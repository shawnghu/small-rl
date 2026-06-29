"""Warm-start SFT for gradient-routing adapters.

Before RL begins, supervised-fine-tune the adapters on collected samples so the
retain/forget behaviors are pre-localized into their respective adapters:

  Phase 1 (retain): retain-only adapter config (set_scales(1, 0)); CE on the
    retain samples; optimizer scoped to the retain params.
  Phase 2 (forget): two-adapter config (set_scales(1, 1)); CE on the forget
    samples; optimizer scoped to the forget params only -- i.e. the retain side
    is frozen / its gradient is "zeroed" (weight-identical to zero-grad hooks,
    but avoids AdamW weight-decay leaking into the frozen retain params).

Loss is plain token-mean cross-entropy on completion tokens only (prompt tokens
masked to -100). Prompts are re-rendered through the chat template so the
SFT-time prompt format is byte-identical to the RL-time format; a per-sample
assert enforces this (fail-loud).

The warm-started weights live in the same parameter tensors the RL trainer
uses, so they persist into training. A separate AdamW is used here and
discarded; RL then proceeds with its own fresh optimizer state ("warm start,
then train as usual").
"""
import json
import os
import random

import torch
import torch.nn.functional as F

from gradient_routing import set_scales, collect_routing_params, has_dual_adapters


def _resolve_path(warmstart_data, environment):
    """warmstart_data may be a jsonl file or a directory of <env>.jsonl files."""
    if os.path.isdir(warmstart_data):
        path = os.path.join(warmstart_data, f"{environment}.jsonl")
    else:
        path = warmstart_data
    assert os.path.isfile(path), f"warmstart data not found: {path}"
    return path


def _build_example(tokenizer, prompt_text, completion_text):
    """Reconstruct the exact RL-format prompt and build (input_ids, labels).

    The stored prompt is the chat-template render decoded with special tokens
    stripped. We recover the user content and re-apply the chat template with
    add_generation_prompt=True, asserting the decode round-trips so the format
    matches RL exactly. Loss is on completion tokens (+EOS) only.
    """
    assert "\nuser\n" in prompt_text and prompt_text.endswith("\nassistant\n"), (
        f"unexpected warmstart prompt format: {prompt_text!r}"
    )
    user = prompt_text.split("\nuser\n", 1)[1].rsplit("\nassistant\n", 1)[0]
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True,
        tokenize=True, return_dict=False,
    )
    assert tokenizer.decode(prompt_ids, skip_special_tokens=True) == prompt_text, (
        "warmstart prompt reconstruction mismatch -- chat template / system prompt "
        f"differs from how this data was generated.\n  got:    "
        f"{tokenizer.decode(prompt_ids, skip_special_tokens=True)!r}\n  stored: {prompt_text!r}"
    )
    comp_ids = tokenizer.encode(completion_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids
    return input_ids, labels


def _split(examples, val_frac, rng):
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(examples) * val_frac))) if examples else 0
    val = [examples[i] for i in idx[:n_val]]
    train = [examples[i] for i in idx[n_val:]]
    return train, val


def _batches(examples, batch_size, pad_id, shuffle, rng):
    """Yield right-padded (input_ids, attention_mask, labels) batches.

    Length-sorted into buckets to keep padding tight; bucket order shuffled."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    buckets = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(buckets)
    for bucket in buckets:
        rows = [examples[i] for i in bucket]
        maxlen = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), maxlen), dtype=torch.long)
        labels = torch.full((len(rows), maxlen), -100, dtype=torch.long)
        for r, (ids, labs) in enumerate(rows):
            n = len(ids)
            input_ids[r, :n] = torch.tensor(ids, dtype=torch.long)
            attn[r, :n] = 1
            labels[r, :n] = torch.tensor(labs, dtype=torch.long)
        yield input_ids, attn, labels


def _ce(model, input_ids, attn, labels, device):
    out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device), use_cache=False)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:].to(device)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        tgt.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


@torch.no_grad()
def _eval_ce(model, examples, batch_size, pad_id, device):
    """Token-weighted mean CE over completion tokens."""
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    rng = random.Random(0)
    for input_ids, attn, labels in _batches(examples, batch_size, pad_id, False, rng):
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device), use_cache=False)
        logits = out.logits[:, :-1, :]
        tgt = labels[:, 1:].to(device)
        ntok = (tgt != -100).sum().item()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        tot_loss += loss.item()
        tot_tok += ntok
    return tot_loss / max(1, tot_tok)


def _run_phase(model, params, train_ex, val_ex, scales, lr, epochs, batch_size,
               pad_id, device, log, tag, rng):
    set_scales(model, *scales)
    opt = torch.optim.AdamW(params, lr=lr)
    log(f"[warmstart:{tag}] start  scales={scales}  n_train={len(train_ex)} "
        f"n_val={len(val_ex)}  init_val_ce={_eval_ce(model, val_ex, batch_size, pad_id, device):.4f}")
    for ep in range(epochs):
        model.train()
        run_loss, n = 0.0, 0
        for input_ids, attn, labels in _batches(train_ex, batch_size, pad_id, True, rng):
            opt.zero_grad(set_to_none=True)
            loss = _ce(model, input_ids, attn, labels, device)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            n += 1
        val = _eval_ce(model, val_ex, batch_size, pad_id, device)
        log(f"[warmstart:{tag}] epoch {ep + 1}/{epochs}  train_ce={run_loss / max(1, n):.4f}  val_ce={val:.4f}")


def run_warmstart(model, tokenizer, args, *, device=None, log=print):
    """Run the two-phase warm-start SFT in place on `model`."""
    assert has_dual_adapters(model), "warmstart requires dual adapters (GR run)"
    device = device or next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    path = _resolve_path(args.warmstart_data, args.environment)
    recs = [json.loads(line) for line in open(path)]
    retain = [_build_example(tokenizer, r["prompt"], r["completion"]) for r in recs if r["cls"] == "retain"]
    forget = [_build_example(tokenizer, r["prompt"], r["completion"]) for r in recs if r["cls"] == "forget"]
    assert retain and forget, f"warmstart needs both classes; got retain={len(retain)} forget={len(forget)}"

    rng = random.Random(args.seed)
    r_train, r_val = _split(retain, args.warmstart_val_frac, rng)
    f_train, f_val = _split(forget, args.warmstart_val_frac, rng)

    lr = args.warmstart_lr if getattr(args, "warmstart_lr", None) is not None else args.lr
    retain_params, forget_params = collect_routing_params(model)
    log(f"[warmstart] env={args.environment} data={path}")
    log(f"[warmstart] retain {len(r_train)}+{len(r_val)}val  forget {len(f_train)}+{len(f_val)}val  "
        f"lr={lr} epochs={args.warmstart_epochs} bs={args.warmstart_batch_size}  "
        f"|retain_params|={sum(p.numel() for p in retain_params):,} "
        f"|forget_params|={sum(p.numel() for p in forget_params):,}")

    _run_phase(model, list(retain_params), r_train, r_val, (1.0, 0.0), lr,
               args.warmstart_epochs, args.warmstart_batch_size, pad_id, device, log, "retain", rng)
    _run_phase(model, list(forget_params), f_train, f_val, (1.0, 1.0), lr,
               args.warmstart_epochs, args.warmstart_batch_size, pad_id, device, log, "forget", rng)

    set_scales(model, 1.0, 1.0)  # restore two-adapter training config
    log("[warmstart] done")
