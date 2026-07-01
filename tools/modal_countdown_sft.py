"""LoRA SFT of Qwen2.5-Coder-7B on the Countdown-Code distillation traces, in our
existing Modal image (HF Trainer + peft) — produces the SFT'd artifact whose
zero-shot hack rate we then measure with tools/modal_countdown_hack_baserate.py.

Equivalent to the paper's verl fsdp_sft_trainer config (LoRA r128/alpha128,
all-linear, lr 1e-4, 5 epochs, completion-masked next-token loss); verl's
FSDP/ulysses/remove-padding are efficiency, not behavior. See COUNTDOWN_CODE_NOTES.md.

Data must be on the volume first:
    .venv/bin/python tools/countdown_build_sft_data.py
    modal volume put gr-modal-pilot output/countdown_sft /countdown_sft --force

Run:
    modal run tools/modal_countdown_sft.py::run            # full: 5 epochs
    modal run tools/modal_countdown_sft.py::run --limit 64 --epochs 1   # smoke

Then point the base-rate harness at the merged model dir on the volume:
    modal run tools/modal_countdown_hack_baserate.py::run \
        --model /output/countdown_sft_model/qwen2.5-coder-7b-sft --no-thinking ...
"""
from __future__ import annotations
import os
import modal
from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

DATA_REMOTE = "/output/countdown_sft"
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUT_ROOT = "/output/countdown_sft_model"

app = modal.App("countdown-sft")


def _slug(model):
    return model.split("/")[-1].lower().replace(".", "-").replace("_", "-")


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=8 * 60 * 60)
def sft(
    model: str = MODEL,
    epochs: int = 5,
    lr: float = 1e-4,
    lora_rank: int = 128,
    lora_alpha: int = 128,
    max_len: int = 2560,
    per_device_bs: int = 4,
    grad_accum: int = 8,
    gradient_checkpointing: bool = False,
    max_steps: int = 0,
    limit: int = 0,
    out_remote: str = "",
) -> dict:
    import json, sys, time
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    if not out_remote:
        out_remote = os.path.join(OUT_ROOT, _slug(model))
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(model)
    eos = tok.eos_token_id

    def load_rows(name):
        path = os.path.join(DATA_REMOTE, f"{name}.jsonl")
        assert os.path.isfile(path), f"missing {path} — `modal volume put` the SFT data first"
        rows = [json.loads(l) for l in open(path)]
        return rows[:limit] if limit else rows

    def encode(row):
        # completion-masked: prompt tokens labeled -100, response tokens trained
        prompt_ids = tok.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                             tokenize=True, return_dict=False)
        if not isinstance(prompt_ids, list):  # some versions return tensors/BatchEncoding
            prompt_ids = list(prompt_ids["input_ids"]) if hasattr(prompt_ids, "keys") else list(prompt_ids)
        resp_ids = tok(row["response"], add_special_tokens=False)["input_ids"] + [eos]
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids
        return input_ids, labels

    def build(name):
        kept, dropped = [], 0
        for r in load_rows(name):
            ids, lab = encode(r)
            if len(ids) > max_len:
                dropped += 1
                continue
            kept.append({"input_ids": ids, "labels": lab})
        print(f"[{name}] {len(kept)} examples ({dropped} dropped > {max_len} tok)", flush=True)
        return kept

    train_ds, val_ds = build("train"), build("val")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        ii, lab, am = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad] * n)
            lab.append(b["labels"] + [-100] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(am)}

    print(f"Loading {model} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2")
    base.config.use_cache = False
    peft_model = get_peft_model(base, LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear"))
    peft_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="/tmp/cc_sft_ckpt", num_train_epochs=epochs, learning_rate=lr,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=grad_accum,
        bf16=True, gradient_checkpointing=gradient_checkpointing, lr_scheduler_type="cosine",
        warmup_ratio=0.03, logging_steps=10, save_strategy="no", report_to=[],
        optim="adamw_torch", dataloader_num_workers=2)
    trainer = Trainer(model=peft_model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    train_s = time.time() - t0
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    m = out.metrics
    eff_batch = per_device_bs * grad_accum
    samples_per_s = m.get("train_samples_per_second", 0.0)
    tok_per_s = round(samples_per_s * 1168, 1)  # mean seq len ~1168
    print(f"[BENCH] gc={gradient_checkpointing} pdbs={per_device_bs} ga={grad_accum} "
          f"max_len={max_len} eff_batch={eff_batch} | runtime={round(train_s,1)}s "
          f"samples/s={round(samples_per_s,3)} tok/s={tok_per_s} peak_gb={peak_gb}", flush=True)

    if max_steps > 0:
        # benchmark run: skip the slow merge+save
        meta = {"model": model, "bench": True, "gc": gradient_checkpointing,
                "per_device_bs": per_device_bs, "grad_accum": grad_accum, "max_len": max_len,
                "train_seconds": round(train_s, 1), "samples_per_second": round(samples_per_s, 3),
                "tokens_per_second": tok_per_s, "peak_gb": peak_gb}
        print(json.dumps(meta, indent=2), flush=True)
        return meta

    print("Merging LoRA and saving ...", flush=True)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_remote, safe_serialization=True)
    tok.save_pretrained(out_remote)
    vol.commit()
    meta = {"model": model, "epochs": epochs, "lr": lr, "lora_rank": lora_rank,
            "n_train": len(train_ds), "n_val": len(val_ds), "train_seconds": round(train_s, 1),
            "tokens_per_second": tok_per_s, "peak_gb": peak_gb, "out": out_remote}
    print(json.dumps(meta, indent=2), flush=True)
    return meta


@app.local_entrypoint()
def run(model: str = MODEL, epochs: int = 5, lr: float = 1e-4, limit: int = 0,
        max_len: int = 2560, per_device_bs: int = 4, grad_accum: int = 8,
        gradient_checkpointing: bool = False, max_steps: int = 0):
    import json
    res = sft.remote(model=model, epochs=epochs, lr=lr, limit=limit, max_len=max_len,
                     per_device_bs=per_device_bs, grad_accum=grad_accum,
                     gradient_checkpointing=gradient_checkpointing, max_steps=max_steps)
    print(json.dumps(res, indent=2))
    out = os.path.join(OUT_ROOT, _slug(model))
    print(f"\nmeasure the SFT'd model:")
    print(f"  modal run tools/modal_countdown_hack_baserate.py::run "
          f"--model {out} --no-thinking --limit 300 --k 8")
