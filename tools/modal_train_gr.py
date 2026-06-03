"""Modal app for running GR retrain pilot on H100s.

Image: nvidia/cuda 12.4 + Python 3.11 + uv sync from /workspace/small-rl/{pyproject.toml,uv.lock}.
Codebase: bundled via add_local_dir at /repo.
Volume: gr-modal-pilot mounted at /output (checkpoints + logs persist).
Secrets: gr-pilot-keys (WANDB_API_KEY, OPENAI_API_KEY).

Each `train_one(params)` call gets one H100, runs train.train_main with
output_dir redirected to /output/<sweep_name>/<run_name>/. vllm_spawn=True so
train.py spawns its own in-function vLLM server.

Launch: see launch_modal_3envs() at the bottom for the 6-run config.
Sync back: `modal volume get gr-modal-pilot / /workspace/small-rl/output/`.
"""
from __future__ import annotations

import modal

REPO_LOCAL = "/workspace/small-rl"
REPO_REMOTE = "/repo"
OUTPUT_REMOTE = "/output"

# NOTE: renamed from "gr-pilot" — jnward lost write access to the "gr-pilot"
# app name on the team-shard-c9-b workspace (a teammate/ACL claimed it). New
# name is owned by jnward; same volume (gr-modal-pilot) so outputs co-locate.
app = modal.App("gr-pilot-jnward")

# Outputs (checkpoints, train.log, routing_eval.jsonl) persist on this volume.
vol = modal.Volume.from_name("gr-modal-pilot", create_if_missing=True)

# Wandb + OpenAI API keys (used by training + topic env retain reward).
secrets = [modal.Secret.from_name("gr-pilot-keys")]

# Image: build deps from pyproject.toml + uv.lock so it matches the local venv.
# vLLM 0.17.0 has broken dep metadata (declared bounds too tight for torch 2.10 /
# transformers 5.2 — see DEPENDENCIES.md); uv handles via the override section
# already in pyproject.toml.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .workdir("/build")
    .add_local_file(f"{REPO_LOCAL}/pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file(f"{REPO_LOCAL}/uv.lock", "/build/uv.lock", copy=True)
    .add_local_file(f"{REPO_LOCAL}/requirements-modal.txt", "/build/requirements-modal.txt", copy=True)
    # vLLM patches must be applied after install — they add a hook attribute
    # (LoRAModelManager._post_create_module_hooks) and a qwen3_5 config fix
    # the codebase depends on at runtime. See vllm_patches/apply.sh.
    .add_local_file(f"{REPO_LOCAL}/vllm_patches/model_manager.py",
                    "/build/vllm_patches/model_manager.py", copy=True)
    .add_local_file(f"{REPO_LOCAL}/vllm_patches/qwen3_5_config.py",
                    "/build/vllm_patches/qwen3_5_config.py", copy=True)
    # Install the full pip-freeze from the working local venv (394 pkgs). Uses
    # --no-deps so pinned versions are respected as-is (vllm 0.17 has broken
    # declared bounds — see DEPENDENCIES.md). flash_attn/flash_attn_3 are
    # prebuilt-wheel URLs so no compile step.
    .run_commands(
        "uv venv --python 3.11 --seed",
        "uv pip install --python /build/.venv/bin/python --no-deps -r /build/requirements-modal.txt",
        # Apply vLLM patches over the installed package.
        "cp /build/vllm_patches/model_manager.py "
        "/build/.venv/lib/python3.11/site-packages/vllm/lora/model_manager.py",
        "cp /build/vllm_patches/qwen3_5_config.py "
        "/build/.venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/qwen3_5.py",
        "ln -s /build/.venv /opt/venv",
    )
    .env({"PATH": "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "PYTHONPATH": REPO_REMOTE,
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          # Quiet HF transfer warnings, point cache to volume so model downloads persist.
          "HF_HOME": "/output/_hf_cache",
          })
    # Add the codebase last so changes don't bust the deps cache.
    .add_local_dir(
        REPO_LOCAL,
        REPO_REMOTE,
        ignore=[
            "__pycache__", "*.pyc",
            ".git", ".venv", ".venv-vllm", ".claude", ".prompt_cache", ".pytest_cache",
            "wandb", "output", "media", "benchmarks",
            "*.log", "*.pdf", "*.png",
            ".*.un~", ".*.sw[opn]", ".*.swp",
            "figures_pareto/figs", "paper_figures",
        ],
        copy=False,  # fresh mount each call — captures recent edits cheaply
    )
    # The leetcode env loads jsonls from ~/rl-rewardhacking-private/results/data/
    # AND persistent_code_eval.py does `from src.evaluate.code.helpers ...`
    # against the same repo. RH_REPO_PATH defaults to ~/rl-rewardhacking-private
    # (= /root/rl-rewardhacking-private inside the container), and
    # ensure_importable() inserts it into sys.path. Mount only the two
    # subdirs we need (data 369 MB + src 31 MB); skip the rest because the
    # uv-installed .venv pushes the full repo to 22 GB.
    .add_local_dir(
        "/workspace/rl-rewardhacking-private/results/data",
        "/root/rl-rewardhacking-private/results/data",
        copy=False,
    )
    .add_local_dir(
        "/workspace/rl-rewardhacking-private/src",
        "/root/rl-rewardhacking-private/src",
        copy=False,
    )
)


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=60 * 60,
)
def eval_math_base(plan: dict) -> dict:
    """Evaluate base Qwen3-8B (no RL, reasoning DISABLED) on a MATH subset.

    plan: {data_path, model, n_eval, max_tokens, temperature}. Reads a jsonl of
    {problem, gold} from the volume, runs offline vLLM with enable_thinking=False,
    grades \\boxed answers with math_verify. Returns pass@1.
    """
    import os, sys, json, subprocess, time
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    # math_verify isn't in the pinned image; install at runtime (fast, wheels).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "math-verify", "latex2sympy2_extended"], check=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from math_verify import parse, verify

    model = plan.get("model", "Qwen/Qwen3-8B")
    rows = [json.loads(l) for l in open(os.path.join(OUTPUT_REMOTE, plan["data_path"]))]
    rows = rows[: plan.get("n_eval", 100)]

    tok = AutoTokenizer.from_pretrained(model)
    SYS = ("Solve the math problem. Put your final answer in \\boxed{}.")
    prompts = []
    for r in rows:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": r["problem"]}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=plan.get("max_tokens", 8192) + 1024, enforce_eager=False)
    sp = SamplingParams(temperature=plan.get("temperature", 0.0),
                        max_tokens=plan.get("max_tokens", 8192))
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0

    n_correct = 0; n_boxed = 0; details = []
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        gold = parse("\\boxed{" + r["gold"] + "}")
        pred = parse(txt)
        ok = False
        try:
            ok = bool(verify(gold, pred))
        except Exception:
            ok = False
        has_box = "\\boxed" in txt
        n_boxed += int(has_box); n_correct += int(ok)
        details.append({"gold": r["gold"], "correct": ok, "has_boxed": has_box,
                        "gen_len": len(o.outputs[0].token_ids)})

    out_path = os.path.join(OUTPUT_REMOTE, "math_eval", "qwen3_8b_base_l5_100_results.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for d in details: f.write(json.dumps(d) + "\n")
    vol.commit()
    n = len(rows)
    return {"n": n, "pass@1": n_correct / n, "boxed_rate": n_boxed / n,
            "mean_gen_len": sum(d["gen_len"] for d in details) / n,
            "duration_s": dur, "out": out_path}


@app.function(
    image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
    timeout=60 * 60,
)
def eval_dualenv_math(plan: dict) -> dict:
    """Post-hoc MATH eval of a dual-env checkpoint at both + retain_only scales.

    Loads the checkpoint's DualLoRA adapter, generates on the math_l5 eval set
    (HF generate via eval_gradient_routing), grades math_correct. Gives the
    coherence-env (math) capability metric that the in-training eval path is
    mis-logging (and confirms it's an eval-path bug, since the model trains math).

    plan: {ckpt, n_eval, max_new_tokens, temperature}.
    """
    import os, sys, json
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch
    from eval_utils import load_gradient_routing_model, eval_gradient_routing, score_eval_samples
    from envs import get_env
    from transformers import AutoTokenizer
    from rewards import get_reward_fn

    ckpt = plan["ckpt"]
    base_model = plan.get("base_model", "Qwen/Qwen3-8B")
    n_eval = plan.get("n_eval", 100)
    tok = AutoTokenizer.from_pretrained(base_model)
    if plan.get("base_only"):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="bfloat16").cuda()
    else:
        model = load_gradient_routing_model(ckpt, base_model=base_model, mlp_config="m64")

    spec = get_env("math_l5")
    eval_data = spec.load_eval_prompts(n_eval, None)
    prompts = [d["prompt"] for d in eval_data]
    metrics = {"math_correct": get_reward_fn("math_correct")}
    modes = ([("both", 1.0, 1.0)] if plan.get("base_only")
             else [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0)])

    results = eval_gradient_routing(
        model, tok, metrics, n_samples=len(prompts),
        max_new_tokens=plan.get("max_new_tokens", 2048),
        temperature=plan.get("temperature", 0.7),
        prompts=prompts, eval_data=eval_data, modes=modes,
    )
    out = {}; diag = {}
    for mode_name, mode_data in results.items():
        mc = mode_data["metrics"].get("math_correct", {})
        out[mode_name] = mc.get("mean")
        samps = mode_data.get("samples", [])
        comps = [(s if isinstance(s, str) else s.get("completion", "")) for s in samps]
        boxed = sum(1 for c in comps if "\\boxed" in c) / max(1, len(comps))
        diag[mode_name] = {"boxed_rate": round(boxed, 3),
                           "sample_completion_tail": (comps[0][-300:] if comps else None)}
    print(f"[dualenv-math] {ckpt} base_only={plan.get('base_only')} math_correct={out}")
    for m, d in diag.items():
        print(f"  [{m}] boxed_rate={d['boxed_rate']}")
        print(f"    comp_tail: {d['sample_completion_tail']!r}")
    return {"ckpt": ckpt, "math_correct": out, "n": len(prompts), "diag": diag}


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=60 * 60)
def kl_from_base(plan: dict) -> dict:
    """Measure how far the trained policy moved from base: KL(ckpt || base) on
    ckpt-generated completions (on-policy). Generates N completions from the
    checkpoint, then teacher-forces both ckpt and base to get per-token logps;
    KL_k1 = mean(logp_ckpt - logp_base), KL_k3 = mean(exp(-d)+d-1), d=logp_ckpt-logp_base.
    ~0 nats/token => training was a near-no-op; larger => real policy movement."""
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from eval_utils import load_gradient_routing_model
    from envs import get_env

    base_model = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    spec = get_env("math_l5")
    n = plan.get("n_eval", 40)
    eval_data = spec.load_eval_prompts(n, None)
    texts = [tok.apply_chat_template(d["prompt"], add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False) for d in eval_data]

    # 1) generate completions from the checkpoint (batched)
    ckpt = load_gradient_routing_model(plan["ckpt"], base_model=base_model, mlp_config="m64").eval()
    seqs = []  # (full_ids, p_len) per sample
    B = 20
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = ckpt.generate(**enc, max_new_tokens=plan.get("max_new_tokens", 1024),
                              do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                              pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for r in range(g.shape[0]):
            # strip left-pad of prompt: real prompt tokens are the non-pad in enc row r
            real = enc["attention_mask"][r].bool()
            p_ids = enc["input_ids"][r][real]
            c_ids = g[r][plen:]
            c_ids = c_ids[c_ids != tok.pad_token_id]
            if c_ids.numel() > 0:
                seqs.append((torch.cat([p_ids, c_ids]), p_ids.numel()))

    def comp_logps(model, seqs):
        outs = []
        with torch.no_grad():
            for full, p_len in seqs:
                logits = model(full.unsqueeze(0).cuda()).logits[0].float()
                lp = torch.log_softmax(logits[:-1], dim=-1)
                idx = torch.arange(p_len - 1, full.numel() - 1, device="cuda")
                tgt = full[p_len:].cuda()
                outs.append(lp[idx].gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
        return outs

    lp_ckpt = comp_logps(ckpt, seqs)
    del ckpt; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="bfloat16").cuda().eval()
    lp_base = comp_logps(base, seqs)

    import torch as T
    all_d = T.cat([(a - b) for a, b in zip(lp_ckpt, lp_base)])  # logp_ckpt - logp_base, per token
    kl_k1 = all_d.mean().item()
    kl_k3 = (T.exp(-all_d) + all_d - 1.0).mean().item()
    per_seq = T.tensor([(a - b).sum().item() for a, b in zip(lp_ckpt, lp_base)])
    res = {"n_seq": len(seqs), "n_tokens": all_d.numel(),
           "KL_k1_nats_per_token": round(kl_k1, 4),
           "KL_k3_nats_per_token": round(kl_k3, 4),
           "mean_KL_per_sequence_nats": round(per_seq.mean().item(), 2),
           "frac_tokens_changed_gt0.1": round((all_d.abs() > 0.1).float().mean().item(), 3)}
    print(f"[KL] {res}", flush=True)
    return res


@app.local_entrypoint()
def launch_modal_kl_from_base():
    plan = {"ckpt": f"{OUTPUT_REMOTE}/math_l5_fixtest/math_l5_fixtest_s22/checkpoint-3200",
            "n_eval": 40, "max_new_tokens": 1024}
    res = kl_from_base.remote(plan)
    print(f"=== KL(ckpt-3200 || base): {res} ===")


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=60 * 60)
def boxed_rate_compare(plan: dict) -> dict:
    """Compare \\boxed{} rate (and mean length / correctness) on FULL completions
    between base and the fixtest checkpoint, on the math_l5 eval set. The
    in-eval boxed_rate diag is computed on truncated stored samples and is
    unreliable; this generates and inspects full completions."""
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from eval_utils import load_gradient_routing_model
    from envs import get_env
    from rewards import get_reward_fn

    base_model = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    spec = get_env("math_l5")
    n = plan.get("n_eval", 60)
    eval_data = spec.load_eval_prompts(n, None)
    prompts = [d["prompt"] for d in eval_data]
    gold = [d.get("gold") for d in eval_data]
    mc_fn = get_reward_fn("math_correct")

    tok.padding_side = "left"
    texts = [tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False) for p in prompts]
    out = {}
    for name, ckpt in [("base", None), ("ckpt3200", plan["ckpt"])]:
        if ckpt is None:
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="bfloat16").cuda().eval()
        else:
            model = load_gradient_routing_model(ckpt, base_model=base_model, mlp_config="m64").eval()
        comps = []
        n_trunc = 0
        gen_lens = []
        eos_id = tok.eos_token_id
        max_new = plan.get("max_new_tokens", 1536)
        B = 20
        for i in range(0, len(texts), B):
            enc = tok(texts[i:i + B], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new,
                                   do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                                   pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            for row in g:
                gen = row[plen:]
                comps.append(tok.decode(gen, skip_special_tokens=True))
                # truncated = no EOS emitted within the budget
                truncated = (eos_id not in gen.tolist())
                n_trunc += int(truncated)
                # generated token count (excl. trailing pad after EOS)
                nonpad = (gen != tok.pad_token_id).sum().item()
                gen_lens.append(nonpad)
        boxed = sum(1 for c in comps if "\\boxed" in c) / len(comps)
        mean_len = sum(len(c) for c in comps) / len(comps)
        scores = mc_fn(comps, gold=gold)
        gen_lens.sort()
        out[name] = {"boxed_rate": round(boxed, 3),
                     "mean_char_len": round(mean_len, 0),
                     "math_correct": round(sum(scores) / len(scores), 3),
                     "trunc_rate": round(n_trunc / len(comps), 3),
                     "n_trunc": n_trunc, "n": len(comps),
                     "mean_gen_tokens": round(sum(gen_lens) / len(gen_lens)),
                     "p90_gen_tokens": gen_lens[int(len(gen_lens) * 0.9)],
                     "max_new_tokens": max_new}
        del model; torch.cuda.empty_cache()
    print(f"[BOXED] {out}", flush=True)
    return out


@app.local_entrypoint()
def launch_modal_boxed_rate_compare():
    plan = {"ckpt": f"{OUTPUT_REMOTE}/math_l5_fixtest/math_l5_fixtest_s22/checkpoint-3200",
            "n_eval": 60, "max_new_tokens": 2048}
    res = boxed_rate_compare.remote(plan)
    print("=== boxed rate: base vs ckpt-3200 (full completions) ===")
    for k, v in res.items():
        print(f"  {k}: boxed_rate={v['boxed_rate']} mean_char_len={v['mean_char_len']} math_correct={v['math_correct']}")


@app.local_entrypoint()
def launch_modal_eval_fixtest_math():
    """Trustworthy offline math eval of the fixtest checkpoint (the old-logp fix
    run) — confirms whether the model RETAINS math (vs base=0.65) and whether
    the in-training eval=0 is just the harness path. Evals checkpoint-400."""
    plan = {"ckpt": f"{OUTPUT_REMOTE}/math_l5_fixtest/math_l5_fixtest_s22/checkpoint-3200",
            "n_eval": 100, "max_new_tokens": 2048, "temperature": 0.7}
    res = eval_dualenv_math.remote(plan)
    print(f"=== fixtest ckpt-400 offline math_correct: {res['math_correct']} (base ref = 0.65) ===")
    print(f"    diag: {res['diag']}")


@app.local_entrypoint()
def launch_modal_eval_dualenv_math():
    """Post-hoc math eval of the dual-env checkpoint-600 (s22) at both+retain_only."""
    # Eval the MATH BASELINE checkpoint (single-env, trained only on math) to
    # confirm whether math collapses under training (vs base=0.65).
    plan = {"ckpt": f"{OUTPUT_REMOTE}/math_l5_baseline/math_l5_baseline_s22/checkpoint-2000",
            "n_eval": 100, "max_new_tokens": 2048, "temperature": 0.7}
    res = eval_dualenv_math.remote(plan)
    print(f"[dualenv-math] math_baseline ckpt-2000  math_correct={res['math_correct']}  n={res['n']}")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,
)
def eval_math_pass_k(plan: dict) -> dict:
    """pass@k filtering shard. plan: {rows, model, k, max_tokens, temperature,
    shard_id}. Runs k samples per problem (no reasoning), grades \\boxed with
    math_verify, returns per-problem n_correct."""
    import os, sys, subprocess, time
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "math-verify", "latex2sympy2_extended"], check=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from math_verify import parse, verify

    rows = plan["rows"]; model = plan.get("model", "Qwen/Qwen3-8B")
    k = plan.get("k", 8); max_tokens = plan.get("max_tokens", 4096)
    tok = AutoTokenizer.from_pretrained(model)
    SYS = "Solve the math problem. Put your final answer in \\boxed{}."
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": r["problem"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=max_tokens + 1024)
    sp = SamplingParams(n=k, temperature=plan.get("temperature", 0.7), top_p=0.95,
                        max_tokens=max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0

    results = []
    for r, o in zip(rows, outs):
        gold = parse("\\boxed{" + r["gold"] + "}")
        nc = 0
        for samp in o.outputs:
            try:
                if verify(gold, parse(samp.text)): nc += 1
            except Exception:
                pass
        results.append({"id": r["id"], "n_correct": nc, "k": k})
    return {"shard_id": plan.get("shard_id"), "results": results, "duration_s": dur}


@app.local_entrypoint()
def launch_modal_math_pass8_filter():
    """pass@8 filter the MATH L5 train set across 8 H200 shards. Keeps problems
    with 0 < n_correct < 8 (drops always/never-solved). Writes filtered pool +
    full pass-count table to the volume."""
    import json, subprocess
    # Pull the L5 train set locally from the volume.
    subprocess.run([".venv/bin/python", "-m", "modal", "volume", "get", "gr-modal-pilot",
                    "math_eval/math_l5_train.jsonl", "/tmp/_l5_train.jsonl", "--force"],
                   cwd=REPO_LOCAL, check=True)
    rows = [json.loads(l) for l in open("/tmp/_l5_train.jsonl")]
    n_shards = 8
    shards = [rows[i::n_shards] for i in range(n_shards)]
    plans = [{"rows": s, "model": "Qwen/Qwen3-8B", "k": 8, "max_tokens": 4096,
              "temperature": 0.7, "shard_id": i} for i, s in enumerate(shards)]
    print(f"[pass8] {len(rows)} problems across {n_shards} H200 shards (k=8, max_tokens=4096)")
    out = list(eval_math_pass_k.map(plans))
    by_id = {}
    for r in out:
        print(f"  shard {r['shard_id']}: {len(r['results'])} problems ({r['duration_s']:.0f}s)")
        for x in r["results"]: by_id[x["id"]] = x["n_correct"]
    # Merge counts back onto rows, write full table + filtered pool.
    import os
    full = [{**r, "n_correct": by_id.get(r["id"], 0), "k": 8} for r in rows]
    kept = [r for r in full if 0 < r["n_correct"] < 8]
    from collections import Counter
    dist = Counter(r["n_correct"] for r in full)
    print(f"[pass8] pass-count dist (n_correct/8): {dict(sorted(dist.items()))}")
    print(f"[pass8] kept (0<nc<8): {len(kept)} / {len(full)}")
    os.makedirs("/tmp/pass8", exist_ok=True)
    with open("/tmp/pass8/math_l5_train_passcounts.jsonl", "w") as f:
        for r in full: f.write(json.dumps(r) + "\n")
    with open("/tmp/pass8/math_l5_train_filtered.jsonl", "w") as f:
        for r in kept: f.write(json.dumps({k: r[k] for k in ("id", "problem", "gold", "subject")}) + "\n")
    for fn in ("math_l5_train_passcounts.jsonl", "math_l5_train_filtered.jsonl"):
        subprocess.run([".venv/bin/python", "-m", "modal", "volume", "put", "gr-modal-pilot",
                        f"/tmp/pass8/{fn}", f"/math_eval/{fn}", "--force"], cwd=REPO_LOCAL, check=True)
    print(f"[pass8] wrote filtered pool + pass-count table to volume math_eval/")


@app.local_entrypoint()
def launch_modal_eval_math_base():
    """Eval base Qwen3-8B (no reasoning) on 100 MATH level-5 problems, max 8k tokens."""
    plan = {"data_path": "math_eval/math_l5_100.jsonl", "model": "Qwen/Qwen3-8B",
            "n_eval": 100, "max_tokens": 2048, "temperature": 0.7}  # match in-training eval
    res = eval_math_base.remote(plan)
    print(f"[math-eval] n={res['n']}  pass@1={res['pass@1']:.3f}  "
          f"boxed_rate={res['boxed_rate']:.3f}  mean_gen_len={res['mean_gen_len']:.0f}  "
          f"({res['duration_s']:.1f}s)")
    print(f"  results: {res['out']}")


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=15 * 60,  # 15 min smoke is plenty
)
def smoke() -> dict:
    """Sanity check: container has Python 3.11, repo importable, 1 H100, secrets set."""
    import os, sys, subprocess
    info = {}
    info["python"] = sys.version.split()[0]
    info["pwd"] = os.getcwd()
    info["repo_exists"] = os.path.isdir(REPO_REMOTE)
    info["wandb_key"] = bool(os.environ.get("WANDB_API_KEY"))
    info["openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))
    info["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    try:
        import torch
        # Cast versions to plain str so the result deserializes locally even
        # when torch isn't installed on the caller side (TorchVersion subclass
        # of str carries a module reference that breaks pickle.loads).
        info["torch_version"] = str(torch.__version__)
        info["torch_cuda"] = bool(torch.cuda.is_available())
        info["gpu_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception as e:
        info["torch_err"] = str(e)
    try:
        import transformers, trl, vllm, peft
        info["transformers"] = str(transformers.__version__)
        info["trl"] = str(trl.__version__)
        info["vllm"] = str(vllm.__version__)
        info["peft"] = str(peft.__version__)
    except Exception as e:
        info["deps_err"] = str(e)
    try:
        from train import train_main  # noqa: F401
        info["train_import"] = True
    except Exception as e:
        info["train_import_err"] = str(e)
    return info


def _run_training(params: dict, sweep_name: str) -> dict:
    """Shared single-job training body. GPU is chosen by the calling Modal
    function (train_one = H100, train_one_h200 = H200)."""
    import os
    import sys
    import time
    import traceback

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    # Output to the mounted volume, partitioned by sweep_name + run_name.
    run_name = params.get("run_name") or "unnamed"
    out_dir = os.path.join(OUTPUT_REMOTE, sweep_name, run_name)
    os.makedirs(out_dir, exist_ok=True)
    params = {**params, "output_dir": out_dir, "gpu_id": 0}

    # Inside the container we have exactly 1 H100. Spawn vLLM in-process.
    params.setdefault("vllm_spawn", True)
    params.setdefault("vllm_gpu_memory", 0.05)
    # MPS isn't relevant for 1-run-per-container.

    # Tee stdout/stderr to train.log on the volume, in addition to Modal's stdout.
    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                try: st.write(s); st.flush()
                except Exception: pass
        def flush(self):
            for st in self.streams:
                try: st.flush()
                except Exception: pass

    real_stdout = sys.stdout
    real_stderr = sys.stderr
    sys.stdout = _Tee(real_stdout, log_f)
    sys.stderr = _Tee(real_stderr, log_f)

    t0 = time.time()
    status = "ok"
    err = None
    try:
        # Import here so any import-time side effects happen with stdout=tee.
        from train import train_main
        train_main(params)
    except SystemExit as e:
        if e.code not in (0, None):
            status = f"sysexit({e.code})"
            err = str(e)
    except BaseException as e:
        status = "crash"
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        dur = time.time() - t0
        log_f.close()
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        vol.commit()  # ensure outputs are flushed to the volume

    return {"status": status, "duration_s": dur, "err": err, "run_name": run_name,
            "output_dir": out_dir}


@app.function(
    image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
    timeout=24 * 60 * 60,
)
def train_one(params: dict, sweep_name: str) -> dict:
    """1×H100 training job."""
    return _run_training(params, sweep_name)


@app.function(
    image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
    timeout=24 * 60 * 60,
)
def train_one_h200(params: dict, sweep_name: str) -> dict:
    """1×H200 training job (more memory — for longer-completion runs that OOM
    a 16-wide forward on H100)."""
    return _run_training(params, sweep_name)


@app.function(
    image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol},
    secrets=secrets + [modal.Secret.from_name("judge-keys")],
    timeout=24 * 60 * 60,
)
def train_one_h200_judge(params: dict, sweep_name: str) -> dict:
    """1×H200 training with OPENROUTER_API_KEY + WANDB_API_KEY in env (for the
    llm_judge routing detector via OpenRouter + wandb logging)."""
    return _run_training(params, sweep_name)


def _dispatch_sweep_judge(sweep_module_name: str, sweep_name: str):
    """Like _dispatch_sweep but dispatches to train_one_h200_judge (judge keys)."""
    import sys, importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs
    print(f"[modal] dispatching {len(runs)} judge runs (sweep={sweep_name}, gpu=H200)")
    for r in runs:
        print(f"  - {r['run_name']}")
    results = list(train_one_h200_judge.starmap([(r, sweep_name) for r in runs]))
    for res in results:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")
    fails = [r for r in results if r["status"] != "ok"]
    print(f"[modal] {'all ok' if not fails else str(len(fails)) + ' failed'}")


@app.local_entrypoint()
def launch_modal_judge_nocoh_classic_smoke():
    """5-step smoke of Run A — validates the OpenRouter judge detector + wandb
    (may31-judge-testing) + classic-routing path before the full launch."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_judge_nocoh_classic.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = {**mod.runs[0], "max_steps": 5, "save_steps": 5,
             "run_name": "smoke_judge_nocoh_classic"}
    print(f"[smoke] {pilot['run_name']} (judge=OpenRouter 235B highprec_nostrip, "
          f"routing=classic, hack_frac={pilot['hack_frac']}, H200)")
    res = train_one_h200_judge.remote(pilot, "leetcode_judge_nocoh_classic_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_judge_kl_coh_merged_smoke():
    """5-step smoke of Run B — validates merged KL-to-base coherence + judge path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_judge_kl_coh_merged.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = {**mod.runs[0], "max_steps": 5, "save_steps": 5,
             "run_name": "smoke_judge_kl_coh_merged"}
    print(f"[smoke] {pilot['run_name']} (KL-coh merged b={pilot['coh_kl_beta']} "
          f"cspr={pilot['coh_samples_per_rollout']}, H200)")
    res = train_one_h200_judge.remote(pilot, "leetcode_judge_kl_coh_merged_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_judge_nocoh_classic_full():
    """Run A: 3 H200 seeds, no coherence + classic routing, 235B judge, 800 steps."""
    _dispatch_sweep_judge("sweeps.leetcode_judge_nocoh_classic", "leetcode_judge_nocoh_classic")


@app.local_entrypoint()
def launch_modal_judge_kl_coh_merged_full():
    """Run B: 1 H200 seed, KL-to-base merged coherence (b=0.1), else identical to Run A."""
    _dispatch_sweep_judge("sweeps.leetcode_judge_kl_coh_merged", "leetcode_judge_kl_coh_merged")


@app.function(image=image, gpu="H200", secrets=secrets, timeout=60 * 60)
def packing_isolation_probe() -> dict:
    """Test whether the padding-free packed forward (train.py _packed_compute_loss)
    isolates sequences correctly. Packs several sequences of varied lengths into a
    single (1,T) forward with position_ids reset per sequence (NO attention_mask,
    same as training), then compares each sequence's completion logps to computing
    that sequence ALONE. If later-in-pack sequences diverge from their solo logps,
    flash-attn-varlen is NOT isolating them -> cross-sequence attention leak, the
    root of the 25-nat new-vs-old gap. Reports per-sequence: position-in-pack,
    length, max|Δ logp|, mean|Δ logp|."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    lm_head = model.lm_head
    dev = "cuda"

    # Build varied-length sequences from real-ish text (mix lengths like the real
    # batch: short ~leetcode, long ~math). Each seq = prompt(=first half) +
    # completion(=second half). Use distinct token content per seq.
    filler = ("Solve the problem step by step. We compute the integral and sum the "
              "series, then simplify the expression to obtain the final boxed answer. ")
    base = tok(filler * 60, return_tensors="pt").input_ids[0].to(dev)
    lengths = [200, 400, 900, 1500, 2048]
    seqs = []
    for k, L in enumerate(lengths):
        # rotate the token stream per seq so contents differ
        rolled = torch.roll(base, shifts=37 * (k + 1))
        reps = (L // rolled.shape[0]) + 1
        seqs.append(rolled.repeat(reps)[:L].contiguous())

    def comp_logps(hidden, ids, p_len, c_len):
        # logp for completion token t predicted by hidden state t-1
        hs = hidden[p_len - 1: p_len + c_len - 1]            # (c_len, H)
        logits = lm_head(hs.unsqueeze(0)).float()            # (1, c_len, V)
        lp = torch.log_softmax(logits, dim=-1)[0]            # (c_len, V)
        comp_ids = ids[p_len: p_len + c_len]
        return lp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)  # (c_len,)

    # ---- PACKED forward (all seqs concatenated, position_ids reset) ----
    packed_ids = torch.cat(seqs).unsqueeze(0)
    packed_pos = torch.cat([torch.arange(s.shape[0], device=dev) for s in seqs]).unsqueeze(0)
    with torch.no_grad():
        h_packed = model.model(input_ids=packed_ids, position_ids=packed_pos,
                               use_cache=False).last_hidden_state[0]  # (T, H)

    results = []
    offset = 0
    for k, s in enumerate(seqs):
        L = s.shape[0]
        p_len = L // 2
        c_len = L - p_len
        h_pk = h_packed[offset: offset + L]
        lp_packed = comp_logps(h_pk, s, p_len, c_len)
        # ---- SOLO forward (this seq alone) ----
        with torch.no_grad():
            h_solo = model.model(
                input_ids=s.unsqueeze(0),
                position_ids=torch.arange(L, device=dev).unsqueeze(0),
                use_cache=False,
            ).last_hidden_state[0]
        lp_solo = comp_logps(h_solo, s, p_len, c_len)
        d = (lp_packed - lp_solo).abs()
        results.append({
            "pos_in_pack": k, "length": L,
            "max_abs_dlogp": round(d.max().item(), 3),
            "mean_abs_dlogp": round(d.mean().item(), 4),
        })
        offset += L

    for r in results:
        print(f"[PROBE] pos={r['pos_in_pack']} len={r['length']:>4} "
              f"max|Δlogp|={r['max_abs_dlogp']:>8} mean|Δlogp|={r['mean_abs_dlogp']}", flush=True)
    return {"results": results}


@app.local_entrypoint()
def launch_modal_packing_probe():
    """Run the packing-isolation probe on H200 and print per-sequence logp gaps."""
    res = packing_isolation_probe.remote()
    print("=== packing isolation probe ===")
    for r in res["results"]:
        print(f"  pos_in_pack={r['pos_in_pack']} length={r['length']} "
              f"max|Δlogp|={r['max_abs_dlogp']} mean|Δlogp|={r['mean_abs_dlogp']}")


@app.function(image=image, gpu="H200", secrets=secrets, timeout=60 * 60)
def position_padding_probe() -> dict:
    """Test whether a LEFT-PADDED batch forward (the `old` path's format) computes
    wrong completion logps when position_ids aren't padding-aware. Builds rows with
    DIFFERENT prompt lengths (so left-pad varies per row), then computes completion
    logps three ways and compares to the packed/correct reference:
      (a) packed (real tokens, positions reset) = ground truth
      (b) left-padded + attention_mask, NO explicit position_ids (model default)
      (c) left-padded + attention_mask, padding-aware position_ids (cumsum-1)
    If (b) diverges from (a) — worse for shorter prompts (more left-pad) — and (c)
    matches (a), the bug is non-padding-aware position_ids in the old-logp path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    lm_head = model.lm_head
    dev = "cuda"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    txt = ("Compute the sum of the series and simplify to a boxed answer. "
           "First expand, then factor, then evaluate at the given point. ")
    stream = tok(txt * 80, return_tensors="pt").input_ids[0].to(dev)
    prompt_lens = [40, 300, 800]          # varied -> varied left-pad
    c_len = 200
    P = max(prompt_lens)
    rows, pmask_rows, cmask = [], [], []
    prompts, comps = [], []
    for k, pl in enumerate(prompt_lens):
        pr = torch.roll(stream, 53 * (k + 1))[:pl].contiguous()
        co = torch.roll(stream, 91 * (k + 1))[pl:pl + c_len].contiguous()
        prompts.append(pr); comps.append(co)

    def comp_logps(hidden, comp_ids, start):
        hs = hidden[start - 1: start - 1 + comp_ids.shape[0]]
        logits = lm_head(hs.unsqueeze(0)).float()
        lp = torch.log_softmax(logits, dim=-1)[0]
        return lp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)

    # (a) packed reference, per sequence solo (correct positions)
    ref = []
    for pr, co in zip(prompts, comps):
        seq = torch.cat([pr, co])
        with torch.no_grad():
            h = model.model(input_ids=seq.unsqueeze(0),
                            position_ids=torch.arange(seq.shape[0], device=dev).unsqueeze(0),
                            use_cache=False).last_hidden_state[0]
        ref.append(comp_logps(h, co, pr.shape[0]))

    # Build left-padded batch (prompts left-padded to P, completions appended)
    N = len(prompts)
    inp = torch.full((N, P + c_len), pad_id, dtype=torch.long, device=dev)
    attn = torch.zeros((N, P + c_len), dtype=torch.long, device=dev)
    for i, (pr, co) in enumerate(zip(prompts, comps)):
        pl = pr.shape[0]
        inp[i, P - pl:P] = pr
        inp[i, P:P + c_len] = co
        attn[i, P - pl:] = 1

    def run_padded(use_aware_pos):
        kw = dict(input_ids=inp, attention_mask=attn, use_cache=False)
        if use_aware_pos:
            pos = attn.long().cumsum(-1) - 1
            pos = pos.masked_fill(attn == 0, 1)
            kw["position_ids"] = pos
        with torch.no_grad():
            h = model.model(**kw).last_hidden_state  # (N, P+c_len, H)
        out = []
        for i in range(N):
            out.append(comp_logps(h[i], comps[i], P))  # completion starts at index P
        return out

    pad_default = run_padded(False)
    pad_aware = run_padded(True)

    results = []
    for i, pl in enumerate(prompt_lens):
        db = (pad_default[i] - ref[i]).abs()
        dc = (pad_aware[i] - ref[i]).abs()
        results.append({
            "prompt_len": pl, "left_pad": P - pl,
            "default_pos_max|d|": round(db.max().item(), 3),
            "aware_pos_max|d|": round(dc.max().item(), 4),
        })
    for r in results:
        print(f"[POSPROBE] prompt_len={r['prompt_len']:>4} left_pad={r['left_pad']:>4} "
              f"default_pos_max|d|={r['default_pos_max|d|']:>8} "
              f"aware_pos_max|d|={r['aware_pos_max|d|']}", flush=True)
    return {"results": results}


@app.local_entrypoint()
def launch_modal_position_probe():
    """Run the left-padding / position_ids probe on H200."""
    res = position_padding_probe.remote()
    print("=== position/padding probe (ref = correct packed logps) ===")
    for r in res["results"]:
        print(f"  prompt_len={r['prompt_len']} left_pad={r['left_pad']} "
              f"default_pos_max|d|={r['default_pos_max|d|']} aware_pos_max|d|={r['aware_pos_max|d|']}")


@app.function(image=image, gpu="H200", secrets=secrets, timeout=60 * 60)
def precision_probe() -> dict:
    """Test whether fp32 vs bf16 log_softmax diverge on REAL generated tokens.
    Generates real completions for math prompts (sampled, like training), then
    computes completion logps two ways on the SAME logits:
      (a) fp32: log_softmax(logits.float())   <- the loss/new path (train.py:1752)
      (b) bf16: log_softmax(logits)            <- native dtype (the old path likely)
    Reports the max |fp32-bf16| gap and the value/probability of the worst tokens.
    If rare (very negative-logp) tokens show ~20-30 nat gaps, the crash is a
    precision mismatch between the two logp computations, triggered by the
    low-prob tokens that long math generations produce."""
    import torch, json, os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    dev = "cuda"

    prompts = [
        "Solve. Put your final answer in \\boxed{}.\nWhat is the remainder when "
        "$2^{2024}$ is divided by $7$?",
        "Solve. Put your final answer in \\boxed{}.\nFind the number of ordered "
        "pairs $(x,y)$ of integers with $x^2 + y^2 = 2025$.",
    ]
    worst = []
    all_gaps = []
    for pr in prompts:
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        p_len = ids.shape[1]
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=400, do_sample=True,
                                 temperature=0.7, top_p=1.0, top_k=0)
        full = gen  # (1, p_len + c_len)
        comp_ids = full[0, p_len:]
        with torch.no_grad():
            logits = model(full).logits[0]            # (T, V), bf16
        # logit at position t predicts token t+1; completion tokens at [p_len, T)
        pred = logits[p_len - 1: full.shape[1] - 1]   # (c_len, V)
        lp_fp32 = torch.log_softmax(pred.float(), dim=-1)
        lp_bf16 = torch.log_softmax(pred, dim=-1).float()
        tgt = comp_ids
        g_fp32 = lp_fp32.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        g_bf16 = lp_bf16.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        gap = (g_fp32 - g_bf16).abs()
        all_gaps.append(gap.max().item())
        k = gap.argmax().item()
        worst.append({
            "max_gap": round(gap.max().item(), 3),
            "n_gap_gt5": int((gap > 5).sum().item()),
            "worst_token_logp_fp32": round(g_fp32[k].item(), 3),
            "worst_token_logp_bf16": round(g_bf16[k].item(), 3),
            "min_logp_fp32": round(g_fp32.min().item(), 2),
            "c_len": int(comp_ids.shape[0]),
        })
    for w in worst:
        print(f"[PRECPROBE] max|fp32-bf16|={w['max_gap']:>8}  n>5={w['n_gap_gt5']}  "
              f"worst_tok: fp32={w['worst_token_logp_fp32']} bf16={w['worst_token_logp_bf16']}  "
              f"min_logp={w['min_logp_fp32']}  c_len={w['c_len']}", flush=True)
    return {"worst": worst}


@app.local_entrypoint()
def launch_modal_precision_probe():
    """Run the fp32-vs-bf16 logp precision probe on real generated math completions."""
    res = precision_probe.remote()
    print("=== precision probe (fp32 vs bf16 log_softmax on real tokens) ===")
    for w in res["worst"]:
        print(f"  max|fp32-bf16|={w['max_gap']} n>5={w['n_gap_gt5']} "
              f"worst_token fp32={w['worst_token_logp_fp32']} bf16={w['worst_token_logp_bf16']} "
              f"min_logp={w['min_logp_fp32']} c_len={w['c_len']}")


@app.function(image=image, gpu=None, secrets=secrets, timeout=10 * 60)
def decode_tokens(ids: list) -> dict:
    """Decode token IDs to see whether the loss-explosion 'worst tokens' are
    byte-level/unicode-fragment tokens (math LaTeX/unicode) vs normal text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    out = {}
    for i in ids:
        s = tok.convert_ids_to_tokens(int(i))
        d = tok.decode([int(i)])
        out[int(i)] = {"token": repr(s), "decoded": repr(d)}
    for i in ids:
        print(f"[DECODE] id={i:>5} token={out[int(i)]['token']:>14} decoded={out[int(i)]['decoded']}", flush=True)
    return out


@app.local_entrypoint()
def launch_modal_decode_tokens():
    """Decode the loss-explosion worst-token IDs."""
    ids = [16, 17, 19, 22, 24, 87, 92, 578, 1124, 3417]
    res = decode_tokens.remote(ids)
    print("=== decoded worst tokens ===")
    for i in ids:
        print(f"  id={i} token={res[i]['token']} decoded={res[i]['decoded']}")


@app.function(image=image, gpu=None, timeout=10 * 60)
def dump_trl_logp_source() -> str:
    """Dump TRL's _get_per_token_logps_and_entropies source to pin the old-logp bug."""
    import inspect, trl
    from trl.trainer.grpo_trainer import GRPOTrainer
    src = inspect.getsource(GRPOTrainer._get_per_token_logps_and_entropies)
    print(f"TRL version: {trl.__version__}")
    print(src, flush=True)
    return src


@app.local_entrypoint()
def launch_modal_dump_trl():
    dump_trl_logp_source.remote()


@app.function(image=image, gpu="H200", secrets=secrets, timeout=60 * 60)
def three_way_logp_probe() -> dict:
    """Decide which logp path is correct. Generate real math completions (ending
    in \\boxed{...}), then compute each completion token's logp two ways on the
    SAME inputs:
      A = standard full forward, logit[t-1] -> softmax -> token[t]   (== 'old'/ground truth)
      B = the packed-and-sliced shortcut used by the loss (== 'new'): pack all
          sequences into (1,T) w/ position resets, base-model forward, manually
          slice each completion's hidden states by (p_len,c_len) offsets, lm_head.
    Pack MULTIPLE sequences so a multi-sequence offset bug (which a single-seq
    test can't see) would show. For each sequence report the worst-divergence
    token: A's logp, B's logp, the token, and distance from the end. Whichever
    of A/B is the outlier (esp. on end-of-answer digits/braces) is the broken one."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    lm_head = model.lm_head
    dev = "cuda"

    prompts = [
        "Solve. Put your final answer in \\boxed{}.\nWhat is the remainder when $2^{2024}$ is divided by $7$?",
        "Solve. Put your final answer in \\boxed{}.\nCompute $\\sum_{k=1}^{10} k^2$.",
        "Solve. Put your final answer in \\boxed{}.\nFind the number of positive divisors of $360$.",
    ]
    prompts_ids, comps_ids = [], []
    for pr in prompts:
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=350, do_sample=True,
                                 temperature=0.7, top_p=1.0, top_k=0)
        prompts_ids.append(ids[0])
        comps_ids.append(gen[0, ids.shape[1]:])

    def logp_A(seq, p_len, c_len):  # standard full forward
        with torch.no_grad():
            logits = model(seq.unsqueeze(0)).logits[0].float()   # (L, V)
        lp = torch.log_softmax(logits, dim=-1)
        idx = torch.arange(p_len - 1, p_len + c_len - 1, device=dev)
        tgt = seq[p_len:p_len + c_len]
        return lp[idx].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    # B: packed multi-seq forward + manual extraction (mirrors _packed_compute_loss)
    seqs = [torch.cat([p, c]) for p, c in zip(prompts_ids, comps_ids)]
    packed = torch.cat(seqs).unsqueeze(0)
    pos = torch.cat([torch.arange(s.shape[0], device=dev) for s in seqs]).unsqueeze(0)
    with torch.no_grad():
        hidden = model.model(input_ids=packed, position_ids=pos, use_cache=False).last_hidden_state[0]

    results = []
    offset = 0
    for k, (p, c) in enumerate(zip(prompts_ids, comps_ids)):
        p_len, c_len = p.shape[0], c.shape[0]
        seq = seqs[k]
        a = logp_A(seq, p_len, c_len)
        # B extraction
        hs = hidden[offset + p_len - 1: offset + p_len + c_len - 1]
        with torch.no_grad():
            b = torch.log_softmax(lm_head(hs.unsqueeze(0)).float(), dim=-1)[0]
            b = b.gather(-1, c.unsqueeze(-1)).squeeze(-1)
        offset += seq.shape[0]
        d = (a - b).abs()
        j = d.argmax().item()
        tokstr = tok.decode([int(c[j].item())])
        results.append({
            "pos_in_pack": k, "c_len": c_len, "max|A-B|": round(d.max().item(), 2),
            "n_gap_gt5": int((d > 5).sum().item()),
            "worst_tok": repr(tokstr), "worst_from_end": c_len - j,
            "A_logp": round(a[j].item(), 2), "B_logp": round(b[j].item(), 2),
        })
    for r in results:
        print(f"[3WAY] pos={r['pos_in_pack']} c_len={r['c_len']} max|A-B|={r['max|A-B|']:>7} "
              f"n>5={r['n_gap_gt5']} worst_tok={r['worst_tok']} from_end={r['worst_from_end']} "
              f"A(full)={r['A_logp']} B(packed)={r['B_logp']}", flush=True)
    return {"results": results}


@app.local_entrypoint()
def launch_modal_three_way_probe():
    res = three_way_logp_probe.remote()
    print("=== three-way logp probe: A=full-forward(=old/truth)  B=packed-shortcut(=new) ===")
    for r in res["results"]:
        print(f"  pos={r['pos_in_pack']} c_len={r['c_len']} max|A-B|={r['max|A-B|']} "
              f"n>5={r['n_gap_gt5']} worst={r['worst_tok']} from_end={r['worst_from_end']} "
              f"A(full)={r['A_logp']} B(packed)={r['B_logp']}")


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=60 * 60)
def method_old_logp_probe(plan: dict) -> dict:
    """STRONGEST verification of the fix: call the ACTUAL trainer method
    `_packed_per_token_logps` (bound to a stub holding the real loaded fixtest
    checkpoint, real temperature 0.7, real max_tokens budget) on real math
    completions, and compare its per-token old-logps against a clean
    single-sequence reference forward: model(full_seq).logits -> /T ->
    log_softmax -> gather. Pads MULTIPLE sequences (left-padded prompts,
    right-padded completions) exactly as trl_overrides does, so any
    multi-sequence offset / microbatch-row-mapping bug shows. Reports per-seq
    max |method - reference|. Pass requires < 0.1 nat."""
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ckpt = plan["ckpt"]
    base_model = plan.get("base_model", "Qwen/Qwen3-8B")
    mlp_config = plan.get("mlp_config", "m64")
    temperature = plan.get("temperature", 0.7)
    n_seq = plan.get("n_seq", 4)
    dev = "cuda"

    # Force flash_attention_2 so the packed varlen forward path is exercised
    # exactly as in training (load_gradient_routing_model uses plain from_pretrained).
    _orig_from_pretrained = AutoModelForCausalLM.from_pretrained

    def _fa2_from_pretrained(*a, **kw):
        kw.setdefault("attn_implementation", "flash_attention_2")
        kw.setdefault("torch_dtype", torch.bfloat16)
        return _orig_from_pretrained(*a, **kw)

    AutoModelForCausalLM.from_pretrained = staticmethod(_fa2_from_pretrained)
    try:
        from eval_utils import load_gradient_routing_model
        model = load_gradient_routing_model(ckpt, base_model=base_model, mlp_config=mlp_config)
    finally:
        AutoModelForCausalLM.from_pretrained = _orig_from_pretrained
    model = model.to(dev).eval()

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Generate real math completions (some long, ending in \boxed{...}).
    prompts = [
        "Solve. Put your final answer in \\boxed{}.\nWhat is the remainder when $2^{2024}$ is divided by $7$?",
        "Solve. Put your final answer in \\boxed{}.\nCompute $\\sum_{k=1}^{10} k^2$.",
        "Solve. Put your final answer in \\boxed{}.\nFind the number of positive divisors of $360$.",
        "Solve. Put your final answer in \\boxed{}.\nWhat is $\\binom{8}{3}$?",
    ][:n_seq]
    prompt_lists, comp_lists = [], []
    for pr in prompts:
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=400, do_sample=True,
                                 temperature=temperature, top_p=1.0, top_k=0)
        prompt_lists.append(ids[0])
        comp_lists.append(gen[0, ids.shape[1]:])

    # Build padded batch tensors EXACTLY as trl_overrides: prompts left-padded,
    # completions right-padded, masks accordingly.
    def left_pad(seqs, pad_id):
        L = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(seqs), L), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            out[i, L - s.shape[0]:] = s
            mask[i, L - s.shape[0]:] = 1
        return out, mask

    def right_pad(seqs, pad_id):
        L = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((len(seqs), L), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            out[i, :s.shape[0]] = s
            mask[i, :s.shape[0]] = 1
        return out, mask

    prompt_ids, prompt_mask = left_pad(prompt_lists, tok.pad_token_id)
    completion_ids, completion_mask = right_pad(comp_lists, tok.pad_token_id)

    # Build a stub exposing exactly the attributes _packed_per_token_logps uses.
    from train import SampleGRPOTrainer

    class _Acc:
        def unwrap_model(self, m):
            return m

    class _Stub:
        pass

    stub = _Stub()
    stub.temperature = temperature
    stub._max_tokens_per_microbatch = plan.get("max_tokens", 12000)
    stub.accelerator = _Acc()
    method = SampleGRPOTrainer._packed_per_token_logps.__get__(stub, _Stub)

    old_ptl = method(model, prompt_ids, prompt_mask, completion_ids, completion_mask)

    # Reference A: clean single-sequence FULL-MODEL forward (lm_head fused),
    #   model(full).logits -> /T -> log_softmax. Dense (non-packed) attention.
    # Reference B: SOLO-PACKED forward through the SAME base-model path the method
    #   uses (model.model + position_ids reset, lm_head applied manually). This
    #   isolates multi-sequence packing/offset bugs from FA2 dense-vs-varlen bf16
    #   numerical noise: B differs from the method ONLY by multi-seq packing.
    unwrapped = model
    results = []
    for i, (p, c) in enumerate(zip(prompt_lists, comp_lists)):
        p_len, c_len = p.shape[0], c.shape[0]
        full = torch.cat([p, c]).unsqueeze(0)
        with torch.no_grad():
            logits = unwrapped(full).logits[0].float() / temperature
        lp = torch.log_softmax(logits, dim=-1)
        idx = torch.arange(p_len - 1, p_len + c_len - 1, device=dev)
        refA = lp[idx].gather(-1, c.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            pos = torch.arange(full.shape[1], device=dev).unsqueeze(0)
            hs = unwrapped.model(input_ids=full, position_ids=pos,
                                 use_cache=False).last_hidden_state[0]
            hsel = hs[p_len - 1: p_len + c_len - 1]
            lgB = unwrapped.lm_head(hsel.unsqueeze(0)).float()[0] / temperature
            refB = torch.log_softmax(lgB, dim=-1).gather(
                -1, c.unsqueeze(-1)).squeeze(-1)

        got = old_ptl[i, :c_len]
        dA = (refA - got).abs()
        dB = (refB - got).abs()
        dAB = (refA - refB).abs()  # full-dense vs solo-packed (no method involved)
        j = dA.argmax().item()
        print(f"    [xcheck seq={i}] max|refA-refB|(dense-vs-solopacked)={dAB.max().item():.4f} "
              f"mean={dAB.mean().item():.5f}; max|method-solopacked|={dB.max().item():.4f}",
              flush=True)
        results.append({
            "seq": i, "c_len": int(c_len),
            "max_abs_diff": round(dA.max().item(), 4),       # vs full dense forward
            "mean_abs_diff": round(dA.mean().item(), 5),
            "n_gap_gt_0.1": int((dA > 0.1).sum().item()),
            "max_abs_diff_solopacked": round(dB.max().item(), 4),  # vs solo-packed
            "mean_abs_diff_solopacked": round(dB.mean().item(), 5),
            "n_gap_gt_0.1_solopacked": int((dB > 0.1).sum().item()),
            "worst_tok": repr(tok.decode([int(c[j].item())])),
            "worst_from_end": int(c_len - j),
            "ref_logp": round(refA[j].item(), 3),
            "method_logp": round(got[j].item(), 3),
        })
    for r in results:
        print(f"[METHOD-OLD] seq={r['seq']} c_len={r['c_len']} "
              f"vsFULL(max={r['max_abs_diff']} mean={r['mean_abs_diff']} n>0.1={r['n_gap_gt_0.1']}) "
              f"vsSOLOPACKED(max={r['max_abs_diff_solopacked']} mean={r['mean_abs_diff_solopacked']} "
              f"n>0.1={r['n_gap_gt_0.1_solopacked']}) worst={r['worst_tok']} from_end={r['worst_from_end']} "
              f"ref={r['ref_logp']} method={r['method_logp']}", flush=True)
    overall_max = max(r["max_abs_diff"] for r in results)
    overall_max_sp = max(r["max_abs_diff_solopacked"] for r in results)
    print(f"[METHOD-OLD] OVERALL max|diff| vs FULL-dense = {overall_max}; "
          f"vs SOLO-PACKED = {overall_max_sp} "
          f"(multi-seq packing is correct if vs-SOLO-PACKED is tiny)", flush=True)
    return {"results": results, "overall_max_abs_diff": overall_max,
            "overall_max_abs_diff_solopacked": overall_max_sp}


@app.local_entrypoint()
def launch_method_old_logp_probe():
    plan = {"ckpt": f"{OUTPUT_REMOTE}/math_l5_fixtest/math_l5_fixtest_s22/checkpoint-3200",
            "base_model": "Qwen/Qwen3-8B", "mlp_config": "m64",
            "temperature": 0.7, "n_seq": 4}
    res = method_old_logp_probe.remote(plan)
    print("=== method_old_logp_probe: _packed_per_token_logps vs references ===")
    for r in res["results"]:
        print(f"  seq={r['seq']} c_len={r['c_len']} vsFULL(max={r['max_abs_diff']} n>0.1={r['n_gap_gt_0.1']}) "
              f"vsSOLOPACKED(max={r['max_abs_diff_solopacked']} n>0.1={r['n_gap_gt_0.1_solopacked']}) "
              f"worst={r['worst_tok']} ref={r['ref_logp']} method={r['method_logp']}")
    print(f"  OVERALL max|diff| vs FULL-dense = {res['overall_max_abs_diff']}; "
          f"vs SOLO-PACKED = {res['overall_max_abs_diff_solopacked']}")


@app.local_entrypoint()
def smoke_test():
    """Build the image (first time) and verify the container env. Run before launch_modal_3envs."""
    result = smoke.remote()
    import json
    print(json.dumps(result, indent=2))


def _dispatch_sweep(sweep_module_name: str, sweep_name: str, gpu: str = "H100"):
    """Shared launch logic: import a sweep config and dispatch each run as a Modal call.
    gpu selects the training function (H100 = train_one, H200 = train_one_h200)."""
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs
    fn = train_one_h200 if gpu == "H200" else train_one

    print(f"[modal] dispatching {len(runs)} runs (sweep={sweep_name}, gpu={gpu})")
    for r in runs:
        print(f"  - {r['run_name']}")

    # .starmap dispatches all in parallel; each gets its own container/GPU.
    results = list(fn.starmap([(r, sweep_name) for r in runs]))
    for res in results:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")
    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        print(f"[modal] {len(failures)} run(s) failed")
    else:
        print(f"[modal] all {len(results)} runs ok")
    print(f"[modal] sync back: modal volume get gr-modal-pilot / {REPO_LOCAL}/output/")


@app.local_entrypoint()
def launch_modal_3envs():
    """6 runs: object_qa + addition_v2 + topic_contains × 2 seeds, exclusive routing."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_3envs_exclusive_nocoh_1k",
        "retrain_gr_modal_3envs_exclusive_nocoh_1k",
    )


@app.local_entrypoint()
def launch_modal_all_classic():
    """14 runs: 7 envs × 2 seeds, classic routing (no coherence, max_steps=1000)."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_all_classic_nocoh_1k",
        "retrain_gr_modal_all_classic_nocoh_1k",
    )


@app.local_entrypoint()
def launch_modal_6envs_classic_coh():
    """12 runs: 6 envs (topic skipped) × 2 seeds, classic + coherence enabled, max_steps=1000."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_6envs_classic_coh_1k",
        "retrain_gr_modal_6envs_classic_coh_1k",
    )


@app.local_entrypoint()
def launch_modal_6envs_excl_coh():
    """12 runs: 6 envs (topic skipped) × 2 seeds, exclusive + coherence enabled, max_steps=1000."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_6envs_excl_coh_1k",
        "retrain_gr_modal_6envs_excl_coh_1k",
    )


@app.local_entrypoint()
def launch_modal_filter_addition_smoke():
    """Smoke test: 1 filter-baseline addition_v2 run, max_steps=20, validates
    config+env+reward+filter_baseline path on Modal before the full 3-seed job."""
    from sweeps.filter_baseline_addition_v2 import runs as _runs
    pilot = dict(_runs[0])
    pilot["max_steps"] = 20
    pilot["save_steps"] = 20
    pilot["eval_every"] = 10
    pilot["run_name"] = "smoke_" + _runs[0]["run_name"]
    print(f"[smoke] launching {pilot['run_name']} (max_steps=20)")
    res = train_one.remote(pilot, "filter_baseline_7envs")
    print(f"[smoke] result: {res}")


@app.local_entrypoint()
def launch_modal_filter_addition_full():
    """3 runs: addition_v2 × 3 seeds, Weak Filtering (filter_baseline, renorm).
    Output -> /output/filter_baseline_7envs/<run_name>/ (matches the figure
    aggregator path so the runs are drop-in for proto_pareto_data.py)."""
    _dispatch_sweep(
        "sweeps.filter_baseline_addition_v2",
        "filter_baseline_7envs",
    )


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=60 * 60,  # 1h max per eval; 6 forget_scales x 1000 samples ~5-15 min
)
def eval_one(plan: dict) -> dict:
    """Run posthoc forget-scale eval on a saved checkpoint.

    plan: {run_name, env, det, ckpt_step, train_sweep, eval_sweep, n_eval,
           forget_scales}.
    Reads /output/<train_sweep>/<run_name>/checkpoint-<step>; writes
    /output/<eval_sweep>/<run_name>.jsonl (one record per forget_scale).
    """
    import os
    import shlex
    import subprocess
    import sys
    import time

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    run_name = plan["run_name"]
    train_sweep = plan["train_sweep"]
    eval_sweep = plan["eval_sweep"]
    ckpt_step = plan["ckpt_step"]
    n_eval = plan["n_eval"]
    scales = plan["forget_scales"]

    ckpt = os.path.join(OUTPUT_REMOTE, train_sweep, run_name, f"checkpoint-{ckpt_step}")
    out_dir = os.path.join(OUTPUT_REMOTE, eval_sweep)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{run_name}.jsonl")
    log_path = os.path.join(log_dir, f"{run_name}.log")

    assert os.path.isdir(ckpt), f"checkpoint missing: {ckpt}"

    cmd = [
        sys.executable, "eval_utils.py",
        "--model_path", ckpt,
        "--n_eval", str(n_eval),
        "--forget_scales", ",".join(s if isinstance(s, str) else f"{s:g}" for s in scales),
        "--output", out_jsonl,
    ]

    t0 = time.time()
    with open(log_path, "w", buffering=1) as logf:
        logf.write(f"# cmd={shlex.join(cmd)}\n\n")
        proc = subprocess.run(cmd, cwd=REPO_REMOTE,
                              stdout=logf, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    vol.commit()
    return {"run_name": run_name, "rc": proc.returncode,
            "duration_s": dur, "out_jsonl": out_jsonl}


_ENV_DET = {
    "persona_qa":     "flattery_any",
    "sorting_copy":   "sorting_copy_threshold",
    "repeat_extra":   "repeat_detector",
    "cities_qa":      "sycophancy_any",
    "object_qa":      "sycophancy_any",
    "addition_v2":    "sycophancy_any",
    "topic_contains": "topic_contains_detector",
}


def _canonical_eval_plans():
    """Build eval plans from the canonical-steps training sweep.

    For each run, ckpt_step = max_steps from the sweep config (2000 for
    long-train envs, 1000 for repeat_extra + topic_contains).
    """
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import (
        runs, _yamls,
    )

    yaml_to_env = {v: k for k, v in _yamls.items()}
    train_sweep = "retrain_gr_modal_all_classic_nocoh_canonical_steps"
    eval_sweep = "gr_forget_scale_eval/canonical_5seed_1k_samples"

    plans = []
    for r in runs:
        env = yaml_to_env[r["config"]]
        plans.append({
            "run_name":      r["run_name"],
            "env":           env,
            "det":           _ENV_DET[env],
            "seed":          r["seed"],
            "ckpt_step":     r["max_steps"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    return plans


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,  # 2h max; batched per-run trajectory eval
)
def eval_trajectory(plan: dict) -> dict:
    """Evaluate every checkpoint of a single run at ONE forget_scale, in a
    single container (amortizing vLLM init across all checkpoints).

    plan: {run_name, env, det, train_sweep, eval_sweep, ckpt_steps:[int],
           forget_scale: float, n_eval: int}.
    Output: {eval_sweep}/{run_name}.jsonl with one record per checkpoint.
    """
    import os, shlex, subprocess, sys, time
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    rn = plan["run_name"]
    train_sweep = plan["train_sweep"]
    eval_sweep = plan["eval_sweep"]
    ckpt_steps = plan["ckpt_steps"]
    fs = plan["forget_scale"]
    n_eval = plan["n_eval"]

    out_dir = os.path.join(OUTPUT_REMOTE, eval_sweep)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{rn}.jsonl")
    log_path = os.path.join(log_dir, f"{rn}.log")

    incremental = bool(plan.get("incremental", False))
    done_steps: set = set()
    if incremental and os.path.exists(out_jsonl):
        # Read existing rows so we can skip checkpoints already evaluated.
        import json as _json
        with open(out_jsonl) as f:
            for line in f:
                try:
                    done_steps.add(int(_json.loads(line).get("step", -1)))
                except Exception:
                    pass
    elif os.path.exists(out_jsonl):
        # Default behavior: start fresh (eval_utils.py appends to --output).
        os.remove(out_jsonl)

    t0 = time.time()
    rc_any_fail = 0
    with open(log_path, "w", buffering=1) as logf:
        for step in ckpt_steps:
            if step in done_steps:
                logf.write(f"[skip] checkpoint-{step} already evaluated in {out_jsonl}\n")
                continue
            ckpt = os.path.join(OUTPUT_REMOTE, train_sweep, rn, f"checkpoint-{step}")
            if not os.path.isdir(ckpt):
                logf.write(f"[skip] missing ckpt {ckpt}\n")
                continue
            cmd = [
                sys.executable, "eval_utils.py",
                "--model_path", ckpt,
                "--n_eval", str(n_eval),
                "--forget_scales", f"{fs:g}",
                "--output", out_jsonl,
            ]
            logf.write(f"\n=== checkpoint-{step}, forget_scale={fs:g} ===\n")
            logf.write(f"# cmd={shlex.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=REPO_REMOTE,
                                  stdout=logf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                rc_any_fail = proc.returncode

    dur = time.time() - t0
    vol.commit()
    return {"run_name": rn, "n_ckpts": len(ckpt_steps),
            "rc": rc_any_fail, "duration_s": dur}


def _canonical_5seed_optimum_plans():
    """Build per-run trajectory eval plans for the canonical 5seed sweep.
    For each (env, seed), pick the per-seed optimum forget_scale from the
    existing final-checkpoint eval results, then enumerate all checkpoints
    for that run."""
    import json
    import os
    from collections import defaultdict
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import runs

    src = os.path.join(
        REPO_LOCAL,
        "output/gr_forget_scale_eval/canonical_5seed_1k_samples/results.jsonl",
    )
    by_es = defaultdict(list)
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("retain") is None or r.get("hack_overall") is None:
                continue
            by_es[(r["env"], r["seed"])].append(r)
    optimum = {}
    for k, rs in by_es.items():
        best = max(rs, key=lambda x: x["retain"] - 2 * x["hack_overall"])
        optimum[k] = float(best["forget_scale"])

    train_sweep = "retrain_gr_modal_all_classic_nocoh_canonical_steps"
    eval_sweep = "gr_forget_scale_eval/canonical_5seed_trajectory_optimum"

    plans = []
    for r in runs:
        env, seed = None, r["seed"]
        for e in ("persona_qa", "sorting_copy", "repeat_extra",
                  "cities_qa", "object_qa", "addition_v2", "topic_contains"):
            if r["run_name"].startswith(e):
                env = e; break
        assert env is not None, r["run_name"]
        fs = optimum.get((env, seed))
        if fs is None:
            continue
        max_step = r["max_steps"]
        ckpt_steps = list(range(100, max_step + 1, 100))
        plans.append({
            "run_name": r["run_name"],
            "env": env,
            "det": _ENV_DET[env],
            "seed": seed,
            "train_sweep": train_sweep,
            "eval_sweep": eval_sweep,
            "ckpt_steps": ckpt_steps,
            "forget_scale": fs,
            "n_eval": 1000,
        })
    return plans


@app.local_entrypoint()
def launch_eval_canonical_5seed_trajectory_smoke():
    """Single-run trajectory eval to verify the per-container time/cost estimate.
    Picks one 2k-step run (20 ckpts) so we see the full amortization curve."""
    plans = _canonical_5seed_optimum_plans()
    pilot = next(p for p in plans if "_2k_s1" in p["run_name"]
                 and p["forget_scale"] > 0.0)
    print(f"[smoke-trajectory] dispatching 1 run:")
    print(f"  - {pilot['run_name']}  f={pilot['forget_scale']:.2g}  "
          f"{len(pilot['ckpt_steps'])} ckpts")
    res = eval_trajectory.remote(pilot)
    tag = "ok" if res["rc"] == 0 else f"FAIL(rc={res['rc']})"
    print(f"  result: {tag}  {res['n_ckpts']} ckpts  ({res['duration_s']:.1f}s)")
    print(f"  -> per-ckpt avg: {res['duration_s']/max(1,res['n_ckpts']):.1f}s")


@app.local_entrypoint()
def launch_eval_canonical_5seed_trajectory():
    """Per-checkpoint partial-forget trajectory eval for the canonical 5seed
    sweep. One container per run; each container loads vLLM once and
    sequentially evals all that run's checkpoints at its per-seed optimum
    forget_scale. ~35 containers, ~15 min wall, ~$20."""
    plans = _canonical_5seed_optimum_plans()
    print(f"[modal-eval-trajectory] dispatching {len(plans)} run-trajectories")
    for p in plans:
        print(f"  - {p['run_name']:80s}  f={p['forget_scale']:.2g}  "
              f"{len(p['ckpt_steps'])} ckpts")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")
    fails = [r for r in results if r["rc"] != 0]
    if fails:
        print(f"[modal-eval-trajectory] {len(fails)} run(s) had failures")
    else:
        print(f"[modal-eval-trajectory] all {len(results)} run-trajectories ok")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_smoke():
    """1-seed, max_steps=20 pilot of the leetcode_rh classic+no-coh sweep.

    Validates the full Qwen3-8B + vLLM + leetcode evaluator pipeline before
    spending $45 on the 5-seed full run. Wall ETA ~15-20 min.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base,
        "max_steps": 20,
        "save_steps": 10,
        "eval_every": 10,
        "run_name": "smoke_pilot_" + base["run_name"],
    }
    print(f"[smoke] dispatching 1 pilot run:")
    print(f"  - {pilot['run_name']}  (max_steps={pilot['max_steps']})")
    res = train_one.remote(pilot, "leetcode_array_classic_nocoh_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_full():
    """5 H100 runs: leetcode_rh_array classic routing + no coherence,
    seeds 22/100/300/7/17, max_steps=3200. ~$45, ~2.5-3h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_nocoh",
        "leetcode_array_classic_nocoh",
    )


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_alpha2_smoke():
    """1-seed, max_steps=20 pilot of the α=2 bad-pass loss scaling sweep.

    Verifies bad_pass_loss_scale=2.0 plumbs through end-to-end before
    spending on the full 5-seed run.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh_alpha2.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base,
        "max_steps": 20,
        "save_steps": 10,
        "eval_every": 10,
        "run_name": "smoke_pilot_" + base["run_name"],
    }
    print(f"[smoke] dispatching 1 pilot run:")
    print(f"  - {pilot['run_name']}  (max_steps={pilot['max_steps']}, "
          f"bad_pass_loss_scale={pilot.get('bad_pass_loss_scale')})")
    res = train_one.remote(pilot, "leetcode_array_classic_nocoh_alpha2_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_alpha2_full():
    """5 H100 runs: classic routing + no coherence + α=2 bad-pass loss
    scaling. Same seeds + steps as the α=1 baseline so the α effect is
    isolated. ~$45, ~2.5-3h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_nocoh_alpha2",
        "leetcode_array_classic_nocoh_alpha2",
    )


_LEETCODE_DET = {
    "leetcode_rh_array": "leetcode_feature_conditional_array",
}

# Full 8-seed cohort for the KL-coh β=0.1 / NoRP deployment figure.
_KLCOH_SEEDS = [22, 100, 300, 7, 17, 33, 44, 55]
# Per-seed optimal deployment forget scale (from the 6-scale post-hoc sweep).
# Original 3 are measured; new seeds default to 0.4 until their 6-scale eval
# is run (launch_modal_eval_leetcode_excl_kl_coh_b01_newseeds_6scale).
_KLCOH_OPT_FORGET = {22: 0.4, 100: 0.4, 300: 0.2,
                     7: 0.4, 17: 0.2, 33: 0.6, 44: 0.2, 55: 0.4}


def _eval_completed_for_sweep(sweep_module: str, train_sweep: str, eval_sweep: str):
    """Generic 'eval completed runs' helper — used by the heavywd + scaled_classic
    eval entrypoints. Identical pattern to launch_modal_eval_leetcode_exclusive_nocoh_completed
    but parameterized by sweep name."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location("m", f"{REPO_LOCAL}/{sweep_module.replace('.', '/')}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet")
        return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    for p in plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_nocoh_heavywd_completed():
    """6-scale forget-scale eval for completed heavywd runs. Skips runs whose
    eval jsonl already exists on the volume."""
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_exclusive_nocoh_heavywd",
        "leetcode_array_exclusive_nocoh_heavywd",
        "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_heavywd_5seed",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_scaled_classic_nocoh_completed():
    """6-scale forget-scale eval for completed scaled_classic runs."""
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_scaled_classic_nocoh",
        "leetcode_array_scaled_classic_nocoh",
        "gr_forget_scale_eval/leetcode_array_scaled_classic_nocoh_5seed",
    )


@app.local_entrypoint()
def launch_modal_leetcode_excl_nocoh_heavywd_smoke():
    """1-seed, max_steps=20 pilot of excl+nocoh+wd=1.0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh_heavywd.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (wd={pilot.get('weight_decay')})")
    res = train_one.remote(pilot, "leetcode_array_exclusive_nocoh_heavywd_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_excl_nocoh_heavywd_full():
    """5 H100 runs: exclusive routing + nocoh + wd=1.0, 5 seeds."""
    _dispatch_sweep(
        "sweeps.leetcode_array_exclusive_nocoh_heavywd",
        "leetcode_array_exclusive_nocoh_heavywd",
    )


@app.local_entrypoint()
def launch_modal_leetcode_scaled_classic_nocoh_smoke():
    """1-seed, max_steps=20 pilot of scaled_classic + nocoh (α=0.5)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_scaled_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (routing_mode={pilot.get('routing_mode')}, "
          f"unlabeled_forget_grad_scale={pilot.get('unlabeled_forget_grad_scale')})")
    res = train_one.remote(pilot, "leetcode_array_scaled_classic_nocoh_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_scaled_classic_nocoh_full():
    """5 H100 runs: scaled_classic routing + nocoh (α=0.5), 5 seeds."""
    _dispatch_sweep(
        "sweeps.leetcode_array_scaled_classic_nocoh",
        "leetcode_array_scaled_classic_nocoh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_b01_newseeds_6scale():
    """6-scale forget-scale eval (final ckpt 3200, n=500) for the 5 NEW KL-coh
    β=0.1 seeds (7/17/33/44/55), to determine each seed's optimal deployment
    forget scale. Output: gr_forget_scale_eval/leetcode_array_excl_kl_coh_5beta/
    (same dir as the original 3, so all 8 are co-located)."""
    train_sweep = "leetcode_array_excl_kl_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_5beta"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for s in (7, 17, 33, 44, 55):
        plans.append({
            "run_name":      f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}",
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "ckpt_step":     3200,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        500,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    print(f"[modal-eval] dispatching {len(plans)} 6-scale evals (KL-coh β=0.1 new seeds), n=500")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_b01_trajectory():
    """Trajectory eval for the 3 KL-coh β=0.1 runs at per-seed optimal forget
    scale (s22=0.4, s100=0.4, s300=0.2). Every checkpoint 200..3200; n_eval=500.

    Output: gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_trajectory/."""
    train_sweep = "leetcode_array_excl_kl_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_trajectory"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for s in _KLCOH_SEEDS:
        fs = _KLCOH_OPT_FORGET.get(s, 0.4)  # new seeds default 0.4 until 6-scale eval
        rn = f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}"
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, 3200 + 1, 200)),
            "forget_scale":  fs,
            "n_eval":        500,
            "incremental":   True,
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} run-trajectories (KL-coh β=0.1)")
    for p in plans:
        print(f"  - {p['run_name']}  f={p['forget_scale']}  {len(p['ckpt_steps'])} ckpts")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_math_coh_smoke():
    """1-seed, max_steps=20 pilot of dual-env (leetcode routing + math coherence).
    Validates DualEnvReward dispatch + coherence-env prompt swap end-to-end."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_math_coh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    # max_steps=30 so the step-10 eval's async scoring (~60-100s) completes
    # before exit — lets us verify both-env eval writes routing_eval.jsonl.
    pilot = {**base, "max_steps": 30, "save_steps": 30, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (coh={pilot.get('coherence_env')}, "
          f"cspr={pilot.get('coh_samples_per_rollout')}, gpu=H200)")
    res = train_one_h200.remote(pilot, "leetcode_math_coh_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_math_coh_full():
    """2 H200 runs: dual-env leetcode routing + math coherence, seeds 22/100."""
    _dispatch_sweep("sweeps.leetcode_math_coh", "leetcode_math_coh", gpu="H200")


@app.local_entrypoint()
def launch_modal_math_l5_baseline_smoke():
    """1-seed, max_steps=20 pilot of the MATH L5 no-intervention baseline.
    Validates env data ships in the image + math_correct reward runs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/math_l5_baseline.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (env={pilot['environment']}, "
          f"max_completion_length={pilot['max_completion_length']}, gpu=H200)")
    res = train_one_h200.remote(pilot, "math_l5_baseline_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_math_l5_baseline_full():
    """3 H200 runs: MATH L5 no-intervention baseline, seeds 22/100/300, 3200 steps."""
    _dispatch_sweep("sweeps.math_l5_baseline", "math_l5_baseline", gpu="H200")


@app.local_entrypoint()
def launch_modal_math_l5_klgraded_smoke():
    """1-seed, max_steps=20 pilot of the KL+graded MATH L5 baseline.
    Validates the graded reward (math_correct + math_boxed_present) and the
    beta>0 KL-to-base path (disabled-adapter ref) run end-to-end."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/math_l5_kl_graded.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (env={pilot['environment']}, beta={pilot['beta']}, "
          f"config={pilot['config']}, max_completion_length={pilot['max_completion_length']}, gpu=H200)")
    res = train_one_h200.remote(pilot, "math_l5_klgraded_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_math_l5_klgraded_full():
    """3 H200 runs: MATH L5 baseline + beta=1e-3 + graded reward, seeds 22/100/300, 3200 steps."""
    _dispatch_sweep("sweeps.math_l5_kl_graded", "math_l5_klgraded", gpu="H200")


@app.local_entrypoint()
def launch_modal_math_l5_lossdiag():
    """1 H200 run, 80 steps: instrumented loss-explosion diagnostic on plain
    math_l5. Surfaces [LOSSDIAG ...] per-step (loss / advantage / trainer-vs-vLLM
    logp divergence) to find which term explodes. Monitor for early-stop."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/math_l5_lossdiag.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = mod.runs[0]
    print(f"[lossdiag] {pilot['run_name']} (max_steps={pilot['max_steps']}, gpu=H200)")
    res = train_one_h200.remote(pilot, "math_l5_lossdiag")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_math_l5_fixtest():
    """1 H200 run, 3200 steps: confirm the old-logp fix prevents the math
    collapse on the exact baseline that originally crashed (plain math_l5,
    beta=0). math_correct should hold near base instead of crashing to ~0."""
    _dispatch_sweep("sweeps.math_l5_fixtest", "math_l5_fixtest", gpu="H200")


@app.local_entrypoint()
def launch_modal_math_l5_nofastis_diag():
    """1 H200 run, 80 steps: loss-explosion FIX test with --no_fast_vllm_is.
    Tests whether using the trainer's own old logps (vs vLLM sampling logps)
    keeps exp(new-old) bounded and stops the crash."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/math_l5_nofastis_diag.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = mod.runs[0]
    print(f"[nofastis] {pilot['run_name']} (no_fast_vllm_is={pilot['no_fast_vllm_is']}, "
          f"max_steps={pilot['max_steps']}, gpu=H200)")
    res = train_one_h200.remote(pilot, "math_l5_nofastis_diag")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_b01_s5_full():
    """5 H100 runs: KL-coh β=0.1, additional seeds 7/17/33/44/55."""
    _dispatch_sweep("sweeps.leetcode_array_excl_kl_coh_b01_s5",
                    "leetcode_array_excl_kl_coh")  # same output dir as original


@app.local_entrypoint()
def launch_modal_leetcode_norp_s5_full():
    """5 H100 runs: NoRP baseline, additional seeds 7/17/33/44/55."""
    _dispatch_sweep("sweeps.leetcode_array_norp_s5",
                    "leetcode_array_norp")  # same output dir as original


@app.local_entrypoint()
def launch_modal_leetcode_norp_smoke():
    """1-seed, max_steps=20 pilot of the NoRP (routing_mode=none) baseline."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_norp.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (routing_mode={pilot.get('routing_mode')})")
    res = train_one.remote(pilot, "leetcode_array_norp_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_norp_full():
    """3 H100 runs: NoRP baseline (routing_mode=none), seeds 22/100/300."""
    _dispatch_sweep("sweeps.leetcode_array_norp", "leetcode_array_norp")


@app.local_entrypoint()
def launch_modal_eval_leetcode_norp_trajectory():
    """Full-model (forget_scale=1.0) trajectory eval for NoRP, n=500. Matches the
    KL-coh trajectory prompt set + n so the no-intervention line is comparable.
    NoRP has no meaningful ablation, so f=1.0 (full model) is the deployment state."""
    train_sweep = "leetcode_array_norp"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_norp_trajectory"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for s in _KLCOH_SEEDS:
        rn = f"leetcode_rh_array_norp_s{s}"
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, 3200 + 1, 200)),
            "forget_scale":  1.0,
            "n_eval":        500,
            "incremental":   True,  # skip already-evaluated ckpts (safe to re-run as training progresses)
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} NoRP full-model (f=1.0) trajectories, n=500")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_b01_both_trajectory():
    """Both-adapters (forget_scale=1.0) trajectory eval for KL-coh β=0.1, n=500.
    Same checkpoints + same n + same prompt set as the optimal-forget trajectory,
    so the two GRAFT lines are directly comparable. 3 seeds."""
    train_sweep = "leetcode_array_excl_kl_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_both_trajectory"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for s in _KLCOH_SEEDS:
        rn = f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}"
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, 3200 + 1, 200)),
            "forget_scale":  1.0,
            "n_eval":        500,
            "incremental":   True,
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} both-adapter (f=1.0) trajectories, n=500")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_base_model_n500():
    """Base-model eval (both adapter scales = 0) at n=500, matching the prompt
    set used by the KL-coh trajectories for a consistent step-0 anchor."""
    plan = {
        "run_name":      "leetcode_rh_array_gr_excl_nocoh_s22",
        "env":           "leetcode_rh_array",
        "det":           _LEETCODE_DET["leetcode_rh_array"],
        "seed":          22,
        "ckpt_step":     3200,
        "train_sweep":   "leetcode_array_exclusive_nocoh",
        "eval_sweep":    "gr_forget_scale_eval/base_model_eval_n500",
        "n_eval":        500,
        "forget_scales": ["0:0"],
    }
    print(f"[base-eval-n500] dispatching {plan['run_name']} both scales=0, n=500")
    res = eval_one.remote(plan)
    tag = "ok" if res["rc"] == 0 else f"FAIL(rc={res['rc']})"
    print(f"  {tag}  ({res['duration_s']:.1f}s)  out={res['out_jsonl']}")


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_extra_full():
    """6 H100 runs: KL-coh extras at β=0.01 and β=1.0 (3 seeds each)."""
    _dispatch_sweep(
        "sweeps.leetcode_array_excl_kl_coh_extra",
        "leetcode_array_excl_kl_coh_extra",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_all_completed():
    """6-scale forget-scale eval (n=1000) for ALL completed KL-coh runs across
    both the base sweep (β=0.03/0.1/0.3) and the extras (β=0.01/1.0). Idempotent
    via skip-if-already-evaluated checks. Outputs all to a combined eval dir."""
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_5beta"
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_excl_kl_coh",
        "leetcode_array_excl_kl_coh",
        eval_sweep,
    )
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_excl_kl_coh_extra",
        "leetcode_array_excl_kl_coh_extra",
        eval_sweep,
    )


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_smoke():
    """1-seed, max_steps=20 pilot of excl + KL-to-base coherence (β=0.1)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_excl_kl_coh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # Pick a β=0.1 run for the smoke (middle of the sweep).
    base = next(r for r in mod.runs if r["coh_kl_beta"] == 0.1 and r["seed"] == 22)
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (coh_loss_type={pilot.get('coh_loss_type')}, "
          f"coh_kl_beta={pilot.get('coh_kl_beta')}, cspr={pilot.get('coh_samples_per_rollout')})")
    res = train_one.remote(pilot, "leetcode_array_excl_kl_coh_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_full():
    """9 H100 runs: exclusive routing + KL-to-base coherence, 3 betas × 3 seeds."""
    _dispatch_sweep(
        "sweeps.leetcode_array_excl_kl_coh",
        "leetcode_array_excl_kl_coh",
    )


@app.local_entrypoint()
def launch_modal_eval_base_model():
    """Base-model eval: re-use an excl+nocoh checkpoint but set BOTH adapter
    scales to 0, which makes every DualLoRA/DualMLP forward reduce to the
    frozen base layer. Equivalent to evaluating Qwen3-8B with no RL training.

    Output: gr_forget_scale_eval/base_model_eval/leetcode_rh_array_base.jsonl
    (one row with metric prefix 'scales_0_0/...')."""
    plan = {
        "run_name":      "leetcode_rh_array_gr_excl_nocoh_s22",
        "env":           "leetcode_rh_array",
        "det":           _LEETCODE_DET["leetcode_rh_array"],
        "seed":          22,
        "ckpt_step":     3200,
        "train_sweep":   "leetcode_array_exclusive_nocoh",
        "eval_sweep":    "gr_forget_scale_eval/base_model_eval",
        "n_eval":        1000,
        "forget_scales": ["0:0"],  # scales_0_0 mode = base model behavior
    }
    print(f"[base-eval] dispatching {plan['run_name']} ckpt-{plan['ckpt_step']} with both scales=0")
    res = eval_one.remote(plan)
    tag = "ok" if res["rc"] == 0 else f"FAIL(rc={res['rc']})"
    print(f"  {tag}  ({res['duration_s']:.1f}s)  out={res['out_jsonl']}")


@app.local_entrypoint()
def launch_modal_eval_leetcode_exclusive_trajectory_f04():
    """Trajectory eval for the 2 exclusive+nocoh leetcode runs at
    forget_scale=0.4 (the per-seed and cross-seed optimum). All checkpoints
    in [200, 400, ..., 3200]; batched per run; n_eval=256 to keep cost low.
    Output: gr_forget_scale_eval/leetcode_array_exclusive_nocoh_trajectory_f04/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_exclusive_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_trajectory_f04"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        plans.append({
            "run_name":      r["run_name"],
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, r["max_steps"] + 1, 200)),
            "forget_scale":  0.4,
            "n_eval":        256,
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} run-trajectories at f=0.4")
    for p in plans:
        print(f"  - {p['run_name']}  {len(p['ckpt_steps'])} ckpts")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_exclusive_trajectory_f04_resume():
    """Resume the excl+nocoh trajectory eval: skip checkpoints already in the
    output jsonl (incremental=True), evaluate only the missing ones.

    Existing jsonls have steps 200–1400; we'll fill 1600–3200 (~9 ckpts/seed)
    at the same forget_scale=0.4. ETA ~75 min per seed (within 2h timeout)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_exclusive_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_trajectory_f04"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        plans.append({
            "run_name":      r["run_name"],
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, r["max_steps"] + 1, 200)),
            "forget_scale":  0.4,
            "n_eval":        256,
            "incremental":   True,
        })
    print(f"[modal-eval-trajectory-resume] dispatching {len(plans)} run-trajectories at f=0.4 (incremental)")
    for p in plans:
        print(f"  - {p['run_name']}  candidate ckpts={len(p['ckpt_steps'])} (already-done will be skipped)")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_condover_smoke():
    """1-seed, max_steps=20 smoke for the conditional_overwrite suffix sweep.
    Validates the new id-hash partition + suffix pair end-to-end on Modal."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_conditional_overwrite_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
        "run_name": "smoke_" + base["run_name"],
    }
    print(f"[smoke-condover] dispatching: {pilot['run_name']}")
    res = train_one.remote(pilot, "leetcode_conditional_overwrite_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_condover_full():
    """5 H100 runs: classic + no coherence + conditional_overwrite suffix.
    Seeds 22/100/300/7/17."""
    _dispatch_sweep(
        "sweeps.leetcode_conditional_overwrite_classic_nocoh",
        "leetcode_conditional_overwrite_classic_nocoh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_condover_completed():
    """6-scale eval on completed conditional_overwrite runs."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_conditional_overwrite_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    train_sweep = "leetcode_conditional_overwrite_classic_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_conditional_overwrite_classic_nocoh_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)"); continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)"); continue
        plans.append({
            "run_name": rn, "env": "leetcode_rh_array", "det": det,
            "seed": r["seed"], "ckpt_step": ckpt,
            "train_sweep": train_sweep, "eval_sweep": eval_sweep,
            "n_eval": 1000, "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet"); return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_negate_smoke():
    """1-seed, max_steps=20 smoke for the inverted-detector classic+nocoh sweep.
    Validates the negate_tags patch end-to-end on Modal."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh_negate.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base,
        "max_steps": 20,
        "save_steps": 10,
        "eval_every": 10,
        "run_name": "smoke_" + base["run_name"],
    }
    print(f"[smoke-negate] dispatching: {pilot['run_name']}")
    res = train_one.remote(pilot, "leetcode_array_classic_nocoh_negate_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_negate_full():
    """5 H100 runs: classic + no coherence + inverted rh_detector
    (fires on non-Array, ~35% of prompts). Seeds 22/100/300/7/17."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_nocoh_negate",
        "leetcode_array_classic_nocoh_negate",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_nocoh_negate_completed():
    """6-scale eval on completed runs of the inverted-detector sweep."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh_negate.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    train_sweep = "leetcode_array_classic_nocoh_negate"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_nocoh_negate_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)"); continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)"); continue
        plans.append({
            "run_name": rn, "env": "leetcode_rh_array", "det": det,
            "seed": r["seed"], "ckpt_step": ckpt,
            "train_sweep": train_sweep, "eval_sweep": eval_sweep,
            "n_eval": 1000, "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet"); return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_classic_coh_full():
    """5 H100 runs: leetcode_rh_array classic routing + coherence (canonical
    cspr=256 config), seeds 22/100/300/7/17, max_steps=3200.
    ~$100 train + ~$10 eval, ~5.5-6h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_coh",
        "leetcode_array_classic_coh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_coh_completed():
    """6-scale eval on whichever classic+coherence runs have the final
    checkpoint. Skips runs whose eval jsonl already exists."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_coh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_classic_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_coh_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet"); return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    for p in plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_exclusive_nocoh_full():
    """2 H100 runs: leetcode_rh_array exclusive routing + no coherence,
    seeds 22/100, max_steps=3200. ~$35 train + ~$5 eval, ~5 h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_exclusive_nocoh",
        "leetcode_array_exclusive_nocoh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_exclusive_nocoh_completed():
    """Eval whichever exclusive-routing runs have the final checkpoint.
    Skips runs whose eval jsonl already exists on the volume."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_exclusive_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_2seed"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        # Final checkpoint present?
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        # Eval jsonl already exists?
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet")
        return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    for p in plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_nocoh_completed():
    """Posthoc forget-scale eval on whichever leetcode_rh_array classic+no-coh
    runs currently have the FINAL checkpoint (max_steps) saved on the volume.
    Skips in-flight runs. Idempotent — re-run later to pick up new completions.
    6 forget_scales x 1000 samples per eval."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_classic_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_nocoh_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]

    completed_plans = []
    for r in mod.runs:
        rn = r["run_name"]
        ckpt = r["max_steps"]
        # Check the volume for the final checkpoint.
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        # Skip if eval jsonl already exists on the volume (otherwise
        # eval_utils.py's --output append would create duplicate records).
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        completed_plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })

    if not completed_plans:
        print("[modal-eval] no completed runs yet")
        return
    print(f"[modal-eval] dispatching {len(completed_plans)} eval(s) "
          f"(out of {len(mod.runs)} total runs)")
    for p in completed_plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(completed_plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_nocoh():
    """Posthoc forget-scale eval on the 5 leetcode_rh_array classic+no-coh
    checkpoints. 6 forget_scales x 1000 samples; outputs to
    /output/gr_forget_scale_eval/leetcode_array_classic_nocoh_5seed/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_classic_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_nocoh_5seed"

    plans = []
    for r in mod.runs:
        plans.append({
            "run_name":      r["run_name"],
            "env":           "leetcode_rh_array",
            "det":           _LEETCODE_DET["leetcode_rh_array"],
            "seed":          r["seed"],
            "ckpt_step":     r["max_steps"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })

    print(f"[modal-eval] dispatching {len(plans)} leetcode evals")
    for p in plans:
        print(f"  - ckpt-{p['ckpt_step']:>4d}  {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")
    failures = [r for r in results if r["rc"] != 0]
    if failures:
        print(f"[modal-eval] {len(failures)} eval(s) failed")
    else:
        print(f"[modal-eval] all {len(results)} leetcode evals ok")


@app.local_entrypoint()
def launch_eval_canonical_steps_missing():
    """Re-run eval for the 6 runs killed by the prior Modal usage limit.

    5 topic_contains seeds + repeat_extra s4. All have checkpoint-1000 still
    on the volume. New jsonls land alongside the existing 29 in
    /output/gr_forget_scale_eval/canonical_5seed_1k_samples/.
    """
    MISSING = {
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s1",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s2",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s3",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s4",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s5",
        "repeat_extra_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s4",
    }
    plans = [p for p in _canonical_eval_plans() if p["run_name"] in MISSING]
    assert len(plans) == len(MISSING), \
        f"plan count mismatch: {len(plans)} vs {len(MISSING)}"
    print(f"[modal-eval] dispatching {len(plans)} recovery evals")
    for p in plans:
        print(f"  - ckpt-{p['ckpt_step']:>4d}  {p['run_name']}")

    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")
    failures = [r for r in results if r["rc"] != 0]
    if failures:
        print(f"[modal-eval] {len(failures)} eval(s) failed")
    else:
        print(f"[modal-eval] all {len(results)} recovery evals ok")


@app.local_entrypoint()
def launch_eval_canonical_steps():
    """35 evals: one per (env, seed) on the canonical-steps training sweep.

    Each container runs all 6 forget_scales x 1000 samples on the final
    checkpoint (2000 for long-train envs; 1000 for repeat/topic). Outputs
    land at /output/gr_forget_scale_eval/canonical_5seed_1k_samples/<run_name>.jsonl.
    """
    plans = _canonical_eval_plans()
    print(f"[modal-eval] dispatching {len(plans)} evals")
    for p in plans:
        print(f"  - ckpt-{p['ckpt_step']:>4d}  {p['run_name']}")

    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")
    failures = [r for r in results if r["rc"] != 0]
    if failures:
        print(f"[modal-eval] {len(failures)} eval(s) failed")
    else:
        print(f"[modal-eval] all {len(results)} evals ok")
    print(f"[modal-eval] sync back: modal volume get gr-modal-pilot "
          f"gr_forget_scale_eval/canonical_5seed_1k_samples/ "
          f"{REPO_LOCAL}/output/gr_forget_scale_eval/canonical_5seed_1k_samples/ --force")


@app.local_entrypoint()
def launch_modal_all_classic_canonical_steps():
    """14 runs: 7 envs × 2 seeds, classic routing + no coherence, canonical per-env max_steps.

    addition_v2, cities_qa, object_qa, persona_qa, sorting_copy -> 2000 steps
    repeat_extra, topic_contains                                -> 1000 steps
    """
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps",
        "retrain_gr_modal_all_classic_nocoh_canonical_steps",
    )


@app.local_entrypoint()
def launch_modal_eval_judge_retain():
    """Retain-only-vs-both hack eval for the may31 judge runs.

    Question: does the hack survive in the retain-only (forget-ablated)
    deployment? Classic routing zeros retain-grad on judge-FLAGGED samples, so a
    judge-VISIBLE hack (s1: frac_rh~0.48) should be routed into forget and
    largely vanish at forget_scale=0, whereas a judge-BLIND hack (s2: frac_rh
    ~0.03) was trained into both adapters and should persist. kl_coh as a third
    point. Trait/correct are code-exec (no judge calls).

    Evals final ckpt-200 at forget_scale 0 (retain-only) / 0.5 / 1 (both).
    Output: gr_forget_scale_eval/judge_retain_eval/{run_name}.jsonl
    """
    eval_sweep = "gr_forget_scale_eval/judge_retain_eval"
    scales = [0.0, 0.5, 1.0]
    plans = []
    for train_sweep, run_name in [
        ("leetcode_judge_nocoh_classic", "leetcode_judge_nocoh_classic_s1"),
        ("leetcode_judge_nocoh_classic", "leetcode_judge_nocoh_classic_s2"),
        ("leetcode_judge_kl_coh_merged", "leetcode_judge_kl_coh_merged_b0.1_s1"),
    ]:
        plans.append({
            "run_name":      run_name,
            "env":           "leetcode_rh_llm_judge",
            "det":           "llm_judge",
            "ckpt_step":     200,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        64,   # single batched HF generate; n=128 OOMs H100 (79GB)
            "forget_scales": scales,
        })
    print(f"[modal-eval] dispatching {len(plans)} retain-vs-both hack evals (scales={scales}, n=64)")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s) -> {r['out_jsonl']}")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=90 * 60,  # 90m; vLLM gen is fast but allow headroom for big n × many modes
)
def eval_one_vllm(plan: dict) -> dict:
    """Posthoc forget-scale eval via a spawned vLLM server (fast; avoids HF OOM).

    Same plan schema as eval_one, plus optional plan['vllm_gpu_memory'] (default
    0.5). Runs on H200 so the HF model (for weight-sync/scales) + the vLLM server
    co-reside comfortably. Passes --vllm to eval_utils.py.
    """
    import os, shlex, subprocess, sys, time
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    run_name = plan["run_name"]
    train_sweep = plan["train_sweep"]
    eval_sweep = plan["eval_sweep"]
    ckpt_step = plan["ckpt_step"]
    n_eval = plan["n_eval"]
    scales = plan["forget_scales"]
    vgm = plan.get("vllm_gpu_memory", 0.5)

    ckpt = os.path.join(OUTPUT_REMOTE, train_sweep, run_name, f"checkpoint-{ckpt_step}")
    out_dir = os.path.join(OUTPUT_REMOTE, eval_sweep)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{run_name}.jsonl")
    log_path = os.path.join(log_dir, f"{run_name}.log")
    assert os.path.isdir(ckpt), f"checkpoint missing: {ckpt}"

    cmd = [
        sys.executable, "eval_utils.py",
        "--model_path", ckpt,
        "--n_eval", str(n_eval),
        "--forget_scales", ",".join(s if isinstance(s, str) else f"{s:g}" for s in scales),
        "--output", out_jsonl,
        "--vllm", "--vllm_gpu_memory", f"{vgm:g}",
    ]

    t0 = time.time()
    with open(log_path, "w", buffering=1) as logf:
        logf.write(f"# cmd={shlex.join(cmd)}\n\n")
        proc = subprocess.run(cmd, cwd=REPO_REMOTE, stdout=logf, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    vol.commit()
    return {"run_name": run_name, "rc": proc.returncode,
            "duration_s": dur, "out_jsonl": out_jsonl}


@app.local_entrypoint()
def launch_modal_eval_judge_retain_vllm():
    """n=1024 retain-only-vs-both-vs-base eval via vLLM, for the may31 judge runs.

    Modes: forget_scale 0 (retain-only), 1 (both), 0:0 (base = both adapters off,
    same prompts). Answers: did retain-only task performance rise above base, and
    how much hacking survives forget-ablation. Trait/correct are code-exec.
    Output: gr_forget_scale_eval/judge_retain_eval_n1024/{run_name}.jsonl
    """
    eval_sweep = "gr_forget_scale_eval/judge_retain_eval_n1024"
    scales = [0.0, 1.0, "0:0"]
    plans = []
    for train_sweep, run_name in [
        ("leetcode_judge_nocoh_classic", "leetcode_judge_nocoh_classic_s1"),
        ("leetcode_judge_nocoh_classic", "leetcode_judge_nocoh_classic_s2"),
        ("leetcode_judge_kl_coh_merged", "leetcode_judge_kl_coh_merged_b0.1_s1"),
    ]:
        plans.append({
            "run_name": run_name, "env": "leetcode_rh_llm_judge", "det": "llm_judge",
            "ckpt_step": 200, "train_sweep": train_sweep, "eval_sweep": eval_sweep,
            "n_eval": 1024, "forget_scales": scales, "vllm_gpu_memory": 0.5,
        })
    print(f"[modal-eval-vllm] dispatching {len(plans)} n=1024 evals (scales={scales})")
    results = list(eval_one_vllm.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s) -> {r['out_jsonl']}")


@app.local_entrypoint()
def launch_modal_eval_judge_retain_vllm_smoke():
    """Smoke the vLLM posthoc eval path: 1 run (s1), n=16, scales {0,1} — validates
    server spawn + adapter weight-sync + set_scales + generate end-to-end before
    the full n=1024 run."""
    plan = {
        "run_name": "leetcode_judge_nocoh_classic_s1", "env": "leetcode_rh_llm_judge",
        "det": "llm_judge", "ckpt_step": 200,
        "train_sweep": "leetcode_judge_nocoh_classic",
        "eval_sweep": "gr_forget_scale_eval/judge_retain_vllm_smoke",
        "n_eval": 16, "forget_scales": [0.0, 1.0], "vllm_gpu_memory": 0.5,
    }
    print("[modal-eval-vllm-smoke] s1 n=16 scales [0,1] via vLLM")
    r = eval_one_vllm.remote(plan)
    print(f"  rc={r['rc']} dur={r['duration_s']:.1f}s -> {r['out_jsonl']}")


@app.local_entrypoint()
def launch_modal_eval_judge_f02():
    """Add forget_scale=0.2 to the s1/s2 trajectory (n=1024, vLLM). Same fixed
    eval prompts as judge_retain_eval_n1024, so directly comparable. Also picks
    up the new standalone rate/ metrics (solve/compile/hack rates) if present.
    Output: gr_forget_scale_eval/judge_retain_f02/{run_name}.jsonl"""
    eval_sweep = "gr_forget_scale_eval/judge_retain_f02"
    plans = [{
        "run_name": rn, "env": "leetcode_rh_llm_judge", "det": "llm_judge",
        "ckpt_step": 200, "train_sweep": "leetcode_judge_nocoh_classic",
        "eval_sweep": eval_sweep, "n_eval": 1024, "forget_scales": [0.2],
        "vllm_gpu_memory": 0.5,
    } for rn in ["leetcode_judge_nocoh_classic_s1", "leetcode_judge_nocoh_classic_s2"]]
    print(f"[modal-eval-vllm] dispatching {len(plans)} f=0.2 evals (n=1024)")
    results = list(eval_one_vllm.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_judge_baseline_offpolicy_smoke():
    """5-step smoke of the off-policy BASELINE-judge run (validates Together
    baseline judge + off-policy + grad-clip path on H200)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_judge_baseline_offpolicy.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = {**mod.runs[0], "max_steps": 5, "save_steps": 5,
             "run_name": "smoke_judge_baseline_offpolicy"}
    print(f"[smoke] {pilot['run_name']} (baseline judge, off-policy rollout "
          f"{pilot['rollout_batch_size']}/optb {pilot['optimizer_batch_size']}, "
          f"max_grad_norm {pilot['max_grad_norm']}, H200)")
    res = train_one_h200_judge.remote(pilot, "leetcode_judge_baseline_offpolicy_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_judge_highprec_offpolicy_smoke():
    """5-step smoke of the off-policy HIGH-PRECISION-judge run."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_judge_highprec_offpolicy.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = {**mod.runs[0], "max_steps": 5, "save_steps": 5,
             "run_name": "smoke_judge_highprec_offpolicy"}
    print(f"[smoke] {pilot['run_name']} (high-precision judge, off-policy, H200)")
    res = train_one_h200_judge.remote(pilot, "leetcode_judge_highprec_offpolicy_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_judge_baseline_offpolicy_full():
    """Full off-policy BASELINE-judge run: 5 seeds (22/100/300/7/17), H200."""
    _dispatch_sweep_judge("sweeps.leetcode_judge_baseline_offpolicy",
                          "leetcode_judge_baseline_offpolicy")


@app.local_entrypoint()
def launch_modal_judge_highprec_offpolicy_full():
    """Full off-policy HIGH-PRECISION-judge run: 5 seeds (22/100/300/7/17), H200."""
    _dispatch_sweep_judge("sweeps.leetcode_judge_highprec_offpolicy",
                          "leetcode_judge_highprec_offpolicy")


@app.local_entrypoint()
def launch_modal_eval_offpolicy_retain():
    """Forget-scale eval (retain-only=0 / both=1, n=512) over all 10 off-policy
    judge checkpoints (5 baseline + 5 highprec, ckpt-3200), via vLLM. Reports
    solve rate (rate/leetcode_correct) + hack rate (hack_freq trait) at each
    scale. No judge calls (detector metric dropped for posthoc).
    Output: gr_forget_scale_eval/judge_offpolicy_retain/{run_name}.jsonl
    """
    eval_sweep = "gr_forget_scale_eval/judge_offpolicy_retain"
    plans = []
    for prefix, sweep in [("baseline", "leetcode_judge_baseline_offpolicy"),
                          ("highprec", "leetcode_judge_highprec_offpolicy")]:
        for s in (22, 100, 300, 7, 17):
            plans.append({
                "run_name": f"leetcode_judge_{prefix}_offpolicy_s{s}",
                "env": "leetcode_rh_llm_judge", "det": "llm_judge",
                "ckpt_step": 3200, "train_sweep": sweep, "eval_sweep": eval_sweep,
                "n_eval": 512, "forget_scales": [0.0, 1.0], "vllm_gpu_memory": 0.5,
            })
    print(f"[modal-eval-vllm] dispatching {len(plans)} off-policy retain evals (scales=[0,1], n=512)")
    results = list(eval_one_vllm.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_judge_baseline_may31_gc_smoke():
    """5-step smoke: may31 on-policy + grad-clip 0.2 + forget_lr_mult 1.0, BASELINE judge."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", f"{REPO_LOCAL}/sweeps/leetcode_judge_baseline_may31_gc.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = {**mod.runs[0], "max_steps": 5, "save_steps": 5, "run_name": "smoke_baseline_may31_gc"}
    print(f"[smoke] {pilot['run_name']} (baseline judge, on-policy, max_grad_norm={pilot['max_grad_norm']}, forget_lr_mult={pilot['forget_lr_mult']}, H200)")
    res = train_one_h200_judge.remote(pilot, "leetcode_judge_baseline_may31_gc_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"): print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_judge_highprec_may31_gc_smoke():
    """5-step smoke: may31 on-policy + grad-clip 0.2 + forget_lr_mult 1.0, HIGH-PRECISION judge."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", f"{REPO_LOCAL}/sweeps/leetcode_judge_highprec_may31_gc.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pilot = {**mod.runs[0], "max_steps": 5, "save_steps": 5, "run_name": "smoke_highprec_may31_gc"}
    print(f"[smoke] {pilot['run_name']} (highprec judge, on-policy, max_grad_norm={pilot['max_grad_norm']}, H200)")
    res = train_one_h200_judge.remote(pilot, "leetcode_judge_highprec_may31_gc_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"): print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_judge_baseline_may31_gc_full():
    """Full: may31 on-policy + grad-clip 0.2, BASELINE judge, 5 seeds (22/100/300/7/17), H200."""
    _dispatch_sweep_judge("sweeps.leetcode_judge_baseline_may31_gc", "leetcode_judge_baseline_may31_gc")


@app.local_entrypoint()
def launch_modal_judge_highprec_may31_gc_full():
    """Full: may31 on-policy + grad-clip 0.2, HIGH-PRECISION judge, 5 seeds (22/100/300/7/17), H200."""
    _dispatch_sweep_judge("sweeps.leetcode_judge_highprec_may31_gc", "leetcode_judge_highprec_may31_gc")


@app.local_entrypoint()
def launch_modal_eval_may31_gc_baseline_retain():
    """Forget-scale eval (retain-only=0 / both=1, n=512) over the 5 baseline-judge
    may31_gc checkpoints (ckpt-200), via vLLM. Solve rate + hack rate per scale;
    no judge calls. Output: gr_forget_scale_eval/may31_gc_baseline_retain/{run}.jsonl"""
    eval_sweep = "gr_forget_scale_eval/may31_gc_baseline_retain"
    plans = [{
        "run_name": f"leetcode_judge_baseline_may31_gc_s{s}",
        "env": "leetcode_rh_llm_judge", "det": "llm_judge",
        "ckpt_step": 200, "train_sweep": "leetcode_judge_baseline_may31_gc",
        "eval_sweep": eval_sweep, "n_eval": 512, "forget_scales": [0.0, 1.0],
        "vllm_gpu_memory": 0.5,
    } for s in (22, 100, 300, 7, 17)]
    print(f"[modal-eval-vllm] dispatching {len(plans)} baseline may31_gc retain evals (scales=[0,1], n=512)")
    results = list(eval_one_vllm.map(plans))
    for r in results:
        print(f"  {r['run_name']}: {'ok' if r['rc']==0 else 'FAIL(rc=%d)'%r['rc']} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_may31_gc_baseline_f3():
    """Forget-scale eval at 0.0 / 0.1 / 1.0 (n=512) over the 5 baseline-judge
    may31_gc checkpoints (ckpt-200), via vLLM. No judge calls.
    Output: gr_forget_scale_eval/may31_gc_baseline_f3/{run}.jsonl"""
    eval_sweep = "gr_forget_scale_eval/may31_gc_baseline_f3"
    plans = [{
        "run_name": f"leetcode_judge_baseline_may31_gc_s{s}",
        "env": "leetcode_rh_llm_judge", "det": "llm_judge",
        "ckpt_step": 200, "train_sweep": "leetcode_judge_baseline_may31_gc",
        "eval_sweep": eval_sweep, "n_eval": 512, "forget_scales": [0.0, 0.1, 1.0],
        "vllm_gpu_memory": 0.5,
    } for s in (22, 100, 300, 7, 17)]
    print(f"[modal-eval-vllm] dispatching {len(plans)} baseline may31_gc evals (scales=[0,0.1,1], n=512)")
    results = list(eval_one_vllm.map(plans))
    for r in results:
        print(f"  {r['run_name']}: {'ok' if r['rc']==0 else 'FAIL(rc=%d)'%r['rc']} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_may31_gc_base():
    """Base-model anchor: forget_scale 0:0 (both adapters OFF = pure Qwen3-8B) on
    the 5 baseline may31_gc checkpoints, n=512, vLLM. Gives the base solve rate on
    the SAME eval prompts as the retain-only(=0) numbers, to test whether the
    retain adapter actually improves the task. Output: gr_forget_scale_eval/may31_gc_base/"""
    eval_sweep = "gr_forget_scale_eval/may31_gc_base"
    plans = [{
        "run_name": f"leetcode_judge_baseline_may31_gc_s{s}",
        "env": "leetcode_rh_llm_judge", "det": "llm_judge",
        "ckpt_step": 200, "train_sweep": "leetcode_judge_baseline_may31_gc",
        "eval_sweep": eval_sweep, "n_eval": 512, "forget_scales": ["0:0"],
        "vllm_gpu_memory": 0.5,
    } for s in (22, 100, 300, 7, 17)]
    print(f"[modal-eval-vllm] dispatching {len(plans)} base-model (0:0) anchor evals, n=512")
    results = list(eval_one_vllm.map(plans))
    for r in results:
        print(f"  {r['run_name']}: {'ok' if r['rc']==0 else 'FAIL(rc=%d)'%r['rc']} ({r['duration_s']:.1f}s)")
