"""Re-evaluate every checkpoint of a single training run with the corrected
detectable partition. Loads K ckpts into K vLLM slots per batch (concurrent
multi-adapter eval), generates 3 modes per ckpt, scores with leetcode rewards,
writes one row per (step, mode, prompt) to {run_dir}/post_eval.jsonl.

Designed to be invoked once per run (one process per GPU); wrap in a shell
launcher that fans out across runs/GPUs.

Example:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/reeval_one_run.py \
      --run_dir output/verified_vary_m_uh/leetcode_rh_array_cspr64_s22_unhinted_frac0.5 \
      --gpu_id 0
"""
import argparse, glob, json, multiprocessing, os, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--gpu_id", type=int, required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-8B")
    p.add_argument("--mlp_config", default="m64")
    p.add_argument("--batch_ckpts", type=int, default=4,
                   help="Concurrent ckpts per vLLM batch (slots needed = this)")
    p.add_argument("--vllm_gpu_memory", type=float, default=0.7)
    p.add_argument("--max_completion_length", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--out_file", default=None,
                   help="Output jsonl; default {run_dir}/post_eval.jsonl")
    p.add_argument("--ckpt_steps", default=None,
                   help="Comma-separated list of step numbers to eval (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.out_file is None:
        args.out_file = os.path.join(args.run_dir, "post_eval.jsonl")
    run_name = os.path.basename(args.run_dir.rstrip("/"))
    log = lambda *a, **kw: print(f"[{run_name}]", *a, flush=True, **kw)

    # --- Read run config (hint, unhinted_frac, seed) ---
    import yaml
    with open(os.path.join(args.run_dir, "run_config.yaml")) as f:
        run_cfg = yaml.safe_load(f)
    leetcode_hint = run_cfg.get("leetcode_hint", "simple_overwrite_tests_aware")
    # Translate legacy unhinted_frac → hack_frac for the refactored env.
    if "hack_frac" in run_cfg:
        hack_frac = float(run_cfg["hack_frac"])
    else:
        hack_frac = 1.0 - float(run_cfg.get("unhinted_frac", 0.0))
    log(f"hint={leetcode_hint} hack_frac={hack_frac} seed={run_cfg.get('seed')}")

    # --- Find checkpoints ---
    ckpts = sorted(
        glob.glob(os.path.join(args.run_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    if args.ckpt_steps:
        wanted = set(int(s) for s in args.ckpt_steps.split(","))
        ckpts = [c for c in ckpts if int(c.split("-")[-1]) in wanted]
    log(f"running {len(ckpts)} ckpts: steps {[int(c.split('-')[-1]) for c in ckpts]}")

    # --- Spawn vLLM server (subprocess) ---
    K = args.batch_ckpts
    socket_path = f"ipc:///tmp/reeval_gpu{args.gpu_id}_{os.getpid()}.sock"
    ready_file = tempfile.mktemp(prefix="reeval_ready_", suffix=f"_gpu{args.gpu_id}")
    log_dir = os.path.join(args.run_dir, "post_eval_logs")
    os.makedirs(log_dir, exist_ok=True)

    from sweep import _vllm_server_worker
    ctx = multiprocessing.get_context("spawn")
    server_proc = ctx.Process(
        target=_vllm_server_worker,
        args=(args.gpu_id, args.base_model, args.mlp_config, K,
              args.vllm_gpu_memory, socket_path),
        kwargs={"ready_file": ready_file, "log_dir": log_dir, "dtype": "bfloat16"},
        daemon=False,
    )
    server_proc.start()
    log(f"spawned vLLM server pid={server_proc.pid}, waiting for ready file...")
    t0 = time.time()
    while not os.path.exists(ready_file):
        if not server_proc.is_alive():
            raise RuntimeError(
                f"vLLM server died before ready. See {log_dir}/vllm_server.log")
        if time.time() - t0 > 600:
            raise TimeoutError("vLLM server did not become ready in 10 min")
        time.sleep(2)
    log(f"vLLM ready in {time.time()-t0:.1f}s")

    from vllm_client import VLLMClient
    client = VLLMClient(socket_path)
    slot_ids = [client.register() for _ in range(K)]
    log(f"registered slots {slot_ids}")

    # --- Load base model + apply dual-MLP adapter (CPU) ---
    log(f"loading base model {args.base_model} on CPU...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gradient_routing import apply_dual_mlp
    from vllm_utils import MLP_PRESETS
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    preset = MLP_PRESETS[args.mlp_config]
    apply_dual_mlp(model, preset["retain_neurons"], preset["forget_neurons"])
    log("base+adapter loaded")

    # --- Eval prompts ---
    from envs.leetcode import _load_eval_prompts, _get_tags_lookup, leetcode_all_components

    class _A: pass
    eval_args = _A()
    eval_args.leetcode_hint = leetcode_hint
    eval_args.hack_frac = hack_frac
    eval_data = _load_eval_prompts(n=10**6, args=eval_args)
    n_eval = len(eval_data)
    log(f"loaded {n_eval} eval prompts")

    # --- Corrected detectable from rh_classifiable_fn (Array tag membership) ---
    from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable
    rh_cfg = run_cfg.get("rh_detector") or {}
    detector_name = rh_cfg.get("name") if isinstance(rh_cfg, dict) else None
    # Fall back to reading from the YAML config file
    if detector_name is None:
        cfg_path = run_cfg.get("config")
        if cfg_path and not os.path.isabs(cfg_path):
            cfg_path = os.path.join("/workspace/small-rl", cfg_path)
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg_yaml = yaml.safe_load(f)
            rh_cfg = cfg_yaml.get("rh_detector") or {}
            detector_name = rh_cfg.get("name")
    log(f"rh_detector name={detector_name}")
    detector_params = rh_cfg.get("params", {}) if isinstance(rh_cfg, dict) else {}
    classifiable_fn = None
    if detector_name in RH_CLASSIFIABLE_REGISTRY:
        classifiable_fn = get_rh_classifiable(detector_name, **detector_params)

    # Build per-row corrected detectable
    cols = {k: [r.get(k) for r in eval_data] for k in eval_data[0].keys()}
    if classifiable_fn is not None:
        flags = list(classifiable_fn(**cols))
        assert len(flags) == n_eval
        detectable_corrected = [bool(f) for f in flags]
    else:
        # No classifiable — fall back to per-row detectable from data file
        detectable_corrected = [bool(r.get("detectable")) for r in eval_data]
    n_mon = sum(detectable_corrected); n_unmon = n_eval - n_mon
    log(f"detectable_corrected: {n_mon} monitored / {n_unmon} unmonitored")

    # --- Tokenize prompts (chat template) ---
    from eval_utils import _tokenize_prompts_for_vllm
    prompts = [r["prompt"] for r in eval_data]
    prompt_ids = _tokenize_prompts_for_vllm(tokenizer, prompts)
    log(f"tokenized {n_eval} prompts")

    # --- Reward kwargs (constant across ckpts) ---
    reward_kwargs = {
        "gt_answer": [r["gt_answer"] for r in eval_data],
        "setup_code": [r["setup_code"] for r in eval_data],
        "test_func_name": [r["test_func_name"] for r in eval_data],
    }

    # --- Modes ---
    modes = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]

    # --- Process batches ---
    out_path = args.out_file
    if os.path.exists(out_path):
        os.remove(out_path)
    log(f"output -> {out_path}")

    from eval_utils import _load_state_dict
    t_run = time.time()
    n_batches = (len(ckpts) + K - 1) // K
    for bi in range(n_batches):
        batch_ckpts = ckpts[bi*K:(bi+1)*K]
        batch_steps = [int(c.split("-")[-1]) for c in batch_ckpts]
        kthis = len(batch_ckpts)
        t_batch = time.time()
        log(f"--- batch {bi+1}/{n_batches}: steps {batch_steps} ---")

        # Load each ckpt's adapter weights into its slot.
        # Strip torch.compile's `_orig_mod.` prefix from saved keys.
        for j, ckpt in enumerate(batch_ckpts):
            sd = _load_state_dict(ckpt)
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys when loading {ckpt}: {unexpected[:5]}...")
            client.update_weights_from_model(slot_ids[j], model)
        log(f"  loaded {kthis} ckpts ({time.time()-t_batch:.1f}s)")

        for mode_name, retain, forget in modes:
            t_mode = time.time()
            for j in range(kthis):
                client.set_scales(slot_ids[j], retain, forget)
            all_eids, all_pids = [], []
            for j in range(kthis):
                all_eids.extend([slot_ids[j]] * n_eval)
                all_pids.extend(prompt_ids)
            comp_texts, _ci, _pi = client.generate_multi(
                all_eids, all_pids, n=1,
                temperature=args.temperature,
                max_tokens=args.max_completion_length,
                top_k=args.top_k, top_p=args.top_p,
            )
            t_gen = time.time() - t_mode
            t_score_start = time.time()
            with open(out_path, "a") as fout:
                for j, ckpt_step in enumerate(batch_steps):
                    completions = comp_texts[j*n_eval:(j+1)*n_eval]
                    correct, trait, compile_ = leetcode_all_components(
                        completions, **reward_kwargs)
                    for i in range(n_eval):
                        rec = {
                            "step": ckpt_step,
                            "mode": mode_name,
                            "id": eval_data[i]["id"],
                            "hackable": bool(eval_data[i].get("hackable")),
                            "detectable_orig": bool(eval_data[i].get("detectable")),
                            "detectable_corrected": bool(detectable_corrected[i]),
                            "correct": float(correct[i]),
                            "trait": float(trait[i]),
                            "compile": float(compile_[i]),
                            "hack_freq": float(trait[i] >= 0.5),
                        }
                        fout.write(json.dumps(rec) + "\n")
            t_score = time.time() - t_score_start
            log(f"  {mode_name}: gen={t_gen:.1f}s score={t_score:.1f}s")
        log(f"--- batch {bi+1} done ({time.time()-t_batch:.1f}s) ---")

    log(f"DONE: {len(ckpts)} ckpts in {time.time()-t_run:.1f}s")

    try:
        client.shutdown()
    except Exception:
        pass
    server_proc.join(timeout=30)
    if server_proc.is_alive():
        server_proc.terminate()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
