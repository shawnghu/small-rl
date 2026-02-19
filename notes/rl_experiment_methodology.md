# RL Experiment Methodology Notes

These notes capture the process and reasoning used to set up efficient RL training runs on a small LM
(SimpleStories 1.25M, 1.25M params, GRPO). Goal: fast iteration with full GPU utilization.

---

## Core Principle: Fast Feedback First

1. **Characterize throughput** (how many concurrent runs can the GPU support?)
2. **Find working hyperparameters** at the throughput ceiling (using early stopping liberally)
3. **Only then** run systematic performance measurements

Doing these out of order wastes effort: optimizing HPs at 2 concurrent runs is 8x slower than
optimizing them at 16. And running a large HP sweep before knowing throughput limits means you can't
size the sweep correctly.

---

## Step 1: Throughput Characterization

**Motivation**: A 1.25M param model almost certainly underutilizes a modern GPU. The question is
whether multiple processes can share the GPU efficiently.

**Key pre-existing fact (not a discovery)**: NVIDIA MPS (Multi-Process Service) should always be
enabled for concurrent GPU workloads. It enables true kernel-level multiplexing. Enable it once
at the start; there's no downside.

```bash
nvidia-cuda-mps-control -d
```

**Process**: Scale up concurrency empirically until step time clearly increases, then back off slightly.

- Double concurrency (5, 10, 20...) until step time noticeably increases
- It's possible that increasing step time is worth it in terms of throughput, so we may like to characterize best-latency concurrency and best-throughput concurrency
- If multiple workload types exist (full FT vs. LoRA rank), try to characterize all of them, making reasonable assumptions (e.g, memory throughput)
- IMPORTANT: for small models, time to first feedback is often 10 seconds or less. Don't sleep for 60-300 seconds to wait for a result.

**Key finding for this setup**: The sweet spot varies by workload:
- Low-rank LoRA (rank 1-8), bs=32: 16-20 concurrent
- High-rank LoRA (rank 32+) or large batch (bs=128): 6 concurrent (efficiency), 12 (max throughput)
- Full fine-tuning: 12 concurrent

---

## Step 2: Hyperparameter Search
In this step we try to roughly characterize what best practices for training a model on a task are, and establish some baselines for reference in later experiments.

**Design principle**: Use known failures to bracket the search space before running a grid. Avoids
exploring obviously broken regions.

**What we knew before starting**:
- beta=0 → degenerate output (model exploits reward with no KL constraint)
- beta=0.1 → reward stuck near baseline (KL too strong, can't learn)
- This brackets the interesting range: ~0.01-0.05

**Order of investigation** (most impactful first):
1. KL penalty (beta) — the most critical lever; sharp cliff on both sides
2. num_generations — stability of GRPO advantages; 16 was a major win over 8
3. batch_size — gradient stability; 32 was sufficient
4. learning rate — matters a lot for LoRA (needs higher LR than full FT)
5. reward function design — semantic rewards (happy_binary) much more tractable than structural
   rewards (sentence_length_10), which collapse to templates regardless of HPs

**Batch size + LR scaling** (subproblem of throughput): 
- First search for configurations of the model on the problem that you think are doable.

- Think explicitly in terms of compute-boundness or memory-boundedness to determine what is or isn't likely scalable. For example, with small models often memory is free and kernels don't saturate even individual cores, so larger batch sizes are very low-cost.

- It's possible that increasing batch size by a factor of 4 allows you to increase LR by a factor of 4, which potentially gives 4x throughput. Even if time per iteration is longer, if this is by less than a factor of 4 we get effective throughput gains.

**Process**: Run as many concurrent experiments as is sensible per round, kill non-starters early (early stopping).
This functionally achieves a lot more throughput in this phase.

** A small sidenote about transferring HPs from full-param to LoRA**: Once HPs are found for full fine-tuning, they
transfer partially but not fully to LoRA:
- beta, batch_size, num_generations transfer directly
- Learning rate does NOT transfer — LoRA needs a higher LR (roughly 10-30x) because fewer
  parameters are being updated

---

## Step 4: Further Characterization

Only after Steps 1 and 2 are done:
- Run validated config across multiple seeds to get variance estimates
- Try gradient routing, using sweep.py
- For gradient routing: eval three adapter modes (both, retain_only, forget_only)
- Write results to a dedicated results file

---

## Summary: Reasoning Behind the Sequence

The sequence (throughput → HPs → measurements) is driven by the question: *what is the most
expensive mistake I can make?*

- Running HPs before throughput: you find good HPs but at 1/8 the speed you could have
- Running measurements before HPs: you measure a bad configuration thoroughly
- Using early stopping during HP search: each failed run killed early frees capacity for another
  attempt, compounding the throughput advantage

The goal is to acquire preliminary knowledge so that informed experimentation can begin in earnest.
Many of the guidelines 
Go for a long time, assuming use of the GPU is free and a resource to be utilized, but proceed in such a way that you can take note of intermediate results. This latter property is useful both because you can adapt your search thoughtfully in response to new results, and also so that we get useful output if the search is stopped at any point.
