# Nsight Systems Profiling Results

## Profile 1: Concurrent Sweep (nsys-report-4a3f)

**Command**: `CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=10000 --sample=cpu --cpuctxsw=process-tree --trace-fork-before-exec=true --cuda-event-trace=false --wait=primary -o ... .venv/bin/python sweep.py --config sweeps/coherence_sweep.py --name nsys-metrics-1855 --no_baseline --no_cache --vllm_async`

**Profile duration**: 306.3s
**GPU**: NVIDIA H200 (single, via CUDA_VISIBLE_DEVICES=0)
**Workload**: Multiple concurrent training runs under MPS on one GPU, using vLLM async server for generation

### GPU Utilization Time Series (10s bins)

| Time | SMs Active | Tensor Active | Compute Warps | DRAM Rd | DRAM Wr |
|------|-----------|---------------|--------------|---------|---------|
| 0s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 10s | 0.2% | 0.0% | 0.0% | 0.1% | 0.1% |
| 20s | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% |
| 30s | 0.0% | 0.0% | 0.0% | 0.1% | 0.2% |
| 40s | 22.7% | 1.8% | 8.2% | 2.9% | 2.0% |
| 50s | 73.0% | 6.5% | 31.0% | 17.0% | 9.7% |
| 60s | 71.6% | 6.5% | 30.6% | 16.8% | 9.6% |
| 70s | 71.1% | 6.3% | 29.7% | 16.6% | 9.3% |
| 80s | 70.6% | 6.3% | 29.7% | 16.5% | 9.3% |
| 90s | 68.9% | 6.0% | 29.6% | 14.8% | 8.8% |
| 100s | 64.3% | 5.3% | 25.5% | 13.9% | 7.8% |
| 110s | 69.6% | 6.4% | 29.9% | 16.1% | 9.4% |
| 120s | 71.9% | 6.3% | 30.0% | 16.7% | 9.3% |
| 130s | 64.9% | 6.0% | 27.7% | 15.1% | 8.7% |
| 140s | 76.9% | 6.9% | 32.9% | 17.7% | 10.2% |
| 150s | 58.9% | 4.8% | 24.2% | 11.6% | 6.9% |
| 160s | 72.9% | 6.4% | 30.3% | 16.8% | 9.6% |
| 170s | 71.4% | 6.4% | 30.3% | 16.5% | 9.5% |
| 180s | 70.9% | 5.9% | 28.5% | 16.1% | 8.9% |
| 190s | 70.5% | 6.9% | 31.8% | 16.9% | 10.0% |
| 200s | 70.4% | 6.2% | 30.4% | 15.4% | 9.1% |
| 210s | 63.9% | 5.5% | 25.9% | 13.9% | 8.0% |
| 220s | 67.5% | 5.6% | 26.7% | 15.5% | 8.4% |
| 230s | 71.2% | 6.5% | 30.4% | 16.9% | 9.5% |
| 240s | 75.7% | 6.7% | 32.2% | 17.7% | 10.0% |
| 250s | 69.8% | 6.5% | 30.2% | 16.3% | 9.5% |
| 260s | 60.1% | 5.0% | 24.6% | 12.0% | 7.1% |
| 270s | 72.8% | 6.7% | 31.0% | 16.7% | 9.7% |
| 280s | 71.1% | 6.3% | 29.7% | 16.4% | 9.3% |
| 290s | 78.9% | 7.0% | 33.7% | 18.0% | 10.4% |
| 300s | 69.2% | 6.9% | 30.9% | 16.7% | 10.0% |

### Steady-State Averages (50s–306s)

| Metric | Value |
|--------|-------|
| SMs Active | 69.9% |
| SM Issue | 36.3% |
| Tensor Active | 6.2% |
| Compute Warps in Flight | 29.5% |
| Unallocated Warps in Active SMs | 40.3% |
| Compute Warps in Flight (avg warps/cycle) | 37.8 |
| Unallocated Warps in Active SMs (avg warps/cycle) | 51.7 |
| DRAM Read BW | 15.9% |
| DRAM Write BW | 9.1% |

### Per-Step Timing (from train.log, single run)

| Step | Rollout | Sync | Gen | Update | Total |
|------|---------|------|-----|--------|-------|
| 1 | 2.62s | 330ms | 1538ms | 1.38s | 4.01s |
| 2 | 3.93s | 775ms | 2023ms | 1.33s | 5.25s |
| 3 | 3.35s | 541ms | 1914ms | 1.40s | 4.75s |
| 4 | 3.64s | 695ms | 1948ms | 1.23s | 4.87s |
| 5 | 3.88s | 602ms | 2150ms | 1.37s | 5.25s |
| 6 | 3.09s | 476ms | 1652ms | 1.63s | 4.73s |
| 7 | 3.78s | 688ms | 2213ms | 1.23s | 5.01s |
| 8 | 3.56s | 679ms | 1917ms | 0.81s | 4.37s |
| 9 | 3.47s | 361ms | 1805ms | 1.15s | 4.62s |
| 10 | 3.21s | 528ms | 1787ms | 1.07s | 4.29s |
| 11 | 2.76s | 262ms | 1456ms | 1.39s | 4.14s |

### OS Runtime Summary (top 10 by total time)

| Function | Calls | Total Time | Avg |
|----------|-------|-----------|-----|
| futex | 556,227 | 236,976s | 426ms |
| epoll_wait | 425,255 | 7,678s | 18ms |
| sem_timedwait | 535 | 5,061s | 9,461ms |
| poll | 230,366 | 4,520s | 20ms |
| pthread_cond_wait | 4,919 | 2,659s | 541ms |
| pthread_cond_timedwait | 294,053 | 2,366s | 8ms |
| sem_wait | 9,743 | 645s | 66ms |
| clock_nanosleep | 603 | 300s | 497ms |
| epoll_pwait | 311 | 288s | 927ms |
| read | 93,665 | 38s | 0.4ms |

### CUDA API Summary (top 10 by total time)

| API | Calls | Total Time | Avg |
|-----|-------|-----------|-----|
| cudaLaunchKernel | 12,362,188 | 446.7s | 36.1us |
| cudaStreamSynchronize | 398,468 | 177.5s | 445.6us |
| cuLaunchKernelEx | 2,581,057 | 48.4s | 18.8us |
| cudaMemcpyAsync | 475,489 | 21.4s | 45.0us |
| cudaMemsetAsync | 295,532 | 12.2s | 41.3us |
| cudaLaunchKernelExC | 62,584 | 2.6s | 41.7us |
| cuKernelGetAttribute | 5,162,110 | 0.6s | 0.1us |
| cudaMalloc | 822 | 0.4s | 506.4us |
| cuLibraryLoadData | 1,370 | 0.4s | 284.7us |
| cudaStreamIsCapturing | 273,585 | 0.3s | 1.0us |

### Top 15 GPU Kernels by Time

Total GPU kernel time: 264.6s, 15,001,084 launches.

| % | Count | Time | Avg | Kernel |
|---|-------|------|-----|--------|
| 14.1% | 20,666 | 37.35s | 1807us | sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma... |
| 13.0% | 41,790 | 34.53s | 826us | sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x64x8_stage3_warpsize2x2x1_ffma... |
| 4.9% | 929,953 | 13.09s | 14us | elementwise_kernel<128, 4> (variant 1) |
| 4.1% | 386,060 | 10.79s | 28us | elementwise_kernel<128, 2> |
| 3.7% | 511,858 | 9.77s | 19us | elementwise_kernel<128, 4> (variant 2) |
| 3.2% | 138,077 | 8.49s | 62us | fmha_cutlassF_bf16_aligned_64x64_rf_sm80 |
| 2.8% | 947,847 | 7.37s | 8us | CUDAFunctor_add (bf16) |
| 2.7% | 8,880 | 7.21s | 812us | fmha_cutlassF_f32_aligned_64x64_rf_sm80 |
| 2.2% | 328,057 | 5.81s | 18us | direct_copy_kernel_cuda |
| 1.9% | 528,487 | 4.93s | 9us | BinaryFunctor (bf16) |
| 1.8% | 16,147 | 4.68s | 290us | fmha_cutlassB_bf16_aligned_64x64_k64_sm80 |
| 1.7% | 283,837 | 4.50s | 16us | CatArrayBatchedCopy |
| 1.7% | 108,252 | 4.49s | 42us | BinaryFunctor (float) |
| 1.5% | 309,160 | 4.07s | 13us | reduce_kernel (MeanOps, variant 1) |
| 1.4% | 190,576 | 3.69s | 19us | reduce_kernel (MeanOps, variant 2) |

---

## Profile 2: Single Large Batch vLLM (nsys-report-c0a6)

**Command**: `CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 --sample=none --cpuctxsw=none --cuda-event-trace=false -o ... .venv/bin/python -u benchmarks/bench_large_batch.py`

**Profile duration**: 121.4s
**GPU**: NVIDIA H200 (single, via CUDA_VISIBLE_DEVICES=0)
**Workload**: vLLM async engine, 16384 sequences, 256 max tokens, SmolLM2-135M-Instruct with MLP adapters (retain=32, forget=32), gpu_memory_utilization=0.8

### GPU Utilization Time Series (10s bins)

| Time | SMs Active | Tensor Active | Compute Warps | DRAM Rd | DRAM Wr |
|------|-----------|---------------|--------------|---------|---------|
| 0s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 10s | 1.9% | 0.3% | 0.5% | 0.3% | 0.6% |
| 20s | 6.7% | 1.0% | 1.7% | 1.3% | 0.9% |
| 30s | 13.7% | 1.9% | 3.8% | 2.8% | 1.1% |
| 40s | 13.4% | 1.8% | 3.7% | 2.8% | 1.0% |
| 50s | 12.7% | 1.7% | 3.4% | 2.6% | 1.0% |
| 60s | 11.6% | 1.5% | 3.1% | 2.4% | 0.9% |
| 70s | 11.2% | 1.5% | 3.0% | 2.3% | 0.9% |
| 80s | 10.8% | 1.4% | 2.9% | 2.2% | 0.8% |
| 90s | 10.9% | 1.4% | 2.9% | 2.3% | 0.9% |
| 100s | 11.6% | 1.5% | 3.1% | 2.4% | 0.9% |
| 110s | 11.4% | 1.5% | 3.1% | 2.4% | 0.9% |
| 120s | 13.0% | 1.7% | 3.5% | 2.7% | 1.0% |

### Steady-State Averages (30s–121s)

| Metric | Value |
|--------|-------|
| SMs Active | 12.0% |
| SM Issue | 3.2% |
| Tensor Active | 1.6% |
| Compute Warps in Flight | 3.2% |
| Unallocated Warps in Active SMs | 8.7% |
| DRAM Read BW | 2.5% |
| DRAM Write BW | 0.9% |

### Effective Batch Size

FlashAttention forward kernel calls: 56,811 (large variant) + 13,590 (small variant) = 70,401 total.
With 4 layers: ~17,600 decode steps.
16,384 seqs × 256 tokens = 4,194,304 tokens total.
Estimated tokens per decode step: ~295.

### Top 15 GPU Kernels by Time

Total GPU kernel time: 13.6s.

| % | Count | Time | Avg | Kernel |
|---|-------|------|-----|--------|
| 50.1% | 56,811 | 6.80s | 120us | FlashAttnFwdSm90 (variant 1) |
| 5.0% | 13,590 | 0.68s | 50us | FlashAttnFwdSm90 (variant 2) |
| 4.3% | 102,792 | 0.59s | 6us | nvjet_hsh_64x96_64x10_2x4_v_bz_TNT |
| 3.5% | 53,511 | 0.47s | 9us | nvjet_hsh_256x96_64x4_2x1_v_bz_coopA_TNT |
| 3.3% | 140,922 | 0.44s | 3us | vllm::fused_add_rms_norm_kernel (half) |
| 3.2% | 2,348 | 0.43s | 182us | cunn_SoftMaxForward (float) |
| 2.5% | 2,348 | 0.34s | 143us | elementwise_kernel<128, 2> |
| 2.3% | 53,511 | 0.31s | 6us | nvjet_hsh_160x64_64x7_2x4_v_bz_TNN |
| 2.3% | 2,348 | 0.31s | 130us | direct_copy_kernel_cuda |
| 2.2% | 2,348 | 0.30s | 129us | BinaryFunctor (float) |
| 2.2% | 70,431 | 0.29s | 4us | vllm::reshape_and_cache_flash_kernel (half) |
| 2.0% | 70,461 | 0.28s | 4us | vllm::rotary_embedding_kernel (half) |
| 1.8% | 70,461 | 0.25s | 4us | vllm::act_and_mul_kernel (half, silu) |
| 1.4% | 2,348 | 0.19s | 81us | distribution_elementwise_grid_stride_kernel (float) |
| 1.1% | 2,350 | 0.15s | 63us | reduce_kernel (MeanOps) |

---

## Profile 3: Qwen3-8B vLLM Single Batch (nsys-report-1701)

**Command**: `.venv/bin/python -u benchmarks/bench_large_batch_qwen.py --model Qwen/Qwen3-8B --n_seqs 512 --max_tokens 16`

**Profile duration**: 120.6s
**GPU**: NVIDIA H200 (single, via CUDA_VISIBLE_DEVICES=0)
**Workload**: vLLM async engine, 512 sequences, 16 max tokens, Qwen3-8B, no adapters. Mostly startup; real generation only in last ~30s.

### GPU Utilization Time Series (10s bins)

| Time | SMs Active | Tensor Active | Compute Warps | DRAM Rd | DRAM Wr |
|------|-----------|---------------|--------------|---------|---------|
| 0s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 10s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 20s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 30s | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% |
| 40s | 0.0% | 0.0% | 0.0% | 0.1% | 0.2% |
| 50s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 60s | 1.7% | 0.0% | 1.3% | 0.3% | 1.4% |
| 70s | 2.2% | 0.3% | 1.6% | 0.5% | 1.3% |
| 80s | 0.4% | 0.0% | 0.3% | 0.2% | 0.3% |
| 90s | 7.0% | 1.0% | 1.1% | 5.8% | 0.5% |
| 100s | 54.9% | 2.0% | 7.7% | 50.9% | 0.9% |
| 110s | 50.5% | 1.9% | 7.1% | 46.9% | 0.8% |
| 120s | 51.9% | 1.9% | 7.3% | 48.2% | 0.8% |

### Steady-State Averages (after 30s)

| Metric | Value |
|--------|-------|
| SMs Active | 13.2% |
| SM Issue | 1.3% |
| Tensor Active | 0.6% |
| Compute Warps in Flight | 2.1% |
| Unallocated Warps in Active SMs | 11.1% |
| DRAM Read BW | 11.9% |
| DRAM Write BW | 0.6% |

### Top 10 GPU Kernels by Time

Total GPU kernel time: 2.1s.

| % | Count | Time | Avg | Kernel |
|---|-------|------|-----|--------|
| 45.8% | 3,300 | 0.98s | 296us | nvjet_tst_384x8_64x4_2x1_v_bz_TNT |
| 12.5% | 18,136 | 0.27s | 15us | FillFunctor (int) |
| 6.0% | 3,302 | 0.13s | 39us | cunn_SoftMaxForward (float) |
| 4.5% | 8,136 | 0.10s | 12us | FlashAttnFwdSm90 |
| 2.1% | 2,866 | 0.05s | 16us | triton_poi_fused_mul_silu_slice_1 |
| 1.6% | 72 | 0.03s | 480us | FillFunctor (short) |
| 1.5% | 144 | 0.03s | 223us | nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT |
| 1.5% | 3,186 | 0.03s | 10us | triton_ |
| 1.5% | 10,692 | 0.03s | 3us | vllm::reshape_and_cache_flash_kernel (bf16) |
| 1.4% | 3,304 | 0.03s | 9us | reduce_kernel (MeanOps) |

---

## Profile 4: Qwen3-8B vLLM Single Batch, warmed up (nsys-report-a14d)

**Command**: `.venv/bin/python -u benchmarks/bench_large_batch_qwen.py --model Qwen/Qwen3-8B --n_seqs 512 --max_tokens 16`

**Profile duration**: 82.9s
**GPU**: NVIDIA H200 (single, via CUDA_VISIBLE_DEVICES=0)
**Workload**: Same as Profile 3 but with faster warmup.

### GPU Utilization Time Series (10s bins)

| Time | SMs Active | Tensor Active | Compute Warps | DRAM Rd | DRAM Wr |
|------|-----------|---------------|--------------|---------|---------|
| 0s | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 10s | 0.1% | 0.1% | 0.0% | 0.1% | 0.3% |
| 20s | 3.0% | 1.0% | 0.6% | 1.8% | 0.5% |
| 30s | 51.2% | 1.9% | 7.2% | 47.9% | 0.8% |
| 40s | 59.1% | 2.2% | 8.3% | 55.4% | 0.9% |
| 50s | 59.3% | 2.2% | 8.3% | 55.5% | 0.9% |
| 60s | 58.7% | 2.2% | 8.2% | 55.0% | 0.9% |
| 70s | 58.3% | 2.2% | 8.1% | 54.6% | 0.9% |
| 80s | 0.0% | 0.0% | 0.0% | 0.2% | 1.0% |

### Steady-State Averages (after 30s)

| Metric | Value |
|--------|-------|
| SMs Active | 54.1% |
| SM Issue | 4.9% |
| Tensor Active | 2.0% |
| Compute Warps in Flight | 7.6% |
| Unallocated Warps in Active SMs | 46.5% |
| DRAM Read BW | 50.7% |
| DRAM Write BW | 0.9% |

### Top 10 GPU Kernels by Time

Total GPU kernel time: 0.5s.

| % | Count | Time | Avg | Kernel |
|---|-------|------|-----|--------|
| 18.6% | 320 | 0.09s | 295us | nvjet_tst_384x8_64x4_2x1_v_bz_TNT |
| 6.9% | 72 | 0.03s | 484us | FillFunctor (short) |
| 6.3% | 144 | 0.03s | 224us | nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT |
| 4.5% | 1,406 | 0.02s | 16us | FlashAttnFwdSm90 |
| 2.6% | 3,962 | 0.01s | 3us | vllm::reshape_and_cache_flash_kernel (bf16) |
| 2.5% | 322 | 0.01s | 39us | cunn_SoftMaxForward (float) |
| 2.3% | 180 | 0.01s | 66us | nvjet_tst_192x128_64x5_1x2_h_bz_coopB_TNT |
| 2.1% | 216 | 0.01s | 49us | nvjet_tst_192x8_64x8_2x1_v_bz_TNT |
| 2.0% | 182 | 0.01s | 56us | nvjet_tst_192x128_64x5_2x1_v_bz_coopB_TNT |
| 1.7% | 2,556 | 0.01s | 3us | cublasLt::splitKreduce_kernel |
