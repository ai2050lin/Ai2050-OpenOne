# CPU bf16 matmul benchmark for 14b mixed placement feasibility (Qwen3-14B layer shapes)
import torch, time, os
torch.set_num_threads(os.cpu_count())
print('threads=%d' % os.cpu_count())
# representative: 16 rows x 100 tokens x 5120 hidden @ 17408 x 5120 (mlp gate/up), bf16
a = torch.randn(1600, 5120, dtype=torch.bfloat16)
w = torch.randn(17408, 5120, dtype=torch.bfloat16)
for _ in range(2):
    b = a @ w.T  # warmup
t0 = time.time()
REP = 8
for _ in range(REP):
    b = a @ w.T
dt = (time.time() - t0) / REP
flops = 2 * 1600 * 5120 * 17408
print('bf16 mlp-gate matmul: %.3f s/call, %.2f TFLOPS' % (dt, flops / dt / 1e12))
# per-forward estimate: 20 CPU layers x (attn 0.19x + mlp 0.81x of 660 MFLOP/row/token) -> use measured TFLOPS
per_layer_flop = 2 * 1600 * 5120 * 5120 + 2 * 1600 * 5120 * 17408 * 3  # q,k,v,o + gate,up,down at batch 1600
per_fwd = 20 * (per_layer_flop * 0.61 / flops * flops)  # scale: single measured op dominates; approx
est = 20 * per_layer_flop / (flops / dt)
print('est 20 CPU layers per forward: %.1f s' % est)
print('est per anchor (108 fwd): %.1f min; 4 anchors: %.1f h' % (est * 108 / 60, est * 108 * 4 / 3600))
