"""Cost of each architecture on the experiment hardware (CPU): parameters, forward FLOPs,
inference latency and peak resident memory of a training step. Run one model per process:
python bench.py CNN"""
import json, resource, sys, time
import numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
from models import build, n_params

name = sys.argv[1]; torch.set_num_threads(1); torch.manual_seed(0)
C, T = 48, 320
m = build(name, C, T)
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
x1 = torch.randn(1, 1, C, T)
with FlopCounterMode(display=False) as fc:
    m.eval(); m(x1)
flops = fc.get_total_flops()
with torch.no_grad():
    for _ in range(5): m(x1)
    ts = []
    for _ in range(50):
        t = time.perf_counter(); m(x1); ts.append(time.perf_counter() - t)
opt = torch.optim.AdamW(m.parameters()); xb = torch.randn(32, 1, C, T); yb = torch.ones(32)
m.train(); ts2 = []
for _ in range(6):
    t = time.perf_counter(); opt.zero_grad()
    torch.nn.functional.binary_cross_entropy_with_logits(m(xb), yb).backward(); opt.step()
    ts2.append(time.perf_counter() - t)
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(json.dumps(dict(model=name, params=n_params(m), gflops_fwd=flops / 1e9,
                      latency_ms_b1=1000 * float(np.median(ts)), step_s_b32=float(np.median(ts2[1:])),
                      train_mem_mb=rss1 - rss0)))
