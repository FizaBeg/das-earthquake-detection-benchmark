"""Robustness of trained (100 % label) models to synthetic perturbations added to the real test patches.
Perturbations are applied at full resolution (288 x 695) before the same pooling used in training.
  - white: zero-mean Gaussian noise; SNR(dB) = 10 log10(P_patch / P_noise), P_patch = mean square of the patch
  - drop : a random fraction of fibre channels set to zero (dead / poorly coupled channels)
The same random perturbation (fixed seed per condition) is used for every model, so comparisons are paired."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from train import pool, predict, metrics, T_CROP, DATA
from models import build

torch.set_num_threads(1)
Xf = np.load(f'{DATA}/test_X.npy').astype(np.float32); y = np.load(f'{DATA}/test_y.npy')
conds = [('clean', None)] + [('white', s) for s in (20, 10, 5, 0, -5)] + [('drop', f) for f in (0.1, 0.3, 0.5)]
inputs = {}
for k, (kind, v) in enumerate(conds):
    rng = np.random.default_rng(777 + k); X = Xf.copy()
    if kind == 'white':
        p = (X ** 2).mean(axis=(1, 2), keepdims=True)
        X += rng.standard_normal(X.shape).astype(np.float32) * np.sqrt(p / 10 ** (v / 10))
    elif kind == 'drop':
        for i in range(len(X)):
            X[i, rng.choice(X.shape[1], int(round(v * X.shape[1])), replace=False)] = 0
    inputs[f'{kind}_{v}'] = pool(X)
out = []
for f in sorted(glob.glob('results/main/*_f1.0_s*.pt')):
    name, _, s = os.path.basename(f)[:-3].split('_')
    m = build(name, 48, T_CROP); m.load_state_dict(torch.load(f)); m.eval()
    for c, X in inputs.items():
        p = predict(m, X)
        out.append(dict(model=name, seed=int(s[1:]), cond=c, **metrics(y, p), prob=p.tolist()))
    print(f, flush=True)
json.dump(out, open('results/noise.json', 'w'))
