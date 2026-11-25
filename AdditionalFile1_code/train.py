"""Train/evaluate one model on the Stanford DAS earthquake-detection data.
Usage: python train.py --model CNN --seed 0 --frac 1.0 --out results/"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from models import build, n_params

DATA = '/home/claude/data/npy'
POOL_C, POOL_T, T_CROP = 6, 2, 320      # 288x695 -> 48x347 -> crop 48x320


def pool(x):
    """Average-pool full-resolution patches (N,288,695) -> (N,48,347) float32."""
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)
    return F.avg_pool2d(x, (POOL_C, POOL_T)).squeeze(1).numpy()


def load(split):
    """Pooled arrays are cached once (make_cache) so parallel runs share them."""
    return np.load(f'{DATA}/{split}_Xp.npy'), np.load(f'{DATA}/{split}_y.npy')


def make_cache():
    for s in ('train', 'eval', 'test'):
        np.save(f'{DATA}/{s}_Xp.npy', pool(np.load(f'{DATA}/{s}_X.npy')))


def center(x):
    s = (x.shape[-1] - T_CROP) // 2
    return x[..., s:s + T_CROP]


def predict(model, X, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(center(X[i:i + bs])).unsqueeze(1)
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)


def metrics(y, p):
    yh = (p >= 0.5).astype(int)
    return dict(auc=float(roc_auc_score(y, p)), acc=float(accuracy_score(y, yh)), f1=float(f1_score(y, yh)))


def subsample(y, frac, seed):
    if frac >= 1: return np.arange(len(y))
    rng = np.random.default_rng(1000 + seed); idx = []
    for c in (0, 1):
        ci = np.where(y == c)[0]
        k = max(2, int(round(frac * len(ci))))
        idx.append(rng.choice(ci, k, replace=False))
    return np.sort(np.concatenate(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frac', type=float, default=1.0); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--min_steps', type=int, default=300); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', default='results')
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tag = f'{a.model}_f{a.frac}_s{a.seed}'

    Xtr, ytr = load('train'); Xva, yva = load('eval'); Xte, yte = load('test')
    idx = subsample(ytr, a.frac, a.seed)
    Xtr, ytr = Xtr[idx].copy(), ytr[idx]
    model = build(a.model, Xtr.shape[1], T_CROP)
    spe = math.ceil(len(ytr) / a.bs)
    steps = max(a.epochs * spe, a.min_steps)
    eval_every = max(spe, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    best = (-1, None, 0); hist = []
    t0 = time.time(); step = 0
    while step < steps:
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), a.bs):
            if step >= steps: break
            b = perm[i:i + a.bs]
            st = rng.integers(0, Xtr.shape[-1] - T_CROP + 1, len(b))
            xb = np.stack([Xtr[j, :, s:s + T_CROP] for j, s in zip(b, st)])
            xb *= rng.choice([-1.0, 1.0], (len(b), 1, 1)).astype(np.float32)   # polarity flip
            xb = torch.from_numpy(xb).unsqueeze(1); yb = torch.from_numpy(ytr[b]).float()
            model.train(); opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % eval_every == 0 or step == steps:
                pv = predict(model, Xva); m = metrics(yva, pv)
                hist.append(dict(step=step, loss=float(loss), val_auc=m['auc'], val_acc=m['acc']))
                score = m['auc'] + 1e-3 * m['acc']
                if score > best[0]:
                    best = (score, {k: v.clone() for k, v in model.state_dict().items()}, step)
    train_time = time.time() - t0
    model.load_state_dict(best[1])
    pte = predict(model, Xte)
    t1 = time.time(); predict(model, Xte); infer = (time.time() - t1) / len(yte)
    res = dict(model=a.model, seed=a.seed, frac=a.frac, n_train=int(len(ytr)), steps=steps,
               best_step=best[2], train_time_s=train_time, infer_ms_per_sample=1000 * infer,
               params=n_params(model), test=metrics(yte, pte),
               val_best_auc=float(max(h['val_auc'] for h in hist)),
               test_prob=pte.tolist(), hist=hist)
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    torch.save(best[1], f'{a.out}/{tag}.pt')
    print(tag, 'n', len(ytr), 'steps', steps, 'time %.0fs' % train_time, res['test'], flush=True)


if __name__ == '__main__':
    main()
