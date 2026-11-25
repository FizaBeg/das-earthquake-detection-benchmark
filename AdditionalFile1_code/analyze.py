"""Aggregate results, run statistical tests, make figures and LaTeX-ready numbers (results/summary.json)."""
import glob, itertools, json
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

MODELS = ['CNN', 'CNN-LSTM', 'ConvLSTM', 'Transformer']
COL = {'CNN': '#1f77b4', 'CNN-LSTM': '#ff7f0e', 'ConvLSTM': '#2ca02c', 'Transformer': '#d62728'}
MK = {'CNN': 'o', 'CNN-LSTM': 's', 'ConvLSTM': '^', 'Transformer': 'D'}
y = np.load('/home/claude/data/npy/test_y.npy')
FIG = '/home/claude/paper/figures'
S = {}


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def boot_auc_diff(pa, pb, B=5000, seed=0):
    rng = np.random.default_rng(seed); d = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(B):   # stratified resampling keeps both classes present
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5]).tolist(), float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def boot_auc_ci(p, B=5000, seed=0):
    rng = np.random.default_rng(seed); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; a = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5]).tolist()


def summarize(runs, key):
    v = np.array([r['test'][key] for r in runs]); return float(v.mean()), float(v.std(ddof=1)), v.tolist()


# ---------------- main (100 % labels) ----------------
main = {m: sorted(load(f'results/main/{m}_f1.0_s*.json'), key=lambda r: r['seed']) for m in MODELS}
S['seeds'] = {m: [r['seed'] for r in main[m]] for m in MODELS}
S['main'] = {}
ens = {}
for m in MODELS:
    R = main[m]
    ens[m] = np.mean([r['test_prob'] for r in R], axis=0)
    yh = (ens[m] >= 0.5).astype(int)
    S['main'][m] = dict(
        auc=summarize(R, 'auc'), acc=summarize(R, 'acc'), f1=summarize(R, 'f1'),
        train_time_s=[float(np.mean([r['train_time_s'] for r in R])), float(np.std([r['train_time_s'] for r in R], ddof=1))],
        best_step=[r['best_step'] for r in R], steps=R[0]['steps'], params=R[0]['params'], n_train=R[0]['n_train'],
        ens_auc=float(roc_auc_score(y, ens[m])), ens_auc_ci=boot_auc_ci(ens[m]),
        ens_acc=float((yh == y).mean()),
        ens_tp=int(((yh == 1) & (y == 1)).sum()), ens_fp=int(((yh == 1) & (y == 0)).sum()),
        ens_fn=int(((yh == 0) & (y == 1)).sum()), ens_tn=int(((yh == 0) & (y == 0)).sum()))
A = np.array([S['main'][m]['auc'][2] for m in MODELS]).T   # seeds x models
ACC = np.array([S['main'][m]['acc'][2] for m in MODELS]).T
S['friedman_auc'] = list(map(float, stats.friedmanchisquare(*A.T)))
S['friedman_acc'] = list(map(float, stats.friedmanchisquare(*ACC.T)))
pairs = list(itertools.combinations(MODELS, 2)); rows = []
for a, b in pairs:
    d, ci, p = boot_auc_diff(ens[a], ens[b])
    ya, yb = ens[a] >= 0.5, ens[b] >= 0.5
    n01 = int(((ya == y) & (yb != y)).sum()); n10 = int(((ya != y) & (yb == y)).sum())
    pm = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 > 0 else 1.0
    rows.append(dict(a=a, b=b, dauc=d, ci=ci, p_boot=p, a_only_correct=n01, b_only_correct=n10, p_mcnemar=pm))
for k, adj in [('p_boot', holm([r['p_boot'] for r in rows])), ('p_mcnemar', holm([r['p_mcnemar'] for r in rows]))]:
    for r, v in zip(rows, adj): r[k + '_holm'] = float(v)
S['pairs'] = rows

# ---------------- noise ----------------
nz = json.load(open('results/noise.json'))
conds = ['clean_None'] + [f'white_{s}' for s in (20, 10, 5, 0, -5)] + [f'drop_{f}' for f in (0.1, 0.3, 0.5)]
S['noise'] = {}
for c in conds:
    S['noise'][c] = {}
    for m in MODELS:
        v = [r for r in nz if r['model'] == m and r['cond'] == c]
        S['noise'][c][m] = dict(auc=[float(np.mean([r['auc'] for r in v])), float(np.std([r['auc'] for r in v], ddof=1))],
                                acc=[float(np.mean([r['acc'] for r in v])), float(np.std([r['acc'] for r in v], ddof=1))],
                                n=len(v))
    M = np.array([[r['auc'] for r in sorted([r for r in nz if r['model'] == m and r['cond'] == c], key=lambda r: r['seed'])] for m in MODELS])
    S['noise'][c]['friedman_p'] = float(stats.friedmanchisquare(*M)[1])

S['noise_pairs'] = {}
for c in ('white_0', 'white_-5', 'drop_0.5'):
    E = {m: np.mean([r['prob'] for r in nz if r['model'] == m and r['cond'] == c], axis=0) for m in MODELS}
    rows_c = []
    for a, b in pairs:
        d, ci, p = boot_auc_diff(E[a], E[b], seed=1)
        rows_c.append(dict(a=a, b=b, dauc=d, ci=ci, p_boot=p))
    for r, v in zip(rows_c, holm([r['p_boot'] for r in rows_c])): r['p_boot_holm'] = float(v)
    S['noise_pairs'][c] = dict(rows=rows_c, ens_auc={m: float(roc_auc_score(y, E[m])) for m in MODELS})
    # per-seed degradation relative to clean, Friedman across models
    D = []
    for m in MODELS:
        cl = {r['seed']: r['auc'] for r in nz if r['model'] == m and r['cond'] == 'clean_None'}
        D.append([cl[r['seed']] - r['auc'] for r in sorted([r for r in nz if r['model'] == m and r['cond'] == c], key=lambda r: r['seed'])])
    S['noise_pairs'][c]['drop_mean'] = {m: float(np.mean(d)) for m, d in zip(MODELS, D)}
    S['noise_pairs'][c]['drop_friedman_p'] = float(stats.friedmanchisquare(*D)[1])

# ---------------- label fraction ----------------
S['frac'] = {}
for f in ('0.01', '0.1', '1.0'):
    S['frac'][f] = {}
    M = []
    for m in MODELS:
        R = load(f'results/{"main" if f == "1.0" else "frac"}/{m}_f{f}_s*.json')
        if not R: continue
        R = sorted(R, key=lambda r: r['seed'])
        S['frac'][f][m] = dict(auc=summarize(R, 'auc')[:2], acc=summarize(R, 'acc')[:2], n_train=R[0]['n_train'],
                               steps=R[0]['steps'], n=len(R),
                               train_time_s=float(np.mean([r['train_time_s'] for r in R])))
        M.append([r['test']['auc'] for r in R])
    if len(M) == 4 and len(set(map(len, M))) == 1:
        S['frac'][f]['friedman_p'] = float(stats.friedmanchisquare(*M)[1])

# ---------------- cost ----------------
S['bench'] = {json.loads(l)['model']: json.loads(l) for l in open('results/bench.jsonl')}
json.dump(S, open('results/summary.json', 'w'), indent=1)

# ---------------- figures ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
snr = [None, 20, 10, 5, 0, -5]; xs = ['clean', '20', '10', '5', '0', '−5']
for m in MODELS:
    mu = [S['noise']['clean_None' if s is None else f'white_{s}'][m]['auc'][0] for s in snr]
    sd = [S['noise']['clean_None' if s is None else f'white_{s}'][m]['auc'][1] for s in snr]
    ax[0].errorbar(range(6), mu, yerr=sd, color=COL[m], marker=MK[m], capsize=3, label=m)
    fr = [0, 0.1, 0.3, 0.5]
    mu = [S['noise']['clean_None' if f == 0 else f'drop_{f}'][m]['auc'][0] for f in fr]
    sd = [S['noise']['clean_None' if f == 0 else f'drop_{f}'][m]['auc'][1] for f in fr]
    ax[1].errorbar([100 * f for f in fr], mu, yerr=sd, color=COL[m], marker=MK[m], capsize=3, label=m)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(xs); ax[0].set_xlabel('Added white-noise SNR (dB)')
ax[0].set_ylabel('Test ROC-AUC (mean ± SD, 5 seeds)'); ax[0].set_title('(a) Additive Gaussian noise', fontsize=10)
ax[1].set_xlabel('Channels set to zero (%)'); ax[1].set_title('(b) Dead / decoupled channels', fontsize=10)
ax[1].set_xticks([0, 10, 30, 50]); ax[0].legend(fontsize=8, loc='lower left')
for a in ax: a.grid(alpha=.3)
plt.savefig(f'{FIG}/fig3_noise.pdf'); plt.savefig(f'{FIG}/fig3_noise.png', dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(5, 3.8), constrained_layout=True)
fr = [f for f in ('0.01', '0.1', '1.0') if all(m in S['frac'][f] for m in MODELS)]
for j, m in enumerate(MODELS):
    xs_ = np.array([S['frac'][f][m]['n_train'] for f in fr]) * (1 + 0.04 * (j - 1.5))
    ax.errorbar(xs_, [S['frac'][f][m]['auc'][0] for f in fr], yerr=[S['frac'][f][m]['auc'][1] for f in fr],
                color=COL[m], marker=MK[m], capsize=3, label=m)
ax.set_xscale('log'); ax.set_xlabel('Number of labelled training patches'); ax.set_ylabel('Test ROC-AUC (mean ± SD)')
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
plt.savefig(f'{FIG}/fig4_labels.pdf'); plt.savefig(f'{FIG}/fig4_labels.png', dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for m in MODELS:
    b = S['bench'][m]; r = S['main'][m]
    ax[0].errorbar(r['train_time_s'][0] / 60, r['auc'][0], xerr=r['train_time_s'][1] / 60, yerr=r['auc'][1],
                   color=COL[m], marker=MK[m], capsize=3, ms=8, label=m)
    ax[1].scatter(b['latency_ms_b1'], r['auc'][0], s=b['params'] / 600, color=COL[m], marker=MK[m], label=m)
ax[0].set_xlabel('Training wall-clock time per run (min, 1 CPU thread)'); ax[0].set_ylabel('Test ROC-AUC')
ax[1].set_xlabel('Inference latency, batch of 1 (ms)'); ax[1].set_ylabel('Test ROC-AUC')
ax[0].set_title('(a) ROC-AUC vs training cost', fontsize=10); ax[1].set_title('(b) ROC-AUC vs latency (marker area ∝ parameters)', fontsize=10)
for a in ax: a.grid(alpha=.3)
ax[0].legend(fontsize=8)
ax[1].margins(0.2)
plt.savefig(f'{FIG}/fig5_cost.pdf'); plt.savefig(f'{FIG}/fig5_cost.png', dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(5, 3.8), constrained_layout=True)
from sklearn.metrics import roc_curve
for m in MODELS:
    fpr, tpr, _ = roc_curve(y, ens[m])
    ax.plot(fpr, tpr, color=COL[m], label=f"{m} (AUC {S['main'][m]['ens_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k:', lw=.8); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=.3)
plt.savefig(f'{FIG}/fig2_roc.pdf'); plt.savefig(f'{FIG}/fig2_roc.png', dpi=130); plt.close()
print(json.dumps({m: {k: S['main'][m][k] for k in ('auc', 'acc', 'ens_auc', 'ens_acc')} for m in MODELS}, indent=0))
print('friedman', S['friedman_auc']); [print(r) for r in rows]
