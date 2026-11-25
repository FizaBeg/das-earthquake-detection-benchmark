"""Convert the Stanford DAS earthquake-detection TFRecords (Zenodo 10.5281/zenodo.6344570)
into numpy arrays. Each example: 288 channels x 695 samples (float32), label 1=earthquake, 0=noise."""
import glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from read_tfr import iterate

SRC = '/home/claude/data/stanford'
OUT = '/home/claude/data/npy'
os.makedirs(OUT, exist_ok=True)
info = {}
hashes = {}
for split, pat in [('train', 'train-*'), ('eval', 'eval-*'), ('test', 'test-*')]:
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(SRC, pat + '.tfrecord.gz'))):
        for x, y in iterate(f):
            X.append(x.reshape(288, 695).astype(np.float16)); Y.append(int(y[0]))
            hashes.setdefault(hashlib.md5(x.tobytes()).hexdigest(), []).append(split)
    X = np.stack(X); Y = np.array(Y, dtype=np.int64)
    np.save(f'{OUT}/{split}_X.npy', X); np.save(f'{OUT}/{split}_y.npy', Y)
    info[split] = dict(n=int(len(Y)), pos=int(Y.sum()), neg=int((Y == 0).sum()),
                       shape=list(X.shape[1:]), min=float(X.min()), max=float(X.max()),
                       nan=int(np.isnan(X.astype(np.float32)).sum()))
    print(split, info[split], flush=True)
dups = {h: s for h, s in hashes.items() if len(s) > 1}
info['exact_duplicates'] = len(dups)
info['cross_split_duplicates'] = sum(1 for s in dups.values() if len(set(s)) > 1)
print('dups', info['exact_duplicates'], 'cross-split', info['cross_split_duplicates'])
json.dump(info, open(f'{OUT}/dataset_info.json', 'w'), indent=1)
