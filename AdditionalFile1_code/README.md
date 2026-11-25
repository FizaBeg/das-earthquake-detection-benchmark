# Code for "Comparing CNN, CNN-LSTM, ConvLSTM and Transformer architectures for earthquake detection on telecommunication-fibre DAS data"

Data: Huot F (2022) Stanford fiber-optic DAS array: Earthquake detection dataset. Zenodo. https://doi.org/10.5281/zenodo.6344570 (CC BY 4.0).
Download the 12 `*.tfrecord.gz` files into `/home/claude/data/stanford` (or edit the paths at the top of `prep.py` and `train.py`).

Environment used: Python 3.13, PyTorch 2.15 (CPU), scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.3, `tfrecord` (only for its protobuf definition).

Steps
1. `python prep.py` – reads the TFRecords, writes numpy arrays and `dataset_info.json` (counts, class balance, duplicate check).
2. `python -c "from train import make_cache; make_cache()"` – average-pools every example to 48 x 347.
3. `for m in CNN CNN-LSTM ConvLSTM Transformer; do python bench.py $m; done > results/bench.jsonl` – cost measurements.
4. `./queue.sh jobs_main.txt results/main` – E1 (5 seeds x 4 models, all labels).
5. `./queue.sh jobs_frac.txt results/frac` – E3 (1 % and 10 % labels).
6. `python eval_noise.py` – E2 (synthetic perturbations of the real test set).
7. `python analyze.py` – statistics (Friedman, paired bootstrap, McNemar, Holm), `results/summary.json` and figures.
8. `python fig_data.py` – Fig. 1.
