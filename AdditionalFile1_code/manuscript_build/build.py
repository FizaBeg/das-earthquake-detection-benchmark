"""Assemble main.tex from parts; every number in the text and tables comes from results/summary.json
(or dataset_info.json / bench.jsonl) through %%KEY%% placeholders, so nothing is typed by hand."""
import json, re
S = json.load(open('/home/claude/work/results/summary.json'))
info = json.load(open('/home/claude/data/npy/dataset_info.json'))
M = ['CNN', 'CNN-LSTM', 'ConvLSTM', 'Transformer']
K = {'CNN': 'CNN', 'CNN-LSTM': 'CNNLSTM', 'ConvLSTM': 'CONVLSTM', 'Transformer': 'TRANS'}
pm = lambda v: f'{v[0]:.3f} $\\pm$ {v[1]:.3f}'
f3 = lambda x: f'{x:.3f}'


def p_fmt(p):
    """for table cells (text mode)"""
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def p_eq(p):
    """for running text inside math: '$p%%KEY%%$' -> 'p<0.001' or 'p=0.012'"""
    return '<0.001' if p < 0.001 else f'={p:.3f}'


R = {'NTRAIN': info['train']['n'], 'NEVAL': info['eval']['n'], 'NTEST': info['test']['n'],
     'NTRAINPOS': info['train']['pos'], 'STEPS': S['main']['CNN']['steps'],
     'N001': S['frac']['0.01']['CNN']['n_train'], 'N01': S['frac']['0.1']['CNN']['n_train'],
     'FRIEDAUCCHI': f"{S['friedman_auc'][0]:.2f}", 'FRIEDAUCP': p_eq(S['friedman_auc'][1]),
     'FRIEDACCCHI': f"{S['friedman_acc'][0]:.2f}", 'FRIEDACCP': p_eq(S['friedman_acc'][1]),
     'MAXDAUC': f3(max(abs(r['dauc']) for r in S['pairs'])),
     'MINPHOLM': (lambda v: 'all Holm-adjusted $p$-values are 1.000' if v > 0.999 else f'smallest Holm-adjusted $p={v:.3f}$')(min(min(r['p_boot_holm'], r['p_mcnemar_holm']) for r in S['pairs'])),
     'MINPRAW': f"{min(r['p_boot'] for r in S['pairs']):.2f}",
     'N5FRIEDP': p_eq(S['noise_pairs']['white_-5']['drop_friedman_p']),
     'N0FRIEDP': p_eq(S['noise_pairs']['white_0']['drop_friedman_p']),
     'D50FRIEDP': p_eq(S['noise_pairs']['drop_0.5']['drop_friedman_p'])}
for f in ('0.01', '0.1'):
    if 'friedman_p' in S['frac'][f]:
        R[f'FRAC{f.replace(".", "")}P'] = p_eq(S['frac'][f]['friedman_p'])
for m in M:
    k, r, b = K[m], S['main'][m], S['bench'][m]
    R.update({f'AUC_{k}': pm(r['auc']), f'ACC_{k}': pm(r['acc']), f'F1_{k}': pm(r['f1']),
              f'EAUC_{k}': f3(r['ens_auc']), f'EAUCLO_{k}': f3(r['ens_auc_ci'][0]), f'EAUCHI_{k}': f3(r['ens_auc_ci'][1]),
              f'EACC_{k}': f3(r['ens_acc']), f'TP_{k}': r['ens_tp'], f'FP_{k}': r['ens_fp'], f'FN_{k}': r['ens_fn'],
              f'TN_{k}': r['ens_tn'], f'TIME_{k}': f"{r['train_time_s'][0]:.0f} $\\pm$ {r['train_time_s'][1]:.0f}",
              f'PAR_{k}': f"{b['params']:,}".replace(',', '\\,'), f'GFLOP_{k}': f"{b['gflops_fwd']:.3f}",
              f'LAT_{k}': f"{b['latency_ms_b1']:.1f}", f'STEP_{k}': f"{b['step_s_b32']:.2f}", f'MEM_{k}': f"{b['train_mem_mb']:.0f}",
              f'ACCMIN_{k}': f3(min(r['acc'][2])), f'ACCMAX_{k}': f3(max(r['acc'][2]))})
    for c, ck in (('white_0', 'N0'), ('white_-5', 'N5'), ('drop_0.5', 'D50')):
        R[f'{ck}EAUC_{k}'] = f3(S['noise_pairs'][c]['ens_auc'][m])
        R[f'{ck}DROP_{k}'] = f3(S['noise_pairs'][c]['drop_mean'][m])
        R[f'{ck}AUC_{k}'] = pm(S['noise'][c][m]['auc'])
    for f in ('0.01', '0.1', '1.0'):
        if m in S['frac'][f]:
            R[f'F{f.replace(".", "")}_{k}'] = pm(S['frac'][f][m]['auc'])
for c, ck in (('white_0', 'N0'), ('white_-5', 'N5'), ('drop_0.5', 'D50')):
    for r in S['noise_pairs'][c]['rows']:
        key = f"{ck}P_{K[r['a']]}_{K[r['b']]}"
        R[key] = p_eq(r['p_boot_holm']); R[key.replace('P_', 'D_', 1)] = f3(r['dauc'])
        R[key.replace('P_', 'CI_', 1)] = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"

R['CODEREPO'] = '[authors: deposit the code in a public repository (e.g. Zenodo or GitHub) and give its DOI or URL here]'
R['FUNDING'] = '[authors: state funding sources, or write "Not applicable".]'
R['CONTRIB'] = '[authors: describe each author\'s contribution, e.g. "AO designed the study, ran the experiments and wrote the manuscript. AT supervised the work and revised the manuscript. All authors read and approved the final manuscript."]'
R['AIUSE'] = 'An AI assistant (Claude, Anthropic) was used to help write the analysis code and draft the manuscript text; the authors checked the code, results, references and text and take full responsibility for the content.'
R['E3DISC'] = (r'\subsection*{Label efficiency is the price of the Transformer}' '\n'
  r'The architecture that was most robust to noise was also the least label-efficient. With 10\,\% and especially 1\,\% of the labels the Transformer had the lowest ROC-AUC, whereas the two models with convolutional front ends (CNN and CNN--LSTM) degraded least. '
  r'This agrees with the observation that Vision Transformers, which lack the locality bias of convolutions, need more training data than CNNs to reach comparable accuracy \citep{Dosovitskiy2021}. '
  r'For DAS arrays where only a few dozen catalogued events can be labelled, a convolutional model is the safer choice; self-supervised pre-training on unlabelled DAS records, which are abundant, is a natural way to reduce the label needs of attention-based models.')
R['E3CONC'] = 'With few labels the ranking reversed: the convolutional models degraded least and the Transformer most.'
R['ABSTRACT'] = open('/home/claude/paper/part_abstract.tex').read().strip()
R['AUCMIN'] = f"{min(S['main'][m]['auc'][0] for m in M):.3f}"; R['AUCMAX'] = f"{max(S['main'][m]['auc'][0] for m in M):.3f}"
R['D50MAX'] = f"{max(S['noise_pairs']['drop_0.5']['drop_mean'][m] for m in M):.2f}"
R['F001LO'] = f"{S['frac']['0.01']['CNN']['auc'][0]:.2f}"; R['F001HI'] = f"{S['frac']['0.01']['CNN-LSTM']['auc'][0]:.2f}"
R['F001TRANSM'] = f"{S['frac']['0.01']['Transformer']['auc'][0]:.2f}"
R['NEVALPOS'] = info['eval']['pos']; R['NTESTPOS'] = info['test']['pos']
NM = {'CNN': 'CNN', 'CNN-LSTM': 'CNN--LSTM', 'ConvLSTM': 'ConvLSTM', 'Transformer': 'Transformer'}
R['PAIRROWS'] = '\n'.join(
    f"{NM[r['a']]} & {NM[r['b']]} & {r['dauc']:.3f} & [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] & {p_fmt(r['p_boot_holm'])} & "
    f"{r['a_only_correct']} / {r['b_only_correct']} & {p_fmt(r['p_mcnemar_holm'])}\\\\" for r in S['pairs'])
CN = {'clean_None': 'Clean', 'white_20': 'White noise, 20 dB', 'white_10': 'White noise, 10 dB',
      'white_5': 'White noise, 5 dB', 'white_0': 'White noise, 0 dB', 'white_-5': 'White noise, $-5$ dB',
      'drop_0.1': '10\\,\\% channels zeroed', 'drop_0.3': '30\\,\\% channels zeroed', 'drop_0.5': '50\\,\\% channels zeroed'}
R['NOISEROWS'] = '\n'.join(
    f"{CN[c]} & " + ' & '.join(pm(S['noise'][c][m]['auc']) for m in M) + f" & {p_fmt(S['noise'][c]['friedman_p'])}\\\\"
    for c in CN)
FL = {'0.01': '1\\,\\%', '0.1': '10\\,\\%', '1.0': '100\\,\\%'}
R['FRACROWS'] = '\n'.join(
    f"{FL[f]} ({S['frac'][f]['CNN']['n_train']}) & " + ' & '.join(pm(S['frac'][f][m]['auc']) for m in M) +
    f" & {p_fmt(S['frac'][f]['friedman_p'])}\\\\" for f in ('0.01', '0.1', '1.0') if 'friedman_p' in S['frac'][f])
R['E3TEXT'] = open('/home/claude/paper/part_e3.tex').read().strip()

parts = [open(f'/home/claude/paper/{p}.tex').read() for p in
         ('part_head', 'part_intro_methods', 'part_results', 'part_tail', 'part_refs', 'part_floats')]
tex = '\n'.join(parts)
tex = tex.replace('%%E3TEXT%%', R.pop('E3TEXT')).replace('%%ABSTRACT%%', R.pop('ABSTRACT'))
for k, v in R.items():
    tex = tex.replace(f'%%{k}%%', str(v))
left = sorted(set(re.findall(r'%%[A-Za-z0-9_]+%%', tex)))
assert not left, left
open('/home/claude/paper/main.tex', 'w').write(tex + '\n\\end{document}\n')
json.dump({k: str(v) for k, v in R.items()}, open('/home/claude/paper/filled_values.json', 'w'), indent=0)
print('ok', len(R), 'keys')
