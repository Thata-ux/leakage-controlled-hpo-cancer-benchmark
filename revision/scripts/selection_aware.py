"""Round-2 (R1.5): dataset-level tuning effect without test-set selection.
(a) confirmatory: each prespecified tuner vs no tuning, Holm over 3 tuners x 3 datasets (9 tests) per outcome;
(b) selection-aware (nested): per run and classifier, the tuner with the highest inner-CV weighted F1 (training data only);
(c) exploratory: tuner with the highest mean test score (original analysis)."""
import sys, numpy as np, pandas as pd
from scipy import stats
Z, R, OUT = sys.argv[1:4]
CORR = 0.35
def nb(d):
    d = np.asarray(d, float); m = d.mean(); se = np.sqrt(CORR * d.var(ddof=1)); q = stats.t.ppf(.975, len(d) - 1)
    p = 2 * stats.t.sf(abs(m / se), len(d) - 1) if se > 0 else 1.0
    return m, m - q * se, m + q * se, p
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
DS = ['Lung Cancer', 'BreastCancer', 'CervicalCancer']; MD = ['XGBoost', 'RandomForest', 'SVM', 'LightGBM']; ALT = ['Grid', 'Random', 'Bayes']
lf = pd.read_csv(f'{Z}/outputs/S4_run_level_results_leakfree.csv').rename(columns={'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'})
C = pd.read_csv(f'{R}/C_matched_space_results.csv'); C['Run'] = C['run'] + 1
rows = []
for design, df in [('Original', lf), ('Matched', C)]:
    for met in ['f1', 'mcc', 'recall_min']:
        conf = []
        for d in DS:
            piv = {m: df[(df.dataset == d) & (df.model == m)].pivot(index='Run', columns='tuner', values=met) for m in MD}
            cv = {m: df[(df.dataset == d) & (df.model == m)].pivot(index='Run', columns='tuner', values='cv_best') for m in MD}
            for t in ALT:
                g = np.mean([(piv[m][t] - piv[m]['No-Tuning']).values for m in MD], axis=0)
                conf.append(dict(design=design, metric=met, dataset=d, procedure=f'Prespecified: {t}', **dict(zip(['mean', 'lo', 'hi', 'p'], nb(g)))))
            sel = np.mean([np.array([piv[m].loc[r, cv[m].loc[r, ALT].astype(float).idxmax()] - piv[m].loc[r, 'No-Tuning'] for r in piv[m].index]) for m in MD], axis=0)
            rows.append(dict(design=design, metric=met, dataset=d, procedure='Selection-aware (inner-CV choice)', **dict(zip(['mean', 'lo', 'hi', 'p'], nb(sel)))))
            ex = np.mean([(piv[m][piv[m][ALT].mean().idxmax()] - piv[m]['No-Tuning']).values for m in MD], axis=0)
            rows.append(dict(design=design, metric=met, dataset=d, procedure='Exploratory: best test mean', **dict(zip(['mean', 'lo', 'hi', 'p'], nb(ex)))))
        adj = holm([c['p'] for c in conf])
        for c, a in zip(conf, adj): c['p_holm9'] = a
        rows += conf
out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
pd.set_option('display.width', 200)
print(out.round(4).to_string())
