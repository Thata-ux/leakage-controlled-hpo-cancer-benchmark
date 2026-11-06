import sys, numpy as np, pandas as pd
from scipy import stats
Z, R = sys.argv[1:3]
CORR = 0.35
def nb(d):
    d = np.asarray(d, float); m = d.mean(); se = np.sqrt(CORR * d.var(ddof=1)); q = stats.t.ppf(.975, len(d) - 1)
    return m, m - q * se, m + q * se, (2 * stats.t.sf(abs(m / se), len(d) - 1) if se > 0 else 1.0)
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
DS = ['Lung Cancer', 'BreastCancer', 'CervicalCancer']; MD = ['XGBoost', 'RandomForest', 'SVM', 'LightGBM']; ALT = ['Grid', 'Random', 'Bayes']
ren = {'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'}
lf = pd.read_csv(f'{Z}/outputs/S4_run_level_results_leakfree.csv').rename(columns=ren)
S4b = pd.read_csv(f'{Z}/outputs/S4b_run_level_results_leaky.csv').rename(columns=ren)
B = pd.read_csv(f'{R}/B_presplit_origtest_results.csv'); B['Run'] = B['run'] + 1
C = pd.read_csv(f'{R}/C_matched_space_results.csv'); C['Run'] = C['run'] + 1
P = {'Leakage-free': lf, 'Pre-split': S4b[S4b.Protocol == 'Leak-PreSplit'], 'Pre-split, original test': B, 'Pre-CV': S4b[S4b.Protocol == 'Leak-PreCV'], 'Matched': C}
def sel_gain(df, d, met, m=None):
    per = []
    for mm in ([m] if m else MD):
        g = df[(df.dataset == d) & (df.model == mm)]
        v = g.pivot(index='Run', columns='tuner', values=met); c = g.pivot(index='Run', columns='tuner', values='cv_best')
        per.append(np.array([v.loc[r, c.loc[r, ALT].astype(float).idxmax()] - v.loc[r, 'No-Tuning'] for r in v.index]))
    return np.mean(per, axis=0)
rows = []
for proto, df in P.items():
    for met in ['f1', 'mcc', 'recall_min']:
        res = [nb(sel_gain(df, d, met)) for d in DS]
        adj = holm([r[3] for r in res])
        for d, r, a in zip(DS, res, adj): rows.append(dict(protocol=proto, metric=met, dataset=d, mean=r[0], lo=r[1], hi=r[2], p=r[3], p_holm3=a))
OUT = sys.argv[3] if len(sys.argv) > 3 else '.'
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/R2_selection_aware_protocols.csv', index=False)
print(T.round(4).to_string())
# per combination (figure)
fr = []
for d in DS:
    for m in MD:
        r = nb(sel_gain(lf, d, 'f1', m)); fr.append(dict(dataset=d, model=m, mean=r[0], lo=r[1], hi=r[2], p=r[3]))
F = pd.DataFrame(fr); F.to_csv(f'{OUT}/R2_selection_aware_per_combination.csv', index=False); print(F.round(4))
# chosen tuner frequencies
ch = []
for d in DS:
    for m in MD:
        c = lf[(lf.dataset == d) & (lf.model == m)].pivot(index='Run', columns='tuner', values='cv_best')[ALT].astype(float).idxmax(axis=1)
        ch += list(c)
print(pd.Series(ch).value_counts())
