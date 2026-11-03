"""Dependence-aware re-analysis of the S7 leak-free run-level results (reviewer R1.2, R1.5, R2.6)."""
import pandas as pd, numpy as np, sys, os
from scipy import stats
D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
lf = pd.read_csv(f'{D}/outputs/S4_run_level_results_leakfree.csv')
DS = ['Lung Cancer', 'BreastCancer', 'CervicalCancer']
MD = ['XGBoost', 'RandomForest', 'SVM', 'LightGBM']
ALT = ['Grid', 'Random', 'Bayes']
MET = ['f1', 'mcc', 'recall_min', 'bal_acc', 'f1_macro', 'auc']
J = 10; RHO = 0.2 / 0.8          # n_test / n_train for the 80:20 hold-out
CORR = 1 / J + RHO               # Nadeau & Bengio (2003) corrected variance factor

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj

def nb(diff, alpha=0.05):
    d = np.asarray(diff, float); n = len(d); m = d.mean(); s2 = d.var(ddof=1)
    se = np.sqrt(CORR * s2) if s2 > 0 else 0.0
    se_naive = np.sqrt(s2 / n) if s2 > 0 else 0.0
    tq = stats.t.ppf(1 - alpha / 2, n - 1); tq90 = stats.t.ppf(0.95, n - 1)
    t = m / se if se > 0 else (np.inf if m != 0 else 0)
    p = 2 * stats.t.sf(abs(t), n - 1) if se > 0 else (0.0 if m != 0 else 1.0)
    pn = 2 * stats.t.sf(abs(m / se_naive), n - 1) if se_naive > 0 else (0.0 if m != 0 else 1.0)
    return dict(mean=m, se=se, lo=m - tq * se, hi=m + tq * se, lo90=m - tq90 * se, hi90=m + tq90 * se, p_nb=p, p_naive_t=pn)

piv = {}
for (d, m), g in lf.groupby(['Dataset', 'Model']):
    piv[(d, m)] = {met: g.pivot(index='Run', columns='Tuner', values=met) for met in MET + ['runtime']}

# ---- 1. all tuner-vs-NT contrasts, every outcome, NB-corrected, Holm within outcome (36 contrasts)
rows = []
for met in MET:
    for d in DS:
        for m in MD:
            P = piv[(d, m)][met]
            for t in ALT:
                diff = (P[t] - P['No-Tuning']).values
                w = stats.wilcoxon(diff, zero_method='wilcox').pvalue if np.any(diff != 0) else 1.0
                rows.append(dict(metric=met, dataset=d, model=m, tuner=t, **nb(diff), p_wilcoxon=w))
C = pd.DataFrame(rows)
C['p_nb_holm36'] = C.groupby('metric')['p_nb'].transform(lambda s: holm(s.values))
C['p_nb_holm3'] = C.groupby(['metric', 'dataset', 'model'])['p_nb'].transform(lambda s: holm(s.values))
C['p_wilcoxon_holm36'] = C.groupby('metric')['p_wilcoxon'].transform(lambda s: holm(s.values))
C.round(5).to_csv(f'{OUT}/R_contrasts_all_outcomes.csv', index=False)

# ---- 2. sensitivity of significance counts (combinations with >=1 tuner significantly different from NT)
def count(df, col, a=0.05):
    s = df[df[col] < a]
    return len(s.groupby(['dataset', 'model'])), sorted({(r.dataset, r.model) for r in s.itertuples()})
sens = []
for met in ['f1', 'mcc', 'recall_min', 'bal_acc']:
    sub = C[C.metric == met]
    for col, lab in [('p_wilcoxon', 'Wilcoxon, uncorrected'), ('p_wilcoxon_holm36', 'Wilcoxon, Holm (36)'),
                     ('p_naive_t', 'paired t, naive'), ('p_nb', 'NB-corrected t, uncorrected'), ('p_nb_holm3', 'NB-corrected t, Holm within combination (3)'),
                     ('p_nb_holm36', 'NB-corrected t, Holm across 36')]:
        n, which = count(sub, col)
        sens.append(dict(metric=met, procedure=lab, n_combinations=n, combinations='; '.join(f'{a}/{b}' for a, b in which)))
S = pd.DataFrame(sens); S.to_csv(f'{OUT}/R_significance_sensitivity.csv', index=False)
print(S[['metric', 'procedure', 'n_combinations']].to_string(index=False))

# ---- 3. equivalence (TOST via 90% NB-corrected CI) for weighted F1, margins 0.005/0.01/0.02
eq = []
for r in C[C.metric == 'f1'].itertuples():
    for delta in [0.005, 0.01, 0.02]:
        if r.lo90 > -delta and r.hi90 < delta: v = 'equivalent'
        elif r.lo > 0: v = 'superior'
        elif r.hi < 0: v = 'inferior'
        elif r.lo90 > -delta: v = 'non-inferior'
        else: v = 'inconclusive'
        eq.append(dict(dataset=r.dataset, model=r.model, tuner=r.tuner, margin=delta, mean=r.mean, lo90=r.lo90, hi90=r.hi90, verdict=v))
E = pd.DataFrame(eq); E.round(5).to_csv(f'{OUT}/R_equivalence_f1.csv', index=False)
print(E.groupby(['margin', 'verdict']).size().unstack(fill_value=0))

# ---- 4. per-dataset pooled gain (mean over 4 classifiers within run → run is the unit), NB-corrected
pool = []
for met in ['f1', 'mcc', 'recall_min']:
    for d in DS:
        for t in ALT + ['best-alt']:
            per_run = []
            for m in MD:
                P = piv[(d, m)][met]
                if t == 'best-alt':
                    b = P[ALT].mean().idxmax(); per_run.append((P[b] - P['No-Tuning']).values)
                else:
                    per_run.append((P[t] - P['No-Tuning']).values)
            diff = np.mean(per_run, axis=0)
            pool.append(dict(metric=met, dataset=d, tuner=t, **nb(diff)))
PL = pd.DataFrame(pool); PL.round(5).to_csv(f'{OUT}/R_pooled_per_dataset.csv', index=False)
print(PL[PL.metric == 'f1'][['dataset', 'tuner', 'mean', 'lo', 'hi', 'p_nb']].round(4).to_string(index=False))

# ---- 5. absolute runtime, extra seconds vs NT with 95% CI over runs
rt = []
for d in DS:
    for m in MD:
        P = piv[(d, m)]['runtime']
        for t in ALT:
            x = (P[t] - P['No-Tuning']).values; h = stats.t.ppf(.975, 9) * x.std(ddof=1) / np.sqrt(10)
            rt.append(dict(dataset=d, model=m, tuner=t, nt_s=P['No-Tuning'].mean(), tuned_s=P[t].mean(), extra_s=x.mean(), extra_lo=x.mean() - h, extra_hi=x.mean() + h))
RT = pd.DataFrame(rt); RT.round(3).to_csv(f'{OUT}/R_runtime_absolute.csv', index=False)
print('extra seconds range', RT.extra_s.min().round(2), RT.extra_s.max().round(2), 'median', RT.extra_s.median().round(2))

# ---- 6. best-tuner post-selection: per outcome, which tuner has best mean; report its NB contrast
best = []
for met in ['f1', 'mcc', 'recall_min', 'bal_acc', 'f1_macro']:
    for d in DS:
        for m in MD:
            sub = C[(C.metric == met) & (C.dataset == d) & (C.model == m)]
            r = sub.loc[sub['mean'].idxmax()]
            best.append(dict(metric=met, dataset=d, model=m, best_tuner=r.tuner, gain=r['mean'], lo=r.lo, hi=r.hi, p_nb=r.p_nb, p_nb_holm3=r.p_nb_holm3, p_nb_holm36=r.p_nb_holm36))
BT = pd.DataFrame(best); BT.round(5).to_csv(f'{OUT}/R_best_tuner_post_selection.csv', index=False)
print(BT[(BT.dataset == 'CervicalCancer')].round(4).to_string(index=False))
