import pandas as pd, numpy as np, json, sys, os
from scipy import stats
R, Z, AN, OUT = sys.argv[1:5]
CORR = 0.35
def nb(d):
    d = np.asarray(d, float); m = d.mean(); se = np.sqrt(CORR * d.var(ddof=1)); q = stats.t.ppf(.975, len(d) - 1)
    p = 2 * stats.t.sf(abs(m / se), len(d) - 1) if se > 0 else 1.0
    return m, m - q * se, m + q * se, p
def f(x, k=3): return (f'{x:+.{k}f}').replace('-', '−')
def ci(m, lo, hi, k=3): return f'{f(m,k)} [{f(lo,k)}, {f(hi,k)}]'
def pv(p): return '<0.001' if p < 0.001 else f'{p:.3f}'
DS = ['Lung Cancer', 'BreastCancer', 'CervicalCancer']; DSN = {'Lung Cancer': 'Lung', 'BreastCancer': 'Breast', 'CervicalCancer': 'Cervical'}
MD = ['XGBoost', 'RandomForest', 'SVM', 'LightGBM']; MDN = {'XGBoost': 'XGBoost', 'RandomForest': 'RF', 'SVM': 'SVM', 'LightGBM': 'LightGBM'}
ALT = ['Grid', 'Random', 'Bayes']
T = {}
s = pd.read_csv(f'{Z}/outputs/ALL_results_summary.csv').set_index(['Dataset', 'Model', 'Tuner'])
Cn = pd.read_csv(f'{AN}/R_contrasts_all_outcomes.csv')
lf = pd.read_csv(f'{Z}/outputs/S4_run_level_results_leakfree.csv')
rt = pd.read_csv(f'{AN}/R_runtime_absolute.csv')
rows = []
for d in DS:
    for m in MD:
        nt = s.loc[(d, m, 'No-Tuning')]
        sub = Cn[(Cn.metric == 'f1') & (Cn.dataset == d) & (Cn.model == m)]
        b = sub.loc[sub['mean'].idxmax()]
        al = s.loc[(d, m, b.tuner)]
        ex = rt[(rt.dataset == d) & (rt.model == m) & (rt.tuner == b.tuner)].iloc[0]
        rows.append([DSN[d], MDN[m], f'{nt.Mean_F1:.3f} ± {nt.Std_F1:.3f}', b.tuner, f'{al.Mean_F1:.3f} ± {al.Std_F1:.3f}', ci(b['mean'], b.lo, b.hi, 3), pv(b.p_nb_holm3), f'{ex.extra_s:.1f} [{ex.extra_lo:.1f}, {ex.extra_hi:.1f}]'])
T['tmain'] = rows

# pooled per dataset: original vs matched
A = pd.read_csv(f'{R}/A_main_logged_results.csv'); C = pd.read_csv(f'{R}/C_matched_space_results.csv')
for df in (A, C): df['Run'] = df['run'] + 1
def pooled(df, d, met):
    per = []
    for m in MD:
        p = df[(df.dataset == d) & (df.model == m)].pivot(index='Run', columns='tuner', values=met); b = p[ALT].mean().idxmax(); per.append((p[b] - p['No-Tuning']).values)
    return nb(np.mean(per, axis=0))
lf2 = lf.rename(columns={'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'})
rows = []
for d in DS:
    for lab, df in [('Original', lf2), ('Matched', C)]:
        r = [DSN[d], lab]
        for met in ['f1', 'mcc', 'recall_min']:
            m, lo, hi, p = pooled(df, d, met); r.append(ci(m, lo, hi) + ('*' if p < 0.05 else ''))
        rows.append(r)
T['tpool'] = rows

# cervical minority table per tuner-selection, NB + Holm(3)
rows = []
for m in MD:
    r = [MDN[m]]
    for met in ['mcc', 'recall_min']:
        sub = Cn[(Cn.metric == met) & (Cn.dataset == 'CervicalCancer') & (Cn.model == m)]; b = sub.loc[sub['mean'].idxmax()]
        nt = lf[(lf.Dataset == 'CervicalCancer') & (lf.Model == m) & (lf.Tuner == 'No-Tuning')][met].mean()
        r += [b.tuner, f'{nt:.3f} → {nt + b["mean"]:.3f}', ci(b['mean'], b.lo, b.hi), pv(b.p_nb_holm3)]
    rows.append(r)
T['t10'] = rows

# sample counts
cnt = pd.read_csv(f'{R}/B_sample_counts.csv')
g = cnt.groupby(['dataset', 'protocol', 'part'])[['n_original', 'n_synthetic', 'n_class0', 'n_class1']].mean()
rows = []
for d in DS:
    for proto, lab in [('Leak-PreSplit (original)', 'Pre-split'), ('Leak-PreSplit-OrigTest (block B)', 'Pre-split, original test')]:
        tr = g.loc[(d, proto, 'train')]; te = g.loc[(d, proto, 'test')]
        rows.append([DSN[d], lab, f'{tr.n_original:.0f} / {tr.n_synthetic:.0f}', f'{te.n_original:.0f} / {te.n_synthetic:.0f}', f'{te.n_class0:.0f} / {te.n_class1:.0f}'])
T['tcount'] = rows

# decomposition
dec = pd.read_csv(f'{AN}/../analysis2/B_decomposition.csv') if os.path.exists(f'{AN}/../analysis2/B_decomposition.csv') else None
S4b = pd.read_csv(f'{Z}/outputs/S4b_run_level_results_leaky.csv')
key = ['Dataset', 'Model', 'Tuner', 'Run']
lfk = lf.set_index(key); pcv = S4b[S4b.Protocol == 'Leak-PreCV'].set_index(key)
rows = []
MN = {'f1': 'Weighted F1', 'auc': 'ROC-AUC', 'mcc': 'MCC', 'recall_min': 'Minority recall'}
for met in ['f1', 'auc', 'mcc', 'recall_min']:
    rr = dec[(dec.metric == met) & (dec.dataset == 'All')].iloc[0]
    c = (pcv[met] - lfk[met]).dropna().mean()
    rows.append([MN[met], f(rr.total), f(rr.contaminated_training), f(rr.synthetic_test), f(c)])
T['tdec'] = rows

# protocol consequences
B = pd.read_csv(f'{R}/B_presplit_origtest_results.csv'); B['Run'] = B['run'] + 1
ps = S4b[S4b.Protocol == 'Leak-PreSplit'].rename(columns={'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'})
pc = S4b[S4b.Protocol == 'Leak-PreCV'].rename(columns={'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'})
rows = []
for lab, df in [('Leakage-free', lf2), ('Pre-split', ps), ('Pre-split, original test', B), ('Pre-CV', pc)]:
    fr = {}
    for met in ['f1', 'mcc']:
        k = 0; fav = 0
        for (d, m), gg in df.groupby(['dataset', 'model']):
            p = gg.pivot(index='Run', columns='tuner', values=met)[['No-Tuning'] + ALT]
            if stats.friedmanchisquare(*[p[c] for c in p]).pvalue < 0.05:
                k += 1; fav += int(p.mean().idxmax() == 'No-Tuning')
        fr[met] = f'{k}/12' + (f' ({fav} favour NT)' if fav else '')
    sig = [DSN[d].split()[0] for d in DS if pooled(df, d, 'f1')[3] < 0.05 and pooled(df, d, 'f1')[0] > 0]
    t = df[df.tuner != 'No-Tuning']
    rows.append([lab, fr['f1'], fr['mcc'], ', '.join(sig) if sig else 'none', f'{t.cv_best.mean():.3f}', f'{t.f1.mean():.3f}', f(t.cv_best.mean() - t.f1.mean())])
T['t9'] = rows
json.dump(T, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in T.items():
    print('==', k); [print(r) for r in v]
