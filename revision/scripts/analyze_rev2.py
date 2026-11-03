"""Analyse the S8 revision blocks A–H against the S7 results."""
import pandas as pd, numpy as np, sys, os, json
from scipy import stats
R, Z, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
J = 10; CORR = 1 / J + 0.25
DS = ['Lung Cancer', 'BreastCancer', 'CervicalCancer']; MD = ['XGBoost', 'RandomForest', 'SVM', 'LightGBM']; ALT = ['Grid', 'Random', 'Bayes']
def nb(d):
    d = np.asarray(d, float); m = d.mean(); s2 = d.var(ddof=1); se = np.sqrt(CORR * s2) if s2 > 0 else 0
    tq = stats.t.ppf(.975, len(d) - 1); p = 2 * stats.t.sf(abs(m / se), len(d) - 1) if se > 0 else (0.0 if m else 1.0)
    return m, m - tq * se, m + tq * se, p
def fmt(m, lo, hi, p, k=3): return f"{m:+.{k}f} [{lo:+.{k}f}, {hi:+.{k}f}] p={p:.3f}"
L = lambda f: pd.read_csv(os.path.join(R, f))
A = L('A_main_logged_results.csv'); B = L('B_presplit_origtest_results.csv'); C = L('C_matched_space_results.csv')
D = L('D_cervical_noscreen_results.csv'); E = L('E_controlled_ir_breast_results.csv'); F = L('F_objective_cervical_results.csv')
G = L('G_sampler_lung_results.csv'); H = L('H_tabpfn_results.csv'); CNT = L('B_sample_counts.csv')
S4 = pd.read_csv(f'{Z}/outputs/S4_run_level_results_leakfree.csv'); S4b = pd.read_csv(f'{Z}/outputs/S4b_run_level_results_leaky.csv')
for df in (A, B, C, D, E, F, G, H): df['Run'] = df['run'] + 1
out = {}

# ---------- A: reproduction
m = S4.merge(A, left_on=['Dataset', 'Model', 'Tuner', 'Run'], right_on=['dataset', 'model', 'tuner', 'Run'], suffixes=('_s7', '_s8'))
d = (m['f1_s7'] - m['f1_s8']).abs()
print(f"A reproduction: {len(m)} rows, identical {int((d < 1e-9).sum())}, max |dF1| {d.max():.6f}")
print(m.assign(diff=d)[d > 1e-9].groupby(['Dataset', 'Model', 'Tuner']).size())
m[['Dataset', 'Model', 'Tuner', 'Run', 'f1_s7', 'f1_s8']].to_csv(f'{OUT}/A_reproduction_check.csv', index=False)

def piv(df, met, extra=None, keys=('dataset', 'model')):
    return {k: g.pivot(index='Run', columns='tuner', values=met) for k, g in df.groupby(list(keys))}

def tuning_table(df, label, keys=('dataset', 'model'), mets=('f1', 'mcc', 'recall_min')):
    rows = []
    for met in mets:
        P = piv(df, met, keys=keys)
        for k, p in P.items():
            nt = p['No-Tuning']; best = p[ALT].mean().idxmax()
            fr = stats.friedmanchisquare(*[p[t] for t in ['No-Tuning'] + ALT]).pvalue
            mm, lo, hi, pp = nb(p[best] - nt)
            rows.append(dict(set=label, key=' | '.join(map(str, k if isinstance(k, tuple) else (k,))), metric=met, nt=nt.mean(), best=best, best_mean=p[best].mean(), gain=mm, lo=lo, hi=hi, p_nb=pp, p_friedman=fr))
    return pd.DataFrame(rows)

def pooled(df, label, group='dataset', mets=('f1', 'mcc', 'recall_min')):
    rows = []
    for met in mets:
        for gname, g in df.groupby(group):
            per = []
            for mdl, gg in g.groupby('model'):
                p = gg.pivot(index='Run', columns='tuner', values=met); b = p[ALT].mean().idxmax(); per.append((p[b] - p['No-Tuning']).values)
            mm, lo, hi, pp = nb(np.mean(per, axis=0))
            rows.append(dict(set=label, group=gname, metric=met, gain=mm, lo=lo, hi=hi, p_nb=pp, n_models=len(per)))
    return pd.DataFrame(rows)

base_tbl = tuning_table(A, 'leak-free (A)'); base_pool = pooled(A, 'leak-free (A)')

# ---------- B: pre-split decomposition
cnt = CNT.groupby(['dataset', 'protocol', 'part'])[['n_original', 'n_synthetic', 'n_class0', 'n_class1']].mean().round(1)
print(cnt); cnt.to_csv(f'{OUT}/B_counts_mean.csv')
key = ['dataset', 'model', 'tuner', 'Run']
lfk = A.set_index(key); bk = B.set_index(key)
s4b = S4b[S4b.Protocol == 'Leak-PreSplit'].rename(columns={'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'}).set_index(key)
dec = []
for met in ['f1', 'auc', 'mcc', 'recall_min']:
    tot = (s4b[met] - lfk[met]).dropna(); contam = (bk[met] - lfk[met]).dropna(); synth = (s4b[met] - bk[met]).dropna()
    for ds in DS:
        dec.append(dict(metric=met, dataset=ds, total=tot.xs(ds, level=0).mean(), contaminated_training=contam.xs(ds, level=0).mean(), synthetic_test=synth.xs(ds, level=0).mean()))
    dec.append(dict(metric=met, dataset='All', total=tot.mean(), contaminated_training=contam.mean(), synthetic_test=synth.mean()))
DEC = pd.DataFrame(dec); DEC.round(4).to_csv(f'{OUT}/B_decomposition.csv', index=False); print(DEC.round(4).to_string(index=False))
# significance of contaminated-training effect per config (Wilcoxon)
sig = []
for met in ['f1', 'mcc']:
    c = (bk[met] - lfk[met]).dropna()
    for k, g in c.groupby(level=[0, 1, 2]):
        p = stats.wilcoxon(g.values).pvalue if np.any(g.values != 0) else 1
        sig.append(dict(metric=met, cfg=k, mean=g.mean(), p=p))
SG = pd.DataFrame(sig); print('B contaminated effect: sig+ / sig- by metric'); print(SG.assign(pos=(SG.p < .05) & (SG['mean'] > 0), neg=(SG.p < .05) & (SG['mean'] < 0)).groupby('metric')[['pos', 'neg']].sum())
B_tbl = tuning_table(B, 'pre-split, original test (B)'); B_pool = pooled(B, 'pre-split, original test (B)')
# optimism gap in B
for nm, df in [('A', A), ('B', B), ('C', C)]:
    g = df[df.tuner != 'No-Tuning']; print(nm, 'optimism gap', (g.cv_best - g.f1).mean().round(4))

# ---------- C: matched domain/budget
C_tbl = tuning_table(C, 'matched domain (C)'); C_pool = pooled(C, 'matched domain (C)')
print('C default_evaluated rate by tuner:'); print(C.groupby(['model', 'tuner']).default_evaluated.mean().unstack().round(2))
rk = []
for k, g in C.groupby(['dataset', 'model']):
    rk.append(g.groupby('tuner').f1.mean())
M = pd.DataFrame(rk)[['No-Tuning'] + ALT]; RB = (-M).rank(axis=1).mean(); print('C mean ranks (12 blocks):', RB.round(2).to_dict())
chi = stats.friedmanchisquare(*[M[c] for c in M]).statistic; N, k = 12, 4; FF = (N - 1) * chi / (N * (k - 1) - chi); print('C Iman-Davenport F', round(FF, 3), 'p', round(1 - stats.f.cdf(FF, 3, 33), 3))
print('C runtime by tuner (s, mean):', C.groupby('tuner').runtime.mean().round(1).to_dict(), ' A:', A.groupby('tuner').runtime.mean().round(1).to_dict())

# ---------- D: cervical no screening
D_tbl = tuning_table(D, 'Cervical without screening (D)', mets=('f1', 'mcc', 'recall_min', 'bal_acc'))
Acv = A[A.dataset == 'CervicalCancer']
print('Cervical with vs without screening (mean over models, NT / best tuned):')
for met in ['f1', 'auc', 'mcc', 'recall_min', 'sens', 'spec', 'sens_at_spec90']:
    print(f"  {met:15s} with: NT {Acv[Acv.tuner=='No-Tuning'][met].mean():.3f}  tuned {Acv[Acv.tuner!='No-Tuning'][met].mean():.3f} | without: NT {D[D.tuner=='No-Tuning'][met].mean():.3f}  tuned {D[D.tuner!='No-Tuning'][met].mean():.3f}")
D_pool = pooled(D, 'Cervical without screening (D)', mets=('f1', 'mcc', 'recall_min', 'bal_acc'))

# ---------- E: controlled IR
Ab = A[(A.dataset == 'BreastCancer') & (A.model.isin(['XGBoost', 'SVM']))].assign(ir=1.68)
EE = pd.concat([Ab, E], ignore_index=True)
E_pool = pooled(EE, 'controlled IR (E)', group='ir', mets=('f1', 'mcc', 'recall_min', 'bal_acc'))
E_tbl = tuning_table(EE, 'controlled IR (E)', keys=('ir', 'model'), mets=('f1', 'mcc', 'recall_min'))
print(E_pool.round(4).to_string(index=False))
print(E_tbl[['key', 'metric', 'nt', 'gain', 'lo', 'hi', 'p_nb', 'p_friedman']].round(4).to_string(index=False))
print('E minority test counts:', E.groupby('ir').n_test_minority.mean().to_dict())

# ---------- F: objective
Fwf = Acv[Acv.tuner.isin(['Grid', 'Random'])].assign(objective='weighted_f1')
FF_ = pd.concat([Fwf, F], ignore_index=True)
fo = FF_.groupby(['objective'])[['f1', 'mcc', 'recall_min', 'bal_acc', 'sens', 'spec', 'ppv', 'sens_at_spec90', 'auc']].mean().round(3); print(fo)
fo.to_csv(f'{OUT}/F_objective_means.csv')
fcmp = []
for obj in ['mcc', 'bal_acc']:
    for met in ['mcc', 'recall_min', 'spec', 'ppv', 'f1', 'sens_at_spec90']:
        a = Fwf.set_index(['model', 'tuner', 'Run'])[met]; b = F[F.objective == obj].set_index(['model', 'tuner', 'Run'])[met]
        dd = (b - a).dropna(); per = dd.groupby(level='Run').mean()   # average over 8 model×tuner configs per run
        mm, lo, hi, pp = nb(per.values); fcmp.append(dict(objective=obj, metric=met, diff_vs_weightedF1=mm, lo=lo, hi=hi, p_nb=pp))
FC = pd.DataFrame(fcmp); FC.round(4).to_csv(f'{OUT}/F_objective_vs_wf1.csv', index=False); print(FC.round(4).to_string(index=False))

# ---------- G: samplers on Lung
Al = A[A.dataset == 'Lung Cancer'].assign(sampler='smote')
GG = pd.concat([Al, G], ignore_index=True)
print(GG.groupby(['sampler', 'tuner'])[['f1', 'mcc', 'recall_min']].mean().round(4).unstack(0))
G_tbl = tuning_table(GG, 'Lung samplers (G)', keys=('sampler', 'model'), mets=('f1', 'mcc', 'recall_min'))
G_pool = pooled(GG, 'Lung samplers (G)', group='sampler', mets=('f1', 'mcc', 'recall_min'))
print(G_pool.round(4).to_string(index=False))
gs = []
for smp in ['smotenc', 'ros']:
    for met in ['f1', 'mcc', 'recall_min']:
        a = Al.set_index(['model', 'tuner', 'Run'])[met]; b = G[G.sampler == smp].set_index(['model', 'tuner', 'Run'])[met]
        per = (b - a).dropna().groupby(level='Run').mean(); mm, lo, hi, pp = nb(per.values)
        gs.append(dict(sampler=smp, metric=met, diff_vs_smote=mm, lo=lo, hi=hi, p_nb=pp))
print(pd.DataFrame(gs).round(4).to_string(index=False))

# ---------- H: TabPFN
ht = []
for ds in DS:
    a = A[A.dataset == ds]
    for var in ['no_resampling', 'smote']:
        h = H[(H.dataset == ds) & (H.variant == var)].set_index('Run')
        for met in ['f1', 'mcc', 'recall_min', 'auc']:
            nt = a[a.tuner == 'No-Tuning'].groupby('model')[met].mean(); best_nt_model = nt.idxmax()
            ref_nt = a[(a.tuner == 'No-Tuning') & (a.model == best_nt_model)].set_index('Run')[met]
            tb = a[a.tuner != 'No-Tuning'].groupby(['model', 'tuner'])[met].mean().idxmax()
            ref_t = a[(a.model == tb[0]) & (a.tuner == tb[1])].set_index('Run')[met]
            m1 = nb(h[met] - ref_nt); m2 = nb(h[met] - ref_t)
            ht.append(dict(dataset=ds, variant=var, metric=met, tabpfn=h[met].mean(), best_default=f'{best_nt_model}', best_default_val=ref_nt.mean(), vs_best_default=fmt(*m1),
                           best_tuned=f'{tb[0]}/{tb[1]}', best_tuned_val=ref_t.mean(), vs_best_tuned=fmt(*m2), runtime=h.runtime.mean()))
HT = pd.DataFrame(ht); HT.to_csv(f'{OUT}/H_tabpfn_comparison.csv', index=False)
print(HT[HT.metric.isin(['f1', 'mcc', 'recall_min'])][['dataset', 'variant', 'metric', 'tabpfn', 'best_default_val', 'vs_best_default', 'best_tuned_val', 'vs_best_tuned']].round(4).to_string(index=False))

# ---------- sens/spec per dataset (A) for R2.2
print(A.groupby(['dataset', 'tuner'])[['sens', 'spec', 'ppv', 'sens_at_spec90']].mean().round(3))
A.groupby(['dataset', 'model', 'tuner'])[['sens', 'spec', 'ppv', 'sens_at_spec90', 'recall_min', 'mcc']].mean().round(4).to_csv(f'{OUT}/A_sens_spec.csv')

allt = pd.concat([base_tbl, B_tbl, C_tbl, D_tbl, E_tbl, G_tbl]); allt.round(5).to_csv(f'{OUT}/tuning_tables_all.csv', index=False)
allp = pd.concat([base_pool, B_pool, C_pool, D_pool, E_pool, G_pool]); allp.round(5).to_csv(f'{OUT}/pooled_all.csv', index=False)
print('\nPOOLED (NB-corrected, per dataset/group):'); print(allp[allp.metric.isin(['f1', 'mcc', 'recall_min'])].round(4).to_string(index=False))
cnts = allt.assign(sig_f=allt.p_friedman < .05, sig_nb=allt.p_nb < .05).groupby(['set', 'metric'])[['sig_f', 'sig_nb']].sum()
print(cnts)
