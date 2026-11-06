"""Revised figures at true print size (fonts >= 10 pt at final width), no panel letters."""
import pandas as pd, numpy as np, sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
R, Z, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10, 'legend.fontsize': 10, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 400})
GRID = dict(color='#e4e3df', lw=0.6)
CORR = 0.1 + 0.25
def nb(d):
    d = np.asarray(d, float); m = d.mean(); se = np.sqrt(CORR * d.var(ddof=1)); q = stats.t.ppf(.975, len(d) - 1); return m, m - q * se, m + q * se
DSN = {'Lung Cancer': 'Lung', 'BreastCancer': 'Breast', 'CervicalCancer': 'Cervical'}
MDN = {'XGBoost': 'XGBoost', 'RandomForest': 'Random forest', 'SVM': 'SVM', 'LightGBM': 'LightGBM'}
ALT = ['Grid', 'Random', 'Bayes']
lf = pd.read_csv(f'{Z}/outputs/S4_run_level_results_leakfree.csv')

# ---------------- forest: NB-corrected, per combination + per-dataset pooled (classifier-averaged within run)
rows = []
for d in ['Lung Cancer', 'BreastCancer', 'CervicalCancer']:
    per = []
    for m in ['XGBoost', 'RandomForest', 'SVM', 'LightGBM']:
        P = lf[(lf.Dataset == d) & (lf.Model == m)].pivot(index='Run', columns='Tuner', values='f1')
        b = P[ALT].mean().idxmax(); diff = (P[b] - P['No-Tuning']).values; per.append(diff)
        rows.append((f'{DSN[d]} · {MDN[m]} ({b.lower() if b != "Bayes" else "Bayesian"})', *nb(diff), False))
    rows.append((f'{DSN[d]}: pooled over classifiers', *nb(np.mean(per, axis=0)), True))
fig, ax = plt.subplots(figsize=(5.6, 3.9))
n = len(rows)
for i, (lab, m, lo, hi, pooled) in enumerate(rows):
    y = n - i
    if pooled:
        ax.fill([lo, m, hi, m], [y, y + 0.32, y, y - 0.32], color='#0d366b')
    else:
        c = '#2a78d6' if lo > 0 or hi < 0 else '#8a8984'
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle='round'); ax.plot(m, y, 's', color=c, ms=6, mec='white', mew=0.8)
ax.set_yticks(range(n, 0, -1)); ax.set_yticklabels([r[0] for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    if r[4]: t.set_fontweight('bold')
ax.axvline(0, color='#0b0b0b', lw=0.9, ls='--'); ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06]); ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f')); ax.grid(axis='x', **GRID); ax.set_axisbelow(True)
ax.set_xlabel('Paired weighted-F1 gain vs. no tuning (corrected 95% CI)')
ax.legend(handles=[Line2D([], [], color='#2a78d6', marker='s', lw=2, label='CI excludes zero'), Line2D([], [], color='#8a8984', marker='s', lw=2, label='CI includes zero'),
                   Line2D([], [], color='#0d366b', marker='D', lw=0, ms=7, label='Dataset-level pooled')], loc='upper center', bbox_to_anchor=(0.3, -0.14), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(); fig.savefig(f'{OUT}/image3.png', bbox_inches='tight'); plt.close(fig)

# ---------------- runtime scatter at print size
s = pd.read_csv(f'{Z}/outputs/ALL_results_summary.csv')
TC = {'No-Tuning': '#8a8984', 'Grid': '#2a78d6', 'Random': '#eb6834', 'Bayes': '#1baf7a'}
TN = {'No-Tuning': 'No tuning', 'Grid': 'Grid', 'Random': 'Random', 'Bayes': 'Bayesian'}
MK = {'XGBoost': 'o', 'RandomForest': 's', 'SVM': '^', 'LightGBM': 'D'}
fig, axs = plt.subplots(1, 3, figsize=(6.7, 2.7))
for ax, d in zip(axs, ['Lung Cancer', 'BreastCancer', 'CervicalCancer']):
    a = s[s.Dataset == d]
    for m in MK:
        b = a[a.Model == m].set_index('Tuner')
        for t in TC: ax.scatter(b.loc[t].Mean_RT, b.loc[t].Mean_F1, marker=MK[m], s=28, color=TC[t], edgecolor='white', lw=0.7, zorder=3)
    ax.set_title(DSN[d] + ' Cancer'); ax.set_xlabel('Runtime (s)'); ax.grid(**GRID); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
axs[0].set_ylabel('Weighted F1')
h = [Line2D([], [], marker='o', ls='', color=TC[t], ms=6, label=TN[t]) for t in TC] + [Line2D([], [], marker=MK[m], ls='', color='#52514e', ms=6, label=MDN[m]) for m in MK]
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0, handletextpad=0.3)
fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(f'{OUT}/image5.png'); plt.close(fig)

# ---------------- leakage figure: MCC decomposition (stacked) + optimism gap by protocol
A = pd.read_csv(f'{R}/A_main_logged_results.csv'); B = pd.read_csv(f'{R}/B_presplit_origtest_results.csv')
S4b = pd.read_csv(f'{Z}/outputs/S4b_run_level_results_leaky.csv')
for df in (A, B): df['Run'] = df['run'] + 1
key = ['dataset', 'model', 'tuner', 'Run']
lfk = A.set_index(key); bk = B.set_index(key)
ps = S4b[S4b.Protocol == 'Leak-PreSplit'].rename(columns={'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'}).set_index(key)
pcv = S4b[S4b.Protocol == 'Leak-PreCV'].rename(columns={'Dataset': 'dataset', 'Model': 'model', 'Tuner': 'tuner'}).set_index(key)
DS = ['Lung Cancer', 'BreastCancer', 'CervicalCancer']
fig, axs = plt.subplots(1, 2, figsize=(6.7, 2.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]; xs = np.arange(3); w = 0.34
for j, met in enumerate(['f1', 'mcc']):
    con = [(bk[met] - lfk[met]).dropna().xs(d, level=0).mean() for d in DS]
    syn = [(ps[met] - bk[met]).dropna().xs(d, level=0).mean() for d in DS]
    x = xs + (j - 0.5) * w
    ax.bar(x, con, w - 0.04, color='#2a78d6' if met == 'mcc' else '#86b6ef', label=f'{met.upper() if met=="mcc" else "F1"}: contaminated training')
    ax.bar(x, syn, w - 0.04, bottom=con, color='#eb6834' if met == 'mcc' else '#f2a07f', label=f'{met.upper() if met=="mcc" else "F1"}: synthetic test cases')
ax.set_xticks(xs); ax.set_xticklabels([DSN[d] for d in DS]); ax.axhline(0, color='#0b0b0b', lw=0.8)
ax.set_ylabel('Mean inflation vs. leakage-free'); ax.set_title('Pre-split leakage, decomposed'); ax.grid(axis='y', **GRID); ax.set_axisbelow(True)
LH = ax.get_legend_handles_labels()
ax = axs[1]
PC = [('Leakage-free', lfk, '#8a8984'), ('Pre-split, original test', bk, '#2a78d6'), ('Pre-split', ps, '#4a3aa7'), ('Pre-CV', pcv, '#e87ba4')]
w = 0.2
for j, (nm, df, c) in enumerate(PC):
    g = df.reset_index(); g = g[g.tuner != 'No-Tuning']; g = g.assign(gap=g.cv_best - g.f1)
    mu = [g[g.dataset == d].gap.mean() for d in DS]
    se = [stats.t.ppf(.975, (g.dataset == d).sum() - 1) * g[g.dataset == d].gap.std(ddof=1) / np.sqrt((g.dataset == d).sum()) for d in DS]
    ax.bar(xs + (j - 1.5) * w, mu, w - 0.03, color=c, label=nm, yerr=se, error_kw=dict(ecolor='#52514e', lw=0.8, capsize=2))
ax.axhline(0, color='#0b0b0b', lw=0.8); ax.set_xticks(xs); ax.set_xticklabels([DSN[d] for d in DS])
ax.set_ylabel('CV F1 − test F1'); ax.set_title('Optimism gap of tuned models'); ax.grid(axis='y', **GRID); ax.set_axisbelow(True)
RH = ax.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.0, 1, 1))
fig.legend(*LH, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=1.2, handletextpad=0.4)
fig.legend(*RH, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=1.0, handletextpad=0.4)
fig.savefig(f'{OUT}/image7.png', bbox_inches='tight'); plt.close(fig)
print('ok')
