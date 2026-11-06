import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
S, F, OUT = sys.argv[1:4]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 400})
pa = pd.read_csv(S); pc = pd.read_csv(F)
DSN = {'Lung Cancer': 'Lung', 'BreastCancer': 'Breast', 'CervicalCancer': 'Cervical'}
MDN = {'XGBoost': 'XGBoost', 'RandomForest': 'Random forest', 'SVM': 'SVM', 'LightGBM': 'LightGBM'}
rows = []
for d in DSN:
    for _, r in pc[pc.dataset == d].iterrows(): rows.append((f'{DSN[d]} · {MDN[r.model]}', r['mean'], r.lo, r.hi, False))
    x = pa[(pa.protocol == 'Leakage-free') & (pa.dataset == d) & (pa.metric == 'f1')].iloc[0]
    rows.append((f'{DSN[d]}: dataset level', x['mean'], x.lo, x.hi, True))
fig, ax = plt.subplots(figsize=(5.6, 3.9)); n = len(rows)
for i, (lab, m, lo, hi, pooled) in enumerate(rows):
    y = n - i
    if pooled: ax.fill([lo, m, hi, m], [y, y + 0.32, y, y - 0.32], color='#0d366b')
    else:
        c = '#2a78d6' if lo > 0 or hi < 0 else '#8a8984'
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle='round'); ax.plot(m, y, 's', color=c, ms=6, mec='white', mew=0.8)
ax.set_yticks(range(n, 0, -1)); ax.set_yticklabels([r[0] for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    if r[4]: t.set_fontweight('bold')
ax.axvline(0, color='#0b0b0b', lw=0.9, ls='--'); ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06])
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
ax.grid(axis='x', color='#e4e3df', lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel('Selection-aware weighted-F1 gain vs. no tuning (corrected 95% CI)')
ax.legend(handles=[Line2D([], [], color='#2a78d6', marker='s', lw=2, label='CI excludes zero'), Line2D([], [], color='#8a8984', marker='s', lw=2, label='CI includes zero'),
                   Line2D([], [], color='#0d366b', marker='D', lw=0, ms=7, label='Dataset level')], loc='upper center', bbox_to_anchor=(0.3, -0.14), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(); fig.savefig(OUT, bbox_inches='tight'); print('ok')
