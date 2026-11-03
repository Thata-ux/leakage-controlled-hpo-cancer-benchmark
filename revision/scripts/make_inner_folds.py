"""Regenerate the inner 5-fold CV indices used by every tuner (leakage-free protocol).
Folds are StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) over the training rows of each run,
in the order listed in results/split_indices.csv (the order produced by train_test_split)."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
sp = pd.read_csv(sys.argv[1]); tr = sp[sp.part == 'train']
rows = []
for (ds, run, seed), g in tr.groupby(['dataset', 'run', 'seed'], sort=False):
    y, ids = g.y.values, g.row_id.values
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=int(seed)).split(np.zeros(len(y)), y), 1):
        rows += [{'dataset': ds, 'run': run, 'seed': seed, 'fold': k, 'row_id': int(r)} for r in ids[va]]
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
