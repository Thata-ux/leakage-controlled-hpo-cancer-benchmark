#!/usr/bin/env bash
# Reproduce every revised table, test, interval and figure from the released run-level results.
# Run from the repository root:  bash revision/scripts/reproduce_all.sh
set -e
python revision/scripts/prepare_inputs.py
mkdir -p out/analysis out/analysis2 out/figures
python revision/scripts/analyze_rev1.py s7 out/analysis                      # corrected contrasts, Holm, equivalence, runtime CIs (S10)
python revision/scripts/analyze_rev2.py revision/results s7 out/analysis2    # blocks A-H, leakage decomposition, reproduction check
python revision/scripts/tables_rev.py revision/results s7 out/analysis out/tables_rev.json   # Tables 4, 6, 8, 9, 10
python revision/scripts/selection_aware.py s7 revision/results out/analysis/R2_selection_aware.csv   # prespecified-tuner and exploratory estimates
python revision/scripts/selection_aware2.py s7 revision/results out/analysis   # confirmatory selection-aware analysis (Table 5, Table 10 column 4)
python revision/scripts/fig_forest_r2.py out/analysis/R2_selection_aware_protocols.csv out/analysis/R2_selection_aware_per_combination.csv out/figures/Fig2_selection_aware_forest.png
python revision/scripts/figs_rev.py revision/results s7 out/figures           # Figs. 3 and 4
python revision/scripts/make_inner_folds.py revision/results/split_indices.csv out/inner_fold_indices.csv
echo "Done: compare out/ with revision/analysis and revision/figures"
