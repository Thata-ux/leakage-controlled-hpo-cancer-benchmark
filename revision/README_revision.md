# Revisions (IJIES Paper ID 20266848) — releases v1.1-revision and v1.2.0

This folder holds everything added for the major revision. The original benchmark (S7) is unchanged in the repository root.

| Path | Content |
|---|---|
| `S8_Revision_IJIES.ipynb` | Revision notebook (blocks A–H), runnable on Google Colab |
| `results/` | Raw outputs per block: `*_results.csv` (run-level metrics, runtime, best CV score), `*_predictions.csv` (test-set predictions and scores with original row IDs), `*_params.jsonl` (selected hyperparameters per run and tuner), `split_indices.csv` (original row IDs of train/test per run), `inner_fold_indices.csv` (row IDs of the 5 inner CV validation folds per run; regenerate with `scripts/make_inner_folds.py`), `B_sample_counts.csv` (original/synthetic and class counts with the original row IDs of the training and test sets for every pre-split protocol and run), `preprocessing_summary.csv`, `missing_share_per_feature.csv` |
| `analysis/` | Derived tables: corrected contrasts (Nadeau–Bengio, Holm), equivalence tests, dataset-level estimates, leakage decomposition, robustness analyses, reproduction check |
| `scripts/` | Scripts that turn `results/` into every table, test, interval and figure of the revised paper |
| `supplementary/` | S8 SHAP summary, S9 critical-difference diagram, S10 significance by inference procedure, S11 leakage inflation per configuration, S12 schematic comparison of protocols (Figure 1 of the submitted version) |

Blocks: A = clean rerun of the leakage-free benchmark with full logging; B = pre-split leakage with original-patient test set; C = matched search domain and budget; D = Cervical Cancer without screening variables; E = controlled imbalance on Breast Cancer (5:1, 10:1, 15:1); F = MCC and balanced-accuracy tuning objectives; G = SMOTENC and random oversampling on Lung Cancer; H = TabPFN v2.

Note: the `run` column in `results/` is 0-based (run r in the paper = `run + 1`); seed = 42 + run.

Reproduce the analysis:
```
python scripts/analyze_rev1.py <S7_results_dir> analysis
python scripts/analyze_rev2.py results <S7_results_dir> analysis
python scripts/tables_rev.py results <S7_results_dir> analysis tables_rev.json
python scripts/figs_rev.py results <S7_results_dir> figures
```

## Inner-fold indices
Every tuner used `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` on the training rows of a run, in the order given in `split_indices.csv`. `results/inner_fold_indices.csv` lists the validation rows of each fold:
```
python scripts/make_inner_folds.py results/split_indices.csv results/inner_fold_indices.csv
```

## Second revision (v1.2.0): selection-aware analysis
- `scripts/selection_aware2.py` → `analysis/R2_selection_aware_protocols.csv` and `analysis/R2_selection_aware_per_combination.csv`. This is the confirmatory dataset-level analysis: in each run, the tuner with the highest inner cross-validated weighted F1 (training data only) is used for each classifier. It produces Table 5, column 4 of Table 10 and Fig. 2.
- `scripts/selection_aware.py` → `analysis/R2_selection_aware.csv`. It gives the prespecified-tuner analysis (Holm over 9 contrasts) and the exploratory best-test-mean estimates.
- `scripts/reproduce_all.sh` reruns every analysis script from the released files; `scripts/prepare_inputs.py` collects the original S7 run-level files from `supplementary/`.
- `figures/` holds Figs. 1–4 of the accepted version.
