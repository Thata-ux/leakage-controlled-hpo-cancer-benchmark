# Does Hyperparameter Tuning Pay Off on Small, Imbalanced Cancer Data?

Code, run-level results and supplementary material for the article:

> T. H. Andika, Widowati, and C. E. Widodo, "Does Hyperparameter Tuning Pay Off on Small, Imbalanced Cancer Data? A Leakage-Controlled Multi-Run Benchmark and Decision Framework", *International Journal of Intelligent Engineering and Systems* (submitted, 2026).

The study compares **no tuning, grid search, random search and Bayesian optimization** for **XGBoost, random forest, SVM and LightGBM** on three small, imbalanced cancer datasets. Every configuration is evaluated over **10 paired runs** under a **leakage-free pipeline** and under **two common leaky pipelines**, giving 1,440 run-level results.

## Main findings

| Question | Result |
|---|---|
| RQ1 – Does tuning improve weighted F1 without leakage? | Not significantly in 9 of 12 dataset-model combinations; the untuned model was best in 4. |
| RQ2 – When does it help? | Only under severe imbalance (Cervical Cancer, 14.46:1): weighted F1 +0.011–0.016, but minority recall +0.17–0.28 and MCC +0.10–0.17. |
| RQ3 – Is it worth the cost? | The best tuner cost a median of about 140× the baseline runtime for a median F1 gain of 0.003. |
| RQ4 – What does leakage do? | SMOTE before splitting inflated weighted F1, AUC and MCC by 0.022, 0.040 and 0.186 and masked tuning effects (3 → 1 significant). SMOTE before cross-validation raised the CV–test optimism gap from 0.000 to 0.029 and made tuning appear harmful in two combinations. |

## Repository structure

```
├── README.md
├── requirements.txt               core packages with the exact versions used
├── requirements-colab-full.txt    full `pip freeze` of the Colab runtime
├── env_versions.txt               Python, CPU, RAM and library versions recorded at run time
├── notebook/
│   └── S7_Paper_1_v4_NOVELTY.ipynb   full pipeline, executed (Supplementary S7)
├── supplementary/
│   ├── S1_per_tuner_statistics.csv           mean ± SD of all metrics and runtime, 48 configurations
│   ├── S2_confidence_intervals_95.csv        95% t-intervals of weighted F1, 48 configurations
│   ├── S3_wilcoxon_bonferroni_effect_sizes.csv   48 paired contrasts with Bonferroni and effect size r
│   ├── S4_run_level_results_leakfree.csv     480 run-level results, leakage-free protocol
│   ├── S4b_run_level_results_leaky.csv       960 run-level results, pre-split and pre-CV leaky protocols
│   ├── S5_mean_ranks_per_combination.csv     mean tuner ranks per dataset-model combination
│   ├── S6_cost_benefit.csv                   F1 gain, runtime multiplier and F1 per second
│   └── S8_shap/                              SHAP and built-in importance plots (9 files)
├── results/                       intermediate statistics used in Section 4
│   ├── friedman_results.csv               Friedman test per combination (weighted F1)
│   ├── secondary_metrics_friedman.csv     Friedman tests for macro-F1, balanced accuracy, MCC, minority recall
│   ├── leakage_inflation.csv              paired inflation per protocol, metric and configuration
│   ├── leakage_tuning_effect.csv          significant tuning effects within each protocol
│   ├── leakage_optimism_gap.csv           best CV F1 minus test F1 per configuration
│   └── FINAL_best_per_dataset_model.csv   best tuner per dataset-model combination
├── figures/                       Figures 1–7 of the article
└── data/                          place the three CSV files here (not redistributed)
```

### Column guide for the run-level files (S4, S4b)

| Column | Meaning |
|---|---|
| `Protocol` | `Leak-Free`, `Leak-PreSplit` (preprocessing + SMOTE before the split) or `Leak-PreCV` (SMOTE on the whole training set before inner CV) |
| `Dataset`, `Model`, `Tuner` | configuration; tuners are `No-Tuning`, `Grid`, `Random`, `Bayes` |
| `Run`, `Seed` | run 1–10 with seed 42–51; the same seed gives the same split for every model, tuner and protocol |
| `runtime` | seconds for fitting (no tuning) or searching and refitting (tuners) |
| `cv_best` | best inner 5-fold cross-validated weighted F1 (empty for no tuning) |
| `f1`, `auc`, `precision`, `recall`, `accuracy` | weighted test-set metrics |
| `f1_macro`, `bal_acc`, `mcc`, `recall_min` | minority-sensitive test-set metrics |

## Data

The datasets are public and are not redistributed here. Download them and place the CSV files in `data/` (or update the paths in Section 1.1 of the notebook):

| Dataset | Source | Target column |
|---|---|---|
| Lung Cancer (276 × 16) | Kaggle, M. A. Bhat: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer | `LUNG_CANCER` |
| Breast Cancer Wisconsin (Diagnostic) (569 × 32) | UCI ML Repository, doi: 10.24432/C5DW2B | `diagnosis` |
| Cervical Cancer (Risk Factors) (835 × 36) | UCI ML Repository, doi: 10.24432/C5Z310 | `Biopsy` |

## How to reproduce

1. Open `notebook/S7_Paper_1_v4_NOVELTY.ipynb` in Google Colab (or Jupyter) and install the packages in `requirements.txt`.
2. Set the dataset paths in Section 1.1.
3. Run all cells. Section 6 (leakage-free benchmark) and Section 11 (leaky protocols) save a checkpoint after every run to `MyDrive/S7_checkpoints`; if the session disconnects, rerun Sections 0–5 and then 6.0 onward, and completed runs are skipped.
4. The full run took about 8.5 hours (about 2.8 h for Section 6 and 5.4 h for Section 11) on a Colab machine with 2 vCPUs and 12.7 GB RAM.

Results can shift slightly between library versions, because default hyperparameters and missing-value handling change between releases. Use the versions in `requirements.txt` to reproduce the reported numbers exactly.

## Citation

If you use this code or these results, please cite the article above. The citation will be updated with volume, issue and DOI after publication.

## Contact

Tahta Herdian Andika, Doctoral Program of Information Systems, Universitas Diponegoro, Indonesia — tahtaherdianandika@students.undip.ac.id
