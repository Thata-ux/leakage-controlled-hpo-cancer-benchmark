# Does Hyperparameter Tuning Pay Off on Small, Imbalanced Cancer Data?

Code, run-level results and supplementary material for the article:

> T. H. Andika, Widowati, and C. E. Widodo, "Does Hyperparameter Tuning Pay Off on Small, Imbalanced Cancer Data? A Leakage-Controlled Multi-Run Benchmark", *International Journal of Intelligent Engineering and Systems* (accepted, 2026).

**Release v1.2.0 is the version that corresponds to the accepted manuscript.** It is published as an immutable GitHub release. Earlier releases: `v1.0-submission` (first submission) and `v1.1-revision` (first revision).

The study compares **no tuning, grid search, random search and Bayesian optimization** for **XGBoost, random forest, SVM and LightGBM** on three small, imbalanced cancer datasets (Lung Cancer, Breast Cancer Wisconsin, Cervical Cancer). Every configuration is evaluated over **10 paired runs** (seeds 42–51) under **four evaluation protocols**:

| Protocol | What is fitted where | Run in |
|---|---|---|
| **Leakage-free** | imputation, scaling and SMOTE fitted on training folds only | S7 (Section 6), rerun with full logging in S8 block A |
| **Pre-split** | preprocessing and SMOTE fitted on the full data before the 80:20 split; the test set contains synthetic cases | S7 (Section 11) |
| **Pre-split, original test** | as pre-split, but tested only on the original test patients of the leakage-free split | S8 block B |
| **Pre-CV** | correct split, but SMOTE applied to the whole training set before inner cross-validation | S7 (Section 11) |

## Main findings (accepted version)

Inference uses the Nadeau–Bengio corrected resampled t-test with Holm correction. Dataset-level effects average the four classifiers within each run. The **confirmatory** dataset-level analysis is **selection-aware**: in each run, the tuner with the highest inner cross-validated weighted F1 (training data only) is used. Estimates that pick the tuner by its mean test score are reported only as exploratory.

| Question | Result |
|---|---|
| RQ1 – Does tuning improve weighted F1 without leakage? | No detectable gain on Lung and Breast Cancer. Equivalence (TOST, margin 0.01) is mostly inconclusive rather than shown. |
| RQ2 – When does it help? | Cervical Cancer only. Selection-aware gains: MCC +0.124 (Holm-adjusted p = 0.015) and minority recall +0.216 (p = 0.009) are confirmatory. The weighted-F1 gain of +0.011 (p = 0.019; Holm-adjusted p = 0.058) is exploratory. The gains reflect an operating-point shift, vanish without screening variables, and are not reproduced by raising imbalance within Breast Cancer (5:1–15:1). |
| RQ3 – Is it worth the cost? | The best tuner added 2.3–69.8 s per search (median 25.1 s) on two vCPUs. |
| RQ4 – What does leakage do? | Pre-split SMOTE inflated MCC by 0.185, of which 0.068 came from contaminated training and 0.117 from synthetic test cases. Pre-CV SMOTE raised cross-validated F1 by 0.025 (0.969 vs. 0.944) and opened a 0.029 gap between cross-validated and test F1. |

## Repository structure

```
├── README.md
├── requirements.txt / requirements-colab-full.txt / env_versions.txt
├── notebook/S7_Paper_1_v4_NOVELTY.ipynb      original benchmark: leakage-free, pre-split and pre-CV protocols
├── supplementary/                            S1–S6 and S8 (original analysis; S4/S4b = run-level results)
├── results/                                  intermediate statistics of the original analysis
├── figures/                                  figures of the first submission
└── revision/                                 everything added in the revisions (see revision/README_revision.md)
    ├── S8_Revision_IJIES.ipynb               revision experiments, blocks A–H
    ├── results/                              run-level results, predictions, hyperparameters, split and inner-fold indices, sample counts
    ├── analysis/                             derived tables, including the selection-aware analysis (R2_*.csv)
    ├── scripts/                              scripts that produce every table, test, interval and figure of the accepted version
    ├── figures/                              Figs. 1–4 of the accepted version
    └── supplementary/                        S9–S12
```

Revision blocks (notebook `revision/S8_Revision_IJIES.ipynb`):

| Block | Content |
|---|---|
| A | clean rerun of the leakage-free benchmark with full logging (464 of 480 results identical to S7) |
| B | pre-split protocol with an original-patient test set |
| C | matched search domain and budget (24 candidates per tuner, default included) |
| D | Cervical Cancer without the Hinselmann, Schiller and cytology variables |
| E | controlled imbalance on Breast Cancer (5:1, 10:1, 15:1) |
| F | MCC and balanced accuracy as tuning objectives |
| G | SMOTENC and random oversampling on Lung Cancer |
| H | TabPFN v2 without tuning |

## Data

The datasets are public and are not redistributed here. Download them and place the CSV files in `data/`:

| Dataset | Source | Target column |
|---|---|---|
| Lung Cancer (309 rows; 276 after removing duplicates) | Kaggle, M. A. Bhat: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer | `LUNG_CANCER` |
| Breast Cancer Wisconsin (Diagnostic) (569 rows) | UCI ML Repository, doi: 10.24432/C5DW2B | `diagnosis` |
| Cervical Cancer (Risk Factors) (858 rows; 835 after removing duplicates) | UCI ML Repository, doi: 10.24432/C5Z310 | `Biopsy` |

Original row identifiers after duplicate removal, the split indices of every run and the inner-fold indices are in `revision/results/`.

## How to reproduce

**A. Reproduce every table, test, interval and figure from the released run-level results** (minutes, no datasets needed):

```bash
pip install -r requirements.txt
bash revision/scripts/reproduce_all.sh
```

The outputs are written to `out/` and can be compared with `revision/analysis/` and `revision/figures/`.

| Manuscript element | Produced by |
|---|---|
| Tables 4, 6, 8, 9, 10 | `tables_rev.py` |
| Table 5 and Table 10 (Cervical selection-aware column), Fig. 2 | `selection_aware2.py`, `fig_forest_r2.py` |
| Prespecified-tuner and exploratory estimates (Section 4.1) | `selection_aware.py` |
| Table 7 and Sections 4.3–4.4 (blocks C–H) | `analyze_rev2.py` |
| Corrected contrasts, Holm, equivalence tests, runtime intervals, S10 | `analyze_rev1.py` |
| Figs. 3 and 4 | `figs_rev.py` |

**B. Rerun the experiments** (hours, needs the datasets):

1. Run `notebook/S7_Paper_1_v4_NOVELTY.ipynb` in Google Colab to obtain the original leakage-free, pre-split and pre-CV results (about 8.5 h on 2 vCPUs). Both notebooks save checkpoints after every run.
2. Run `revision/S8_Revision_IJIES.ipynb` for blocks A–H. The notebook writes the files in `revision/results/`.
3. Run step A above on the new results.

Results can shift slightly between library versions. Use the versions in `requirements.txt` to reproduce the reported numbers exactly.

## Citation

If you use this code or these results, please cite the article above. The citation will be updated with volume, issue and DOI after publication.

## Contact

Tahta Herdian Andika, Doctoral Program of Information Systems, Universitas Diponegoro, Indonesia — tahtaherdianandika@students.undip.ac.id
