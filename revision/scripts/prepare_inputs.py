"""Collect the original (S7) run-level files from supplementary/ into the folder layout the revision scripts expect.
Usage (from the repository root):  python revision/scripts/prepare_inputs.py
Creates s7/outputs/ with S4_run_level_results_leakfree.csv, S4b_run_level_results_leaky.csv and ALL_results_summary.csv."""
import os, shutil
os.makedirs('s7/outputs', exist_ok=True)
for src, dst in [('supplementary/S4_run_level_results_leakfree.csv', 'S4_run_level_results_leakfree.csv'),
                 ('supplementary/S4b_run_level_results_leaky.csv', 'S4b_run_level_results_leaky.csv'),
                 ('supplementary/S1_per_tuner_statistics.csv', 'ALL_results_summary.csv')]:
    shutil.copy(src, os.path.join('s7/outputs', dst)); print('copied', src)
