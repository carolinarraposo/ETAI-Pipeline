"""Saving results: one full report per run, and one row per run in a scoreboard CSV."""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """Writes one run's full report to `results_dir/run_<run>.txt` and returns the path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv, tuning = config["cv"], config.get("tuning", {})
    tuning_line = (f"Optuna TPE, {tuning['n_trials']} trials, {tuning['n_splits']} inner folds"
                   if tuning.get("enabled") else "off")
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}, scaler={config['preprocessing']['scaler']}\n"
        f"Locked test set: size={config['test_set']['size']}  random_state={config['test_set']['random_state']}\n"
        f"CV: {cv['n_splits']} stratified folds, shuffle={cv['shuffle']}, random_state={cv['random_state']}\n"
        f"Metrics: {cv['metrics']}  |  selection metric: {cv['selection_metric']}\n"
        f"Tuning: {tuning_line}\n"
        + "=" * 60 + "\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)
    return path