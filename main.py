"""
Entry point for the pipeline.

Run with:
    python main.py

    load config -> load data -> clean -> drop duplicates -> features/target -> set the locked test set aside
    -> Pipeline(preprocessing + model) -> holdout and cross-validation on the development set
    -> fairness audit -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test, build_preprocessor,
)
from src.model import build_model
from src.evaluate import (
    holdout_report, cross_validate_pipeline, cv_report, oof_classification_report, fairness_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])                              # rules only: keeps every row
    df = drop_duplicate_rows(df, config["diagnostics"]["id_column"])           # training data only, before the split

    X, y, extras = split_features_target(df, config["data"], config["preprocessing"]["mnar_indicator_sources"])

    # the locked test set is set aside here and not used anywhere in this script
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # preprocessing + model in ONE estimator: everything is fitted on training rows only
    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"], shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )

    report = holdout_report(pipeline, X_dev, y_dev, random_state=cv_config["random_state"])

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, cv_config.get("scoring", "accuracy"), cv_config.get("n_jobs", 1)
    )
    report += "\n" + cv_report(fold_scores, cv_config.get("scoring", "accuracy"))
    report += "\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()