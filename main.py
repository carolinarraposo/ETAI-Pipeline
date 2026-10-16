"""
Entry point for the pipeline.

Run with:
    python main.py

    load config -> load data -> clean -> drop duplicates -> features/target -> set the locked test set aside
    -> Pipeline(preprocessing + model) -> holdout and cross-validation on the development set
    -> (if tuning.enabled) nested CV + tuning on the development set -> fairness audit -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, add_engineered_features, split_features_target, split_dev_test
from src.model import build_pipeline
from src.evaluate import holdout_report, cross_validate_pipeline, cv_report, oof_classification_report, fairness_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])                              # rules only: keeps every row
    df = drop_duplicate_rows(df, config["diagnostics"]["id_column"])           # training data only, before the split
    df = add_engineered_features(df)                                           # fixed formulas: safe before the split

    X, y, extras = split_features_target(df, config["data"], config["preprocessing"]["mnar_indicator_sources"])

    # the locked test set is set aside here and not used anywhere in this script
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipeline = build_pipeline(config["preprocessing"], config["model"])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"], shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    report = holdout_report(pipeline, X_dev, y_dev, random_state=cv_config["random_state"])

    # default hyperparameters, 5-fold CV
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs)
    report += "\n" + cv_report(fold_scores, scoring)

    # hyperparameter tuning: development set only, every candidate scored by CV of the whole pipeline
    tuning_config = config.get("tuning", {})
    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(
                f"tuning.enabled is true but tuning.search_spaces has no entry for '{model_type}'. "
                f"Add one, or set tuning.enabled: false. Defined: {list(search_spaces)}"
            )
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # 1. the honest estimate of "tune, then refit" (its predictions replace the untuned ones)
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\nNested cross-validation (tuned pipeline)\n" + cv_report(nested_scores, scoring)

        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n" + tuning_report(study, nested_scores, scoring)

    report += "\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    pipeline.fit(X_dev, y_dev)     # the final model: refit on ALL development rows (tuned, if tuning ran)

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()