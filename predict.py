"""
Predict a CSV with the model saved by main.py -- no training.

    python predict.py data/compas_new_people.csv predictions.csv
    python predict.py data/adult_new.csv adult_predictions.csv --config config_adult.yaml
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, add_engineered_features, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """The id, the probability of the positive class and the 0/1 prediction (at 0.5) for EVERY row of df."""
    clean = clean_dataset(df, config["diagnostics"])      # no drop_duplicate_rows: every row gets an answer
    clean = add_engineered_features(clean)
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    probability = pipeline.predict_proba(X)[:, 1]
    id_column = config["diagnostics"]["id_column"]
    return pd.DataFrame({id_column: df[id_column].to_numpy(),
                         "probability": probability,
                         "prediction": (probability >= 0.5).astype(int)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="config file the model was trained with")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config["output"]["model_path"])
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")