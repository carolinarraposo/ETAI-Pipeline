"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)


def flag_invalid_values(df, rules):
    """Turns values outside each column's min/max (from config) into NaN, in place.
    An impossible value (age -3) is as good as missing, but .isna() would never catch it."""
    for column, bounds in rules.items():
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = pd.Series(False, index=df.index)

        if "min" in bounds:
            invalid = invalid | (numeric < bounds["min"])

        if "max" in bounds:
            invalid = invalid | (numeric > bounds["max"])

        df.loc[invalid, column] = np.nan

    return df

def clean_dataset(df, diagnostics_config):
    """Rule-based cleaning driven by config: placeholders/invalid values -> NaN, category
    spellings fixed, redundant columns dropped. Row-preserving: it also runs on new data,
    where every row needs a prediction."""
    out = df.copy()
    placeholder_tokens = diagnostics_config["placeholder_tokens"]

    for col in diagnostics_config["numeric_text_columns"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(placeholder_tokens, np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config["validity_rules"])

    for col, mapping in diagnostics_config["canonical_categories"].items():
        if col not in out.columns:
            continue
        cleaned = out[col].str.strip() # no .astype(str): it would turn a real NaN into the text "nan"
        mapped = cleaned.str.lower().map(mapping).fillna(cleaned)
        out[col] = mapped.mask(mapped.isin(placeholder_tokens))  # leftover placeholders ("?") -> NaN

    redundant = [c for c in diagnostics_config["redundant_columns"] if c in out.columns]
    return out.drop(columns=redundant)


def drop_duplicate_rows(df, id_column=None):
    """TRAINING DATA ONLY. Drops exact duplicate rows and repeated ids, before the dev/test split,
    so the same person can't land in both the training and the test set."""
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
        return out

def add_missingness_indicators(df, mnar_indicator_sources):
    """Adds a <col>_was_missing flag per MNAR column, BEFORE imputation, so the pattern survives the fill."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df, data_config, mnar_indicator_sources):
    """Returns (X, y, extras). race and score_text go to `extras` (fairness audit), never into X."""
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    X = df[[c for c in df.columns if c not in always_drop]]
    return X, y, extras


def split_dev_test(X, y, extras, test_size, random_state):
    """Sets the locked test set aside. Everything we learn from or compare on is the development set."""
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}


def build_preprocessor(preprocessing_config):
    """Builds the (unfitted) ColumnTransformer from config. Fitting happens later, on training rows only,
    because this goes inside the model's Pipeline."""
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])