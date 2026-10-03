"""Evaluation: a single holdout split, and stratified k-fold cross-validation of the whole pipeline."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate, train_test_split


def holdout_report(pipeline, X, y, val_size=0.25, random_state=42) -> str:
    """One train/validation split of the development set -- what the pipeline did up to week 3."""
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)
    pipeline.fit(X_tr, y_tr)
    train_acc = accuracy_score(y_tr, pipeline.predict(X_tr))
    val_acc = accuracy_score(y_va, pipeline.predict(X_va))
    text = (
        f"Holdout (one {1 - val_size:.0%}/{val_size:.0%} split of the development set)\n"
        f"Train accuracy:      {train_acc:.3f}\n"
        f"Validation accuracy: {val_acc:.3f}\n"
        f"Gap (train - val):   {train_acc - val_acc:+.3f}\n"
    )
    print(text)
    return text


def cross_validate_pipeline(pipeline, X, y, cv, scoring="accuracy", n_jobs=1):
    """
    Fits a fresh copy of the WHOLE pipeline (preprocessing + model) on each fold's training part and scores it
    on that fold's validation part, so imputation, encoding and scaling are re-learned inside every fold.

    Returns (fold_scores, y_oof): the per-fold train/validation/gap table, and out-of-fold predictions --
    each row predicted by the one fold model that did NOT train on it.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])
    return fold_scores, y_oof


def cv_report(fold_scores, scoring="accuracy") -> str:
    """Per-fold table plus mean and std, as text (printed and saved to results/)."""
    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines)
    print(text)
    return text


def oof_classification_report(y_true, y_pred) -> str:
    text = "Classification report (out-of-fold predictions, development set):\n" + \
        classification_report(y_true, y_pred, zero_division=0)
    print(text)
    return text


def fairness_report(y_true, y_pred, extras, sensitive_attr="race") -> str:
    """
    False positive rate by group (people who did NOT reoffend but were predicted to), for our model
    (out-of-fold predictions) and for COMPAS's own score (score_text != "Low" = predicted high risk).
    """
    df = extras.copy()
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text