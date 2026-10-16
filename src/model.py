"""Model construction."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

def oblique_random_forest(n_estimators=100, n_components=10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    A forest of oblique trees: each tree is fit on a bootstrap sample of `n_components` random sparse linear
    combinations of the features (new set per tree), so its splits are slanted hyperplanes. `tree_params` go
    to every tree. Tuned as `model__estimator__tree__<param>`.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)


def adaboost(max_depth=1, **params):
    """AdaBoost over decision trees of depth `max_depth` (1 = stumps). The depth is tuned as
    `model__estimator__max_depth`."""
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


def adaboost_lr(**params):
    """AdaBoost over logistic regressions: a weighted sum of linear models is still linear."""
    return AdaBoostClassifier(estimator=LogisticRegression(max_iter=2000), **params)

_MODELS = {
    "dummy": DummyClassifier,   # always predicts the majority class: the floor every real model must beat
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,   # many trees on random row/feature subsets, predictions averaged (untuned)
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "adaboost_lr": adaboost_lr,
}

def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)

def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing and model as ONE Pipeline,
    so every learned step is re-fit on the training part of each fold. Step names address
    hyperparameters from outside (`model__max_depth`, `prep__numeric__impute__strategy`).
    """
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])