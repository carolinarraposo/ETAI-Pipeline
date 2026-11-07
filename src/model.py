"""Model construction."""
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from src.preprocessing import build_preprocessor

from xgboost import XGBClassifier

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

class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Keeps the features chosen by at least `min_votes` of three models, fitted on the training
    rows only: a random forest and gradient boosting vote for features with above-average
    importance, an L1-penalised logistic regression for features it doesn't shrink to zero.
    """

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(X, y)
        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(lasso.coef_[0]) > 1e-8))
        self.support_ = self.votes_ >= self.min_votes
        return self

    def _get_support_mask(self):
        return self.support_

_MODELS = {
    "dummy": DummyClassifier,   # always predicts the majority class: the floor every real model must beat
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,   # many trees on random row/feature subsets, predictions averaged (untuned)
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "adaboost_lr": adaboost_lr,
    "gradient_boosting": GradientBoostingClassifier,   # many trees in sequence, each one correcting the previous (untuned)
    "hist_gradient_boosting": HistGradientBoostingClassifier,   # faster version of gradient boosting, with histogram binning
    "naive_bayes": GaussianNB,   # naive Bayes with Gaussian likelihoods
    "xgboost": XGBClassifier,   # gradient boosting with regularization and early stopping
}

def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict, selection_config: dict = None,
                   memory=None) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing, feature selection
    (if `selection_config["enabled"]`) and model as one Pipeline, so every learned step is
    re-fit on the training part of each fold. Step names address hyperparameters from outside
    (`model__max_depth`, `select__min_votes`). `memory` (a folder) caches the fitted
    preprocessing and selection steps: a fit on the same rows with the same settings is
    reused instead of repeated, so tuning the model doesn't refit them in every trial.
    """
    steps = [("prep", build_preprocessor(preprocessing_config))]
    if selection_config and selection_config.get("enabled", False):
        steps.append(("select", VoteSelector(min_votes=selection_config.get("min_votes", 2),
                                             random_state=selection_config.get("random_state"))))
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)