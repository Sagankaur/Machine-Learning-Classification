from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import ExperimentConfig


def build_model(config: ExperimentConfig):
    if config.model_name == "lightgbm":
        return LGBMClassifier(**config.tuned_lightgbm_params)
    if config.model_name == "xgboost":
        return XGBClassifier(
            random_state=config.random_state,
            eval_metric="logloss",
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
        )
    if config.model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=config.random_state)
    if config.model_name == "random_forest":
        return RandomForestClassifier(random_state=config.random_state, n_estimators=100)
    if config.model_name == "logistic_regression":
        return LogisticRegression(max_iter=5000, random_state=config.random_state)

    raise ValueError(f"Unsupported model_name: {config.model_name}")
