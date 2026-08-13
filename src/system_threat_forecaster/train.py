from __future__ import annotations

from dataclasses import asdict
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import ExperimentConfig
from .data import ensure_output_dir, load_datasets
from .features import (
    add_date_features,
    drop_low_information_columns,
    remove_duplicate_rows,
    strip_string_columns,
)
from .models import build_model
from .preprocessing import build_preprocessor, cap_outliers, infer_feature_types


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df, test_df = drop_low_information_columns(train_df, test_df, config)
    train_df = add_date_features(train_df, config)
    test_df = add_date_features(test_df, config)
    train_df = strip_string_columns(train_df)
    test_df = strip_string_columns(test_df)

    target = train_df[config.target_column]
    features = train_df.drop(columns=[config.target_column])
    features, target = remove_duplicate_rows(features, target)

    features = cap_outliers(
        features, config.outlier_lower_quantile, config.outlier_upper_quantile
    )
    test_df = cap_outliers(
        test_df, config.outlier_lower_quantile, config.outlier_upper_quantile
    )
    return features, target, test_df


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    train_df, test_df = load_datasets(config)
    features, target, transformed_test = prepare_features(train_df, test_df, config)

    numeric_columns, categorical_columns = infer_feature_types(
        pd.concat([features, target.rename(config.target_column)], axis=1),
        config.target_column,
    )
    preprocessor = build_preprocessor(numeric_columns, categorical_columns, config)
    model = build_model(config)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, output_dict=False)

    ensure_output_dir(config.artifact_dir)
    joblib.dump(pipeline, config.artifact_dir / f"{config.model_name}_pipeline.joblib")

    submission_predictions = pipeline.predict(transformed_test)
    submission = pd.DataFrame(
        {"id": range(len(submission_predictions)), "target": submission_predictions}
    )
    submission.to_csv(config.artifact_dir / "submission.csv", index=False)

    metrics = {
        "model_name": config.model_name,
        "validation_accuracy": accuracy,
        "classification_report": report,
        "config": {
            key: str(value) if hasattr(value, "as_posix") else value
            for key, value in asdict(config).items()
        },
    }
    (config.artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics
