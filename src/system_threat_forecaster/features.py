from __future__ import annotations

import pandas as pd

from .config import ExperimentConfig


def add_date_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    transformed = df.copy()

    for column in config.date_columns:
        transformed[column] = pd.to_datetime(transformed[column], errors="coerce")
        transformed[column] = transformed[column].fillna(transformed[column].min())

    transformed["Days_Between_os_as"] = (
        transformed["DateOS"] - transformed["DateAS"]
    ).dt.days

    for prefix in config.date_columns:
        transformed[f"{prefix}_Year"] = transformed[prefix].dt.year
        transformed[f"{prefix}_Month"] = transformed[prefix].dt.month
        transformed[f"{prefix}_Day"] = transformed[prefix].dt.day

    current_date = pd.Timestamp.now()
    transformed["DaysSinceASUpdate"] = (current_date - transformed["DateAS"]).dt.days
    transformed["DaysSinceOSUpdate"] = (current_date - transformed["DateOS"]).dt.days

    return transformed.drop(columns=list(config.date_columns), errors="ignore")


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda column: column.map(lambda value: value.strip() if isinstance(value, str) else value))


def drop_low_information_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.drop(columns=list(config.drop_columns), errors="ignore")
    test_out = test_df.drop(columns=list(config.drop_columns), errors="ignore")
    return train_out, test_out


def remove_duplicate_rows(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    deduped = features.drop_duplicates()
    aligned_target = target.loc[deduped.index]
    return deduped, aligned_target
