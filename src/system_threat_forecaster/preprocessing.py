from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import ExperimentConfig


def cap_outliers(
    df: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    capped = df.copy()
    numeric_columns = capped.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        lower, upper = capped[column].quantile([lower_quantile, upper_quantile])
        capped[column] = np.clip(capped[column], lower, upper)

    return capped


def reduce_cardinality(
    frame: pd.DataFrame | np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    reduced = frame.copy()
    for column in reduced.columns:
        reduced[column] = reduced[column].astype(str)
        frequencies = reduced[column].value_counts(normalize=True)
        keepers = frequencies[frequencies >= threshold].index
        reduced[column] = np.where(reduced[column].isin(keepers), reduced[column], "Others")

    return reduced.to_numpy()


def infer_feature_types(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[target_column], errors="ignore")
    categorical_columns = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: ExperimentConfig,
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "cardinality_reducer",
                FunctionTransformer(
                    reduce_cardinality,
                    kw_args={"threshold": config.rare_category_threshold},
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )
