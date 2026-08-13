from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ExperimentConfig


def load_datasets(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    return train_df, test_df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
