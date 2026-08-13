from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    data_dir: Path = Path("data")
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_column: str = "target"
    random_state: int = 42
    test_size: float = 0.2
    rare_category_threshold: float = 0.01
    outlier_lower_quantile: float = 0.01
    outlier_upper_quantile: float = 0.99
    model_name: str = "lightgbm"
    artifact_dir: Path = Path("artifacts")
    drop_columns: tuple[str, ...] = (
        "MachineID",
        "IsBetaUser",
        "AutoSampleSubmissionEnabled",
        "IsFlightsDisabled",
    )
    date_columns: tuple[str, ...] = ("DateAS", "DateOS")
    tuned_lightgbm_params: dict[str, int | float] = field(
        default_factory=lambda: {
            "learning_rate": 0.1,
            "max_depth": 7,
            "n_estimators": 100,
            "random_state": 42,
        }
    )

    @property
    def train_path(self) -> Path:
        return self.resolve_data_dir() / self.train_file

    @property
    def test_path(self) -> Path:
        return self.resolve_data_dir() / self.test_file

    def resolve_data_dir(self) -> Path:
        candidates = (
            self.data_dir,
            Path("."),
            Path(".."),
        )
        for candidate in candidates:
            if (candidate / self.train_file).exists() and (candidate / self.test_file).exists():
                return candidate
        return self.data_dir
