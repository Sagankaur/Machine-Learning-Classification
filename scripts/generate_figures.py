from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from system_threat_forecaster import ExperimentConfig
from system_threat_forecaster.train import prepare_features


def main() -> None:
    config = ExperimentConfig()
    assets_dir = Path("docs") / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    train_df["target"].value_counts().sort_index().plot(
        kind="bar",
        ax=ax,
        color=["#60a5fa", "#16a34a"],
    )
    ax.set_title("Class Distribution in Training Data")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(assets_dir / "class-distribution.png", dpi=200)
    plt.close(fig)

    missing_pct = (train_df.isna().mean() * 100).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=missing_pct.values,
        y=missing_pct.index,
        ax=ax,
        color="#f59e0b",
    )
    ax.set_title("Top 10 Features by Missing-Value Percentage")
    ax.set_xlabel("Missing Values (%)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(assets_dir / "missing-values-top10.png", dpi=200)
    plt.close(fig)

    features, target, _ = prepare_features(train_df, test_df, config)
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    pipeline = joblib.load(config.artifact_dir / "lightgbm_pipeline.joblib")
    predictions = pipeline.predict(x_val)
    matrix = confusion_matrix(y_val, predictions)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=matrix).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(assets_dir / "confusion-matrix.png", dpi=200)
    plt.close(fig)

    probabilities = pipeline.predict_proba(x_val)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_val, probabilities, ax=ax)
    ax.set_title(f"Validation ROC Curve (AUC = {roc_auc_score(y_val, probabilities):.3f})")
    fig.tight_layout()
    fig.savefig(assets_dir / "roc-curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_val, probabilities, ax=ax)
    ax.set_title(
        "Validation Precision-Recall Curve "
        f"(AP = {average_precision_score(y_val, probabilities):.3f})"
    )
    fig.tight_layout()
    fig.savefig(assets_dir / "precision-recall-curve.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
