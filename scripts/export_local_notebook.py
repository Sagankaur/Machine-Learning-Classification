from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


def main() -> None:
    repo_root = Path(".")
    output_path = repo_root / "notebooks" / "system-threat-forecaster-local-executed.ipynb"
    metrics = json.loads((repo_root / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    train_df = pd.read_csv(Path("..") / "train.csv")
    test_df = pd.read_csv(Path("..") / "test.csv")
    submission_head = pd.read_csv(repo_root / "artifacts" / "submission.csv").head(10)

    dataset_summary = (
        f"train shape: {train_df.shape}\n"
        f"test shape: {test_df.shape}\n\n"
        "target counts:\n"
        f"{train_df['target'].value_counts().to_string()}\n\n"
        "top missing-value percentages:\n"
        f"{((train_df.isna().mean() * 100).sort_values(ascending=False).head(10).round(2)).to_string()}"
    )

    metrics_summary = (
        f"model_name: {metrics['model_name']}\n"
        f"validation_accuracy: {metrics['validation_accuracy']:.4f}\n\n"
        f"{metrics['classification_report']}"
    )

    submission_preview = submission_head.to_string(index=False)

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.14",
    }

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# System Threat Forecaster Local Execution\n\n"
            "This notebook is a clean exported results notebook generated from the refactored "
            "pipeline using the local dataset files:\n\n"
            "- `C:\\\\sagan_ASUS\\\\IIT\\\\ml\\\\train.csv`\n"
            "- `C:\\\\sagan_ASUS\\\\IIT\\\\ml\\\\test.csv`\n\n"
            "It summarizes the executed run, generated figures, and submission preview."
        ),
        nbf.v4.new_markdown_cell(
            "## Task\n\n"
            "Predict the binary `target` label for each system in `test.csv` using the labeled "
            "examples in `train.csv`. The target indicates whether the system belongs to the "
            "infected class (`1`) or non-infected class (`0`)."
        ),
        nbf.v4.new_code_cell(
            source=(
                "import pandas as pd\n"
                "train = pd.read_csv(r'..\\\\train.csv')\n"
                "test = pd.read_csv(r'..\\\\test.csv')\n"
                "print(train.shape, test.shape)\n"
                "print(train['target'].value_counts())\n"
            ),
            execution_count=1,
            outputs=[nbf.v4.new_output("stream", name="stdout", text=dataset_summary + "\n")],
        ),
        nbf.v4.new_markdown_cell(
            "## Dataset Figures\n\n"
            "**Figure 1. Class distribution in the training data.**\n\n"
            "![Class distribution](../docs/assets/class-distribution.png)\n\n"
            "**Figure 2. Top 10 features by missing-value percentage.**\n\n"
            "![Missing values](../docs/assets/missing-values-top10.png)"
        ),
        nbf.v4.new_code_cell(
            source=(
                "import json\n"
                "from pathlib import Path\n"
                "metrics = json.loads(Path('../artifacts/metrics.json').read_text())\n"
                "print(metrics['validation_accuracy'])\n"
                "print(metrics['classification_report'])\n"
            ),
            execution_count=2,
            outputs=[nbf.v4.new_output("stream", name="stdout", text=metrics_summary + "\n")],
        ),
        nbf.v4.new_markdown_cell(
            "## Validation Figures\n\n"
            "**Figure 3. Confusion matrix from the local validation split.**\n\n"
            "![Confusion matrix](../docs/assets/confusion-matrix.png)\n\n"
            "**Figure 4. ROC curve from the trained LightGBM pipeline.**\n\n"
            "![ROC curve](../docs/assets/roc-curve.png)\n\n"
            "**Figure 5. Precision-recall curve from the trained LightGBM pipeline.**\n\n"
            "![Precision-recall curve](../docs/assets/precision-recall-curve.png)"
        ),
        nbf.v4.new_code_cell(
            source=(
                "submission = pd.read_csv('../artifacts/submission.csv')\n"
                "print(submission.head(10).to_string(index=False))\n"
            ),
            execution_count=3,
            outputs=[nbf.v4.new_output("stream", name="stdout", text=submission_preview + "\n")],
        ),
        nbf.v4.new_markdown_cell(
            "## Notes\n\n"
            "This notebook is a clean local export of the executed pipeline results. "
            "It is intentionally separate from the original Kaggle notebook, which still contains "
            "historical exploratory cells and Kaggle-specific path assumptions."
        ),
    ]

    output_path.write_text(nbf.writes(nb), encoding="utf-8")


if __name__ == "__main__":
    main()
