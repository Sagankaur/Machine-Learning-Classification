# Machine Learning Classification

Classification research project for the Kaggle `System Threat Forecaster` competition.

Kaggle notebook: <https://www.kaggle.com/code/sagandeep/23f2003511-notebook-t12025>  
Competition: <https://www.kaggle.com/competitions/System-Threat-Forecaster>

## Repository Layout

- `docs/technical-report.md`: IEEE-style technical research report
- `docs/experiment-history.md`: condensed summary of the full notebook archive
- `docs/assets/`: report figures
- `notebooks/system-threat-forecaster-final.ipynb`: curated final notebook
- `notebooks/mlp-project-legacy.ipynb`: older notebook preserved for reference
- `src/system_threat_forecaster/`: reusable research code extracted from notebook logic
- `scripts/train_model.py`: entrypoint for running the refactored pipeline
- `requirements.txt`: dependency list
- `pyproject.toml`: package metadata for the `src/` layout

## Recommended Local Layout

```text
Machine-Learning-Classification/
├── artifacts/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

`data/` and `artifacts/` are intentionally gitignored.

## Research Summary

- Total notebook versions studied: `45`
- Best recorded score: `0.63270`
- Best experiment: `v37`, tuned `LightGBM`
- Final curated notebook in this repo: `v45`

## Running the Refactored Pipeline

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Place the Kaggle `train.csv` and `test.csv` files in `data/`.
4. Run `python scripts/train_model.py`.

The script writes outputs to `artifacts/`, including:

- trained pipeline artifact
- `submission.csv`
- `metrics.json`

## Notes

This repository is intentionally curated. The full private Kaggle notebook archive and helper scripts remain outside the repo in the separate `kaggle_notebooks` workspace folder.
