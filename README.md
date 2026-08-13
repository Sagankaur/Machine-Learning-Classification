# Machine Learning Classification

Classification work for the Kaggle `System Threat Forecaster` competition.

Kaggle notebook: <https://www.kaggle.com/code/sagandeep/23f2003511-notebook-t12025>  
Competition: <https://www.kaggle.com/competitions/System-Threat-Forecaster>

## What is in this repo

- `notebooks/system-threat-forecaster-final.ipynb`: the curated final Kaggle notebook.
- `notebooks/mlp-project-legacy.ipynb`: the older repo notebook kept for reference.
- `docs/experiment-history.md`: condensed history of the 45 Kaggle notebook iterations.
- `requirements.txt`: Python packages needed to run the notebook locally.

## What is not in this repo

- Kaggle download scripts and notebook-sync utilities.
- All 40 retained versioned notebooks from the private working archive.
- Competition datasets, submissions, model binaries, and other generated outputs.

Those belong in a separate working/archive folder, not in the public project repo.

## Recommended local layout

```text
Machine-Learning-Classification/
├── docs/
├── notebooks/
├── data/              # keep local, gitignored
├── requirements.txt
├── .gitignore
└── README.md
```

## Running locally

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Place the Kaggle competition files under a local `data/` folder.
4. Update notebook paths from Kaggle input paths to your local dataset paths before running.

## Results summary

- Best Kaggle score in the version history: `0.63270`
- Best-scoring version: `v37` using tuned `LightGBM`
- Final curated notebook in this repo: `v45`, which reflects the cleaned end-state workflow and final modeling pipeline

## Notes

This repo is intentionally curated. The full notebook archaeology lives outside the repo in the separate `kaggle_notebooks` workspace folder.
