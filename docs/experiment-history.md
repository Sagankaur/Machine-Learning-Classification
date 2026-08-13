# Experiment History

This document condenses the private Kaggle notebook archive into a repo-friendly summary.

## Scope

- Total notebook versions: `45`
- Retained versions in the archive: `40`
- Removed errored versions: `v1`, `v2`, `v4`, `v5`, `v22`
- Primary competition metric: `accuracy`
- Best score: `0.63270` on `v37`

## Evolution

### Phase 1: Baseline

- `v03`: Dummy classifier with most-frequent prediction to establish the floor.

### Phase 2: First workable pipelines

- `v06`: Linear regression with a preprocessing pipeline and thresholded output.
- `v07`: Logistic regression using the same preprocessing stack.

### Phase 3: Model exploration

- `v08` to `v15`: Broader experimentation with PCA, feature selection, K-fold validation, random forest, XGBoost, SVC, MLP, SGD, bagging, and lasso/ridge variants.

### Phase 4: Encoding and SVC-heavy experiments

- `v16` to `v21`, `v23` to `v27`: More structured SVC, LightGBM, voting, and encoding experiments.
- Best result in this block: `v17` with SVC plus PCA at `0.61920`.

### Phase 5: Hyperparameter search era

- `v28` to `v41`: Systematic GridSearchCV and RandomizedSearchCV across random forest, XGBoost, LightGBM, gradient boosting, and voting ensembles.
- Strongest results:
  - `v35`: tuned XGBoost at `0.62810`
  - `v36`: tuned GradientBoosting at `0.62740`
  - `v37`: tuned LightGBM at `0.63270` and the best score overall
  - `v41`: voting ensemble of LightGBM, XGBoost, and CatBoost at `0.63070`

### Phase 6: Cleanup and final polish

- `v42`: notebook rewrite with clearer structure and ROC-AUC evaluation.
- `v43`: added precision, recall, and confusion-matrix analysis.
- `v44`: expanded EDA and MLP experiments.
- `v45`: final polished notebook with the cleaned EDA -> pipeline -> split -> feature-selection -> model flow.

## Why only one main notebook is in the repo

The version archive is useful for private iteration history, but it makes the repository noisy and harder to review. The repo keeps:

- one final notebook for the main deliverable
- one legacy notebook that was already in the repo
- one summary document for the experiment trail

That gives you a clean project repo without losing the history context.
