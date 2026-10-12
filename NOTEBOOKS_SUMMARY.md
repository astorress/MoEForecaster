# Notebook Summary: MoE Pipeline for Hourly Demand Forecasting in Colombia's SIN

This document describes what each notebook does, what it reads and what it writes. All paths are relative to `OUTPUT_ROOT`, which is the repository folder by default (see [README](README.md#where-outputs-are-stored-output_root)). Each notebook contains the full methodological detail in its own markdown cells.

## Common structure

Every notebook follows the same layout:

1. **Title and introduction**: objective, inputs (and the notebook that produces them), outputs, and its place in the pipeline.
2. **Environment setup**: the `OUTPUT_ROOT` parameter and a bootstrap cell. The bootstrap cell finds the repository (or clones it in Colab), installs dependencies in Colab and resolves `DATA_DIR`, `MODELS_DIR` and `RESULTS_DIR`.
3. **Numbered sections**: configuration and imports, data loading, the main analysis, and saving the artifacts.

## Pipeline flow

```
01 Data acquisition      → data/raw/demand.parquet
02 Preprocessing         → data/processed/features.parquet
03 Clustering            → data/clustering/cluster_assignments.parquet + centroids
04 Cluster classifiers   → models/classifiers/
05 Expert selection      → models/experts/
06 Gating design         → models/gating/
07 HPO                   → models/hyperparameters/
08 Production assembly   → models/production/moe_forecaster.pkl
09 Benchmark             → results/benchmark/
```

---

### 01_data_acquisition.ipynb

**Objective:** download and validate hourly real demand of the SIN from the public XM API, and characterize the series with exploratory analysis.

**Input:** XM API, metric `DemaReal`, entity `Sistema`, 2020-01-01 to 2025-12-31, one request per day (2,192 days).

**Output:** `data/raw/demand.parquet` (52,608 hourly records), `data/raw/metadata.json`, figures `results/01_*.png`, `results/stationarity_results.csv`.

**Method:**
- Download with retries and exponential back-off. Each day is checked for completeness (24 records) and plausible ranges.
- The download is reused if `demand.parquet` exists; set `FORCE_DOWNLOAD=1` to download again.
- EDA: full series with a 7-day moving average; distributions by hour, day of week and month; hour × day-of-week heatmap; temporal gaps; descriptive statistics by period.
- ACF/PACF and STL decomposition with daily (24 h) and weekly (168 h) periods.
- ADF and KPSS stationarity tests.

---

### 02_preprocessing.ipynb

**Objective:** build the hourly feature matrix used by every later notebook.

**Input:** `data/raw/demand.parquet`.

**Output:** `data/processed/features.parquet` (52,608 × 39: `demand` + 38 features), `data/processed/feature_names.json`, figures `results/02_*.png`.

**Method:**
- The raw series is aligned to a regular hourly grid.
- **38 features:**
  - calendar sine/cosine encodings (6);
  - Colombian holidays (Law 51 of 1983) with `is_holiday`, `is_bridge`, and days to/from the nearest holiday (4);
  - Fourier terms with periods of 24 h and 168 h (10);
  - lags {1, 2, 3, 23, 24, 25, 47, 48, 49, 167, 168, 169} (12);
  - rolling means and standard deviations over 24, 168 and 336 h (6).
- All features use only past information, so no future data leaks into them.
- ADF/KPSS tests and an MSTL decomposition with residual diagnostics.

---

### 03_clustering.ipynb

**Objective:** define the expert domains by clustering in two dimensions: **day type** (24-hour daily profiles) and **week type** (168-hour weekly profiles).

**Input:** `data/processed/features.parquet`, restricted to 2022–2025.

**Output:** `data/clustering/cluster_assignments.parquet` (hourly `cluster_day`, `cluster_week` and `expert_id`), the day and week centroids, `clustering_metrics.json`, `top_configs.json`, and figures `results/03_*.png`.

**Method:**
- Daily and weekly demand profile matrices.
- Sweep of k-means (Euclidean), k-means with DTW and k-Shape over a range of k.
- Configurations are ranked with silhouette, Davies–Bouldin and Calinski–Harabasz indices plus temporal consistency (ARI).
- The final configuration (day k = 3, week k = 2) is fixed in the notebook. Section 6.1 explains this choice.
- Each hour receives the expert identifier `day<i>_week<j>`.

---

### 04_cluster_analysis_classifiers.ipynb

**Objective:**
- check how the clusters relate to calendar variables;
- train classifiers that predict the day type and week type from the features (they are the basis of the gating).

**Input:** `features.parquet`, `cluster_assignments.parquet`.

**Output:** `models/classifiers/` (the best classifier per target), `results/04_classifier_comparison.csv`, `results/04_confidence_analysis.json`, figures `results/04_*.png`.

**Method:**
- Cluster vs. calendar analysis: contingency tables, χ², normalized mutual information and Cohen's κ.
- Six classifiers: logistic regression, random forest, gradient boosting, XGBoost, k-NN and SVM.
- Three feature sets: calendar only, autoregressive only, and all.
- 10-fold stratified cross-validation scored by accuracy and macro F1.
- Confidence, calibration and threshold analysis of the selected classifiers.

---

### 05_expert_selection.ipynb

**Objective:** select the best model for each expert domain.

**Input:** `features.parquet`, `cluster_assignments.parquet`; 2020–2021 are excluded.

**Output:** `models/experts/expert_<id>.pkl`, the selection tables (CSV) and figures `results/05_*.png`.

**Method:**
- Three model families (linear, tree ensembles, deep learning with LSTM/GRU), with five candidate models each.
- Walk-forward evaluation engine; metrics MAE, RMSE, MAPE and MASE.
- **Homogeneous strategy:** one family for all experts, with the best model of that family chosen per expert.
- **Heterogeneous strategy:** the best model per expert across all families.
- Bias–variance constraints per expert (heuristic thresholds).
- The final experts are trained on all the data of their domain.

---

### 06_gating_design.ipynb

**Objective:** design and compare the mechanisms that route each hour to the experts.

**Input:** `features.parquet`, `cluster_assignments.parquet`, the centroids, the classifiers (04) and the experts (05).

**Output:** `models/gating/` (the selected configuration as JSON), comparison tables and figures `results/06_*.png`.

**Method:**
- **Hard gating:** the most likely expert.
- **Distance-based soft gating:** a softmax over distances to the centroids, with temperature τ.
- **Probability-based soft gating:** weights equal to P(day type) × P(week type).
- **Adaptive gating:** hard when the classifiers' minimum confidence reaches the threshold θ, soft otherwise.
- Sweeps of θ and τ, with analysis by hour, by day type and by confidence level.

---

### 07_hpo.ipynb

**Objective:** tune the hyperparameters of each expert and the gating parameters.

**Input:** `features.parquet`, `cluster_assignments.parquet`, the expert selection from 05.

**Output:** `models/hyperparameters/` (best hyperparameters per expert, as JSON), search-space tables, figures `results/07_*.png`.

**Method:**
- `HalvingRandomSearchCV` (random search with successive halving) and `TimeSeriesSplit`, using family-specific search spaces.
- Convergence curves and parallel-coordinate plots.
- Grid search of the gating threshold and temperature on a validation period.

---

### 08_production_assembly.ipynb

**Objective:** assemble the final MoE system and serialize it.

**Input:** `features.parquet`, `cluster_assignments.parquet`, the outputs of 05–07.

**Output:**
- `models/production/moe_forecaster.pkl` (the complete object) and `models/production/experts/`;
- `model_card.json`, `feature_config.json` and `gating_params.json`;
- figures `results/08_*.png`.

**Method:**
- Retrain the experts and the classifiers on the full data set (2022 onwards).
- Assemble `MoEForecaster` (`src/moe_forecaster/forecaster.py`), which implements adaptive gating. Its methods are `predict()`, `predict_with_metadata()` (expert, confidences and gating mode for each hour) and `get_model_card()`.
- In-sample sanity check, then a recursive demonstration forecast for January 2026.
- Serialize the model, then reload it to check that it works.

---

### 09_benchmark_comparison.ipynb

**Objective:** compare the MoE with conventional forecasting models.

**Input:** `features.parquet`, `models/production/moe_forecaster.pkl`. Section 11 also downloads January 2026 data from the XM API.

**Output:** `results/benchmark/` (ranking tables, statistical tests, interpretability and robustness analyses, figures).

**Method:**
1. **Screening of 30 baselines:** 10 statistical, 10 machine-learning and 10 deep-learning (PyTorch) models, on a chronological train/validation/test split.
2. **Model selection for HPO:** one model per family (Lasso, AdaBoost, Transformer), chosen in Section 7.
3. **HPO:** random search on the selected models.
4. **Final evaluation on the test set:**
   - predictive performance (MAE, RMSE, MAPE);
   - computational efficiency;
   - interpretability (1–5 scale, SHAP, t-SNE/UMAP views of the MoE routing, routing stability);
   - statistical tests (Friedman, Wilcoxon with Bonferroni correction);
   - robustness across COVID-19 periods (lockdown, Holy Week, reopening, Christmas, national strike).
5. **Out-of-sample forecast for January 2026,** compared against actual XM data.
