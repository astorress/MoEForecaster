# Mixture of Experts (MoE) for Hourly Electricity Demand Forecasting in Colombia's SIN

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](docker-compose.yml)
[![Conda](https://img.shields.io/badge/conda-environment.yml-44A833.svg)](environment.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astorress/MoEForecaster/blob/master/notebooks/01_data_acquisition.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Reproducible research code for an hourly electricity demand forecasting system for Colombia's National Interconnected System (SIN, *Sistema Interconectado Nacional*), based on a Mixture of Experts.

This repository contains the full model-building pipeline as nine Jupyter notebooks. Data download, preprocessing, clustering, expert training, gating design, hyperparameter optimization, production assembly and benchmarking are all included. Every step can be re-executed from the public data source on a local machine (pip or conda), in Docker or in Google Colab.

---

## Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Where outputs are stored (`OUTPUT_ROOT`)](#where-outputs-are-stored-output_root)
- [How to run](#how-to-run)
  - [Option A: Google Colab](#option-a-google-colab)
  - [Option B: Conda](#option-b-conda)
  - [Option C: pip + virtual environment](#option-c-pip--virtual-environment)
  - [Option D: Docker](#option-d-docker)
- [Runtime and hardware](#runtime-and-hardware)
- [Using the production model](#using-the-production-model)
- [Reproducibility notes](#reproducibility-notes)
- [Further documentation](#further-documentation)
- [Author](#author)
- [License](#license)

---

## Overview

### Problem

Short-term forecasts of hourly electricity demand (from one day to one week ahead) support generation dispatch, maintenance planning and operating-cost reduction in Colombia's SIN. Hourly demand combines several overlapping patterns:

- **Daily (24 h)**: intraday cycle (low overnight, daytime peak, evening level).
- **Weekly (168 h)**: working days vs. weekends and holidays.
- **Seasonal**: monthly and yearly variation.

### Approach: Mixture of Experts

1. **Domains by clustering.** Days are clustered by their 24-hour profile (*day type*) and weeks by their 168-hour profile (*week type*). Each (day type, week type) combination defines an expert domain.
2. **One expert per domain.** For each domain, the best model is selected from several model families with walk-forward validation.
3. **Gating.** Classifiers estimate the probability of each day type and week type. When both are confident, the most likely expert is used (hard gating). Otherwise the experts' forecasts are combined, weighted by probability (soft gating). This combination is called *adaptive gating*.

### Data

Hourly real demand of the SIN (`DemaReal`, kWh) from the public API of XM S.A., the Colombian market operator, for 2020-01-01 to 2025-12-31 (2,192 days, 52,608 hourly records). Notebook 01 downloads it. No data is stored in this repository.

---

## Pipeline

The notebooks run in order; each reads the artifacts written by the previous ones.

| # | Notebook | Purpose | Main outputs |
|---|---|---|---|
| 01 | `01_data_acquisition` | Download and validate hourly demand from the XM API; exploratory analysis (EDA), ACF/PACF, STL decomposition, stationarity tests | `data/raw/demand.parquet`, `data/raw/metadata.json` |
| 02 | `02_preprocessing` | Build the hourly feature matrix: calendar, Colombian holidays, Fourier terms, lags and rolling statistics (38 features); MSTL and stationarity analysis | `data/processed/features.parquet`, `data/processed/feature_names.json` |
| 03 | `03_clustering` | Cluster daily (24 h) and weekly (168 h) profiles; choose the configuration; build hourly cluster assignments and expert identifiers | `data/clustering/cluster_assignments.parquet`, centroids |
| 04 | `04_cluster_analysis_classifiers` | Relate clusters to calendar variables; train and select classifiers that predict the day type and week type | `models/classifiers/` |
| 05 | `05_expert_selection` | Evaluate candidate model families per expert with walk-forward validation (homogeneous vs. heterogeneous strategy); train the final experts | `models/experts/` |
| 06 | `06_gating_design` | Compare hard, distance-based soft, probability-based soft and adaptive gating | `models/gating/`, comparison tables |
| 07 | `07_hpo` | Hyperparameter optimization per expert (`HalvingRandomSearchCV`) and gating parameter search | `models/hyperparameters/` |
| 08 | `08_production_assembly` | Retrain on the full data set, assemble the `MoEForecaster` and serialize it | `models/production/moe_forecaster.pkl`, model card |
| 09 | `09_benchmark_comparison` | Benchmark the MoE against statistical, machine-learning and deep-learning baselines | `results/benchmark/` |

The opening cells of each notebook list its exact inputs, outputs and method. [NOTEBOOKS_SUMMARY.md](NOTEBOOKS_SUMMARY.md) gives a longer description of each step.

```
XM API ──► 01 raw demand ──► 02 features ──► 03 day/week clusters ──► 04 cluster classifiers
                                                    │                         │
                                                    ▼                         ▼
                                          05 experts per cluster ──► 06 gating design
                                                    │                         │
                                                    └────────► 07 HPO ◄───────┘
                                                                 │
                                                                 ▼
                                               08 production MoEForecaster (.pkl)
                                                                 │
                                                                 ▼
                                                    09 benchmark vs. baselines
```

---

## Repository structure

```
MoEForecaster/
├── README.md                  ← this file
├── NOTEBOOKS_SUMMARY.md       ← detailed description of each notebook
├── requirements.txt           ← pinned dependencies (single source for every environment)
├── environment.yml            ← conda environment (Python 3.12 + requirements.txt)
├── Makefile                   ← make pipeline, make nb, make up, ...
├── docker-compose.yml         ← JupyterLab and runner services
├── docker/Dockerfile
│
├── src/moe_forecaster/
│   ├── config.py              ← resolves OUTPUT_ROOT and creates data/, models/, results/
│   └── forecaster.py          ← MoEForecaster class (used by notebooks 08 and 09)
│
├── notebooks/                 ← the pipeline, 01 to 09
│
├── data/                      ← created by the notebooks (empty in git)
│   ├── raw/  processed/  clustering/
├── models/                    ← created by the notebooks (empty in git)
│   ├── classifiers/  experts/  gating/  hyperparameters/  production/
└── results/                   ← figures and tables (empty in git)
```

---

## Where outputs are stored (`OUTPUT_ROOT`)

The notebooks contain no hard-coded paths. All of them store and read their artifacts under a single folder, **`OUTPUT_ROOT`**, which contains `data/`, `models/` and `results/`. It is resolved in this order:

1. The `OUTPUT_ROOT` parameter in the first code cell of each notebook.
2. The environment variable `MOE_OUTPUT_ROOT`.
3. The repository folder (default).

Relative paths are resolved from the repository root. In Colab, a path under `/content/drive` mounts Google Drive automatically.

| Setting | Artifacts are stored in |
|---|---|
| nothing (default) | `MoEForecaster/data`, `MoEForecaster/models`, `MoEForecaster/results` |
| `export MOE_OUTPUT_ROOT=/data/moe_run1` | `/data/moe_run1/{data,models,results}` |
| `OUTPUT_ROOT = "/content/drive/MyDrive/MoEForecaster"` (Colab) | your Google Drive |

> Use the **same** `OUTPUT_ROOT` for every notebook of a run, because each notebook reads what the previous ones wrote. Setting `MOE_OUTPUT_ROOT` once in your shell (or in Colab, editing the same line in each notebook) keeps them consistent.

---

## How to run

The four options below use the same notebooks and the same pinned versions from `requirements.txt`. Internet access is needed only for notebook 01, which downloads from the XM API.

### Option A: Google Colab

1. Open a notebook in Colab, e.g. [01_data_acquisition](https://colab.research.google.com/github/astorress/MoEForecaster/blob/master/notebooks/01_data_acquisition.ipynb) (or *File → Open notebook → GitHub* and paste the repository URL).
2. *Optional:* to keep the results between sessions, set `OUTPUT_ROOT` in the first code cell to a folder in your Drive, for example `"/content/drive/MyDrive/MoEForecaster"`. Otherwise outputs go to the Colab VM and are lost when the session ends.
3. *Optional:* for notebooks 05 and 09 (LSTM/GRU models), choose *Runtime → Change runtime type → GPU*.
4. Run all cells. The setup cell clones the repository and installs the pinned dependencies. Colab's own PyTorch build with CUDA is kept. If Colab asks to restart the session after the installation, restart it and run the notebook again.
5. Continue with notebooks 02, 03, … using the same `OUTPUT_ROOT`.

### Option B: Conda

```bash
git clone https://github.com/astorress/MoEForecaster.git
cd MoEForecaster
conda env create -f environment.yml
conda activate moe-forecaster

# optional: store outputs outside the repository
export MOE_OUTPUT_ROOT=/path/to/outputs

jupyter lab                      # open notebooks/ and run 01 → 09 interactively
# or, headless:
make pipeline-local              # notebooks 01–08 (executed copies in outputs/notebooks/)
make benchmark-local             # notebook 09
make nb-local NB=03_clustering   # a single notebook
```

### Option C: pip + virtual environment

Requires Python 3.12.

```bash
git clone https://github.com/astorress/MoEForecaster.git
cd MoEForecaster
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# CPU-only PyTorch (smaller download on Linux):
# pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

jupyter lab                      # or the make *-local targets shown above
```

Without `make`, e.g. on Windows, a notebook can be run headless with papermill:

```bash
papermill --cwd notebooks notebooks/01_data_acquisition.ipynb outputs/notebooks/01_data_acquisition.ipynb
```

### Option D: Docker

Requires Docker with Compose v2 and `make`. Python does not need to be installed on the host.

```bash
git clone https://github.com/astorress/MoEForecaster.git
cd MoEForecaster
make build                       # build the image (Python 3.12, CPU PyTorch)

make up                          # JupyterLab at http://localhost:8888
make down                        # stop it

make pipeline                    # notebooks 01–08, headless
make benchmark                   # notebook 09
make nb NB=03_clustering         # a single notebook

# store outputs in a host folder outside the repository:
make pipeline OUTPUT_DIR=/path/on/host
```

The executed notebooks, with their outputs and figures, are saved in `outputs/notebooks/`. `make clean` deletes everything generated inside the repository. Run `make help` to list all targets.

---

## Runtime and hardware

At least 16 GB of RAM is recommended; with Docker, give it at least 8 GB.

Approximate times on CPU (Docker on a 14-core Mac):

| Notebook | Time |
|---|---|
| 01 | ~45 min the first time (2,192 daily requests to the XM API); afterwards `data/raw/demand.parquet` is reused |
| 02 | ~2 min |
| 03 | ~1.5 h (k-means with DTW) |
| 04 | ~1.5 h (10-fold cross-validation of six classifiers) |
| 05–09 | several hours; 05 and 09 train LSTM/GRU models and are much faster on a GPU |

To force a new download in notebook 01, set the environment variable `FORCE_DOWNLOAD=1`.

---

## Using the production model

```python
import pickle
import sys

sys.path.insert(0, "src")                 # not needed in Docker (src/ is on PYTHONPATH)
from moe_forecaster import MoEForecaster  # required to unpickle the model

with open("models/production/moe_forecaster.pkl", "rb") as f:   # under your OUTPUT_ROOT
    model = pickle.load(f)

preds, gating = model.predict_with_metadata(X)   # X: DataFrame with model.feature_cols
print(model.get_model_card())
```

`predict_with_metadata` returns the forecast (kWh) and, for every hour, the selected expert, the classifier confidences and the gating mode (`hard`, `hard_fallback` or `soft`).

---

## Reproducibility notes

- **Pinned environment.** All environments install the versions in `requirements.txt`. The only exception is PyTorch in Colab, where the preinstalled CUDA build is used.
- **Random seeds.** Each notebook sets `RANDOM_SEED = 42`. Results can still differ slightly between CPU and GPU, and between hardware platforms, because of non-deterministic floating-point operations in PyTorch, LightGBM and XGBoost.
- **Source data.** XM may revise historical values. `data/raw/metadata.json` records the download timestamp of the data used in a run.
- **No synthetic fallbacks.** If an artifact from a previous notebook is missing, the notebook stops with an error naming the notebook to run first. It never continues with placeholder data.

---

## Further documentation

- [NOTEBOOKS_SUMMARY.md](NOTEBOOKS_SUMMARY.md): detailed description of every notebook.
- The notebooks themselves: each opens with its objective, inputs and outputs, and explains every section.

---

## Author

**Sebastián Torres**

- Email: [sebastiantorres8800@gmail.com](mailto:sebastiantorres8800@gmail.com)
- GitHub: [@astorress](https://github.com/astorress)

---

## License

[MIT](LICENSE)
