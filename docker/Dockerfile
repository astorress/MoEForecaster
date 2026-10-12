# Runtime image for the notebooks (CPU). On a machine with an NVIDIA GPU the
# notebooks use CUDA automatically when a CUDA build of PyTorch is available.
FROM python:3.12-slim

# libgomp1 is required by LightGBM and XGBoost; git lets the notebooks find the repository
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# src/ on the PYTHONPATH so moe_forecaster (and moe_forecaster.pkl) can be imported anywhere
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
