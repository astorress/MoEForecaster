"""Mixture of Experts for hourly electricity demand forecasting in Colombia's National Interconnected System (SIN)."""

from moe_forecaster.config import setup_paths
from moe_forecaster.forecaster import MoEForecaster

__all__ = ["MoEForecaster", "setup_paths"]
