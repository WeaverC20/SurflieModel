"""Wave forecast inference module

Production inference code for wave forecasting models.
This module is imported by the backend API.
"""

from .predictor import WaveForecastPredictor

__all__ = ["WaveForecastPredictor"]
__version__ = "0.1.0"
