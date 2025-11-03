"""
Forest Fire Detection Package
=============================

This package contains modules for:
- Data preprocessing (PNG → GeoTIFF, mask generation)
- Weather data integration (Open-Meteo API)
- Dataset loading and augmentation
- U-Net model definition and training
- Configuration and utility functions
"""

import os

# Automatically set the base project path when imported
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Handy re-exports (optional)
from .config import MODEL_CONFIG, REGION_COORDS, WEATHER_API

__all__ = [
    "MODEL_CONFIG",
    "REGION_COORDS",
    "WEATHER_API",
]
