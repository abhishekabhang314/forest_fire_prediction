"""
Configuration file for Forest Fire Detection Project
-----------------------------------------------------
This file contains:
- Directory paths
- Region coordinates
- API URLs and keys (if needed)
- Model and training parameters
"""

import os

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === Region Coordinates (approximate center points for weather API) ===
REGION_COORDS = {
    "alberta": {
        "lat_min": 49.0,
        "lat_max": 60.0,
        "lon_min": -120.0,
        "lon_max": -110.0
    },
    "uttarakhand": (30.3, 78.0),
    "california": (37.2, -119.5),
    "australia_nsw": (-33.0, 146.0),
}

# === Weather API ===
# Using Open-Meteo (Free, no API key required)
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# Example:
# f"{WEATHER_API}?latitude={lat}&longitude={lon}&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max&timezone=auto"

# === Model Parameters ===
MODEL_CONFIG = {
    "input_channels": 6,   # RGB + weather (temp, humidity, wind)
    "output_classes": 1,   # binary mask (fire/no-fire)
    "encoder": "resnet34",
    "encoder_weights": "imagenet",
    "learning_rate": 1e-4,
    "epochs": 5,
    "batch_size": 4,
    "image_size": (256, 256),
}

# === Logging and Checkpoints ===
LOG_FILE = os.path.join(BASE_DIR, "train_log.txt")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "checkpoint.pth")

# === Misc Settings ===
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.system("nvidia-smi >nul 2>&1") == 0 else "cpu"

# === Utility Function ===
def print_config_summary():
    """Prints a short summary of key configuration values."""
    print("\n🌲 Forest Fire Detection Configuration")
    print(f"Base Dir     : {BASE_DIR}")
    print(f"Data Dir     : {DATA_DIR}")
    print(f"Models Dir   : {MODELS_DIR}")
    print(f"Device       : {DEVICE}")
    print(f"Default Model: {MODEL_CONFIG['encoder']} ({MODEL_CONFIG['input_channels']} → {MODEL_CONFIG['output_classes']})")
    print(f"Weather API  : {WEATHER_API}\n")


if __name__ == "__main__":
    print_config_summary()
