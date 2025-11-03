---

# 🌲 Forest Fire Detection & Simulation using Deep Learning (U-Net)

### 🔥 AI-Powered Satellite Image Analysis for Fire Risk Mapping

---

## 📖 Overview

Forest fires are one of the most devastating natural disasters, leading to biodiversity loss, air pollution, and property damage.
This project focuses on **detecting and simulating forest fires** using **AI/ML techniques**, leveraging geospatial and weather data.

We use a **U-Net deep learning model** to segment fire zones from satellite imagery and provide a **Streamlit-based frontend** to visualize predictions interactively.

---

## 🎯 Objectives

1. **Detect fire zones** from satellite images (binary segmentation: fire / no-fire).
2. **Integrate weather data** (temperature, humidity, wind speed, etc.) to enhance accuracy.
3. **Visualize predictions** with a web app for real-time uploads and overlays.
4. **Evaluate model accuracy** using IoU, Dice, and Accuracy metrics.

---

## 🧠 Project Workflow

### **1️⃣ Data Collection**

* **Satellite Imagery:**
  PNG images with orange overlay (fire zones) from Alberta and other regions.
  (Can use MODIS, VIIRS, or Sentinel-2 sources).

* **Metadata (PGW/XML):**
  Geospatial coordinates and projection details.

* **Weather Data:**
  Retrieved using the **Open-Meteo API** or ERA5 Reanalysis.
  Parameters used:

  * Temperature (°C)
  * Relative Humidity (%)
  * Wind Speed (m/s)
  * Precipitation (mm)

---

### **2️⃣ Data Processing (`src/data_preprocessor.py`)**

* Converts raster data and overlays into model-ready image/mask pairs.
* Resizes and normalizes inputs to `256x256`.
* Aligns weather data with imagery date stamps.

---

### **3️⃣ Model Training (`src/train_model.py`)**

* Model: **U-Net** with a **ResNet-34** encoder (`segmentation-models-pytorch`).
* Input channels: RGB (optionally extended with weather data).
* Output: Binary mask indicating fire zones.
* Loss: Binary Cross Entropy + Dice Loss.
* Evaluation Metrics: IoU, Dice, Accuracy.

Trained models are saved in the `models/` directory:

```
models/unet_alberta.pth
```

---

### **4️⃣ Model Evaluation (`src/evaluate_model.py`)**

* Calculates:

  * **IoU (Intersection over Union)**
  * **Dice Coefficient**
  * **Accuracy**
* Generates visual comparison charts:

  * Ground truth vs. predicted mask
  * Mean metric chart (`outputs/metrics_chart.png`)

---

### **5️⃣ Prediction (`src/predict_fire.py`)**

* Loads the trained model.
* Predicts fire zones for a new satellite image.
* Saves overlayed output (`outputs/custom_prediction.png`).

---

### **6️⃣ Streamlit Frontend (`app/streamlit_app.py`)**

Run the web app to upload satellite images and see predictions instantly.

```bash
streamlit run app/streamlit_app.py
```

**Features:**

* Upload `.png` / `.jpg` satellite images.
* View predicted fire overlays in red.
* Download prediction result.
* Interactive UI built with Streamlit.

---

## 🗂️ Project Structure

```
forest_fire_prediction/
│
├── app/
│   └── streamlit_app.py        # Streamlit frontend
│
├── src/
│   ├── config.py               # Configuration and paths
│   ├── dataset_loader.py       # Dataset class
│   ├── train_model.py          # Model training script
│   ├── evaluate_model.py       # Evaluation script
│   ├── predict_fire.py         # Custom prediction
│   ├── weather_fetcher.py      # Weather data collection
│   └── __init__.py
│
├── data/
│   └── processed/
│       └── alberta/
│           ├── images/         # Input satellite images
│           ├── masks/          # Corresponding fire masks
│           └── weather.csv     # Weather data
│
├── models/
│   └── unet_alberta.pth        # Saved model weights
│
├── outputs/
│   ├── predictions/            # Predicted overlays
│   ├── metrics_chart.png       # Evaluation results
│   └── custom_prediction.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/forest_fire_prediction.git
cd forest_fire_prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train or Use Pretrained Model

If you already have a trained model:

```bash
python -m src.evaluate_model
```

Or train again:

```bash
python -m src.train_model
```

### 5️⃣ Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Sample Results

|   Metric  |  Value |
| :-------: | :----: |
|  Mean IoU | 0.0002 |
| Mean Dice | 0.0004 |
|  Accuracy | 0.6578 |

*These values can improve with more data and weather integration.*

---

## 🚀 Future Improvements

* Integrate **live weather data** during prediction.
* Add **fire spread simulation** using **Cellular Automata**.
* Deploy Streamlit app to **Streamlit Cloud** or **AWS EC2**.
* Build **temporal models** (ConvLSTM) for multi-day predictions.

---

## 🧑‍💻 Authors

**Abhishek Abhang** AI & Geospatial Enthusiast
* 📧 [abhishekabhang2004@gmail.com](mailto:abhishekabhang2004@gmail.com)
* 💼 GitHub: [github.com/abhishekabhang314](https://github.com/abhishekabhang314)

---