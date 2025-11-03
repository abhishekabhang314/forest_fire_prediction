# src/weather_fetcher.py
import os
import requests
import pandas as pd
from datetime import datetime
from src.config import REGION_COORDS

def fetch_weather_data(region_name, start_date, end_date):
    coords = REGION_COORDS[region_name]
    lat = (coords["lat_min"] + coords["lat_max"]) / 2
    lon = (coords["lon_min"] + coords["lon_max"]) / 2

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant,"
        f"relative_humidity_2m_max"
        f"&timezone=auto"
    )

    print(f"📡 Fetching weather data for {region_name} ({start_date} → {end_date})")
    resp = requests.get(url)
    if resp.status_code != 200:
        print("❌ API request failed:", resp.text)
        return None

    data = resp.json()
    daily = data.get("daily", {})
    if not daily:
        print("⚠️ No data returned from API. Check date range or coords.")
        return None

    df = pd.DataFrame(daily)
    df.rename(columns={
        "time": "date",
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "precipitation_sum": "rainfall",
        "wind_speed_10m_max": "wind_speed",
        "wind_direction_10m_dominant": "wind_dir",
        "relative_humidity_2m_max": "humidity"
    }, inplace=True)

    # Compute mean temp
    df["temperature"] = (df["temperature_max"] + df["temperature_min"]) / 2
    df = df[["date", "temperature", "humidity", "rainfall", "wind_speed", "wind_dir"]]

    out_dir = os.path.join("data", region_name, "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weather.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Weather data saved to: {out_path}")
    print(df.head())
    return df


if __name__ == "__main__":
    region = "alberta"
    start_date = "2024-06-01"
    end_date = "2024-06-30"
    fetch_weather_data(region, start_date, end_date)
