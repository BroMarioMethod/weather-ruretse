"""
Generate predictions from the latest NWP data.
Run via cron every 6 hours:  0 */6 * * *  uv run python predict.py
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config
from features import build_feature_set, get_feature_columns
from models import WeatherPredictor, PrecipitationPredictor


def fetch_current_forecast() -> pd.DataFrame:
    """Grab the latest Open-Meteo forecast and return as DataFrame."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(config.HOURLY_FORECAST_VARS),
        "timezone": "UTC",
        "forecast_days": config.FORECAST_DAYS,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Rename to match training column names
    rename = {
        "temperature_2m": "fcst_temp",
        "dewpoint_2m": "fcst_dewpoint",
        "relative_humidity_2m": "fcst_humidity",
        "pressure_msl": "fcst_pressure",
        "surface_pressure": "fcst_surface_pressure",
        "wind_speed_10m": "fcst_wind_speed",
        "wind_direction_10m": "fcst_wind_dir",
        "wind_gusts_10m": "fcst_wind_gust",
        "precipitation": "fcst_precip",
        "precipitation_probability": "fcst_precip_prob",
        "cloud_cover": "fcst_cloud",
        "cape": "fcst_cape",
        "visibility": "fcst_visibility",
    }
    df.rename(columns=rename, inplace=True)
    df["lead_hours"] = range(len(df))

    # For lag/tendency features we need obs columns —
    # use forecast values as proxy (they'll be NaN for future)
    # In production the most recent obs fill in from collector.
    for obs_col, fcst_col in [
        ("obs_temp", "fcst_temp"),
        ("obs_precip", "fcst_precip"),
        ("obs_humidity", "fcst_humidity"),
        ("obs_pressure", "fcst_pressure"),
        ("obs_wind_speed", "fcst_wind_speed"),
        ("obs_wind_dir", "fcst_wind_dir"),
        ("obs_dewpoint", "fcst_dewpoint"),
        ("obs_cloud", "fcst_cloud"),
    ]:
        df[obs_col] = df[fcst_col]  # proxy — improves once real obs fill in

    return df


def main() -> None:
    print(f"[predict] Loading models...")
    wp = WeatherPredictor()
    wp.load(config.MODEL_PATH)

    pp = PrecipitationPredictor()
    pp.load(config.PRECIP_MODEL_PATH)

    print(f"[predict] Fetching latest forecast...")
    df = fetch_current_forecast()
    df = build_feature_set(df)

    feature_cols = get_feature_columns(df)
    # Only use features that the model was trained on
    trained_features = wp.models["temperature"].feature_name_
    available = [c for c in trained_features if c in df.columns]
    missing = [c for c in trained_features if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing {len(missing)} features, filling with 0: {missing[:5]}...")
        for col in missing:
            df[col] = 0
    X = df[trained_features]

    weather = wp.predict(X)
    precip = pp.predict(X)

    # ── Build output ──
    forecast_rows = []
    for i, t in enumerate(df.index):
        row = {
            "time": t.isoformat(),
            "lead_hours": int(df["lead_hours"].iloc[i]),
            "temperature_c": round(float(weather["temperature"][i]), 1),
            "humidity_pct": round(float(weather["humidity"][i]), 0),
            "wind_speed_kmh": round(float(weather["wind_speed"][i]), 1),
            "precip_probability_pct": round(float(precip["precip_probability"][i]) * 100, 0),
            "precip_expected_mm": round(float(precip["precip_expected_mm"][i]), 2),
        }
        if "temperature_q10" in weather:
            row["temperature_range_c"] = [
                round(float(weather["temperature_q10"][i]), 1),
                round(float(weather["temperature_q90"][i]), 1),
            ]
        if "precip_q10_mm" in precip:
            row["precip_range_mm"] = [
                round(float(precip["precip_q10_mm"][i]), 2),
                round(float(precip["precip_q90_mm"][i]), 2),
            ]
        forecast_rows.append(row)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "location": {"lat": config.LATITUDE, "lon": config.LONGITUDE, "name": config.LOCATION_NAME},
        "hourly": forecast_rows,
    }

    outpath = config.BASE_DIR / "latest_forecast.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[predict] Wrote {len(forecast_rows)} hours → {outpath}")

    # Preview next 24h
    print(f"\n{'Time':>20s} {'Temp':>6s} {'Hum':>5s} {'Wind':>6s} {'P(rain)':>8s} {'Rain mm':>8s}")
    print("-" * 60)
    for row in forecast_rows[:24]:
        print(
            f"{row['time'][11:16]:>20s} "
            f"{row['temperature_c']:>5.1f}° "
            f"{row['humidity_pct']:>4.0f}% "
            f"{row['wind_speed_kmh']:>5.1f} "
            f"{row['precip_probability_pct']:>7.0f}% "
            f"{row['precip_expected_mm']:>7.2f}"
        )


if __name__ == "__main__":
    main()