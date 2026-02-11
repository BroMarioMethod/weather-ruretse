"""
Daily verification: how did yesterday's forecasts perform?
Run daily:  0 8 * * *  uv run python verify.py
"""
import numpy as np
import pandas as pd

import db


def main() -> None:
    conn = db.get_conn()

    query = """
        WITH ranked AS (
            SELECT f.*, o.temperature_2m AS obs_temp,
                   o.precipitation AS obs_precip,
                   o.wind_speed_10m AS obs_wind,
                   ROW_NUMBER() OVER (
                       PARTITION BY f.valid_time
                       ORDER BY f.fetched_at DESC
                   ) AS rn
            FROM forecasts f
            JOIN observations o ON f.valid_time = o.time
            WHERE f.source = 'best_match'
              AND date(f.valid_time) >= date('now', '-2 days')
              AND date(f.valid_time) < date('now')
        )
        SELECT * FROM ranked WHERE rn = 1
        ORDER BY valid_time
    """
    df = pd.read_sql(query, conn, parse_dates=["valid_time"])
    conn.close()

    if df.empty:
        print("[verify] No verification data for yesterday.")
        return

    print(f"[verify] {len(df)} hours verified\n")

    # Temperature
    temp_errors = df["temperature_2m"] - df["obs_temp"]
    print(f"Temperature:")
    print(f"  MAE  = {temp_errors.abs().mean():.2f}°C")
    print(f"  Bias = {temp_errors.mean():+.2f}°C")

    # Precipitation
    precip_errors = df["precipitation"] - df["obs_precip"]
    print(f"\nPrecipitation:")
    print(f"  MAE  = {precip_errors.abs().mean():.2f} mm")
    print(f"  Bias = {precip_errors.mean():+.2f} mm")

    obs_rain = (df["obs_precip"] >= 0.1).astype(int)
    fcst_rain = (df["precipitation"] >= 0.1).astype(int)
    hits = ((obs_rain == 1) & (fcst_rain == 1)).sum()
    misses = ((obs_rain == 1) & (fcst_rain == 0)).sum()
    false_alarms = ((obs_rain == 0) & (fcst_rain == 1)).sum()
    pod = hits / max(hits + misses, 1)
    far = false_alarms / max(hits + false_alarms, 1)
    print(f"  POD (hit rate) = {pod:.2%}")
    print(f"  FAR            = {far:.2%}")

    # Wind
    wind_errors = df["wind_speed_10m"] - df["obs_wind"]
    print(f"\nWind speed:")
    print(f"  MAE  = {wind_errors.abs().mean():.2f} km/h")
    print(f"  Bias = {wind_errors.mean():+.2f} km/h")

    # Alerts
    temp_mae = temp_errors.abs().mean()
    if temp_mae > 3.0:
        print(f"\n⚠️  Temperature MAE ({temp_mae:.1f}°C) exceeds threshold — consider retraining")
    if abs(precip_errors.mean()) > 1.0:
        print(f"\n⚠️  Precipitation bias ({precip_errors.mean():+.1f} mm) — consider retraining")


if __name__ == "__main__":
    main()