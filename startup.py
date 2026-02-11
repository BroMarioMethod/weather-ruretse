"""
Render startup: backfill data, train models, generate forecast.
Runs once before gunicorn starts serving.
"""
import sys
import time

def run():
    print("=" * 60)
    print("  RENDER STARTUP: Initializing weather predictor")
    print("=" * 60)

    # Step 1: Initialize database
    print("\n[startup] Initializing database...")
    import db
    db.init_db()

    # Step 2: Backfill historical data (observations + forecasts)
    # Use a shorter history on Render to stay within RAM and time limits
    print("\n[startup] Backfilling historical data...")
    import config
    original_years = config.HISTORY_YEARS
    config.HISTORY_YEARS = 2  # 2 years is enough, faster than 3

    import collector
    try:
        collector.backfill_history()
    except Exception as e:
        print(f"[startup] ⚠ History backfill error (continuing): {e}")

    try:
        collector.backfill_forecasts()
    except Exception as e:
        print(f"[startup] ⚠ Forecast backfill error (continuing): {e}")
        # Fallback: copy observations as pseudo-forecasts
        print("[startup] Using archive fallback for forecasts...")
        try:
            collector.backfill_forecasts_from_archive()
        except Exception as e2:
            print(f"[startup] ⚠ Fallback also failed: {e2}")

    config.HISTORY_YEARS = original_years

    # Check data
    conn = db.get_conn()
    obs = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    fcst = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    conn.close()
    print(f"\n[startup] Observations: {obs}")
    print(f"[startup] Forecasts:    {fcst}")

    if obs < 100 or fcst < 100:
        print("[startup] ⚠ Not enough data to train. Will serve raw NWP forecasts.")
        # Still generate a forecast from raw Open-Meteo data
        try:
            generate_raw_forecast()
        except Exception as e:
            print(f"[startup] ⚠ Raw forecast failed: {e}")
        return

    # Step 3: Train models
    print("\n[startup] Training models...")
    try:
        import train
        train.main()
    except Exception as e:
        print(f"[startup] ⚠ Training failed: {e}")
        # Try raw forecast as fallback
        try:
            generate_raw_forecast()
        except Exception as e2:
            print(f"[startup] ⚠ Raw forecast also failed: {e2}")
        return

    # Step 4: Generate predictions
    print("\n[startup] Generating predictions...")
    try:
        import predict
        predict.main()
    except Exception as e:
        print(f"[startup] ⚠ Prediction failed: {e}")

    print("\n" + "=" * 60)
    print("  STARTUP COMPLETE — ready to serve")
    print("=" * 60)


def generate_raw_forecast():
    """Fallback: serve raw Open-Meteo forecasts without ML correction."""
    import json
    from datetime import datetime, timezone
    import requests
    import config

    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(config.HOURLY_FORECAST_VARS),
            "timezone": "UTC",
            "forecast_days": config.FORECAST_DAYS,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["hourly"]

    hourly = []
    for i, t in enumerate(data["time"]):
        hourly.append({
            "time": t,
            "lead_hours": i,
            "temperature_c": data["temperature_2m"][i],
            "humidity_pct": data["relative_humidity_2m"][i],
            "wind_speed_kmh": data["wind_speed_10m"][i],
            "precip_probability_pct": data.get("precipitation_probability", [0] * len(data["time"]))[i] or 0,
            "precip_expected_mm": data["precipitation"][i] or 0,
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "location": {
            "lat": config.LATITUDE,
            "lon": config.LONGITUDE,
            "name": config.LOCATION_NAME,
        },
        "note": "Raw NWP forecast — ML models not yet trained",
        "hourly": hourly,
    }

    outpath = config.BASE_DIR / "latest_forecast.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[startup] Wrote raw forecast → {outpath}")


if __name__ == "__main__":
    run()