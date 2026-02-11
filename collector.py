"""
Fetch forecast and observation data from Open-Meteo.

Usage:
    uv run python collector.py --backfill     # one-time historical seed
    uv run python collector.py                # regular cron run

Cron:  0 */6 * * *  cd $PROJECT && uv run python collector.py
"""
import sys
import time as _time
from datetime import datetime, timedelta, timezone

import requests

import config
import db


# ═══════════════════════════════════════════════════════
#  API Fetchers
# ═══════════════════════════════════════════════════════

def fetch_forecast() -> dict:
    """Current forecast from the live API."""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(config.HOURLY_FORECAST_VARS),
            "models": ",".join(config.FORECAST_MODELS),
            "timezone": "UTC",
            "forecast_days": config.FORECAST_DAYS,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_history(start_date: str, end_date: str) -> dict:
    """Historical observations (ERA5 reanalysis) from the archive API."""
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(config.HOURLY_HISTORY_VARS),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_historical_forecast(start_date: str, end_date: str) -> dict:
    """What NWP models actually predicted for past dates."""
    resp = requests.get(
        config.HISTORICAL_FORECAST_URL,
        params={
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(config.HOURLY_HISTORICAL_FORECAST_VARS),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════
#  Storage helpers
# ═══════════════════════════════════════════════════════

def store_forecast(data: dict) -> None:
    """Store live forecast data."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    hourly = data["hourly"]
    n = len(hourly["time"])

    rows = []
    for i in range(n):
        row = {"time": hourly["time"][i], "lead_hours": i}
        for var in config.HOURLY_FORECAST_VARS:
            row[var] = hourly.get(var, [None] * n)[i]
        rows.append(row)

    db.insert_forecasts(rows, source="best_match", fetched_at=fetched_at)
    print(f"[collector] Stored {n} forecast rows ({fetched_at})")


def store_historical_forecast_chunk(data: dict) -> None:
    """Store historical forecast data into the forecasts table."""
    fetched_at = "backfill"  # marker so we know this was backfilled
    hourly = data["hourly"]
    n = len(hourly["time"])

    rows = []
    for i in range(n):
        row = {"time": hourly["time"][i], "lead_hours": None}
        for var in config.HOURLY_HISTORICAL_FORECAST_VARS:
            row[var] = hourly.get(var, [None] * n)[i]
        # Variables not in historical set → None
        for var in config.HOURLY_FORECAST_VARS:
            if var not in row:
                row[var] = None
        rows.append(row)

    db.insert_forecasts(rows, source="best_match", fetched_at=fetched_at)
    print(f"[collector] Stored {n} historical forecast rows")


def store_history(data: dict) -> None:
    """Store observation / reanalysis data."""
    hourly = data["hourly"]
    n = len(hourly["time"])

    rows = []
    for i in range(n):
        row = {"time": hourly["time"][i]}
        for var in config.HOURLY_HISTORY_VARS:
            row[var] = hourly.get(var, [None] * n)[i]
        rows.append(row)

    db.insert_observations(rows)
    print(f"[collector] Stored {n} observation rows")


# ═══════════════════════════════════════════════════════
#  Backfill routines
# ═══════════════════════════════════════════════════════

def backfill_history() -> None:
    """Seed database with historical observations."""
    end = datetime.now(timezone.utc) - timedelta(days=5)
    start = end - timedelta(days=365 * config.HISTORY_YEARS)
    _chunked_fetch(start, end, fetch_history, store_history, "observations")


def backfill_forecasts() -> None:
    """Seed database with historical NWP forecast data."""
    end = datetime.now(timezone.utc) - timedelta(days=5)
    start = end - timedelta(days=365 * config.HISTORY_YEARS)
    _chunked_fetch(
        start, end,
        fetch_historical_forecast,
        store_historical_forecast_chunk,
        "historical forecasts",
    )

def backfill_forecasts_from_archive() -> None:
    """
    Fallback: copy archive/reanalysis data into the forecasts table.
    Less ideal than real historical forecasts but gets training working.
    """
    conn = db.get_conn()
    count = conn.execute("""
        INSERT OR REPLACE INTO forecasts
            (fetched_at, valid_time, source, lead_hours,
             temperature_2m, dewpoint_2m, relative_humidity_2m,
             pressure_msl, surface_pressure,
             wind_speed_10m, wind_direction_10m, wind_gusts_10m,
             precipitation, precipitation_probability,
             cloud_cover, cape, visibility)
        SELECT
            'backfill-archive', time, 'best_match', NULL,
            temperature_2m, dewpoint_2m, relative_humidity_2m,
            pressure_msl, surface_pressure,
            wind_speed_10m, wind_direction_10m, NULL,
            precipitation, NULL,
            cloud_cover, NULL, NULL
        FROM observations
    """).rowcount
    conn.commit()
    conn.close()
    print(f"[collector] Copied {count} observation rows → forecasts table (fallback)")


def _chunked_fetch(start, end, fetch_fn, store_fn, label, chunk_days=30):
    """Fetch data in chunks to stay within API limits."""
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        s = chunk_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"[collector] Fetching {label} {s} → {e}")

        try:
            data = fetch_fn(s, e)
            store_fn(data)
        except requests.exceptions.HTTPError as exc:
            print(f"[collector] ⚠ HTTP error for {s}→{e}: {exc}")
            print(f"[collector]   Skipping chunk, continuing...")
        except Exception as exc:
            print(f"[collector] ⚠ Error for {s}→{e}: {exc}")
            print(f"[collector]   Skipping chunk, continuing...")

        chunk_start = chunk_end + timedelta(days=1)
        _time.sleep(1.5)  # be polite to the free API


# ═══════════════════════════════════════════════════════
#  Regular update
# ═══════════════════════════════════════════════════════

def update_recent_observations() -> None:
    """Fetch last 7 days of observations to fill recent gaps."""
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=7)
    data = fetch_history(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    store_history(data)
    print("[collector] Updated recent observations")


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main() -> None:
    db.init_db()

    if "--backfill" in sys.argv:
        print("═══ Backfilling observations ═══")
        backfill_history()
        print("\n═══ Backfilling historical forecasts ═══")
        backfill_forecasts()
        # Verify
        conn = db.get_conn()
        obs = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        fcst = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        conn.close()
        print(f"\n✓ Observations: {obs}")
        print(f"✓ Forecasts:    {fcst}")
        return

    # Normal run: grab new forecasts + recent obs
    print("[collector] Fetching forecast...")
    forecast_data = fetch_forecast()
    store_forecast(forecast_data)

    print("[collector] Updating recent observations...")
    update_recent_observations()

    print("[collector] Done.")


if __name__ == "__main__":
    main()