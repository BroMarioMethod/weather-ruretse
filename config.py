from pathlib import Path

# ── Your location ──────────────────────────────────────
LATITUDE = -24.601389
LONGITUDE = 26.067500
TIMEZONE = "Africa/Gaborone"
LOCATION_NAME = "Ruretse"

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "weather_data.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "weather_model.pkl"
PRECIP_MODEL_PATH = MODEL_DIR / "precip_model.pkl"

# ── Collection settings ───────────────────────────────
FORECAST_DAYS = 7
HISTORY_YEARS = 15  # how far back to seed on first run

# ── Thresholds ─────────────────────────────────────────
PRECIP_THRESHOLD_MM = 0.1  # minimum mm to count as "rain"

# ── Open-Meteo hourly variables ────────────────────────
HOURLY_FORECAST_VARS = [
    "temperature_2m",
    "dewpoint_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_probability",
    "cloud_cover",
    "cape",
    "visibility",
]

HOURLY_HISTORY_VARS = [
    "temperature_2m",
    "dewpoint_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

# ── Model names to pull from Open-Meteo ───────────────
FORECAST_MODELS = ["best_match", "gfs_seamless", "icon_seamless"]

# ── Historical forecast API (for backfill) ─────────────
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Some variables (precipitation_probability, visibility) may not be
# available historically — use this subset for backfill
HOURLY_HISTORICAL_FORECAST_VARS = [
    "temperature_2m",
    "dewpoint_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "cape",
]