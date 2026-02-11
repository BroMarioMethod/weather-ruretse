# 🌦 Ruretse Weather Predictor

A lightweight, resource-efficient machine learning weather prediction system for **Ruretse, Botswana**. Uses Model Output Statistics (MOS) to correct and downscale free NWP (Numerical Weather Prediction) model forecasts for hyperlocal accuracy.

Built to run on limited hardware (Raspberry Pi, old laptop) or free-tier cloud services — no GPU required.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Data](https://img.shields.io/badge/Data-Open--Meteo-orange)

## What It Does

NWP Model Forecasts (GFS, ICON, etc.)
        │
        ▼
┌──────────────────────────┐
│   Feature Engineering    │  ← pressure tendencies, dewpoint depression,
│                          │    wind components, lag features, cyclical time
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   LightGBM MOS Models   │  ← trained on years of forecast-vs-observation pairs
│                          │
│  • Temperature corrector │
│  • Humidity corrector    │
│  • Wind corrector        │
│  • 2-stage precip model  │  ← classifier (will it rain?) + Tweedie regressor (how much?)
│  • Quantile models       │  ← prediction intervals (uncertainty)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  7-Day Hourly Forecast   │  ← served via web dashboard + JSON API
│  with charts & tables    │
└──────────────────────────┘


### Predictions Include

| Variable | Detail |
|----------|--------|
| 🌡 Temperature | Hourly °C with confidence bands |
| 💧 Humidity | Hourly % |
| 💨 Wind Speed | Hourly km/h |
| 🌧 Rain Probability | Calibrated % chance per hour |
| 🌧 Rain Amount | Expected mm with uncertainty range |
| 📊 Daily Summaries | Min/max temp, total rain, max wind per day |

---

## Screenshots

### Charts Dashboard (`/`)

The default landing page shows a 7-day visual overview with interactive charts:

- **Combined overview** — temperature, precipitation, wind, and humidity stacked
- **Daily summary bars** — at-a-glance temp range, rain totals, wind, humidity per day
- **Individual detail charts** — temperature, precipitation, wind, humidity

### Data Table (`/table`)

Hourly data grouped by day for precise value lookup.

### API (`/forecast`)

Full JSON output for integration with other tools.

---

## Architecture


weather-predictor/
│
├── config.py           # Location, paths, API settings
├── db.py               # SQLite schema, queries, data access
├── collector.py        # Fetches forecasts + observations from Open-Meteo
├── features.py         # Feature engineering (pure functions)
├── models.py           # LightGBM model classes (WeatherPredictor, PrecipitationPredictor)
├── train.py            # Training pipeline orchestrator
├── predict.py          # Generate predictions → latest_forecast.json
├── verify.py           # Daily verification of forecast accuracy
├── charts.py           # Matplotlib/Seaborn chart generation
├── serve.py            # Flask web server (dashboard + API)
│
├── models/             # Saved model files (.pkl) — not tracked in git
├── logs/               # Cron job logs — not tracked in git
├── weather_data.db     # SQLite database — not tracked in git
├── latest_forecast.json # Current forecast output — not tracked in git
│
├── pyproject.toml      # Project metadata + dependencies
├── crontab.txt         # Cron schedule for automation
└── README.md


### File Dependency Graph


config.py               ← no project imports
    ↑
db.py                   ← config
    ↑
collector.py            ← config, db
features.py             ← standalone (pure functions)
models.py               ← config
    ↑
train.py                ← config, db, features, models
predict.py              ← config, features, models
verify.py               ← db
charts.py               ← standalone (matplotlib/seaborn)
serve.py                ← config, charts

---

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/weather-predictor.git
cd weather-predictor

# Install dependencies
uv sync
```

### Configure Location

The project is preconfigured for Ruretse, Botswana. To verify or adjust, edit `config.py`:

```python
LATITUDE = -24.48       # Ruretse, Botswana
LONGITUDE = 25.98
TIMEZONE = "Africa/Gaborone"
LOCATION_NAME = "Ruretse, Botswana"
```

### Seed Historical Data (One-Time)

This downloads ~3 years of observations and historical NWP forecasts from Open-Meteo (free, no API key needed). Takes a few minutes.

```bash
uv run python collector.py --backfill
```

### Train Models

```bash
uv run python train.py
```

Expected output:


Loading paired forecast ↔ observation data...
Loaded XXXXX paired rows: 2022-XX-XX → 2025-XX-XX
Engineering features...
  58 features

═══ Training WeatherPredictor ═══
  temperature: val MAE = X.XX
  humidity:    val MAE = X.XX
  wind_speed:  val MAE = X.XX

═══ Training PrecipitationPredictor ═══
  Precip classification: Brier=0.XXXX, AUC=0.XXXX

═══ Validation by Lead Time ═══
  Lead  1h: Temp MAE ML=X.XX°C  NWP=X.XX°C  Δ=+XX.X%
  Lead 24h: Temp MAE ML=X.XX°C  NWP=X.XX°C  Δ=+XX.X%


### Generate Forecast

```bash
uv run python predict.py
```

### Start Dashboard

```bash
uv run python serve.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Web Dashboard & API

### Endpoints

| URL | Method | Format | Description |
|-----|--------|--------|-------------|
| `/` | GET | HTML | Charts dashboard — visual overview |
| `/table` | GET | HTML | Hourly data table grouped by day |
| `/chart/overview` | GET | PNG | Combined 4-panel chart |
| `/chart/temperature` | GET | PNG | Temperature detail chart |
| `/chart/precipitation` | GET | PNG | Precipitation detail chart |
| `/chart/wind` | GET | PNG | Wind detail chart |
| `/chart/humidity` | GET | PNG | Humidity detail chart |
| `/chart/daily_summary` | GET | PNG | Daily summary bar charts |
| `/forecast` | GET | JSON | Full hourly forecast (7 days) |
| `/forecast/today` | GET | JSON | Today's hours only |
| `/forecast/summary` | GET | JSON | Daily aggregated summaries |
| `/health` | GET | JSON | Status and data freshness |
| `/refresh` | POST | JSON | Trigger a new prediction run |

### JSON Response Example

```json
{
  "generated_at": "2025-07-11T12:00:00+00:00",
  "location": {
    "lat": -24.48,
    "lon": 25.98,
    "name": "Ruretse, Botswana"
  },
  "hourly": [
    {
      "time": "2025-07-11T13:00",
      "lead_hours": 1,
      "temperature_c": 22.4,
      "humidity_pct": 35,
      "wind_speed_kmh": 12.3,
      "precip_probability_pct": 5,
      "precip_expected_mm": 0.0
    }
  ]
}
```

### Access from Other Devices

The server binds to `0.0.0.0:5000`, so any device on your local network can access it:

```bash
# Find your machine's local IP
hostname -I | awk '{print $1}'

# Then open from phone/tablet/other PC:
# http://192.168.x.x:5000/
```

### Expose to the Internet (Free)

```bash
# Option 1: Cloudflare Tunnel (recommended — free, permanent)
sudo apt install cloudflared
cloudflared tunnel --url http://localhost:5000

# Option 2: ngrok (quick demo link)
ngrok http 5000
```

---

## Automation

### Cron Schedule

Install with `crontab crontab.txt` or `crontab -e`:

```cron
# Collect new forecasts + observations every 6 hours
0 */6 * * *   cd /path/to/weather-predictor && uv run python collector.py

# Generate predictions 5 minutes after collection
5 */6 * * *   cd /path/to/weather-predictor && uv run python predict.py

# Daily verification at 8am
0 8 * * *     cd /path/to/weather-predictor && uv run python verify.py

# Weekly retrain on Sunday at 2am
0 2 * * 0     cd /path/to/weather-predictor && uv run python train.py
```

### Systemd Service (Run Dashboard on Boot)

```bash
sudo nano /etc/systemd/system/weather.service
```

```ini
[Unit]
Description=Ruretse Weather Forecast Dashboard
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/weather-predictor
ExecStart=/path/to/weather-predictor/.venv/bin/python serve.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable weather
sudo systemctl start weather
```

---

## How It Works

### The MOS Approach

This project does **not** simulate atmospheric physics. Instead it uses **Model Output Statistics (MOS)** — the same technique used by professional meteorological services worldwide:

1. **Collect** free NWP model forecasts (what GFS/ICON predicted)
2. **Collect** actual observations (what really happened)
3. **Train** ML models to learn the systematic biases and errors
4. **Apply** those corrections to new forecasts

This is dramatically more efficient than running your own weather model, and on resource-limited hardware it's the only viable approach.

### Why LightGBM

| Factor | LightGBM | Deep Learning (LSTM/Transformer) |
|--------|----------|----------------------------------|
| Training time (1yr data) | 2–10 minutes | 2–8 hours |
| RAM needed | < 1 GB | 4–16 GB |
| GPU required | No | Practically yes |
| Tabular data performance | ★★★★★ | ★★★☆☆ |
| Handles missing data | Natively | Needs imputation |
| Model file size | 1–10 MB | 50–500 MB |

### Precipitation Handling

Rain is the hardest variable to predict. This project uses a **two-stage approach**:

1. **Stage 1 — Classification:** Will it rain? (LightGBM binary classifier)
2. **Stage 2 — Regression:** If yes, how much? (LightGBM with Tweedie loss — ideal for zero-inflated, right-skewed distributions)
3. **Calibration:** Probabilities are calibrated using isotonic regression
4. **Uncertainty:** Quantile regression models provide 10th–90th percentile prediction intervals

### Key Engineered Features

| Feature | Why It Matters |
|---------|---------------|
| NWP model forecast values | Strongest raw signal |
| Pressure tendency (3h, 6h, 12h, 24h) | Falling pressure → approaching rain |
| Dewpoint depression | Small gap → high precipitation potential |
| Wind U/V components | Better for ML than speed + direction |
| Wind component changes | Detect frontal passages |
| Cyclical time encodings (sin/cos) | Diurnal and seasonal patterns |
| Recent precipitation lags and rolling sums | Persistence signal |
| CAPE (instability) | Convective storm potential |
| NWP running bias | How wrong has the model been recently? |

---

## Data Sources

All data is free, no API keys required.

| Source | What | URL |
|--------|------|-----|
| Open-Meteo Forecast API | Current NWP forecasts (GFS, ICON, etc.) | [open-meteo.com](https://open-meteo.com/) |
| Open-Meteo Historical Forecast API | Past NWP predictions | [open-meteo.com](https://open-meteo.com/) |
| Open-Meteo Archive API | Historical observations (ERA5 reanalysis) | [open-meteo.com](https://open-meteo.com/) |

### Storage Requirements

- **Database:** ~50–100 MB for 3 years of hourly data
- **Models:** ~10–20 MB (all .pkl files combined)
- **Forecast output:** < 1 MB

---

## Local Context: Ruretse, Botswana

### Climate Characteristics

Ruretse is located in the semi-arid southeastern Botswana lowveld. Key weather patterns the model learns to handle:

- **Dry winters** (May–September): Clear skies, cold nights, warm days, minimal precipitation
- **Wet summers** (October–April): Convective afternoon thunderstorms, highly variable rainfall
- **Diurnal temperature range:** Large (often 15–20°C between night and day)
- **Dominant rain mechanism:** Convective — CAPE and dewpoint depression features are particularly important here
- **Wind patterns:** Light and variable in winter; gusty with afternoon thermals and storm outflows in summer

### Why MOS Works Well Here

Semi-arid regions with strong convective rainfall are exactly where NWP models struggle most — they systematically mishandle the timing and intensity of afternoon thunderstorms. The ML correction layer learns these local biases effectively, especially with a year or more of training data.

---

## Resource Requirements

### Minimum Hardware

| Component | Requirement |
|-----------|-------------|
| CPU | Any x86/ARM from the last decade |
| RAM | 512 MB (training), 100 MB (serving) |
| Storage | 500 MB total |
| Network | Internet access for API calls |
| GPU | Not needed |

### Tested On

- Old Dell laptop (Intel i5, 8 GB)

### Training Time

| Dataset Size | Time |
|-------------|------|
| 1 year (~8,760 rows) | ~2 minutes |
| 3 years (~26,000 rows) | ~5 minutes |
| 5 years (~44,000 rows) | ~10 minutes |

---

## Expected Accuracy

| Variable | 6h Forecast | 24h Forecast | 48h Forecast |
|----------|-------------|--------------|--------------|
| Temperature | ±0.8–1.2°C | ±1.2–2.0°C | ±1.5–2.5°C |
| Humidity | ±5–8% | ±8–12% | ±10–15% |
| Wind Speed | ±2–3 km/h | ±3–5 km/h | ±4–6 km/h |
| Precip Probability | Brier 0.08–0.12 | Brier 0.12–0.18 | Brier 0.15–0.22 |
| Improvement over raw NWP | 10–25% | 5–15% | 3–10% |

> Accuracy improves significantly with more training data. The single most impactful thing is consistent data collection over time.

---

## Development

### Run Tests

```bash
# Quick data check
uv run python -c "
import db
conn = db.get_conn()
obs = conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
fcst = conn.execute('SELECT COUNT(*) FROM forecasts').fetchone()[0]
print(f'Observations: {obs}')
print(f'Forecasts:    {fcst}')
conn.close()
"

# Check forecast output
uv run python -c "
import json
with open('latest_forecast.json') as f:
    data = json.load(f)
print(f'Hours: {len(data[\"hourly\"])}')
print(f'First: {data[\"hourly\"][0][\"time\"]}')
print(f'Last:  {data[\"hourly\"][-1][\"time\"]}')
"
```

### Verify Model Performance

```bash
uv run python verify.py
```

### Force Re-collect + Retrain + Predict

```bash
uv run python collector.py
uv run python train.py
uv run python predict.py
```

---

## License

MIT — use freely, modify freely, attribute if you're feeling generous.

---

## Acknowledgements

- **[Open-Meteo](https://open-meteo.com/)** — free weather API, no key required
- **[LightGBM](https://lightgbm.readthedocs.io/)** — fast gradient boosted trees
- **[Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/)** — chart generation
- MOS methodology as developed by the US National Weather Service

---

## Roadmap

- [ ] Add multiple NWP model comparison (GFS vs ICON spread as uncertainty feature)
- [ ] Lightning / severe weather indicators
- [ ] SMS / Telegram alerts for high-probability rain events
- [ ] Historical accuracy tracking dashboard
- [ ] Seasonal model variants (separate dry/wet season models)
- [ ] Integration with local rain gauges or personal weather stations


---

**Don't forget to update `config.py` with the correct Ruretse coordinates:**

```python
LATITUDE = -24.48
LONGITUDE = 25.98
TIMEZONE = "Africa/Gaborone"
LOCATION_NAME = "Ruretse, Botswana"
```