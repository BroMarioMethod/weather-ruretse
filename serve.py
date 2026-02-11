"""
Serve predictions via HTTP with HTML dashboard + visual charts.
Start:  uv run python serve.py
"""
import json
import subprocess
import sys
from base64 import b64decode
from collections import OrderedDict
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template_string

from config import BASE_DIR, LOCATION_NAME

app = Flask(__name__)

FORECAST_FILE = BASE_DIR / "latest_forecast.json"


def load_forecast() -> dict | None:
    if not FORECAST_FILE.exists():
        return None
    with open(FORECAST_FILE) as f:
        return json.load(f)


def rain_class(prob_pct: float) -> str:
    if prob_pct >= 60:
        return "rain-high"
    elif prob_pct >= 30:
        return "rain-med"
    return "rain-low"


# ═══════════════════════════════════════════════════════
#  HTML Templates
# ═══════════════════════════════════════════════════════

LAYOUT_HEAD = """
<!DOCTYPE html>
<html>
<head>
    <title>Weather Forecast — {{ location }}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: #0f1923;
            color: #e0e0e0;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 { color: #4fc3f7; margin-bottom: 5px; }
        .meta { color: #888; font-size: 0.85em; margin-bottom: 15px; }
        nav {
            display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;
        }
        nav a {
            background: #1a2733; color: #4fc3f7; padding: 8px 16px;
            border-radius: 6px; text-decoration: none; font-size: 0.9em;
            border: 1px solid #243447; transition: background 0.2s;
        }
        nav a:hover, nav a.active {
            background: #243447; border-color: #4fc3f7;
        }
        .day-group {
            margin-bottom: 25px; background: #1a2733;
            border-radius: 8px; overflow: hidden;
        }
        .day-header {
            background: #243447; padding: 10px 15px;
            font-weight: bold; color: #4fc3f7; font-size: 1.1em;
        }
        table { width: 100%; border-collapse: collapse; }
        th {
            background: #1e2d3d; padding: 8px 10px; text-align: right;
            font-size: 0.8em; color: #90a4ae; position: sticky; top: 0;
        }
        th:first-child { text-align: left; }
        td {
            padding: 6px 10px; text-align: right;
            border-bottom: 1px solid #1e2d3d; font-size: 0.9em;
        }
        td:first-child { text-align: left; color: #b0bec5; }
        tr:hover { background: #243447; }
        .rain-high { color: #42a5f5; font-weight: bold; }
        .rain-med  { color: #90caf9; }
        .rain-low  { color: #546e7a; }
        .temp { color: #ffab40; }
        .wind { color: #a5d6a7; }
        .summary-bar {
            display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px;
        }
        .summary-card {
            background: #1a2733; border-radius: 8px;
            padding: 15px 20px; flex: 1; min-width: 140px;
        }
        .summary-card .label { color: #90a4ae; font-size: 0.8em; }
        .summary-card .value {
            font-size: 1.4em; font-weight: bold; margin-top: 5px;
        }
        .chart-container {
            background: #1a2733; border-radius: 8px;
            padding: 10px; margin-bottom: 20px; text-align: center;
        }
        .chart-container img {
            max-width: 100%; height: auto; border-radius: 4px;
        }
        .chart-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 15px; margin-bottom: 20px;
        }
        @media (max-width: 900px) {
            .chart-grid { grid-template-columns: 1fr; }
        }
        .endpoints {
            margin-top: 30px; padding: 15px; background: #1a2733;
            border-radius: 8px; font-size: 0.85em;
        }
        .endpoints a { color: #4fc3f7; }
        .endpoints code {
            background: #243447; padding: 2px 6px; border-radius: 3px;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
"""

CHARTS_TEMPLATE = LAYOUT_HEAD + """
    <h1>🌦 {{ location }} Forecast</h1>
    <div class="meta">
        Generated: {{ generated_at }} &nbsp;|&nbsp;
        {{ total_hours }} hours &nbsp;|&nbsp; {{ num_days }} days
    </div>

    <nav>
        <a href="/" class="{{ 'active' if page == 'charts' else '' }}">📊 Charts</a>
        <a href="/table" class="{{ 'active' if page == 'table' else '' }}">📋 Table</a>
        <a href="/forecast/summary">📅 Daily JSON</a>
        <a href="/forecast">🔧 Full JSON</a>
    </nav>

    <div class="summary-bar">
        <div class="summary-card">
            <div class="label">Temperature Range</div>
            <div class="value temp">{{ temp_min }}° – {{ temp_max }}°C</div>
        </div>
        <div class="summary-card">
            <div class="label">Max Rain Probability</div>
            <div class="value rain-high">{{ max_rain_prob }}%</div>
        </div>
        <div class="summary-card">
            <div class="label">Total Expected Rain</div>
            <div class="value rain-med">{{ total_rain_mm }} mm</div>
        </div>
        <div class="summary-card">
            <div class="label">Max Wind</div>
            <div class="value wind">{{ max_wind }} km/h</div>
        </div>
    </div>

    <!-- Overview chart (full width) -->
    <div class="chart-container">
        <img src="data:image/png;base64,{{ charts.overview }}" alt="Overview">
    </div>

    <!-- Daily summary bar charts -->
    <div class="chart-container">
        <img src="data:image/png;base64,{{ charts.daily_summary }}" alt="Daily Summary">
    </div>

    <!-- Detail charts (2-column grid) -->
    <div class="chart-grid">
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.temperature }}" alt="Temperature">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.precipitation }}" alt="Precipitation">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.wind }}" alt="Wind">
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.humidity }}" alt="Humidity">
        </div>
    </div>

    <div class="endpoints">
        <strong>All Endpoints:</strong><br>
        <code>GET</code> <a href="/">/</a> — charts dashboard<br>
        <code>GET</code> <a href="/table">/table</a> — hourly data table<br>
        <code>GET</code> <a href="/chart/overview">/chart/overview</a> — overview chart (PNG)<br>
        <code>GET</code> <a href="/chart/temperature">/chart/temperature</a> — temperature (PNG)<br>
        <code>GET</code> <a href="/chart/precipitation">/chart/precipitation</a> — precipitation (PNG)<br>
        <code>GET</code> <a href="/chart/wind">/chart/wind</a> — wind (PNG)<br>
        <code>GET</code> <a href="/chart/humidity">/chart/humidity</a> — humidity (PNG)<br>
        <code>GET</code> <a href="/chart/daily_summary">/chart/daily_summary</a> — daily summary (PNG)<br>
        <code>GET</code> <a href="/forecast">/forecast</a> — full JSON<br>
        <code>GET</code> <a href="/forecast/today">/forecast/today</a> — today JSON<br>
        <code>GET</code> <a href="/forecast/summary">/forecast/summary</a> — daily summaries JSON<br>
        <code>POST</code> <code>/refresh</code> — re-run predict.py
    </div>
</body>
</html>
"""

TABLE_TEMPLATE = LAYOUT_HEAD + """
    <h1>🌦 {{ location }} Forecast</h1>
    <div class="meta">
        Generated: {{ generated_at }} &nbsp;|&nbsp;
        {{ total_hours }} hours &nbsp;|&nbsp; {{ num_days }} days
    </div>

    <nav>
        <a href="/" class="{{ 'active' if page == 'charts' else '' }}">📊 Charts</a>
        <a href="/table" class="{{ 'active' if page == 'table' else '' }}">📋 Table</a>
        <a href="/forecast/summary">📅 Daily JSON</a>
        <a href="/forecast">🔧 Full JSON</a>
    </nav>

    <div class="summary-bar">
        <div class="summary-card">
            <div class="label">Temperature Range</div>
            <div class="value temp">{{ temp_min }}° – {{ temp_max }}°C</div>
        </div>
        <div class="summary-card">
            <div class="label">Max Rain Probability</div>
            <div class="value rain-high">{{ max_rain_prob }}%</div>
        </div>
        <div class="summary-card">
            <div class="label">Total Expected Rain</div>
            <div class="value rain-med">{{ total_rain_mm }} mm</div>
        </div>
        <div class="summary-card">
            <div class="label">Max Wind</div>
            <div class="value wind">{{ max_wind }} km/h</div>
        </div>
    </div>

    {% for day_name, hours in days %}
    <div class="day-group">
        <div class="day-header">{{ day_name }}</div>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Temp °C</th>
                    <th>Humidity</th>
                    <th>Wind km/h</th>
                    <th>P(Rain)</th>
                    <th>Rain mm</th>
                </tr>
            </thead>
            <tbody>
                {% for h in hours %}
                <tr>
                    <td>{{ h.hour }}</td>
                    <td class="temp">{{ h.temperature_c }}</td>
                    <td>{{ h.humidity_pct|int }}%</td>
                    <td class="wind">{{ h.wind_speed_kmh }}</td>
                    <td class="{{ h.rain_class }}">{{ h.precip_probability_pct|int }}%</td>
                    <td class="{{ h.rain_class }}">{{ h.precip_expected_mm }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endfor %}
</body>
</html>
"""


# ═══════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════

def _summary_vars(data: dict) -> dict:
    """Compute summary stats for templates."""
    hourly = data["hourly"]
    temps = [h["temperature_c"] for h in hourly
             if isinstance(h.get("temperature_c"), (int, float))]
    probs = [h.get("precip_probability_pct", 0) for h in hourly]
    rains = [h.get("precip_expected_mm", 0) for h in hourly]
    winds = [h.get("wind_speed_kmh", 0) for h in hourly
             if isinstance(h.get("wind_speed_kmh"), (int, float))]

    # Group by day for table view
    days = OrderedDict()
    for h in hourly:
        dt = datetime.fromisoformat(h["time"])
        day_key = dt.strftime("%A, %B %d")
        if day_key not in days:
            days[day_key] = []
        days[day_key].append({
            "hour": dt.strftime("%H:%M"),
            "temperature_c": h.get("temperature_c", "—"),
            "humidity_pct": h.get("humidity_pct", 0),
            "wind_speed_kmh": h.get("wind_speed_kmh", "—"),
            "precip_probability_pct": h.get("precip_probability_pct", 0),
            "precip_expected_mm": h.get("precip_expected_mm", 0),
            "rain_class": rain_class(h.get("precip_probability_pct", 0)),
        })

    return {
        "location": data.get("location", {}).get("name", LOCATION_NAME),
        "generated_at": data.get("generated_at", "unknown"),
        "total_hours": len(hourly),
        "num_days": len(days),
        "temp_min": round(min(temps), 1) if temps else "—",
        "temp_max": round(max(temps), 1) if temps else "—",
        "max_rain_prob": round(max(probs), 0) if probs else 0,
        "total_rain_mm": round(sum(rains), 1) if rains else 0,
        "max_wind": round(max(winds), 1) if winds else "—",
        "days": list(days.items()),
    }


# ═══════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════

@app.route("/")
def index():
    """Charts dashboard (default landing page)."""
    data = load_forecast()
    if not data:
        return ("<h1 style='color:#e0e0e0;background:#0f1923;padding:40px'>"
                "No forecast available.</h1>"
                "<p style='color:#888;background:#0f1923;padding:0 40px'>"
                "Run: <code>uv run python predict.py</code></p>"), 404

    # Import here so matplotlib isn't loaded until needed
    from charts import generate_all_charts
    charts = generate_all_charts(data["hourly"])

    ctx = _summary_vars(data)
    ctx["charts"] = charts
    ctx["page"] = "charts"

    return render_template_string(CHARTS_TEMPLATE, **ctx)


@app.route("/table")
def table_view():
    """Hourly data table (the original view)."""
    data = load_forecast()
    if not data:
        return "<h1>No forecast available.</h1>", 404

    ctx = _summary_vars(data)
    ctx["page"] = "table"

    return render_template_string(TABLE_TEMPLATE, **ctx)


@app.route("/chart/<name>")
def single_chart(name: str):
    """Serve a single chart as a downloadable PNG image."""
    data = load_forecast()
    if not data:
        return "No forecast data", 404

    from charts import generate_single_chart
    b64 = generate_single_chart(data["hourly"], name)
    if b64 is None:
        valid = ["overview", "temperature", "precipitation",
                 "wind", "humidity", "daily_summary"]
        return jsonify({"error": f"Unknown chart: {name}",
                        "valid_names": valid}), 404

    png_bytes = b64decode(b64)
    return Response(png_bytes, mimetype="image/png",
                    headers={"Content-Disposition": f"inline; filename={name}.png"})


@app.route("/forecast")
def forecast_json():
    """Full JSON forecast."""
    data = load_forecast()
    if not data:
        return jsonify({"error": "No forecast available."}), 404
    return jsonify(data)


@app.route("/forecast/today")
def forecast_today():
    """Today's hourly forecast only."""
    data = load_forecast()
    if not data:
        return jsonify({"error": "No forecast available."}), 404
    today = datetime.now().strftime("%Y-%m-%d")
    today_hours = [h for h in data["hourly"] if h["time"].startswith(today)]
    return jsonify({"date": today, "location": data.get("location"),
                    "hourly": today_hours})


@app.route("/forecast/summary")
def forecast_summary():
    """Daily summaries."""
    data = load_forecast()
    if not data:
        return jsonify({"error": "No forecast available."}), 404

    days_agg = OrderedDict()
    for h in data["hourly"]:
        day = h["time"][:10]
        if day not in days_agg:
            days_agg[day] = {"temps": [], "rain": [], "wind": [], "probs": []}
        days_agg[day]["temps"].append(h.get("temperature_c", 0))
        days_agg[day]["rain"].append(h.get("precip_expected_mm", 0))
        days_agg[day]["wind"].append(h.get("wind_speed_kmh", 0))
        days_agg[day]["probs"].append(h.get("precip_probability_pct", 0))

    summaries = []
    for day, vals in days_agg.items():
        summaries.append({
            "date": day,
            "temp_min_c": round(min(vals["temps"]), 1),
            "temp_max_c": round(max(vals["temps"]), 1),
            "total_rain_mm": round(sum(vals["rain"]), 2),
            "max_rain_probability_pct": round(max(vals["probs"]), 0),
            "max_wind_kmh": round(max(vals["wind"]), 1),
        })

    return jsonify({"location": data.get("location"), "daily": summaries})


@app.route("/health")
def health():
    data = load_forecast()
    status = {"status": "ok" if data else "no_data",
              "forecast_file_exists": FORECAST_FILE.exists()}
    if data:
        status["generated_at"] = data.get("generated_at")
        status["hours_in_forecast"] = len(data.get("hourly", []))
    return jsonify(status)


@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        result = subprocess.run(
            [sys.executable, "predict.py"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=60,
        )
        return jsonify({"status": "ok" if result.returncode == 0 else "error",
                        "stdout": result.stdout[-500:],
                        "stderr": result.stderr[-500:]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Serving forecasts for {LOCATION_NAME}")
    print(f"  📊 Dashboard:  http://localhost:5000/")
    print(f"  📋 Table:      http://localhost:5000/table")
    print(f"  🖼  Charts:     http://localhost:5000/chart/overview")
    print(f"  🔧 JSON API:   http://localhost:5000/forecast")
    print(f"  📅 Summaries:  http://localhost:5000/forecast/summary")
    print(f"  ❤️  Health:     http://localhost:5000/health")
    app.run(host="0.0.0.0", port=5000, debug=True)