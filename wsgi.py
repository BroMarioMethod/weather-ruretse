"""
WSGI entry point for Render / gunicorn.
Runs startup initialization, then serves the Flask app.
"""
import os
import sys

# Run startup if the forecast file doesn't exist yet
# (avoids re-running on gunicorn worker respawns)
from config import BASE_DIR

forecast_file = BASE_DIR / "latest_forecast.json"

if not forecast_file.exists():
    print("[wsgi] No forecast file found — running startup...")
    import startup
    startup.run()
else:
    print("[wsgi] Forecast file exists — skipping startup.")

# Start background refresh thread
from background import start_background_thread
start_background_thread()

# Import the Flask app for gunicorn
from serve import app

# Render sets PORT env var
port = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)