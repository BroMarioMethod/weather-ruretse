#!/bin/bash
# Generate forecast on startup and every 6 hours in background
python collector.py --backfill &
python predict.py

# Background refresh loop
while true; do
    sleep 21600  # 6 hours
    python collector.py
    python predict.py
done &

# Start web server
python serve.py
