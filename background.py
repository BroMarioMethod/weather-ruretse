"""
Background refresh thread for Render deployment.
Periodically collects new data and regenerates predictions.
"""
import threading
import time
import traceback


def _refresh_loop():
    """Run collector + predict every 6 hours."""
    # Wait for initial startup to finish
    time.sleep(60)

    while True:
        try:
            print("\n[background] Refreshing data...")
            import collector
            collector.main()  # fetch new forecast + recent observations

            print("[background] Regenerating predictions...")
            import predict
            predict.main()

            print("[background] Refresh complete.")
        except Exception:
            print(f"[background] ⚠ Refresh failed:")
            traceback.print_exc()

        # Sleep 6 hours
        time.sleep(6 * 60 * 60)


def start_background_thread():
    """Start the refresh loop in a daemon thread."""
    thread = threading.Thread(target=_refresh_loop, daemon=True, name="bg-refresh")
    thread.start()
    print("[background] Started refresh thread (every 6 hours)")