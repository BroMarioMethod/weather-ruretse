"""
Train all models on historical data.
Run manually or weekly via cron:  0 2 * * 0  uv run python train.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config
import db
from features import build_feature_set, get_feature_columns
from models import WeatherPredictor, PrecipitationPredictor


def main() -> None:
    db.init_db()

    print("Loading paired forecast ↔ observation data...")
    df = db.load_paired_data()

    if len(df) < 500:
        print(f"Only {len(df)} paired rows. Need at least 500. Run collector.py --backfill first.")
        sys.exit(1)

    print(f"Loaded {len(df)} paired rows: {df.index.min()} → {df.index.max()}")

    # Feature engineering
    print("Engineering features...")
    df = build_feature_set(df)
    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} features")

    # Drop rows where all features are NaN (first few rows due to lags)
    df = df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2)
    print(f"  {len(df)} rows after dropping sparse rows")

    # Time-based split: last 20% for validation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"  Train: {len(train_df)} rows ({train_df.index.min()} → {train_df.index.max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df.index.min()} → {val_df.index.max()})")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    # ── Train weather predictor (temp, humidity, wind) ──
    print("\n═══ Training WeatherPredictor ═══")
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    wp = WeatherPredictor()
    wp.train(X_train, train_df[obs_cols], X_val, val_df[obs_cols])
    wp.save(config.MODEL_PATH)
    print(f"  Saved → {config.MODEL_PATH}")

    # ── Train precipitation predictor ──
    print("\n═══ Training PrecipitationPredictor ═══")
    pp = PrecipitationPredictor()

    # Align on non-null precip observations
    precip_train = train_df["obs_precip"].dropna()
    precip_val = val_df["obs_precip"].dropna()
    common_train = precip_train.index.intersection(X_train.index)
    common_val = precip_val.index.intersection(X_val.index)

    pp.train(
        X_train.loc[common_train],
        precip_train.loc[common_train],
        X_val.loc[common_val],
        precip_val.loc[common_val],
    )
    pp.save(config.PRECIP_MODEL_PATH)
    print(f"  Saved → {config.PRECIP_MODEL_PATH}")

    # ── Summary by lead time ──
    print("\n═══ Validation by Lead Time ═══")
    if "lead_hours" in val_df.columns:
        wp_loaded = WeatherPredictor()
        wp_loaded.load(config.MODEL_PATH)
        preds = wp_loaded.predict(X_val)

        for lead in [1, 3, 6, 12, 24, 48]:
            mask = val_df["lead_hours"] == lead
            n = mask.sum()
            if n < 10:
                continue
            temp_mae = np.mean(np.abs(
                preds["temperature"][mask.values] - val_df.loc[mask, "obs_temp"].values
            ))
            nwp_mae = np.mean(np.abs(
                val_df.loc[mask, "fcst_temp"].values - val_df.loc[mask, "obs_temp"].values
            ))
            improvement = (1 - temp_mae / nwp_mae) * 100 if nwp_mae > 0 else 0
            print(f"  Lead {lead:2d}h (n={n:4d}): "
                  f"Temp MAE ML={temp_mae:.2f}°C  NWP={nwp_mae:.2f}°C  "
                  f"Δ={improvement:+.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()