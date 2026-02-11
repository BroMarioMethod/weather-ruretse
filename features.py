"""Feature engineering for weather prediction."""
import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encodings from the DatetimeIndex."""
    hour = df.index.hour
    doy = df.index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df.index.month
    return df


def add_derived_meteo(df: pd.DataFrame) -> pd.DataFrame:
    """Meteorologically meaningful derived features."""
    # Dewpoint depression — key rain predictor
    df["dewpoint_depression"] = df["fcst_temp"] - df["fcst_dewpoint"]

    # Wind components (better than speed + direction for ML)
    wind_rad = np.radians(df["fcst_wind_dir"])
    df["wind_u"] = -df["fcst_wind_speed"] * np.sin(wind_rad)
    df["wind_v"] = -df["fcst_wind_speed"] * np.cos(wind_rad)

    # Gust ratio
    df["gust_ratio"] = df["fcst_wind_gust"] / df["fcst_wind_speed"].replace(0, np.nan)
    df["gust_ratio"] = df["gust_ratio"].fillna(1.0)

    return df


def add_tendency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pressure / temperature / humidity tendencies (change over N hours)."""
    for var, col in [
        ("pressure", "fcst_pressure"),
        ("temp", "fcst_temp"),
        ("humidity", "fcst_humidity"),
    ]:
        for window in [3, 6, 12, 24]:
            df[f"{var}_tend_{window}h"] = df[col] - df[col].shift(window)

    # Wind component shifts (frontal passage signal)
    for comp in ["wind_u", "wind_v"]:
        for window in [3, 6]:
            df[f"{comp}_change_{window}h"] = df[comp] - df[comp].shift(window)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recent observed conditions as features."""
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"obs_precip_lag_{lag}h"] = df["obs_precip"].shift(lag)
        df[f"obs_temp_lag_{lag}h"] = df["obs_temp"].shift(lag)

    # Rolling precipitation sums
    for window in [6, 12, 24]:
        df[f"obs_precip_roll_{window}h_sum"] = (
            df["obs_precip"].rolling(window, min_periods=1).sum()
        )
        df[f"obs_precip_roll_{window}h_max"] = (
            df["obs_precip"].rolling(window, min_periods=1).max()
        )

    return df


def add_nwp_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    """Running NWP bias — how wrong has the model been recently?"""
    if "obs_temp" in df.columns and "fcst_temp" in df.columns:
        temp_error = df["obs_temp"] - df["fcst_temp"]
        df["nwp_temp_bias_24h"] = temp_error.rolling(24, min_periods=1).mean()
        df["nwp_temp_bias_72h"] = temp_error.rolling(72, min_periods=1).mean()

    if "obs_precip" in df.columns and "fcst_precip" in df.columns:
        precip_error = df["obs_precip"] - df["fcst_precip"]
        df["nwp_precip_bias_24h"] = precip_error.rolling(24, min_periods=1).mean()

    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    df = add_temporal_features(df)
    df = add_derived_meteo(df)
    df = add_tendency_features(df)
    df = add_lag_features(df)
    df = add_nwp_bias_features(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the column names that are valid ML features (not targets)."""
    exclude_prefixes = ("obs_",)  # observations are targets, not features
    exclude_exact = {"lead_hours"}  # keep lead_hours AS a feature actually

    # The obs lag/roll columns ARE features (they're historical, not future)
    include_even_if_obs = {c for c in df.columns if "lag_" in c or "roll_" in c or "bias_" in c}

    feature_cols = []
    for col in df.columns:
        if col in include_even_if_obs:
            feature_cols.append(col)
        elif any(col.startswith(p) for p in exclude_prefixes):
            continue
        elif col in exclude_exact:
            continue
        elif df[col].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            feature_cols.append(col)

    # Also include lead_hours — it's critical
    if "lead_hours" in df.columns and "lead_hours" not in feature_cols:
        feature_cols.append("lead_hours")

    return sorted(set(feature_cols))