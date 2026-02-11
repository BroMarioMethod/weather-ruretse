import sqlite3
import pandas as pd
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS forecasts (
            fetched_at    TEXT    NOT NULL,
            valid_time    TEXT    NOT NULL,
            source        TEXT    NOT NULL,
            lead_hours    INTEGER,
            temperature_2m          REAL,
            dewpoint_2m             REAL,
            relative_humidity_2m    REAL,
            pressure_msl            REAL,
            surface_pressure        REAL,
            wind_speed_10m          REAL,
            wind_direction_10m      REAL,
            wind_gusts_10m          REAL,
            precipitation           REAL,
            precipitation_probability REAL,
            cloud_cover             REAL,
            cape                    REAL,
            visibility              REAL,
            PRIMARY KEY (fetched_at, valid_time, source)
        );

        CREATE TABLE IF NOT EXISTS observations (
            time              TEXT PRIMARY KEY,
            temperature_2m    REAL,
            dewpoint_2m       REAL,
            relative_humidity_2m REAL,
            pressure_msl      REAL,
            surface_pressure   REAL,
            wind_speed_10m    REAL,
            wind_direction_10m REAL,
            precipitation     REAL,
            cloud_cover       REAL
        );

        CREATE INDEX IF NOT EXISTS idx_fcst_valid
            ON forecasts(valid_time);
        CREATE INDEX IF NOT EXISTS idx_fcst_source
            ON forecasts(source, valid_time);
    """)
    conn.commit()
    conn.close()


def insert_forecasts(rows: list[dict], source: str, fetched_at: str) -> None:
    conn = get_conn()
    for row in rows:
        conn.execute("""
            INSERT OR REPLACE INTO forecasts
            (fetched_at, valid_time, source, lead_hours,
             temperature_2m, dewpoint_2m, relative_humidity_2m,
             pressure_msl, surface_pressure,
             wind_speed_10m, wind_direction_10m, wind_gusts_10m,
             precipitation, precipitation_probability,
             cloud_cover, cape, visibility)
            VALUES (?,?,?,?, ?,?,?, ?,?, ?,?,?, ?,?, ?,?,?)
        """, (
            fetched_at, row["time"], source, row.get("lead_hours"),
            row.get("temperature_2m"),
            row.get("dewpoint_2m"),
            row.get("relative_humidity_2m"),
            row.get("pressure_msl"),
            row.get("surface_pressure"),
            row.get("wind_speed_10m"),
            row.get("wind_direction_10m"),
            row.get("wind_gusts_10m"),
            row.get("precipitation"),
            row.get("precipitation_probability"),
            row.get("cloud_cover"),
            row.get("cape"),
            row.get("visibility"),
        ))
    conn.commit()
    conn.close()


def insert_observations(rows: list[dict]) -> None:
    conn = get_conn()
    for row in rows:
        conn.execute("""
            INSERT OR REPLACE INTO observations
            (time, temperature_2m, dewpoint_2m, relative_humidity_2m,
             pressure_msl, surface_pressure,
             wind_speed_10m, wind_direction_10m,
             precipitation, cloud_cover)
            VALUES (?,?,?,?, ?,?, ?,?, ?,?)
        """, (
            row["time"],
            row.get("temperature_2m"),
            row.get("dewpoint_2m"),
            row.get("relative_humidity_2m"),
            row.get("pressure_msl"),
            row.get("surface_pressure"),
            row.get("wind_speed_10m"),
            row.get("wind_direction_10m"),
            row.get("precipitation"),
            row.get("cloud_cover"),
        ))
    conn.commit()
    conn.close()


def load_observations(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    conn = get_conn()
    query = "SELECT * FROM observations"
    conditions = []
    params = []
    if start:
        conditions.append("time >= ?")
        params.append(start)
    if end:
        conditions.append("time <= ?")
        params.append(end)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY time"
    df = pd.read_sql(query, conn, params=params, parse_dates=["time"])
    df.set_index("time", inplace=True)
    conn.close()
    return df


def load_forecasts(source: str = "best_match",
                   start: str | None = None,
                   end: str | None = None) -> pd.DataFrame:
    conn = get_conn()
    query = "SELECT * FROM forecasts WHERE source = ?"
    params: list = [source]
    if start:
        query += " AND valid_time >= ?"
        params.append(start)
    if end:
        query += " AND valid_time <= ?"
        params.append(end)
    query += " ORDER BY valid_time, fetched_at"
    df = pd.read_sql(query, conn, params=params, parse_dates=["valid_time", "fetched_at"])
    conn.close()
    return df


def load_paired_data() -> pd.DataFrame:
    """Join most-recent forecast per valid_time with observations."""
    conn = get_conn()
    query = """
        WITH latest_fcst AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY valid_time, source
                       ORDER BY fetched_at DESC
                   ) AS rn
            FROM forecasts
            WHERE source = 'best_match'
        )
        SELECT
            o.time,
            -- observations (labels)
            o.temperature_2m    AS obs_temp,
            o.dewpoint_2m       AS obs_dewpoint,
            o.relative_humidity_2m AS obs_humidity,
            o.pressure_msl      AS obs_pressure,
            o.wind_speed_10m    AS obs_wind_speed,
            o.wind_direction_10m AS obs_wind_dir,
            o.precipitation     AS obs_precip,
            o.cloud_cover       AS obs_cloud,
            -- forecast features
            f.temperature_2m    AS fcst_temp,
            f.dewpoint_2m       AS fcst_dewpoint,
            f.relative_humidity_2m AS fcst_humidity,
            f.pressure_msl      AS fcst_pressure,
            f.surface_pressure  AS fcst_surface_pressure,
            f.wind_speed_10m    AS fcst_wind_speed,
            f.wind_direction_10m AS fcst_wind_dir,
            f.wind_gusts_10m    AS fcst_wind_gust,
            f.precipitation     AS fcst_precip,
            f.precipitation_probability AS fcst_precip_prob,
            f.cloud_cover       AS fcst_cloud,
            f.cape              AS fcst_cape,
            f.visibility        AS fcst_visibility,
            f.lead_hours        AS lead_hours
        FROM observations o
        INNER JOIN latest_fcst f
            ON f.valid_time = o.time AND f.rn = 1
        ORDER BY o.time
    """
    df = pd.read_sql(query, conn, parse_dates=["time"])
    df.set_index("time", inplace=True)
    conn.close()

    # ── Force all feature/target columns to numeric ──────────
    # SQLite stores everything as TEXT internally; when columns
    # contain NULLs mixed with numbers, pandas infers 'object'.
    numeric_cols = [c for c in df.columns if c != "time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df