"""
Generate forecast charts as base64-encoded PNGs.
Uses matplotlib + seaborn with a dark theme matching the dashboard.
"""
import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
import seaborn as sns


# ── Colors ─────────────────────────────────────────────
COLORS = {
    "bg": "#0f1923",
    "panel": "#1a2733",
    "grid": "#243447",
    "text": "#ffffff",
    "subtext": "#e0e0e0",
    "label": "#f0f0f0",
    "tick": "#d0d0d0",
    "title": "#4fc3f7",
    "temp": "#ffab40",
    "temp_band": "#ff6d00",
    "rain_bar": "#42a5f5",
    "rain_prob": "#90caf9",
    "wind": "#a5d6a7",
    "wind_gust": "#66bb6a",
    "humidity": "#ce93d8",
    "cloud": "#78909c",
    "accent": "#4fc3f7",
    "danger": "#ef5350",
    "warning": "#ffa726",
}


def _apply_dark_theme():
    """Set matplotlib rcParams with fully white text."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["title"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "text.color": COLORS["text"],
        "xtick.color": COLORS["tick"],
        "ytick.color": COLORS["tick"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 9,
        "legend.labelcolor": COLORS["text"],
        "font.family": "sans-serif",
        "font.size": 10,
    })


def _fig_to_base64(fig, dpi=130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_df(hourly: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    for col in df.columns:
        if col not in ("temperature_range_c", "precip_range_mm"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════
#  Individual Charts
# ═══════════════════════════════════════════════════════

def chart_temperature(df: pd.DataFrame) -> str:
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    times = df.index

    if "temperature_range_c" in df.columns:
        try:
            lows = [r[0] if isinstance(r, list) else np.nan for r in df["temperature_range_c"]]
            highs = [r[1] if isinstance(r, list) else np.nan for r in df["temperature_range_c"]]
            ax.fill_between(times, lows, highs,
                            alpha=0.15, color=COLORS["temp_band"], label="10th–90th percentile")
        except (TypeError, IndexError):
            pass

    ax.plot(times, df["temperature_c"], color=COLORS["temp"],
            linewidth=2, label="Temperature", zorder=5)

    for day in pd.date_range(times.min().normalize(), times.max().normalize(), freq="D"):
        night_start = day + pd.Timedelta(hours=20)
        night_end = day + pd.Timedelta(hours=32)
        ax.axvspan(night_start, night_end, alpha=0.08, color="white", zorder=0)

    if len(df) > 0:
        idx_max = df["temperature_c"].idxmax()
        idx_min = df["temperature_c"].idxmin()
        ax.annotate(f'{df["temperature_c"].max():.1f}°',
                    xy=(idx_max, df["temperature_c"].max()),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color=COLORS["danger"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["danger"], lw=0.8))
        ax.annotate(f'{df["temperature_c"].min():.1f}°',
                    xy=(idx_min, df["temperature_c"].min()),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color=COLORS["accent"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["accent"], lw=0.8))

    ax.set_ylabel("Temperature (°C)", color=COLORS["text"])
    ax.set_title("Temperature Forecast", fontsize=13, fontweight="bold",
                 color=COLORS["title"], pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=0, ha="center")

    return _fig_to_base64(fig)


def chart_precipitation(df: pd.DataFrame) -> str:
    _apply_dark_theme()
    fig, ax1 = plt.subplots(figsize=(14, 3.5))
    times = df.index
    width = pd.Timedelta(minutes=45)

    rain = df["precip_expected_mm"].fillna(0)
    bar_colors = [COLORS["rain_bar"] if r > 0.1 else COLORS["grid"] for r in rain]
    ax1.bar(times, rain, width=width, color=bar_colors, alpha=0.7,
            label="Expected rainfall", zorder=3)

    if "precip_range_mm" in df.columns:
        try:
            lows = [r[0] if isinstance(r, list) else 0 for r in df["precip_range_mm"]]
            highs = [r[1] if isinstance(r, list) else 0 for r in df["precip_range_mm"]]
            ax1.vlines(times, lows, highs, color=COLORS["rain_bar"],
                       alpha=0.3, linewidth=3, zorder=2)
        except (TypeError, IndexError):
            pass

    ax1.set_ylabel("Rainfall (mm)", color=COLORS["rain_bar"])
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis="y", colors=COLORS["rain_bar"])

    ax2 = ax1.twinx()
    prob = df["precip_probability_pct"].fillna(0)
    ax2.plot(times, prob, color=COLORS["rain_prob"], linewidth=1.5,
             alpha=0.9, label="Probability", zorder=4)
    ax2.fill_between(times, 0, prob, alpha=0.05, color=COLORS["rain_prob"])
    ax2.set_ylabel("Probability (%)", color=COLORS["rain_prob"])
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax2.tick_params(axis="y", colors=COLORS["rain_prob"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Precipitation Forecast", fontsize=13, fontweight="bold",
                  color=COLORS["title"], pad=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    fig.autofmt_xdate(rotation=0, ha="center")

    return _fig_to_base64(fig)


def chart_wind(df: pd.DataFrame) -> str:
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 3))
    times = df.index
    speed = df["wind_speed_kmh"].fillna(0)

    ax.fill_between(times, 0, speed, alpha=0.3, color=COLORS["wind"], zorder=2)
    ax.plot(times, speed, color=COLORS["wind"], linewidth=1.5,
            label="Wind speed", zorder=3)

    ax.axhline(y=20, color=COLORS["warning"], linewidth=0.8,
               linestyle="--", alpha=0.6, label="Breezy (20 km/h)")
    ax.axhline(y=40, color=COLORS["danger"], linewidth=0.8,
               linestyle="--", alpha=0.6, label="Strong (40 km/h)")

    ax.set_ylabel("Wind Speed (km/h)", color=COLORS["text"])
    ax.set_ylim(bottom=0)
    ax.set_title("Wind Forecast", fontsize=13, fontweight="bold",
                 color=COLORS["title"], pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=0, ha="center")

    return _fig_to_base64(fig)


def chart_humidity(df: pd.DataFrame) -> str:
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 2.5))
    times = df.index
    hum = df["humidity_pct"].fillna(0)

    ax.axhspan(30, 60, alpha=0.08, color=COLORS["wind"], zorder=0, label="Comfort zone")
    ax.fill_between(times, 0, hum, alpha=0.2, color=COLORS["humidity"], zorder=2)
    ax.plot(times, hum, color=COLORS["humidity"], linewidth=1.5, zorder=3)

    ax.set_ylabel("Humidity (%)", color=COLORS["text"])
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title("Humidity Forecast", fontsize=13, fontweight="bold",
                 color=COLORS["title"], pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=0, ha="center")

    return _fig_to_base64(fig)


def chart_daily_summary(df: pd.DataFrame) -> str:
    _apply_dark_theme()
    sns.set_theme(style="dark", rc={
        "axes.facecolor": COLORS["panel"],
        "figure.facecolor": COLORS["bg"],
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["tick"],
        "ytick.color": COLORS["tick"],
    })

    daily = df.resample("D").agg({
        "temperature_c": ["min", "max", "mean"],
        "precip_expected_mm": "sum",
        "precip_probability_pct": "max",
        "wind_speed_kmh": "max",
        "humidity_pct": "mean",
    })
    daily.columns = ["temp_min", "temp_max", "temp_mean",
                     "rain_total", "rain_prob_max", "wind_max", "humidity_mean"]
    daily = daily.dropna()

    if len(daily) == 0:
        fig, ax = plt.subplots(figsize=(14, 2))
        ax.text(0.5, 0.5, "No daily data available", ha="center", va="center",
                transform=ax.transAxes, color=COLORS["text"])
        return _fig_to_base64(fig)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2),
                              gridspec_kw={"wspace": 0.4})
    fig.suptitle("Daily Summary", fontsize=13, fontweight="bold",
                 color=COLORS["title"], y=1.02)

    day_labels = [d.strftime("%a\n%b %d") for d in daily.index]
    x = range(len(daily))

    # 1) Temperature range
    ax = axes[0]
    ax.bar(x, daily["temp_max"] - daily["temp_min"],
           bottom=daily["temp_min"], color=COLORS["temp"], alpha=0.7, width=0.6)
    for i, (lo, hi) in enumerate(zip(daily["temp_min"], daily["temp_max"])):
        ax.text(i, hi + 0.3, f"{hi:.0f}°", ha="center", fontsize=7,
                color=COLORS["text"], fontweight="bold")
        ax.text(i, lo - 1.2, f"{lo:.0f}°", ha="center", fontsize=7,
                color=COLORS["subtext"])
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontsize=7, color=COLORS["tick"])
    ax.set_title("Temp Range", fontsize=9, color=COLORS["text"])
    ax.set_ylabel("°C", fontsize=8, color=COLORS["text"])

    # 2) Rain total
    ax = axes[1]
    colors = [COLORS["rain_bar"] if r > 1 else COLORS["grid"] for r in daily["rain_total"]]
    ax.bar(x, daily["rain_total"], color=colors, alpha=0.8, width=0.6)
    for i, v in enumerate(daily["rain_total"]):
        if v > 0.1:
            ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=7,
                    color=COLORS["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontsize=7, color=COLORS["tick"])
    ax.set_title("Rain (mm)", fontsize=9, color=COLORS["text"])
    ax.set_ylim(bottom=0)

    # 3) Rain probability
    ax = axes[2]
    colors = [COLORS["danger"] if p >= 70 else COLORS["rain_prob"] if p >= 30
              else COLORS["grid"] for p in daily["rain_prob_max"]]
    ax.bar(x, daily["rain_prob_max"], color=colors, alpha=0.8, width=0.6)
    for i, v in enumerate(daily["rain_prob_max"]):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=7,
                color=COLORS["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontsize=7, color=COLORS["tick"])
    ax.set_title("Max P(Rain)", fontsize=9, color=COLORS["text"])
    ax.set_ylim(0, 110)

    # 4) Max wind
    ax = axes[3]
    colors = [COLORS["danger"] if w >= 40 else COLORS["warning"] if w >= 20
              else COLORS["wind"] for w in daily["wind_max"]]
    ax.bar(x, daily["wind_max"], color=colors, alpha=0.8, width=0.6)
    for i, v in enumerate(daily["wind_max"]):
        ax.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=7,
                color=COLORS["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontsize=7, color=COLORS["tick"])
    ax.set_title("Max Wind", fontsize=9, color=COLORS["text"])
    ax.set_ylabel("km/h", fontsize=8, color=COLORS["text"])
    ax.set_ylim(bottom=0)

    # 5) Mean humidity
    ax = axes[4]
    ax.bar(x, daily["humidity_mean"], color=COLORS["humidity"], alpha=0.6, width=0.6)
    for i, v in enumerate(daily["humidity_mean"]):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=7,
                color=COLORS["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontsize=7, color=COLORS["tick"])
    ax.set_title("Avg Humidity", fontsize=9, color=COLORS["text"])
    ax.set_ylim(0, 110)

    # Force white ticks on all panels
    for ax in axes:
        ax.tick_params(colors=COLORS["tick"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    return _fig_to_base64(fig)


def chart_combined_overview(df: pd.DataFrame) -> str:
    _apply_dark_theme()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.12, "height_ratios": [3, 3, 2, 2]}
    )
    times = df.index

    # Panel 1: Temperature
    ax1.plot(times, df["temperature_c"], color=COLORS["temp"], linewidth=2)
    if "temperature_range_c" in df.columns:
        try:
            lows = [r[0] if isinstance(r, list) else np.nan for r in df["temperature_range_c"]]
            highs = [r[1] if isinstance(r, list) else np.nan for r in df["temperature_range_c"]]
            ax1.fill_between(times, lows, highs, alpha=0.12, color=COLORS["temp_band"])
        except (TypeError, IndexError):
            pass
    ax1.set_ylabel("Temp (°C)", fontsize=9, color=COLORS["text"])
    ax1.set_title("7-Day Weather Overview", fontsize=14, fontweight="bold",
                  color=COLORS["title"], pad=15)

    # Panel 2: Precipitation
    rain = df["precip_expected_mm"].fillna(0)
    prob = df["precip_probability_pct"].fillna(0)
    ax2.bar(times, rain, width=pd.Timedelta(minutes=45),
            color=COLORS["rain_bar"], alpha=0.7, label="Rain (mm)")
    ax2.set_ylabel("Rain (mm)", fontsize=9, color=COLORS["text"])
    ax2.set_ylim(bottom=0)

    ax2b = ax2.twinx()
    ax2b.plot(times, prob, color=COLORS["rain_prob"], linewidth=1.2, alpha=0.8)
    ax2b.set_ylabel("Prob %", fontsize=8, color=COLORS["rain_prob"])
    ax2b.set_ylim(0, 105)
    ax2b.tick_params(axis="y", colors=COLORS["rain_prob"])

    # Panel 3: Wind
    ax3.fill_between(times, 0, df["wind_speed_kmh"].fillna(0),
                     alpha=0.3, color=COLORS["wind"])
    ax3.plot(times, df["wind_speed_kmh"], color=COLORS["wind"], linewidth=1.2)
    ax3.set_ylabel("Wind (km/h)", fontsize=9, color=COLORS["text"])
    ax3.set_ylim(bottom=0)

    # Panel 4: Humidity
    ax4.fill_between(times, 0, df["humidity_pct"].fillna(0),
                     alpha=0.2, color=COLORS["humidity"])
    ax4.plot(times, df["humidity_pct"], color=COLORS["humidity"], linewidth=1.2)
    ax4.set_ylabel("Humidity %", fontsize=9, color=COLORS["text"])
    ax4.set_ylim(0, 105)

    # Day separators + white ticks on all panels
    for ax in [ax1, ax2, ax3, ax4]:
        for day in pd.date_range(times.min().normalize(), times.max().normalize(), freq="D"):
            ax.axvline(day, color=COLORS["subtext"], linewidth=0.3, alpha=0.5)
        ax.tick_params(colors=COLORS["tick"])
        ax.yaxis.label.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
    ax4.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    return _fig_to_base64(fig)


# ═══════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════

def generate_all_charts(hourly: list[dict]) -> dict[str, str]:
    df = _build_df(hourly)
    return {
        "overview": chart_combined_overview(df),
        "temperature": chart_temperature(df),
        "precipitation": chart_precipitation(df),
        "wind": chart_wind(df),
        "humidity": chart_humidity(df),
        "daily_summary": chart_daily_summary(df),
    }


def generate_single_chart(hourly: list[dict], name: str) -> str | None:
    df = _build_df(hourly)
    chart_fns = {
        "overview": chart_combined_overview,
        "temperature": chart_temperature,
        "precipitation": chart_precipitation,
        "wind": chart_wind,
        "humidity": chart_humidity,
        "daily_summary": chart_daily_summary,
    }
    fn = chart_fns.get(name)
    if fn is None:
        return None
    return fn(df)