import pandas as pd
import numpy as np
import pulp
import os
import tempfile
from datetime import datetime, timedelta, date
import requests
from bs4 import BeautifulSoup
import io
import zipfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo

from functionlib import fetch_combined_aemo_data, run_optimization

DEFAULT_PUMPING_TARGET_ML = 40.0
SYD_TZ = ZoneInfo("Australia/Sydney")


def _as_date_list(dates_in_data):
    """
    Ensure dates_in_data is a list of python datetime.date.
    """
    out = []
    for d in dates_in_data:
        if isinstance(d, date):
            out.append(d)
        else:
            out.append(pd.to_datetime(d).date())
    return sorted(out)


def get_daily_targets(csv_path: str, dates_in_data) -> dict:
    """
    Build daily_targets dict keyed by 'YYYY-MM-DD'.

    New CSV format (recommended):
        day_offset,target_ml
        0,40
        1,42

    Meaning:
        day_offset=0 -> today in Australia/Sydney
        day_offset=1 -> tomorrow in Australia/Sydney

    Only offsets that match dates_in_data will be applied.
    """
    dates_in_data = _as_date_list(dates_in_data)

    # Default: for every date we have in price data, use DEFAULT_PUMPING_TARGET_ML
    daily_targets = {d.strftime("%Y-%m-%d"): float(DEFAULT_PUMPING_TARGET_ML) for d in dates_in_data}

    if not os.path.exists(csv_path):
        return daily_targets

    df = pd.read_csv(csv_path)

    # Support the new offset-based CSV
    required_cols = {"day_offset", "target_ml"}
    df_cols_lower = {c.lower(): c for c in df.columns}
    if required_cols.issubset(set(df_cols_lower.keys())):
        day_offset_col = df_cols_lower["day_offset"]
        target_ml_col = df_cols_lower["target_ml"]

        # Today based on Sydney timezone (important!)
        today_syd = datetime.now(SYD_TZ).date()

        # Parse + apply
        df[day_offset_col] = pd.to_numeric(df[day_offset_col], errors="coerce").astype("Int64")
        df[target_ml_col] = pd.to_numeric(df[target_ml_col], errors="coerce")

        for _, row in df.iterrows():
            if pd.isna(row[day_offset_col]) or pd.isna(row[target_ml_col]):
                continue
            offset = int(row[day_offset_col])
            target = float(row[target_ml_col])

            # Only allow 0/1 as you requested (but you can expand later)
            if offset not in (0, 1):
                continue

            d = today_syd + timedelta(days=offset)
            d_str = d.strftime("%Y-%m-%d")

            if d_str in daily_targets:
                daily_targets[d_str] = target

        return daily_targets

    # Backward compatibility: your old CSV format (Date, Target_ML)
    # (kept so you won't brick if someone uploads the old file by mistake)
    if "Date" in df.columns and "Target_ML" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d")
        df["Target_ML"] = df["Target_ML"].astype(float)

        for _, row in df.iterrows():
            d = row["Date"]
            if d in daily_targets:
                daily_targets[d] = float(row["Target_ML"])

        return daily_targets

    raise ValueError(
        "daily_targets.csv format wrong. Use either:\n"
        "1) day_offset,target_ml  (recommended)\n"
        "or\n"
        "2) Date,Target_ML        (legacy)\n"
    )


def main():
    logs = []

    def log_cb(msg):
        logs.append(str(msg))

    # Get AEMO prices
    df_price, now_naive = fetch_combined_aemo_data(log_cb)
    if df_price is None or len(df_price) == 0:
        raise RuntimeError("No price data")

    # Fixed parameters, temporarily
    params = {
        "daily_target": float(DEFAULT_PUMPING_TARGET_ML),  # default is 40
        "q_on": 1050.0,
        "p_on": 1950.0,
        "p_standby": 0.0,
        "min_on_hours": 4.0,
        "daily_min_hours": 6.0,
    }

    # Get pumping targets (today/tomorrow via day_offset)
    df_price["Date"] = pd.to_datetime(df_price["DateTime"]).dt.date
    dates_in_data = sorted(df_price["Date"].unique())
    daily_targets = get_daily_targets("daily_targets.csv", dates_in_data)

    # (Optional) log what targets were applied for transparency
    applied = {k: daily_targets[k] for k in sorted(daily_targets.keys())[:5]}
    log_cb(f"Daily targets (sample): {applied}")

    # Run optimisation
    result = run_optimization(df_price, params, daily_targets, log_cb)

    # Export optimised pumping schedule
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    result["df_price"].to_csv(f"{out_dir}/detailed_results.csv", index=False)
    with pd.ExcelWriter(f"{out_dir}/optimisation_result.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(result["schedule"]).to_excel(writer, sheet_name="Schedule", index=False)
        result["df_price"].to_excel(writer, sheet_name="Detailed", index=False)

    with open(f"{out_dir}/run_logs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    print("Pumping schedule has been saved to ./output/")


if __name__ == "__main__":
    main()
