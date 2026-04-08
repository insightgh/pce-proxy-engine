"""
seasonal.py
Builds and applies residual seasonal adjustment (RSA) factors using LOESS decomposition.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def build_seasonal_factors(actual_pce_df, proxy_estimates_df, min_observations=3):
    actual = actual_pce_df[["date", "mom_pct"]].copy()
    actual["date"] = actual["date"].dt.to_period("M").dt.to_timestamp()
    actual = actual.rename(columns={"mom_pct": "actual_mom"})

    proxy = proxy_estimates_df[["date", "proxy_mom_pct"]].copy()
    proxy["date"] = proxy["date"].dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(actual, proxy, on="date", how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between actual PCE and proxy.")

    # Calculate raw error
    merged["error"] = merged["actual_mom"] - merged["proxy_mom_pct"]

    # Ensure chronological order and proper monthly frequency for statsmodels
    merged = merged.sort_values("date").set_index("date").asfreq("MS")

    # --- THE PANDEMIC EXCLUSION UPGRADE ---
    # Mask out the massive COVID-19 anomalies (March 2020 - Dec 2021) with NaNs
    pandemic_mask = (merged.index >= '2020-03-01') & (merged.index <= '2021-12-01')
    merged.loc[pandemic_mask, 'error'] = np.nan

    # Linearly interpolate across the pandemic gap so the LOESS algorithm
    # doesn't break, while completely ignoring the massive lockdown outliers.
    filled_error = merged["error"].interpolate(method='linear').bfill().ffill()

    # Apply Classical Decomposition to extract pure seasonality, filtering out random shocks
    decomp = seasonal_decompose(filled_error, model="additive", period=12, extrapolate_trend='freq')

    merged["seasonal_factor"] = decomp.seasonal
    merged["calendar_month"]  = merged.index.month

    # Group by calendar month to establish the stable RSA factor
    rsa = merged.groupby("calendar_month")["seasonal_factor"].mean()
    return rsa.reindex(range(1, 13), fill_value=0.0)


def apply_seasonal_adjustment(raw_proxy_mom, target_month, rsa_factors):
    factor   = rsa_factors.get(target_month, 0.0)
    adjusted = raw_proxy_mom + factor
    return {
        "raw_proxy_mom": round(raw_proxy_mom, 4),
        "rsa_factor":    round(factor, 4),
        "adjusted_mom":  round(adjusted, 4),
        "target_month":  target_month,
    }


def backtest_accuracy(actual_pce_df, proxy_estimates_df, rsa_factors=None, expanding_window=False):
    actual = actual_pce_df[["date", "mom_pct"]].copy()
    actual["date"] = actual["date"].dt.to_period("M").dt.to_timestamp()
    actual = actual.rename(columns={"mom_pct": "actual_mom"})

    proxy = proxy_estimates_df[["date", "proxy_mom_pct"]].copy()
    proxy["date"] = proxy["date"].dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(actual, proxy, on="date", how="inner")
    merged["calendar_month"] = merged["date"].dt.month

    if expanding_window and len(merged) >= 24:
        # Out-of-sample RSA: for each month, only use data available before it
        merged = merged.sort_values("date").reset_index(drop=True)
        rsa_col = []
        for i in range(len(merged)):
            if i < 24:
                rsa_col.append(0.0)
                continue
            hist_actual = actual_pce_df[actual_pce_df["date"].dt.to_period("M").dt.to_timestamp() < merged.loc[i, "date"]]
            hist_proxy = proxy_estimates_df[proxy_estimates_df["date"].dt.to_period("M").dt.to_timestamp() < merged.loc[i, "date"]]
            try:
                hist_rsa = build_seasonal_factors(hist_actual, hist_proxy)
                rsa_col.append(hist_rsa.get(merged.loc[i, "calendar_month"], 0.0))
            except Exception:
                rsa_col.append(0.0)
        merged["rsa_factor"] = rsa_col
        merged["adjusted_proxy"] = merged["proxy_mom_pct"] + merged["rsa_factor"]
    elif rsa_factors is not None:
        merged["rsa_factor"]     = merged["calendar_month"].map(rsa_factors)
        merged["adjusted_proxy"] = merged["proxy_mom_pct"] + merged["rsa_factor"]
    else:
        merged["rsa_factor"]     = 0.0
        merged["adjusted_proxy"] = merged["proxy_mom_pct"]

    merged["raw_error"]      = merged["actual_mom"] - merged["proxy_mom_pct"]
    merged["adjusted_error"] = merged["actual_mom"] - merged["adjusted_proxy"]

    cols = ["date","actual_mom","proxy_mom_pct","rsa_factor",
            "adjusted_proxy","raw_error","adjusted_error"]
    return merged[cols].sort_values("date").reset_index(drop=True)


def print_accuracy_summary(backtest_df):
    # Standard calculation (Including the COVID misses)
    raw_mae  = backtest_df["raw_error"].abs().mean()
    adj_mae  = backtest_df["adjusted_error"].abs().mean()
    raw_rmse = np.sqrt((backtest_df["raw_error"] ** 2).mean())
    adj_rmse = np.sqrt((backtest_df["adjusted_error"] ** 2).mean())

    # Ex-COVID calculation (The true metric of the model's accuracy)
    clean_df = backtest_df[~((backtest_df["date"] >= '2020-03-01') & (backtest_df["date"] <= '2021-12-01'))]
    clean_raw_mae = clean_df["raw_error"].abs().mean()
    clean_adj_mae = clean_df["adjusted_error"].abs().mean()

    print("=== Backtest Accuracy Summary ===")
    print(f"  Observations : {len(backtest_df)} (Includes {len(backtest_df) - len(clean_df)} Pandemic Months)")
    print(f"  Raw Proxy (All-Time) — MAE: {raw_mae:.4f} pp  |  RMSE: {raw_rmse:.4f} pp")
    print(f"  Adj Proxy (All-Time) — MAE: {adj_mae:.4f} pp  |  RMSE: {adj_rmse:.4f} pp")
    print(f"  --------------------------------------------------")
    print(f"  Raw Proxy (Ex-COVID) — MAE: {clean_raw_mae:.4f} pp")
    print(f"  Adj Proxy (Ex-COVID) — MAE: {clean_adj_mae:.4f} pp")
    print(f"  RSA improvement      : {clean_raw_mae - clean_adj_mae:+.4f} pp MAE\n")