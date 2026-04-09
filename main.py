"""
main.py — PCE Proxy Engine

Modes:
  python main.py                        ->  Live Headline PCE estimate
  python main.py --core                 ->  Live Core PCE estimate
  python main.py --backtest             ->  Historical accuracy (3 years default)
  python main.py --backtest --core      ->  Historical Core PCE accuracy
  python main.py --decompose 2026-01    ->  Component breakdown for a month
  python main.py --weights              ->  Dynamic vs static weight comparison
  python main.py --backtest --verbose   ->  Run backtest with component count diagnostics
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from src.crosswalk import load_crosswalk, get_series_by_type, CONTRIBUTION_CAP
from src.api_clients import (
    fetch_bls_data, fetch_import_price_data, fetch_ppi_data,
    fetch_fred_data, fetch_bea_pce_weights,
)
from src.seasonal import (
    build_seasonal_factors, apply_seasonal_adjustment,
    backtest_accuracy, print_accuracy_summary,
)
from src.weights import (
    build_dynamic_weights, weight_coverage_report,
)


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------

def load_weights(crosswalk, start_year, is_core=False):
    bea_key = os.getenv("BEA_API_KEY")
    if not bea_key:
        print("  [Weights] No BEA_API_KEY → static weights. "
              "Register free at apps.bea.gov/API/signup")
        return None, False
    try:
        bea_df = fetch_bea_pce_weights(start_year=start_year)
        dyn_w  = build_dynamic_weights(bea_df, crosswalk, is_core=is_core)
        weight_coverage_report(dyn_w, crosswalk)
        return dyn_w, True
    except Exception as e:
        print(f"  [Warning] BEA weights failed ({e}) → static fallback.")
        return None, False


# ---------------------------------------------------------------------------
# Core proxy builder (Törnqvist Log-Aggregation)
# ---------------------------------------------------------------------------

OER_DAMPENING = 0.85  # BEA smooths OER more than CPI; scale CPI OER by this factor
OER_SERIES_ID = 'CUSR0000SEHC'
ADAPTIVE_CAP_ZSCORE = 2.5  # Cap at 2.5 sigma of trailing contribution volatility
ADAPTIVE_CAP_LOOKBACK = 24  # Months of history for rolling volatility


def build_proxy_series(crosswalk, start_year, end_year, dynamic_weights=None):
    all_mom = []

    cpi_ids = get_series_by_type(crosswalk, 'CPI')
    if cpi_ids:
        all_mom.append(_compute_mom(fetch_bls_data(cpi_ids, start_year, end_year)))

    import_ids = get_series_by_type(crosswalk, 'IMPORT')
    if import_ids:
        all_mom.append(_compute_mom(fetch_import_price_data(import_ids, start_year, end_year)))

    ppi_ids = get_series_by_type(crosswalk, 'PPI')
    if ppi_ids:
        all_mom.append(_compute_mom(fetch_ppi_data(ppi_ids, start_year, end_year)))

    fred_ids = get_series_by_type(crosswalk, 'FRED')
    if fred_ids:
        for fid in fred_ids:
            fred_df = fetch_fred_data(fid, observation_start=f"{start_year}-01-01")
            fred_df['series_id'] = fid
            all_mom.append(fred_df[['date', 'series_id', 'mom_pct']])

    if not all_mom:
        raise ValueError("No data fetched.")

    combined = pd.concat(all_mom, ignore_index=True)

    # Housing OER-to-PCE dampening: BEA smooths OER more aggressively than CPI
    oer_mask = combined['series_id'] == OER_SERIES_ID
    combined.loc[oer_mask, 'mom_pct'] = combined.loc[oer_mask, 'mom_pct'] * OER_DAMPENING

    merged = pd.merge(
        combined,
        crosswalk[['series_id','source_type','pce_weight','cap_contribution','target_pce_name']],
        on='series_id', how='inner'
    )

    # Apply dynamic weights
    if dynamic_weights is not None:
        dw = dynamic_weights[['date', 'series_id', 'dynamic_weight']].copy()
        merged = pd.merge(merged, dw, on=['date', 'series_id'], how='left')
        merged['effective_weight'] = merged['dynamic_weight'].fillna(merged['pce_weight'])
        merged.drop(columns=['dynamic_weight'], inplace=True)
    else:
        merged['effective_weight'] = merged['pce_weight']

    # Törnqvist Geometric Aggregation
    merged['log_change'] = np.log1p(merged['mom_pct'])
    merged['weighted_log_contrib'] = merged['log_change'] * merged['effective_weight']

    # --- The "Ragged Edge" Completeness Gate ---
    expected_components = len(crosswalk[crosswalk['active'] == 1])
    component_counts = merged.groupby('date')['series_id'].nunique()
    valid_dates = component_counts[component_counts == expected_components].index
    merged = merged[merged['date'].isin(valid_dates)]

    # Adaptive contribution caps: use rolling volatility per series, bounded by hard cap
    cap_mask = merged['cap_contribution'] == 1
    if cap_mask.any():
        hard_cap = CONTRIBUTION_CAP / 100
        merged = merged.sort_values('date')
        for sid in merged.loc[cap_mask, 'series_id'].unique():
            sid_mask = merged['series_id'] == sid
            rolling_std = (
                merged.loc[sid_mask, 'weighted_log_contrib']
                .rolling(window=ADAPTIVE_CAP_LOOKBACK, min_periods=6)
                .std()
            )
            # Adaptive cap: tighter in calm periods, never exceeds hard cap
            adaptive_cap = (rolling_std * ADAPTIVE_CAP_ZSCORE).clip(upper=hard_cap).fillna(hard_cap)
            merged.loc[sid_mask, 'weighted_log_contrib'] = merged.loc[sid_mask, 'weighted_log_contrib'].clip(
                lower=-adaptive_cap.values, upper=adaptive_cap.values
            )

    # Standard arithmetic contributions purely for the decompose readout display
    merged['weighted_contribution_pp'] = (merged['mom_pct'] * merged['effective_weight']) * 100
    merged['mom_pct_pp']               = merged['mom_pct'] * 100

    proxy = (
        merged.groupby('date')['weighted_log_contrib']
        .sum().reset_index()
    )

    proxy['proxy_mom_pct'] = np.expm1(proxy['weighted_log_contrib']) * 100
    proxy = proxy.drop(columns=['weighted_log_contrib'])

    return proxy.sort_values('date').reset_index(drop=True), merged


def _compute_mom(raw_df):
    pivot = raw_df.pivot(index='date', columns='series_id', values='value')
    mom   = pivot.pct_change().dropna()
    return mom.reset_index().melt(id_vars='date', var_name='series_id', value_name='mom_pct')


# ---------------------------------------------------------------------------
# Dynamic Formula Effect (The 12-Month Trailing Spread)
# ---------------------------------------------------------------------------

def apply_dynamic_formula_drag(proxy_df, actual_pce_df, span=12):
    """
    Apply exponentially-weighted formula drag correction.
    Uses EWM (exponential weighted mean) instead of simple rolling mean
    so recent spreads are weighted more heavily, allowing faster adaptation
    to BLS/BEA methodology shifts.
    """
    p_df = proxy_df.copy()
    a_df = actual_pce_df[['date', 'mom_pct']].rename(columns={'mom_pct': 'actual_mom'}).copy()

    p_df['date'] = p_df['date'].dt.to_period('M').dt.to_timestamp()
    a_df['date'] = a_df['date'].dt.to_period('M').dt.to_timestamp()

    merged = pd.merge(p_df, a_df, on='date', how='left')
    merged['raw_spread'] = merged['proxy_mom_pct'] - merged['actual_mom']

    # Exponentially weighted mean: recent months matter more than distant ones
    merged['dynamic_drag'] = merged['raw_spread'].ewm(span=span, min_periods=3).mean().shift(1)
    # Use median of available spreads as fallback instead of hardcoded constant
    fallback = merged['raw_spread'].median() if merged['raw_spread'].notna().any() else 0.0
    merged['dynamic_drag'] = merged['dynamic_drag'].fillna(fallback)
    merged['proxy_mom_pct'] = merged['proxy_mom_pct'] - merged['dynamic_drag']

    return merged[['date', 'proxy_mom_pct']]


# ---------------------------------------------------------------------------
# Mode 1: Live
# ---------------------------------------------------------------------------

def run_live(is_core=False, verbose=False):
    now = datetime.now()
    cy  = now.year
    label_text = "CORE PCE" if is_core else "HEADLINE PCE"

    print(f"\n{'='*44}\n  {label_text} Proxy Engine — Live Estimate")
    print(f"  Running: {now.strftime('%B %Y')}\n{'='*44}\n")

    print("[1/5] Loading crosswalk...")
    cw = load_crosswalk()

    if is_core:
        # Note: Nondurables (CUSR0000SAN) has been restored to the core basket
        core_drops = ['CUSR0000SAF11', 'CUSR0000SETB', 'CUSR0000SAH21']
        cw = cw[~cw['series_id'].isin(core_drops)].reset_index(drop=True)

    print("\n[2/5] Loading weights...")
    dyn_w, using_dyn = load_weights(cw, cy - 6, is_core=is_core)
    wlabel = "BEA dynamic" if using_dyn else "static"

    print(f"\n[3/5] Building raw proxy ({wlabel} weights)...")
    proxy_df, merged_detail = build_proxy_series(cw, cy - 6, cy, dyn_w)
    
    if verbose:
        print("\n  [Verbose] Component Counts per Month (Post-Gate):")
        counts = merged_detail.groupby('date')['series_id'].nunique().tail(6)
        for d, c in counts.items():
            print(f"    {d.strftime('%Y-%m')}: {c} components")

    print("\n[4/5] Fetching actual PCE from FRED to apply dynamic formula drag...")
    fred_series = "PCEPILFE" if is_core else "PCEPI"
    actual_pce = fetch_fred_data(fred_series, observation_start=f"{cy-6}-01-01")
    
    proxy_df = apply_dynamic_formula_drag(proxy_df, actual_pce)

    latest       = proxy_df.iloc[-1]
    tgt_month    = latest['date'].month
    tgt_year     = latest['date'].year
    raw_proxy    = latest['proxy_mom_pct']
    print(f"      Most recent complete data: {latest['date'].strftime('%B %Y')}")

    print("\n[5/5] Building seasonal factors (statsmodels decomposition)...")
    cutoff   = pd.Timestamp(f"{cy-6}-01-01")
    hist     = proxy_df[
        (proxy_df['date'] >= cutoff) &
        ~((proxy_df['date'].dt.year == tgt_year) &
          (proxy_df['date'].dt.month == tgt_month))
    ]
    rsa_act = actual_pce[actual_pce['date'] >= cutoff]
    rsa     = build_seasonal_factors(rsa_act, hist)
    result  = apply_seasonal_adjustment(raw_proxy, tgt_month, rsa)

    print(f"\n{'='*44}\n  Results — {latest['date'].strftime('%B %Y')} | {wlabel} weights")
    print(f"{'='*44}")
    print(f"  Raw Proxy MoM      : {result['raw_proxy_mom']:+.3f} pp")
    print(f"  RSA Factor         : {result['rsa_factor']:+.3f} pp")
    print(f"  Adjusted Proxy MoM : {result['adjusted_mom']:+.3f} pp")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print(f"\n  Seasonal Factors:")
    for m, n in enumerate(months, 1):
        print(f"    {n}: {rsa[m]:+.4f} pp{'  <--' if m == tgt_month else ''}")
    print()

    # Return data for Streamlit frontend
    return {
        "date_str": latest['date'].strftime('%B %Y'),
        "raw_proxy_mom": result['raw_proxy_mom'],
        "rsa_factor": result['rsa_factor'],
        "adjusted_mom": result['adjusted_mom']
    }


# ---------------------------------------------------------------------------
# Mode 2: Backtest
# ---------------------------------------------------------------------------

def run_backtest(years=3, is_core=False, verbose=False):
    now = datetime.now()
    fetch_start_year = now.year - years - 1 
    sy = now.year - years
    
    label_text = "CORE PCE" if is_core else "HEADLINE PCE"
    print(f"\n{'='*44}\n  {label_text} Proxy Engine — Backtest ({years} years)\n{'='*44}\n")

    print("[1/4] Loading crosswalk...")
    cw = load_crosswalk()

    if is_core:
        shape_before = cw.shape[0]
        core_drops = ['CUSR0000SAF11', 'CUSR0000SETB', 'CUSR0000SAH21']
        cw = cw[~cw['series_id'].isin(core_drops)].reset_index(drop=True)
        shape_after = cw.shape[0]
        if verbose:
            print(f"  [Verbose] Core Filter: Dropped {shape_before - shape_after} components. "
                  f"Rows remaining: {shape_after}")

    print("\n[2/4] Loading weights...")
    dyn_w, using_dyn = load_weights(cw, fetch_start_year, is_core=is_core)
    wlabel = "BEA dynamic" if using_dyn else "static"

    print(f"\n[3/4] Building raw historical proxy ({wlabel} weights)...")
    proxy_df, merged_detail = build_proxy_series(cw, fetch_start_year, now.year, dyn_w)
    
    if verbose:
        print("\n  [Verbose] Component Counts per Month (Post-Gate):")
        counts = merged_detail.groupby('date')['series_id'].nunique().tail(12)
        for d, c in counts.items():
            print(f"    {d.strftime('%Y-%m')}: {c} components")

    print("\n[4/4] Fetching actual PCE and calculating dynamic drag...\n")
    fred_series = "PCEPILFE" if is_core else "PCEPI"
    actual_pce  = fetch_fred_data(fred_series, observation_start=f"{fetch_start_year}-01-01")
    
    proxy_df = apply_dynamic_formula_drag(proxy_df, actual_pce)
    
    cutoff = pd.Timestamp(f"{sy}-01-01")
    proxy_df = proxy_df[proxy_df['date'] >= cutoff].reset_index(drop=True)
    actual_pce = actual_pce[actual_pce['date'] >= cutoff].reset_index(drop=True)
    
    # Build RSA factors excluding the most recent month (same as run_live)
    # so that backtest and live forecast produce consistent numbers
    latest_date = proxy_df['date'].max()
    rsa_proxy = proxy_df[proxy_df['date'] < latest_date]
    rsa_actual = actual_pce[actual_pce['date'] < latest_date]
    rsa         = build_seasonal_factors(rsa_actual, rsa_proxy)
    results     = backtest_accuracy(actual_pce, proxy_df, rsa)

    print_accuracy_summary(results)
    print(f"  {'Date':<12} {'Actual':>8} {'Raw Proxy':>10} {'Adj Proxy':>10} "
          f"{'Raw Err':>9} {'Adj Err':>9}")
    print("  " + "-"*62)
    for _, row in results.iterrows():
        print(f"  {row['date'].strftime('%Y-%m'):<12}"
              f"  {row['actual_mom']:>7.3f}"
              f"  {row['proxy_mom_pct']:>9.3f}"
              f"  {row['adjusted_proxy']:>9.3f}"
              f"  {row['raw_error']:>8.3f}"
              f"  {row['adjusted_error']:>8.3f}")
    print()

    
    #Return data for Streamlit frontend
    
    # Calculate MAE metrics
    clean_df = results[~((results["date"] >= '2020-03-01') & (results["date"] <= '2021-12-01'))]
    ex_covid_mae = clean_df["adjusted_error"].abs().mean()
    
    return {
        "dataframe": results,
        "mae": ex_covid_mae
    }


# ---------------------------------------------------------------------------
# Mode 3: Decompose
# ---------------------------------------------------------------------------

def run_decompose(date_str):
    try:
        tgt = pd.to_datetime(date_str + "-01")
    except Exception:
        raise ValueError(f"Use YYYY-MM format, e.g. 2023-08")

    print(f"\n{'='*64}\n  Component Decomposition — {tgt.strftime('%B %Y')}\n{'='*64}\n")

    cw = load_crosswalk()
    print("\nLoading weights...")
    dyn_w, using_dyn = load_weights(cw, tgt.year - 1)

    print("\nFetching data...")
    _, detail = build_proxy_series(cw, tgt.year - 1, tgt.year, dyn_w)

    month = detail[
        (detail['date'].dt.year == tgt.year) &
        (detail['date'].dt.month == tgt.month)
    ].copy()

    if month.empty:
        print(f"[Error] No data for {date_str}.")
        return

    month = month.sort_values('weighted_contribution_pp', key=abs, ascending=False)
    total = month['weighted_contribution_pp'].sum()
    wcol  = 'effective_weight' if 'effective_weight' in month.columns else 'pce_weight'

    print(f"\n  {'Component':<35} {'Src':>4} {'Wt%':>5} {'MoM%':>7} {'Contrib':>8} {'Cap':>5}")
    print("  " + "-"*72)
    for _, r in month.iterrows():
        print(f"  {r['target_pce_name']:<35}"
              f"  {r['source_type']:>3}"
              f"  {r[wcol]*100:>4.1f}"
              f"  {r['mom_pct_pp']:>6.3f}"
              f"  {r['weighted_contribution_pp']:>7.3f} pp"
              f"{'  YES' if r['cap_contribution']==1 else ''}")
    print("  " + "-"*72)
    print(f"  {'TOTAL RAW PROXY (Pre-Drag)':>55}  {total:>7.3f} pp\n")
    print("  *Note: Total excludes Dynamic Trailing Spread and Residual Seasonal Adjustment.\n")

    try:
        pce = fetch_fred_data("PCEPI", observation_start=f"{tgt.year}-01-01")
        row = pce[(pce['date'].dt.year == tgt.year) & (pce['date'].dt.month == tgt.month)]
        if not row.empty:
            actual = row.iloc[0]['mom_pct']
            print(f"  Actual PCE MoM : {actual:+.3f} pp")
    except Exception:
        pass
    print()


# ---------------------------------------------------------------------------
# Mode 4: Weights report
# ---------------------------------------------------------------------------

def run_weights_report():
    now = datetime.now()
    print(f"\n{'='*64}\n  Weight Comparison — Static vs BEA Dynamic\n{'='*64}\n")

    cw = load_crosswalk()
    dyn_w, using_dyn = load_weights(cw, now.year - 2)

    if not using_dyn:
        print("  BEA dynamic weights not available. Add BEA_API_KEY to .env\n")
        return

    latest_date = dyn_w['date'].max()
    latest      = dyn_w[dyn_w['date'] == latest_date]

    print(f"  {'Series':<25} {'Component':<30} {'Static%':>8} {'Dynamic%':>9} {'Diff':>7}")
    print("  " + "-"*80)
    for _, r in cw.sort_values('pce_weight', ascending=False).iterrows():
        sid = r['series_id']
        sw  = r['pce_weight'] * 100
        d   = latest[latest['series_id'] == sid]
        dw  = d.iloc[0]['dynamic_weight'] * 100 if not d.empty else None
        if dw is not None:
            print(f"  {sid:<25} {r['target_pce_name']:<30} {sw:>7.2f}%  {dw:>7.2f}%  {dw-sw:>+6.2f}%")
        else:
            print(f"  {sid:<25} {r['target_pce_name']:<30} {sw:>7.2f}%  {'N/A':>8}  {'—':>7}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCE Proxy Engine")
    parser.add_argument("--backtest",  action="store_true")
    parser.add_argument("--years",     type=int, default=3)
    parser.add_argument("--decompose", type=str, metavar="YYYY-MM")
    parser.add_argument("--weights",   action="store_true")
    parser.add_argument("--core",      action="store_true", help="Run proxy for Core PCE")
    parser.add_argument("--verbose",   action="store_true", help="Print advanced diagnostic data")
    args = parser.parse_args()

    try:
        if args.decompose:
            run_decompose(args.decompose)
        elif args.backtest:
            run_backtest(years=args.years, is_core=args.core, verbose=args.verbose)
        elif args.weights:
            run_weights_report()
        else:
            run_live(is_core=args.core, verbose=args.verbose)
    except Exception as e:
        import traceback
        import sys
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)