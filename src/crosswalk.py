"""
crosswalk.py
Loads and validates data/crosswalk.csv.
"""

import pandas as pd
import os

REQUIRED_COLUMNS   = {'target_pce_name', 'series_id', 'source_type', 'pce_weight', 'active'}
VALID_SOURCE_TYPES = {'CPI', 'IMPORT', 'PPI', 'FRED'}
CONTRIBUTION_CAP   = 0.07


def load_crosswalk(filepath="data/crosswalk.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}.")

    df = pd.read_csv(filepath)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"crosswalk.csv missing columns: {missing}")

    df['pce_weight']       = pd.to_numeric(df['pce_weight'],  errors='coerce')
    df['active']           = pd.to_numeric(df['active'],      errors='coerce').fillna(0).astype(int)
    df['source_type']      = df['source_type'].str.strip().str.upper()
    df['cap_contribution'] = pd.to_numeric(
        df.get('cap_contribution', 0), errors='coerce'
    ).fillna(0).astype(int)

    bad = df['pce_weight'].isna()
    if bad.any():
        raise ValueError(f"Non-numeric pce_weight: {df.loc[bad,'target_pce_name'].tolist()}")

    bad_types = ~df['source_type'].isin(VALID_SOURCE_TYPES)
    if bad_types.any():
        raise ValueError(f"Invalid source_type: {df.loc[bad_types,'source_type'].tolist()}")

    active_df = df[df['active'] == 1].copy()
    if active_df.empty:
        raise ValueError("No active rows in crosswalk.csv.")

    total = active_df['pce_weight'].sum()
    if not (0.98 <= total <= 1.02):
        print(f"  [Warning] Weights sum to {total:.4f} — normalising.")
        active_df['pce_weight'] = active_df['pce_weight'] / total

    capped = (active_df['cap_contribution'] == 1).sum()
    print(f"  Crosswalk: {len(active_df)} components | "
          f"CPI: {(active_df['source_type']=='CPI').sum()} | "
          f"Import: {(active_df['source_type']=='IMPORT').sum()} | "
          f"PPI: {(active_df['source_type']=='PPI').sum()} | "
          f"Capped: {capped} | Weights: {total:.4f}")

    return active_df.reset_index(drop=True)


def get_series_by_type(crosswalk_df, source_type):
    mask = crosswalk_df['source_type'] == source_type.upper()
    return crosswalk_df.loc[mask, 'series_id'].unique().tolist()