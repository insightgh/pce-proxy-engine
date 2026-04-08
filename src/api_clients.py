"""
api_clients.py
All external data fetching:
  BLS  — CPI, Import Price Index, PPI (same endpoint, different series prefixes)
  FRED — actual PCE for backtesting
  BEA  — monthly PCE expenditure by category for time-varying weights
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def _get_secret(key):
    """Check Streamlit secrets first (for cloud deploy), then fall back to env vars."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

BLS_API_KEY  = _get_secret("BLS_API_KEY")
FRED_API_KEY = _get_secret("FRED_API_KEY")
BEA_API_KEY  = _get_secret("BEA_API_KEY")

BLS_API_URL  = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
BEA_API_URL  = "https://apps.bea.gov/api/data"


# ---------------------------------------------------------------------------
# Internal BLS fetcher — shared by CPI, Import Price, PPI
# ---------------------------------------------------------------------------

def _fetch_bls_series(series_ids, start_year, end_year, label="series", max_retries=3):
    """
    Fetch any BLS v2 series. Batches in groups of 50 (API hard limit).
    Returns DataFrame: series_id, date, value
    """
    end_year    = min(end_year, datetime.now().year)
    all_records = []
    empty_ids   = []

    for i in range(0, len(series_ids), 50):
        batch = series_ids[i:i + 50]
        print(f"  Fetching {len(batch)} {label} ({start_year}-{end_year})...")

        payload = {
            "seriesid":        batch,
            "startyear":       str(start_year),
            "endyear":         str(end_year),
            "registrationkey": BLS_API_KEY,
        }
        
        # Robust Retry Loop for BLS
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    BLS_API_URL, json=payload,
                    headers={'Content-type': 'application/json'}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                
                if data['status'] != 'REQUEST_SUCCEEDED':
                    raise Exception(f"BLS error: {data.get('message')}")
                break # Success, break out of retry loop
                
            except (requests.exceptions.RequestException, Exception) as e:
                if attempt < max_retries - 1:
                    print(f"  [BLS API] Connection blip. Retrying in 2s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(2)
                else:
                    raise Exception(f"BLS API failed after {max_retries} attempts: {e}")

        for series in data['Results']['series']:
            sid = series['seriesID']
            obs = [
                {
                    'series_id': sid,
                    'date':      pd.to_datetime(f"{it['year']}-{it['period'][1:]}-01"),
                    'value':     float(it['value']),
                }
                for it in series['data']
                if it['value'] not in ("-", "") and it['period'][1:].isdigit()
            ]
            if obs:
                all_records.extend(obs)
            else:
                empty_ids.append(sid)

    if empty_ids:
        print(f"  [Warning] No data for: {empty_ids} — verify IDs at bls.gov")
    if not all_records:
        raise ValueError(f"BLS returned no data for {label}: {series_ids}")

    df = pd.DataFrame(all_records)
    return df.sort_values(['series_id', 'date']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public BLS fetchers
# ---------------------------------------------------------------------------

def fetch_bls_data(series_ids, start_year, end_year):
    return _fetch_bls_series(series_ids, start_year, end_year, label="CPI series")

def fetch_import_price_data(series_ids, start_year, end_year):
    return _fetch_bls_series(series_ids, start_year, end_year, label="Import Price series")

def fetch_ppi_data(series_ids, start_year, end_year):
    return _fetch_bls_series(series_ids, start_year, end_year, label="PPI series")


# ---------------------------------------------------------------------------
# FRED — actual PCE for backtesting
# ---------------------------------------------------------------------------

def fetch_fred_data(series_id="PCEPI", observation_start="2015-01-01", max_retries=3):
    """Fetch monthly FRED series, return MoM % changes."""
    params = {
        "series_id":         series_id,
        "observation_start": observation_start,
        "frequency":         "m",
        "units":             "lin",
        "file_type":         "json",
        "api_key":           FRED_API_KEY,
    }
    
    # Robust Retry Loop for FRED
    for attempt in range(max_retries):
        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            break # Success, break out of retry loop
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  [FRED API] Connection blip. Retrying in 2s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(2)
            else:
                raise Exception(f"FRED API failed after {max_retries} attempts: {e}")

    records = [
        {"date": pd.to_datetime(o["date"]), "value": float(o["value"])}
        for o in raw.get("observations", [])
        if o["value"] != "."
    ]
    if not records:
        raise ValueError(f"FRED returned no data for '{series_id}'.")

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["mom_pct"] = df["value"].pct_change() * 100
    return df.dropna(subset=["mom_pct"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# BEA — time-varying PCE expenditure weights
# ---------------------------------------------------------------------------

def fetch_bea_pce_weights(start_year=2018, max_retries=3):
    """
    Fetch monthly PCE expenditure by category from BEA.
    """
    if not BEA_API_KEY:
        raise EnvironmentError(
            "BEA_API_KEY not set in .env. "
            "Register free at apps.bea.gov/API/signup"
        )

    years = ",".join(str(y) for y in range(start_year, datetime.now().year + 1))

    params = {
        "UserID":       BEA_API_KEY,
        "method":       "GetData",
        "DataSetName":  "NIUnderlyingDetail",
        "TableName":    "U20405",
        "Frequency":    "M",
        "Year":         years,
        "ResultFormat": "JSON",
    }

    print(f"  Fetching BEA Table U20405 (NIUnderlyingDetail), {start_year}-present...")
    
    # Robust Retry Loop for BEA
    for attempt in range(max_retries):
        try:
            resp = requests.get(BEA_API_URL, params=params, timeout=60)
            resp.raise_for_status()
            raw = resp.json()
            break # Success, break out of retry loop
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  [BEA API] Connection blip. Retrying in 2s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(2)
            else:
                raise Exception(f"BEA API failed after {max_retries} attempts: {e}")

    if "BEAAPI" not in raw:
        raise ValueError(f"Unexpected BEA response: {list(raw.keys())}")

    results = raw["BEAAPI"].get("Results", {})

    if "Error" in results:
        raise ValueError(f"BEA API error: {results['Error']}")

    data = results.get("Data", [])
    if not data:
        raise ValueError(
            f"BEA Table U20405 returned no data. "
            f"Response keys: {list(results.keys())}. "
            f"Check that your BEA_API_KEY is valid and the table/dataset name is correct."
        )

    records = []
    for obs in data:
        try:
            period = obs["TimePeriod"]          
            if "M" not in period:
                continue                        
            year  = int(period[:4])
            month = int(period[5:])
            val   = obs["DataValue"].replace(",", "")
            if val in ("", "---", "(D)"):
                continue
            records.append({
                "date":             pd.to_datetime(f"{year}-{month:02d}-01"),
                "line_number":      int(obs["LineNumber"]),
                "line_description": obs["LineDescription"].strip(),
                "value_billions":   float(val),
            })
        except (KeyError, ValueError):
            continue

    if not records:
        raise ValueError("Could not parse any rows from BEA U20405 response.")

    df = (
        pd.DataFrame(records)
        .sort_values(["line_number", "date"])
        .reset_index(drop=True)
    )
    print(f"  BEA loaded: {df['line_description'].nunique()} categories, "
          f"{df['date'].nunique()} months.")
    return df