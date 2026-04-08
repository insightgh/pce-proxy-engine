"""
weights.py
Handles static and dynamic weight fetching and processing for the PCE Proxy.
"""

import pandas as pd

# 1-to-1 Mapping: BEA Line Number -> BLS/PPI Series ID
BEA_CONCORDANCE = {
    53:  "CUSR0000SETC",   # Motor vehicle parts
    68:  "CUSR0000SAH3",   # Furnishings
    73:  "CUSR0000SAR",    # Recreation (FIXED ID)
    74:  "CUSR0000SAG1",   # Other durable goods
    76:  "CUSR0000SAF11",  # Food and beverages
    108: "CUSR0000SAA",    # Clothing and footwear
    115: "CUSR0000SETB",   # Gasoline and other energy
    130: "CUSR0000SAN",    # Other nondurable goods
    148: "CUSR0000SEHA",   # Rent
    149: "CUSR0000SEHC",   # Owners equivalent rent
    169: "CUSR0000SAH21",  # Electricity and gas
    170: "CUSR0000SEHG",   # Water and sanitation
    199: "CUSR0000SAS4",   # Transportation services
    204: "CUSR0000SARS",   # Recreation services (FIXED ID)
    228: "CUSR0000SEFV",   # Food services and accommodations
    232: "CUSR0000SAE2",   # Communication
    233: "CUSR0000SETG01", # Airline Fares (FIXED ID)
    253: "WPSFD4131",      # Financial services
}

# 1-to-Many Mapping: BEA Line Number -> [(Series ID, Allocation Share)]
BEA_SPLIT_LINES = {
    172: [ # Health care
        ("CUSR0000SAM", 0.38),  
        ("WPS511104",   0.31),  
        ("WPS512101",   0.31),  
    ],
    342: [ # NPISHs (Non-profits)
        ("CUSR0000SEEB", 0.30),  
        ("WPS512101",    0.70),  
    ],
}

# Fallback list used if BEA API fails or connection is lost
STATIC_WEIGHT_SERIES = [
    "CUSR0000SETC", "CUSR0000SAH3", "CUSR0000SAR", "CUSR0000SAG1",
    "CUSR0000SAF11", "CUSR0000SAA", "CUSR0000SETB", "CUSR0000SAN",
    "CUSR0000SEHA", "CUSR0000SEHC", "CUSR0000SAH21", "CUSR0000SEHG",
    "CUSR0000SAS4", "CUSR0000SARS", "CUSR0000SEFV", "CUSR0000SAE2",
    "WPSFD4131", "CUSR0000SETG01", "CUSR0000SAM", "WPS511104", "WPS512101",
    "CUSR0000SEEB"
]

def build_dynamic_weights(bea_df, crosswalk_df, is_core=False):
    target_line = 374 if is_core else 1
    total = (
        bea_df[bea_df['line_number'] == target_line]
        [['date', 'value_billions']]
        .rename(columns={'value_billions': 'total_pce'})
    )
    if total.empty:
        raise ValueError(f"BEA line {target_line} not found.")

    records = []
    
    # 74 = Food at Home, 115 = Gasoline, 169 = Electricity/Gas
    CORE_EXCLUSIONS = {74, 115, 169}

    for date, month_data in bea_df.groupby('date'):
        total_row = total[total['date'] == date]
        if total_row.empty: continue
        total_val = total_row.iloc[0]['total_pce']
        line_values = month_data.set_index('line_number')['value_billions'].to_dict()

        for line_num, series_id in BEA_CONCORDANCE.items():
            if is_core and line_num in CORE_EXCLUSIONS:
                continue 
            if line_num in line_values:
                records.append({
                    'date': date, 'series_id': series_id,
                    'dynamic_weight': line_values[line_num] / total_val,
                    'source': 'BEA',
                })
                
        for line_num, splits in BEA_SPLIT_LINES.items():
            if is_core and line_num in CORE_EXCLUSIONS:
                continue
            if line_num in line_values:
                for split_series, share in splits:
                    records.append({
                        'date': date, 'series_id': split_series,
                        'dynamic_weight': (line_values[line_num] * share) / total_val,
                        'source': 'BEA',
                    })

    dynamic_df = pd.DataFrame(records)

    # Only use static weights for series NOT already covered by BEA dynamic weights
    bea_covered_series = set(dynamic_df['series_id'].unique()) if not dynamic_df.empty else set()
    all_dates = bea_df['date'].unique()
    static_rows = crosswalk_df[
        (crosswalk_df['series_id'].isin(STATIC_WEIGHT_SERIES)) &
        (~crosswalk_df['series_id'].isin(bea_covered_series))
    ]

    static_records = [
        {'date': date, 'series_id': row['series_id'], 'dynamic_weight': row['pce_weight'], 'source': 'static'}
        for date in all_dates for _, row in static_rows.iterrows()
    ]

    combined = pd.concat([dynamic_df, pd.DataFrame(static_records)], ignore_index=True)

    # Aggregate duplicate (date, series_id) pairs (e.g. WPS512101 in multiple split lines)
    combined = combined.groupby(['date', 'series_id'], as_index=False).agg({
        'dynamic_weight': 'sum',
        'source': 'first',
    })

    # Normalize per-month so weights sum to 1.0
    month_totals = combined.groupby('date')['dynamic_weight'].transform('sum')
    combined['dynamic_weight'] = combined['dynamic_weight'] / month_totals.where(month_totals > 0, 1.0)

    return combined.sort_values(['date', 'series_id']).reset_index(drop=True)

def weight_coverage_report(dynamic_weights_df, crosswalk_df):
    if dynamic_weights_df.empty: return
    latest_date = dynamic_weights_df['date'].max()
    latest = dynamic_weights_df[dynamic_weights_df['date'] == latest_date]
    bea_coverage = latest[latest['source'] == 'BEA']['dynamic_weight'].sum() * 100
    static_coverage = latest[latest['source'] == 'static']['dynamic_weight'].sum() * 100