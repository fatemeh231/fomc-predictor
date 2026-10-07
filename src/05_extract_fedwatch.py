# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:16:08 2026

@author: fatemeh
"""

import pandas as pd
import os
from pathlib import Path

base = r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor'
data_folder = Path(base) / 'data' / 'polymarket_fed_analysis-main' / 'poly_fed_data'

# Find all fedwatch_outcomes files
outcome_files = sorted(data_folder.glob('fedwatch_outcomes_*.csv'))
print(f"Found {len(outcome_files)} outcome files:")
for f in outcome_files:
    print(f"  - {f.name}")
print()

results = []

for f in outcome_files:
    # Extract meeting date from filename: fedwatch_outcomes_20250730.csv
    meeting_date_str = f.stem.split('_')[-1]  # '20250730'
    meeting_date = pd.to_datetime(meeting_date_str, format='%Y%m%d')

    # Load
    df = pd.read_csv(f)

    # Find row closest to 7 days before decision
    df['diff'] = abs(df['days_before_decision'] - 7)
    row = df.loc[df['diff'].idxmin()]

    # Extract probabilities
    prob_cut = row['50+ bps decrease'] + row['25 bps decrease']
    prob_hold = row['No change']
    prob_hike = row['25+ bps increase']

    results.append({
        'meeting_date': meeting_date,
        'source': 'fedwatch',
        'prob_cut': round(prob_cut, 4),
        'prob_hold': round(prob_hold, 4),
        'prob_hike': round(prob_hike, 4),
        'days_before': row['days_before_decision']
    })

result_df = pd.DataFrame(results)
print("Extracted:")
print(result_df.to_string(index=False))

# Save
output = Path(base) / 'data' / 'processed' / 'fedwatch_extracted.csv'
result_df.to_csv(output, index=False)
print(f"\nSaved to: {output}")