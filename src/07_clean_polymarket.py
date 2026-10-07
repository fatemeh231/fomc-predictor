# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:19:44 2026

@author: fatemeh
"""

import pandas as pd
from pathlib import Path

base = r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor'
input_path = Path(base) / 'data' / 'processed' / 'polymarket_extracted.csv'

df = pd.read_csv(input_path)

print("Before cleaning:", len(df), "rows")

# 1. Drop rows with NaN
df = df.dropna(subset=['prob_cut', 'prob_hold', 'prob_hike'])

# 2. Drop rows where probabilities don't sum to ~1.0
df['total'] = df['prob_cut'] + df['prob_hold'] + df['prob_hike']
df = df[(df['total'] > 0.9) & (df['total'] < 1.1)]

# 3. Ensure last_data_date is within 14 days of meeting
df['meeting_date'] = pd.to_datetime(df['meeting_date'])
df['last_data_date'] = pd.to_datetime(df['last_data_date'], format='mixed', errors='coerce')
df['days_gap'] = (df['meeting_date'] - df['last_data_date']).dt.days
df = df[(df['days_gap'] >= 0) & (df['days_gap'] <= 14)]

# Drop temporary columns
df = df.drop(columns=['total', 'days_gap'])

print("After cleaning:", len(df), "rows")
print()
print(df.to_string(index=False))

# Save
output = Path(base) / 'data' / 'processed' / 'polymarket_clean.csv'
df.to_csv(output, index=False)
print(f"\nSaved to: {output}")