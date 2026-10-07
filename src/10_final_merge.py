import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor')

features = pd.read_parquet(base / 'data' / 'processed' / 'features_dataset.parquet')
fedwatch = pd.read_csv(base / 'data' / 'processed' / 'fedwatch_extracted.csv')
kalshi   = pd.read_csv(base / 'data' / 'processed' / 'kalshi_probs.csv')
poly     = pd.read_csv(base / 'data' / 'processed' / 'polymarket_clean.csv')

features['date'] = pd.to_datetime(features['date'])
fedwatch['meeting_date'] = pd.to_datetime(fedwatch['meeting_date'])
kalshi['meeting_date']   = pd.to_datetime(kalshi['meeting_date'])
poly['meeting_date']     = pd.to_datetime(poly['meeting_date'])

fw = fedwatch[['meeting_date', 'prob_cut', 'prob_hold', 'prob_hike']].copy()
kl = kalshi[['meeting_date', 'prob_cut', 'prob_hold', 'prob_hike']].copy()
pl = poly[['meeting_date', 'prob_cut', 'prob_hold', 'prob_hike']].copy()

all_market = pd.concat([fw, kl, pl], ignore_index=True)

combined = all_market.groupby('meeting_date').agg({
    'prob_cut':  'mean',
    'prob_hold': 'mean',
    'prob_hike': 'mean',
}).reset_index()

merged = features.merge(combined, left_on='date', right_on='meeting_date', how='left')
merged = merged.drop(columns=['meeting_date'])

# Add has_market_data flag
merged['has_market_data'] = merged['prob_cut'].notna().astype(int)

# KEEP NaN in probabilities (do NOT fill)
print(f"Meetings with real market data: {merged['has_market_data'].sum()} of {len(merged)}")
print(f"Meetings with missing market data: {(merged['has_market_data'] == 0).sum()}")

output = base / 'data' / 'processed' / 'master_final_v3.parquet'
merged.to_parquet(output)
print(f"\nSaved to: {output}")