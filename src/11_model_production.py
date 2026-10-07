# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:42:52 2026

@author: fatemeh
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor')

df = pd.read_parquet(base / 'data' / 'processed' / 'master_final_v3.parquet')
df = df.sort_values('date').reset_index(drop=True)
df['target'] = (df['decision'] != 'hold').astype(int)

macro_features = [
    'fed_funds', 'cpi', 'unemployment', 'gdp',
    'treasury_2y', 'treasury_10y',
    'btc_close', 'dxy_close', 'gld_close', 'spy_close',
    'fed_funds_change', 'cpi_change', 'cpi_yoy', 'unemployment_change',
    'yield_spread', 'treasury_2y_change', 'gold_dxy_ratio',
    'spy_momentum', 'btc_momentum', 'days_since_last_meeting'
]
all_features = macro_features + ['prob_cut', 'prob_hold', 'prob_hike', 'has_market_data']

# ---- FIXED SPLIT: train < 2024, test >= 2024 ----
CUTOFF = pd.Timestamp('2024-01-01')
train_df = df[df['date'] < CUTOFF].copy()
test_df  = df[df['date'] >= CUTOFF].copy()

print("=" * 78)
print("FIXED CHRONOLOGICAL SPLIT")
print("=" * 78)
print(f"Train: {len(train_df)} meetings  ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"Test:  {len(test_df)} meetings  ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
print(f"\nTrain target mix: {train_df['target'].value_counts().to_dict()}")
print(f"Test target mix:  {test_df['target'].value_counts().to_dict()}")
print(f"Test meetings WITH market data: {test_df['has_market_data'].sum()}")
print()

# ---- Train Random Forest on training data ----
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(train_df[all_features], train_df['target'])

# ---- Hybrid prediction function ----
def hybrid_predict(row, rf):
    if row['has_market_data'] == 1:
        if row['prob_cut'] > 0.6 or row['prob_hike'] > 0.6:
            return 1, 'market→change'
        elif row['prob_hold'] > 0.6:
            return 0, 'market→hold'
        else:
            return rf.predict(row[all_features].to_frame().T)[0], 'ML (uncertain market)'
    else:
        return rf.predict(row[all_features].to_frame().T)[0], 'ML (no market data)'

# ---- Generate predictions ----
results = []
for _, row in test_df.iterrows():
    pred, method = hybrid_predict(row, rf)
    results.append({
        'date':         row['date'].date(),
        'actual':       row['decision'],
        'actual_bin':   row['target'],
        'predicted':    int(pred),
        'pred_label':   'change' if pred == 1 else 'hold',
        'correct':      int(pred) == row['target'],
        'method':       method,
        'prob_cut':     row['prob_cut'],
        'prob_hold':    row['prob_hold'],
        'prob_hike':    row['prob_hike'],
    })

pred_df = pd.DataFrame(results)

print("=" * 78)
print("TEST SET PREDICTIONS")
print("=" * 78)
display = pred_df[['date', 'actual', 'pred_label', 'correct', 'method', 'prob_cut', 'prob_hold', 'prob_hike']]
print(display.to_string(index=False))

# ---- Overall metrics ----
acc = pred_df['correct'].mean()
print("\n" + "=" * 78)
print("OVERALL PERFORMANCE")
print("=" * 78)
print(f"Accuracy: {acc:.4f}  ({pred_df['correct'].sum()}/{len(pred_df)})")
print()
print(classification_report(
    pred_df['actual_bin'], pred_df['predicted'],
    zero_division=0, target_names=['hold', 'change']
))
print("Confusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(pred_df['actual_bin'], pred_df['predicted'], labels=[0, 1]))

# ---- Performance by method ----
print("\n" + "=" * 78)
print("PERFORMANCE BY PREDICTION METHOD")
print("=" * 78)
for m in sorted(pred_df['method'].unique()):
    sub = pred_df[pred_df['method'] == m]
    print(f"  {m:30s}: {sub['correct'].mean():.2%} on {len(sub)} meetings")

# ---- Performance by market data availability ----
print("\n" + "=" * 78)
print("PERFORMANCE BY MARKET DATA AVAILABILITY")
print("=" * 78)
with_m  = pred_df[pred_df['method'].str.startswith('market')]
no_m_ml = pred_df[pred_df['method'] == 'ML (no market data)']
if len(with_m) > 0:
    print(f"  With market data:     {with_m['correct'].mean():.2%} on {len(with_m)} meetings")
if len(no_m_ml) > 0:
    print(f"  Without market data:  {no_m_ml['correct'].mean():.2%} on {len(no_m_ml)} meetings")

# ---- Save ----
out = base / 'data' / 'processed' / 'final_predictions.csv'
pred_df.to_csv(out, index=False)
print(f"\nSaved to: {out}")