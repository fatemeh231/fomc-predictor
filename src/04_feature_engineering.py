# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:41:02 2026

@author: fatemeh
"""

import pandas as pd
import numpy as np

# Path
base = r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor'
df = pd.read_parquet(base + r'\data\processed\master_dataset.parquet')

# Sort by date (important for delta calculations)
df = df.sort_values('date').reset_index(drop=True)

# 1. Fed funds change since last meeting
df['fed_funds_change'] = df['fed_funds'].diff()

# 2. CPI change and year-over-year
df['cpi_change'] = df['cpi'].diff()
df['cpi_yoy'] = df['cpi'].pct_change(periods=4) * 100  # 4 meetings ≈ 1 year

# 3. Unemployment change
df['unemployment_change'] = df['unemployment'].diff()

# 4. Yield spread (classic recession signal)
df['yield_spread'] = df['treasury_10y'] - df['treasury_2y']

# 5. Treasury 2Y change
df['treasury_2y_change'] = df['treasury_2y'].diff()

# 6. Gold vs Dollar ratio (inflation vs strength)
df['gold_dxy_ratio'] = df['gld_close'] / df['dxy_close']

# 7. SPY momentum (3-meeting return)
df['spy_momentum'] = df['spy_close'].pct_change(periods=3) * 100

# 8. BTC momentum
df['btc_momentum'] = df['btc_close'].pct_change(periods=3) * 100

# 9. Days since last meeting
df['days_since_last_meeting'] = df['date'].diff().dt.days

# Drop the first row (has NaN from diff)
df = df.dropna().reset_index(drop=True)

print("Shape after feature engineering:", df.shape)
print("\nNew columns:", list(df.columns))
print("\nFirst 3 rows of new features:")
print(df[['date', 'decision', 'fed_funds_change', 'cpi_change', 'yield_spread', 'spy_momentum']].head(3))

# Save
output_path = base + r'\data\processed\features_dataset.parquet'
df.to_parquet(output_path)
print(f"\nSaved feature dataset to: {output_path}")