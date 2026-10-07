import pandas as pd

# Paths
base = r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor'
macro_path = base + r'\data\raw\macro_data.parquet'
market_path = base + r'\data\raw\market_data.parquet'
fomc_path = base + r'\data\raw\fomc_dates.csv'

# Load
macro = pd.read_parquet(macro_path)
market = pd.read_parquet(market_path)
fomc = pd.read_csv(fomc_path, parse_dates=['date'])

# Sort by date (required for merge_asof)
macro = macro.sort_index()
market = market.sort_index()
fomc = fomc.sort_values('date')

# THE FIX: Forward-fill so each day carries the last known value
macro = macro.ffill()
market = market.ffill()

# Merge macro (backward = grab most recent value BEFORE the meeting)
merged = pd.merge_asof(
    fomc,
    macro,
    left_on='date',
    right_index=True,
    direction='backward'
)

# Merge market
merged = pd.merge_asof(
    merged,
    market,
    left_on='date',
    right_index=True,
    direction='backward'
)

# Preview
print("Shape:", merged.shape)
print("\nColumns:", list(merged.columns))
print("\nLast 5 rows:")
print(merged.tail())
print("\nAny missing values?")
print(merged.isna().sum())

# Save the merged dataset
output_path = base + r'\data\processed\master_dataset.parquet'
merged.to_parquet(output_path)
print(f"\nSaved master dataset to: {output_path}")