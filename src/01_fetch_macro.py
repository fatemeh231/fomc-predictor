import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

# Load API key
load_dotenv()
api_key = os.getenv('FRED_API_KEY')
fred = Fred(api_key=api_key)

# Define the series we want
series = {
    'fed_funds':     'FEDFUNDS',    # Federal Funds Rate
    'cpi':           'CPIAUCSL',    # Consumer Price Index
    'unemployment':  'UNRATE',      # Unemployment Rate
    'gdp':           'GDP',         # Gross Domestic Product
    'treasury_2y':   'DGS2',        # 2-Year Treasury Yield
    'treasury_10y':  'DGS10',       # 10-Year Treasury Yield
}

# Pull each series one by one
data = {}
for name, code in series.items():
    print(f"Pulling {name} ({code})...")
    data[name] = fred.get_series(code)

# Combine into one DataFrame
df = pd.DataFrame(data)

# Show a preview
print("\n--- Preview ---")
print(df.tail(5))
print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))

# Save raw data to Parquet
df.to_parquet(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor\data\raw\macro_data.parquet')
print("\nSaved successfully.")