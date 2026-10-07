# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:14:52 2026

@author: fatemeh
"""

import yfinance as yf
import pandas as pd

# Download market data
tickers = ['SPY', 'GLD', 'DX-Y.NYB', 'BTC-USD']
data = yf.download(tickers, start='2014-01-01', end='2026-10-05', interval='1d')

# Keep only Close prices
close = data['Close']

# Rename columns for clarity
close = close.rename(columns={
    'SPY': 'spy_close',
    'GLD': 'gld_close',
    'DX-Y.NYB': 'dxy_close',
    'BTC-USD': 'btc_close',
})

# Preview
print("Shape:", close.shape)
print("\nColumns:", list(close.columns))
print("\nLast 5 rows:")
print(close.tail())
print("\nAny missing values?")
print(close.isna().sum())

# Save raw market data
close.to_parquet(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor\data\raw\market_data.parquet')
print("\nSaved to: data/raw/market_data.parquet")