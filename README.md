
# FOMC Rate Decision Predictor

A hybrid machine learning system that predicts Federal Reserve (FOMC) interest rate decisions by combining macroeconomic data, market data, and prediction market probabilities.

**Author:** Seyedeh Fatemeh Hosseininassab
**Course:** Computer Science — University Project
**Date:** October 2026

---

## 🎯 Project Overview

The Federal Reserve's FOMC meets **8 times per year** to decide the federal funds rate. These decisions move global markets — currencies, gold, equities, and crypto.

This project builds a system that predicts each FOMC decision (hike / hold / cut) using:

1. **Macroeconomic data** from FRED (CPI, unemployment, GDP, Treasury yields)
2. **Market data** from yfinance (SPY, GLD, DXY, Bitcoin)
3. **Prediction market probabilities** from CME FedWatch, Kalshi, and Polymarket

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **95.45%** (21/22 test meetings) |
| Accuracy on meetings with market data | **100%** (15/15) |
| Accuracy on meetings without market data | **85.71%** (6/7) |
| Training period | Jul 2015 – Dec 2023 (68 meetings) |
| Test period | Jan 2024 – Sep 2026 (22 meetings) |

**Key finding:** Prediction market probabilities are near-perfect predictors of FOMC decisions. The hybrid model achieves 100% accuracy when market data exists.

---

## 🗂️ Project Structure

fomc-predictor/
├── data/
│   ├── raw/
│   │   ├── macro_data.parquet
│   │   ├── market_data.parquet
│   │   ├── fomc_dates.csv
│   │   ├── kalshi_historical_markets.csv
│   │   └── kalshi_fed_markets.csv
│   └── processed/
│       ├── master_dataset.parquet
│       ├── features_dataset.parquet
│       ├── master_final_v3.parquet
│       └── final_predictions.csv
│
├── src/
│   ├── 01_fetch_macro.py
│   ├── 02_fetch_market.py
│   ├── 03_build_dataset.py
│   ├── 04_feature_engineering.py
│   ├── 05_extract_fedwatch.py
│   ├── 06_extract_polymarket.py
│   ├── 07_clean_polymarket.py
│   ├── 08_extract_kalshi.py
│   ├── 09_kalshi_to_probs.py
│   ├── 10_final_merge.py
│   ├── 11_model_production.py
│   ├── 12_make_plots.py
│   └── archive/
│
├── reports/
│   └── figures/
│
├── docs/
│   └── HANDOFF.md
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md

---

## 🚀 Quick Start

### 1. Install dependencies

    pip install -r requirements.txt

**Note for users in Iran:** Use a mirror if PyPI is blocked:

    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### 2. Get a FRED API key

1. Go to https://fredaccount.stlouisfed.org/apikeys
2. Create a free account
3. Request an API key
4. Create a `.env` file in the project root with:

    FRED_API_KEY=your_key_here

### 3. Run the pipeline (in order)

    python src/01_fetch_macro.py
    python src/02_fetch_market.py
    python src/03_build_dataset.py
    python src/04_feature_engineering.py
    python src/05_extract_fedwatch.py
    python src/06_extract_polymarket.py
    python src/07_clean_polymarket.py
    python src/08_extract_kalshi.py
    python src/09_kalshi_to_probs.py
    python src/10_final_merge.py
    python src/11_model_production.py
    python src/12_make_plots.py

The final result is printed to console and saved to `data/processed/final_predictions.csv`.

---

## 🧠 Methodology

### Hybrid Model

The system uses **rules + ML**:

    For each FOMC meeting:
        IF we have market data (Kalshi / Polymarket / FedWatch):
            IF prob_cut > 60% OR prob_hike > 60%:
                predict = "change"
            ELIF prob_hold > 60%:
                predict = "hold"
            ELSE:
                use ML prediction
        ELSE:
            use ML prediction (Random Forest on macro features)

### Random Forest Configuration

- n_estimators = 100
- class_weight = 'balanced'
- random_state = 42
- Handles NaN natively

### Features (24 total)

**Macro (20):**
fed_funds, cpi, unemployment, gdp, treasury_2y, treasury_10y, btc_close, dxy_close, gld_close, spy_close, fed_funds_change, cpi_change, cpi_yoy, unemployment_change, yield_spread, treasury_2y_change, gold_dxy_ratio, spy_momentum, btc_momentum, days_since_last_meeting

**Market (4):**
prob_cut, prob_hold, prob_hike, has_market_data

---

## 📚 Data Sources

| Source | What It Provides | Frequency |
|--------|------------------|-----------|
| FRED | Fed rates, CPI, unemployment, GDP, Treasury yields | Monthly / quarterly |
| yfinance | SPY, GLD, DXY, BTC daily prices | Daily |
| CME FedWatch | Market-implied rate probabilities | Per meeting |
| Kalshi | Fed rate contracts (2024-2025) | Per meeting |
| Polymarket | Fed rate contracts (2023-2025) | Per meeting |


## 📖 Citations

1. Diercks, A., Katz, J. D., & Wright, J. H. (2026). Kalshi and the Rise of Macro Markets. NBER Working Paper w34702.
2. Chan-Lau, J. A., Quach, T. L., & Shi, A. R. (2025). Forecasting Federal Fund Rates with AI. AMRO Working Paper WP/25-10.
3. Xiao Jingyi, F., & Liu, L. (2025). Can We Reliably Predict the Fed's Next Move? arXiv:2506.22763.
4. Gössi, S., et al. (2023). FinBERT-FOMC. ACM ICAIF '23. DOI: 10.1145/3604237.3626843.

---

## 📄 License

This project is for academic use only.

---

## 📬 Contact

**Seyedeh Fatemeh Hosseininassab**
Computer Science Student
GitHub: github.com/fatemeh231
linkedin: https://www.linkedin.com/in/seyedeh-fatemeh-hosseininasab-7320bb322/?isSelfProfile=true