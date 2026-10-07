import pandas as pd
from pathlib import Path

base = r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor'
data_folder = Path(base) / 'data' / 'polymarket_fed_analysis-main' / 'poly_fed_data'

# Find all polymarket price files
poly_files = sorted(data_folder.glob('polymarket-price-data-*.csv'))
print(f"Found {len(poly_files)} Polymarket files\n")

results = []

for f in poly_files:
    # Extract meeting date: polymarket-price-data-20230614.csv
    meeting_date_str = f.stem.split('-')[-1]
    meeting_date = pd.to_datetime(meeting_date_str, format='%Y%m%d')

    # Load
    df = pd.read_csv(f)

    # Find columns dynamically
    cut_cols = [c for c in df.columns if 'decrease' in c.lower()]
    hold_cols = [c for c in df.columns if '0 bps' in c or 'no change' in c.lower()]
    hike_cols = [c for c in df.columns if 'increase' in c.lower()]

    # Take the last row (closest to meeting)
    last_row = df.iloc[-1]

    prob_cut = sum(last_row[c] for c in cut_cols)
    prob_hold = sum(last_row[c] for c in hold_cols)
    prob_hike = sum(last_row[c] for c in hike_cols)

    last_date = last_row['Date (UTC)']

    results.append({
        'meeting_date': meeting_date,
        'source': 'polymarket',
        'prob_cut': round(prob_cut, 4),
        'prob_hold': round(prob_hold, 4),
        'prob_hike': round(prob_hike, 4),
        'last_data_date': last_date
    })

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

output = Path(base) / 'data' / 'processed' / 'polymarket_extracted.csv'
result_df.to_csv(output, index=False)
print(f"\nSaved to: {output}")