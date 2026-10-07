import pandas as pd
from pathlib import Path

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor')

kalshi = pd.read_csv(base / 'data' / 'processed' / 'kalshi_meeting_consensus.csv')
kalshi['meeting_date'] = pd.to_datetime(kalshi['meeting_date'])

master = pd.read_parquet(base / 'data' / 'processed' / 'master_with_market.parquet')
master['date'] = pd.to_datetime(master['date'])
master = master.sort_values('date').reset_index(drop=True)

# Compute "prior rate" — the fed_funds from the PREVIOUS meeting
master['prior_fed_funds'] = master['fed_funds'].shift(1)
# For the first meeting, fall back to current
master['prior_fed_funds'] = master['prior_fed_funds'].fillna(master['fed_funds'])

print(f"Kalshi meetings: {len(kalshi)}")
print(f"Master rows:     {len(master)}")
print()

results = []

for _, k_row in kalshi.iterrows():
    match = master[
        (master['date'].dt.year == k_row['year']) &
        (master['date'].dt.month == k_row['month'])
    ]

    if len(match) == 0:
        print(f"  SKIP {k_row['event_ticker']} - no master match")
        continue

    m_row = match.iloc[0]

    # Use PRIOR meeting's rate as the "before" reference
    prior_rate = m_row['prior_fed_funds']
    current_upper = prior_rate + 0.125

    consensus_high = k_row['consensus_high'] if pd.notna(k_row['consensus_high']) else k_row['consensus_low']
    diff = consensus_high - current_upper

    if diff < -0.10:
        prob_cut, prob_hold, prob_hike = 0.85, 0.10, 0.05
        classification = 'cut'
    elif diff > 0.10:
        prob_cut, prob_hold, prob_hike = 0.05, 0.10, 0.85
        classification = 'hike'
    else:
        prob_cut, prob_hold, prob_hike = 0.10, 0.80, 0.10
        classification = 'hold'

    results.append({
        'meeting_date':   m_row['date'],
        'event_ticker':   k_row['event_ticker'],
        'actual':         m_row['decision'],
        'classified':     classification,
        'match':          m_row['decision'] == classification,
        'prob_cut':       prob_cut,
        'prob_hold':      prob_hold,
        'prob_hike':      prob_hike,
        'market_source':  'kalshi',
        'prior_rate':     round(prior_rate, 3),
        'current_upper':  round(current_upper, 3),
        'consensus_high': k_row['consensus_high'],
        'diff':           round(diff, 3)
    })

kalshi_probs = pd.DataFrame(results)

print("=" * 105)
print("KALSHI CLASSIFICATION RESULTS (using prior meeting's rate)")
print("=" * 105)
display_cols = ['meeting_date', 'event_ticker', 'actual', 'classified', 'match',
                'prior_rate', 'consensus_high', 'diff']
print(kalshi_probs[display_cols].to_string(index=False))

n_correct = kalshi_probs['match'].sum()
n_total = len(kalshi_probs)
print()
print("=" * 105)
print(f"Correct: {n_correct} / {n_total} ({100*n_correct/n_total:.1f}%)")
print("=" * 105)

output = base / 'data' / 'processed' / 'kalshi_probs.csv'
kalshi_probs.to_csv(output, index=False)
print(f"\nSaved to: {output}")