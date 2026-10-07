import pandas as pd
import re
from pathlib import Path

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor')
df = pd.read_csv(base / 'data' / 'processed' / 'kalshi_fed_markets.csv')

# Keep only FED-24* and FED-25* meetings
meeting_pattern = re.compile(r'^FED-(\d{2})([A-Z]{3})$')
df['is_meeting'] = df['event_ticker'].apply(lambda x: bool(meeting_pattern.match(str(x))))

meetings = df[df['is_meeting']].copy()
print(f"FOMC meeting markets found: {len(meetings)}")

# Extract year + month
meetings['year'] = meetings['event_ticker'].str.extract(r'FED-(\d{2})')[0].apply(lambda x: 2000 + int(x))
meetings['month'] = meetings['event_ticker'].str.extract(r'FED-\d{2}([A-Z]{3})')[0]

month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
             'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
meetings['month_num'] = meetings['month'].map(month_map)

unique_meetings = meetings[['year', 'month_num', 'event_ticker']].drop_duplicates()
unique_meetings = unique_meetings.sort_values(['year', 'month_num'])
print("\nUnique meetings found:")
print(unique_meetings.to_string(index=False))

# Extract consensus per meeting
consensus_list = []
for event_ticker in unique_meetings['event_ticker']:
    strikes = meetings[meetings['event_ticker'] == event_ticker].copy()
    strikes = strikes.dropna(subset=['floor_strike'])

    if len(strikes) == 0:
        print(f"  Skipping {event_ticker} - no strikes")
        continue

    strikes = strikes.sort_values('floor_strike')
    yes_strikes = strikes[strikes['result'] == 'yes']

    if len(yes_strikes) > 0:
        consensus_rate = yes_strikes['floor_strike'].max()
    else:
        consensus_rate = None

    no_strikes = strikes[
        (strikes['result'] == 'no') &
        (strikes['floor_strike'] > (consensus_rate if consensus_rate is not None else 0))
    ]
    next_no = no_strikes['floor_strike'].min() if len(no_strikes) > 0 else None

    consensus_list.append({
        'event_ticker': event_ticker,
        'year': int(strikes['year'].iloc[0]),
        'month': int(strikes['month_num'].iloc[0]),
        'consensus_low': consensus_rate,
        'consensus_high': next_no,
        'n_strikes': len(strikes)
    })

consensus_df = pd.DataFrame(consensus_list)

# Add meeting date
consensus_df['meeting_date'] = pd.to_datetime(
    dict(year=consensus_df['year'], month=consensus_df['month'], day=1)
)

print("\n" + "=" * 70)
print("MARKET CONSENSUS BY MEETING")
print("=" * 70)
print(consensus_df.to_string(index=False))

# Save
output = base / 'data' / 'processed' / 'kalshi_meeting_consensus.csv'
consensus_df.to_csv(output, index=False)
print(f"\nSaved to: {output}")