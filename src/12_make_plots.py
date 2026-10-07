# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:51:56 2026

@author: fatemeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

base = Path(r'C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\fomc-predictor')
figs = base / 'reports' / 'figures'
figs.mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
pred_df = pd.read_csv(base / 'data' / 'processed' / 'final_predictions.csv')
df = pd.read_parquet(base / 'data' / 'processed' / 'master_final_v3.parquet')
df = df.sort_values('date').reset_index(drop=True)
df['target'] = (df['decision'] != 'hold').astype(int)

# ============================================================
# PLOT 1: Accuracy by Method (bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

methods = {
    'market→hold': 'Market says HOLD\n(rule-based)',
    'market→change': 'Market says CHANGE\n(rule-based)',
    'ML (no market data)': 'ML only\n(no market data)',
}
labels = []
accs = []
counts = []

for key, label in methods.items():
    sub = pred_df[pred_df['method'] == key]
    if len(sub) > 0:
        labels.append(f"{label}\n(n={len(sub)})")
        accs.append(sub['correct'].mean() * 100)
        counts.append(len(sub))

colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax.bar(labels, accs, color=colors, width=0.6, edgecolor='black', linewidth=1)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 1.5, f"{acc:.1f}%",
            ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy by Prediction Method', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(y=95.45, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.text(2.4, 96.5, 'Overall: 95.45%', color='red', fontsize=10, ha='right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figs / 'plot1_accuracy_by_method.png', dpi=200, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 2: Confusion Matrix
# ============================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred_df['actual_bin'], pred_df['predicted'], labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())

for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        ax.text(j, i, cm[i, j], ha='center', va='center',
                fontsize=20, fontweight='bold', color=text_color)

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted: HOLD', 'Predicted: CHANGE'], fontsize=11)
ax.set_yticklabels(['Actual: HOLD', 'Actual: CHANGE'], fontsize=11)
ax.set_title('Confusion Matrix — Test Set (22 meetings)', fontsize=13, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(figs / 'plot2_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 3: Feature Importance (Top 10)
# ============================================================
macro_features = [
    'fed_funds', 'cpi', 'unemployment', 'gdp',
    'treasury_2y', 'treasury_10y',
    'btc_close', 'dxy_close', 'gld_close', 'spy_close',
    'fed_funds_change', 'cpi_change', 'cpi_yoy', 'unemployment_change',
    'yield_spread', 'treasury_2y_change', 'gold_dxy_ratio',
    'spy_momentum', 'btc_momentum', 'days_since_last_meeting'
]
all_features = macro_features + ['prob_cut', 'prob_hold', 'prob_hike', 'has_market_data']

train_df = df[df['date'] < pd.Timestamp('2024-01-01')]
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(train_df[all_features], train_df['target'])

imp = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#A23B72' if 'prob_' in f or 'has_market' in f else '#2E86AB' for f in imp['feature']]
ax.barh(imp['feature'], imp['importance'], color=colors, edgecolor='black', linewidth=0.7)

ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 10 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#A23B72', edgecolor='black', label='Market features'),
    Patch(facecolor='#2E86AB', edgecolor='black', label='Macro features'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(figs / 'plot3_feature_importance.png', dpi=200, bbox_inches='tight')
plt.show()

# ============================================================
# PLOT 4: Timeline of Predictions
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))

pred_df['date'] = pd.to_datetime(pred_df['date'])

for i, row in pred_df.iterrows():
    if row['correct']:
        color = '#2E86AB' if row['actual_bin'] == 0 else '#A23B72'
        marker = 'o'
    else:
        color = 'red'
        marker = 'X'
    ax.scatter(row['date'], row['actual_bin'], color=color, s=180, marker=marker,
               zorder=3, edgecolor='black', linewidth=1)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB',
           markersize=12, markeredgecolor='black', label='Correct — HOLD'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#A23B72',
           markersize=12, markeredgecolor='black', label='Correct — CHANGE'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
           markersize=14, markeredgecolor='black', label='Incorrect'),
]
ax.legend(handles=legend_elements, loc='center left', fontsize=10)

ax.set_yticks([0, 1])
ax.set_yticklabels(['HOLD', 'CHANGE'], fontsize=12)
ax.set_ylabel('Actual Decision', fontsize=12)
ax.set_title('Test Set Predictions Over Time (2024–2026)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(figs / 'plot4_timeline.png', dpi=200, bbox_inches='tight')
plt.show()

print("=" * 60)
print("ALL PLOTS SAVED")
print("=" * 60)
print(f"Location: {figs}")
print("  1. plot1_accuracy_by_method.png")
print("  2. plot2_confusion_matrix.png")
print("  3. plot3_feature_importance.png")
print("  4. plot4_timeline.png")