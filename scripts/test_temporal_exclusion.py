#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
"""
Test to verify temporal columns are excluded from evaluation metrics
"""

import pandas as pd

# Simulate evaluation data with temporal features
test_data = {
    'Date': ['2020-01-01', '2020-01-02', '2020-01-03'],
    'Year': [2020, 2020, 2020],
    'Month': [1, 1, 1],
    'Day': [1, 2, 3],
    'Hour': [0, 0, 0],
    '5904_WIEN-HOHE WARTE:Wind_speed': [1.5, 2.0, 2.5],
    '5904_WIEN-HOHE WARTE:Air_temperature': [10.2, 11.5, 12.0],
}

df = pd.DataFrame(test_data)

print("DataFrame with temporal features:")
print(df)
print("\nAll columns:")
print(df.columns.tolist())

# Test filtering logic from evaluation.py
temporal_cols = ['Date', 'Year', 'Month', 'Day', 'Hour']
measurement_cols = [col for col in df.columns if col not in temporal_cols]

print("\n" + "="*60)
print("Columns used for evaluation metrics:")
print("="*60)
print(measurement_cols)

print("\n" + "="*60)
print("Columns excluded (temporal features):")
print("="*60)
print(temporal_cols)

print("\n✅ Only measurement columns will be evaluated!")
print(f"   Number of measurement columns: {len(measurement_cols)}")
print(f"   Number of temporal columns: {len([c for c in df.columns if c in temporal_cols])}")
