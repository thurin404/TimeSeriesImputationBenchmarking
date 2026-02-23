#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
"""
Quick test to verify Wide partition column name parsing
"""

import pandas as pd

# Test data simulating Wide partition format
test_data = {
    'Date': ['2020-01-01', '2020-01-02'],
    '5904_WIEN-HOHE WARTE:Wind_speed': [1.5, 2.0],
    '5925_WIEN-INNERE STADT:Air_temperature': [10.2, 11.5],
    '3202_LINZ-STADT:Relative_humidity': [65, 70]
}

df = pd.DataFrame(test_data)

print("Test DataFrame (Wide format):")
print(df)
print("\nColumns:")
print(df.columns.tolist())

# Test parsing logic from evaluation.py
print("\n" + "="*60)
print("Testing station/feature parsing for Wide partition:")
print("="*60)

partition_method = "Wide"
partition = "wide"  # Dictionary key

for element in df.columns:
    if element == 'Date':
        continue
        
    # Logic from evaluation.py
    if partition_method.lower() == "wide":
        if ":" in element:
            station_name, feature_name = element.split(":", 1)
        else:
            station_name = partition
            feature_name = element
    else:
        station_name = partition
        feature_name = element
    
    print(f"\nColumn: {element}")
    print(f"  → Station: {station_name}")
    print(f"  → Feature: {feature_name}")

print("\n✅ Wide partition parsing test completed!")
