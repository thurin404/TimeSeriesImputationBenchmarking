#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
"""
Script to add metadata columns to existing evaluation_results.csv
Preserves existing results and adds:
- scaling: false (not scaled)
- resample: true (resampled to 1d)
- resample_freq: '1d' (daily frequency)
"""

import pandas as pd
import os

def add_metadata_columns():
    """Add metadata columns to existing evaluation results"""
    
    results_file = "evaluation_results.csv"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return
    
    # Read existing results
    print(f"Reading {results_file}...")
    df = pd.read_csv(results_file)
    
    print(f"Current shape: {df.shape}")
    print(f"Current columns: {df.columns.tolist()}")
    
    # Check if metadata columns already exist
    metadata_cols = ['scaling', 'resample', 'resample_freq']
    existing_metadata = [col for col in metadata_cols if col in df.columns]
    
    if existing_metadata:
        print(f"Warning: Metadata columns already exist: {existing_metadata}")
        print("Updating values...")
    
    # Add or update metadata columns based on params.yaml comment:
    # "preserve the already gathered results. they were not resampled and scaled."
    # But looking at params.yaml, resample is true, so the data WAS resampled
    # The user means it was NOT scaled (scaling: false)
    df['scaling'] = False
    df['resample'] = True
    df['resample_freq'] = '1d'
    
    # Create backup
    backup_file = "evaluation_results_backup.csv"
    if os.path.exists(backup_file):
        print(f"Backup already exists: {backup_file}")
    else:
        print(f"Creating backup: {backup_file}")
        df_original = pd.read_csv(results_file)
        df_original.to_csv(backup_file, index=False)
    
    # Save updated results
    print(f"Saving updated results to {results_file}...")
    df.to_csv(results_file, index=False)
    
    print(f"Updated shape: {df.shape}")
    print(f"Updated columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Show unique values for metadata columns
    print("\nMetadata column values:")
    print(f"scaling: {df['scaling'].unique()}")
    print(f"resample: {df['resample'].unique()}")
    print(f"resample_freq: {df['resample_freq'].unique()}")
    
    print("\n✅ Successfully added metadata columns!")
    print(f"📁 Backup saved to: {backup_file}")

if __name__ == "__main__":
    add_metadata_columns()
