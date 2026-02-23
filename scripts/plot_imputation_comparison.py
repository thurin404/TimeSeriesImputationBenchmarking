#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "numpy",
#   "pyyaml",
#   "scikit-learn",
# ]
# ///
"""
Create detailed time series plots comparing ground truth and all imputation methods.
Picks one example per feature with missing rate <= 40% spanning 1 month to 1 year.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import yaml
import joblib
from pathlib import Path

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def apply_inverse_scaling(data: pd.DataFrame, scaler_dir: str) -> pd.DataFrame:
    """Apply inverse scaling transformation to data using saved scalers.
    
    Args:
        data: DataFrame to inverse transform (wide format with MultiIndex columns)
        scaler_dir: Directory containing saved scaler files
    
    Returns:
        Inverse-transformed DataFrame
    """
    scaler_file = os.path.join(scaler_dir, "scaler_global.pkl")
    
    if not os.path.exists(scaler_file):
        print(f"Warning: Global scaler file {scaler_file} not found. Skipping inverse scaling.")
        return data
    
    try:
        scaler = joblib.load(scaler_file)
        
        # Apply inverse transform to the wide format data
        # The scaler was fitted on all columns (Station, Feature) tuples
        df_copy = data.copy()
        
        # Inverse transform all columns at once
        inverse_transformed = scaler.inverse_transform(df_copy.values)
        df_copy.iloc[:, :] = inverse_transformed
        
        print(f"  Applied inverse scaling using {scaler_file}")
        return df_copy
    except Exception as e:
        print(f"  Warning: Error applying inverse scaling: {e}")
        return data

def find_suitable_series(df_mask, df_ground_truth, features, max_missing_rate=0.30):
    """
    Find one suitable time series for each feature to plot.
    Suitable means having a reasonable amount of missing data (between 5% and max_missing_rate).
    
    Args:
        df_mask: Mask dataframe with MultiIndex columns (Station, Feature)
        df_ground_truth: Ground truth dataframe with MultiIndex columns (Station, Feature)
        features: List of feature names
        max_missing_rate: Maximum missing rate to consider
    
    Returns:
        dict: {feature: (column_tuple, missing_rate)}
    """
    selected = {}
    
    for feature in features:
        # Find columns for this feature (columns are tuples: (Station, Feature))
        if isinstance(df_mask.columns, pd.MultiIndex):
            feature_cols = [col for col in df_mask.columns if col[1] == feature]
        else:
            # Fallback for non-MultiIndex
            feature_cols = [col for col in df_mask.columns 
                           if feature in str(col) and col not in ['Date', 'Year', 'Month', 'Day', 'Hour']]
        
        # For each potential series
        best_candidate = None
        best_missing_rate = 0
        
        for col in feature_cols:
            if col in df_mask.columns and col in df_ground_truth.columns:
                # Calculate missing rate
                mask_series = df_mask[col]
                missing_rate = mask_series.sum() / len(mask_series)
                
                # Check if this is a good candidate
                if 0.05 <= missing_rate <= max_missing_rate:
                    # Prefer higher missing rates (more interesting to visualize)
                    if best_candidate is None or missing_rate > best_missing_rate:
                        best_candidate = col
                        best_missing_rate = missing_rate
        
        if best_candidate:
            selected[feature] = (best_candidate, best_missing_rate)
            # Format column name for display
            if isinstance(best_candidate, tuple):
                col_display = f"{best_candidate[0]}:{best_candidate[1]}"
            else:
                col_display = str(best_candidate)
            print(f"Selected for {feature}: {col_display} (missing rate: {best_missing_rate:.1%})")
    
    return selected

def plot_imputation_comparison(data_dict, output_dir, params):
    """
    Create comprehensive plots comparing all imputation methods.
    
    Args:
        data_dict: Dictionary with keys 'ground_truth', 'mask', and imputation method names
                  Data is in long format (Date, Station, Feature, Value)
        output_dir: Where to save plots
        params: Configuration parameters
    """
    ground_truth_long = data_dict['ground_truth']
    mask_long = data_dict['mask']
    methods = [k for k in data_dict.keys() if k not in ['ground_truth', 'mask']]
    
    partition_by = params['prepare_working_files']['partition_by']
    resample = params['preprocessing']['resample']
    resample_freq = params['preprocessing'].get('resample_freq', '1h')
    
    print(f"\nPartition by: {partition_by}")
    print(f"Resample: {resample} ({'to ' + resample_freq if resample else 'no resampling'})")
    print(f"Found {len(methods)} imputation methods")
    
    # Convert from long to wide format for plotting
    # Long format: Date, Station, Feature, Value
    # Wide format: Date as index, columns as (Station, Feature) tuples
    print(f"Converting from long to wide format...")
    ground_truth_long['Date'] = pd.to_datetime(ground_truth_long['Date'])
    mask_long['Date'] = pd.to_datetime(mask_long['Date'])
    
    ground_truth = ground_truth_long.pivot(index='Date', columns=['Station', 'Feature'], values='Value')
    mask = mask_long.pivot(index='Date', columns=['Station', 'Feature'], values='Value')
    
    # Convert imputed data to wide format and apply inverse scaling
    imputed_wide = {}
    scaling = params['prepare_working_files'].get('scaling', False)
    scaler_dir = 'data/00_tools/scalers'
    
    if scaling:
        print(f"Applying inverse scaling to imputed data...")
    
    for method in methods:
        data_dict[method]['Date'] = pd.to_datetime(data_dict[method]['Date'])
        temp_wide = data_dict[method].pivot(index='Date', columns=['Station', 'Feature'], values='Value')
        
        # Apply inverse scaling if scaling was used
        if scaling:
            temp_wide = apply_inverse_scaling(temp_wide, scaler_dir)
        
        imputed_wide[method] = temp_wide
    
    # Extract unique features from the multi-level column index
    # Columns are now tuples: (Station, Feature)
    if isinstance(ground_truth.columns, pd.MultiIndex):
        features = list(set([col[1] for col in ground_truth.columns]))
    else:
        # Fallback if not multi-index (shouldn't happen with pivot)
        features = ['Wind_speed', 'Air_temperature', 'Relative_humidity', 'Air_pressure', 'Wind_direction']
    
    print(f"Detected features: {features}")
    
    # Find suitable series for plotting
    selected_series = find_suitable_series(mask, ground_truth, features, max_missing_rate=0.40)
    
    if not selected_series:
        print("No suitable series found for plotting!")
        return
    
    # Create plots for each selected feature
    for feature, (col_name, missing_rate) in selected_series.items():
        print(f"\nCreating plot for {feature} ({col_name})...")
        
        # Determine time window based on resampling
        total_points = len(ground_truth)
        if resample and 'd' in resample_freq.lower():
            # Daily data - show up to 1 year or all data
            max_points = min(365, total_points)
        elif resample and 'w' in resample_freq.lower():
            # Weekly data - show up to 1 year
            max_points = min(52, total_points)
        else:
            # Hourly or no resampling - show 1 month
            max_points = min(30 * 24, total_points)
        
        # Use the last max_points for visualization (most recent data)
        idx_start = max(0, total_points - max_points)
        
        # Extract data - Date is now the index after pivot
        dates = ground_truth.index[idx_start:]
        gt = ground_truth[col_name].iloc[idx_start:].values
        mask_vals = mask[col_name].iloc[idx_start:].values
        
        # Get imputed values from all methods (use imputed_wide, not data_dict)
        imputed_data = {}
        for method in methods:
            if col_name in imputed_wide[method].columns:
                imputed_data[method] = imputed_wide[method][col_name].iloc[idx_start:].values
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 6))
        
        # Plot ground truth
        ax.plot(dates, gt, 'k-', linewidth=2, label='Ground Truth', alpha=0.7, zorder=10)
        
        # Highlight missing regions
        missing_mask = mask_vals == 1
        if missing_mask.any():
            # Create patches for missing regions
            ax.fill_between(dates, gt.min() - 0.1 * (gt.max() - gt.min()), 
                           gt.max() + 0.1 * (gt.max() - gt.min()),
                           where=missing_mask, alpha=0.15, color='red', 
                           label='Missing Data Region', zorder=1)
        
        # Plot imputed values for missing parts
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(methods)))
        for i, method in enumerate(sorted(imputed_data.keys())):
            imp = imputed_data[method]
            # Only plot imputed values where data was missing
            imp_masked = imp.copy()
            imp_masked[~missing_mask] = np.nan
            ax.plot(dates, imp_masked, 'o-', color=colors[i], 
                   label=f'{method} (imputed)', markersize=4, linewidth=1.5, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{feature} Value', fontsize=12, fontweight='bold')
        
        # Extract station name from column tuple or string
        if isinstance(col_name, tuple):
            # Column is (Station, Feature) tuple
            station, feat = col_name
            title = f'{feat} at {station}'
        elif ':' in str(col_name):
            station, feat = str(col_name).split(':', 1)
            title = f'{feat} at {station}'
        else:
            title = f'{feature} - {col_name}'
        
        ax.set_title(f'Imputation Comparison: {title}\n'
                    f'Missing Rate: {missing_rate:.1%} | '
                    f'Time Period: {dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, 
                 framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Improve date formatting
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save plot - handle tuple column names
        if isinstance(col_name, tuple):
            safe_filename = f"{col_name[0]}_{col_name[1]}".replace(':', '_').replace('/', '-').replace(' ', '_')
        else:
            safe_filename = str(col_name).replace(':', '_').replace('/', '-').replace(' ', '_')
        output_file = os.path.join(output_dir, f'imputation_comparison_{safe_filename}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")

def main():
    """Main function to load data and create plots"""
    
    if len(sys.argv) < 3:
        print("Usage: python plot_imputation_comparison.py <working_files_dir> <imputed_dir> <output_dir>")
        print("Example: python plot_imputation_comparison.py data/04_working_files data/05_imputed data/08_plots")
        sys.exit(1)
    
    working_dir = sys.argv[1]
    imputed_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/08_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parameters
    params = load_params()
    partition_by = params['prepare_working_files']['partition_by']
    
    print("="*70)
    print("IMPUTATION COMPARISON VISUALIZATION")
    print("="*70)
    print(f"Working files directory: {working_dir}")
    print(f"Imputed files directory: {imputed_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Partition method: {partition_by}")
    
    # Load original data and mask from the imputed directory structure
    # New structure: data/05_imputed/original/ts_feature_imputed.csv
    #                data/05_imputed/mask/ts_feature_imputed.csv
    original_file = os.path.join(imputed_dir, "original", f"ts_{partition_by.lower()}_imputed.csv")
    mask_file = os.path.join(imputed_dir, "mask", f"ts_{partition_by.lower()}_imputed.csv")
    
    if not os.path.exists(original_file):
        print(f"ERROR: Original data file not found: {original_file}")
        sys.exit(1)
    
    if not os.path.exists(mask_file):
        print(f"ERROR: Mask file not found: {mask_file}")
        sys.exit(1)
    
    print(f"\nLoading data from new structure...")
    print(f"  Ground truth: {original_file}")
    print(f"  Mask: {mask_file}")
    
    # Get imputation methods (exclude 'original' and 'mask' directories)
    imputed_subdirs = [d for d in os.listdir(imputed_dir) 
                      if os.path.isdir(os.path.join(imputed_dir, d)) 
                      and d not in ['original', 'mask']]
    
    print(f"\nFound {len(imputed_subdirs)} imputation methods:")
    for method in sorted(imputed_subdirs):
        print(f"  - {method}")
    
    # Load ground truth and mask
    data_dict = {
        'ground_truth': pd.read_csv(original_file),
        'mask': pd.read_csv(mask_file)
    }
    
    print(f"  Ground truth shape: {data_dict['ground_truth'].shape}")
    print(f"  Mask shape: {data_dict['mask'].shape}")
    
    # Load imputed data from each method
    for method in imputed_subdirs:
        method_file = os.path.join(imputed_dir, method, f"ts_{partition_by.lower()}_imputed.csv")
        if os.path.exists(method_file):
            data_dict[method] = pd.read_csv(method_file)
            print(f"  Loaded {method}: shape {data_dict[method].shape}")
        else:
            print(f"  WARNING: File not found for {method}: {method_file}")
    
    # Create plots
    if len(data_dict) > 2:  # Has ground truth, mask, and at least one imputation method
        plot_imputation_comparison(data_dict, output_dir, params)
    else:
        print("ERROR: Not enough data loaded to create plots!")
    
    print("\n" + "="*70)
    print("✅ Imputation comparison plots created successfully!")
    print(f"📁 Plots saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
