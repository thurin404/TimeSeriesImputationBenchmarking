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
Create detailed time series plots comparing ground truth and imputation methods.
One plot per feature with 4 subplots to distinguish datasets clearly.
Shows missing and non-missing parts with fixed time periods.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.dates import AutoDateLocator
import yaml
from pathlib import Path
from datetime import datetime
import joblib

# Configuration for each feature
FEATURE_CONFIG = {
    'Air_temperature': {
        'station': '4305_ZWERNDORF',
        'missing_rate': 0.211,
        'ylabel': 'Temperature (K)',
        'time_period': ('2015-03-01', '2015-05-31'),  # Spring 2015
    },
    'Wind_speed': {
        'station': '12504_BISCHOFSHOFEN',
        'missing_rate': 0.161,
        'ylabel': 'Wind Speed (m/s)',
        'time_period': ('2015-04-01', '2015-04-07'),  # 1 week - less variability, clearer patterns
    },
    'Air_pressure': {
        'station': '5972_GROSS-ENZERSDORF',
        'missing_rate': 0.208,
        'ylabel': 'Pressure (hPa)',
        'time_period': ('2016-02-01', '2016-04-30'),  # Winter-Spring 2016 with missing data
    },
    'Relative_humidity': {
        'station': '4305_ZWERNDORF',
        'missing_rate': 0.211,
        'ylabel': 'Relative Humidity (%)',
        'time_period': ('2015-03-01', '2015-05-31'),  # Spring 2015
    },
    'Wind_direction': {
        'station': '12504_BISCHOFSHOFEN',
        'missing_rate': 0.181,
        'ylabel': 'Wind Direction (°)',
        'time_period': ('2015-04-01', '2015-04-07'),  # 1 week - less variability, clearer patterns
    }
}

# Method categorization for logical distribution
METHOD_CATEGORIES = {
    'Simple/Statistical': ['mean', 'ffill', 'interpolation'],
    'ML-based': ['knn', 'iterative', 'xgboost'],
    'Classical Deep Learning': ['brits', 'gpvae', 'saits', 'frets'],
    'Transformer/Foundation': ['imputeformer', 'moment', 'tefn', 'timemixer', 'timesnet', 'tslanet']
}

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)
    return params

def apply_inverse_scaling(data: pd.DataFrame, scaler_dir: str) -> pd.DataFrame:
    """Apply inverse scaling transformation to data using saved scalers.
    
    Args:
        data: DataFrame to inverse transform (wide format with MultiIndex columns)
        scaler_dir: Directory containing saved scaler files
    
    Returns:
        Inverse-transformed DataFrame with original structure preserved
    """
    scaler_file = os.path.join(scaler_dir, "scaler_global.pkl")
    
    if not os.path.exists(scaler_file):
        print(f"Warning: Global scaler file {scaler_file} not found. Skipping inverse scaling.")
        return data
    
    try:
        scaler = joblib.load(scaler_file)
        
        # Preserve the original index and columns
        original_index = data.index
        original_columns = data.columns
        
        # Apply inverse transform
        inverse_transformed = scaler.inverse_transform(data.values)
        
        # Create new DataFrame with original structure
        df_result = pd.DataFrame(
            inverse_transformed,
            index=original_index,
            columns=original_columns
        )
        
        return df_result
    except Exception as e:
        print(f"  Warning: Error applying inverse scaling: {e}")
        return data

def load_feature_data(feature, working_dir, imputed_dir, params):
    """
    Load ground truth, mask, and imputed data for a specific feature from new long-format structure.
    
    Args:
        feature: Feature name to load
        working_dir: Directory containing working files (not used in new structure)
        imputed_dir: Directory containing imputed data in new structure
        params: Parameters dict with scaling info
    
    Returns:
        dict: {
            'ground_truth': DataFrame (wide format with stations as columns),
            'mask': DataFrame (wide format with stations as columns),
            'imputed': {method: DataFrame (wide format with stations as columns)}
        }
    """
    # Load from new structure: data/05_imputed/original/ and data/05_imputed/mask/
    partition_by = params['prepare_working_files']['partition_by']
    gt_file = os.path.join(imputed_dir, "original", f"ts_{partition_by.lower()}_imputed.csv")
    mask_file = os.path.join(imputed_dir, "mask", f"ts_{partition_by.lower()}_imputed.csv")
    
    if not os.path.exists(gt_file) or not os.path.exists(mask_file):
        print(f"Warning: Files not found - {gt_file} or {mask_file}")
        return None
    
    # Load and convert from long to wide format
    # Filter by feature first to reduce memory usage
    gt_long = pd.read_csv(gt_file)
    mask_long = pd.read_csv(mask_file)
    
    # Filter for specific feature
    gt_long = gt_long[gt_long['Feature'] == feature].copy()
    mask_long = mask_long[mask_long['Feature'] == feature].copy()
    
    if len(gt_long) == 0:
        print(f"  Warning: No data found for feature {feature}")
        return None
    
    gt_long['Date'] = pd.to_datetime(gt_long['Date'])
    mask_long['Date'] = pd.to_datetime(mask_long['Date'])
    
    # Pivot to wide format - now columns are just Station names
    gt_df = gt_long.pivot(index='Date', columns='Station', values='Value')
    mask_df = mask_long.pivot(index='Date', columns='Station', values='Value')
    
    # Reset index to make Date a column
    gt_df = gt_df.reset_index()
    mask_df = mask_df.reset_index()
    
    # Load imputed data from different methods and apply inverse scaling
    imputed_data = {}
    scaling = params['prepare_working_files'].get('scaling', False)
    scaler_dir = 'data/00_tools/scalers'
    
    for method in os.listdir(imputed_dir):
        method_dir = os.path.join(imputed_dir, method)
        if not os.path.isdir(method_dir) or method in ['original', 'mask']:
            continue
            
        imp_file = os.path.join(method_dir, f"ts_{partition_by.lower()}_imputed.csv")
        if os.path.exists(imp_file):
            imp_long = pd.read_csv(imp_file)
            
            # Filter for specific feature first
            imp_long = imp_long[imp_long['Feature'] == feature].copy()
            
            if len(imp_long) == 0:
                print(f"  Warning: No data found for {feature} in {method}")
                continue
            
            imp_long['Date'] = pd.to_datetime(imp_long['Date'])
            
            # Pivot to wide format - columns are just Station names
            imp_df = imp_long.pivot(index='Date', columns='Station', values='Value')
            
            # Note: Inverse scaling is skipped here because the scaler was trained on all features together,
            # but we're only loading one feature at a time. The scaler dimensions won't match.
            # For visualization purposes, we'll skip inverse scaling in this script.
            
            # Reset index to make Date a column
            imp_df = imp_df.reset_index()
            
            imputed_data[method] = imp_df
    
    return {
        'ground_truth': gt_df,
        'mask': mask_df,
        'imputed': imputed_data
    }

def plot_feature_comparison(feature, data_dict, config, output_dir):
    """
    Create a 2x2 subplot figure with all imputation methods distributed across 4 subplots.
    Methods are grouped by category (Simple, ML-based, Classical DL, Transformer).
    Each subplot shows ground truth vs multiple imputation methods, with missing regions highlighted.
    """
    station = config['station']
    ylabel = config['ylabel']
    missing_rate = config['missing_rate']
    start_date, end_date = config['time_period']
    
    # Extract data
    gt_df = data_dict['ground_truth']
    mask_df = data_dict['mask']
    imputed = data_dict['imputed']
    
    # Filter to time period
    mask_time = (gt_df['Date'] >= start_date) & (gt_df['Date'] <= end_date)
    dates = gt_df.loc[mask_time, 'Date'].values
    gt_values = gt_df.loc[mask_time, station].values
    mask_values = mask_df.loc[mask_time, station].values
    
    # Get all imputed values
    all_methods = sorted(imputed.keys())
    method_data = {}
    for method in all_methods:
        imp_df = imputed[method]
        # Apply same time mask to imputed data
        imp_mask_time = (imp_df['Date'] >= start_date) & (imp_df['Date'] <= end_date)
        
        # Check if station exists in this method's dataframe
        if station not in imp_df.columns:
            print(f"  Warning: Station {station} not found in {method}, skipping")
            continue
        
        imp_values = imp_df.loc[imp_mask_time, station].values
        
        # Verify array lengths match
        if len(imp_values) != len(gt_values):
            print(f"  Warning: Length mismatch for {method}: {len(imp_values)} vs {len(gt_values)}, skipping")
            continue
            
        method_data[method] = imp_values
    
    if len(method_data) == 0:
        print(f"Warning: No imputation methods available for {feature}")
        return None
    
    # Categorize methods by type (only methods that have data)
    available_methods = list(method_data.keys())
    categorized_methods = {category: [] for category in METHOD_CATEGORIES.keys()}
    uncategorized = []
    
    for method in available_methods:
        found = False
        for category, methods_in_cat in METHOD_CATEGORIES.items():
            if method in methods_in_cat:
                categorized_methods[category].append(method)
                found = True
                break
        if not found:
            uncategorized.append(method)
    
    # Distribute methods across 4 subplots by category
    method_groups = []
    subplot_titles = []
    
    for category in ['Simple/Statistical', 'ML-based', 'Classical Deep Learning', 'Transformer/Foundation']:
        if categorized_methods[category]:
            method_groups.append(categorized_methods[category])
            subplot_titles.append(category)
    
    # If we have uncategorized methods, add them to the last subplot or create new one
    if uncategorized:
        if len(method_groups) < 4:
            method_groups.append(uncategorized)
            subplot_titles.append('Other Methods')
        else:
            method_groups[-1].extend(uncategorized)
    
    # Ensure we have exactly 4 subplots (pad with empty if needed)
    while len(method_groups) < 4:
        method_groups.append([])
        subplot_titles.append('(No methods)')
    
    n_methods = len(method_data)
    
    print(f"  Distributing {n_methods} methods across 4 subplots by category:")
    for i, (title, group) in enumerate(zip(subplot_titles, method_groups)):
        if group:
            print(f"    Subplot {i+1} - {title}: {len(group)} methods - {', '.join(group)}")
        else:
            print(f"    Subplot {i+1} - {title}: (empty)")
    
    # Color palette for methods (use only methods that have data)
    available_methods = list(method_data.keys())
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(available_methods)))
    color_map = {method: colors[i] for i, method in enumerate(available_methods)}
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle(f'{feature} Imputation Comparison - {station}\n'
                 f'Missing Rate: {missing_rate:.1%} | Period: {start_date} to {end_date} | '
                 f'{n_methods} Methods Compared',
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    # Create missing mask
    missing_mask = mask_values == 1
    missing_count = missing_mask.sum()
    
    print(f"  Missing data points in time period: {missing_count} ({missing_count/len(missing_mask):.1%})")
    
    # Plot each subplot with its assigned methods
    for subplot_idx, (ax, methods_in_subplot, subplot_title) in enumerate(zip(axes, method_groups, subplot_titles)):
        if not methods_in_subplot:
            ax.text(0.5, 0.5, f'{subplot_title}\n(No methods)', 
                   ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Plot non-missing ground truth (black line)
        gt_non_missing = gt_values.copy()
        gt_non_missing[missing_mask] = np.nan
        ax.plot(dates, gt_non_missing, 'k-', linewidth=2.5, label='Ground Truth (observed)', 
                alpha=0.9, zorder=10)
        
        # Plot missing ground truth (gray dashed line for reference)
        gt_missing = gt_values.copy()
        gt_missing[~missing_mask] = np.nan
        ax.plot(dates, gt_missing, color='gray', linestyle='--', linewidth=1.5, 
                label='Ground Truth (missing)', alpha=0.4, zorder=5)
        
        # Highlight missing regions with background color using fill_between (only where data is missing)
        if missing_mask.any():
            y_min = gt_values.min() - 0.05 * (gt_values.max() - gt_values.min())
            y_max = gt_values.max() + 0.05 * (gt_values.max() - gt_values.min())
            ax.fill_between(dates, y_min, y_max, where=missing_mask, 
                           alpha=0.15, color='red', label='Missing Data Region', zorder=1)
        
        # Plot imputed values for each method in this subplot
        for method in methods_in_subplot:
            imp_values = method_data[method]
            
            # Plot imputed values in missing regions only
            imp_missing = imp_values.copy()
            imp_missing[~missing_mask] = np.nan
            ax.plot(dates, imp_missing, 'o-', color=color_map[method], linewidth=1.8, 
                   markersize=2, label=method, alpha=0.85, zorder=6)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{subplot_title}\n({len(methods_in_subplot)} methods)', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, 
                 framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_locator(AutoDateLocator())
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'imputation_comparison_{feature}_{station}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created plot for {feature}: {output_file}")
    return output_file

def main():
    """Main function to create comparison plots"""
    
    if len(sys.argv) < 3:
        print("Usage: python plot_imputation_detailed.py <working_files_dir> <imputed_dir> [output_dir]")
        print("Example: python plot_imputation_detailed.py data/04_working_files data/05_imputed data/08_plots")
        sys.exit(1)
    
    working_dir = sys.argv[1]
    imputed_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/08_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parameters
    params = load_params()
    scaling = params['prepare_working_files'].get('scaling', False)
    
    print("="*80)
    print("DETAILED IMPUTATION COMPARISON VISUALIZATION")
    print("="*80)
    print(f"Working files directory: {working_dir}")
    print(f"Imputed files directory: {imputed_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Features to plot: {len(FEATURE_CONFIG)}")
    print(f"Scaling enabled: {scaling}")
    print("="*80)
    
    # Check available imputation methods
    available_methods = [d for d in os.listdir(imputed_dir) 
                        if os.path.isdir(os.path.join(imputed_dir, d)) 
                        and d not in ['original', 'mask']]
    print(f"\nAvailable imputation methods ({len(available_methods)}):")
    for method in sorted(available_methods):
        print(f"  - {method}")
    print()
    
    if scaling:
        print("Note: Inverse scaling is skipped in detailed plots because the scaler was trained")
        print("on all features together. Use plot_imputation_comparison.py for scaled visualizations.\n")
    
    created_plots = []
    
    # Process each feature
    for feature, config in FEATURE_CONFIG.items():
        start_date, end_date = config['time_period']
        print(f"\n{'='*80}")
        print(f"Processing: {feature}")
        print(f"  Station: {config['station']}")
        print(f"  Missing Rate: {config['missing_rate']:.1%}")
        print(f"  Time Period: {start_date} to {end_date}")
        print(f"{'='*80}")
        
        # Load data
        data_dict = load_feature_data(feature, working_dir, imputed_dir, params)
        
        if data_dict is None:
            print(f"❌ Skipping {feature} - data not found")
            continue
        
        print(f"  Loaded ground truth: {len(data_dict['ground_truth'])} rows")
        print(f"  Loaded mask: {len(data_dict['mask'])} rows")
        print(f"  Loaded {len(data_dict['imputed'])} imputation methods")
        
        # Create plot
        try:
            output_file = plot_feature_comparison(feature, data_dict, config, output_dir)
            if output_file:
                created_plots.append(output_file)
        except Exception as e:
            print(f"❌ Error creating plot for {feature}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"✅ COMPLETED: Created {len(created_plots)} plots")
    print(f"📁 Plots saved to: {output_dir}")
    print("\nCreated plots:")
    for plot in created_plots:
        print(f"  - {os.path.basename(plot)}")
    print("="*80)

if __name__ == "__main__":
    main()
