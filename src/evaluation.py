from typing import Optional

import os
import sys

# Suppress TensorFlow warnings before any imports that might load TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import dvclive
import pandas as pd
import numpy as np
import yaml
import joblib
from skimage.metrics import structural_similarity as ssim
from fastdtw import fastdtw # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import polars as pl
warnings.filterwarnings('ignore')

def load_working_files(data_files: list[str], missing_data_files: list[str]) -> tuple[dict, dict, str]:
    """Load working files and files with missing values.
    
    This is a copy of the function from imputation.tools.utilities to avoid
    loading PyPOTS during evaluation (which causes TensorFlow warnings).

    Args:
        data_files: List of paths to working files
        missing_data_files: List of paths to files with missing values
    Returns:
        tuple: (working_data dict, missing_data dict, partition_method string)
    """
    working_data = {}
    missing_data = {}

    for file in data_files:
        if file.endswith(".csv"):
            df = pd.read_csv(file, sep=',', index_col=0)
            set_name = file.strip(".csv").split("/")[-1].split("_")[2:]
            set_name = ("_").join(set_name)
            working_data[set_name] = df

    for file in missing_data_files:
        if file.endswith(".csv"):
            df = pd.read_csv(file, sep=',', index_col=0)
            set_name = file.strip(".csv").split("/")[-1].split("_")[2:]
            if "missing" in set_name:
                set_name.remove("missing")
            if "mask" in set_name:
                set_name.remove("mask")
            set_name = ("_").join(set_name)
            missing_data[set_name] = df

    # Determine partition method from filename
    prep_method = data_files[0].split("/")[-1].split("_")[1] if data_files else "Unknown"

    return working_data, missing_data, prep_method

def load_evaluation_files(data_files: list[str], sep: str) -> tuple[dict, str]:
    """Load working files and files with missing values.

    Args:
        data_files (str): Path to the working file.
        missing_data_files (str): Path to the file with missing values.
    Returns:
        tuple: A tuple containing two dictionaries:
            - Dictionary of working data dataframes.
            - Dictionary of missing data dataframes.
    """
    data = {}

    for file in data_files:
        if file.endswith(".csv"):
            df = pd.read_csv(file, sep=sep, index_col=0)
            set_name = file.strip(".csv").split("/")[-1].split("_")[2:]
            if "missing" in set_name in set_name:
                set_name.remove("missing")
            if "mask" in set_name:
                set_name.remove("mask")
            if "imputed" in set_name:
                set_name.remove("imputed")
            set_name = ("_").join(set_name)
            data[set_name] = df
    prep_method = data_files[0].split("/")[-1].split("_")[1]

    return data, prep_method

def apply_inverse_scaling(data: pd.DataFrame, 
                         scaler_dir: str) -> pd.DataFrame:
    """Apply inverse scaling transformation to data using saved scalers.
    
    Args:
        data: Dictionary of dataframes to inverse transform
        scaler_dir: Directory containing saved scaler files
        partition_by: Partition method ('Wide', 'Station', or 'Feature')
    
    Returns:
        Dictionary of inverse-transformed dataframes
    """
    if not os.path.exists(scaler_dir):
        print(f"Warning: Scaler directory {scaler_dir} not found. Skipping inverse scaling.")
        return data
    
    scaler_file = os.path.join(scaler_dir, "scaler_global.pkl")
    
    if not os.path.exists(scaler_file):
        print(f"Warning: Global scaler file {scaler_file} not found. Skipping inverse scaling.")
        return data
    
    try:
        scaler = joblib.load(scaler_file)
        
        # Apply inverse transform
        # Note: We need to handle the dataframe structure carefully
        # The scaler was fitted on columns excluding 'Date', so we need to match that
        df_copy = data.copy()
        data_columns = [col for col in data.columns if col not in ['Date', 'Year', 'Month', 'Day', 'Hour']]
        
        if len(data_columns) > 0:
            # Inverse transform
            inverse_transformed = scaler.inverse_transform(data[data_columns].values)
            df_copy[data_columns] = inverse_transformed
            print(f"Applied inverse scaling to data using {scaler_file}")
            inverse_data = df_copy
    except Exception as e:
        print(f"Error applying inverse scaling to data: {e}")
        inverse_data = data.copy()
    
    return inverse_data

def run_evaluation(run: int, 
                   original_data: dict[str, pd.DataFrame], 
                   imputed_data: dict[str, pd.DataFrame], 
                   mask: dict[str, pd.DataFrame], 
                   imputation_method: str, 
                   partition_method: str,
                   scaling: bool = False,
                   resample: bool = False,
                   resample_freq: Optional[str] = None) -> pd.DataFrame:
    """
    Enhanced evaluation function with improved error handling and additional metrics
    
    Args:
        run: Run number
        original_data: Ground truth data (in original domain after inverse scaling)
        imputed_data: Imputed data (in original domain after inverse scaling)
        mask: Missing data mask
        imputation_method: Name of imputation method
        partition_method: Partition strategy used
        scaling: Whether scaling was applied
        resample: Whether resampling was applied
        resample_freq: Resampling frequency if applied
    """
    results = pd.DataFrame(columns=[
        "run", "imputation_method", "partition_method", "station", "feature", 
        "RMSE", "MAE", "MAPE", "R2", "SSIM", "dtw_distance", "cosine_similarity",
        "missing_rate", "n_missing_values", "n_total_values",
        "scaling", "resample", "resample_freq"
    ])
    
    for partition in original_data.keys():
        if partition not in imputed_data:
            print(f"Warning: Partition {partition} not found in imputed data for method {imputation_method}")
            continue
            
        if partition not in mask:
            print(f"Warning: Mask for partition {partition} not found")
            continue
        
        # Calculate SSIM only for larger matrices
        if original_data[partition].shape[0] > 7 and original_data[partition].shape[1] > 7:
            try:
                # Align indices before calculating SSIM
                imputed_aligned = imputed_data[partition].reindex(original_data[partition].index)
                
                # Ensure data is numeric and handle NaN values
                orig_vals = original_data[partition].astype(float).fillna(0).values
                imp_vals = imputed_aligned.astype(float).fillna(0).values
                
                data_range = max(
                    np.nanmax(orig_vals), np.nanmax(imp_vals)
                ) - min(
                    np.nanmin(orig_vals), np.nanmin(imp_vals)
                )
                
                if data_range > 0:
                    ssim_score = ssim(orig_vals, imp_vals, data_range=data_range)
                else:
                    ssim_score = np.nan
            except Exception as e:
                print(f"Warning: SSIM calculation failed for {partition}: {e}")
                ssim_score = np.nan
        else:
            ssim_score = np.nan
        
        # Evaluate each feature/station (exclude temporal feature columns)
        temporal_cols = ['Date', 'Year', 'Month', 'Day', 'Hour']
        
        for element in original_data[partition].columns:
            # Skip temporal feature columns - they're not measurement data
            if element in temporal_cols:
                continue
                
            if element not in imputed_data[partition].columns:
                print(f"Warning: Element {element} not found in imputed data for {partition}")
                continue
                
            if element not in mask[partition].columns:
                print(f"Warning: Element {element} not found in mask for {partition}")
                continue
            
            mask_series = mask[partition][element]
            n_missing = mask_series.sum()
            n_total = len(mask_series)
            missing_rate = n_missing / n_total if n_total > 0 else 0
            
            # Skip if no missing values to evaluate
            if n_missing == 0:
                continue
                
            try:
                original_series = original_data[partition][element].astype(float)
                imputed_series = imputed_data[partition][element].astype(float)
                
                # Align indices (some methods add padding rows)
                # Use mask_series index as the reference since it matches original data
                imputed_series = imputed_series.reindex(mask_series.index)
                
                # Get missing positions
                missing_positions = mask_series == 1
                orig_missing = original_series[missing_positions]
                imp_missing = imputed_series[missing_positions]
                
                # Remove any remaining NaN values
                valid_indices = ~(np.isnan(orig_missing) | np.isnan(imp_missing))
                if valid_indices.sum() == 0:
                    print(f"Warning: No valid values for evaluation in {partition}/{element}")
                    continue
                    
                orig_valid = orig_missing[valid_indices]
                imp_valid = imp_missing[valid_indices]
                
                # Convert to numpy arrays for calculations
                orig_vals = np.array(orig_valid)
                imp_vals = np.array(imp_valid)
                
                # Calculate metrics
                rmse = np.sqrt(np.mean((orig_vals - imp_vals) ** 2))
                mae = np.mean(np.abs(orig_vals - imp_vals))
                
                # MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((orig_vals - imp_vals) / np.maximum(np.abs(orig_vals), 1e-8))) * 100
                
                # R-squared
                ss_res = np.sum((orig_vals - imp_vals) ** 2)
                ss_tot = np.sum((orig_vals - np.mean(orig_vals)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
                
                # DTW distance
                try:
                    dtw_distance, _ = fastdtw(orig_vals, imp_vals)
                except Exception as e:
                    print(f"Warning: DTW calculation failed for {partition}/{element}: {e}")
                    dtw_distance = np.nan
                
                # Cosine similarity
                try:
                    cos_sim = cosine_similarity(
                        orig_vals.reshape(1, -1), 
                        imp_vals.reshape(1, -1)
                    )[0][0]
                except Exception as e:
                    print(f"Warning: Cosine similarity calculation failed for {partition}/{element}: {e}")
                    cos_sim = np.nan
                
                # Add results
                # Handle station/feature assignment based on partition method
                
                station_name = element
                feature_name = partition
                
                result_row = {
                    "run": run,
                    "imputation_method": imputation_method,
                    "partition_method": partition_method,
                    "station": station_name,
                    "feature": feature_name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "MAPE": mape,
                    "R2": r2,
                    "SSIM": ssim_score,
                    "dtw_distance": dtw_distance,
                    "cosine_similarity": cos_sim,
                    "missing_rate": missing_rate,
                    "n_missing_values": n_missing,
                    "n_total_values": n_total,
                    "scaling": scaling,
                    "resample": resample,
                    "resample_freq": resample_freq if resample else None
                }
                
                results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)
                
            except Exception as e:
                print(f"Error evaluating {partition}/{element} for {imputation_method}: {e}")
                continue
    
    return results

def create_comprehensive_visualizations(results: pd.DataFrame, output_dir: str, run: int):
    """Create comprehensive visualizations for evaluation results - PER FEATURE"""
    
    # Filter for current run
    current_results = results[results['run'] == run]
    if current_results.empty:
        print(f"No results found for run {run}")
        return
    
    #partition_method = current_results["partition_method"].iloc[0]
    
    # Get all metrics for visualization
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2', 'SSIM', 'dtw_distance', 'cosine_similarity']
    
    # Create visualizations PER FEATURE (not mixed across features!)
    features = current_results['feature'].unique()
    
    print(f"\nCreating visualizations for {len(features)} features separately...")
    
    for feature in features:
        feature_results = current_results[current_results['feature'] == feature]
        
        print(f"  Creating plots for {feature}...")
        
        # Create subplots for all metrics for this feature
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        fig.suptitle(f'Imputation Method Comparison - {feature}', fontsize=16, fontweight='bold')
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                
                # Remove any infinite or very large values for better visualization
                plot_data = feature_results.copy()
                plot_data[metric] = plot_data[metric].replace([np.inf, -np.inf], np.nan)
                
                # Create bar plot grouped by method
                try:
                    method_agg = plot_data.groupby('imputation_method')[metric].agg(['mean', 'std']).reset_index()
                    
                    ax.bar(range(len(method_agg)), method_agg['mean'], 
                          yerr=method_agg['std'], capsize=5, alpha=0.7)
                    ax.set_xticks(range(len(method_agg)))
                    ax.set_xticklabels(method_agg['imputation_method'], rotation=45, ha='right')
                    ax.set_title(f"{metric} - {feature}")
                    ax.set_ylabel(metric)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                except Exception as e:
                    print(f"    Error plotting {metric} for {feature}: {e}")
                    ax.text(0.5, 0.5, f"Error plotting {metric}", ha='center', va='center', transform=ax.transAxes)
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        safe_feature_name = feature.replace('/', '_').replace(':', '_')
        plt.savefig(os.path.join(output_dir, f"metrics_comparison_{safe_feature_name}_run_{run}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also create a comparison heatmap PER FEATURE
    print("\n  Creating heatmaps per feature...")
    for feature in features:
        feature_results = current_results[current_results['feature'] == feature]
        
        # Create heatmaps for key metrics
        metrics_for_heatmap = ['RMSE', 'MAE', 'R2']
        fig, axes = plt.subplots(1, len(metrics_for_heatmap), figsize=(18, 6))
        fig.suptitle(f'Method Comparison Heatmap - {feature}', fontsize=14, fontweight='bold')
        
        for i, metric in enumerate(metrics_for_heatmap):
            try:
                # Aggregate by method and station (if multiple stations per feature)
                if 'station' in feature_results.columns:
                    pivot_data = feature_results.pivot_table(
                        index="imputation_method", 
                        columns="station", 
                        values=metric,
                        aggfunc='mean'
                    )
                else:
                    # Just show methods if no station dimension
                    pivot_data = feature_results.groupby('imputation_method')[metric].mean().to_frame()
                
                # Handle infinite values
                pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
                
                sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis', ax=axes[i])
                axes[i].set_title(f'{metric}')
                axes[i].set_xlabel('Station' if pivot_data.shape[1] > 1 else '')
                axes[i].set_ylabel('Method')
                
            except Exception as e:
                print(f"    Error creating heatmap for {metric} in {feature}: {e}")
                axes[i].text(0.5, 0.5, "Error", ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        safe_feature_name = feature.replace('/', '_').replace(':', '_')
        plt.savefig(os.path.join(output_dir, f"heatmap_{safe_feature_name}_run_{run}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  Visualizations created successfully!")
    
    # Create overall ranking table (with caveat that averaging across features can be misleading)
    print("\n  Creating overall ranking table (cross-feature average with caution)...")
    ranking_data = current_results.groupby("imputation_method").agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'MAPE': 'mean', 
        'R2': 'mean',
        'SSIM': 'mean',
        'dtw_distance': 'mean',
        'cosine_similarity': 'mean'
    }).round(4)
    
    # Rank methods (lower is better for RMSE, MAE, MAPE, dtw_distance; higher is better for R2, SSIM, cosine_similarity)
    ranking_data['RMSE_rank'] = ranking_data['RMSE'].rank()
    ranking_data['MAE_rank'] = ranking_data['MAE'].rank()
    ranking_data['MAPE_rank'] = ranking_data['MAPE'].rank()
    ranking_data['dtw_rank'] = ranking_data['dtw_distance'].rank()
    ranking_data['R2_rank'] = ranking_data['R2'].rank(ascending=False)
    ranking_data['SSIM_rank'] = ranking_data['SSIM'].rank(ascending=False)
    ranking_data['cosine_sim_rank'] = ranking_data['cosine_similarity'].rank(ascending=False)
    
    # Calculate average rank
    rank_cols = ['RMSE_rank', 'MAE_rank', 'MAPE_rank', 'dtw_rank', 'R2_rank', 'SSIM_rank', 'cosine_sim_rank']
    ranking_data['average_rank'] = ranking_data[rank_cols].mean(axis=1)
    ranking_data = ranking_data.sort_values('average_rank')
    
    # Save ranking to CSV
    ranking_data.to_csv(os.path.join(output_dir, f"method_ranking_run_{run}.csv"))

    live = dvclive.Live("dvclive", resume=False)

    # Log metrics per feature (NOT averaged across features - that would be misleading!)
    # Features have different value ranges, so we report them separately
    features = current_results['feature'].unique()
    
    for feature in features:
        feature_data = current_results[current_results['feature'] == feature]
        feature_agg = feature_data.groupby('imputation_method').agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'MAPE': 'mean',
            'R2': 'mean',
            'SSIM': 'mean',
            'dtw_distance': 'mean',
            'cosine_similarity': 'mean'
        }).round(4)
        
        # Log each method's metrics for this feature
        for method in feature_agg.index:
            for metric in ['RMSE', 'MAE', 'MAPE', 'R2', 'SSIM', 'dtw_distance', 'cosine_similarity']:
                metric_name = f"{feature}/{method}/{metric}"
                live.log_metric(metric_name, feature_agg.loc[method, metric])
        
        # Log the best method for this feature
        best_rmse_method = feature_agg['RMSE'].idxmin()
        best_mae_method = feature_agg['MAE'].idxmin()
        best_r2_method = feature_agg['R2'].idxmax()
        
        live.log_param(f"best_method_{feature}_RMSE", best_rmse_method)
        live.log_param(f"best_method_{feature}_MAE", best_mae_method)
        live.log_param(f"best_method_{feature}_R2", best_r2_method)

    # Also log overall ranking (for reference, but with warning)
    for _, row in ranking_data.iterrows():
        method = row.name
        live.log_metric(f"overall_avg/{method}/average_rank", row['average_rank'])

    # Finalize logging
    live.end()
    
    print(f"Visualizations saved to {output_dir}")
    print("\nOverall method ranking (averaged across features - use with caution!):")
    for i, (method, row) in enumerate(ranking_data.iterrows(), 1):
        print(f"{i}. {method} (avg rank: {row['average_rank']:.2f})")

def to_partitions(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert wide DataFrame to partitioned dictionary by Feature."""
    temp_data_long_polors = pl.from_pandas(data.melt(ignore_index=False),include_index=True)
    temp_data_patitioned = temp_data_long_polors.partition_by("Feature", as_dict=True)
    data_pandas_dict = {key[0]: temp_data_patitioned[key].pivot(index="Date", on="Station", values="value").to_pandas().set_index("Date") for key in temp_data_patitioned.keys()}
    return data_pandas_dict
    


def main():
    """Enhanced main function with better error handling and reporting"""
    
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate_imputations.py <input_dir> <output_dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load preprocessing configuration from params.yaml
        print("Loading preprocessing configuration...")
        with open("params.yaml", 'r') as file:
            params = yaml.safe_load(file)
        
        # Extract metadata
        partition_by = params.get("prepare_working_files", {}).get("partition_by", "Unknown")
        scaling = params.get("prepare_working_files", {}).get("scaling", False)
        resample = params.get("preprocessing", {}).get("resample", False)
        resample_freq = params.get("preprocessing", {}).get("resample_freq", None) if resample else None
        
        print(f"Configuration: partition_by={partition_by}, scaling={scaling}, resample={resample}, resample_freq={resample_freq}")
        
        # Determine scaler directory
        scaler_dir = "data/00_tools/scalers"

        # Get imputation methods
        files = os.listdir(input_dir)
        methods = [f for f in files if (not f.startswith(".") and not f.endswith(".csv") and not f.endswith(".yaml"))]
        print(f"Found imputation methods: {methods}")

        # Load imputed data for each method
        data_files_imputed = {}
        data_files = []
        mask_files = []
        for method in methods:
            method_dir = os.path.join(input_dir, method)
            if os.path.isdir(method_dir):
                method_files = [os.path.join(method_dir, f) for f in os.listdir(method_dir) 
                              if (not f.startswith(".") and f.endswith(".csv"))]
                if method == "original":
                    data_files_imputed[method] = method_files
                    data_files = method_files
                elif method == "mask":
                    mask_files = method_files
                else:
                    data_files_imputed[method] = method_files
                print(f"  {method}: {len(method_files)} files")

        # Load original data and masks
        print("Loading original data and masks...")
        data = pd.read_csv(data_files[0], sep=',')
        mask = pd.read_csv(mask_files[0], sep=',')
        print("...Successfully loaded original data and mask")
        print(f"Original data shape: {data.shape}, Mask shape: {mask.shape}")
        data_wide = data.pivot(index="Date", columns=["Station", "Feature"], values="Value")
        mask_wide = mask.pivot(index="Date", columns=["Station", "Feature"], values="Value")
        data = to_partitions(data_wide)
        mask = to_partitions(mask_wide)
        partition_method = data_files[0].split("/")[-1].split("_")[1]
        # Load imputed data
        print("Loading imputed data...")
        imputed_data = {}
        for method in data_files_imputed.keys():
            try:
                temp_data = pd.read_csv(data_files_imputed[method][0], sep=',')
                temp_data_wide = temp_data.pivot(index="Date", columns=["Station", "Feature"], values="Value")

                
                # Apply inverse scaling to imputed data if scaling was used
                if scaling:
                    temp_data_wide = apply_inverse_scaling(temp_data_wide, scaler_dir)
                data_pandas_dict = to_partitions(temp_data_wide)
                imputed_data[method] = data_pandas_dict
                print(f"  Successfully loaded {method}")
            except Exception as e:
                print(f"  Error loading {method}: {e}")
                continue

        # Load or create results DataFrame
        results_file = os.path.join(output_dir, "evaluation_results.csv")
        if os.path.exists(results_file):
            results = pd.read_csv(results_file)
        else:
            results = pd.DataFrame(columns=[
                "run", "imputation_method", "partition_method", "station", "feature", 
                "RMSE", "MAE", "MAPE", "R2", "SSIM", "dtw_distance", "cosine_similarity",
                "missing_rate", "n_missing_values", "n_total_values",
                "scaling", "resample", "resample_freq"
            ])

        # Determine run number
        run = 1 if results.empty or pd.isna(results['run'].max()) else int(results['run'].max()) + 1
        print(f"Starting evaluation run {run}")

        # Evaluate each method
        for imp_method in imputed_data.keys():
            print(f"Evaluating method: {imp_method}")
            try:
                temp_results = run_evaluation(
                    run=run, 
                    original_data=data, 
                    imputed_data=imputed_data[imp_method], 
                    mask=mask, 
                    imputation_method=imp_method, 
                    partition_method=partition_method,
                    scaling=scaling,
                    resample=resample,
                    resample_freq=resample_freq
                )
                results = pd.concat([results, temp_results], ignore_index=True)
                print(f"  Added {len(temp_results)} evaluation records")
            except Exception as e:
                print(f"  Error evaluating {imp_method}: {e}")
                continue

        # Save results
        results.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        # Create visualizations
        print("Creating visualizations...")
        try:
            create_comprehensive_visualizations(results, output_dir, run)
        except Exception as e:
            print(f"Error creating visualizations: {e}")

        # Print summary statistics
        current_results = results[results['run'] == run]
        if not current_results.empty:
            print(f"\n{'='*60}")
            print(f"EVALUATION SUMMARY - Run {run}")
            print(f"{'='*60}")
            print("Configuration:")
            print(f"  - Partition by: {partition_by}")
            print(f"  - Scaling: {scaling}")
            print(f"  - Resample: {resample}")
            if resample:
                print(f"  - Resample frequency: {resample_freq}")
            print(f"\nMethods evaluated: {current_results['imputation_method'].nunique()}")
            print(f"Total evaluations: {len(current_results)}")
            
            if scaling:
                print("\n⚠️  Note: All metrics calculated in ORIGINAL DOMAIN (inverse scaling applied)")
            
            # Print feature-specific summaries (don't mix different value ranges)
            print("\n" + "="*80)
            print("METHOD PERFORMANCE BY FEATURE (avoiding misleading cross-feature averages)")
            print("="*80)
            
            for feature in sorted(current_results['feature'].unique()):
                feature_results = current_results[current_results['feature'] == feature]
                
                print(f"\n{'─'*80}")
                print(f"Feature: {feature}")
                print(f"{'─'*80}")
                
                feature_stats = feature_results.groupby('imputation_method').agg({
                    'RMSE': ['mean', 'std', 'min', 'max'],
                    'MAE': ['mean', 'std', 'min', 'max'],
                    'MAPE': ['mean', 'std'],
                    'R2': ['mean', 'std'],
                    'cosine_similarity': ['mean', 'std']
                }).round(4)
                
                print(feature_stats)
                
                # Rank methods for this feature
                best_rmse = feature_results.loc[feature_results['RMSE'].idxmin(), 'imputation_method']
                best_mae = feature_results.loc[feature_results['MAE'].idxmin(), 'imputation_method']
                best_r2 = feature_results.loc[feature_results['R2'].idxmax(), 'imputation_method']
                
                print(f"\nBest methods for {feature}:")
                print(f"  • Lowest RMSE: {best_rmse}")
                print(f"  • Lowest MAE: {best_mae}")
                print(f"  • Highest R²: {best_r2}")
            
            print(f"\n{'='*80}")
            print("⚠️  WARNING: Cross-feature averages are NOT shown as they can be misleading")
            print("   Different features have different value ranges (e.g., Wind_speed vs Air_pressure)")
            print("   Always compare methods within the same feature!")
            print(f"{'='*80}\n")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()