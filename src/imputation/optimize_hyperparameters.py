#!/usr/bin/env python3
"""
Standalone hyperparameter optimization script for PyPOTS imputation models.

This script can be used to optimize hyperparameters for any PyPOTS-based imputation
model independently of the main pipeline.

Usage:
    python optimize_hyperparameters.py <model_type> <input_dir> <model_dir> [--trials N] [--timeout T]

Example:
    python optimize_hyperparameters.py saits data/04_working_files models/saits --trials 100 --timeout 7200
"""

import argparse
import sys
import os
from pathlib import Path

# Add the imputation directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from saits import SAITSImputer
from brits import BRITSImputer
from tools import utilities


def get_imputer(model_type: str):
    """Get the appropriate imputer class based on model type"""
    imputers = {
        'saits': SAITSImputer,
        'brits': BRITSImputer,
    }
    
    if model_type.lower() not in imputers:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(imputers.keys())}")
    
    return imputers[model_type.lower()]()


def main():
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for PyPOTS imputation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "model_type", 
        help="Type of model to optimize (saits, brits)"
    )
    
    parser.add_argument(
        "input_dir", 
        help="Directory containing input data files"
    )
    
    parser.add_argument(
        "model_dir", 
        help="Directory to save optimized model and reports"
    )
    
    parser.add_argument(
        "--trials", 
        type=int, 
        default=50, 
        help="Number of optimization trials (default: 50)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=3600, 
        help="Optimization timeout in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--dataset-key",
        type=str,
        help="Specific dataset key to use for optimization (default: first available)"
    )
    
    args = parser.parse_args()
    
    try:
        # Get the imputer
        imputer = get_imputer(args.model_type)
        print(f"Optimizing hyperparameters for {args.model_type.upper()} model")
        
        # Load data
        data_files, missing_data_files = utilities.get_working_files(dir=args.input_dir)
        _, missing_data, _ = utilities.load_working_files(data_files, missing_data_files)
        
        if not missing_data:
            print("No data found in input directory!")
            return 1
        
        # Select dataset for optimization
        if args.dataset_key and args.dataset_key in missing_data:
            sample_key = args.dataset_key
        else:
            sample_key = next(iter(missing_data.keys()))
        
        sample_df = missing_data[sample_key]
        print(f"Using dataset: {sample_key}")
        print(f"Dataset shape: {sample_df.shape}")
        
        # Create model directory if it doesn't exist
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Run optimization
        best_params = imputer.optimize_hyperparameters(
            sample_df,
            args.model_dir,
            n_trials=args.trials,
            timeout=args.timeout
        )
        
        if best_params:
            print("\nOptimization completed successfully!")
            print(f"Best parameters: {best_params}")
            
            # Update model configuration if it exists
            try:
                from pathlib import Path
                config_path = Path(f"params/{args.model_type}.yaml")
                if config_path.exists():
                    imputer.update_params_with_optimization(str(config_path), best_params)
                    print(f"Updated configuration file: {config_path}")
                else:
                    print(f"Configuration file not found: {config_path}")
            except Exception as e:
                print(f"Failed to update configuration: {e}")
        else:
            print("Optimization failed!")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())