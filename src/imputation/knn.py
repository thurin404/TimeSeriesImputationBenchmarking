import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from tools import utilities

# KNN imputation with hyperparameter optimization and parallelization

def optimize_knn_params(df: pd.DataFrame, params_grid: dict, n_trials: int = 25) -> dict:
    """
    Optimize KNN parameters using Optuna on a sample of the data.
    """
    print("Starting KNN hyperparameter optimization...")
    
    # Sample data for faster optimization (use first 1000 rows if dataset is large)
    sample_df = df.head(min(1000, len(df))).copy()
    
    # Create a mask for validation - use actual missing values
    mask = sample_df.isna()
    
    # For optimization, we need some non-missing data to validate on
    # Artificially mask 10% of the non-missing data for validation
    available_data = sample_df[~mask]
    int(0.1 * available_data.count().sum())
    
    validation_mask = mask.copy()
    for col in sample_df.columns:
        non_missing_idx = sample_df[col].dropna().index
        if len(non_missing_idx) > 0:
            n_to_mask = max(1, int(0.1 * len(non_missing_idx)))
            mask_idx = np.random.choice(non_missing_idx, size=min(n_to_mask, len(non_missing_idx)), replace=False)
            validation_mask.loc[mask_idx, col] = True
    
    # Store ground truth
    ground_truth = sample_df.copy()
    
    # Create training data with validation values masked
    train_df = sample_df.copy()
    train_df[validation_mask & ~mask] = np.nan
    
    def objective(trial):
        # Suggest parameters from grid - only those supported by KNNImputer
        n_neighbors = trial.suggest_categorical("n_neighbors", params_grid["n_neighbors"])
        weights = trial.suggest_categorical("weights", params_grid["weights"])
        # KNNImputer only supports 'nan_euclidean' metric
        
        # Create imputer with suggested parameters
        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric='nan_euclidean',  # Only supported metric
            keep_empty_features=True
        )
        
        try:
            # Impute
            imputed_values = imputer.fit_transform(train_df.to_numpy())
            imputed_df = pd.DataFrame(imputed_values, index=train_df.index, columns=train_df.columns)
            
            # Calculate MSE on validation set (artificially masked non-missing data)
            val_mask = validation_mask & ~mask
            mse = mean_squared_error(
                ground_truth[val_mask].values.flatten(),
                imputed_df[val_mask].values.flatten()
            )
            
            return mse
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')
    
    # Create study with parallelization support
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # n_jobs=-1 uses all CPU cores
    
    best_params = study.best_params
    best_mse = study.best_value
    
    print(f"Best parameters found: {best_params}")
    print(f"Best validation MSE: {best_mse:.4f}")
    
    return best_params


def imputation(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Perform KNN imputation with given parameters"""
    imputer = KNNImputer(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        metric='nan_euclidean',  # Only supported metric for KNNImputer
        keep_empty_features=True
    )
    
    # Identify temporal columns to exclude from imputation
    temporal_cols = ['Date', 'Year', 'Month', 'Day', 'Hour']
    cols_to_impute = [col for col in df.columns if col not in temporal_cols]
    
    # Create output dataframe with original structure
    df_imputed = df.copy()
    
    # Only impute non-temporal columns
    if len(cols_to_impute) > 0:
        imputed_values = imputer.fit_transform(df[cols_to_impute].to_numpy())
        df_imputed[cols_to_impute] = imputed_values
    
    return df_imputed


def main():
    imputation_method = "knn"
    if len(sys.argv) != 3:
        print("Usage: python src/imputation/knn.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Load YAML configurations
    params_all = yaml.safe_load(open("params.yaml"))
    knn_config_path = params_all["imputation"]["knn"]["params_file"]

    with open(knn_config_path, "r") as f:
        knn_yaml = yaml.safe_load(f)

    knn_params = knn_yaml["params"]
    params_grid = knn_yaml["params_grid"]
    optimized = knn_yaml.get("optimized", False)

    data_files, missing_data_files = utilities.get_working_files(dir=input_dir)
    _, missing_data, prep_method = utilities.load_working_files(data_files, missing_data_files)
    
    imputed_data = {}

    # If parameters are not optimized, perform optimization first
    if not optimized:
        print("Running hyperparameter optimization for KNN...")
        
        # Run optimization on the first dataset (representative)
        sample_key = next(iter(missing_data.keys()))
        sample_df = missing_data[sample_key]
        
        best_params = optimize_knn_params(sample_df, params_grid, n_trials=25)
        
        # Update YAML
        knn_yaml["params"].update(best_params)
        knn_yaml["optimized"] = True
        
        with open(knn_config_path, "w") as f:
            yaml.dump(knn_yaml, f, sort_keys=False)
        
        knn_params.update(best_params)
        print("Optimization complete. Updated YAML file saved.")
    else:
        print("Using optimized parameters for KNN imputation.")

    for key, df in missing_data.items():
        print(f"Imputing data for: {key} using {imputation_method} method")
        imputed_data[key] = imputation(df, knn_params)
        print(f"Imputation completed for: {key}")

    utilities.save_imputed_files(imputed_data, output_dir=output_dir, imputation_method=imputation_method, prep_method=prep_method)




if __name__ == "__main__":
    main()