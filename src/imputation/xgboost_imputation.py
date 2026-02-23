import sys
import yaml
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from tools import utilities


def optimize_xgboost_params(df: pd.DataFrame, params_grid: dict, base_params: dict, n_trials: int = 25, n_samples: int = 5000):
    """
    Optimize XGBoost parameters using Optuna on non-missing values from one representative column.
    """
    print("Starting XGBoost hyperparameter optimization...")

    # Identify temporal columns to exclude
    temporal_cols = ['Date', 'Year', 'Month', 'Day', 'Hour']
    cols_to_use = [col for col in df.columns if col not in temporal_cols]
    df_subset = df[cols_to_use]

    # Pick the first column with missing values as a proxy
    target_column = next((col for col in df_subset.columns if df_subset[col].isna().sum() > 0), df_subset.columns[0])
    non_missing = df_subset.loc[df_subset[target_column].notna()]
    X = non_missing.drop(columns=[target_column])
    y = non_missing[target_column]

    # Sample if dataset is large
    if len(X) > n_samples:
        sample_idx = np.random.choice(len(X), n_samples, replace=False)
        X = X.iloc[sample_idx]
        y = y.iloc[sample_idx]

    # Split data once for all trials
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        # Suggest parameters
        learning_rate = trial.suggest_float("learning_rate", params_grid["learning_rate"][0], 
                                           params_grid["learning_rate"][1], log=True)
        max_depth = trial.suggest_int("max_depth", params_grid["max_depth"][0], 
                                     params_grid["max_depth"][1])
        subsample = trial.suggest_float("subsample", params_grid["subsample"][0], 
                                       params_grid["subsample"][1])
        colsample_bytree = trial.suggest_float("colsample_bytree", params_grid["colsample_bytree"][0], 
                                              params_grid["colsample_bytree"][1])
        reg_lambda = trial.suggest_float("reg_lambda", params_grid["reg_lambda"][0], 
                                        params_grid["reg_lambda"][1], log=True)
        n_estimators = trial.suggest_int("n_estimators", params_grid["n_estimators"][0], 
                                        params_grid["n_estimators"][1])

        try:
            model = xgb.XGBRegressor(
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                n_estimators=n_estimators,
                random_state=base_params.get("random_state", 42),
                device=base_params.get("device", "cpu"),
                early_stopping_rounds=10,
                verbosity=0
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            
            return rmse
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')

    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"Best parameters found: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    return best_params

def imputation(df: pd.DataFrame, params) -> pd.DataFrame:
    df_imputed = df.copy()
    
    # Identify temporal columns to exclude from imputation
    temporal_cols = ['Date', 'Year', 'Month', 'Day', 'Hour']
    cols_to_impute = [col for col in df_imputed.columns if col not in temporal_cols]
    
    for column in cols_to_impute:
        model = xgb.XGBRegressor(**params)

        non_missing = df_imputed.loc[df[column].notna()]
        missing = df_imputed.loc[df[column].isna()]

        # Use only non-temporal columns as features (excluding current target column)
        feature_cols = [col for col in cols_to_impute if col != column]
        
        X_train = non_missing[feature_cols]
        y_train = non_missing[column]
        X_missing = missing[feature_cols]

        model.fit(X_train, y_train)
        predictions = model.predict(X_missing)
        df_imputed.loc[df_imputed[column].isna(), column] = predictions

    return df_imputed


def main():
    imputation_method = "xgboost"

    if len(sys.argv) != 3:
        print("Usage: python src/prepare.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Load YAML configurations
    params_all = yaml.safe_load(open("params.yaml"))
    xgb_config_path = params_all["imputation"]["xgboost"]["params_file"]

    with open(xgb_config_path, "r") as f:
        xgb_yaml = yaml.safe_load(f)

    xgboost_params = xgb_yaml["params"]
    params_grid = xgb_yaml["params_grid"]
    optimized = xgb_yaml.get("optimized", False)

    data_files, missing_data_files = utilities.get_working_files(dir=input_dir)
    _, missing_data, prep_method = utilities.load_working_files(data_files, missing_data_files)

    imputed_data = {}

    # If parameters are not optimized, perform optimization first
    if not optimized:
        print("Using default parameters for XGBoost imputation and running optimization...")

        # Run optimization on the first dataset (representative)
        sample_key = next(iter(missing_data.keys()))
        sample_df = missing_data[sample_key]

        best_params = optimize_xgboost_params(sample_df, params_grid, xgboost_params)

        # Update YAMLs
        xgb_yaml["params"].update(best_params)
        xgb_yaml["optimized"] = True

        with open(xgb_config_path, "w") as f:
            yaml.dump(xgb_yaml, f, sort_keys=False)

        xgboost_params.update(best_params)
        print("Optimization complete. Updated YAML files saved.")

    else:
        print("Using optimized parameters for XGBoost imputation.")

    # Perform imputation
    for key, df in missing_data.items():
        print(f"Imputing data for: {key} using {imputation_method} method")
        imputed_data[key] = imputation(df, xgboost_params)
        print(f"Imputation completed for: {key}")

    utilities.save_imputed_files(imputed_data, output_dir=output_dir,
                                 imputation_method=imputation_method, prep_method=prep_method)

if __name__ == "__main__":
    main()
