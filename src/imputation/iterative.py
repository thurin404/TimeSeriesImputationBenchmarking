import sys
import yaml
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from tools import utilities

# IterativeImputer with default BayesianRidge estimator (simplified)


def imputation(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Perform iterative imputation with BayesianRidge estimator"""
    # Use default BayesianRidge estimator with simple parameters
    imputer_params = {
        "random_state": int(params.get("random_state", 42)),
        "max_iter": int(params.get("max_iter", 10)),
        "tol": float(params.get("tol", 0.001)),
        "verbose": int(params.get("verbose", 0)),
        "initial_strategy": 'mean'
    }
    
    # Identify temporal columns to exclude from imputation
    temporal_cols = ['Date', 'Year', 'Month', 'Day', 'Hour']
    cols_to_impute = [col for col in df.columns if col not in temporal_cols]
    
    # Create imputer with default BayesianRidge
    imputer = IterativeImputer(**imputer_params)
    
    # Create output dataframe with original structure
    df_imputed = df.copy()
    
    # Only impute non-temporal columns
    if len(cols_to_impute) > 0:
        imputed_values = imputer.fit_transform(df[cols_to_impute].to_numpy())
        df_imputed[cols_to_impute] = imputed_values
    
    return df_imputed


def main():
    imputation_method = "iterative"

    if len(sys.argv) != 3:
        print("Usage: python src/imputation/iterative.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Load YAML configurations
    params_all = yaml.safe_load(open("params.yaml"))
    iterative_config_path = params_all["imputation"]["iterative"]["params_file"]

    with open(iterative_config_path, "r") as f:
        iterative_yaml = yaml.safe_load(f)

    iterative_params = iterative_yaml["params"]

    data_files, missing_data_files = utilities.get_working_files(dir=input_dir)
    _, missing_data, prep_method = utilities.load_working_files(data_files, missing_data_files)

    imputed_data = {}

    print("Using default BayesianRidge estimator for Iterative imputation.")

    # Perform imputation
    for key, df in missing_data.items():
        print(f"Imputing data for: {key} using {imputation_method} method")
        imputed_data[key] = imputation(df, iterative_params)
        print(f"Imputation completed for: {key}")

    utilities.save_imputed_files(imputed_data, output_dir=output_dir,
                                 imputation_method=imputation_method, prep_method=prep_method)



if __name__ == "__main__":
    main()