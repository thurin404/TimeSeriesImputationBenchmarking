import os
import joblib
import pandas as pd


def get_working_files(dir: str = "../data/04_working/") -> tuple[list[str], list[str]]:
    """Get list of working files and files with missing values from YAML files.

    Args:
        dir (str): Directory where the YAML files are located. Defaults to "../data/04_working_files/".

    Returns:
        tuple: A tuple containing two lists:
            - List of working file paths.
            - List of file paths with missing values.
    """
    datafiles = []
    datafiles_missing = []
    files = os.listdir(dir)
    for file in files:
        if "gt" in file:
            datafiles.append(os.path.join(dir, file))
        if "missing" in file:
            datafiles_missing.append(os.path.join(dir, file))

    return datafiles, datafiles_missing

def load_working_files(data_files: list[str], missing_data_files: list[str]) -> tuple[dict, dict, str]:
    """Load working files and files with missing values.

    Args:
        data_files (str): Path to the working file.
        missing_data_files (str): Path to the file with missing values.
    Returns:
        tuple: A tuple containing two dictionaries:
            - Dictionary of working data dataframes.
            - Dictionary of missing data dataframes.
    """
    working_data = joblib.load(data_files[0])
    missing_data = joblib.load(missing_data_files[0])
    prep_method = data_files[0].split("/")[-1].split("_")[1]
    return working_data, missing_data, prep_method

def save_imputed_files(imputed_data: dict, output_dir: str = "../data/05_imputed_data/", 
                       imputation_method: str = "ffill", prep_method: str = "station"):
    """Save imputed dataframes to CSV files.

    Args:
        imputed_data (dict): Dictionary of imputed dataframes.
        output_dir (str): Directory to save the imputed files. Defaults to "../data/05_imputed_data/".
        imputation_method (str): Imputation method used. Defaults to "ffil".
        prep_method (str): Preparation method used. Defaults to "station".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, imputation_method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    temp_df = pd.DataFrame()
    time_features = ["Year", "Month", "Day", "Hour"]

    for key, df in imputed_data.items():
        if any(feature in df.columns for feature in time_features):
            df = df.drop(columns=[feature for feature in time_features if feature in df.columns])
        if prep_method == "wide":
            df = df.melt(ignore_index=False, var_name=["Station", "Feature"], value_name="Value")
        elif prep_method == "feature":
            df = df.melt(ignore_index=False, var_name="Station", value_name="Value")
            df["Feature"] = key
        elif prep_method == "station":
            df = df.melt(ignore_index=False, var_name="Feature", value_name="Value")
            df["Station"] = key
        else:
            raise ValueError(f"Unknown prep_method: {prep_method}")

        # Collect all processed DataFrames
        temp_df = pd.concat([temp_df, df], axis=0)

    # Save once, after all processing
    output_path = os.path.join(output_dir, f"ts_{prep_method}_imputed.csv")
    temp_df.to_csv(output_path, index=True, sep=',')
    print(f"✅ Imputed data saved to {output_path}")
