import os
import shutil
import sys
from tools import utilities
import pandas as pd

# mean imputation

def imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(df.mean(), axis=0)
    return df

def main():
    imputation_method = "mean"
    if len(sys.argv) != 3:
        print("Usage: python src/prepare.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    data_files, missing_data_files = utilities.get_working_files(dir=input_dir)
    _, missing_data, prep_method = utilities.load_working_files(data_files, missing_data_files)
    
    imputed_data = {}

    for key, df in missing_data.items():
        imputed_data[key] = imputation(df)

    utilities.save_imputed_files(imputed_data, output_dir=output_dir, imputation_method=imputation_method, prep_method=prep_method)

if __name__ == "__main__":
    main()