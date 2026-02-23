# Preprocessing module for time series data
# loads data from ../data/raw/ and saves processed data to ../data/processed/
import os
import pandas as pd
import sys

import yaml

def load_data(data_dir: str, resample: bool = False, resample_freq: str = '1d') -> pd.DataFrame:
    ''' Load data from CSV files in the given directory.
    Each CSV file corresponds to a different feature and contains time series data for multiple stations.
    The function reads each CSV file, resamples the data to hourly frequency, and reshapes it into a long format.
    Finally, it concatenates all the data into a single DataFrame with columns: Date, Feature, Station, Value.
    '''
    df = pd.DataFrame()
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            print(f"Loading file: {file}")

            feature = file.replace(".csv", "")
            temp_df = pd.read_csv(os.path.join(data_dir, file), parse_dates=True, sep=",", index_col=0)
            temp_df.index = pd.to_datetime(temp_df.index, errors='coerce')

            if temp_df.index.tz is not None:
                temp_df.index = temp_df.index.tz_convert(None)  # Remove timezone if present

            print(f"The length of {feature}-DataFrame is {len(temp_df)}, before resampling")
            temp_df.columns = [str(col).split(" ")[0] for col in temp_df.columns]
            temp_df = temp_df.asfreq('h')
            temp_df = temp_df.reindex()
            if resample:
                temp_df = temp_df.resample(resample_freq).mean()

            temp_df["Feature"] = feature
            temp_df["Date"] = temp_df.index
            print(f"The length of {feature}-DataFrame is {len(temp_df)}, after resampling")
            temp_df = temp_df.melt(id_vars=["Date", "Feature"], var_name='Station', value_name='Value')
            df = pd.concat([df, temp_df], axis=0)
            df = df.reset_index(drop=True)
    df = transform_data(df)
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    ''' Transform the loaded data by handling missing values and converting temperature to Kelvin.
    '''
    # Handle temperature conversion to Kelvin
    temp_mask = df['Feature'].str.contains('temperature', case=False, na=False)
    df.loc[temp_mask, 'Value'] = df.loc[temp_mask, 'Value'] + 273.15
    #df.loc[temp_mask, 'Value'] = df.loc[temp_mask, 'Value'].clip(lower=0, upper=360)

    # Handle direction values to be within 0-360
    dir_mask = df['Feature'].str.contains('direction', case=False, na=False)
    df.loc[dir_mask, 'Value'] = df.loc[dir_mask, 'Value'].clip(lower=0, upper=360)

    hum_mask = df['Feature'].str.contains('humidity', case=False, na=False)
    df.loc[hum_mask, 'Value'] = df.loc[hum_mask, 'Value'].clip(lower=0, upper=100)

    """ # filter data according to start and end date
    start_date = "2008-08-05T00:00"
    end_date = "2018-08-04T23:00:00"
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df.loc[mask] """

    return df

def save_data(df: pd.DataFrame, output_path: str):
    ''' Save the processed DataFrame to a CSV file.
    '''
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def preprocess_data(input_dir: str, output_path: str, resample: bool = False, resample_freq: str = '1d'):
    ''' Main function to load, process, and save the data.
    '''
    df = load_data(input_dir, resample=resample, resample_freq=resample_freq)
    save_data(df, output_path)

def main():
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)

    resample = params['preprocessing']['resample']
    resample_freq = params['preprocessing']['resample_freq']

    if len(sys.argv) != 4:
        print("Usage: python src/preprocessing.py <input_dir_org> <input_dir_ref> <output_dir>")
        sys.exit(1)
    input_dir_org = sys.argv[1]
    input_dir_ref = sys.argv[2]
    output_dir = sys.argv[3]
    output_path_org = os.path.join(output_dir, 'processed_org.csv')
    output_path_ref = os.path.join(output_dir, 'processed_ref.csv')

    preprocess_data(input_dir_org, output_path_org, resample=resample, resample_freq=resample_freq)
    preprocess_data(input_dir_ref, output_path_ref, resample=resample, resample_freq=resample_freq)
    print("Preprocessing completed.")

if __name__ == "__main__":
    main()