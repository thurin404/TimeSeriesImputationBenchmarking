import os
import sys
from typing import Optional
import joblib
import polars as pl
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(data_dir: str, data_file: str) -> pl.DataFrame:
    ''' Load and preprocess data from the specified directory.
    '''
    df = pl.read_csv(os.path.join(data_dir, data_file), try_parse_dates=True)
    return df


def generate_working_files(dir: str = "../data/04_working", df: Optional[pl.DataFrame] = None, 
                           partition_by: str = "Feature", scaling: bool = False, scaler_dir: str = "../data/03_scalers", resampling: bool = False) -> None:
    ''' Generate working files partitioned by Feature or Station.
    '''
    if df is None:
        print("DataFrame is None, please provide a valid DataFrame.")
        return

    DIR = dir
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    df_pandas = df.to_pandas()
    df_pandas_wide = df_pandas.pivot(index="Date", columns=["Station", "Feature"], values="Value")
    df_pandas_wide_missing = df_pandas.pivot(index="Date", columns=["Station", "Feature"], values="Value_missing")
    df_pandas_wide_mask = df_pandas.pivot(index="Date", columns=["Station", "Feature"], values="missing")

    if scaling:
        scaler = StandardScaler()
        scaler.fit(df_pandas_wide.loc[:, :])
        df_pandas_wide_missing.loc[:, :] = scaler.transform(df_pandas_wide_missing.loc[:, :])
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        scaler_file = os.path.join(scaler_dir, "scaler_global.pkl")
        joblib.dump(scaler, scaler_file)

    def add_features(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("Date").dt.year().alias("Year"),
                              pl.col("Date").dt.month().alias("Month"),
                              pl.col("Date").dt.day().alias("Day"),
                              pl.col("Date").dt.hour().alias("Hour") if not resampling else pl.lit(0).alias("Hour"))
    
    def save_data(data_dict: dict, filename: str) -> None:
        with open(os.path.join(DIR, filename), "wb") as f:
            joblib.dump(data_dict, f)
            #
    def scale_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
        datetime_scaler = MinMaxScaler()
        datetime_features = df[["Year", "Month", "Day", "Hour"]]
        df[["Year", "Month", "Day", "Hour"]] = datetime_scaler.fit_transform(datetime_features)
        return df


    if partition_by == "Feature":
        df_long = pl.from_pandas(df_pandas_wide.melt(ignore_index=False),include_index=True)
        df_long_missing = pl.from_pandas(df_pandas_wide_missing.melt(ignore_index=False),include_index=True)
        df_long_mask = pl.from_pandas(df_pandas_wide_mask.melt(ignore_index=False),include_index=True)

        data = df_long.partition_by("Feature", as_dict=True)
        data_missing = df_long_missing.partition_by("Feature", as_dict=True)
        data_mask = df_long_mask.partition_by("Feature", as_dict=True)

        data_pandas, data_missing_pandas, data_mask_pandas = {}, {}, {}

        for key in data_missing.keys():
            data_pandas[key[0]] = data[key].pivot(index="Date", on="Station", values="value").to_pandas().set_index('Date')
            data_missing_pandas[key[0]] = scale_datetime_features(add_features(data_missing[key].pivot(index="Date", on="Station", values="value")).to_pandas().set_index('Date'))
            data_mask_pandas[key[0]] = data_mask[key].pivot(index="Date", on="Station", values="value").to_pandas().set_index('Date')
        
        save_data(data_pandas, "ts_feature_gt.pkl")
        save_data(data_missing_pandas, "ts_feature_missing.pkl")
        save_data(data_mask_pandas, "ts_feature_mask.pkl")

    elif partition_by == "Station":
        df_long = pl.from_pandas(df_pandas_wide.melt(ignore_index=False),include_index=True)
        df_long_missing = pl.from_pandas(df_pandas_wide_missing.melt(ignore_index=False),include_index=True)
        df_long_mask = pl.from_pandas(df_pandas_wide_mask.melt(ignore_index=False),include_index=True)

        data = df_long.partition_by("Station", as_dict=True)
        data_missing = df_long_missing.partition_by("Station", as_dict=True)
        data_mask = df_long_mask.partition_by("Station", as_dict=True)

        data_pandas, data_missing_pandas, data_mask_pandas = {}, {}, {}

        for key in data_missing.keys():
            data_pandas[key[0]] = data[key].pivot(index="Date", on="Feature", values="value").to_pandas().set_index('Date')
            data_missing_pandas[key[0]] = scale_datetime_features(add_features(data_missing[key].pivot(index="Date", on="Feature", values="value")).to_pandas().set_index('Date'))
            data_mask_pandas[key[0]] = data_mask[key].pivot(index="Date", on="Feature", values="value").to_pandas().set_index('Date')

        save_data(data_pandas, "ts_station_gt.pkl")
        save_data(data_missing_pandas, "ts_station_missing.pkl")
        save_data(data_mask_pandas, "ts_station_mask.pkl")

    elif partition_by == "Wide":
        df_pandas_wide_missing["Year"] = df_pandas_wide_missing.index.year  # type: ignore
        df_pandas_wide_missing["Month"] = df_pandas_wide_missing.index.month # type: ignore
        df_pandas_wide_missing["Day"] = df_pandas_wide_missing.index.day  # type: ignore
        df_pandas_wide_missing["Hour"] = df_pandas_wide_missing.index.hour if not resampling else 0  # type: ignore
        df_pandas_wide_missing = scale_datetime_features(df_pandas_wide_missing)
        
        data_pandas, data_missing_pandas, data_mask_pandas = {}, {}, {}

        data_pandas["Wide"] = df_pandas_wide
        data_missing_pandas["Wide"] = df_pandas_wide_missing
        data_mask_pandas["Wide"] = df_pandas_wide_mask

        save_data(data_pandas, "ts_wide_gt.pkl")
        save_data(data_missing_pandas, "ts_wide_missing.pkl")
        save_data(data_mask_pandas, "ts_wide_mask.pkl")


def main():
    if len(sys.argv) != 5:
        print("Usage: python prepare_working_files.py <data_dir> <data_file> <data_dir_output>")
        sys.exit(1)

    data_dir = sys.argv[1]
    data_file = sys.argv[2]
    data_dir_output = sys.argv[3]
    scaler_dir = sys.argv[4]

    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)
    partition_by = params["prepare_working_files"]["partition_by"]
    scaling = params["prepare_working_files"]["scaling"]
    resampling = params["preprocessing"]["resample"]

    df = load_data(data_dir, data_file)
    generate_working_files(dir=data_dir_output, df=df, partition_by=partition_by, scaling=scaling, scaler_dir=scaler_dir, resampling=resampling)

if __name__ == "__main__":
    main()