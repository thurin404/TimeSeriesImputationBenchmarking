import os
import polars as pl
import sys
import yaml
import numpy as np

def load_data(data_dir: str, data_file: str) -> pl.DataFrame:
    ''' Load and preprocess data from the specified directory.
    '''
    df = pl.read_csv(os.path.join(data_dir, data_file), try_parse_dates=True)
    return df

def save_data(df: pl.DataFrame, output_dir: str, output_file: str):
    ''' Save the processed DataFrame to a CSV file.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_file)
    df.write_csv(output_path)
    print(f"Processed data saved to {output_path}")

def apply_missing_pattern(df_org: pl.DataFrame, df_ref: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    ''' Apply the missing data pattern from the original data to the reference data.
    '''
    df_ref = df_ref.with_columns(pl.col("Value").interpolate())
    df_missing = df_org.with_columns(pl.col("Value").is_null().alias("missing"))
    df_missing_features = dict(df_missing.partition_by("Feature", as_dict=True))
    df_missing_stations = dict(df_missing.partition_by("Station", as_dict=True))
    stations = df_org["Station"].unique().to_list()
    features = df_org["Feature"].unique().to_list()

    for f in features:
        print(f"Processing feature: {f}")
        feat_df = df_missing_features[(f,)].pivot(index="Date", on="Station", values="missing")
        # if a station is missing append it with mean missing pattern of other stations
        cols = feat_df.columns
        cols.remove("Date")
        cols.remove("ZAC")
        print(cols)
        feat_df = feat_df.with_columns(mean_missing=pl.mean_horizontal(cols))
        for s in stations:
            if s not in feat_df.columns:
                feat_df = feat_df.with_columns(pl.col("mean_missing").alias(s))
        df_missing_features[(f,)] = feat_df.drop("mean_missing").unpivot(index=["Date"], variable_name="Station", value_name="missing").with_columns(pl.lit(f).alias("Feature"))

    for s in stations:
        print(f"Processing station: {s}")
        stat_df = df_missing_stations[(s,)].pivot(index="Date", on="Feature", values="missing")
        # if a feature is missing append it with mean missing pattern of other features
        cols = stat_df.columns
        cols.remove("Date")
        print(cols)
        stat_df = stat_df.with_columns(mean_missing=pl.mean_horizontal(cols))
        for f in features:
            if f not in stat_df.columns:
                stat_df = stat_df.with_columns(pl.col("mean_missing").alias(f))
        df_missing_stations[(s,)] = stat_df.drop("mean_missing").unpivot(index=["Date"], variable_name="Feature", value_name="missing").with_columns(pl.lit(s).alias("Station"))


    stat_feat_list = ['{"' + s + '","' + f + '"}' for s in stations for f in features]
    columns_list = df_missing.pivot(index="Date", on=["Station", "Feature"], values="missing").columns

    for s_f in stat_feat_list:
        if s_f not in columns_list:
            station = s_f.split('","')[0][2:]
            feature = s_f.split('","')[1][:-2]
            print(f"Adding missing combination: {s_f}")
            feat_missing = df_missing_features[(feature,)].filter(pl.col("Station") == station).with_columns(pl.col("missing").alias("feat_missing")).drop("missing")
            stat_missing = df_missing_stations[(station,)].filter(pl.col("Feature") == feature).with_columns(pl.col("missing").alias("stat_missing")).drop("missing")
            mean_missing = pl.concat([feat_missing, stat_missing], how="align").with_columns(np.rint(pl.mean_horizontal(["feat_missing", "stat_missing"])).cast(pl.Boolean).alias("missing")).drop(["feat_missing", "stat_missing"])
            df_missing = pl.concat([df_missing, mean_missing], how="align")

    # replace station names of df_missing to df_ref
    stations_ref = df_ref["Station"].unique().to_list()
    stations_org = df_org["Station"].unique().to_list()
    station_map = {key: value for key, value in zip(stations_org, stations_ref)}
    df_missing = df_missing.with_columns(pl.col("Station").map_elements(lambda s: station_map.get(s, None)))

    df_ref = df_ref.join(df_missing, on=["Date", "Station", "Feature"], how="left")
    df_ref = df_ref.with_columns(
        pl.when(~pl.col("missing"))
        .then(pl.col("Value"))
        .otherwise(pl.lit(None).cast(pl.Float64))
        .alias("Value_missing")
    ).drop("Value_right")
    return df_missing, df_ref
    

def clip_data(df_org: pl.DataFrame, df_ref: pl.DataFrame, method: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    ''' Clip the values in the DataFrames based on the specified method.
    '''
    df_org = df_org.filter(pl.col("Station")!="DAN")
    df_ref = df_ref.filter(pl.col("Station")!=df_ref["Station"].unique().to_list()[-1])
    if method == "default":
        # determine the index range of the reference data and apply it to both dataframes
        start_idx = df_ref['Date'].min()
        end_idx = df_ref['Date'].max()
        df_org = df_org.filter((df_org['Date'] >= start_idx) & (df_org['Date'] <= end_idx))
        df_ref = df_ref.filter((df_ref['Date'] >= start_idx) & (df_ref['Date'] <= end_idx))

    elif method == "moderate":
        pass  # No clipping applied for moderate
    elif method == "aggressive":
        pass  # No clipping applied for aggressive
    elif method == "False":
        pass  # No clipping applied for False

    return df_org, df_ref

def main():
    """ 
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)

    input_dir = params['preproc']['dir']
    data_file_org = params['preproc']['files']['org']
    data_file_ref = params['preproc']['files']['ref']

    output_dir = params['prepared']['dir']
    output_file_org = params['prepared']['files']['org']
    output_file_ref = params['prepared']['files']['ref']
 """
    if len(sys.argv) != 5:
        print("Usage: python src/prepare.py <input_dir> <data_file_org> <data_file_ref> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    data_file_org = sys.argv[2]
    data_file_ref = sys.argv[3]
    output_dir = sys.argv[4]
    output_file_org = "prepared_org.csv"
    output_file_ref = "prepared_ref.csv"
    
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)
    clip_method = params['prepare']['clip']

    df_org: pl.DataFrame = load_data(input_dir, data_file_org)
    df_ref: pl.DataFrame = load_data(input_dir, data_file_ref)
    df_org, df_ref = clip_data(df_org, df_ref, clip_method)
    df_org, df_ref = apply_missing_pattern(df_org, df_ref)
    save_data(df_org, output_dir, output_file_org)
    save_data(df_ref, output_dir, output_file_ref)
    print("Data preparation completed.")

if __name__ == "__main__":
    main()