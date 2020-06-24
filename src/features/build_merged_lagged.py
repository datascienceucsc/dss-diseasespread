import pandas as pd 
import os
import sys

from build_lagged_features import add_lagged_features
from build_merged_features import merge_features  

def make_features(raw_path: str, processed_path: str, data: str):

    df = pd.read_csv(
        os.path.join(raw_path, 'dengue_features_' + data + '.csv')
    )

    df = merge_features(df)
    df = df.drop( # drop unused and correlated features
        ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg'],
        axis = 1
    )
    ts_features = list(df.loc[:, 'precipitation_amt_mm':].columns.values)

    df_sj = df[df['city'] == 'sj']
    df_iq = df[df['city'] == 'iq']

    df_sj = add_lagged_features(
        df_sj, 7, ts_features).fillna(method = 'backfill')
    df_iq = add_lagged_features(
        df_iq, 7, ts_features).fillna(method = 'backfill')
    df = pd.concat([df_sj, df_iq], axis = 0)

    print(df.columns.values)

    df.to_csv(
        os.path.join(processed_path, 'merged_lag7_features_' + data + '.csv'),
        index = False
    )

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            'Usage: python build_merged_lagged.py <raw data path> <processed data path>'
        )

    make_features(sys.argv[1], sys.argv[2], data = 'train')
    make_features(sys.argv[1], sys.argv[2], data = 'test')



 