import pandas as pd 
import os 
import sys
from typing import List

def add_lagged_features(df: pd.DataFrame, max_lag: int,
 features: List[str]) -> pd.DataFrame:
    """
    Creates columns with lagged data up to lag max_lag for each column 
    listed in features
    """

    lag_df = [
        df[features].shift(k).add_prefix('lag' + str(k) + '_') 
        for k in range(1, max_lag+1)
    ]
    return pd.concat([df] + lag_df, axis = 1)

def make_lagged_data(
    raw_path: str, processed_path: str, data: str, build_files: bool = True
) -> pd.DataFrame:

    if data not in ['train', 'test']:
        raise ValueError('Argument \'data\' must be one of \'train\', \'test')

    features = pd.read_csv(
        os.path.join(raw_path, 'dengue_features_' + data + '.csv'))
    
    features = (features
        .drop( # correlated features
            ['reanalysis_sat_precip_amt_mm', 'reanalysis_dew_point_temp_k', 
             'reanalysis_air_temp_k', 'reanalysis_tdtr_k'],
            axis = 1
        )
        .fillna(method = 'backfill')
        .assign(
            ndvi_n = lambda x : x['ndvi_ne'] + x['ndvi_nw'] / 2,
            ndvi_s = lambda x : x['ndvi_se'] + x['ndvi_sw'] / 2,
            monthofyear = lambda x: pd.to_datetime(x['week_start_date']).dt.month
        )
        .drop( # unused features
            ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'year', 'weekofyear',
             'week_start_date'], 
            axis = 1
        )
    )
    ts_features = list(
        features.loc[
            :, 'reanalysis_relative_humidity_percent' : 'ndvi_s'
    ].columns.values)

    features_sj = features[
        features['city'] == 'sj']
    features_iq = features[
        features['city'] == 'iq']

    features_sj = add_lagged_features(
        features_sj, 7, ts_features).fillna(method = 'backfill')
    features_iq = add_lagged_features(
        features_iq, 7, ts_features).fillna(method = 'backfill')

    features = pd.concat([features_sj, features_iq], axis = 0)
    if build_files:
        features.to_csv(
            os.path.join(processed_path, 'lag7_features_' + data + '.csv'),
            index = False
        )

    return features
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            'Usage: python build_lagged_features.py <raw data path> <processed data path>'
        )

    make_lagged_data(sys.argv[1], sys.argv[2], data = 'train')
    make_lagged_data(sys.argv[1], sys.argv[2], data = 'test')