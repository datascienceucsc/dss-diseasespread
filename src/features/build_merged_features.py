import pandas as pd 
import os
import sys
from typing import List


def merge_features(df: pd.DataFrame)-> pd.DataFrame:
    """ 
    Merges features that estimate the same thing
    """

    # kelvin conversions
    df['station_max_temp_c'] += 273.15
    df['station_min_temp_c'] += 273.15
    df['station_avg_temp_c'] += 273.15

    df = (df
        .fillna(method = 'backfill')
        .assign( # create average estimates for data
            ndvi_n = lambda x: 
                (x['ndvi_ne'] + x['ndvi_nw']) / 2,
            ndvi_s = lambda x: 
                (x['ndvi_se'] + x['ndvi_sw']) / 2,
            max_temp_k = lambda x: 
                (x['station_max_temp_c'] + x['reanalysis_max_air_temp_k']) /2,
            min_temp_k = lambda x: 
                (x['station_min_temp_c'] + x['reanalysis_min_air_temp_k']) /2,
            avg_temp_k = lambda x:
                (x['station_avg_temp_c'] + x['reanalysis_avg_temp_k'] + x['reanalysis_air_temp_k']) / 3,
            precip_mm = lambda x:
                (x['station_precip_mm'] + x['reanalysis_sat_precip_amt_mm']) / 2,
            diur_temp_rng_k = lambda x:
                (x['station_diur_temp_rng_c'] + x['reanalysis_tdtr_k']) / 2
            
        )
        .drop( # features for which we created average estimates
            ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw',
            'station_max_temp_c', 'reanalysis_max_air_temp_k',
            'station_min_temp_c', 'reanalysis_min_air_temp_k',
            'station_avg_temp_c', 'reanalysis_avg_temp_k', 'reanalysis_air_temp_k',
            'station_precip_mm', 'reanalysis_sat_precip_amt_mm',
            'station_diur_temp_rng_c', 'reanalysis_tdtr_k'
            ],
            axis = 1
        )
    )
    return df

def make_merged_data(
    raw_path: str, processed_path: str,data: str, build_files: bool = True
) -> pd.DataFrame:

    if data not in ['train', 'test']:
        raise ValueError('Argument \'data\' must be one of \'train\', \'test')

    features = pd.read_csv(
         os.path.join(raw_path, 'dengue_features_' + data + '.csv')
    )
    features = merge_features(features)
    features.drop(
        ['year', 'weekofyear', 'reanalysis_specific_humidity_g_per_kg'],
        axis = 1
    )

    if build_files:
        features.to_csv(
            os.path.join(processed_path, 'merged_features_' + data + '.csv'),
            index = False
    )

    return features

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            'Usage: python build_merged_features.py <raw data path> <processed data path>'
        )

    make_merged_data(sys.argv[1], sys.argv[2], data = 'train')
    make_merged_data(sys.argv[1], sys.argv[2], data = 'test')