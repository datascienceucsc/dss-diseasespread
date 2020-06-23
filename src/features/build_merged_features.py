import pandas as pd 
import os
from typing import List


def make_merged_data(
    raw_path: str, processed_path: str, data: str, build_files: bool = True
)-> pd.DataFrame:

    if data not in ['train', 'test']:
        raise ValueError('Argument \'data\' must be one of \'train\', \'test')

    features = pd.read_csv(
        os.path.join(raw_path, 'dengue_features_' + data + '.csv'))

    # kelvin conversions
    features['station_max_temp_c'] += 273.15
    features['station_min_temp_c'] += 273.15
    features['station_avg_temp_c'] += 273.15

    features = (features
        .drop(['weekofyear', 'year'], axis = 1)    
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
        .drop( # features strongly correlated to other features
            ['reanalysis_specific_humidity_g_per_kg'],
            axis = 1
        )
    )
    if build_files:
        features.to_csv(
            os.path.join(processed_path, 'merged_features_' + data + '.csv'),
            index = False
        )

    return features

if __name__ == "__main__":
    
    make_merged_data(
        '../../data/raw','../../data/processed', data = 'train'
    )
    make_merged_data(
        '../../data/raw','../../data/processed',data = 'test'
    )