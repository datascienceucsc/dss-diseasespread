import pandas as pd 
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
            [
                'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw',
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
