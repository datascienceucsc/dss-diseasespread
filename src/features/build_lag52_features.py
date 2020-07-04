import pandas as pd 
import os
import sys

from typing import Tuple
from utilities import add_lagged_features

def make_lag52_features(features: pd.DataFrame) -> pd.DataFrame:
    """ 
    Takes test set  concatenated to training set. Drops correlated
    features and builds features up to lag 52
    """

    features = (features
        .drop( # correlated features
            ['reanalysis_sat_precip_amt_mm', 'reanalysis_dew_point_temp_k', 
             'reanalysis_air_temp_k', 'reanalysis_tdtr_k'],
            axis = 1
        )
        .fillna(method = 'backfill')
        .drop( # unused features
            ['year', 'weekofyear','week_start_date'], 
            axis = 1
        )
    )
    ts_features = list(features.loc[:, 'ndvi_ne' :].columns.values)

    features = add_lagged_features(
        features, 52, ts_features)
    
    return features


def process_city(
    train_features: pd.DataFrame, 
    test_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    """
    Builds lagged features for a given city, then splits back 
    training and test sets and removes the first year of data
    """
    features = pd.concat([train_features, test_features])
    features = make_lag52_features(features)

    train_features = features.iloc[52:-test_size, :] 
    test_features = features.iloc[-test_size:, :]
    train_labels = train_labels.iloc[52:]

    return (train_features, test_features, train_labels)

def make_lag_52_dataset(raw_path: str, processed_path: str, build_files: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_features = pd.read_csv(
        os.path.join(raw_path, 'dengue_features_train.csv')) 
    test_features = pd.read_csv(
        os.path.join(raw_path, 'dengue_features_test.csv')) 
    train_labels = pd.read_csv(
        os.path.join(raw_path, 'dengue_labels_train.csv'))

    train_features_sj, test_features_sj, train_labels_sj = process_city(
        train_features[train_features['city'] == 'sj'],
        test_features[test_features['city'] == 'sj'],
        train_labels[train_labels['city'] == 'sj'],
        test_size = 260
    )

    train_features_iq, test_features_iq, train_labels_iq = process_city(
        train_features[train_features['city'] == 'iq'],
        test_features[test_features['city'] == 'iq'],
        train_labels[train_labels['city'] == 'iq'],
        test_size = 156
    )

    train_features = pd.concat([train_features_sj, train_features_iq])
    test_features = pd.concat([test_features_sj, test_features_iq])
    train_labels = pd.concat([train_labels_sj, train_labels_iq])

    if build_files == True:
        train_features.to_csv(
            os.path.join(processed_path, 'lag52_train_features.csv'), 
            index = False
        )
        test_features.to_csv(
            os.path.join(processed_path, 'lag52_test_features.csv'),
            index = False
        )
        train_labels.to_csv(
            os.path.join(processed_path, 'lag52_train_labels.csv'),
            index = False
        )
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(
            'Usage: python build_merged_lagged.py <raw data path> <processed data path>'
        )
    make_lag_52_dataset(sys.argv[1], sys.argv[2])