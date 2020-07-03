import pandas as pd 
import os
import sys
from typing import List

from utilies import merge_features

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