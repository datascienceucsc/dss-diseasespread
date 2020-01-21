# get_data.py
# 
# Anders Poirel
# 21-01-2020

import requests
from os.path import join

RAW_DATA_PATH = "../../data/raw/"

train_features_url = "https://s3.amazonaws.com/drivendata/data/44/public/dengue_features_train.csv"

train_labels_url = "https://s3.amazonaws.com/drivendata/data/44/public/dengue_labels_train.csv"

test_features_url = "https://s3.amazonaws.com/drivendata/data/44/public/dengue_features_test.csv"

sample_sub_url = "https://s3.amazonaws.com/drivendata/data/44/public/submission_format.csv"

r = requests.get(train_features_url)
with open(join(RAW_DATA_PATH, "train_features.csv"), 'wb') as f:
    f.write(r.content)

r = requests.get(train_labels_url)
with open(join(RAW_DATA_PATH, "train_labels.csv"), 'wb') as f:
    f.write(r.content)

r = requests.get(test_features_url)
with open(join(RAW_DATA_PATH, "test_features.csv"), 'wb') as f:
    f.write(r.content)

