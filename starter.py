#-----------------------------------------------------------------------------
# starter.py
#
# Anders Poirel
# 2-01-2020
#-----------------------------------------------------------------------------

import numpy as np 
import pandas as pd

TRAIN_LEN = 1456
TEST_LEN = 416

X_train = pd.read_csv("data/dengue_features_train.csv")
X_test = pd.read_csv("data/dengue_features_test.csv")

y_train_pred = np.zeroes(LEN) 
y_test_pred = np.zeroes(LEN)

#-----------------------------------------------------------------------------
#
# Code saving predicitons to y_train_pred and y_test_pred here
#
#-----------------------------------------------------------------------------

np.save("y_train_pred.npy", y_train_pred, allow_pickle = False)
np.save("y_test_pred.npy", y_test_pred, allow_pickle = False)
