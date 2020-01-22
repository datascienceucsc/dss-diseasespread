# hyperopt attempt
#
# 22-10-2019

#%%
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

#%% 
DATA_PATH = '../../data/raw/'
X_train = pd.read_csv(os.path.join(DATA_PATH, 'dengue_features_train.csv'))
y_train = pd.read_csv(os.path.join(DATA_PATH, 'dengue_labels_train.csv'))

#%%
X_train = pd.get_dummies(X_train, columns = ['city'], drop_first = True)
X_train.set_index('week_start_date', inplace  = True)
y_train = y_train['total_cases']


#%%
imputer = IterativeImputer(
    estimator = ExtraTreesRegressor(
        n_estimators = 100, 
        criterion = 'mae', 
        max_depth = 10)
)
X_train = imputer.fit_transform(X_train)

#%%
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
SEED = 21


#%% Defining search space
hp_space = {
    'n_estimators' :  hp.quniform('n_estimator', 25, 500, 10),
    'learning_rate' : hp.choice('learning_rate', 
                        [hp.uniform('learning_rate', 0, 0.3) , 1],
    'max_depth' : hp.qlognormal('max_depth', 0.7, 1, 1) + 1,
    'subsample' : hp.uniform('subsample', 0.7, 1),
    'reg_lambda' : hp.lognormal('reg_lambda', 2.3, 2)
}

#%%
ho_sample(hp_space)

#%%
def objective(n_folds):


    return score