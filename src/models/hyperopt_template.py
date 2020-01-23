# hyperopt attempt
#
# 22-10-2019

#%%
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

#%% 
MODELS_PATH = '../../models'
DATA_PATH = '../../data/raw/'
SEED = 42

X_train = pd.read_csv(os.path.join(DATA_PATH, 'dengue_features_train.csv'))
y_train = pd.read_csv(os.path.join(DATA_PATH, 'dengue_labels_train.csv'))

#%%
X_train = pd.get_dummies(X_train, columns = ['city'], drop_first = True)
X_train.set_index('week_start_date', inplace  = True)
y_train = y_train['total_cases']

#%%
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)

#%%
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample


#%% Define hyperparameter space
default_param_space = {
    'n_estimators' :  ho_scope.int(hp.quniform('n_estimator', 25, 500, 10)),
    'learning_rate' : hp.uniform('learning_rate', 0, 0.3),
    'max_depth' : ho_scope.int(hp.qlognormal('max_depth', 0.6, 1, 1)) + 1,
    'subsample' : hp.uniform('subsample', 0.7, 1),
    'reg_lambda' : hp.lognormal('reg_lambda', 2.3, 2)
}

#%%
ho_sample(default_param_space)

#%%
scores = objective(X_train, y_train, **ho_sample(default_param_space))

#%%
def objective(X, y, **hp_params):
    """
    Target function for optimization

    Parameters:
    - X: training features matrix
    - y: training target array
    - hp_params: dictionary of 
    """

    xgb_model = XGBRegressor(**hp_params)
    scores = cross_validate(
        xgb_model, X, y,
        scoring = 'neg_mean_absolute_error',
        cv = 5, 
        n_jobs = -1)    

    neg_mae = np.mean(scores['test_score'])
    return {'loss' : (-neg_mae), 'status' : STATUS_OK}

#%% 
def optimize_xgb(X_train, y_train, prev_trials = Trials(),
    hp_space = default_param_space):
    """
    Parameters:

    """
    best_param = fmin(
        fn = partial(objective, X_train, y_train), 
        space = hp_space, 
        algo = tpe.suggest,
        max_evals = 20, 
        trials = prev_trials, 
        rstate = np.random.RandomState(SEED)
    )
    
    return best_params

# %%
part = partial(objective, X_train, y_train)

# %%
