# hyperopt function wrapper
#
# 22-10-2019
# Anders Poirel
#
# To-do: change cross-validation type to timeseries?

#%%
import os
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope

default_param_space = {
    'n_estimators' :  ho_scope.int(hp.quniform('n_estimator', 25, 500, 10)),
    'learning_rate' : hp.uniform('learning_rate', 0, 0.3),
    'max_depth' : ho_scope.int(hp.qlognormal('max_depth', 0.6, 1, 1)) + 1,
    'subsample' : hp.uniform('subsample', 0.7, 1),
    'reg_gamma' : hp.lognormal('reg_gamma', 2.3, 2),
    'reg_lamba' : hp.uniform('reg_lambda', 0, 1),
    'reg_alpha' : hp.uniform('reg_alpha', 0, 1)
}

def objective(X, y, hp_params, n_folds):
    """
    Objective function for bayesian optimization

    Parameters:
    - X: training features matrix
    - y: training target vector
    - hp_params: parameter dictionary for xgboost
    - n_folds: number of folds to use in cross_validation

    Returns:
    Mean absoute error across the n folds
    """

    xgb_model = XGBRegressor(**hp_params)
    scores = cross_validate(
        xgb_model, X, y,
        scoring = 'neg_mean_absolute_error',
        cv = n_folds, 
        n_jobs = -1)    

    neg_mae = np.mean(scores['test_score'])
    return {'loss' : (-neg_mae), 'status' : STATUS_OK}

def optimize_xgb(X_train, y_train, n_iter = 20, prev_trials = Trials(),
    hp_space = default_param_space):
    """
    Parameters:
    - X_train: training feature matrix. All columns must be numeric.
    - y_train: training target vector
    - prev_trials: hyperopt.Trials() object used to save progress
    - hp_space: space of hyperparameters to explore 

    Default: 
    
    {
    'n_estimators' :  ho_scope.int(hp.quniform('n_estimator', 25, 500, 10)),

    'learning_rate' : hp.uniform('learning_rate', 0, 0.3),

    'max_depth' : ho_scope.int(hp.qlognormal('max_depth', 0.6, 1, 1)) + 1,

    'subsample' : hp.uniform('subsample', 0.7, 1),

    'reg_gamma' : hp.lognormal('reg_gamma', 2.3, 2),

    'reg_lamba' : hp.uniform('reg_lambda', 0, 1),

    'reg_alpha' : hp.uniform('reg_alpha', 0, 1)
    }

    Returns:
    Dictionary of best parameters for an XGBRegressor

    """
    best_params = fmin(
        fn = partial(objective, X_train, y_train),
        space = hp_space,
        algo = tpe.suggest,
        max_evals = n_iter, 
        trials = prev_trials)

    return best_params
