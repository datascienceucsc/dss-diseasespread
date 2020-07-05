import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

def build_submission(
    model_sj: BaseEstimator, 
    model_iq: BaseEstimator, 
    test_features_sj: pd.DataFrame,
    test_features_iq: pd.DataFrame,
    raw_path: str,
    pred_path: str,
    name: str
)-> pd.DataFrame:

    submission = pd.read_csv(
        os.path.join(raw_path, 'submission_format.csv'))

    y_pred_sj = model_sj.predict(test_features_sj)
    y_pred_iq = model_iq.predict(test_features_iq)
    y_pred = np.concatenate((y_pred_sj, y_pred_iq))

    submission['total_cases'] = np.round(y_pred).astype(int)
    submission.to_csv(
        os.path.join(pred_path, name + '.csv'),
        index = None
    )

    return submission
    

# class from wrapping statsmodels regressors with a scikit-learn
# interface

class SMWrapper(BaseEstimator, RegressorMixin):
    """ 
    A universal sklearn-style wrapper for statsmodels regressors 
    """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)