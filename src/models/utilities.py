import pandas as pd
import numpy as np
import sklearn
import os

def build_submission(
    model_sj: sklearn.base.BaseEstimator, 
    model_iq: sklearn.base.BaseEstimator, 
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
