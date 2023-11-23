from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt


class Regressor(BaseEstimator, RegressorMixin):
    '''
    This class is a very specific class for the project Month to Annual
    It has functions for OLS and MixedLM from statsmodels
    This class inherits from scikit-learn BaseEstimator
    It creates a regression model based on December months only from VIEWS data
    But it can be used to predict the outcomes for every month
    '''

    def __init__(self, use_mixed_effects=False, groups=None):
        self.model = None
        self.use_mixed_effects = use_mixed_effects
        if groups is not None:
            self.groups = groups.loc[groups.index.get_level_values(
                'month_id') % 12 == 0]
        else:
            self.groups = groups

    def fit(self, X, y):

        # X, y = check_X_y(X, y, y_numeric=True)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X should be a pandas DataFrame.")
        X = X.query('month_id % 12 == 0')
        y = y.loc[y.index.get_level_values(
            'month_id') % 12 == 0]
        if self.use_mixed_effects:
            if self.groups is None:
                raise ValueError(
                    "Groups should be provided for mixed-effects modeling.")
            X = sm.add_constant(X)
            self.model = MixedLM(y, X, groups=self.groups).fit()
        else:

            X = sm.add_constant(X)
            self.model = sm.OLS(y, X).fit()
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X should be a pandas DataFrame.")

        X_const = sm.add_constant(X)
        predictions = self.model.predict(X_const)

        # Adding the indexes to the predictions
        predictions_with_indexes = pd.DataFrame(
            predictions, index=X.index, columns=["Predictions"])
        predictions_with_indexes.index.name = X.index.name

        return predictions_with_indexes

    def predict_annual(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X should be a pandas DataFrame.")

        X_const = sm.add_constant(X.query('country_id % 12 == 0'))
        predictions = self.model.predict(X_const)

        # Adding the indexes to the predictions
        predictions_with_indexes = pd.DataFrame(
            predictions, index=X.index, columns=["Annual Predictions"])
        predictions_with_indexes.index.name = X.index.name

        return predictions_with_indexes
