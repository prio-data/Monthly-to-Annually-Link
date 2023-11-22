from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from sklearn.linear_model import LinearRegression
import numpy as np


class OLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.model = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


