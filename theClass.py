from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt


class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, use_mixed_effects=False, groups=None):
        self.model = None
        self.use_mixed_effects = use_mixed_effects
        self.groups = groups

    def fit(self, X, y):
        #X, y = check_X_y(X, y, y_numeric=True)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X should be a pandas DataFrame.")

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

    def plot_predictions(self, X, y, ged_sb_tlag1):
        predictions_with_indexes = self.predict(X)
        predictions_with_indexes["Actual"] = y
        predictions_with_indexes["ged_sb_tlag1"] = ged_sb_tlag1

        plt.figure(figsize=(10, 6))
        plt.scatter(predictions_with_indexes["Predictions"],
                    predictions_with_indexes["Actual"], label="Predictions vs Actual")
        plt.scatter(predictions_with_indexes["ged_sb_tlag1"],
                    predictions_with_indexes["Actual"], label="ged_sb_tlag1 vs Actual", marker='x')
        plt.xlabel("Predictions and ged_sb_tlag1")
        plt.ylabel("Actual")
        plt.legend()
        plt.title("Predictions vs Actual and ged_sb_tlag1")
        plt.show()
