import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt


class MonthToAnnual(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='or'):
        self.model_type = model_type
        self.model = None

    def fit(self, X, y):
        X = sm.add_constant(X)
        # Adjust for other model types
        self.model = sm.OLS(y, X).fit() if self.model_type == 'or' else None
        return self

    def predict(self, X):
        check_is_fitted(self, 'model')
        X = sm.add_constant(X)
        # Adjust for other model types
        return self.model.predict(X) if self.model else None

    def plot_predictions(self, X, y):
        check_is_fitted(self, 'model')
        X_with_const = sm.add_constant(X)
        predictions = self.model.predict(X_with_const) if self.model else None

        if predictions is not None:
            plt.scatter(X, y, color='black', label='Actual')
            plt.plot(X, predictions, color='blue', linewidth=3, label='Predicted')
            plt.xlabel('Independent Variable')
            plt.ylabel('Dependent Variable')
            plt.title('Predictions')
            plt.legend()
            plt.show()
        else:
            print("Model has not been fitted or predictions are unavailable.")

