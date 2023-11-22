import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
from scipy import stats
from viewser.operations import fetch
from viewser import Queryset, Column
import subprocess


class MonthToAnnual:
    def __init__(self, dependent_variable, independent_variable, model_type):
        self.independent_variable = independent_variable
        self.dependent_variable = dependent_variable
        self.model_type = model_type
        self.model = None
        self.predicted_annual = None

    def fit(self):
        if self.model_type == 'or':
            self.model = sm.OLS(self.dependent_variable,
                                sm.add_constant(self.independent_variable)).fit()
        elif self.model_type == 'fe':
           pass
        elif self.model_type == 'me':
            pass

    def predict_annual(self):
        # Your logic to predict annual values based on the fitted model
        # Store predictions in self.predicted_annual
        pass  # Placeholder, replace with prediction logic

    def predict(self, new_data):
        # Your logic to predict values for new data using the fitted model
        # Example:
        # return self.model.predict(new_data)
        print(new_data)
        return self.model.predict(sm.add_constant(new_data))
        # Placeholder, replace with prediction logic
    
    def plot_predictions(self):
        # Your logic to plot predictions
        pass  # Placeholder, replace with plotting logic

