import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
from scipy import stats
from viewser.operations import fetch
from viewser import Queryset, Column
import subprocess


class StatsmodelsFixedAndRandomEffectsRegression:
    def __init__(self, data):
        self.data = data
        self.models = {}  # Dictionary to store regression models
        self.results = {}  # Dictionary to store regression results

    def fixed_effects_regression(self, x_columns, y_column,fixed_effect_index):
        X = self.data[x_columns]
        X = sm.add_constant(X)  # Add a constant term for the intercept
        y = self.data[y_column]
        y[fixed_effect_index] = y.index.get_level_values(fixed_effect_index)


        model = sm.OLS(y, X).fit()
        self.models['Fixed Effects Regression'] = model
        self.results['Fixed Effects Regression'] = model.summary()

    def random_effects_regression(self, x_columns, y_column, group_column):
        X = self.data[x_columns]
        X = sm.add_constant(X)  # Add a constant term for the intercept
        y = self.data[y_column]
        groups = self.data[group_column]

        model = MixedLM(y, X, groups)
        result = model.fit()
        
        self.models['Random Effects Regression'] = result
        self.results['Random Effects Regression'] = result.summary()

    def evaluate_model(self, model_name):
        return self.results.get(model_name, None)

    def plot_regression(self, x_columns, y_column, model_name):
        X = self.data[x_columns]
        y = self.data[y_column]
        model = self.models[model_name]
        y_pred = model.predict(sm.add_constant(X))
        
        plt.scatter(y, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Actual vs. Predicted')
        plt.show()

    @staticmethod
    def f_test(model1, model2):
        f_value = (model1.ssr - model2.ssr) / (model2.df_model - model1.df_model) / (model2.ssr / model2.df_resid)
        p_value = 1 - stats.f.cdf(f_value, model2.df_model - model1.df_model, model2.df_resid)
        return f_value, p_value
'''
# Example usage:
data1 = pd.DataFrame({'X1': [1, 2, 3, 4, 5],
                     'X2': [2, 3, 4, 5, 6],
                     'Y': [2, 4, 6, 8, 10]})

data2 = pd.DataFrame({'X1': [1, 2, 3, 4, 5],
                     'X2': [2, 3, 4, 5, 6],
                     'Y': [3, 6, 8, 10, 12]})

independent_variables = ['X1', 'X2']
dependent_variable = 'Y'
statsmodels_regression1 = StatsmodelsFixedAndRandomEffectsRegression(data1)
statsmodels_regression2 = StatsmodelsFixedAndRandomEffectsRegression(data2)

# Fixed Effects Regression for data1
statsmodels_regression1.fixed_effects_regression(independent_variables, dependent_variable)

# Random Effects Regression for data2
statsmodels_regression2.random_effects_regression(independent_variables, dependent_variable, group_column)

# Perform an F-test to compare the two models
f_value, p_value = StatsmodelsFixedAndRandomEffectsRegression.f_test(
    statsmodels_regression1.models['Fixed Effects Regression'],
    statsmodels_regression2.models['Random Effects Regression']
)
print(f"F-value: {f_value}, p-value: {p_value}")
'''