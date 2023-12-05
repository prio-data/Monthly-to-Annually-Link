import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
from scipy import stats
from viewser.operations import fetch
from viewser import Queryset, Column
import subprocess
from scipy.stats import chi2


class MonthToAnnualRegression:
    def __init__(self, data):
        self.data = data
        self.predictions = {}
        self.models = {}  # Dictionary to store regression models
        self.results = {}  # Dictionary to store regression results

    def ols_regression(self, x_columns, y_column, model_name):
        filtered_rows = self.data.index.get_level_values('month_id') % 12 == 0

        # Filter X and y Series based on the condition
        X = self.data.loc[filtered_rows, x_columns]
        dummy_vars = pd.get_dummies(
            self.data.loc[filtered_rows, 'country_id'], prefix='country', drop_first=True, dtype=int)
        X = pd.concat([X, dummy_vars], axis=1)
        # X = sm.add_constant(X)  # Add a constant term for the intercept
        print(list(X.columns))
        y = self.data.loc[filtered_rows, y_column]
        print(y)

        model = sm.OLS(y, X).fit()
        self.models[model_name] = model
        self.results[model_name] = model.summary()

    def prediction(self, x_columns, y_column, model_name):
        filtered_rows = self.data.index.get_level_values('month_id') % 1 == 0

        X = self.data.loc[filtered_rows, x_columns]
        dummy_vars = pd.get_dummies(
            self.data.loc[filtered_rows, 'country_id'], prefix='country', drop_first=True, dtype=int)
        X = pd.concat([X, dummy_vars], axis=1)
        # X = sm.add_constant(X)  # Add a constant term for the intercept
        print(list(X.columns))
        y = self.data.loc[filtered_rows, y_column]

        model = self.models[model_name]
        y_pred = model.predict(X)
        self.predictions[model_name] = y_pred

    def plot_time_series_regression(self, x_columns, y_column, model_name, country_id, shock_variable):
        country_data = self.data[(self.data['country_id'] == country_id)]
        # Get the country_name for the given country_id
        country_name = country_data['country_name'].iloc[0]

        # country_data = country_data.set_index('month_id').sort_index()
        x = country_data[x_columns]
        y = country_data[y_column]
        # model = self.models[model_name]
        y_pred = self.predictions[model_name][self.predictions[model_name].index.get_level_values(
            'country_id') == country_id]
        # plt.plot(y)
        # plt.plot(y_pred)
        # print(y)
        # print(y_pred)
        df1 = y_pred.reset_index()
        df1.columns = ['month_id', 'country_id', 'value']
        df2 = y.reset_index()
        df2.columns = ['month_id', 'country_id', 'value']
        # Plotting
        plt.plot(df1['month_id'], df1['value'], label='Predicted')
        plt.plot(df2['month_id'], df2['value'], label='Actual')

        plt.xlabel('Month ID')
        plt.ylabel('Infant Mortality')
        plt.title(f'{model_name} for Country: {country_name}')
        plt.legend()
        # Create a twin y-axis for shock variable
        ax2 = plt.gca().twinx()
        ax2.bar(df2['month_id'], x[shock_variable],
                color='gray', alpha=0.5, label=shock_variable)
        ax2.set_ylabel(shock_variable, color='gray')
        plt.show()

    @staticmethod
    def f_test(model1, model2):
        # f_value = (model1.ssr - model2.ssr) / (model2.df_model - model1.df_model) / (model2.ssr / model2.df_resid)
        # p_value = 1 - stats.f.cdf(f_value, model2.df_model - model1.df_model, model2.df_resid)
        # return f_value, p_value

        ftest = model1.compare_f_test(model2)
        print(ftest)

    @staticmethod
    def countries_with_missing_data(df, country_id):
        return len(df.query(f'country_id == {country_id}'))

    @staticmethod
    def add_column_from_multiindex(df, new_column_name, level_name):
        # Extract the column from the multi-index and assign it as a new column
        df[new_column_name] = df.index.get_level_values(level_name)
        return df

    @staticmethod
    def likelihood_ratio_test(model1, model2, degrees):
        lr_test = model1.compare_lr_test(model2)
        test_statistic = lr_test[0]
        p_value = lr_test[1]

        # Output the test statistic and p-value
        print(f"Likelihood Ratio Test Statistic: {test_statistic}")
        print(f"P-value: {p_value}")
        # Degrees of freedom: Difference in the number of parameters between models
        degrees_of_freedom = degrees  # Adjust as per your actual difference in parameters

        # Calculate the critical value from the chi-square distribution
        # Use a significance level of 0.05
        critical_value = chi2.ppf(0.95, degrees_of_freedom)
        print(f"Critical value: {critical_value}")

        # Compare the test statistic with the critical value
        if test_statistic > critical_value:
            print("Test Statistic > Critical Value: Reject the null hypothesis")
        else:
            print("Test Statistic <= Critical Value: Fail to reject the null hypothesis")
