import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
from scipy import stats
from viewser.operations import fetch
from viewser import Queryset, Column
import subprocess


class MonthToAnnualRegression:
    def __init__(self, data):
        self.data = data
        self.predictions = {}
        self.models = {}  # Dictionary to store regression models
        self.results = {}  # Dictionary to store regression results


    def ols_regression(self, x_columns, y_column,model_name):
        filtered_rows = self.data.index.get_level_values('month_id') % 12 == 0

        # Filter X and y Series based on the condition
        X = self.data.loc[filtered_rows, x_columns]        
        X = sm.add_constant(X)  # Add a constant term for the intercept
        y = self.data.loc[filtered_rows, y_column]
        model = sm.OLS(y, X).fit()
        self.models[model_name] = model
        self.results[model_name] = model.summary()
        
      

    # def plot_time_series_regression(self, x_columns, y_column, model_name, country_id):
    #     country_data = self.data[(self.data['country_id'] == country_id)]
    #     #country_data = country_data.set_index('month_id').sort_index()
    #     X = country_data[x_columns]
    #     y = country_data[y_column]
    #     model = self.models[model_name]
    #     y_pred = model.predict(sm.add_constant(X))
    #     #plt.plot(y)
    #     #plt.plot(y_pred)
        
    #     df1 = y_pred.reset_index()
    #     df1.columns = ['month_id', 'country_id', 'value']
    #     df2 = y.reset_index()
    #     df2.columns = ['month_id', 'country_id', 'value']
    #     # Plotting
    #     plt.plot(df1['month_id'], df1['value'], marker='o',label='Predicted')
    #     plt.plot(df2['month_id'], df2['value'], marker='o',label='Actual')

    #     plt.xlabel('Month ID')
    #     plt.ylabel('Infant Mortality')
    #     plt.title('Plotting Series with MultiIndex')
    #     plt.legend()

    #     plt.show()
    
    
    
    def plot_time_series_regression(self, x_columns, y_column, model_name, country_id, shock_variable):
        country_data = self.data[(self.data['country_id'] == country_id)]
        # country_data = country_data.set_index('month_id').sort_index()
        x = country_data[x_columns]
        y = country_data[y_column]
        #model = self.models[model_name]
        y_pred = self.predictions[model_name][self.predictions[model_name].index.get_level_values('country_id') == country_id]
        #plt.plot(y)
        #plt.plot(y_pred)
        print(y)
        print(y_pred)
        df1 = y_pred.reset_index()
        df1.columns = ['month_id', 'country_id', 'value']
        df2 = y.reset_index()
        df2.columns = ['month_id', 'country_id', 'value']
        # Plotting
        plt.plot(df1['month_id'], df1['value'], marker='o', label='Predicted')
        plt.plot(df2['month_id'], df2['value'], marker='o', label='Actual')
        plt.plot(df2['month_id'], x[shock_variable], marker='o', label=shock_variable)

        plt.xlabel('Month ID')
        plt.ylabel('Infant Mortality')
        plt.title('Plotting Series with MultiIndex')
        plt.legend()

        plt.show()
        
    def random_effects_regression(self, x_columns, y_column, group_column, model_name):
        X = self.data[x_columns]
        X = sm.add_constant(X)  # Add a constant term for the intercept
        y = self.data[y_column]
        groups = self.data[group_column]

        model = MixedLM(y, X, groups)
        result = model.fit()
        
        self.models[model_name] = result
        self.results[model_name] = result.summary()

    def get_regression_matrix(self, model_name):
        return self.results.get(model_name, None)

    def prediction(self, x_columns, y_column, model_name):
        X = self.data[x_columns]
        y = self.data[y_column]
        model = self.models[model_name]
        y_pred = model.predict(sm.add_constant(X))
        self.predictions[model_name] = y_pred
        
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
        # f_value = (model1.ssr - model2.ssr) / (model2.df_model - model1.df_model) / (model2.ssr / model2.df_resid)
        # p_value = 1 - stats.f.cdf(f_value, model2.df_model - model1.df_model, model2.df_resid)
        # return f_value, p_value
    
        ftest = model1.compare_f_test(model2)
        print(ftest)
    
    @staticmethod  
    def countries_with_missing_data(df,country_id):
        return len(df.query(f'country_id == {country_id}'))

