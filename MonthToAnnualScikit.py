from typing import Optional, Union
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import XGBRFRegressor, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

# from lightgbm import LGBMClassifier, LGBMRegressor


class MotanRegression(BaseEstimator):
    """ 
    Regression model to fit using December data and predict on all monthly data
    Implementeted as a valid sklearn estimator, so it can be used in pipelines and GridSearch objects.
    Args:
        reg_name: currently supports either 'linear'
        reg_params: dict of parameters to pass to regression sub-model when initialized
    """

    def __init__(self,
                 reg_name: str = 'linear',
                 reg_params: Optional[dict] = None):

        self.reg_name = reg_name
        self.reg_params = reg_params
        # self.reg_fi = []

    @staticmethod
    def _resolve_estimator(func_name: str):
        """ Lookup table for supported estimators.
        This is necessary because sklearn estimator default arguments
        must pass equality test, and instantiated sub-estimators are not equal. """

        funcs = {'linear': LinearRegression(),
                 'logistic': LogisticRegression(solver='liblinear'),
                 'LGBMRegressor': LGBMRegressor(n_estimators=250),
                 'LGBMClassifier': LGBMClassifier(n_estimators=250),
                 'RFRegressor': XGBRFRegressor(n_estimators=250, n_jobs=-2),
                 'RFClassifier': XGBRFClassifier(n_estimators=250, n_jobs=-2),
                 'GBMRegressor': GradientBoostingRegressor(n_estimators=200),
                 'GBMClassifier': GradientBoostingClassifier(n_estimators=200),
                 'XGBRegressor': XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-2),
                 'XGBClassifier': XGBClassifier(n_estimators=100, learning_rate=0.05, n_jobs=-2),
                 'HGBRegressor': HistGradientBoostingRegressor(max_iter=200),
                 'HGBClassifier': HistGradientBoostingClassifier(max_iter=200),
                 }

        return funcs[func_name]

    def fit(self, X, y):
        """
        Fits the model based on the provided features X and target y.

        Parameters:
        -----------
        if 
            independent_variable = ['wdi_ny_gdp_mktp_kd', 'wdi_sp_pop_totl', 'ged_sb_tlag1', 'ged_sb_tlag2', 'ged_sb_tlag3', 'ged_sb_tlag4', 'ged_sb_tlag5', 'ged_sb_tlag6', 'ged_sb_tlag7', 'ged_sb_tlag8', 'ged_sb_tlag9', 'ged_sb_tlag10', 'ged_sb_tlag11', 'ged_sb_tlag12', 'ged_sb_tlag13', 'ged_sb_tlag14', 'ged_sb_tlag15', 'ged_sb_tlag16', 'ged_sb_tlag17',
                            'ged_sb_tlag18', 'ged_sb_tlag19', 'ged_sb_tlag20', 'ged_sb_tlag21', 'ged_sb_tlag22', 'ged_sb_tlag23', 'ged_sb_tlag24', 'ged_sb_tlag25', 'ged_sb_tlag26', 'ged_sb_tlag27', 'ged_sb_tlag28', 'ged_sb_tlag29', 'ged_sb_tlag30', 'ged_sb_tlag31', 'ged_sb_tlag32', 'ged_sb_tlag33', 'ged_sb_tlag34', 'ged_sb_tlag35', 'ged_sb_tlag36','country_id','month_id','ged_sb_nolag']
            dependent_variable = 'wdi_sh_dyn_mort_fe'
        then
        X : pandas.DataFrame
            df[independent_variable]
        y : pandas.Series
            df[dependent_variable]

        Returns:
        --------
        self : object
            Returns an instance of the fitted model.
        """
        X_, y_ = check_X_y(X, y, dtype=None,
                           accept_sparse=False,
                           accept_large_sparse=False,
                           force_all_finite='allow-nan')

        if X_.shape[1] < 2:
            raise ValueError('Cannot fit model when n_features = 1')

        # Adding transformation here - fixed effect + December
        X = X[X['month_id'] % 12 == 0]
        dummy_vars = pd.get_dummies(
            X['country_id'], prefix='country', drop_first=True, dtype=int)
        X = pd.concat([X, dummy_vars], axis=1)
        y = y.loc[X.index[X['month_id'] % 12 == 0]]

        #########

        self.reg_ = self._resolve_estimator(self.reg_name)  # regression model
        if self.reg_params:
            self.reg_.set_params(**self.reg_params)  # regression parameters
        self.reg_.fit(X[y > 0], y[y > 0])
        # self.reg_fi = self.reg_.feature_importances_

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predicts target variable based on the input features X.

        Parameters:
        -----------
        X : pandas.DataFrame
            Input features including 'country_id'.

        Returns:
        --------
        predictions : array-like
            Predicted values based on the input features.
        """
        X_ = check_array(X, accept_sparse=False, accept_large_sparse=False)
        # Adding dummies to input data
        dummy_vars = pd.get_dummies(
            X['country_id'], prefix='country', drop_first=True, dtype=int)
        X = pd.concat([X, dummy_vars], axis=1)
        check_is_fitted(self, 'is_fitted_')
        return self.reg_.predict(X)


def manual_test():
    reg = MotanRegression()
    # check_estimator(reg) # uncomment if we need to check requirements for scikit-learn estimator
    from sklearn.datasets import make_regression
    X, y = make_regression()
    reg = MotanRegression()
    reg.fit(X, y)
    print(reg.predict(X))
    print('Done')


if __name__ == '__main__':
    manual_test()
