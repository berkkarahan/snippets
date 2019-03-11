import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def _vif_sklearn(exog, exog_idx):
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    lr = LinearRegression()
    lr.fit(x_i, x_noti)
    x_noti_pred = lr.predict(x_i)
    r_squared_i = r2_score(x_noti, x_noti_pred)
    vif = 1. / (1. - r_squared_i)
    return vif

class CollinearityReducer:
    def __init__(self, X, threshold=5.0, lib='statsmodels', vars='float'):
        # Type check lib, sklearn or statsmodels
        if lib == 'sklearn':
            self.viffunc = _vif_sklearn
        elif lib == 'statsmodels':
            self.viffunc = variance_inflation_factor
        else:
            print('Lib is entered wrongly, falling back to default lib: statsmodels.')

        # Type check X, pandas.DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X should be a pandas.DataFrame.')
        else:
            self.X = X

        # Type check vars, float or all.
        if vars == 'float':
            self.X = self.X.select_dtypes([np.float])
        elif vars == 'all':
            self.X = self.X.select_dtypes([np.number])
        else:
            print('Vars is entered wrongly, falling back to default vars: float')
            self.X = self.X.select_dtypes([np.float])

    def evaluate(self):
        variables = list(range(self.X.shape[1]))
        dropped = True
        while dropped:
            vif = [self.viffunc(self.X.iloc[:, variables].values, ix)
                    for ix in range(self.X.iloc[:, variables].shape[1])]
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print('dropping ' + X.iloc[:, variables].columns[maxloc] + ' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True
        print('Remaining variables:')
        print(self.X.columns[variables])
        return self.X.iloc[:, variables]
