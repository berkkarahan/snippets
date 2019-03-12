import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import Parallel, delayed, parallel_backend


# Calculating VIF with sklearn is much faster due to parallelized linear-regression.
def _vif_sklearn(exog, exog_idx):
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    x_i = x_i.reshape(-1,1)
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    lr = LinearRegression()
    lr.fit(x_noti, x_i)
    x_i_pred = lr.predict(x_noti)
    r_squared_i = r2_score(x_i, x_i_pred)
    vif = 1. / (1. - r_squared_i)
    return vif

class CollinearityReducer:
    def __init__(self, X, threshold=5.0, lib='statsmodels', vars='float', n_jobs=1):
        self.n_jobs = n_jobs
        self.threshold = threshold
        # Type check lib, sklearn or statsmodels
        if lib == 'sklearn':
            self.viffunc = _vif_sklearn
        elif lib == 'statsmodels':
            self.viffunc = variance_inflation_factor
        else:
            print('Lib is entered wrongly, falling back to default lib: scikit-learn.')
            self.viffunc = _vif_sklearn

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
            dropped = False
            with parallel_backend('threading', n_jobs=self.n_jobs):
                vif = Parallel()(delayed(self.viffunc)(self.X.iloc[:, variables].values, ix) for ix in range(self.X.iloc[:, variables].shape[1]))
            maxloc = vif.index(max(vif))
            if max(vif) > self.threshold:
                print('dropping ' + self.X.iloc[:, variables].columns[maxloc] + ' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True
        print('Remaining variables:')
        print(self.X.columns[variables])
        return self.X.iloc[:, variables]
