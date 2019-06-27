import numpy as np
import pandas as pd

class OddDetector():
    def __init__(self, df):
        self._df = df
        self._ft_cnt = len(self._df.columns)

    def _iqr_outlier(self, feature):
        df = self._df.loc[:, feature]
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        LB = Q1 - multiplier*(Q3-Q1)
        UB = Q3 + multiplier*(Q3-Q1)
        bl = np.array((((df < LB) | (df > UB))))
        self._df.loc[bl, feature+'_outlier'] = 1
        self._df.loc[~bl, feature+'_outlier'] = 0

    def run(self):
        for f in self._df.columns.values:
            self._iqr_outlier(f)
        self._df['sum_outlier'] = self._df.loc[:,self._df.filter(like='_outlier').columns].sum(axis=1)
        self._df['odd'] = self._df.apply(lambda row: 0 if row['sum_outlier']<self._ft_cnt/2 else 1, axis=1)
        self._df = self._df.drop(self._df.filter(like='_outlier').columns, axis=1)
