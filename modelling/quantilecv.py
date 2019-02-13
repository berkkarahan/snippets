import numpy as np
from sklearn.model_selection import StratifiedKFold as SKF, RepeatedStratifiedKFold as RSKF


class QuantileCV:
    def __init__(self, n_splits=5, n_repeats=None, shuffle=False, random_state=None):
        self.n_splits=n_splits
        self.shuffle=shuffle
        self.random_state=random_state
        self.n_repeats=n_repeats

        if self.n_repeats is not None:
            self.cvcls = RSKF(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        else:
            self.cvcls = SKF(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

    def _get_quantiles(self, y):
        qs = np.percentile(y, q=[25,50,75])
        yq = np.zeros_like(y, dtype=np.int)
        for i in range(len(qs)):
            if i == len(qs) - 1:
                break
            yq[(y > qs[i]) & (y < qs[i + 1])] = i + 1
        return yq

    def split(self, X, y):
        yq = self._get_quantiles(y)
        return self.cvcls.split(X, yq)
