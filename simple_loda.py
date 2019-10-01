from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import scipy


@dataclass
class SimpleLODA:
    """ Simplified LODA

    LODA: Lightweight on-line detector of anomalies
    https://link.springer.com/article/10.1007/s10994-015-5521-0
    """

    k: int
    bins: Union[int, str]
    d: int

    def __post_init__(self):
        self.weights = np.zeros((self.k, self.d))

        non_zero_w = np.rint(self.d * (self.d ** (-1 / 2))).astype(int)

        indexes = np.random.rand(self.k, self.d).argpartition(non_zero_w, axis=1)[
            :, :non_zero_w
        ]

        rand_values = np.random.normal(size=indexes.shape)

        for weight, chosen_d, values in zip(self.weights, indexes, rand_values):
            weight[chosen_d] = values

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> SimpleLODA:

        if isinstance(X, pd.DataFrame):
            X = X.values

        w_X = X @ self.weights.T

        self.hists = [
            scipy.stats.rv_histogram(np.histogram(w_x, bins=self.bins)) for w_x in w_X.T
        ]

        return self

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        if isinstance(X, pd.DataFrame):
            X = X.values

        w_X = X @ self.weights.T

        X_prob = np.array([hist.pdf(w_x) for hist, w_x in zip(self.hists, w_X.T)])
        X_prob[X_prob <= 0] = X_prob[X_prob > 0].min()

        X_scores = -np.mean(np.log(X_prob), axis=0)

        return -X_scores
