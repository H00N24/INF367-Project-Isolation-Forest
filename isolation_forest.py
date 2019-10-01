from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Union

import numpy as np
import pandas as pd


def c_norm(n: int) -> float:
    if n > 2:
        return 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n
    elif n == 2:
        return 1.0
    return 0.0


@dataclass
class IsolationTree:
    max_depth: int
    current_depth: int = 0

    __split_feat_index: int = field(init=False, default=-1, repr=None)
    __split_value: np.float = field(init=False, default=np.nan, repr=None)

    __c: float = field(init=False, default=np.nan, repr=None)

    __left_child: Optional[IsolationTree] = field(init=False, default=None, repr=None)
    __right_child: Optional[IsolationTree] = field(init=False, default=None, repr=None)

    def fit(self, X: np.ndarray) -> IsolationTree:

        if self.current_depth < self.max_depth and X.shape[0] > 1:
            self.__split_feat_index = np.random.choice(X.shape[1])

            s_feat = X[:, self.__split_feat_index]

            feat_min = s_feat.min()
            feat_max = s_feat.max()

            self.__split_value = feat_min + np.random.random() * (feat_max - feat_min)

            self.__left_child = IsolationTree(
                max_depth=self.max_depth, current_depth=self.current_depth + 1
            ).fit(X[s_feat < self.__split_value])

            self.__right_child = IsolationTree(
                max_depth=self.max_depth, current_depth=self.current_depth + 1
            ).fit(X[s_feat >= self.__split_value])
        else:
            self.__c = c_norm(X.shape[0])

        return self

    def path_length(self, X: np.ndarray) -> np.ndarray:
        res = np.zeros(X.shape[0])

        if self.__split_feat_index == -1:
            return np.full(X.shape[0], self.current_depth) + self.__c
        else:
            s_feat = X[:, self.__split_feat_index]

            left_mask = s_feat < self.__split_value
            right_mask = s_feat >= self.__split_value

            res[left_mask] = self.__left_child.path_length(X[left_mask])
            res[right_mask] = self.__right_child.path_length(X[right_mask])

        return res

@dataclass
class IsolationForest:
    n_estimators: int
    max_samples: int = 256

    estimators_: List[IsolationTree] = field(init=False, repr=None)
    __c: float = field(init=False, repr=None)

    def __post_init__(self):
        max_depth = np.ceil(np.log2(self.max_samples)).astype(int)
        self.estimators_ = [
            IsolationTree(max_depth=max_depth) for _ in range(self.n_estimators)
        ]

        self.__c = c_norm(self.max_samples)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> IsolationForest:

        if isinstance(X, pd.DataFrame):
            X = X.values

        for estimator in self.estimators_:
            estimator.fit(
                X[np.random.choice(X.shape[0], self.max_samples, replace=False)]
            )

        return self

    def __mean_path_length(self, X: np.ndarray) -> np.ndarray:
        return np.mean(
            [estimator.path_length(X) for estimator in self.estimators_], axis=0
        )

    def paper_score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """ Original scoring proposed in IsolationForest paper
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        return np.power(2, -self.__mean_path_length(X) / self.__c)

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """ Negative score, used in Scikit-learn
        """

        return -self.paper_score_samples(X)
