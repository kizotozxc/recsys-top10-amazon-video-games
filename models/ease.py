"""EASE closed-form constrained ridge reconstruction, implemented directly."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, interactions):
        self.history = interactions.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0]+1] += self.regularization
        precision = cho_solve(cho_factor(gram, lower=True, overwrite_a=True,
                                         check_finite=False),
                              np.eye(gram.shape[0]), overwrite_b=True,
                              check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score_users(self, users):
        return np.asarray(self.history[users] @ self.weights)
