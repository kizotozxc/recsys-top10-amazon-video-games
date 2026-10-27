"""Weighted implicit ALS, solved explicitly with NumPy/SciPy linear algebra."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class ImplicitALS:
    def __init__(self, factors=32, regularization=1.0, alpha=20.0, iterations=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    @staticmethod
    def binary_target(events):
        return (events.rating.to_numpy() >= 4).astype(np.float64)

    def fit(self, events, shape, cutoff=None):
        from scipy import sparse
        # Explicit zero targets MUST remain stored: they still have c=1+alpha.
        target = self.make_target(events, cutoff)
        preferences = sparse.csr_matrix((target, (events.u, events.i)), shape=shape)
        preferences.sort_indices()
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.01, (shape[1], self.factors))
        self.loss_history = []
        transposed = preferences.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._update(preferences, self.item_factors)
            self.item_factors = self._update(transposed, self.user_factors)
            self.loss_history.append(self._objective(preferences))
        self.target_values = target
        return self

    def make_target(self, events, cutoff):
        return self.binary_target(events)

    def _update(self, preferences, fixed):
        gram = fixed.T @ fixed
        base = gram + self.regularization * np.eye(self.factors)
        result = np.zeros((preferences.shape[0], self.factors))
        for row in range(preferences.shape[0]):
            lo, hi = preferences.indptr[row:row+2]
            if lo == hi:
                continue
            vectors = fixed[preferences.indices[lo:hi]]
            lhs = base + self.alpha * (vectors.T @ vectors)
            rhs = (1.0 + self.alpha) * (vectors.T @ preferences.data[lo:hi])
            result[row] = cho_solve(cho_factor(lhs, lower=True, check_finite=False),
                                    rhs, check_finite=False)
        return result

    def _objective(self, preferences):
        # Sum over all unobserved pairs without constructing a dense U x I matrix.
        rows = np.repeat(np.arange(preferences.shape[0]), np.diff(preferences.indptr))
        scores = np.einsum('ij,ij->i', self.user_factors[rows],
                           self.item_factors[preferences.indices])
        total = np.sum((self.user_factors.T @ self.user_factors) *
                       (self.item_factors.T @ self.item_factors))
        total += np.sum((1+self.alpha)*(preferences.data-scores)**2 - scores**2)
        total += self.regularization * (np.sum(self.user_factors**2) +
                                        np.sum(self.item_factors**2))
        return float(total)

    def score_users(self, users):
        return self.user_factors[users] @ self.item_factors.T
