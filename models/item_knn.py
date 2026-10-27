"""Item cosine neighbours with overlap shrinkage and column-wise top neighbours."""
import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbours=100, shrinkage=20.0):
        self.neighbours = neighbours
        self.shrinkage = shrinkage

    def fit(self, interactions):
        self.history = interactions.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).tocoo()
        norms = np.sqrt(np.asarray(self.history.power(2).sum(axis=0)).ravel())
        valid = gram.row != gram.col
        row, col, overlap = gram.row[valid], gram.col[valid], gram.data[valid]
        similarity = overlap / np.maximum(norms[row] * norms[col], 1e-12)
        similarity *= overlap / (overlap + self.shrinkage)
        columns = sparse.csc_matrix((similarity, (row, col)), shape=gram.shape)
        # Each target item keeps its k strongest incoming neighbours.
        for j in range(columns.shape[1]):
            lo, hi = columns.indptr[j:j+2]
            if hi - lo > self.neighbours:
                indices = np.lexsort((columns.indices[lo:hi], -columns.data[lo:hi]))
                columns.data[lo:hi][indices[self.neighbours:]] = 0
        columns.eliminate_zeros()
        self.weights = columns.tocsr()
        return self

    def score_users(self, users):
        return (self.history[users] @ self.weights).toarray()
