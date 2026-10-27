"""Full-catalog, equal-user macro evaluation shared by EVERY model."""
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse


def utc_timestamp(date):
    return int(pd.Timestamp(date, tz='UTC').timestamp())


@dataclass
class Protocol:
    history: pd.DataFrame
    future: pd.DataFrame
    cutoff: int
    name: str
    min_positive_history: int = 3

    def __post_init__(self):
        assert (self.history.timestamp < self.cutoff).all()
        assert (self.future.timestamp >= self.cutoff).all()
        self.user_ids = np.array(sorted(self.history.user.unique()))
        self.item_ids = np.array(sorted(self.history.item.unique()))
        umap = {u: j for j, u in enumerate(self.user_ids)}
        imap = {i: j for j, i in enumerate(self.item_ids)}
        self.events = self.history.assign(u=self.history.user.map(umap), i=self.history.item.map(imap))
        self.shape = (len(umap), len(imap))
        self.seen = sparse.csr_matrix((np.ones(len(self.events)), (self.events.u, self.events.i)), shape=self.shape)
        positives = self.events[self.events.rating >= 4]
        self.positive = sparse.csr_matrix((np.ones(len(positives)), (positives.u, positives.i)), shape=self.shape)
        warm = self.future.user.isin(umap) & self.future.item.isin(imap)
        mapped = self.future[warm].assign(u=self.future[warm].user.map(umap), i=self.future[warm].item.map(imap))
        history_keys = set(zip(self.events.u, self.events.i))
        unseen = np.array([(u, i) not in history_keys for u, i in zip(mapped.u, mapped.i)])
        relevant = mapped[(mapped.rating >= 4) & unseen]
        positive_counts = np.asarray(self.positive.sum(axis=1)).ravel()
        self.truth = {int(u): set(g.i) for u, g in relevant.groupby('u')
                      if positive_counts[u] >= self.min_positive_history and
                      self.shape[1] - self.seen[u].nnz >= 10}
        self.users = np.array(sorted(self.truth), dtype=np.int64)
        assert len(self.users) > 0
        self.audit = {
            'stage': self.name, 'cutoff': str(pd.to_datetime(self.cutoff, unit='s', utc=True)),
            'history_events': len(self.history), 'future_events': len(self.future),
            'history_users': self.shape[0], 'candidate_items': self.shape[1],
            'cold_user_events': int((~self.future.user.isin(umap)).sum()),
            'cold_item_events': int((~self.future.item.isin(imap)).sum()),
            'warm_relevant_events': len(relevant), 'evaluation_users': len(self.users),
            'evaluated_relevant_events': sum(map(len, self.truth.values())),
            'evaluation_user_sha256': hashlib.sha256('\n'.join(self.user_ids[self.users]).encode()).hexdigest(),
            'candidate_sha256': hashlib.sha256('\n'.join(self.item_ids).encode()).hexdigest(),
        }


def deterministic_topk(scores, k=10):
    """Exact top-k; ties resolve by lexicographically sorted source item ID.

    Partition to find the boundary, then explicitly resolve ties there. This avoids
    arbitrary tie behaviour in argpartition for sparse-neighbour scores.
    """
    result = np.empty((len(scores), k), dtype=np.int64)
    for row, s in enumerate(scores):
        threshold = np.partition(s, len(s)-k)[len(s)-k]
        better = np.flatnonzero(s > threshold)
        ties = np.flatnonzero(s == threshold)[:k-len(better)]
        selected = np.r_[better, ties]
        result[row] = selected[np.lexsort((selected, -s[selected]))]
    return result


def user_metrics(recommendations, truth, k=10):
    discount = 1/np.log2(np.arange(2, k+2))
    hits = np.array([item in truth for item in recommendations[:k]], dtype=float)
    recall = hits.sum()/len(truth)
    ndcg = (hits*discount).sum()/discount[:min(len(truth), k)].sum()
    return float(ndcg), float(recall), float(hits.sum()/k)


def evaluate(model, protocol, k=10, batch_size=128):
    rows, all_recommendations = [], []
    for start in range(0, len(protocol.users), batch_size):
        users = protocol.users[start:start+batch_size]
        scores = np.asarray(model.score_users(users), dtype=np.float64)
        assert scores.shape == (len(users), protocol.shape[1])
        assert np.isfinite(scores).all()
        for row, u in enumerate(users):
            scores[row, protocol.seen[u].indices] = -np.inf
        recommendations = deterministic_topk(scores, k)
        for u, recs in zip(users, recommendations):
            assert len(set(recs)) == k
            assert not set(recs).intersection(protocol.seen[u].indices)
            metrics = user_metrics(recs, protocol.truth[u], k)
            rows.append((int(u), *metrics))
        all_recommendations.append(recommendations)
    per_user = pd.DataFrame(rows, columns=['u','NDCG@10','Recall@10','Precision@10']).set_index('u')
    recs = np.vstack(all_recommendations)
    aggregate = per_user.mean().to_dict()
    aggregate['Coverage@10'] = len(np.unique(recs))/protocol.shape[1]
    return aggregate, per_user, recs


def paired_bootstrap(left, right, metric='NDCG@10', seed=42, repetitions=2000):
    assert left.index.equals(right.index)
    delta = (left[metric] - right[metric]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repetitions)])
    return {'delta': float(delta.mean()), 'ci_low': float(np.quantile(means, .025)),
            'ci_high': float(np.quantile(means, .975))}
