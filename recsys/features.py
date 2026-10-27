"""Strictly causal user/item metadata, including tied timestamp handling."""
import numpy as np


class EntityTimeline:
    def __init__(self, events, key, size):
        self.times = [np.empty(0, dtype=np.int64) for _ in range(size)]
        self.sums = [np.zeros(1, dtype=np.float64) for _ in range(size)]
        for entity, group in events.groupby(key, sort=False):
            group = group.sort_values('timestamp', kind='stable')
            self.times[entity] = group.timestamp.to_numpy()
            self.sums[entity] = np.r_[0., group.rating.to_numpy().cumsum()]

    def query(self, entities, timestamps):
        entities = np.asarray(entities, dtype=np.int64)
        timestamps = np.broadcast_to(np.asarray(timestamps, dtype=np.int64), entities.shape)
        counts = np.zeros(len(entities), dtype=np.int64)
        sums = np.zeros(len(entities))
        last = np.zeros(len(entities), dtype=np.int64)
        first = np.zeros(len(entities), dtype=np.int64)
        # Group queries once; no use of the current or same-day review in metadata.
        order = np.argsort(entities, kind='stable')
        sorted_entities = entities[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_entities))+1, len(order)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            ix = order[start:stop]
            entity = entities[ix[0]]
            ts = self.times[entity]
            n = np.searchsorted(ts, timestamps[ix], side='left')
            counts[ix] = n
            sums[ix] = self.sums[entity][n]
            if len(ts):
                known = n > 0
                last[ix[known]] = ts[n[known]-1]
                first[ix[known]] = ts[0]
        return counts, sums, last, first


class CausalFeatures:
    # Each categorical field has its OWN vocabulary/embedding parameters.
    COUNT_BINS = np.array([1, 3, 7, 15, 31, 63, 127])
    RATING_BINS = np.array([2.5, 3.0, 3.5, 4.0, 4.5])
    DAYS_BINS = np.array([7, 30, 90, 180, 365, 730, 1460])

    def __init__(self, events, shape):
        self.shape = shape
        self.users = EntityTimeline(events, 'u', shape[0])
        self.items = EntityTimeline(events, 'i', shape[1])
        sizes = [shape[0], 8, 6, 8, shape[1], 8, 6, 8]
        self.offsets = np.r_[0, np.cumsum(sizes)[:-1]]
        self.n_features = sum(sizes)

    def _side(self, entities, timestamps, user):
        timeline = self.users if user else self.items
        entities = np.asarray(entities, dtype=np.int64)
        timestamps = np.broadcast_to(timestamps, entities.shape)
        counts, sums, last, first = timeline.query(entities, timestamps)
        smoothed_mean = (sums + 5*3.5) / (counts+5)
        days = (timestamps - (last if user else first)) / 86400
        days[counts == 0] = 1e6
        values = np.column_stack([
            entities, np.searchsorted(self.COUNT_BINS, counts, side='right'),
            np.searchsorted(self.RATING_BINS, smoothed_mean, side='right'),
            np.searchsorted(self.DAYS_BINS, days, side='right')])
        offsets = self.offsets[:4] if user else self.offsets[4:]
        return values + offsets

    def encode(self, users, items, timestamps, use_metadata=True):
        uf = self._side(users, timestamps, True)
        itf = self._side(items, timestamps, False)
        return np.c_[uf, itf] if use_metadata else np.c_[uf[:, 0], itf[:, 0]]

    def sides_at(self, cutoff, use_metadata=True):
        users = self._side(np.arange(self.shape[0]), cutoff, True)
        items = self._side(np.arange(self.shape[1]), cutoff, False)
        if not use_metadata:
            users, items = users[:, :1], items[:, :1]
        return users, items


def make_fm_examples(events, shape, seed=42, negatives_per_positive=3):
    """Uniform negatives from items already observed at t; exclude only history <= t.

    Future interactions are NOT consulted when rejecting negative samples.
    Positive and low-rated observed reviews use their observed binary relevance.
    """
    rng = np.random.default_rng(seed)
    sorted_events = events.sort_values(['timestamp', 'u', 'i'], kind='stable')
    first_time = events.groupby('i').timestamp.min().reindex(np.arange(shape[1]))
    available_order = np.lexsort((np.arange(shape[1]), first_time.to_numpy()))
    available_times = first_time.to_numpy()[available_order]
    seen = [set() for _ in range(shape[0])]
    users, items, times, labels = [], [], [], []
    for timestamp, same_time in sorted_events.groupby('timestamp', sort=False):
        # Exclude all reviews at this timestamp; no invented within-day ordering.
        for u, i in same_time[['u','i']].itertuples(index=False, name=None):
            seen[u].add(i)
        n_available = np.searchsorted(available_times, timestamp, side='left')
        for u, i, rating in same_time[['u','i','rating']].itertuples(index=False, name=None):
            users.append(u); items.append(i); times.append(timestamp); labels.append(float(rating>=4))
            if rating < 4 or n_available == 0:
                continue
            for _ in range(negatives_per_positive):
                for _ in range(100):
                    negative = int(available_order[rng.integers(n_available)])
                    if negative not in seen[u]:
                        users.append(u); items.append(negative); times.append(timestamp); labels.append(0.)
                        break
    return (np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64),
            np.asarray(times, dtype=np.int64), np.asarray(labels, dtype=np.float32))
