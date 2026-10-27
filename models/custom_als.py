"""Original rating-calibrated and recency-aware preference target; same ALS solver."""
import numpy as np
from scipy.special import expit
from .ials import ImplicitALS


def calibrated_target(events, cutoff, half_life_days=365, temperature=0.75, floor=0.5):
    """Use only reviews strictly preceding each review to estimate user generosity.

    p_ui = (r_ui-1)/4 * sigmoid((r_ui-mu_u_before)/temperature)
           * [floor + (1-floor)*2**(-(cutoff-t_ui)/half_life)].
    Confidence is exactly the SAME as baseline iALS, including low-rated reviews.
    """
    from recsys.features import EntityTimeline
    timeline = EntityTimeline(events, 'u', int(events.u.max())+1)
    counts, sums, _, _ = timeline.query(events.u.to_numpy(), events.timestamp.to_numpy())
    mean_before = (sums + 5*3.5) / (counts + 5)
    ratings = events.rating.to_numpy(dtype=float)
    age = np.maximum(0, (cutoff - events.timestamp.to_numpy()) / 86400)
    freshness = floor + (1-floor) * np.exp2(-age / half_life_days)
    target = (ratings-1)/4 * expit((ratings-mean_before)/temperature) * freshness
    assert np.all((target >= 0) & (target <= 1))
    return target


class CustomTargetALS(ImplicitALS):
    def __init__(self, half_life_days=365, **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def make_target(self, events, cutoff):
        if cutoff is None:
            raise ValueError('CustomTargetALS requires the recommendation cutoff')
        return calibrated_target(events, cutoff, self.half_life_days)
