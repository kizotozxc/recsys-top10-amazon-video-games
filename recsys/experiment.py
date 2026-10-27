"""Validation search, locked configuration, refit, one final test comparison."""
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from models import ItemKNN, ImplicitALS, EASE, FactorizationMachine, CustomTargetALS
from .features import CausalFeatures, make_fm_examples
from .evaluation import evaluate, paired_bootstrap

SEED = 42
PRIMARY_METRIC = 'NDCG@10'  # fixed BEFORE any validation or test scores
# Small declared search budget, no test-informed changes.
SEARCH_SPACE = {
    'ItemKNN': [dict(neighbours=k, shrinkage=s) for k in [50, 150] for s in [0., 20.]],
    'iALS': [dict(factors=f, regularization=r, alpha=20., iterations=12, seed=SEED)
             for f in [32, 64] for r in [1., 10.]],
    'EASE': [dict(regularization=r) for r in [50., 200., 800.]],
    'FM': [dict(factors=32, learning_rate=lr, regularization=reg, epochs=8,
                batch_size=4096, seed=SEED, use_metadata=True)
           for lr in [.003, .01] for reg in [1e-6, 1e-4]],
    'CustomALS': [dict(factors=f, regularization=r, alpha=20., iterations=12,
                       half_life_days=h, seed=SEED)
                  for f, r in [(32, 1.), (64, 10.)] for h in [180, 730]],
}
CLASSES = {'ItemKNN': ItemKNN, 'iALS': ImplicitALS, 'EASE': EASE,
           'FM': FactorizationMachine, 'CustomALS': CustomTargetALS}


def fit_model(name, params, protocol, fm_cache=None):
    model = CLASSES[name](**params)
    if name in ['ItemKNN', 'EASE']:
        model.fit(protocol.positive)
    elif name == 'FM':
        features, examples = fm_cache if fm_cache else (None, None)
        model.fit(protocol.events, protocol.shape, protocol.cutoff, features, examples)
    else:
        model.fit(protocol.events, protocol.shape, protocol.cutoff)
    return model


def fm_data(protocol):
    return (CausalFeatures(protocol.events, protocol.shape),
            make_fm_examples(protocol.events, protocol.shape, SEED))


def search(protocol, output='results'):
    """Function deliberately has no test argument."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, selected = [], {}
    cache = None
    with threadpool_limits(limits=4):
        for name, configs in SEARCH_SPACE.items():
            if name == 'FM':
                cache = fm_data(protocol)
            scores = []
            for trial, params in enumerate(configs):
                print(f'Validation {name} {trial+1}/{len(configs)}: {params}', flush=True)
                start = perf_counter()
                model = fit_model(name, params, protocol, cache if name == 'FM' else None)
                fit_seconds = perf_counter()-start
                metrics, _, _ = evaluate(model, protocol)
                row = dict(model=name, trial=trial, params=json.dumps(params, sort_keys=True),
                           fit_seconds=fit_seconds, **metrics)
                rows.append(row); scores.append(metrics[PRIMARY_METRIC])
                print(f'  NDCG@10={scores[-1]:.6f}; fit={fit_seconds:.1f}s', flush=True)
                pd.DataFrame(rows).to_csv(output/'validation_trials.csv', index=False)
                del model; gc.collect()
            selected[name] = dict(configs[int(np.argmax(scores))])  # deterministic first tie
    # Control: custom target at baseline's selected ALS hyperparameters.
    # Pick half-life from validation, no further test tuning.
    matched = dict(selected['iALS'])
    matched['half_life_days'] = selected['CustomALS']['half_life_days']
    selected['CustomALS_matched'] = matched
    # Control: SAME FM architecture/optimizer/seeds/sampling, IDs only.
    selected['FM_ID_only'] = dict(selected['FM'], use_metadata=False)
    locked = {'seed': SEED, 'primary_metric': PRIMARY_METRIC, 'configs': selected,
              'validation_protocol': protocol.audit}
    text = json.dumps(locked, indent=2, ensure_ascii=False, sort_keys=True)
    (output/'selected_config.json').write_text(text+'\n')
    (output/'selected_config.sha256').write_text(hashlib.sha256(text.encode()).hexdigest()+'\n')
    return pd.DataFrame(rows), selected


def final_test(protocol, selected, output='results'):
    output = Path(output)
    rows, per_user, recommendations, curves = [], {}, {}, {}
    cache = None
    with threadpool_limits(limits=4):
        for name, params in selected.items():
            method = 'FM' if name == 'FM_ID_only' else ('CustomALS' if name == 'CustomALS_matched' else name)
            if method == 'FM' and cache is None:
                cache = fm_data(protocol)
            print(f'Final refit and test: {name}', flush=True)
            start = perf_counter()
            model = fit_model(method, params, protocol, cache if method == 'FM' else None)
            fit_seconds = perf_counter()-start
            start = perf_counter()
            metrics, by_user, recs = evaluate(model, protocol)
            eval_seconds = perf_counter()-start
            row = dict(model=name, params=json.dumps(params, sort_keys=True),
                       fit_seconds=fit_seconds, evaluation_seconds=eval_seconds, **metrics)
            rows.append(row); per_user[name] = by_user; recommendations[name] = recs
            if hasattr(model, 'loss_history'):
                curves[name] = model.loss_history
            if name == 'CustomALS':
                np.save(output/'custom_target.npy', model.target_values)
            print(f'  NDCG@10={metrics[PRIMARY_METRIC]:.6f}; fit={fit_seconds:.1f}s', flush=True)
            pd.DataFrame(rows).to_csv(output/'test_results.csv', index=False)
            by_user.to_csv(output/f'metrics_{name}.csv', index=True)
            # Keep output compact: full recommendations are cached locally, never committed.
            np.save(output/f'recommendations_{name}.npy', recs)
            del model; gc.collect()
    comparisons = []
    for left, right in [('FM','FM_ID_only'),('CustomALS','iALS'),('CustomALS_matched','iALS')]:
        for metric in ['NDCG@10','Recall@10']:
            comparisons.append(dict(left=left, right=right, metric=metric,
                                    **paired_bootstrap(per_user[left], per_user[right], metric)))
    pd.DataFrame(comparisons).to_csv(output/'paired_comparisons.csv', index=False)
    (output/'learning_curves.json').write_text(json.dumps(curves, indent=2))
    return pd.DataFrame(rows), per_user, recommendations, pd.DataFrame(comparisons), curves
