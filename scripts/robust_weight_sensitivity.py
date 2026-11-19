"""Matched-mask sensitivity to priorities in the VOSTOK-612 site model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from robust_site_evaluation import FEATURES, ROOT, TOLS, WEIGHTS, decision_model, mask_scores

ATOL = 1e-10


def maximum_excluding_self(values):
    if values.shape[1] == 1:
        return np.full_like(values, -np.inf)
    first = values.argmax(axis=1)
    largest_two = np.partition(values, -2, axis=1)[:, -2:]
    return np.where(np.arange(values.shape[1]) == first[:, None],
                    largest_two[:, :1], largest_two[:, 1:])


def bounds(lower, upper, groups):
    """Compare site-shared and independent bounds using group maxima."""
    n_profiles = lower.shape[0]
    group_lower = np.full((n_profiles, groups.max() + 1), -np.inf)
    group_upper = group_lower.copy()
    np.maximum.at(group_lower.T, groups, lower.T)
    np.maximum.at(group_upper.T, groups, upper.T)
    alternatives = np.maximum(group_lower[:, groups], maximum_excluding_self(group_upper)[:, groups])
    shared = (alternatives - lower).max(axis=0)
    independent = np.maximum(0, (maximum_excluding_self(upper) - lower).max(axis=0))
    return shared, independent


def weight_profiles():
    yield 'nominal', WEIGHTS
    for component in WEIGHTS:
        for label, factor in [('lower', .5), ('higher', 1.5)]:
            varied = {k: v * (factor if k == component else 1) for k, v in WEIGHTS.items()}
            yield f'{component}_{label}', {k: v / sum(varied.values()) for k, v in varied.items()}


def study(repeats):
    _, configs, site_ids, scores, nominal, coef = decision_model()
    _, groups = np.unique(site_ids, return_inverse=True)
    profiles = list(weight_profiles())
    models = [('nominal', nominal, coef)] + [
        (label, *decision_model(weights)[-2:]) for label, weights in profiles[1:]
    ]
    intercept = np.concatenate([m[1] for m in models])
    coefficients = np.concatenate([np.repeat(m[2][None, :], m[1].shape[0], axis=0) for m in models])
    selections = {label: slice(i * len(nominal), (i + 1) * len(nominal))
                  for i, (label, _, _) in enumerate(models)}
    selections['joint'] = slice(None)
    full = intercept + (scores @ coefficients.T).T[:, site_ids]
    references = {label: (full[sl].max(axis=1, keepdims=True) - full[sl]).max(axis=0)
                  for label, sl in selections.items()}
    records = []
    rng = np.random.default_rng(612)

    def evaluate_mask(masked, label, fraction, repeat):
        lo = intercept + (np.nan_to_num(masked, nan=1) @ coefficients.T).T[:, site_ids]
        hi = intercept + (np.nan_to_num(masked, nan=5) @ coefficients.T).T[:, site_ids]
        for name, selection in selections.items():
            shared, independent = bounds(lo[selection], hi[selection], groups)
            reference = references[name]
            if not ((independent + ATOL >= shared).all() and (shared + ATOL >= reference).all()):
                raise AssertionError('Inconsistent regret bounds')
            for method, regret in [('shared', shared), ('independent', independent)]:
                i = regret.argmin()
                records.append(dict(mask=label, fraction=fraction, repeat=repeat, weights=name,
                                    method=method, choice=int(i), bound=regret[i],
                                    excess=reference[i] - reference.min()))

    for block in (False, True):
        for fraction in (.1, .2, .3):
            for rep in range(repeats):
                masked = mask_scores(scores, rng, round(fraction * scores.size), block)
                evaluate_mask(masked, 'site_block' if block else 'uniform', fraction, rep)

    for i, feature in enumerate(FEATURES):
        masked = scores.copy()
        masked[:, i] = np.nan
        evaluate_mask(masked, f'missing_{feature}', 1 / len(FEATURES), 0)

    trials = pd.DataFrame(records)
    deltas = trials.pivot(index=['mask', 'fraction', 'repeat', 'weights'],
                          columns='method', values='bound').reset_index()
    deltas['bound_reduction'] = deltas.independent - deltas.shared
    deltas['reduced'] = deltas.bound_reduction > ATOL
    for tol in TOLS:
        deltas[f'shared_at_{tol:g}'] = deltas.shared <= tol + ATOL
        deltas[f'independent_at_{tol:g}'] = deltas.independent <= tol + ATOL
        deltas[f'extra_at_{tol:g}'] = deltas[f'shared_at_{tol:g}'] & ~deltas[f'independent_at_{tol:g}']
    summary = deltas.groupby(['mask', 'fraction', 'weights'], as_index=False).mean(numeric_only=True)
    summary = summary.drop(columns='repeat')
    weights = pd.DataFrame([{'profile': label, **w} for label, w in profiles])
    complete = pd.DataFrame([{'weights': label, **configs.iloc[r.argmin()].to_dict(),
                              'minimum_regret_with_complete_scores': r.min()}
                             for label, r in references.items()])
    return trials, summary, weights, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=300)
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments' / 'robust_weights_2026')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    tables = study(args.repeats)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in zip(('trials', 'summary', 'weight_profiles', 'complete_data'), tables):
        table.to_csv(args.out / f'{name}.csv', index=False)
    summary = tables[1]
    selected = summary.weights.isin(['nominal', 'joint']) & summary['mask'].isin(['uniform', 'site_block'])
    columns = ['mask', 'fraction', 'weights', 'independent_at_1', 'shared_at_1', 'bound_reduction']
    print(summary.loc[selected, columns].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
