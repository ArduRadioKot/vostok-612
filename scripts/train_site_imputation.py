"""Train rubric imputation models and benchmark them on held-out sites."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from robust_site_evaluation import FEATURES, ROOT, decision_model

METHODS = ('mean', 'median', 'knn3', 'ridge', 'bayesian_ridge')


def predict(training, partial, method):
    """Fit on complete historical rows and fill the incoming site's hidden cells."""
    missing = np.isnan(partial)
    observed = ~missing
    restored = partial.copy()
    if not missing.any():
        return restored

    targets = training[:, missing]
    if method == 'median':
        restored[missing] = np.median(targets, axis=0)
    elif method == 'mean' or not observed.any():
        restored[missing] = targets.mean(axis=0)
    else:
        inputs = training[:, observed]
        query = partial[observed][None, :]
        if method == 'knn3':
            distance = np.linalg.norm(inputs - query, axis=1)
            exact = distance == 0
            if exact.any():
                restored[missing] = targets[exact].mean(axis=0)
            else:
                neighbors = np.argsort(distance, kind='stable')[:3]
                restored[missing] = np.average(targets[neighbors], axis=0,
                                               weights=1 / distance[neighbors])
        elif method == 'ridge':
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            restored[missing] = model.fit(inputs, targets).predict(query)[0]
        elif method == 'bayesian_ridge':
            model = make_pipeline(StandardScaler(), BayesianRidge())
            restored[missing] = [model.fit(inputs, target).predict(query)[0] for target in targets.T]
        else:
            raise ValueError(f'Unknown method: {method}')
    return restored.clip(1, 5)


def profile_regret(scores):
    return (scores.max(axis=1, keepdims=True) - scores).max(axis=0)


def benchmark(sites, x, intercept, coefficients, site_ids):
    full_scores = intercept + (x @ coefficients)[site_ids]
    full_regret = profile_regret(full_scores)
    records = []
    for i, name in enumerate(sites.site):
        train = np.delete(x, i, axis=0)
        affected = site_ids == i
        for count in range(1, x.shape[1] + 1):
            for hidden in combinations(range(x.shape[1]), count):
                partial = x[i].copy()
                partial[list(hidden)] = np.nan
                for method in METHODS:
                    result = predict(train, partial, method)
                    errors = result - x[i]
                    delta_score = errors @ coefficients
                    scores = full_scores + delta_score * affected[None, :]
                    choice = profile_regret(scores).argmin()
                    excess = full_regret[choice] - full_regret.min()
                    records.append({
                        'site': name, 'n_hidden': count,
                        'hidden': '+'.join(FEATURES[k] for k in hidden),
                        'method': method, 'mae': np.abs(errors).sum() / count,
                        'mse': (errors ** 2).sum() / count,
                        'site_score_error': abs(delta_score), 'choice': choice,
                        'oracle_match': np.isclose(excess, 0), 'regret': excess,
                    })
    return pd.DataFrame(records)


def summarize(trials, keys):
    result = trials.groupby(keys, as_index=False).agg(
        n=('mae', 'size'), mae=('mae', 'mean'), mse=('mse', 'mean'),
        site_score_mae=('site_score_error', 'mean'),
        oracle_agreement=('oracle_match', 'mean'), regret=('regret', 'mean'))
    result['rmse'] = np.sqrt(result.pop('mse'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments' / 'trained_imputation_2026')
    parser.add_argument('--site', help='Site name to omit from training, or a new name')
    parser.add_argument('--scores', nargs=6, metavar='VALUE',
                        help=f'Values in order {", ".join(FEATURES)}. Use NA for an unknown value.')
    parser.add_argument('--method', choices=METHODS, default='ridge')
    args = parser.parse_args()
    sites, configs, site_ids, x, intercept, coefficients = decision_model()
    if args.scores is not None:
        if not args.site:
            parser.error('--scores requires --site')
        try:
            partial = np.array([np.nan if v.upper() == 'NA' else float(v) for v in args.scores])
        except ValueError:
            parser.error('--scores accepts numbers from 1 through 5 or NA')
        recognized = sites.site.eq(args.site)
        if not all(v.upper() == 'NA' or 1 <= p <= 5 for v, p in zip(args.scores, partial)):
            parser.error('--scores accepts numbers from 1 through 5 or NA')
        filled = predict(x[~recognized], partial, args.method)
        print(pd.Series(filled, index=FEATURES).to_string())
        return

    if args.site:
        parser.error('--site requires --scores')
    trials = benchmark(sites, x, intercept, coefficients, site_ids)
    partial = trials[trials.n_hidden < x.shape[1]]
    by_count = summarize(trials, ['n_hidden', 'method'])
    headline = summarize(partial, ['method']).sort_values('mae')
    site_results = summarize(partial, ['site', 'method'])
    errors_by_site = site_results.pivot(index='site', columns='method', values='mae')
    rng = np.random.default_rng(612)
    samples = rng.integers(len(errors_by_site), size=(10000, len(errors_by_site)))
    for method in METHODS:
        diff = (errors_by_site[method] - errors_by_site['mean']).to_numpy()
        quantiles = np.quantile(diff[samples].mean(axis=1), [.025, .975])
        row = headline.method.eq(method)
        headline.loc[row, 'mae_delta_vs_mean'] = diff.mean()
        headline.loc[row, ['site_bootstrap_lo', 'site_bootstrap_hi']] = quantiles

    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in [('summary', headline), ('by_hidden_count', by_count),
                        ('by_site', site_results), ('trials', trials), ('configuration_keys', configs)]:
        table.to_csv(args.out / f'{name}.csv', index=name == 'configuration_keys')
    print(headline.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
