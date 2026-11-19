"""Robust pre-screening under missing site scores and discrete cost profiles."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nuclear_dss.data import FEATURES, load_scenario, read_sources
from nuclear_dss.ranking import WEIGHTS, _scale_percentile, risk_proxy

KEYS = ['site', 'reactor', 'modules']
PROFILES = ['legacy_21x26', 'updated_21x26']
LOGISTICS = FEATURES.index('logistics')
TOLS = [0, .5, 1, 2, 5]


def worst_regret(lower, upper, site_ids, shared=True):
    """Exact max regret for additive site effects and discrete economic profiles."""
    same_site = site_ids[:, None] == site_ids[None, :]
    if not shared:
        same_site = np.eye(len(site_ids), dtype=bool)
    # Rows are choices and columns are challengers. A shared site effect cancels.
    gap = np.where(same_site, lower[:, None, :] - lower[:, :, None],
                   upper[:, None, :] - lower[:, :, None])
    return gap.max(axis=(0, 2))


def decision_model(weights=None):
    custom_weights = weights is not None
    weights = WEIGHTS if weights is None else weights
    sites, _ = read_sources(ROOT)
    scenario = load_scenario(ROOT / 'data/scenario_demo.csv', sites)
    x = sites[list(FEATURES)].to_numpy(float)
    profiles = [pd.read_csv(ROOT / 'experiments' / p / 'ranking.csv').set_index(KEYS).sort_index()
                for p in PROFILES]
    if not all(p.index.equals(profiles[0].index) for p in profiles):
        raise ValueError('Cost profiles must use the same candidate set')
    configs = profiles[0].index.to_frame(index=False)
    site_ids = pd.Index(sites.site).get_indexer(configs.site)
    if (site_ids < 0).any():
        raise ValueError('Unknown site in cost profile')

    def proxy(logistics):
        return np.array([risk_proxy({**s, 'logistics': logistics}) for s in scenario.to_dict('records')])

    risk_at_five = proxy(5)
    if not all(np.allclose(proxy(k), risk_at_five + 5 * (5-k)) for k in range(1, 6)):
        raise ValueError('Risk clipping makes the site effect nonlinear')
    intercept = 100 * np.array([
        weights['npv'] * _scale_percentile(p.npv_bln)
        + weights['risk'] * (1 - (risk_at_five[site_ids] + 25) / 100)
        + weights['demand_fit'] * p.demand_fit.to_numpy()
        for p in profiles
    ])
    coefficients = np.full(len(FEATURES), 100 * weights['expert'] / 30)
    coefficients[LOGISTICS] += 5 * weights['risk']
    if not custom_weights:
        expected = intercept + (x @ coefficients)[site_ids]
        np.testing.assert_allclose(expected, [p.composite_0_100 for p in profiles], atol=.00051, rtol=0)
    return sites, configs, site_ids, x, intercept, coefficients


def evaluate(x, intercept, coefficients, site_ids):
    lower = intercept + (np.nan_to_num(x, nan=1) @ coefficients)[site_ids]
    upper = intercept + (np.nan_to_num(x, nan=5) @ coefficients)[site_ids]
    shared = worst_regret(lower, upper, site_ids)
    independent = worst_regret(lower, upper, site_ids, shared=False)
    return lower, upper, shared, independent


def mask_scores(x, rng, count, block):
    if block:
        # Cluster missingness by site by hiding complete rows plus a partial row.
        order = rng.permutation(x.shape[0])
        cells = np.concatenate([i * x.shape[1] + rng.permutation(x.shape[1]) for i in order])
    else:
        cells = rng.permutation(x.size)
    masked = x.copy()
    masked.flat[cells[:count]] = np.nan
    return masked


def run_experiment(x, intercept, coefficients, site_ids, repeats):
    records = []
    rng = np.random.default_rng(612)
    full = intercept + (x @ coefficients)[site_ids]
    realized = (full.max(axis=1, keepdims=True) - full).max(axis=0)
    full_oracle = realized.min()
    for block in (False, True):
        for level in (.1, .2, .3):
            for rep in range(repeats):
                masked = mask_scores(x, rng, round(x.size * level), block)
                lower, upper, shared, independent = evaluate(masked, intercept, coefficients, site_ids)
                if not ((independent + 1e-10 >= shared).all() and
                        (shared + 1e-10 >= realized).all()):
                    raise AssertionError('Regret bounds must contain the full-data reference')
                choices = {'shared_site': int(shared.argmin()),
                           'independent_intervals': int(independent.argmin())}
                for name, estimator in [('mean', SimpleImputer()),
                                        ('knn', KNNImputer(n_neighbors=3, weights='distance'))]:
                    restored = estimator.fit_transform(masked)
                    score = intercept + (restored @ coefficients)[site_ids]
                    choices[name] = int((score.max(axis=1, keepdims=True) - score).max(axis=0).argmin())
                for name, index in choices.items():
                    bound = independent[index] if name == 'independent_intervals' else shared[index]
                    records.append(dict(mask='site_block' if block else 'uniform', fraction=level,
                                        repeat=rep, method=name, choice=index,
                                        upper_regret=bound, exact_upper_regret=shared[index],
                                        full_regret=realized[index],
                                        excess_full_regret=realized[index] - full_oracle))
    df = pd.DataFrame(records)
    df['oracle_match'] = np.isclose(df.excess_full_regret, 0)
    summary = df.groupby(['mask', 'fraction', 'method']).agg(
        repeats=('repeat', 'size'), oracle_agreement=('oracle_match', 'mean'),
        mean_full_regret=('full_regret', 'mean'), mean_excess=('excess_full_regret', 'mean'),
        mean_upper_bound=('upper_regret', 'mean')).reset_index()
    for tol in TOLS:
        coverage = df.assign(covered=df.upper_regret <= tol + 1e-10).groupby(
            ['mask', 'fraction', 'method']).covered.mean().reset_index(name=f'coverage_at_{tol:g}')
        summary = summary.merge(coverage, on=['mask', 'fraction', 'method'])
    return df, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-scores', type=Path, help='CSV with site and six rubric columns; blank = unknown')
    parser.add_argument('--tolerance', type=float, default=1, help='Maximum regret in points (0–100)')
    parser.add_argument('--repeats', type=int, default=0, help='Trials per mask type and level on the original CSV')
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments' / 'robust_site_2026')
    args = parser.parse_args()
    if args.tolerance < 0 or args.repeats < 0 or (args.site_scores and args.repeats):
        parser.error('Require nonnegative tolerance and repeats, and use at most one experiment mode')
    sites, configs, site_ids, x, intercept, coefficients = decision_model()
    if args.site_scores:
        data = pd.read_csv(args.site_scores).set_index('site')
        if not data.index.is_unique or set(data.index) != set(sites.site):
            parser.error('Site scores must contain one row for each original site')
        x = data.loc[sites.site, list(FEATURES)].to_numpy(float)
        if not (np.isnan(x) | ((x >= 1) & (x <= 5))).all():
            parser.error('Site scores must be blank or between 1 and 5')
    if not np.isfinite(args.tolerance):
        parser.error('Tolerance must be finite')
    lower, upper, shared, independent = evaluate(x, intercept, coefficients, site_ids)
    results = configs.assign(score_lo=lower.min(axis=0), score_hi=upper.max(axis=0),
                             shared_regret=shared, independent_regret=independent,
                             within_tolerance=shared <= args.tolerance + 1e-10)
    results = results.sort_values('shared_regret', kind='stable')
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / 'ranking.csv', index=False)
    best = results.iloc[0]
    status = 'within_tolerance' if best.within_tolerance else 'insufficient_to_recommend'
    print(f'{status}; tolerance={args.tolerance:g}; best={best.site} / {best.reactor} / '
          f'{best.modules}; worst regret={best.shared_regret:.4f} points')
    if args.repeats:
        trials, summary = run_experiment(x, intercept, coefficients, site_ids, args.repeats)
        trials.to_csv(args.out / 'trials.csv', index=False)
        summary.to_csv(args.out / 'summary.csv', index=False)
        print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
