"""Transparent expert-label collection and aggregation (no generated labels)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {'site', 'expert_id', 'independent_score', 'method', 'source_reference'}


def blank_template(sites, n_experts=3):
    """Create an empty field sheet. Does not populate any synthetic ratings."""
    if not 2 <= n_experts <= 25:
        raise ValueError('n_experts must be 2..25')
    form = pd.DataFrame([{'site': site, 'expert_id': f'expert_{i+1}',
                          'independent_score': '', 'method': '', 'source_reference': ''}
                         for site in sites.site for i in range(n_experts)])
    return form.astype({'independent_score': 'object'}) if not form.empty else form


def aggregate_ratings(path, sites, min_experts=2):
    """Build independent site-level labels from documented per-expert ratings."""
    if min_experts < 2:
        raise ValueError('min_experts must be at least 2')
    frame = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
    if not REQUIRED.issubset(frame.columns):
        raise ValueError('Ratings columns missing: '+', '.join(sorted(REQUIRED-set(frame.columns))))
    if frame[list(REQUIRED)].isin(['']).any().any():
        raise ValueError('Expert rating form contains empty required values')
    if frame.duplicated(subset=['site', 'expert_id']).any():
        raise ValueError('Each (site, expert_id) can only have one recorded rating')
    if not set(frame.site).issubset(set(sites.site)):
        raise ValueError('Ratings contain unknown site')
    frame['independent_score'] = pd.to_numeric(frame['independent_score'], errors='raise')
    if not np.isfinite(frame['independent_score']).all() or not frame['independent_score'].between(0,30).all():
        raise ValueError('Scores outside 0..30')
    grouped = frame.groupby('site', sort=True)['independent_score'].agg(['mean','std','count'])
    eligible = grouped.loc[grouped['count'] >= min_experts]
    if len(eligible) < 8:
        raise ValueError('At least 8 sites with sufficiently independent expert scores needed')
    labels = eligible[['mean']].reset_index().rename(columns={'mean':'independent_expert_label'})
    audit = eligible.reset_index().rename(columns={'mean':'mean_score', 'std':'between_expert_std', 'count':'n_experts'})
    return labels, audit
