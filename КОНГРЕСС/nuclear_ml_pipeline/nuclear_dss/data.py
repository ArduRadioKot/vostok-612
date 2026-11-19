"""Transparent source data and illustrative scenario input checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
import numpy as np

FEATURES = ('deficit', 'oyat', 'logistics', 'industry', 'economy', 'geo_climate')
SITE_REQUIRED = ('site', *FEATURES, 'expert_score')
REACTOR_REQUIRED = ('reactor', 'P_el_MW', 'P_th_MW', 'service_life', 'dims', 'footprint')
SCENARIO_REQUIRED = ('site', 'demand_mw', 'tariff_rub_per_kwh', 'data_status')

@dataclass(frozen=True)
class Diagnostic:
    level: str
    code: str
    message: str


def read_sources(root: Path):
    sites = pd.read_csv(root / 'data' / 'sites.csv', encoding='utf-8-sig')
    reactors = pd.read_csv(root / 'data' / 'reactors.csv', encoding='utf-8-sig')
    for name, frame, required in [('sites.csv', sites, SITE_REQUIRED), ('reactors.csv', reactors, REACTOR_REQUIRED)]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'{name}: missing columns: {sorted(missing)}')
    if sites['site'].isna().any() or sites['site'].duplicated().any():
        raise ValueError('Duplicate or empty site identifiers')
    if reactors['reactor'].isna().any() or reactors['reactor'].duplicated().any():
        raise ValueError('Duplicate or empty reactor identifiers')
    for f in (*FEATURES, 'expert_score'):
        sites[f] = pd.to_numeric(sites[f], errors='raise')
    if not sites[list(FEATURES)].apply(lambda col: col.between(1,5).all()).all():
        raise ValueError('All six expert criteria must be in 1..5')
    reactors['P_el_MW'] = pd.to_numeric(reactors['P_el_MW'], errors='coerce')
    if not np.isfinite(reactors['P_el_MW']).all() or (reactors['P_el_MW']<=0).any():
        raise ValueError('Invalid electrical capacity in reactors.csv')
    return sites, reactors


def num_from_text(value):
    """Only a single numeric field, or missing; never manufacture a measurement."""
    if pd.isna(value) or str(value).strip().casefold() in ('нет','нкт','', 'none'):
        return None
    token = re.search(r'\d+(?:[.,]\d+)?', str(value))
    return float(token.group().replace(',', '.')) if token else None


def load_scenario(path: Path, sites: pd.DataFrame):
    scenario = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
    missing = set(SCENARIO_REQUIRED) - set(scenario.columns)
    if missing:
        raise ValueError(f'Scenario is missing columns: {sorted(missing)}')
    if scenario['site'].duplicated().any() or scenario['site'].eq('').any():
        raise ValueError('Scenario has duplicate/empty site names')
    if set(scenario['site']) != set(sites['site']):
        raise ValueError('Scenario must have exactly one row per site in sites.csv')
    for col in ('demand_mw','tariff_rub_per_kwh'):
        scenario[col] = pd.to_numeric(scenario[col], errors='raise')
        if (~np.isfinite(scenario[col])).any() or (scenario[col] <= 0).any():
            raise ValueError(f'Non-positive/invalid scenario input: {col}')
    if not scenario['data_status'].isin(('illustrative','measured')).all():
        raise ValueError('data_status must be illustrative or measured')
    # A source reference is required for a claim of measured data. This verifies
    # traceability of the claim, not the truth or authority of the referenced data.
    measured=scenario['data_status'].eq('measured')
    if measured.any() and ('source_reference' not in scenario.columns or
            scenario.loc[measured,'source_reference'].astype(str).str.strip().eq('').any()):
        raise ValueError('Measured scenario rows require a non-empty source_reference')
    return sites.merge(scenario, on='site', how='left', validate='one_to_one')


def diagnose(sites, reactors):
    issues=[]
    weak_label = np.isclose(sites['expert_score'], sites[list(FEATURES)].sum(axis=1))
    if weak_label.all():
        issues.append(Diagnostic('warning','derived_target',
          f'All {len(sites)} expert_score values are exactly the sum of six features: NOT independent ML labels.'))
    for field in ('dims','footprint'):
        missing = reactors[field].fillna('').str.strip().str.casefold().isin(('нет', '', 'нкт'))
        issues.append(Diagnostic('warning','missing_'+field,
          f'{int(missing.sum())} of {len(reactors)} reactors lack a documented {field}.'))
    if 'service_life' in reactors.columns:
        missing = reactors['service_life'].apply(num_from_text).isna().sum()
        issues.append(Diagnostic('warning','missing_lifetime',f'{missing} reactor lifetimes missing/ambiguous.'))
    issues.append(Diagnostic('warning','unknown_demand',
      'The source site table contains ordinal urgency scores, NOT measured electricity demand in MW.'))
    issues.append(Diagnostic('warning','unverified_siting',
      'No licensing, emergency planning, heat-sink or site-specific safety assessment is supplied.'))
    return [dict(level=x.level,code=x.code,message=x.message) for x in issues]
