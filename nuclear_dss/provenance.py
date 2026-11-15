"""Source-aware financial profiles and checks for conflicting project documents.

These are source-quoted assumptions, not licensed nuclear siting conclusions.
"""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from .economics import Assumptions

DATA = Path(__file__).resolve().parents[1] / 'data'
BIGSTONE_SITE = 'Большой камень (Приморский край)'
BIGSTONE_REACTOR = 'СВБР-100'


def load_workbook_case(path: Path | None = None) -> dict:
    src = Path(path) if path else DATA / 'economics_bigstone_2026.json'
    d = json.loads(src.read_text(encoding='utf-8'))
    if abs(sum(v['value_bln_rub'] for v in d['capex_items']) - d['capex_total_bln_rub']) > 1e-6:
        raise ValueError('Workbook CAPEX reconciliation failed')
    if abs(sum(v['value_bln_rub_year'] for v in d['opex_items']) - d['gross_opex_bln_rub_year']) > 1e-6:
        raise ValueError('Workbook gross OPEX reconciliation failed')
    if abs((d['gross_opex_bln_rub_year'] - d['noncash_depreciation_bln_rub_year']) - d['cash_opex_bln_rub_year']) > 1e-6:
        raise ValueError('Workbook cash OPEX reconciliation failed')
    return d


def cost_profile(name: str = 'bigstone_2026') -> tuple[Assumptions, dict]:
    if name == 'legacy_2026':
        return Assumptions(anchor_capex_bln_per_100mw=93.5,
                           anchor_opex_bln_per_100mw_year=6.75,
                           max_operation_years=40), {
            'name': name, 'status': 'illustrative_original_model',
            'source': 'research_article(1).md / economics.py (2026-11-11)',
            'applicability': 'Illustrative power-law scenario; not calibrated to actual sites or projects',
            'depreciation_handling': 'Not separately provided in original anchor',
        }
    if name == 'bigstone_2026':
        d = load_workbook_case()
        return Assumptions(anchor_capex_bln_per_100mw=d['capex_total_bln_rub'],
                           anchor_opex_bln_per_100mw_year=d['cash_opex_bln_rub_year'],
                           max_operation_years=60), {
            'name': name, 'status': d['data_status'], 'source': d['source_file'],
            'applicability': d['applicability'],
            'gross_opex_bln_year': d['gross_opex_bln_rub_year'],
            'noncash_depreciation_bln_year': d['noncash_depreciation_bln_rub_year'],
            'cash_opex_bln_year': d['cash_opex_bln_rub_year'],
            'quality_flags': d['quality_flags'],
            'extrapolation_policy': 'All combinations outside the exact 100 MW single-module Большой Камень/SVBR reference are unvalidated power-law extrapolations.',
        }
    raise ValueError(f'Unknown cost profile {name!r}')


def applicability(site: str, reactor: str, modules: int, *, name: str) -> str:
    if name != 'bigstone_2026':
        return 'ILLUSTRATIVE_LEGACY_ANCHOR'
    if str(site).casefold() == BIGSTONE_SITE.casefold() and str(reactor) == BIGSTONE_REACTOR and int(modules) == 1:
        return 'SOURCE_CASE_UNVERIFIED_ESTIMATE'
    return 'UNVALIDATED_TRANSFER_FROM_SINGLE_BIGSTONE_CASE'


def source_consistency(sites, reactors) -> list[dict]:
    def issue(code, message, action):
        return {'severity':'warning', 'code': code, 'finding': message, 'action': action}
    issues = [
        issue('site_count_mismatch',
              'Thesis and Congress narrative report 17 sites, whereas the supplied sites.csv contains 21.',
              'Publish an explicit versioned inclusion/exclusion list; do not assume 17=21.'),
        issue('rubric_mismatch',
              'Thesis reports 9 criteria and 25/27 points; CSV has six ordinal columns 1..5 and 25/30 for Большой Камень.',
              'Keep these methodologies separate until original nine-criterion scorecard is sourced.'),
        issue('capacity_factor_mismatch',
              'Thesis cites 80–85% capacity factor; Congress paragraph 4.2 uses 90%; scenario currently assumes 85%.',
              'Perform scenario sensitivity; state capacity factor explicitly.'),
        issue('service_life_mismatch',
              'Thesis and cost amortization use 60 years; reactor catalogue gives СВБР-100 50 years.',
              'Keep catalogue service life in the base case and 60 years as a transparent what-if only.'),
        issue('demand_not_site_verified',
              'Congress mentions 500–800 MW regional deficit; this is not demonstrated contracted demand for a site.',
              'Obtain local hourly consumption, grid capacity and offtake terms; keep scenario_demo illustrative.'),
        issue('port_name_mismatch',
              'Economics workbook cell A5 says port Korsakov while cell B5 describes Большой Камень.',
              'Resolve location and verify cost item against logistics estimates.'),
        issue('reactor_project_maturity',
              'Technology catalog and concept documents are not independent evidence of installed cost or licensing.',
              'Require vendor references and regulatory engineering review outside this software.'),
    ]
    matching = reactors[reactors['reactor'] == BIGSTONE_REACTOR]
    if len(matching) != 1:
        issues.append(issue('anchor_technology_missing', 'Reference СВБР-100 absent or duplicated in reactor catalog.', 'Check reactor identity.'))
    if len(sites) != 21:
        issues.append(issue('catalogue_changed', f'Expecting 21 original site rows, found {len(sites)}.', 'Review data diff and audit assumptions.'))
    return issues
