"""Updated-document reconciliation and optional modern challenger tests."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from nuclear_dss.provenance import load_workbook_case, cost_profile, applicability, source_consistency
from nuclear_dss.economics import cashflow
from nuclear_dss.data import read_sources, load_scenario
from nuclear_dss.ranking import build_ranking
from nuclear_dss.modern import compare_models, paired_bootstrap_mae_delta

ROOT=Path(__file__).resolve().parents[1]


def test_2026_financial_statement_reconciles_without_double_depreciation():
    data=load_workbook_case()
    assert data['capex_total_bln_rub']==pytest.approx(77.0)
    assert data['gross_opex_bln_rub_year']==pytest.approx(5.18)
    assert data['noncash_depreciation_bln_rub_year']==pytest.approx(1.28)
    assert data['cash_opex_bln_rub_year']==pytest.approx(3.90)
    p,details=cost_profile('bigstone_2026')
    assert p.anchor_opex_bln_per_100mw_year==pytest.approx(3.90)
    assert details['status']=='unverified_project_estimate'


def test_legacy_profile_remains_reproducible_and_different():
    older,_=cost_profile('legacy_2026')
    newer,_=cost_profile('bigstone_2026')
    assert older.anchor_capex_bln_per_100mw==pytest.approx(93.5)
    assert older.anchor_opex_bln_per_100mw_year==pytest.approx(6.75)
    assert newer.max_operation_years==60
    a=cashflow(100,1,100,15,50,older)
    b=cashflow(100,1,100,15,50,newer)
    assert b['npv_bln']>a['npv_bln']
    assert b['breakeven_tariff_rub_kwh']==pytest.approx(b['lcoe_rub_kwh'])
    assert b['operating_years']==50


def test_financial_case_is_not_silently_propagated_to_every_candidate():
    assert applicability('Большой камень (Приморский край)','СВБР-100',1,name='bigstone_2026')=='SOURCE_CASE_UNVERIFIED_ESTIMATE'
    for site,reactor,n in [('Большой камень (Приморский край)','СВБР-100',2),
                            ('Батагай(Якутия)','СВБР-100',1),
                            ('Большой камень (Приморский край)','4S',1)]:
        assert applicability(site,reactor,n,name='bigstone_2026').startswith('UNVALIDATED')


def test_cross_doc_audit_reports_count_and_method_conflicts():
    sites,reactors=read_sources(ROOT)
    codes={x['code'] for x in source_consistency(sites,reactors)}
    assert {'site_count_mismatch','rubric_mismatch','service_life_mismatch','port_name_mismatch','demand_not_site_verified'}<=codes


def test_ranking_marks_unvalidated_cost_transfer_and_override():
    sites,reactors=read_sources(ROOT)
    scenario=load_scenario(ROOT/'data'/'scenario_demo.csv',sites)
    p,_=cost_profile('bigstone_2026')
    result, excluded, stats=build_ranking(sites,reactors,scenario,reactor_filter='СВБР-100',
            samples=10,rank_stability_samples=0,p=p,cost_profile_name='bigstone_2026',
            operation_life_override=60)
    assert result and excluded
    assert stats['operation_life_override_years']==60
    assert all(x['operation_years_assumed']==60 for x in result)
    assert stats['n_extrapolated_financial_anchors']==len(result) # demo Big Stone needs >1 module
    assert all(x['cost_anchor_applicability'].startswith('UNVALIDATED') for x in result)


def test_small_sample_comparisons_are_opt_in_and_not_false_claims():
    sites,_=read_sources(ROOT)
    # SOFTWARE-ONLY SYNTHETIC SIGNAL; NOT AN EXTERNAL EXPERT DATASET.
    fake=sites[['site','deficit','oyat','logistics','industry','economy','geo_climate']].copy()
    fake['independent_expert_label']=np.clip(sites.expert_score + np.sin(np.arange(len(sites)))*.6,0,30)
    predictions,report=compare_models(fake,include_tabpfn=False)
    assert len(predictions)==len(sites)
    assert report['tabpfn_3_5_status']=='NOT_RUN_OPTIONAL'
    assert {'gp','bayesian_ridge','svr_rbf'} <= set(report['metrics'])
    assert all(np.isfinite(x['mae']) for x in report['metrics'].values())
    assert 'external' in report['selection_warning'].lower()


def test_bootstrap_delta_zero_for_identical_predictions():
    data=paired_bootstrap_mae_delta([0,1,2],[0,1,1],[0,1,1],n_resamples=25)
    assert data['observed_delta_mae']==0
    assert data['bootstrap_p025']==0
    assert data['bootstrap_p975']==0
