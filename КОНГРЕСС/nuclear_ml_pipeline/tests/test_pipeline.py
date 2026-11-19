from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from nuclear_dss.data import read_sources,diagnose,load_scenario
from nuclear_dss.ranking import screen, pareto_indices, build_ranking, _scale_percentile
from nuclear_dss.economics import cashflow, Assumptions, common_shocks,monte_carlo
from nuclear_dss.expert import load_independent_labels,loo_evaluate,fit_predict

ROOT=Path(__file__).resolve().parents[1]


def data():
    s,r=read_sources(ROOT)
    return s,r,load_scenario(ROOT/'data/scenario_demo.csv',s)


def test_legacy_data_and_derived_label_flag():
    s,r,_=data()
    assert len(s)==21 and len(r)==26
    assert any(z['code']=='derived_target' for z in diagnose(s,r))


def test_scenario_requires_real_positive_demand(tmp_path):
    s,r,sc=data()
    x=sc[['site','demand_mw','tariff_rub_per_kwh','data_status']].copy()
    x.loc[0,'demand_mw']=0
    p=tmp_path/'bad.csv';x.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Non-positive'):load_scenario(p,s)


def test_large_power_is_screened_out():
    s,r,sc=data()
    site=sc.iloc[0]
    large=r[r.reactor=='БН-1200М'].iloc[0]
    reasons,_=screen(site,large,3)
    assert reasons and 'threshold' in reasons[0]


def test_overbuilding_blocked_even_if_smaller_than_350mw():
    s,r,sc=data()
    site=sc[sc['demand_mw']==25].iloc[0]
    medium=r[r.reactor=='СВБР-100'].iloc[0]
    reasons,_=screen(site,medium,1)
    assert any('150%' in z for z in reasons)


def test_electricity_sale_is_demand_capped():
    a=cashflow(100,1,50,15,40)
    b=cashflow(100,1,200,15,40)
    assert b['annual_sold_gwh']>a['annual_sold_gwh']
    assert a['annual_sold_gwh']==pytest.approx(50*0.7*8.76)


def test_high_tariff_increases_npv():
    low=cashflow(100,1,100,10,40)
    high=cashflow(100,1,100,20,40)
    assert high['npv_bln']>low['npv_bln']
    assert high['lcoe_rub_kwh']==pytest.approx(low['lcoe_rub_kwh'])


def test_pareto_correctness_equality_and_tradeoffs():
    def cand(npv,risk,expert,fit):return dict(npv_bln=npv,risk_proxy=risk,expert_prediction=expert,demand_fit=fit)
    rows=[cand(10,10,15,1),cand(5,20,14,.8),cand(11,30,10,1),cand(10,10,15,1)]
    assert set(pareto_indices(rows))=={0,2,3}


def test_npv_scaling_is_increasing():
    scaled=_scale_percentile([-20,0,20])
    assert np.all(np.diff(scaled)>0)


def test_uncertainty_reproducible():
    p=Assumptions()
    a=monte_carlo(100,1,90,15,40,p,common_shocks(32,42))
    b=monte_carlo(100,1,90,15,40,p,common_shocks(32,42))
    assert a==b
    assert a['npv_p05_bln']<=a['npv_p50_bln']<=a['npv_p95_bln']


def test_all_scenarios_unverified_and_default_no_gp_training():
    s,r,sc=data()
    rows,excluded,report=build_ranking(s,r,sc,reactor_filter='СВБР-100',samples=12)
    assert rows and excluded
    assert all(x['screening_status']=='REQUIRES_VERIFICATION' for x in rows)
    assert report['expert_method']=='exact_sum_of_six_rubric_features_not_ML'
    assert len(rows)+len(excluded)==21*3
    assert report['n_on_pareto_front']>=1


def test_independent_labels_reject_derived_target(tmp_path):
    s,_,_=data()
    p=tmp_path/'labels.csv'
    pd.DataFrame({'site':s.site, 'independent_expert_label':s.expert_score}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='derived'):
        load_independent_labels(p,s)


def test_independent_labels_gpr_no_label_leakage(tmp_path):
    s,_,_=data()
    labels=s.expert_score.copy()
    labels.iloc[:10] += np.linspace(0.3,1.2,10)
    p=tmp_path/'labels.csv'
    pd.DataFrame({'site':s.site,'independent_expert_label':labels}).to_csv(p,index=False)
    labeled=load_independent_labels(p,s)
    metrics=loo_evaluate(labeled)
    assert metrics['n']==21 and metrics['gp_loo_mae']>=0
    pred,sd,model=fit_predict(s,labeled)
    assert len(pred)==len(s) and np.isfinite(sd).all()


def test_extended_cross_validation_metrics_and_artifact(tmp_path):
    """Synthetic labels ONLY for software tests, never claimed as expert measurements."""
    from nuclear_dss.expert import (loo_predictions, evaluate_predictions,
                                    train_and_export, load_trained_model, predict_model)
    s, r, sc = data()
    synthetic = s[['site', *('deficit','oyat','logistics','industry','economy','geo_climate')]].copy()
    synthetic['independent_expert_label'] = (s.expert_score +
        np.sin(np.arange(len(s)))*0.8).clip(0,30)
    metrics = evaluate_predictions(loo_predictions(synthetic))
    assert metrics['n'] == 21
    assert metrics['gp_loo_mae'] >= 0
    assert 'ridge_baseline_loo_rmse' in metrics
    assert 0 <= metrics['gp_loo_nominal90_coverage_uncalibrated'] <= 1
    metadata = train_and_export(s,synthetic,tmp_path/'trained',tmp_path/'SYNTHETIC_LABELS_ONLY.csv')
    assert metadata['target'] == 'independent_expert_label'
    model, meta = load_trained_model(tmp_path/'trained'/'expert_gp.joblib')
    pred, std = predict_model(s, model)
    assert len(pred) == len(s) and np.isfinite(std).all()
    rows, excluded, info = build_ranking(s,r,sc,reactor_filter='СВБР-100',samples=10,
                              independent_model_path=tmp_path/'trained'/'expert_gp.joblib')
    assert rows and info['expert_method']=='trained_independent_expert_GPR'


def test_weights_and_economic_reporting_reproducible():
    s,r,sc=data()
    a,_,ra=build_ranking(s,r,sc,reactor_filter='СВБР-100',samples=10,
                            rank_stability_samples=32,seed=9)
    b,_,rb=build_ranking(s,r,sc,reactor_filter='СВБР-100',samples=10,
                            rank_stability_samples=32,seed=9)
    assert ra['economic_decision_status'] in ('NO_POSITIVE_NPV_UNDER_ASSUMPTIONS',
                                             'SOME_POSITIVE_NPV_UNDER_ASSUMPTIONS')
    assert [x['weight_sensitivity_top5_share'] for x in a] == [x['weight_sensitivity_top5_share'] for x in b]
    assert all(0<=x['weight_sensitivity_top5_share']<=1 for x in a)


def test_rejects_concurrent_train_labels_and_model():
    s,r,sc=data()
    with pytest.raises(ValueError,match='either'):
        build_ranking(s,r,sc,independent_labels_path='anything.csv',
                      independent_model_path='something.joblib')


def test_expert_label_collection_and_aggregation(tmp_path):
    from nuclear_dss.labeling import blank_template,aggregate_ratings
    from nuclear_dss.expert import load_independent_labels
    s,_,_=data()
    template=blank_template(s,n_experts=3)
    assert len(template)==63
    with pytest.raises(ValueError,match='empty'):
        form=tmp_path/'empty.csv'
        template.to_csv(form,index=False)
        aggregate_ratings(form,s)
    # Synthetic ratings are strictly for pipeline unit testing; no research labels created.
    for i,idx in enumerate(template.index):
        template.loc[idx,'independent_score']=float((s.expert_score.iloc[i//3]+np.sin(i+1)).clip(0,30))
        template.loc[idx,'method']='SYNTHETIC_TEST_ONLY'
        template.loc[idx,'source_reference']='SYNTHETIC_TEST_ONLY'
    template.to_csv(form,index=False)
    labels,audit=aggregate_ratings(form,s,min_experts=3)
    assert len(labels)==21 and (audit.n_experts==3).all()
    path=tmp_path/'independent_labels.csv';labels.to_csv(path,index=False)
    joined=load_independent_labels(path,s)
    assert len(joined)==21


def test_measured_scenario_requires_reference(tmp_path):
    s,r,sc=data()
    cols=['site','demand_mw','tariff_rub_per_kwh','data_status']
    new=sc[cols].copy()
    new.loc[0,'data_status']='measured'
    p=tmp_path/'scenario.csv'
    new.to_csv(p,index=False)
    with pytest.raises(ValueError,match='source_reference'):
        load_scenario(p,s)
    new['source_reference']=''
    new.loc[0,'source_reference']='SOURCED_TEST_RECORD'
    new.to_csv(p,index=False)
    validated=load_scenario(p,s)
    assert validated.loc[0,'data_status']=='measured'
