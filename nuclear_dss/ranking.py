"""Candidate screening, independent normalization and actual Pareto dominance."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from .data import num_from_text
from .economics import Assumptions, cashflow, monte_carlo, common_shocks
from .expert import fit_predict, load_independent_labels, loo_evaluate, load_trained_model, predict_model
from .provenance import applicability, cost_profile, source_consistency

WEIGHTS={'npv':.45,'expert':.25,'risk':.20,'demand_fit':.10}


def screen(site, reactor, n, *, max_reactor_mw=350):
    reasons=[]; warnings=[]
    p=float(reactor['P_el_MW']); demand=float(site['demand_mw'])
    if p>max_reactor_mw:
        reasons.append('Electric reactor power above exploratory small-reactor catalogue threshold')
    if p*n<.8*demand:
        reasons.append('Installed capacity below 80% of assumed demand')
    if p*n>1.5*demand:
        reasons.append('Installed capacity exceeds 150% of assumed demand')
    if str(site.get('data_status','')).lower()!='measured':
        warnings.append('Demand and tariff are illustrative assumptions, NOT verified')
    for key in ('seismic_balls','water_available','port_type'):
        if not str(site.get(key,'')).strip(): warnings.append(f'Site factor {key} not verified')
    if num_from_text(reactor.get('footprint')) is None:
        warnings.append('Reactor footprint unknown; engineering fit not evaluated')
    if num_from_text(reactor.get('dims')) is None:
        warnings.append('Reactor dimensions unknown; transport fit not evaluated')
    if num_from_text(reactor.get('service_life')) is None:
        warnings.append('Service life not stated; illustrative default used')
    warnings.append('No licensing/safety/transport engineering assessment: screening only')
    return reasons,warnings


def risk_proxy(site):
    """Ordinal screening proxy, not probability of accident or license rejection."""
    score=float(max(0,(5-float(site['logistics']))))*5.
    ses=num_from_text(site.get('seismic_balls'))
    if ses is not None and ses>=8: score+=25
    elif ses is None: score+=10 # data incompleteness penalty, not a hazard estimate
    if str(site.get('permafrost','')).strip().lower()=='yes': score+=20
    if str(site.get('port_type','')).strip().lower()=='seasonal':score+=15
    if str(site.get('water_available','')).strip().lower()=='no':score+=20
    elif not str(site.get('water_available','')).strip():score+=10
    return min(100.,score)


def _scale_percentile(values):
    values=np.asarray(values,dtype=float)
    lo,hi=np.quantile(values,[.05,.95]);
    if hi-lo<1e-9: return np.full(len(values),.5)
    return np.clip((values-lo)/(hi-lo),0,1)


def pareto_indices(rows):
    """Maximize NPV, expert, demand fit; minimize risk. Equality != dominance."""
    vals=np.array([[r['npv_bln'],-r['risk_proxy'],r['expert_prediction'],r['demand_fit']] for r in rows],float)
    answer=[]
    for i,a in enumerate(vals):
        dominates_other=(vals>=a).all(axis=1) & (vals>a).any(axis=1)
        dominates_other[i]=False
        if not dominates_other.any():answer.append(i)
    return answer


def build_ranking(sites, reactors, scenario, reactor_filter=None, max_modules=3,
                  samples=250, seed=42, independent_labels_path=None,
                  max_reactor_mw=350., p=Assumptions(), independent_model_path=None,
                  rank_stability_samples=100, weight_concentration=30.0,
                  cost_profile_name="bigstone_2026", operation_life_override=None):
    if operation_life_override is not None and not 1 <= int(operation_life_override) <= 100:
        raise ValueError("Operation life override must be within 1..100 years")
    if cost_profile_name not in ("bigstone_2026", "legacy_2026"):
        raise ValueError("Unknown cost profile")
    if not 1<=max_modules<=12 or not 10<=samples<=20000:
        raise ValueError('max_modules must be 1..12, samples 10..20000')
    if not 10<=max_reactor_mw<=2000:
        raise ValueError('max_reactor_mw must be 10..2000')
    if independent_labels_path is not None and independent_model_path is not None:
        raise ValueError('Use either independent labels OR trained model, not both')
    if not 0 <= rank_stability_samples <= 10000 or weight_concentration <= 0:
        raise ValueError('Invalid weight sensitivity settings')
    labels_report=None
    if independent_model_path:
        fitted,meta=load_trained_model(independent_model_path)
        mu,sd=predict_model(sites,fitted)
        labels_type='trained_independent_expert_GPR'
        labels_report=meta.get('validation')
        expert_predictions=dict(zip(sites.site,(float(z) for z in mu)))
        expert_std=dict(zip(sites.site,(float(z) for z in sd)))
    elif independent_labels_path:
        labels=load_independent_labels(independent_labels_path, sites)
        labels_report=loo_evaluate(labels)
        mu,sd,_=fit_predict(sites,labels)
        labels_type='independent_expert_labels_GPR'
        expert_predictions=dict(zip(sites.site,(float(z) for z in mu)))
        expert_std=dict(zip(sites.site,(float(z) for z in sd)))
    else:
        labels_type='exact_sum_of_six_rubric_features_not_ML'
        expert_predictions=dict(zip(sites.site,sites.expert_score.astype(float)))
        expert_std=dict.fromkeys(sites.site,None)
    shocks=common_shocks(samples,seed)
    rows=[];excluded=[]
    for _,s in scenario.iterrows():
        for _,r in reactors.iterrows():
            if reactor_filter and reactor_filter.casefold() not in str(r.reactor).casefold(): continue
            for n in range(1,max_modules+1):
                rejected,warnings=screen(s,r,n,max_reactor_mw=max_reactor_mw)
                if rejected:
                    excluded.append(dict(site=s['site'],reactor=r['reactor'],modules=n,reason='; '.join(rejected)))
                    continue
                years=num_from_text(r.get('service_life'))
                if years is None: years=p.max_operation_years
                if operation_life_override is not None: years=int(operation_life_override)
                source_applicability = applicability(s['site'], r['reactor'], n, name=cost_profile_name)
                if source_applicability.startswith('UNVALIDATED'):
                    warnings.append('Financial anchor transferred from one Big Stone/SVBR pilot case: unvalidated extrapolation')
                if operation_life_override is not None:
                    warnings.append('Operation lifetime is a user-defined what-if, not the reactor catalog life')
                # Avoid silent positive economic effects from unknown site parameters.
                econ=cashflow(float(r['P_el_MW']),n,float(s['demand_mw']),
                              float(s['tariff_rub_per_kwh']),int(years),p)
                mc=monte_carlo(float(r['P_el_MW']),n,float(s['demand_mw']),
                               float(s['tariff_rub_per_kwh']),int(years),p,shocks)
                delta=abs(float(r['P_el_MW'])*n-float(s['demand_mw']))
                fit=max(0.,1.-delta/(.5*float(s['demand_mw'])))
                rows.append({
                    'site':s['site'],'reactor':r['reactor'],'modules':n,
                    'demand_mw_assumed':round(float(s['demand_mw']),2),
                    'tariff_rub_kwh_assumed':float(s['tariff_rub_per_kwh']),
                    'source_status':s['data_status'],
                    'screening_status':'REQUIRES_VERIFICATION',
                    'cost_profile':cost_profile_name,
                    'cost_anchor_applicability':source_applicability,
                    'operation_years_assumed':int(min(int(years),p.max_operation_years)),
                    'capex_bln':round(econ['capex_bln'],4),
                    'cash_opex_bln_year':round(econ['opex_bln_year'],4),
                    'breakeven_tariff_rub_kwh':round(econ['breakeven_tariff_rub_kwh'],4),
                    'power_mw':econ['plant_mw'],
                    'npv_bln':round(econ['npv_bln'],4),
                    'lcoe_rub_kwh':round(econ['lcoe_rub_kwh'],4),
                    'annual_sold_gwh':round(econ['annual_sold_gwh'],4),
                    'simple_payback_years':None if econ['simple_payback_years'] is None else round(econ['simple_payback_years'],2),
                    'expert_prediction':float(expert_predictions[s['site']]),
                    'expert_std':expert_std[s['site']],
                    'risk_proxy':risk_proxy(s),
                    'demand_fit':round(fit,6),
                    'warnings':' | '.join(warnings),
                    **{k:round(v,4) if isinstance(v,float) else v for k,v in mc.items()},
                    'expert_method':labels_type
                })
    if not rows: return [],excluded,{'n_candidates':0,'n_excluded':len(excluded),'expert_label_validation':labels_report}
    rows_unsorted=list(rows)
    econ_scores=_scale_percentile([r['npv_bln'] for r in rows])
    # Normalize economics across current scenario; compare scores only WITHIN that run.
    for r,e in zip(rows,econ_scores):
        r['composite_0_100']=round(100*(WEIGHTS['npv']*float(e)+
                                     WEIGHTS['expert']*r['expert_prediction']/30.+
                                     WEIGHTS['risk']*(1-r['risk_proxy']/100.)+
                                     WEIGHTS['demand_fit']*r['demand_fit']),3)
    front=set(pareto_indices(rows))
    for i,r in enumerate(rows):r['pareto_front']=(i in front)
    rows.sort(key=lambda r:(r['composite_0_100'],r['npv_bln']),reverse=True)
    for idx,r in enumerate(rows,1):r['rank']=idx
    # Stress-test the weights without pretending the elicited priorities are exact.
    # Dirichlet draws centre around the declared weights and share one RNG seed.
    if rank_stability_samples:
        rng=np.random.default_rng(seed + 1009)
        weight_samples=rng.dirichlet(np.array(list(WEIGHTS.values()))*weight_concentration,
                                     size=rank_stability_samples)
        econ_by_key={(r['site'],r['reactor'],r['modules']):float(e)
                     for r,e in zip(rows_unsorted,econ_scores)}
        components=np.array([
            [econ_by_key[(r['site'],r['reactor'],r['modules'])],
             r['expert_prediction']/30.,1.-r['risk_proxy']/100.,r['demand_fit']]
            for r in rows],float)
        score_matrix=weight_samples@components.T
        rank_matrix=np.empty(score_matrix.shape,dtype=int)
        for i, row_scores in enumerate(score_matrix):
            order=np.argsort(-row_scores,kind='stable')
            rank_matrix[i,order]=np.arange(1,len(rows)+1)
        for j,r in enumerate(rows):
            ranks=rank_matrix[:,j]
            r['weight_sensitivity_top5_share']=round(float(np.mean(ranks<=5)),4)
            r['weight_sensitivity_rank_median']=float(np.median(ranks))
            r['weight_sensitivity_rank_p95']=float(np.quantile(ranks,.95))
    report={ 'n_candidates':len(rows),'n_excluded':len(excluded),
             'n_on_pareto_front':len(front),
             'n_positive_npv':int(sum(r['npv_bln']>0 for r in rows)),
             'n_positive_npv_nonextrapolated':int(sum(r['npv_bln']>0 and
                   not r['cost_anchor_applicability'].startswith('UNVALIDATED') for r in rows)),
             'positive_npv_evidence_warning': ('Every positive NPV comes from unvalidated cross-site/reactor cost scaling; none is supported by a directly applicable cost case.'
                 if any(r['npv_bln']>0 for r in rows) and all(r['cost_anchor_applicability'].startswith('UNVALIDATED') for r in rows if r['npv_bln']>0)
                 else 'Positive NPV, if any, still depends on unverified assumptions.'),
             'best_npv_bln':float(max(r['npv_bln'] for r in rows)),
             'economic_decision_status':('NO_POSITIVE_NPV_UNDER_ASSUMPTIONS' if all(r['npv_bln']<=0 for r in rows) else 'SOME_POSITIVE_NPV_UNDER_ASSUMPTIONS'),
             'weight_sensitivity':{'samples':rank_stability_samples,'seed':seed+1009,'dirichlet_concentration':weight_concentration},
             'assumptions':vars(p),
             'weights':WEIGHTS,
             'expert_method':labels_type,
             'expert_label_validation':labels_report,
             'cost_profile':cost_profile_name,
             'source_cross_checks':source_consistency(sites,reactors),
             'operation_life_override_years':operation_life_override,
             'n_extrapolated_financial_anchors':sum(x['cost_anchor_applicability'].startswith('UNVALIDATED') for x in rows),
             'monte_carlo_note':'Scenario shares reflect chosen triangular priors, not calibrated probabilities.',
             'disclaimer':'Research screening only; no safety, licensed siting, grid adequacy or investment recommendation.'}
    return rows,excluded,report
