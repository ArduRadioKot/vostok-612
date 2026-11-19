# -*- coding: utf-8 -*-
"""Полный Pareto/Weighted-счётчик по 4 критериям (Economy, Expert, Risk, Strategy).
Текущий дефолт — Economy 40%, Expert 20%, Risk 20%, Strategy 20%.
Возвращает Top-K + список Парето-фронта.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gate'))
import csv
import numpy as np
from load_data import load_sites, load_reactors
from gate_rules import HardGate
from econ_model import calc_npvr, _site_tariff
from expert_gp import ExpertGPR
from risk_model import calc_risk
from strategy_model import DEFAULT_WEIGHTS, strategy_weighted

def run(reactor_filter='СВБР-100', max_modules=3, n_top=15, weights=None):
    if weights is None: weights = DEFAULT_WEIGHTS
    sites=load_sites(); reactors=load_reactors()
    gate=HardGate()
    gp=ExpertGPR()
    # prepare train data for GPR (all sites, 6 features -> expert_score)
    X_raw=[]; y_raw=[]
    for s in sites:
        X_raw.append([s.deficit_score, s.logistics_score, s.geo_climate_score,
                      getattr(s,'_industry_score',3), getattr(s,'_economy_score',3),
                      s.expert_score])  # actually 6th is dummy
        y_raw.append(s.expert_score)
    X_raw=np.array(X_raw); y_raw=np.array(y_raw)
    if len(X_raw)>=4:
        gp.fit(X_raw, y_raw)
    else:
        gp.X_train=X_raw; gp.y_train=y_raw
    results=[]
    for s in sites:
        for r in reactors:
            if reactor_filter and reactor_filter.lower() not in r.name.lower(): continue
            for n in range(1, max_modules+1):
                gr=gate.evaluate(s,r,n)
                if gr.blocked: continue
                tariff=_site_tariff(s)
                econ=calc_npvr(r,s,n,tariff=tariff)
                mu, sigma = gp.predict_site(np.array([s.deficit_score, s.logistics_score, s.geo_climate_score, 3, 3, s.expert_score]))
                risk_score, risk_drivers = calc_risk(s)
                composite = strategy_weighted(econ['npv_bln'], mu, risk_score, strategic_priority=s.expert_score)
                results.append({
                    'site': s.name, 'reactor': r.name, 'n_modules': n,
                    'npv_bln': round(econ['npv_bln'],2),
                    'lcoe_rub_kwh': round(econ['lcoe_rub_per_kwh'],2),
                    'payback_yr': round(econ['payback_years'],1),
                    'expert_mu': round(mu,2),
                    'risk_score': round(risk_score,1),
                    'composite': round(composite,1),
                    'warnings': '; '.join(gr.reasons_passed)
                })
    results.sort(key=lambda x: x['composite'], reverse=True)
    # simple pareto front: non-dominated by all four metrics
    def dominated(a,b):
        return (a['npv_bln']>=b['npv_bln'] and a['lcoe_rub_kwh']<=b['lcoe_rub_kwh']
                and a['risk_score']<=b['risk_score'] and a['expert_mu']>=b['expert_mu']
                and (a['npv_bln']>b['npv_bln'] or a['lcoe_rub_kwh']<b['lcoe_rub_kwh']))
    pareto=[]
    for i,a in enumerate(results):
        if not any(dominated(b,a) for j,b in enumerate(results) if j!=i):
            pareto.append(a)
    return results[:n_top], pareto[:10]

if __name__=='__main__':
    top,pf=run('СВБР-100', max_modules=3, n_top=12)
    print('=== TOP-12 WEIGHTED ===')
    for i,r in enumerate(top,1):
        print(f"{i:2}. {r['site']:<30} n={r['n_modules']}  comp={r['composite']:>5.1f}  NPV={r['npv_bln']:>+9.1f}B  LCOE={r['lcoe_rub_kwh']:.2f}  Risk={r['risk_score']:.0f}")
