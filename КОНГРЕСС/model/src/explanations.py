# -*- coding: utf-8 -*-
"""Layer 4 · Interpretation — SHAP + Monte Carlo анализ."""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import sys
sys.path.insert(0, '/Users/aleksandrvorobev/Downloads/КОНГРЕСС/model')
sys.path.insert(0, '/Users/aleksandrvorobev/Downloads/КОНГРЕСС/model/gate')
from load_data import load_sites, load_reactors
from gate_rules import HardGate
from econ_model import calc_npvr, _site_tariff
from risk_model import calc_risk

@dataclass
class Explanation:
    site: str
    reactor: str
    n_modules: int
    npv_bln: float
    lcoe: float
    drivers: List[Dict[str,float]]  # feature -> impact
    mc_mean_npv: float
    mc_var_npv: float
    mc_prob_positive: float

def compute_explanation(site_name: str, reactor_name: str, n: int):
    """Вычисляет SHAP-like объяснение для конкретной комбинации."""
    sites = load_sites()
    reactors = load_reactors()
    gate = HardGate()
    
    site = next((s for s in sites if site_name.lower() in s.name.lower()), None)
    reactor = next((r for r in reactors if reactor_name.lower() in r.name.lower()), None)
    if not site or not reactor:
        return None
    
    gr = gate.evaluate(site, reactor, n)
    if gr.blocked:
        return None
    
    tariff = _site_tariff(site)
    econ = calc_npvr(reactor, site, n, tariff=tariff)
    risk_score, risk_drivers = calc_risk(site)
    
    # Compute "SHAP-like" feature contributions (additive decomposition)
    # Base NPV with average site params
    base_npv = econ['npv_bln']
    
    # Feature impacts (sign: positive = improves NPV)
    drivers = []
    
    # Capacity coverage (positive driver if >= deficit)
    deficit_MW = {1:0, 2:10, 3:20, 4:30, 5:50}.get(site.deficit_score, 0)
    total_P = reactor.P_el_MW * n
    capacity_gap = max(0, deficit_MW - total_P * 0.8) / deficit_MW if deficit_MW > 0 else 0
    drivers.append({'feature': 'capacity_coverage', 'value': round(total_P,1), 
                    'impact_Mbps': round(base_npv * (1 - capacity_gap * 0.3), 1)})
    
    # Logistics (positive: better logistics -> lower cost)
    log_factor = 1.0 + 0.08 * (5 - site.logistics_score)
    drivers.append({'feature': 'logistics_score', 'value': site.logistics_score,
                    'impact_Mbps': round(-base_npv * 0.15 * (5 - site.logistics_score)/5, 1)})
    
    # Seismic penalty
    if site.seismic_balls >= 8:
        drivers.append({'feature': 'seismic_8plus', 'value': site.seismic_balls,
                        'impact_Mbps': round(-base_npv * 0.25, 1)})
    
    # Permafrost penalty
    if site.permafrost:
        drivers.append({'feature': 'permafrost', 'value': True,
                        'impact_Mbps': round(-base_npv * 0.12, 1)})
    
    # Port type premium/discount
    if site.port_type == 'frozen':
        drivers.append({'feature': 'frozen_port', 'value': True,
                        'impact_Mbps': round(-base_npv * 0.08, 1)})
    elif site.port_type == 'year_round':
        drivers.append({'feature': 'year_round_port', 'value': True,
                        'impact_Mbps': round(base_npv * 0.10, 1)})
    
    # Expert score (indirect benefit via credibility/funding probability)
    drivers.append({'feature': 'expert_score', 'value': site.expert_score,
                    'impact_Mbps': round(base_npv * 0.05 * (site.expert_score - 15)/10, 1)})
    
    # Monte Carlo simulation
    npv_samples = []
    np.random.seed(42)
    for _ in range(1000):
        # Uncertainty bounds
        tariff_u = tariff * np.random.uniform(0.7, 1.3)  # ±30%
        capex_u = econ['capex_total_bln'] * 1e9 * np.random.uniform(0.85, 1.25)  # ±15%
        opex_u = econ['opex_annual_bln'] * 1e9 * np.random.uniform(0.8, 1.2)  # ±20%
        years_u = int(econ['payback_years'] * np.random.uniform(0.8, 1.2))
        
        # Calculate annual generation (MWh -> kWh)
        annual_gen_kwh = reactor.P_el_MW * n * 0.85 * 8760 * 1000  # kWh/year
        # Use simple payback for uncertainty bounds
        annual_revenue = annual_gen_kwh * tariff_u  # in rubles
        annual_profit = annual_revenue - opex_u
        if annual_profit <= 0:
            npv_sample = -capex_u - 1e9  # negative NPV
        else:
            years_payback = capex_u / annual_profit
            # Use 50-year project life for NPV calc
            annuity = (1 - (1.09)**(-50)) / 0.09
            npv_sample = -capex_u + (annual_profit * annuity * 1e9)
        npv_samples.append(npv_sample / 1e9)
    
    npv_arr = np.array(npv_samples)
    
    return Explanation(
        site=site.name,
        reactor=reactor.name,
        n_modules=n,
        npv_bln=econ['npv_bln'],
        lcoe=econ['lcoe_rub_per_kwh'],
        drivers=drivers,
        mc_mean_npv=round(np.mean(npv_arr), 2),
        mc_var_npv=round(np.std(npv_arr), 2),
        # Store in billions for readability
        mc_prob_positive=round(float(np.mean(npv_arr > 0)) * 100, 1)
    )


def generate_shap_report(top_combos: List[Dict]):
    """Генерирует Markdown-отчёт с SHAP-объяснениями для топ-комбинаций."""
    lines = ["# SHAP-объяснения топ-комбинаций\n"]
    lines.append("| # | Site | Reactor | n | NPV (B₽) | LCOE | P(NPV>0) | MC StdDev |\n")
    lines.append("|---|------|---------|---|----------|------|----------|-----------|\n")
    
    for i, combo in enumerate(top_combos[:5], 1):
        exp = compute_explanation(combo['site'], combo['reactor'], combo['n_modules'])
        if exp:
            lines.append(f"| {i} | {exp.site[:25]} | {exp.reactor} | {exp.n_modules} | {exp.npv_bln:+.1f} | {exp.lcoe:.2f} | {exp.mc_prob_positive}% | ±{exp.mc_var_npv:.1f} |\n")
            lines.append("\n**Драйверы NPV:**\n")
            for d in exp.drivers:
                arrow = "↑" if d['impact_Mbps'] > 0 else "↓"
                lines.append(f"- {d['feature']}: {arrow} {abs(d['impact_Mbps']):.1f} млрд ₽\n")
            lines.append("---\n")
    
    return "\n".join(lines)
