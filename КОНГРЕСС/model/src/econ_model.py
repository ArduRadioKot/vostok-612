# -*- coding: utf-8 -*-
"""СЛОЙ 2b · Экономическая модель NPV (калибровка по Сахалину)."""
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'gate'))
from gate_rules import ReactorInfo, SiteInfo, HardGate

# calibrated from Yuzhno-Sakhalinsk case study (full 1-mod CBRR + ZYATC)
# CAPEX ~93.5B RUB for single module (reactor+turbine+port inf+infrastructure)
# OPEX ~6.75B rub/year per module (fuel cycle + staff + amortization + insurance)
ANCHOR_CAPEX_1MOD_RUB = 93.5e9
ANCHOR_OPEX_PER_YR_RUB = 6.75e9
# Realistic selling price for small modular reactors in remote ISOLED grids (rub/kWh)
# Based on Sakhalin diesel avoidance (~50 rub/kWh avoided) but real contracted price is lower
SAKHALIN_EL_TARIFF_RUB_PER_KWH = 35.0
TARIFF_DEFAULT = SAKHALIN_EL_TARIFF_RUB_PER_KWH
# Tariff proxy: remote sites with poor logistics have higher diesel alternatives
_LOGISTICS_TARIFF_PROXY = {
    1: 30.0,  # excellent logistics
    2: 38.0,  # good
    3: 50.0,  # moderate (Sakhalin baseline)
    4: 62.0,  # poor (remote)
    5: 75.0,  # very poor (Arctic/off-grid)
}
def _site_tariff(site, default=TARIFF_DEFAULT):
    return _LOGISTICS_TARIFF_PROXY.get(site.logistics_score, default)
CAPACITY_FACTOR = 0.85

@dataclass
class ProjectParams:
    seismic_factor: float = 1.0
    permafrost_factor: float = 1.0
    port_frozen_factor: float = 1.05

def calc_npvr(reactor: ReactorInfo, site: SiteInfo, n_modules: int,
              params=None, tariff=TARIFF_DEFAULT) -> dict:
    if params is None: params = ProjectParams()
    base_capex = ANCHOR_CAPEX_1MOD_RUB
    scale = n_modules ** 0.8 / max(n_modules, 1)
    capex_total = base_capex * n_modules * scale
    if site.seismic_balls >= HardGate.SEISMIC_HIGH_THRESHOLD:
        seism = 1.6
    else:
        seism = 1.0 + 0.05 * (site.seismic_balls - 6)
    if site.permafrost: perm = 1.25
    else: perm = 1.0
    if site.port_type == 'frozen': port = 1.15
    else: port = 1.0
    site_mult = seism * perm * port
    capex_total *= site_mult
    log_prem = 1.0 + 0.08 * (5 - site.logistics_score)
    opex_annual = ANCHOR_OPEX_PER_YR_RUB * n_modules * log_prem
    # P in MW -> generate MWh/year; convert to kWh for rub/kWh tariff
    annual_gen_kwh = reactor.P_el_MW * n_modules * CAPACITY_FACTOR * 8760 * 1000
    revenue_per_yr = annual_gen_kwh * tariff  # keep consistent: revenue in rub/year from MWh*1e3*kWh_rate
    years = int(reactor.service_life_years) if reactor.service_life_years > 0 else 60
    disc = 0.09
    annuity = (1 - (1 + disc) ** (-years)) / disc
    npv = -capex_total + revenue_per_yr * annuity - opex_annual * annuity
    lcoe = (capex_total / annuity + opex_annual) / annual_gen_kwh
    payback = capex_total / max(revenue_per_yr - opex_annual, 1e-6)
    return {
        'n_modules': n_modules,
        'capex_total_bln': capex_total / 1e9,
        'opex_annual_bln': opex_annual / 1e9,
        'revenue_annual_bln': revenue_per_yr / 1e9,
        'npv_bln': npv / 1e9,
        'lcoe_rub_per_kwh': lcoe,
        'capacity_factor': CAPACITY_FACTOR,
        'payback_years': payback,
        'site_mult': site_mult,
    }
