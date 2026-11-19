"""Scenario cash flow with sold energy capped by stated demand (not a feasibility study)."""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np

@dataclass(frozen=True)
class Assumptions:
    anchor_capex_bln_per_100mw: float = 77.0
    anchor_opex_bln_per_100mw_year: float = 3.9
    power_scale_exponent: float = 0.80
    module_scale_exponent: float = 0.92
    capacity_factor: float = 0.85
    demand_use_factor: float = 0.70
    discount_rate: float = 0.09
    construction_years: int = 5
    max_operation_years: int = 60

    def __post_init__(self):
        if not 0 < self.capacity_factor <= 1 or not 0 < self.demand_use_factor <= 1:
            raise ValueError('Capacity/demand use factors must lie in (0,1]')
        if self.discount_rate < 0 or self.construction_years < 1 or self.max_operation_years < 1:
            raise ValueError('Invalid economic horizon/rate')
        if self.anchor_capex_bln_per_100mw <= 0 or self.anchor_opex_bln_per_100mw_year <= 0:
            raise ValueError('Positive economic anchors required')


def cashflow(power_mw: float, modules: int, demand_mw: float, tariff: float, years: int,
             p: Assumptions = Assumptions(), *, capex_factor: float = 1.,
             opex_factor: float = 1., tariff_factor: float = 1.,
             capacity_factor: float | None = None, demand_factor: float = 1.,
             delay_years: int = 0, capital_cost_bln: float | None = None,
             operating_cost_bln_year: float | None = None):
    if min(power_mw, modules, demand_mw, tariff, years) <= 0:
        raise ValueError('All model inputs must be positive')
    plant_mw = power_mw * modules
    # Power law is strictly a proxy extrapolated from ONE unverified anchor case.
    if (capital_cost_bln is None) != (operating_cost_bln_year is None):
        raise ValueError('Provide both explicit cost inputs or neither')
    if capital_cost_bln is None:
        capex = 1e9*p.anchor_capex_bln_per_100mw*(power_mw/100.)**p.power_scale_exponent*modules**p.module_scale_exponent
        opex = 1e9*p.anchor_opex_bln_per_100mw_year*(power_mw/100.)**p.power_scale_exponent*modules
    else:
        if not (0 < capital_cost_bln < float('inf') and 0 <= operating_cost_bln_year < float('inf')):
            raise ValueError('Invalid explicit cost inputs')
        capex, opex = 1e9 * capital_cost_bln, 1e9 * operating_cost_bln_year
    capex *= capex_factor
    opex *= opex_factor
    annual_kwh = min(plant_mw*(p.capacity_factor if capacity_factor is None else capacity_factor),
                     demand_mw*demand_factor*p.demand_use_factor)*8760*1000
    if annual_kwh <= 0: raise ValueError('Zero energy sold')
    annual_revenue = annual_kwh * tariff * tariff_factor
    op_years = max(1, min(int(years), p.max_operation_years))
    build = p.construction_years + int(delay_years)
    disc = 1. + p.discount_rate
    # Capital spending spread equally across construction years t=0,...,build-1
    pv_capex = capex/build*sum(disc**(-t) for t in range(build))
    operational_discount = sum(disc**(-t) for t in range(build,build+op_years))
    pv_revenue = annual_revenue*operational_discount
    pv_opex = opex*operational_discount
    pv_energy = annual_kwh*operational_discount
    npv_bln = (pv_revenue-pv_opex-pv_capex)/1e9
    lcoe = (pv_capex+pv_opex)/pv_energy
    # NPV=0 break-even tariff is LCOE here (before taxes, financing and any subsidies).
    # This is NOT an observed market tariff or independently validated cost forecast.
    margin = annual_revenue-opex
    simple_payback = build + capex/margin if margin > 0 else None
    return dict(npv_bln=float(npv_bln), lcoe_rub_kwh=float(lcoe),
                capex_bln=capex/1e9, opex_bln_year=opex/1e9,
                revenue_bln_year=annual_revenue/1e9, annual_sold_gwh=annual_kwh/1e6,
                plant_mw=float(plant_mw), simple_payback_years=simple_payback,
                project_years=int(build+op_years), operating_years=int(op_years),
                breakeven_tariff_rub_kwh=float(lcoe))


def common_shocks(n: int, seed=42):
    """Common random numbers for comparable stress tests; priors are assumptions, not frequencies."""
    rng=np.random.default_rng(seed)
    return dict(capex=rng.triangular(.8, 1., 1.4, n),
                opex=rng.triangular(.8, 1., 1.3, n),
                tariff=rng.triangular(.7, 1., 1.3, n),
                demand=rng.triangular(.8, 1., 1.2, n),
                availability=rng.triangular(.80, 1., 1.05, n),
                delay=rng.choice([0,1,2],size=n,p=[.5,.3,.2]))


def monte_carlo(power_mw, modules, demand_mw, tariff, years, p, shocks):
    vals=np.empty(len(shocks['capex']),float)
    levelized=np.empty_like(vals)
    for i in range(len(vals)):
        v=cashflow(power_mw, modules, demand_mw, tariff, years, p,
             capex_factor=float(shocks['capex'][i]),
             opex_factor=float(shocks['opex'][i]),
             tariff_factor=float(shocks['tariff'][i]),
             demand_factor=float(shocks['demand'][i]),
             capacity_factor=min(.99, p.capacity_factor*float(shocks['availability'][i])),
             delay_years=int(shocks['delay'][i]))
        vals[i]=v['npv_bln']
        levelized[i]=v['lcoe_rub_kwh']
    q05=float(np.quantile(vals,.05))
    return dict(npv_mean_bln=float(vals.mean()),npv_p05_bln=q05,
                npv_p50_bln=float(np.median(vals)),npv_p95_bln=float(np.quantile(vals,.95)),
                npv_cvar05_bln=float(vals[vals<=q05].mean()),
                share_positive_in_assumed_scenarios=float(np.mean(vals>0)),
                lcoe_p05_rub_kwh=float(np.quantile(levelized,.05)),
                lcoe_p50_rub_kwh=float(np.median(levelized)),
                lcoe_p95_rub_kwh=float(np.quantile(levelized,.95)),
                scenarios=int(len(vals)))
