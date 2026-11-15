"""Reference-case sensitivity; no screening eligibility or safety claim.

Demonstrates the 100 MW financial workbook on a HYPOTHETICAL locally consumed
100 MW load with 70% annual demand factor. The site-level 100 MW demand is
NOT documented in the supplied primary materials.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from nuclear_dss.economics import cashflow
from nuclear_dss.provenance import cost_profile

ROOT=Path(__file__).resolve().parents[1]


def run(output=ROOT/'experiments'/'reference_case_whatif.csv'):
    rows=[]
    for name in ('legacy_2026','bigstone_2026'):
        p,_=cost_profile(name)
        for years in (50,60):
            for capacity_factor in (.80,.85,.90):
                for tariff in (12.,15.,20.):
                    scenario=type(p)(**{**vars(p),'capacity_factor':capacity_factor})
                    result=cashflow(100,1,100,tariff,years,scenario)
                    rows.append({'case':'100 MW one-module; 100 MW hypothetical peak demand',
                                 'cost_profile':name,'life_input_years':years,
                                 'operating_years_used':result['operating_years'],
                                 'capacity_factor_assumed':capacity_factor,
                                 'tariff_rub_kwh_assumed':tariff,
                                 'demand_use_factor_assumed':p.demand_use_factor,
                                 'npv_bln':round(result['npv_bln'],4),
                                 'lcoe_rub_kwh':round(result['lcoe_rub_kwh'],4),
                                 'annual_sold_gwh':round(result['annual_sold_gwh'],4),
                                 'cost_source_status':'unverified_single_case_or_legacy_illustrative',
                                 'warning':'Demand and tariff assumed; no safety/licensing/economic validation'})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    return rows

if __name__=='__main__':
    result=run()
    print(f'Saved {len(result)} what-if rows: {ROOT / "experiments/reference_case_whatif.csv"}')
