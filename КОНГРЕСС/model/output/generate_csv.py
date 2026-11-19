# -*- coding: utf-8 -*-
from pathlib import Path
import sys
base = Path(__file__).parent.parent
sys.path.insert(0, str(base / "src"))
sys.path.insert(0, str(base / "gate"))
from load_data import load_sites, load_reactors
from gate_rules import HardGate
from econ_model import calc_npvr, _site_tariff
import csv

sites = load_sites(); reactors = load_reactors()
gate = HardGate()
rows = []
for s in sites:
    for r in reactors:
        for n in range(1, 4):
            gr = gate.evaluate(s, r, n)
            if gr.blocked: continue
            tariff = _site_tariff(s)
            econ = calc_npvr(r, s, n, tariff=tariff)
            rows.append({
                'site': s.name,
                'reactor': r.name,
                'n_modules': n,
                'P_el_total_MW': r.P_el_MW * n,
                'npv_bln': round(econ['npv_bln'], 2),
                'capex_bln': round(econ['capex_total_bln'], 2),
                'lcoe_rub_kwh': round(econ['lcoe_rub_per_kwh'], 2),
                'payback_yr': round(econ['payback_years'], 1),
                'tariff_rub_kwh': tariff,
                'warnings': '; '.join(gr.reasons_passed)
            })

rows.sort(key=lambda x: x['npv_bln'], reverse=True)
out_csv = base / 'output' / 'full_ranking.csv'
with open(out_csv, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print('Wrote', out_csv)
