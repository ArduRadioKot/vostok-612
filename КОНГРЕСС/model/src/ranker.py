# -*- coding: utf-8 -*-
"""Раннер: полный перебор (site, reactor, n) -> Top-N по NPV."""
from pathlib import Path
import sys
base = Path(__file__).parent.parent
sys.path.insert(0, str(base / 'src'))
sys.path.insert(0, str(base / 'gate'))
from load_data import load_sites, load_reactors
from gate_rules import HardGate
from econ_model import calc_npvr, _site_tariff

def top_n(n_top=10, max_modules=4):
    sites = load_sites(); reactors = load_reactors()
    results, total = [], 0
    gate = HardGate()
    for s in sites:
        for r in reactors:
            for n in range(1, max_modules + 1):
                total += 1
                gr = gate.evaluate(s, r, n)
                if gr.blocked: continue
                econ = calc_npvr(r, s, n)
                results.append({
                    'site': s.name, 'reactor': r.name, 'n_modules': n,
                    'npv_bln': round(econ['npv_bln'], 2),
                    'capex_bln': round(econ['capex_total_bln'], 2),
                    'lcoe_rub_kwh': round(econ['lcoe_rub_per_kwh'], 2),
                    'payback_yr': round(econ['payback_years'], 1),
                    'blocked_reasons': gr.reasons_blocked,
                    'warnings': gr.reasons_passed,
                })
    results.sort(key=lambda x: x['npv_bln'], reverse=True)
    return results[:n_top], total

if __name__ == '__main__':
    top, total = top_n(15, max_modules=3)
    print(f"\n=== TOP-{len(top)} from {total} valid combos ===\n")
    for i, row in enumerate(top, 1):
        warn = [w for w in row['warnings'] if w != 'OK']
        wstr = " | WARN: " + ", ".join(warn[:2]) if warn else ""
        print(f"{i:2}. {row['site']:<22} {row['reactor']:<16} n={row['n_modules']}  "
              f"NPV={row['npv_bln']:>+9.1f}B rub  LCOE={row['lcoe_rub_kwh']:.1f} RUB/kWh  "
              f"Payback={row['payback_yr']:.1f}y{wstr}")
