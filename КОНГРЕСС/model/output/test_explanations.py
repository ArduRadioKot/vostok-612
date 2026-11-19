from pathlib import Path
#!/usr/bin/env python3
"""Test SHAP explanations for top combinations."""
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gate'))

# Ensure model directory is in path
sys.path.insert(0, '/Users/aleksandrvorobev/Downloads/КОНГРЕСС/model')

from src.explanations import compute_explanation, generate_shap_report
from src.pareto_ranker import run

def main():
    print("=" * 60)
    print("SHAP EXPLANATIONS FOR TOP SVBR-100 COMBINATIONS")
    print("=" * 60)
    
    # Run pareto ranking to get top combos
    top, pareto = run(reactor_filter='СВБР-100', max_modules=3, n_top=5)
    
    results = []
    for combo in top[:5]:
        exp = compute_explanation(combo['site'], combo['reactor'], combo.get('n_modules', combo.get('n', 1)))
        if exp:
            results.append({
                'site': exp.site,
                'reactor': exp.reactor, 
                'n': exp.n_modules,
                'npv_bln': exp.npv_bln,
                'lcoe': exp.lcoe,
                'mc_prob_positive': exp.mc_prob_positive,
                'mc_var_npv': exp.mc_var_npv,
                'drivers': exp.drivers
            })
    
    # Print results
    print("\n=== TOP 5 COMBINATIONS WITH UNCERTAINTY ANALYSIS ===\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['site'][:40]} (×{r['n']})")
        print(f"   NPV: {r['npv_bln']:>+8.1f} млрд ₽ | P(NPV>0): {r['mc_prob_positive']}% | σ: ±{r['mc_var_npv']:.1f}")
        print(f"   LCOE: {r['lcoe']:.2f} ₽/кВт·ч")
        print(f"   Key drivers:")
        for d in r['drivers'][:3]:
            arrow = "↑" if d['impact_Mbps'] > 0 else "↓"
            print(f"     - {d['feature']}: {arrow} {abs(d['impact_Mbps']):.1f} млрд ₽")
        print()
    
    # Save full report
    report_md = generate_shap_report(results)
    output_path = Path('/Users/aleksandrvorobev/Downloads/КОНГРЕСС/model/output/SHAP_REPORT.md')
    output_path.write_text(report_md, encoding='utf-8')
    print(f"\nReport saved to: {output_path}")

if __name__ == '__main__':
    main()
