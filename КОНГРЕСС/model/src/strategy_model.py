# -*- coding: utf-8 -*-
"""Layer 2d · Strategy Score — веса приоритетов (Economy 40%, Expert 20%, Risk 20%, Strategy 20%).
Текущий дефолт — Economy-first; можно переключить weights=dict(...).
"""
DEFAULT_WEIGHTS = {'economy':0.40, 'expert':0.20, 'risk':0.20, 'strategy':0.20}
def strategy_weighted(econ_score, expert_mu, risk_score, strategic_priority=0.0):
    """Композитная оценка 0..100, higher=better."""
    # Normalize inputs to 0-100
    econ_norm=max(0,min(100, 100 - econ_score/100*50))  # lower NPV -> worse
    exp_norm=max(0,min(100, expert_mu))
    risk_norm=max(0, 100-risk_score)  # invert: lower risk=better
    strat_norm=max(0,min(100,strategic_priority))
    w=DEFAULT_WEIGHTS
    composite = w['economy']*econ_norm + w['expert']*exp_norm + w['risk']*risk_norm + w['strategy']*strat_norm
    return round(composite,1)
