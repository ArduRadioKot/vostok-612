# -*- coding: utf-8 -*-
"""Layer 2c · Risk Score — эвристическая функция риска (сейсмика, мерзлота, удалённость).
Возвращает risk_score 0..100 (выше = хуже) и ключевые risk drivers.
"""
def calc_risk(site):
    score = 0.0
    drivers = []
    # Seismic penalty: 0-8 баллов → 0-40 очков
    seism_bonus = max(0, (site.seismic_balls - 6) * 10) if site.seismic_balls else 0
    score += min(seism_bonus, 40)
    if site.seismic_balls >= 8: drivers.append(f"Seismic {site.seismic_balls}B")
    # Permafrost +30
    if site.permafrost: score+=30; drivers.append("Permafrost foundations")
    # Seasonal port +15
    if site.port_type=='frozen': score+=15; drivers.append("Seasonal port")
    # No rail/port +20
    if not site.has_rail and not site.port_type: score+=20; drivers.append("No rail/port")
    # Low logistics score (bad) + up to 20
    log_penalty = max(0, (5 - site.logistics_score) * 5)
    score += min(log_penalty, 20)
    return min(score, 100.0), drivers
