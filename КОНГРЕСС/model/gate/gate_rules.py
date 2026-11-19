# -*- coding: utf-8 -*-
"""СЛОЙ 1 · Hard Gate — булевы правила для отсева невозможных комбинаций."""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import re, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class GateResult:
    site: str; reactor: str; n_modules: int
    passed: bool
    reasons_blocked: List[str]
    reasons_passed: List[str]
    @property
    def blocked(self): return not self.passed

@dataclass
class ReactorInfo:
    name: str; country: str; coolant: str; safety: str
    P_el_MW: float; P_th_MW: float
    core_life_years: float; service_life_years: float
    dims: Tuple[float, float]; footprint_m2: float
    max_module_mass_t: float

@dataclass
class SiteInfo:
    name: str
    deficit_score: int; logistics_score: int; geo_climate_score: int
    expert_score: float
    seismic_balls: int = 0
    port_type: str = ''           # none | frozen | year_round
    has_rail: bool = False
    has_road_year: bool = False
    water_for_cooling: str = ''   # sea | river | none
    permafrost: bool = False

def parse_gabaryty(s):
    """Parse '24м/3,5м' or '16,7м/7,9м' into (width_m, height_m). Handles EU decimal comma."""
    if not s or not isinstance(s, str): return None
    # Replace comma with dot (European decimal), then split by / or x or whitespace
    norm = s.replace(',', '.').replace('м', '').replace('M', '')
    # Split on /, -, x, or space separators
    import re as _re
    parts = _re.split(r'[/\s\-]+', norm)
    nums = [float(x) for x in parts if x.strip().replace('.','',1).isdigit() and float(x) > 0]
    if len(nums) >= 2:
        # Usually format is W/H; take first two numbers after filtering zeros
        return (nums[1], nums[0]) if len(nums) == 2 else (min(nums), max(nums))
    return None

def parse_footprint(fp):
    if not fp or not isinstance(fp, str): return None
    m = re.search(r'\d+\.?\d*', fp)
    return float(m.group()) if m else None

class HardGate:
    MAX_RAIL_HEIGHT_M = 4.8
    MAX_MODULE_MASS_T = 500
    SEISMIC_HIGH_THRESHOLD = 8

    def check_capacity(self, s, r, n):
        dm = {1:0, 2:10, 3:20, 4:30, 5:50}.get(s.deficit_score, 0)
        total = r.P_el_MW * n
        if total < dm * 0.8:
            return [f"Дефицит ~{dm:.0f}МВт, {n}x{r.P_el_MW}MW={total:.0f}MW (<80%)"]
        return []

    def check_logistics(self, s, r, n):
        res = []
        mx = max(r.dims) if r.dims else 999
        if not s.port_type:
            res.append("Нет порта доставки тяжеловесного оборудования")
        elif s.port_type == 'frozen':
            res.append("Порт сезонный (ледостав) - завоз только летом")
        # Rail height constraint only applies when rail is the PRIMARY delivery mode
        # For sea-port sites (Yuzhno-Sakhalinsk -> Korsakov ferry), rail gauge not restrictive
        if s.has_rail and r.dims and mx > self.MAX_RAIL_HEIGHT_M:
            if s.port_type in ('none', ''):
                res.append(f"Высота моноблока {mx:.1f}m > rail gauge ({self.MAX_RAIL_HEIGHT_M}m) - rail transport impossible")
            else:
                res.append(f"Высота моноблока {mx:.1f}m > rail gauge; possible modular reassembly at site (sea delivery)")
        if not s.has_rail and not s.port_type:
            res.append("Доставка оборудования невозможна (нет rail/port)")
        return res

    def check_area(self, s, r, n):
        if not r.footprint_m2:
            return ["Footprint unknown"]
        req = r.footprint_m2 * n
        if getattr(s, 'expert_score', 0) < 15:
            return [f"Area uncertain; low score suggests possible shortage"]
        return []

    def check_seismic(self, s):
        if s.seismic_balls >= self.SEISMIC_HIGH_THRESHOLD:
            return [f"Seismic {s.seismic_balls}B - needs reinforcement (CAPEX x1.5-2.0)"]
        return []

    def check_cooling(self, s):
        if not s.water_for_cooling:
            return ["Cooling source unknown"]
        return []

    def check_permafrost(self, s):
        if s.permafrost:
            return ["Permafrost - special foundations (+CAPEX)"]
        return []

    def evaluate(self, s, r, n):
        caps = self.check_capacity(s,r,n)
        logs = self.check_logistics(s,r,n)
        areas = self.check_area(s,r,n)
        seism = self.check_seismic(s)
        cool = self.check_cooling(s)
        perm = self.check_permafrost(s)
        all_reasons = caps + logs + areas + seism + cool + perm
        # Hard block only for capacity deficit or delivery impossible
        hard_keywords = ['невозможна', 'Дефицит', 'Нет порта']
        critical = [r for r in all_reasons if any(kw in r for kw in hard_keywords)]
        # The rest are warnings; if no warnings, keep 'OK'
        warnings = [r for r in all_reasons if r not in critical]
        passed = len(critical) == 0
        return GateResult(site=s.name, reactor=r.name, n_modules=n,
                          passed=passed, reasons_blocked=critical,
                          reasons_passed=warnings if warnings else ['OK'])
