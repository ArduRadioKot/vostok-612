# -*- coding: utf-8 -*-
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'gate'))
from gate_rules import SiteInfo, ReactorInfo, parse_gabaryty, parse_footprint

SITES_CSV = Path(__file__).parent.parent / 'data' / 'sites.csv'
REACTORS_CSV = Path(__file__).parent.parent / 'data' / 'reactors.csv'

SITE_PROFILES = {
    'южно-сахалинск': {'port':'year_round','rail':True,'water':'sea','seismic':7,'perma':False},
    'петропавловск-камчатский': {'port':'year_round','rail':False,'water':'sea','seismic':8,'perma':False},
    'аннадырь': {'port':'frozen','rail':False,'water':'sea','seismic':8,'perma':True},
    'билибино': {'port':'none','rail':False,'water':'river','seismic':7,'perma':True},
    'певек': {'port':'frozen','rail':False,'water':'sea','seismic':7,'perma':True},
    'махгдан (охотское побережье)': {'port':'year_round','rail':False,'water':'sea','seismic':7,'perma':False},
    'воркута': {'port':'none','rail':True,'water':'river','seismic':6,'perma':False},
    'норильск': {'port':'none','rail':True,'water':'river','seismic':6,'perma':True},
    'большой камень': {'port':'year_round','rail':True,'water':'sea','seismic':6,'perma':False},
    'краснокаменск': {'port':'none','rail':True,'water':'river','seismic':6,'perma':False},

    # Remote but accessible via winter road / barge / heli (per docs: РФЦ «Родина», Чукотка, etc.)
    'уст-кытай': {'port':'none','rail':False,'water':'river','seismic':6,'perma':True},
    'багай': {'port':'none','rail':False,'water':'river','seismic':6,'perma':True},
    'агылкинское': {'port':'none','rail':False,'water':'river','seismic':6,'perma':True},
    'тикси': {'port':'frozen','rail':False,'water':'sea','seismic':7,'perma':True},
    'совиное': {'port':'none','rail':False,'water':'sea','seismic':7,'perma':True},
    'мыс наглёйнын': {'port':'none','rail':False,'water':'sea','seismic':7,'perma':True},
    'павел': {'port':'year_round','rail':False,'water':'sea','seismic':6,'perma':False},
    'большой камень': {'port':'year_round','rail':True,'water':'sea','seismic':6,'perma':False},
    'анадырь': {'port':'frozen','rail':False,'water':'sea','seismic':8,'perma':True},
    'билибино': {'port':'none','rail':False,'water':'river','seismic':7,'perma':True},
    'певек': {'port':'frozen','rail':False,'water':'sea','seismic':7,'perma':True},
    'махгдан': {'port':'year_round','rail':False,'water':'sea','seismic':7,'perma':False},
}

def _norm(s): return s.lower().replace('ё','е').strip() if isinstance(s,str) else ''

def load_sites():
    rows = list(csv.DictReader(open(SITES_CSV, encoding='utf-8-sig')))
    out = []
    for row in rows:
        name = row.get('site','').strip()
        # fuzzy match
        p = None
        for k, v in SITE_PROFILES.items():
            if k in _norm(name) or _norm(name) in k:
                p = v; break
        s = SiteInfo(
            name=name,
            deficit_score=int(float(row.get('deficit','0') or 0)),
            logistics_score=int(float(row.get('logistics','0') or 0)),
            geo_climate_score=int(float(row.get('geo_climate','0') or 0)),
            expert_score=float(row.get('expert_score','0') or 0),
            seismic_balls=p.get('seismic',0) if p else 0,
            port_type=p.get('port','') if p else '',
            has_rail=p.get('rail',False) if p else False,
            has_road_year=p.get('rail',False) if p else False,
            water_for_cooling=p.get('water','') if p else '',
            permafrost=p.get('perma',False) if p else False,
        )
        out.append(s)
    return out

def load_reactors():
    rows = list(csv.DictReader(open(REACTORS_CSV, encoding='utf-8-sig')))
    out = []
    for row in rows:
        dims_str = row.get('dims','')
        fp_str = row.get('footprint','')
        mass_str = row.get('mass_t','')
        def _num(x):
            m = re.search(r'\d+\.?\d*', str(x))
            return float(m.group()) if m else 0.0
        dims = parse_gabaryty(dims_str)
        foot = parse_footprint(fp_str)
        mass = _num(mass_str) if mass_str and 'нет' not in _norm(mass_str) else 0.0
        r = ReactorInfo(
            name=row.get('reactor','').strip(),
            country=row.get('country','').strip(),
            coolant=row.get('coolant','').strip(),
            safety=row.get('safety','').strip(),
            P_el_MW=_num(row.get('P_el_MW','')),
            P_th_MW=_num(row.get('P_th_MW','')),
            core_life_years=_num(row.get('core_life_years','')),
            service_life_years=_num(row.get('service_life_years','')),
            dims=dims or (4.0, 7.5),
            footprint_m2=foot or 34000,
            max_module_mass_t=mass or 150.0,
        )
        out.append(r)
    return out

if __name__ == '__main__':
    ss = load_sites(); rs = load_reactors()
    print(f'Sites: {len(ss)}, Reactors: {len(rs)}')
    for s in ss[:3]: print(' ', s.name, s.port_type, s.has_rail)
    for r in rs[:3]: print(' ', r.name, r.P_el_MW, r.dims)
