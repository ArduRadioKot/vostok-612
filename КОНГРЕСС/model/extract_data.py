# -*- coding: utf-8 -*-
"""Слой 0: извлечение sites.csv и reactors.csv из исходных файлов проекта."""
import openpyxl, csv, re, os, docx
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def num(s):
    s = str(s or '')
    # убрать множитель вида " *2 ", " ×3 модуля" и т.п.
    s = re.sub(r'[\u00d7\u00d8\*]\s*\d+\s*', ' ', s, flags=re.I)
    # оставить только цифры и десятичную точку
    s = re.sub(r'[^0-9.]', '', s)
    m = re.search(r'[\d.]+', s)
    try: return float(m.group()) if m else None
    except: return None

def clean(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip()

# ---------- ПЛОЩАДКИ ----------
wb = openpyxl.load_workbook(os.path.join(BASE,'Выбор места.xlsx'), data_only=True)
ws = wb['Лист1']
rows = list(ws.iter_rows(values_only=True))
# найдём заголовок
hdr_idx = None
for i,r in enumerate(rows):
    if r and r[0] and 'Локация' in str(r[0]):
        hdr_idx = i; break
hdr = rows[hdr_idx]
sites = []
for r in rows[hdr_idx+1:]:
    if not r or not r[0]: continue
    name = clean(r[0])
    if not re.match(r'^[А-ЯЁа-яёA-Za-z0-9]', name): continue  # допускаем имена с цифры (4S)
    vals = [num(r[j]) for j in range(1,7)]
    if all(v is None for v in vals): continue
    total = num(r[7])
    if total is None:
        total = round(sum(v for v in vals if v is not None), 0)
    sites.append({
        'site': name, 'deficit': vals[0], 'oyat': vals[1], 'logistics': vals[2],
        'industry': vals[3], 'economy': vals[4], 'geo_climate': vals[5], 'expert_score': total})

with open(os.path.join(BASE,'model/data/sites.csv'),'w',newline='',encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(sites[0].keys()))
    w.writeheader(); w.writerows(sites)
print(f'SITES: {len(sites)} rows')

# ---------- РЕАКТОРЫ ----------
wb = openpyxl.load_workbook(os.path.join(BASE,'Выбор реактора.xlsx'), data_only=True)
ws = wb['Лист1']
rows = list(ws.iter_rows(values_only=True))
hdr_idx = None
for i,r in enumerate(rows):
    if r and r[0] and 'Реактор' in str(r[0]):
        hdr_idx = i; break
hdr = rows[hdr_idx]
reactors = []
for r in rows[hdr_idx+1:]:
    if not r or not r[0]: continue
    name = clean(r[0])
    if not re.match(r'^[А-ЯЁа-яёA-Za-z0-9]', name): continue
    def g(j): return clean(r[j]) if j < len(r) else ''
    try:
        reactors.append({
            'reactor': name, 'country': g(1), 'coolant': g(2), 'safety': g(3),
            'p_thermal': g(4), 'p_electric': g(5), 'core_life': g(6), 'service_life': g(7),
            'dims': g(8), 'footprint': g(9),
            'P_el_MW': num(g(5)), 'P_th_MW': num(g(4))})
    except Exception as e:
        print(f'ERR row {r[0]!r}: {e}')

with open(os.path.join(BASE,'model/data/reactors.csv'),'w',newline='',encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(reactors[0].keys()))
    w.writeheader(); w.writerows(reactors)
print(f'REACTORS: {len(reactors)} rows')
