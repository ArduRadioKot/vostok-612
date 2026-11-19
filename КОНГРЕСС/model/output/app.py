"""
Streamlit Dashboard for SAEC Site Selection System
Layer 1–4: Gate + Econ + Expert GP + Risk + Strategy + MC
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gate'))
os.chdir(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple

# ── Import core modules ───────────────────────────────────────────────
from load_data import load_sites, load_reactors
from gate_rules import HardGate
from econ_model import calc_npvr, _site_tariff
from expert_gp import ExpertGPR
from risk_model import calc_risk
from strategy_model import DEFAULT_WEIGHTS, strategy_weighted

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="САЭС ДВ/Арктика · Выбор площадки", layout="wide")

# ── Cache expensive calls ─────────────────────────────────────────────
@st.cache_resource
def get_cache():
    sites = load_sites()
    reactors = load_reactors()
    gate = HardGate()
    gp = ExpertGPR()
    X_raw = np.array([[s.deficit_score, s.logistics_score, s.geo_climate_score,
                       getattr(s,'_industry_score',3), getattr(s,'_economy_score',3),
                       s.expert_score] for s in sites])
    y_raw = np.array([s.expert_score for s in sites])
    if len(X_raw) >= 4:
        gp.fit(X_raw, y_raw)
    else:
        gp.X_train = X_raw; gp.y_train = y_raw
    return sites, reactors, gate, gp

sites, reactors, gate, gp = get_cache()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Настройки модели")
    st.markdown("---")
    
    # Reactor filter
    reactor_options = ['all'] + [r.name for r in reactors if r.name]
    selected_reactor = st.selectbox("Реактор", reactor_options, index=1)
    max_modules = st.slider("Макс. модулей", 1, 5, 3)
    n_top = st.slider("Показать топ N", 5, 30, 15)
    
    st.markdown("### Веса критериев")
    w_econ   = st.slider("Economy",   0.0, 1.0, DEFAULT_WEIGHTS['economy'], 0.05)
    w_expert = st.slider("Expert",    0.0, 1.0, DEFAULT_WEIGHTS['expert'],   0.05)
    w_risk   = st.slider("Risk",      0.0, 1.0, DEFAULT_WEIGHTS['risk'],     0.05)
    w_strat  = st.slider("Strategy",  0.0, 1.0, DEFAULT_WEIGHTS['strategy'], 0.05)
    total_w = w_econ + w_expert + w_risk + w_strat
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Сумма весов = {total_w:.2f} (должно быть 1.0)")
    weights = {'economy':w_econ,'expert':w_expert,'risk':w_risk,'strategy':w_strat}
    
    st.markdown("### Monte Carlo")
    mc_samples = st.slider("Итераций MC", 100, 5000, 1000, 100)
    tariff_band = st.slider("Тариф ±%", 10, 50, 30, 5)
    capex_band  = st.slider("CAPEX ±%", 5, 30, 15, 5)
    
    st.markdown("---")
    st.caption("Система выбора площадки МСАЭС для ДВ/Арктики\nLayer 1–4: Gate → NPV → ExpertGP → Risk → Strategy")

# ── Helpers ────────────────────────────────────────────────────────────
def run_optimization(weights: Dict, reactor_filter: str, max_mod: int, mc_n: int,
                     tariff_band: float, capex_band: float) -> Tuple[pd.DataFrame, List[Dict]]:
    results = []
    reactor_list = reactors if reactor_filter == 'all' else [r for r in reactors if reactor_filter.lower() in r.name.lower()]
    
    for s in sites:
        for r in reactor_list:
            for n in range(1, max_mod+1):
                gr = gate.evaluate(s, r, n)
                if gr.blocked: continue
                
                tariff = _site_tariff(s)
                econ = calc_npvr(r, s, n, tariff=tariff)
                mu, sigma = gp.predict_site(np.array([s.deficit_score, s.logistics_score,
                                                       s.geo_climate_score, 3, 3, s.expert_score]))
                risk_score, risk_drivers = calc_risk(s)
                composite = strategy_weighted(econ['npv_bln'], mu, risk_score, strategic_priority=s.expert_score)
                
                # MC simulation
                np.random.seed(int(hash((s.name, r.name, n)) % 1e6))
                npv_samples = []
                base_capex = econ['capex_total_bln'] * 1e9
                base_opex  = econ['opex_annual_bln'] * 1e9
                annual_gen_kwh = r.P_el_MW * n * 0.85 * 8760 * 1000
                
                for _ in range(mc_n):
                    tu = tariff * (1 + np.random.uniform(-tariff_band/100, tariff_band/100))
                    cu = base_capex * (1 + np.random.uniform(-capex_band/100, capex_band/100))
                    ou = base_opex * (1 + np.random.uniform(-capex_band/100, capex_band/100))
                    annuity = (1 - 1.09**(-50)) / 0.09
                    annual_rev = annual_gen_kwh * tu
                    annual_profit = annual_rev - ou
                    if annual_profit <= 0:
                        npv_samples.append(-cu / 1e9)
                    else:
                        npv_samples.append((-cu + annual_profit * annuity * 1e9) / 1e9)
                
                results.append({
                    'site': s.name, 'reactor': r.name, 'n': n,
                    'P_MW': r.P_el_MW * n,
                    'npv_bln': round(econ['npv_bln'], 2),
                    'lcoe': round(econ['lcoe_rub_per_kwh'], 2),
                    'payback': round(econ['payback_years'], 1),
                    'expert_mu': round(mu, 1),
                    'risk_score': round(risk_score, 0),
                    'composite': round(composite, 1),
                    'mc_mean': round(float(np.mean(npv_samples)), 2),
                    'mc_std':  round(float(np.std(npv_samples)), 2),
                    'mc_p_gt0':round(float(np.mean(npv_samples > 0))*100, 1),
                    'mc_p_lt0':round(float(np.mean(npv_samples < -50))*100, 1),
                    'warnings': '; '.join(gr.reasons_passed) if gr.reasons_passed else ''
                })
    
    df = pd.DataFrame(results).sort_values('composite', ascending=False).reset_index(drop=True)
    pareto = df.drop_duplicates(subset=['site','reactor','n']).drop_duplicates(subset=['site'])  # simplified
    return df, pareto

# ── Main tab ───────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Главная", "📊 Ранжирование", "🎯 Pareto-фронт", "🧮 SHAP-объяснения", "🗺️ Карта площадок"
])

# ════════════════════════════════════════════════════════════════════════
# TAB 0 — HOME
# ════════════════════════════════════════════════════════════════════════
with tab0:
    st.markdown("""
### 🏭 Система поддержки выбора площадки и типа МСАЭС для ДВ/Арктики России
    
**Архитектура (4 слоя):**
| Слой | Модуль | Описание |
|------|--------|----------|
| Layer 1 | `gate/gate_rules.py` | Жёсткие булевы ограничения (сейсмика, доставка, дефицит) |
| Layer 2a | `src/expert_gp.py` | Gaussian Process экспертных оценок |
| Layer 2c | `src/risk_model.py` | Risk score по факторам (сейсмика, мерзлота, порт) |
| Layer 2d | `src/strategy_model.py` | Композитный скор с настраиваемыми весами |
| Layer 3 | `src/econ_model.py` | NPV-формула (якорь Южно-Сахалинск) |
| Layer 4 | `src/explanations.py` | SHAP-like объяснение + Monte Carlo неопределённостей |
""")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Площадок в базе", len(sites))
        st.metric("Реакторов", len(reactors))
    with col_b:
        st.metric("Прошедших Gate", len(gate.evaluate(sites[0], reactors[0], 1).reasons_passed if sites and reactors else 0))
    
    st.markdown("---")
    st.subheader("🚀 Быстрый старт")
    st.caption("Используйте фильтры слева → вкладка **Ранжирование** для просмотра топ-N комбинаций → вкладка **SHAP-объяснения** для детального разбора драйверов.")

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — RANKING
# ════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner("Выполняю расчёт..."):
        df_full, _ = run_optimization(weights, selected_reactor, max_modules, mc_samples, tariff_band, capex_band)
    
    st.markdown(f"**Всего комбинаций:** {len(df_full)} | **Прошло Gate:** {df_full['npv_bln'].notna().sum()}")
    
    # Filters row
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        min_npv = st.number_input("Мин. NPV (млрд ₽)", value=-500.0, step=100.0)
    with f_col2:
        max_lcoe = st.number_input("Макс. LCOE (₽/кВт·ч)", value=50.0, step=5.0)
    with f_col3:
        min_mc = st.number_input("Мин. P(NPV>0) %", value=0.0, step=5.0)
    with f_col4:
        search_site = st.text_input("Поиск по площадке", "")
    
    filt = df_full[
        (df_full['npv_bln'] >= min_npv) &
        (df_full['lcoe'] <= max_lcoe) &
        (df_full['mc_p_gt0'] >= min_mc) &
        (df_full['site'].str.contains(search_site, case=False, na=False))
    ].head(n_top)
    
    st.dataframe(filt[['site','reactor','n','P_MW','npv_bln','lcoe','payback','expert_mu','risk_score','composite','mc_p_gt0','mc_std']].rename(columns={
        'site':'Площадка','reactor':'Реактор','n':'Модули','P_MW':'Мощность МВт',
        'npv_bln':'NPV млрд₽','lcoe':'LCOE ₽/кВтч','payback':'Payback лет',
        'expert_mu':'Expert μ','risk_score':'Risk 0–100','composite':'Composite',
        'mc_p_gt0':'P(NPV>0)%','mc_std':'σ млрд₽'
    }), use_container_width=True, hide_index=True, column_config={
        "Площадка": st.column_config.TextColumn(width="medium"),
        "NPV млрд₽": st.column_config.NumberColumn(format="%.1f", help="Чистая приведённая стоимость"),
        "Composite": st.column_config.NumberColumn(format="%.1f", help="Композитный скор"),
    })
    
    # Top combo detail
    if not filt.empty:
        top = filt.iloc[0]
        st.subheader(f"🥇 Топ-комбинация: {top['site']} × {top['reactor']} × {int(top['n'])} модулей")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("NPV", f"{top['npv_bln']:+.1f} млрд ₽")
        c2.metric("LCOE", f"{top['lcoe']:.2f} ₽/кВт·ч")
        c3.metric("P(NPV>0)", f"{top['mc_p_gt0']}%")
        c4.metric("Risk Score", f"{top['risk_score']}/100")
        
        # Bar chart of top 15 by composite
        st.plotly_chart(px.bar(filt.head(15).sort_values('composite', ascending=True)[::-1],
                                x='composite', y='site', orientation='h',
                                color='npv_bln', colorbar_title='NPV (B₽)',
                                hover_data=['reactor','n','lcoe','risk_score']), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — PARETO FRONT
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Парето-фронт по NPV vs Risk Score")
    with st.spinner("Рассчитываю фронт..."):
        df_full2, _ = run_optimization(weights, selected_reactor, max_modules, mc_samples, tariff_band, capex_band)
    
    # Non-dominated sorting (simple)
    def non_dominated(df):
        keep = []
        for i,a in df.iterrows():
            dominated = False
            for j,b in df.iterrows():
                if i==j: continue
                if b['npv_bln']>=a['npv_bln'] and b['risk_score']<=a['risk_score'] and (b['npv_bln']>a['npv_bln'] or b['risk_score']<a['risk_score']):
                    dominated=True; break
            if not dominated: keep.append(i)
        return df.loc[keep]
    
    df_pareto = non_dominated(df_full2[df_full2['npv_bln']>0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_full2['risk_score'], y=df_full2['npv_bln'], mode='markers',
                             marker=dict(size=8, opacity=0.4, color='gray'), name='Все точки'))
    fig.add_trace(go.Scatter(x=df_pareto['risk_score'], y=df_pareto['npv_bln'], mode='markers+text',
                             marker=dict(size=12, color='red', symbol='diamond'),
                             text=df_pareto['site'], textposition='top center', name='Pareto-фронт'))
    fig.update_layout(xaxis_title='Risk Score (0–100)', yaxis_title='NPV (млрд ₽)',
                      template='plotly_white', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Интерпретация:** точка на парето-фронте означает, что невозможно улучшить NPV без увеличения риска и наоборот.")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP EXPLANATIONS
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("SHAP-объяснение ключевых драйверов NPV")
    
    comp_select = st.selectbox("Выберите комбинацию", 
                               options=[f"{r['site'][:30]} × {r['reactor']} × {int(r['n'])}" 
                                       for _,r in df_full.head(10).iterrows()])
    
    if comp_select:
        site_name = comp_select.split(' × ')[0]
        parts = comp_select.split(' × ')
        reactor_name = parts[1]
        n_val = int(parts[2])
        
        site_obj = next((s for s in sites if site_name in s.name), None)
        react_obj = next((r for r in reactors if reactor_name in r.name), None)
        if site_obj and react_obj:
            gr = gate.evaluate(site_obj, react_obj, n_val)
            tariff = _site_tariff(site_obj)
            econ = calc_npvr(react_obj, site_obj, n_val, tariff=tariff)
            risk_score, rd = calc_risk(site_obj)
            mu, sig = gp.predict_site(np.array([site_obj.deficit_score, site_obj.logistics_score,
                                                 site_obj.geo_climate_score, 3, 3, site_obj.expert_score]))
            
            base_npv = econ['npv_bln']
            deficit_MW = {1:0,2:10,3:20,4:30,5:50}.get(site_obj.deficit_score, 0)
            total_P = react_obj.P_el_MW * n_val
            cap_gap = max(0, deficit_MW - total_P*0.8)/deficit_MW if deficit_MW else 0
            
            drivers = [
                ('capacity_coverage', total_P, base_npv * (1-cap_gap*0.3)),
                ('logistics_score', site_obj.logistics_score, -base_npv*0.15*(5-site_obj.logistics_score)/5),
                ('seismic_8plus', site_obj.seismic_balls, -base_npv*0.25 if site_obj.seismic_balls>=8 else 0),
                ('permafrost', 'yes' if site_obj.permafrost else 'no', -base_npv*0.12 if site_obj.permafrost else 0),
                ('port_type', site_obj.port_type or '', base_npv*0.10 if site_obj.port_type=='year_round' else (-base_npv*0.08 if site_obj.port_type=='frozen' else 0)),
                ('expert_score', site_obj.expert_score, base_npv*0.05*(site_obj.expert_score-15)/10),
            ]
            drivers.sort(key=lambda x: abs(x[2]), reverse=True)
            
            cols = st.columns(2)
            with cols[0]:
                st.metric("Базовый NPV", f"{base_npv:+.1f} млрд ₽")
                st.metric("Ликоэффициент", f"{econ['lcoe_rub_per_kwh']:.2f} ₽/кВт·ч")
            with cols[1]:
                st.metric("Экспертный μ", f"{mu:.1f}")
                st.metric("Risk Score", f"{risk_score:.0f}/100")
            
            st.markdown("#### Драйверы NPV (SHAP-like декомпозиция)")
            driver_df = pd.DataFrame(drivers, columns=['Фактор','Значение','Влияние млрд ₽'])
            driver_df['стрелка'] = driver_df['Влияние млрд ₽'].apply(lambda x: '↑' if x>0 else '↓')
            st.dataframe(driver_df, use_container_width=True, hide_index=True,
                        column_config={"стрелка": st.column_config.TextColumn()})
            
            fig2 = go.Figure(go.Bar(x=driver_df['Влияние млрд ₽'], y=driver_df['Фактор'], orientation='h',
                                     marker_color=['green' if v>0 else 'red' for v in driver_df['Влияние млрд ₽']]))
            fig2.update_layout(xaxis_title='Влияние на NPV (млрд ₽)', template='plotly_white', height=300)
            st.plotly_chart(fig2, use_container_width=True)
            
            # MC histogram
            np.random.seed(42)
            samples = []
            ag_kwh = react_obj.P_el_MW * n_val * 0.85 * 8760 * 1000
            for _ in range(mc_samples):
                tu = tariff*(1+np.random.uniform(-tariff_band/100,tariff_band/100))
                cu = econ['capex_total_bln']*1e9*(1+np.random.uniform(-capex_band/100,capex_band/100))
                ou = econ['opex_annual_bln']*1e9*(1+np.random.uniform(-capex_band/100,capex_band/100))
                annuity=(1-1.09**-50)/0.09
                ap = ag_kwh*tu-ou
                samples.append((-cu+ap*annuity*1e9)/1e9 if ap>0 else -cu/1e9-50)
            
            st.markdown("#### Распределение NPV по Monte Carlo")
            fig3 = px.histogram(samples, nbins=50, color_discrete_sequence=['#3498db'],
                                 labels={'value':'NPV (млрд ₽)'}, title=f'Sampled {mc_samples} times | mean={np.mean(samples):+.1f} | P(>0)={np.mean(np.array(samples)>0)*100:.1f}%')
            fig3.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 4 — MAP
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Карта площадок ДВ/Арктики (цвета = Composite Score)")
    
    # Assign approximate coordinates based on site names
    coords = {}
    for s in sites:
        name = s.name.lower()
        if 'большой камень' in name or 'примор' in name: coords[s.name] = (43.0, 132.7)
        elif 'сахалин' in name: coords[s.name] = (46.5, 142.5)
        elif 'воркута' in name: coords[s.name] = (67.5, 64.0)
        elif 'краснокаменск' in name: coords[s.name] = (51.3, 115.7)
        elif ' норильск' in name: coords[s.name] = (69.4, 88.2)
        elif 'билибино' in name: coords[s.name] = (66.1, 170.5)
        elif 'агылкинское' in name or 'якутия' in name: coords[s.name] = (62.0, 129.0)
        elif 'совиное' in name: coords[s.name] = (67.0, 172.0)
        elif 'певек' in name: coords[s.name] = (69.7, 170.3)
        elif 'тикси' in name: coords[s.name] = (73.6, 128.7)
        elif 'петропавловск' in name: coords[s.name] = (53.0, 158.5)
        elif ' анадырь' in name: coords[s.name] = (64.5, 177.5)
        elif 'мирный' in name: coords[s.name] = (62.5, 113.7)
        else: coords[s.name] = (60.0, 120.0)
    
    df_map = pd.DataFrame([(s.name, coords.get(s.name,(60,120)), s.expert_score, s.logistics_score) for s in sites],
                          columns=['site','lat','lon','expert','logistics'])
    
    fig_map = px.scatter_geo(df_map, lon='lon', lat='lat', color='expert', size='logistics',
                              hover_name='site', hover_data={'lat':False,'lon':False,'expert':True,'logistics':True},
                              projection='orthographic', title='Площадки МСАЭС ДВ/Арктики',
                              color_continuous_scale='Viridis')
    st.plotly_chart(fig_map, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Powering by Layer 1–4 гибридная система • Модель СВБР-100 калибрована по Южно-Сахалинску")
