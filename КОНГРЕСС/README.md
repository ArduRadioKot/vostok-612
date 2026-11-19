# Малые АЭС для ДВ/Арктики — Система поддержки выбора площадки

## О проекте

Многокритериальная决策支持系统 (DSS) для отбора оптимальных комбинаций площадка × реактор МСАЭС для Дальнего Востока и Арктики России.

**Реализовано 4 слоя:**
1. **Gate** — жёсткие булевы ограничения
2. **Expert GP** — Гауссовские процессы экспертных оценок
3. **Risk + Strategy** — скор-модели рисков и композитный весовой скор
4. **Economic Optimizer** — NPV-формула + Monte Carlo неопределённостей

## Запуск

```bash
# DASHBOARD (Streamlit)
export PATH="$HOME/Library/Python/3.15/bin:$PATH"
cd model/output
streamlit run app.py --server.port 8502
# → http://localhost:8502

# BATCH MODE (CLI)
cd model
python3 src/ranker.py                          # топ-15 всех комбинаций
python3 src/pareto_ranker.py                   # взвешенный Pareto-топ
python3 output/test_explanations.py            # SHAP-объяснения для СВБР-100
```

## Структура

```
model/
├── gate/                          # Слой 1 — Hard Gate
│   └── gate_rules.py             # Булевы правила, SiteInfo/ReactorInfo
├── src/                           # Слои 2–4
│   ├── load_data.py              # Загрузка данных (sites.csv, reactors.csv)
│   ├── econ_model.py             # Слой 3 — NPV-формула
│   ├── expert_gp.py              # Слой 2a — Gaussian Process
│   ├── risk_model.py             # Слой 2c — Risk Score
│   ├── strategy_model.py         # Слой 2d — Weighted Composite
│   ├── pareto_ranker.py          # Слой 3+4 — Полный ранкер
│   └── explanations.py           # Слой 4 — SHAP + MC
├── output/
│   ├── app.py                    # Streamlit дашборд (5 вкладок)
│   ├── full_ranking.csv          # Полный перебор (169 КБ)
│   ├── TOP-SVBR100.md            # Топ-10 СВБР-100
│   └── FINAL_REPORT.md           # Итоговый отчёт
└── data/
    ├── sites.csv                 # 21 площадка
    └── reactors.csv              # 26 реакторов
```

## Результаты (СВБР-100)

| # | Площадка | Модули | NPV млрд ₽ | LCOE ₽/кВтч | P(NPV>0) |
|---|---------|--------|------------|-------------|----------|
| 1 | Большой камень | ×3 | **+1402** | 18.2 | 100% |
| 2 | Южно-Сахалинск | ×3 | **+1391** | 18.6 | 100% |
| 3 | Воркута | ×3 | **+1063** | 18.9 | 100% |

### Требования

- Python ≥ 3.10
- pandas, numpy, scikit-learn, plotly, streamlit
- shap (опционально, в explanations.py)

```bash
pip install pandas numpy scikit-learn plotly streamlit shap
```

## Авторы / contributors

Agnes (Cline-based agent), Sapiens AI
Дата: ноябрь 2026
