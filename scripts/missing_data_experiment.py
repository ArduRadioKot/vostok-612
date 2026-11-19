"""Masked observed rubric recovery: exploratory, not expert preference prediction."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nuclear_dss.data import FEATURES, read_sources
from nuclear_dss.ranking import WEIGHTS, _scale_percentile

OUT = ROOT / 'experiments' / 'missing_data_2026'
REPEATS = 300
SEED = 612


def main():
    sites, _ = read_sources(ROOT)
    ranking_path = OUT / 'baseline' / 'ranking.csv'
    ranking = pd.read_csv(ranking_path)
    x = sites[list(FEATURES)].to_numpy(float)
    site_index = {s: i for i, s in enumerate(sites.site)}
    idx = ranking.site.map(site_index).to_numpy()
    econ = _scale_percentile(ranking.npv_bln.to_numpy())
    risk = ranking.risk_proxy.to_numpy()
    fit = ranking.demand_fit.to_numpy()
    original_site_score = x.sum(axis=1)
    original = (WEIGHTS['npv'] * econ + WEIGHTS['expert'] * original_site_score[idx] / 30
                + WEIGHTS['risk'] * (1 - risk / 100) + WEIGHTS['demand_fit'] * fit)
    original_best = original.max()
    rng = np.random.default_rng(SEED)
    records = []
    for frac in (.1, .2, .3):
        for rep in range(REPEATS):
            # Exact count, shared masks for paired comparison; every column observed.
            while True:
                mask = np.zeros(x.size, dtype=bool)
                mask[rng.choice(x.size, size=round(frac*x.size), replace=False)] = True
                mask = mask.reshape(x.shape)
                if (~mask).sum(axis=0).min() >= 3:
                    break
            missing = x.copy()
            missing[mask] = np.nan
            for name, estimator in [('mean', SimpleImputer()),
                                    ('knn_3', KNNImputer(n_neighbors=3, weights='distance'))]:
                recovered = np.clip(estimator.fit_transform(missing), 1, 5)
                score = recovered.sum(axis=1)
                # Only logistics changes the existing risk proxy. Other components remain fixed.
                changed_risk = np.clip(risk + 5 * (x[idx, 2] - recovered[idx, 2]), 0, 100)
                modified = (WEIGHTS['npv'] * econ + WEIGHTS['expert'] * score[idx] / 30
                            + WEIGHTS['risk'] * (1 - changed_risk / 100)
                            + WEIGHTS['demand_fit'] * fit)
                chosen = int(np.argmax(modified))
                # Equal reference maxima are all accepted; no arbitrary tie penalty.
                records.append(dict(missing_fraction=frac, repeat=rep, method=name,
                    masked_count=int(mask.sum()), mae=float(np.abs(recovered[mask]-x[mask]).mean()),
                    rmse=float(np.sqrt(np.mean((recovered[mask]-x[mask])**2))),
                    site_loss=float(original_site_score.max() - original_site_score[int(np.argmax(score))]),
                    site_best_retained=int(np.isclose(original_site_score[int(np.argmax(score))],original_site_score.max())),
                    config_best_retained=int(np.isclose(original[chosen], original_best, atol=1e-12)),
                    config_regret=float(original_best-original[chosen])))
    df=pd.DataFrame(records)
    summary=(df.groupby(['missing_fraction','method'],as_index=False)
             .agg(n=('repeat','size'), mae=('mae','mean'), rmse=('rmse','mean'),
                  site_best_retained=('site_best_retained','mean'), site_loss=('site_loss','mean'),
                  config_best_retained=('config_best_retained','mean'), config_regret=('config_regret','mean')))
    OUT.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT/'trials.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    info={'seed':SEED,'repeats':REPEATS,'features':list(FEATURES),'shape':list(x.shape),
          'mask':'uniform exact cell count, at least 3 observed per column',
          'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT/'data/sites.csv',ranking_path]},
          'reference':'existing six-criterion site scores, not independently verified labels',
          'decision':'site sum; existing configuration score with recalculated site sum and logistics risk; other variables fixed',
          'caveats':['Artificial missingness only','KNN and mean fitted to partially observed 21-site matrix',
                     'Source data not ground truth','Fixed economic inputs and candidate set','Not independent expert preference validation']}
    (OUT/'protocol.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
