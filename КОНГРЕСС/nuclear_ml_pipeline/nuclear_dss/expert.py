"""Leakage-aware optional supervised learning of independently obtained expert scores.

The bundled site data do NOT contain an independently measured target. Training is
therefore opt-in and must be supplied with genuine external expert labels.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .data import FEATURES


def load_independent_labels(path: Path, sites: pd.DataFrame):
    """Read one independently obtained aggregate score per site (range [0, 30]).

    No labels are invented or copied from the legacy derived expert_score column.
    The input must be one site per row to prevent the same site leaking across folds.
    """
    df = pd.read_csv(path, encoding='utf-8-sig')
    required = {'site', 'independent_expert_label'}
    if not required.issubset(df.columns):
        raise ValueError('Expected columns: site,independent_expert_label')
    if df['site'].duplicated().any() or df['site'].isna().any():
        raise ValueError('Duplicate or empty site labels')
    if not set(df['site']).issubset(set(sites['site'])):
        raise ValueError('Labels contain unknown site identifiers')
    joined = sites.merge(df[['site', 'independent_expert_label']],
                         on='site', how='inner', validate='one_to_one')
    if len(joined) < 8:
        raise ValueError('At least 8 independently labeled sites required (diagnostic minimum)')
    joined['independent_expert_label'] = pd.to_numeric(
        joined['independent_expert_label'], errors='raise')
    y = joined['independent_expert_label'].to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any() or (y > 30).any():
        raise ValueError('Independent labels must be numeric, 0..30')
    rubric_sum = joined[list(FEATURES)].sum(axis=1).to_numpy(float)
    if np.allclose(y, rubric_sum):
        raise ValueError('Labels are identical to the arithmetic sum: derived, not independent target')
    return joined


def make_model():
    """Fixed kernel removes hyperparameter-selection bias in a tiny sample."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    kernel = (ConstantKernel(1., constant_value_bounds='fixed') *
              Matern(length_scale=2., length_scale_bounds='fixed', nu=1.5) +
              WhiteKernel(noise_level=0.1, noise_level_bounds='fixed'))
    return Pipeline([
        ('scale', StandardScaler()),
        ('gp', GaussianProcessRegressor(
            kernel=kernel, alpha=1e-5, normalize_y=True,
            optimizer=None, random_state=42)),
    ])


def _predict_with_std(model, x):
    # Pipeline.predict() does not expose return_std; use its fitted scaler.
    x_scaled = model.named_steps['scale'].transform(x)
    return model.named_steps['gp'].predict(x_scaled, return_std=True)


def loo_predictions(labeled):
    """Strict leave-one-SITE-out evaluation, with refitted preprocessing per fold."""
    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = labeled[list(FEATURES)].to_numpy(dtype=float)
    y = labeled['independent_expert_label'].to_numpy(dtype=float)
    mu = np.empty_like(y)
    sigma = np.empty_like(y)
    ridge_mu = np.empty_like(y)
    mean_mu = np.empty_like(y)
    for tr, te in LeaveOneOut().split(x):
        gp = make_model()
        gp.fit(x[tr], y[tr])
        m, s = _predict_with_std(gp, x[te])
        mu[te] = m
        sigma[te] = s
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(x[tr], y[tr])
        ridge_mu[te] = ridge.predict(x[te])
        mean_mu[te] = y[tr].mean()
    return pd.DataFrame({
        'site': labeled['site'].to_numpy(),
        'true_expert_label': y,
        'gp_prediction': np.clip(mu, 0, 30),
        'gp_std_uncalibrated': sigma,
        'ridge_prediction': np.clip(ridge_mu, 0, 30),
        'train_mean_prediction': mean_mu,
        'rubric_sum_prediction': x.sum(axis=1),
    })


def evaluate_predictions(predictions):
    """Metrics computed only from out-of-fold predictions, not training fit."""
    y = predictions['true_expert_label'].to_numpy(float)
    answer = {'n': len(y), 'validation': 'leave-one-site-out', 'model': 'GPR-Matern-3/2-fixed',
              'target': 'independent_expert_label', 'uncertainty_calibration': 'NOT_CALIBRATED'}
    for name, field in (
            ('gp', 'gp_prediction'),
            ('ridge_baseline', 'ridge_prediction'),
            ('rubric_sum_baseline', 'rubric_sum_prediction'),
            ('train_mean_baseline', 'train_mean_prediction')):
        pred = predictions[field].to_numpy(float)
        answer[f'{name}_loo_mae'] = round(float(np.mean(np.abs(y-pred))), 4)
        answer[f'{name}_loo_rmse'] = round(float(np.sqrt(np.mean((y-pred)**2))), 4)
    sig = np.maximum(predictions['gp_std_uncalibrated'].to_numpy(float), 1e-8)
    mu = predictions['gp_prediction'].to_numpy(float)
    z = 1.6448536269514722  # normal two-sided 90% interval; diagnostic ONLY
    answer['gp_loo_nominal90_coverage_uncalibrated'] = round(
        float(np.mean((y >= mu-z*sig) & (y <= mu+z*sig))), 4)
    answer['gp_loo_mean_std_uncalibrated'] = round(float(np.mean(sig)), 4)
    answer['gp_loo_gaussian_nll_uncalibrated'] = round(float(np.mean(
        np.log(sig*np.sqrt(2*np.pi)) + 0.5*((y-mu)/sig)**2)), 4)
    answer['note'] = ('Metrics require real independently collected labels. '
                      'Small N, unknown label quality, and uncalibrated GP intervals '
                      'do not establish out-of-domain generalization.')
    # Retain old short names for downstream/backward compatibility.
    answer['arithmetic_baseline_mae'] = answer['rubric_sum_baseline_loo_mae']
    return answer


def loo_evaluate(labeled):
    return evaluate_predictions(loo_predictions(labeled))


def fit_predict(sites, labeled):
    model = make_model()
    model.fit(labeled[list(FEATURES)].to_numpy(float),
              labeled['independent_expert_label'].to_numpy(float))
    x = sites[list(FEATURES)].to_numpy(float)
    mu, std = _predict_with_std(model, x)
    return np.clip(mu, 0, 30), std, model


def train_and_export(sites, labeled, output_dir, labels_path):
    """Persist a fitted model, out-of-fold evidence, and a machine-readable model card."""
    import hashlib
    import json
    import joblib
    import sklearn

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = loo_predictions(labeled)
    metrics = evaluate_predictions(predictions)
    _, _, model = fit_predict(sites, labeled)
    # Match exact data/schema via hash of the training input, not absolute paths.
    hash_input = labeled[['site', *FEATURES, 'independent_expert_label']].sort_values('site')
    fingerprint = hashlib.sha256(hash_input.to_csv(index=False).encode('utf-8')).hexdigest()
    metadata = {
        'model_version': '3.0.0', 'method': 'GPR Matérn 3/2 with fixed hyperparameters',
        'features': list(FEATURES), 'target': 'independent_expert_label',
        'n_train_sites': len(labeled), 'train_data_sha256': fingerprint,
        'scikit_learn_version': sklearn.__version__,
        'validation': metrics, 'label_source_file': Path(labels_path).name,
        'caveats': ['External expert labels must be authentic and independently produced.',
                    'Model uncertainty not calibrated.',
                    'Research use only; not a nuclear-safety or siting model.'],
    }
    joblib.dump({'model': model, 'metadata': metadata}, output_dir/'expert_gp.joblib')
    predictions.to_csv(output_dir/'cv_predictions.csv', index=False, encoding='utf-8-sig')
    (output_dir/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir/'model_card.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def load_trained_model(path):
    """Only load trusted self-generated joblib files; pickle deserialization is unsafe."""
    import joblib
    obj = joblib.load(path)
    if not isinstance(obj, dict) or 'model' not in obj or 'metadata' not in obj:
        raise ValueError('Invalid model artifact')
    meta = obj['metadata']
    if meta.get('features') != list(FEATURES) or meta.get('target') != 'independent_expert_label':
        raise ValueError('Unexpected trained model schema')
    return obj['model'], meta


def predict_model(sites, model):
    mu, sigma = _predict_with_std(model, sites[list(FEATURES)].to_numpy(float))
    return np.clip(mu, 0, 30), sigma
