"""Experimental challengers for independently labelled expert preference regression.

This module NEVER trains from the supplied six-score arithmetic target. TabPFN-3.5
is opt-in: weights and license authorization are not included or downloaded by
project packaging. Running it may require separate authentication/download.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import FEATURES
from .expert import loo_predictions


def paired_bootstrap_mae_delta(y, trial, reference, seed=42, n_resamples=2000):
    """Site-resampling interval for MAE(trial) - MAE(reference); descriptive only."""
    y=np.asarray(y,float)
    absolute_delta=np.abs(y-np.asarray(trial,float))-np.abs(y-np.asarray(reference,float))
    rng=np.random.default_rng(seed)
    subsamples=rng.integers(0,len(y),size=(n_resamples,len(y)))
    dist=absolute_delta[subsamples].mean(axis=1)
    return {'observed_delta_mae':round(float(absolute_delta.mean()),4),
            'bootstrap_p025':round(float(np.quantile(dist,.025)),4),
            'bootstrap_p975':round(float(np.quantile(dist,.975)),4),
            'note':'Descriptive paired site bootstrap; no independent external validation.'}


def _make_tabpfn():
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except ImportError as exc:
        raise RuntimeError('Optional TabPFN benchmark requires pip install tabpfn; model weights require separate license acceptance.') from exc
    # Explicit pin to the proposed 2026 checkpoint, not whichever model becomes default.
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)


def compare_models(labeled: pd.DataFrame, *, include_tabpfn: bool=False, seed=42):
    """Use full Leave-One-SITE-Out for every challenger; no random row splitting.

    The TabPFN challenger is never installed/loaded unless explicitly requested.
    Results are hypothesis-generating, because model selection on same 21 sites
    biases the best-observed number. Final selection requires an external cohort.
    """
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import BayesianRidge
    from sklearn.svm import SVR

    pred=loo_predictions(labeled).copy()
    x=labeled[list(FEATURES)].to_numpy(float)
    y=labeled['independent_expert_label'].to_numpy(float)
    algorithms = {
        'bayesian_ridge':lambda:make_pipeline(StandardScaler(),BayesianRidge()),
        'svr_rbf':lambda:make_pipeline(StandardScaler(),SVR(C=3.,epsilon=.25,gamma='scale')),
    }
    if include_tabpfn:
        algorithms['tabpfn_3_5']=_make_tabpfn
    for name, constructor in algorithms.items():
        values=np.empty_like(y)
        for train,test in LeaveOneOut().split(x):
            model=constructor()
            model.fit(x[train],y[train])
            values[test]=model.predict(x[test])[0]
        pred[name+'_prediction']=np.clip(values,0.,30.)
    scores={}
    for col in [c for c in pred.columns if c.endswith('_prediction')]:
        val=pred[col].to_numpy(float)
        short=col.removesuffix('_prediction')
        scores[short]={
            'mae':round(float(np.abs(y-val).mean()),4),
            'rmse':round(float(np.sqrt(np.mean((y-val)**2))),4),
            'vs_gp':paired_bootstrap_mae_delta(y,val,pred['gp_prediction'],seed=seed),
            'vs_rubric_sum':paired_bootstrap_mae_delta(y,val,pred['rubric_sum_prediction'],seed=seed+1),
        }
    report={
        'n_independent_sites':len(y),
        'cv':'leave-one-site-out with refitted transformations inside every fold',
        'challenger_models':[k for k in algorithms],
        'tabpfn_3_5_status':'RUN_AND_VALIDATED_ON_PROVIDED_LABELS' if include_tabpfn else 'NOT_RUN_OPTIONAL',
        'selection_warning':'Do not choose best CV model and report its same CV error as unbiased generalization; require external validation on new sites.',
        'target_warning':'External independent expert labels required. Arithmetic sum of rubrics is NOT a learnable research target.',
        'metrics':scores,
    }
    return pred,report


def export_comparison(labeled, out: Path, include_tabpfn=False):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    predictions,report=compare_models(labeled,include_tabpfn=include_tabpfn)
    predictions.to_csv(out/'cv_comparison.csv',index=False,encoding='utf-8-sig')
    (out/'model_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
