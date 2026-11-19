"""Estimate missing ordinal site preferences from observed historical rubric cells."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data import FEATURES
from .expert import make_model as make_gpr
from .regional_design import number
from .site_context import context_design, site_context

METHODS = ("median", "bayesian-ridge", "blend")
CITY_METHODS = (*METHODS, "auto", "context-ridge", "gpr", "none")


def site_scores(row: dict) -> np.ndarray:
    values = [number(row.get(key)) for key in FEATURES]
    for key, value in zip(FEATURES, values):
        if value is not None and not 1 <= value <= 5:
            raise ValueError(f"Rubric {key} must be in [1, 5]")
    return np.array(values, dtype=float)


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"site", *FEATURES}.issubset(reader.fieldnames or []):
            raise ValueError("Imputation history requires site and all six rubric columns")
        result = {}
        for row in reader:
            key = row["site"].strip().casefold()
            scores = site_scores(row)
            if not key or key in result or np.isnan(scores).any():
                raise ValueError("History must contain unique sites and complete rubric rows")
            result[key] = scores
    if len(result) < 4:
        raise ValueError("Imputation history requires at least four complete sites")
    return result


def _predict_missing(train: np.ndarray, queries: np.ndarray, observed: np.ndarray,
                     method: str) -> np.ndarray:
    targets = train[:, ~observed]
    median = np.median(targets, axis=0)
    if method == "median":
        return np.broadcast_to(median, (len(queries), len(median)))
    if method == "gpr":
        # The fixed kernel is shared across targets, with each response normalized separately.
        model = make_gpr().fit(train[:, observed], targets)
        predictions = model.predict(queries[:, observed]).reshape(len(queries), targets.shape[1])
        return np.clip(predictions, 1, 5)
    if method not in {"bayesian-ridge", "blend"}:
        raise ValueError(f"Unknown imputation method: {method!r}")
    from sklearn.linear_model import BayesianRidge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    predictions = []
    for target in targets.T:
        model = make_pipeline(StandardScaler(), BayesianRidge())
        model.fit(train[:, observed], target)
        predictions.append(model.predict(queries[:, observed]))
    ridge = np.clip(np.column_stack(predictions), 1, 5)
    return (median + ridge) / 2 if method == "blend" else ridge


def predict_missing(train: np.ndarray, query: np.ndarray, observed: np.ndarray,
                    method: str) -> np.ndarray:
    return _predict_missing(train, query[None, :], observed, method)[0]


def predict_with_context(train, queries, observed, train_context, query_context):
    """Fit ridge with and without context on the same rubric values."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    base_train = scaler.fit_transform(train[:, observed])
    base_query = scaler.transform(queries[:, observed])
    targets = train[:, ~observed]
    baseline = Ridge(alpha=4.0).fit(base_train, targets).predict(base_query).reshape(len(queries), -1)
    extra_train, extra_query, used = context_design(train_context, query_context)
    contextual = baseline.copy()
    supported = np.array([bool(fields) for fields in used])
    if supported.any():
        augmented_train = np.column_stack((base_train, extra_train))
        augmented_query = np.column_stack((base_query, extra_query))
        predicted = Ridge(alpha=4.0).fit(augmented_train, targets).predict(augmented_query[supported])
        contextual[supported] = predicted.reshape(supported.sum(), -1)
    return np.clip(baseline, 1, 5), np.clip(contextual, 1, 5), used


def validate_context(train, observed, context, *, groups=None):
    """Compute per-site residuals and report where additional covariates apply."""
    groups = range(len(train)) if groups is None else groups
    grouped = {}
    for i, group in enumerate(groups):
        grouped.setdefault(group, []).append(i)
    errors = {key: np.empty((len(train), (~observed).sum()))
              for key in ("median", "ridge", "context-ridge")}
    supported = np.zeros(len(train), dtype=bool)
    for held_out in grouped.values():
        training = np.ones(len(train), dtype=bool)
        training[held_out] = False
        fold, queries = train[training], train[held_out]
        if len(fold) < 2:
            raise ValueError("Each validation fold needs at least two training sites")
        ridge, enriched, used = predict_with_context(
            fold, queries, observed, [row for row, keep in zip(context, training) if keep],
            [context[i] for i in held_out],
        )
        truth = queries[:, ~observed]
        median = _predict_missing(fold, queries, observed, "median")
        fold_supported = np.array([bool(fields) for fields in used])
        errors["median"][held_out] = truth - median
        errors["ridge"][held_out] = truth - ridge
        errors["context-ridge"][held_out] = truth - np.where(fold_supported[:, None], enriched, median)
        supported[held_out] = fold_supported
    return errors, supported


def _all_predictions(train: np.ndarray, queries: np.ndarray, observed: np.ndarray) -> np.ndarray:
    median = _predict_missing(train, queries, observed, "median")
    ridge = _predict_missing(train, queries, observed, "bayesian-ridge")
    return np.stack((median, ridge, (median + ridge) / 2), axis=1)


def _preferred_method(errors: np.ndarray) -> int:
    """Select by mean absolute error and prefer the median on numerical ties."""
    scores = np.abs(errors).mean(axis=(0, 2))
    return int(np.flatnonzero(np.isclose(scores, scores.min(), rtol=0, atol=1e-12))[0])


def cross_validate_imputation(train: np.ndarray, observed: np.ndarray, *, nested: bool,
                              include_gpr: bool = False):
    """Return held-out residuals for fixed methods and, optionally, the selector.

    Pairwise holdouts serve the inner folds in both directions. The outer site's
    hidden values are never used to choose its own imputation method.
    """
    indices = np.arange(len(train))
    truth = train[:, ~observed]
    errors = np.empty((len(train), len(METHODS), truth.shape[1]))
    gpr_errors = np.empty_like(truth) if include_gpr else None
    for i in indices:
        fold, query = train[indices != i], train[[i]]
        errors[i] = truth[i] - _all_predictions(fold, query, observed)[0]
        if gpr_errors is not None:
            gpr_errors[i] = truth[i] - _predict_missing(fold, query, observed, "gpr")[0]
    residuals = dict(zip(METHODS, errors.transpose(1, 0, 2)))
    if gpr_errors is not None:
        residuals["gpr"] = gpr_errors
    if not nested:
        return residuals, None

    inner = np.zeros((len(train), *errors.shape))
    for i in indices:
        for j in indices[i + 1:]:
            fold = train[(indices != i) & (indices != j)]
            pair = [i, j]
            pair_errors = truth[pair, None, :] - _all_predictions(fold, train[pair], observed)
            inner[i, j] = pair_errors[1]
            inner[j, i] = pair_errors[0]
    selected = [_preferred_method(inner[i, indices != i]) for i in indices]
    residuals["auto"] = errors[indices, selected]
    return residuals, METHODS[_preferred_method(errors)]


def scenario_site_scores(site: dict, history: dict[str, np.ndarray] | None, method: str,
                         n_scenarios: int, rng: np.random.Generator, *, history_context=None):
    """Sample whole held-out residual vectors so gaps on a site move together."""
    query = site_scores(site)
    observed = np.isfinite(query)
    missing = [key for key, known in zip(FEATURES, observed) if not known]
    if observed.all():
        return np.full(n_scenarios, (query.mean() - 1) / 4), {"status": "OBSERVED", "missing": []}
    if method == "none" or not observed.any() or history is None:
        return None, {"status": "INSUFFICIENT_RUBRIC", "missing": missing}
    explicit_key = (site.get("historical_site") or "").strip().casefold()
    if explicit_key and explicit_key not in history:
        raise ValueError(f"Unknown historical_site for {site['site_id']!r}")
    excluded = {explicit_key, site["site_id"].strip().casefold(), site["name"].strip().casefold()}
    names = [name for name in sorted(history) if name not in excluded]
    train = np.array([history[name] for name in names])
    if len(train) < (4 if method == "auto" else 3):
        return None, {"status": "INSUFFICIENT_HISTORY", "missing": missing}
    details = {}
    if method == "context-ridge":
        context = (history_context or {})
        train_context = [context.get(name, {}) for name in names]
        query_context = site_context(site)
        _, predictions, (used,) = predict_with_context(train, query[None, :], observed,
                                                        train_context, [query_context])
        errors, coverage = validate_context(train, observed, train_context)
        selected = method if used else "median"
        estimate = predictions[0] if used else predict_missing(train, query, observed, selected)
        details = {"context_features_provided": sorted(query_context), "context_features_used": sorted(used),
                   "held_out_context_sites": int(coverage.sum())}
        validation = "leave_one_site_out"
    else:
        errors, auto_selected = cross_validate_imputation(
            train, observed, nested=method == "auto", include_gpr=method == "gpr")
        selected = auto_selected or method
        estimate = predict_missing(train, query, observed, selected)
        validation = "nested_leave_one_site_out" if auto_selected else "leave_one_site_out"
    residuals = errors[method if method == "auto" else selected]
    drawn = residuals[rng.integers(len(residuals), size=n_scenarios)]
    filled = np.clip(estimate + drawn, 1, 5)
    normalized = ((query[observed].sum() + filled.sum(axis=1)) / len(FEATURES) - 1) / 4
    return normalized, {
        "status": "IMPUTED_SOFT_RUBRIC",
        "method": method,
        "selected_method": selected,
        "validation": validation,
        "fixed_method_mae": {key: float(np.abs(value).mean()) for key, value in errors.items() if key != "auto"},
        "missing": missing,
        "estimate": dict(zip(missing, estimate.tolist())),
        "n_history_sites": len(train),
        "held_out_site_mae": float(np.abs(residuals).mean()),
        "held_out_site_median_mae": float(np.abs(errors["median"]).mean()),
        **details,
    }
