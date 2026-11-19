"""Optional site covariates and their fold-local encoding."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .regional_design import is_missing, number

NUMERIC_FIELDS = (
    "latitude", "longitude", "population", "local_demand_min_mw", "local_demand_max_mw",
    "annual_electricity_gwh", "grid_connection_distance_km", "rail_distance_km", "port_distance_km",
    "heating_degree_days_18c", "transport_max_diameter_m", "available_land_m2",
)
REGION_FIELD = "admin_region"
CONTEXT_FIELDS = (REGION_FIELD, *NUMERIC_FIELDS)
COORDINATE_LIMITS = {"latitude": 90, "longitude": 180}
POSITIVE_FIELDS = {"local_demand_min_mw", "local_demand_max_mw", "transport_max_diameter_m", "available_land_m2"}
MIN_NUMERIC_ROWS = 4
MIN_REGION_ROWS = 2


def site_context(row: dict) -> dict[str, str | float]:
    context = {}
    region = row.get(REGION_FIELD)
    if not is_missing(region):
        context[REGION_FIELD] = " ".join(str(region).casefold().split())
    for field in NUMERIC_FIELDS:
        value = number(row.get(field))
        if value is None:
            continue
        if field in COORDINATE_LIMITS:
            valid = abs(value) <= COORDINATE_LIMITS[field]
        elif field == "population":
            valid = value >= 0 and value.is_integer()
        elif field in POSITIVE_FIELDS:
            valid = value > 0
        else:
            valid = value >= 0
        if not valid:
            raise ValueError(f"Invalid {field}: {value:g}")
        context[field] = value
    low = context.get("local_demand_min_mw")
    high = context.get("local_demand_max_mw")
    if low is not None and high is not None and low > high:
        raise ValueError("local_demand_min_mw must not exceed local_demand_max_mw")
    return context


def read_history_context(path: Path) -> dict[str, dict]:
    """Read aligned context only. Illustrative historical fields are ignored."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        result = {}
        for row in csv.DictReader(stream):
            name = row["site"].strip().casefold()
            status = (row.get("context_source_status") or "").strip().casefold()
            if status not in {"", "documented", "illustrative", "unknown"}:
                raise ValueError(f"Invalid context_source_status for {name!r}")
            if status in {"illustrative", "unknown"}:
                result[name] = {}
                continue
            context = site_context(row)
            if set(context) - {REGION_FIELD} and (
                status != "documented" or is_missing(row.get("context_source_reference"))
            ):
                raise ValueError(f"Numeric historical context for {name!r} requires documented status and a source")
            result[name] = context
        return result


def context_design(train: list[dict], queries: list[dict]):
    """Encode only features supported by the current training fold."""
    training_columns, query_columns = [], []
    used = [set() for _ in queries]
    labels = [row.get(REGION_FIELD) for row in train]
    query_labels = [row.get(REGION_FIELD) for row in queries]
    for label in sorted(set(labels) - {None}):
        if labels.count(label) >= MIN_REGION_ROWS:
            training_columns.append([x == label for x in labels])
            query_columns.append([x == label for x in query_labels])
            for fields, value in zip(used, query_labels):
                if value == label:
                    fields.add(REGION_FIELD)

    for field in NUMERIC_FIELDS:
        values = np.array([row.get(field, np.nan) for row in train], dtype=float)
        present = values[np.isfinite(values)]
        if len(present) < MIN_NUMERIC_ROWS:
            continue
        scale = present.std()
        if scale == 0:
            continue
        center = np.median(present)
        training_columns.append(np.where(np.isfinite(values), (values - center) / scale, 0.0))
        query_columns.append([(row[field] - center) / scale if field in row else 0.0 for row in queries])
        for fields, row in zip(used, queries):
            if field in row:
                fields.add(field)
    if not training_columns:
        return np.empty((len(train), 0)), np.empty((len(queries), 0)), used
    return np.column_stack(training_columns), np.column_stack(query_columns), used
