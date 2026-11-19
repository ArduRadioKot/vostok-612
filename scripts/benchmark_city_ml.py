"""Replay historical rubric scores in synthetic cities with controlled missing data."""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from nuclear_dss.city_pipeline import DEMO, KEY, ROOT, analyze_city, local_sites, read_scenarios
from nuclear_dss.data import FEATURES
from nuclear_dss.regional_design import read_inputs, write_csv
from nuclear_dss.site_context import REGION_FIELD, read_history_context
from nuclear_dss.site_imputation import read_history, site_scores

METHODS = ("none", "median", "bayesian-ridge", "context-ridge", "gpr")
POLICIES = ("random", "uncertainty", "decision-value")


def key(row):
    return tuple(row[field] for field in KEY)


def top3(rows):
    return {(row["scenario_id"], key(row)) for row in rows
            if row["rank"] is not None and row["rank"] <= 3}


def decision_metrics(result, oracle):
    """Evaluate returned leaders on the same complete inputs and scenario set."""
    _, rows, _, _, report = result
    _, full_rows, _, _, full_report = oracle
    leaders = {key(row) for row in report["leaders"]}
    reference = {key(row) for row in full_report["leaders"]}
    full_top = top3(full_rows)
    # Utility is bounded by [0, 1]. Use a one-point penalty when a leader is unavailable.
    losses = {x: [] for x in sorted(leaders)}
    for row in full_rows:
        x = key(row)
        if x in losses:
            loss = row["utility_loss"]
            losses[x].append(1.0 if loss is None else loss)
    return {
        "abstained": int(not leaders),
        "leader_overlap": int(bool(leaders & reference)),
        "exact_leader_set": int(leaders == reference),
        "scenario_top3_recall": len(top3(rows) & full_top) / len(full_top),
        "mean_full_loss_points": 100 * np.mean(list(losses.values())) if losses else None,
    }


def query_policies(sites, hidden, training, neutral, scenarios, rng):
    """Choose one cell using only observed scores, simulated worlds and known inputs."""
    ranking, outcomes, _, _, _ = neutral
    candidates = sorted({key(row) for row in ranking if row["status"] == "SCENARIO_SHORTLIST"})
    if not candidates:
        return {}
    scenarios_by_id = {scenario.scenario_id: i for i, scenario in enumerate(scenarios)}
    candidates_by_key = {candidate: i for i, candidate in enumerate(candidates)}
    site_by_id = {site["site_id"]: site_scores(site) for site in sites}
    missing_sites = {site_id for site_id, _ in hidden}
    worlds = {}
    for site_id in sorted(missing_sites):
        query = site_by_id[site_id]
        observed = np.isfinite(query)
        distance = np.abs(training[:, observed] - query[observed]).sum(axis=1)
        weights = 1 / (1 + distance)
        sample = training[rng.choice(len(training), size=256, p=weights / weights.sum())]
        worlds[site_id] = sample

    shifts = {}
    for site_id, sample in worlds.items():
        columns = [FEATURES.index(field) for sid, field in hidden if sid == site_id]
        shifts[site_id] = (sample[:, columns] - 3).sum(axis=1) / (4 * len(FEATURES))

    best = np.full((256, len(scenarios)), -np.inf)
    utilities = np.full((256, len(scenarios), len(candidates)), np.nan)
    for row in outcomes:
        if row["utility"] is None:
            continue
        i = scenarios_by_id[row["scenario_id"]]
        utility = row["utility"] + scenarios[i].weights[2] * shifts.get(row["site_id"], 0.0)
        best[:, i] = np.maximum(best[:, i], utility)
        index = candidates_by_key.get(key(row))
        if index is not None:
            utilities[:, i, index] = utility
    loss = np.where(np.isfinite(utilities), best[:, :, None] - utilities, 1.0).mean(axis=1)
    original_loss = loss.mean(axis=0).min()
    scores = {}
    for cell in hidden:
        site_id, field = cell
        values = worlds[site_id][:, FEATURES.index(field)]
        expected_loss = 0.0
        for value in np.unique(values):
            subset = values == value
            expected_loss += subset.mean() * loss[subset].mean(axis=0).min()
        scores[cell] = (float(values.std()), float(original_loss - expected_loss))
    return {
        "random": hidden[int(rng.integers(len(hidden)))],
        "uncertainty": max(hidden, key=lambda cell: scores[cell][0]),
        "decision-value": max(hidden, key=lambda cell: scores[cell][1]),
    }


def benchmark(history_path: Path, demo: Path, seed: int):
    history = read_history(history_path)
    context = read_history_context(history_path)
    if any(REGION_FIELD not in context[name] for name in history):
        raise ValueError("Grouped replay requires admin_region for every historical site")
    region, sites, reactors = read_inputs(demo / "region.json", demo / "sites.csv", ROOT / "data/reactors.csv")
    city = "Учебный Северный"
    sites = local_sites(sites, city)
    scenarios = read_scenarios(demo / "scenarios.csv", region["region_id"], city)
    reference_site = next(site for site in sites if site["site_id"] == "S2")
    other_sites = [site for site in sites if site["site_id"] != "S2"]
    physical_fields = {"local_demand_min_mw", "local_demand_max_mw", "transport_max_diameter_m", "available_land_m2"}
    pairs = list(combinations(FEATURES, 2))
    rng = np.random.default_rng(seed)
    trials, actions = [], []

    for name, truth in history.items():
        group = context[name][REGION_FIELD]
        # Site geometry, hard inputs, costs and scenario parameters remain synthetic.
        covariates = {f: v for f, v in context[name].items() if f not in physical_fields}
        target = {**reference_site, **dict(zip(FEATURES, truth)), **covariates, "historical_site": None}
        full = sorted([target, *other_sites], key=lambda row: row["site_id"])
        original = {row["site_id"]: row for row in full}
        oracle = analyze_city(region, full, reactors, scenarios, imputation="none", seed=seed)
        patterns = [(f"scatter-{index + 1}", [("S2", field) for field in pairs[pair]])
                    for index, pair in enumerate(rng.choice(len(pairs), size=2, replace=False))]
        patterns.append(("block-logistics", [(sid, "logistics") for sid in ("S1", "S2")]))
        for validation in ("leave-site-out", "leave-region-out"):
            train = {n: scores for n, scores in history.items() if n != name
                     and (validation == "leave-site-out" or context[n][REGION_FIELD] != group)}
            train_context = {n: context[n] for n in train}
            training = np.array(list(train.values()))
            for label, hidden in patterns:
                masked = [{**row, **{field: "" for sid, field in hidden if sid == row["site_id"]}}
                          for row in full]
                inputs = dict(history=train, history_context=train_context, seed=seed)
                base = {
                    "historic_site": name, "historic_region": group, "holdout": validation,
                    "pattern": "block" if label.startswith("block") else "scatter",
                    "mask": label, "training_count": len(train),
                    "hidden": "|".join(f"{sid}:{field}" for sid, field in hidden),
                }
                for method in METHODS:
                    result = analyze_city(region, masked, reactors, scenarios, imputation=method, **inputs)
                    models = result[4]["site_rubric_model"]
                    fields = {f for model in models.values() for f in model.get("context_features_used", [])}
                    if method == "none":
                        cell_mae = None
                    else:
                        cell_mae = float(np.mean([
                            abs(models[sid]["estimate"][field] - float(original[sid][field]))
                            for sid, field in hidden
                        ]))
                    trials.append({**base, "method": method, "context_features_used": "|".join(sorted(fields)),
                                   "cell_mae": cell_mae, **decision_metrics(result, oracle)})

                # Measure acquisition after one reveal, with median completion for every policy.
                neutral_sites = [{**row, **{field: 3 for sid, field in hidden if sid == row["site_id"]}}
                                 for row in masked]
                neutral = analyze_city(region, neutral_sites, reactors, scenarios, imputation="none", seed=seed)
                selected = query_policies(masked, hidden, training, neutral, scenarios, rng)
                for policy, (sid, field) in selected.items():
                    revealed = [{**row, field: original[sid][field]} if row["site_id"] == sid else row
                                for row in masked]
                    result = analyze_city(region, revealed, reactors, scenarios, imputation="median", **inputs)
                    actions.append({**base, "policy": policy, "requested": f"{sid}:{field}",
                                    **decision_metrics(result, oracle)})
    return trials, actions


def summarize(rows, column):
    groups = {}
    for row in rows:
        group = (row["holdout"], row["pattern"], row[column])
        groups.setdefault(group, []).append(row)
    reports = []
    for (holdout, pattern, name), group in sorted(groups.items()):
        losses = [row["mean_full_loss_points"] for row in group if row["mean_full_loss_points"] is not None]
        reports.append({
            "holdout": holdout, "pattern": pattern, column: name, "n": len(group),
            "recommendations": len(losses),
            **{metric: float(np.mean([row[metric] for row in group])) for metric in
               ("abstained", "leader_overlap", "exact_leader_set", "scenario_top3_recall")},
            "mean_full_loss_points": None if not losses else float(np.mean(losses)),
        })
        if column == "method":
            errors = [row["cell_mae"] for row in group if row["cell_mae"] is not None]
            reports[-1]["cell_mae"] = float(np.mean(errors)) if errors else None
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "data/sites.csv")
    parser.add_argument("--demo", type=Path, default=DEMO)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=612)
    args = parser.parse_args()
    trials, actions = benchmark(args.history, args.demo, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "imputation_trials": trials, "query_trials": actions,
        "imputation_summary": summarize(trials, "method"), "query_summary": summarize(actions, "policy"),
    }
    for name, records in outputs.items():
        write_csv(args.out / f"{name}.csv", records, list(records[0]))
    report = {
        "seed": args.seed, "historical_sites": len({r["historic_site"] for r in trials}),
        "context_features_available": sorted({f for row in read_history_context(args.history).values() for f in row}),
        "imputation_runs": len(trials), "query_runs": len(actions),
        "imputation_methods": METHODS,
        "design": "Historical ordinal rows replayed at a synthetic city site; physical data and costs are synthetic.",
    }
    (args.out / "protocol.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
