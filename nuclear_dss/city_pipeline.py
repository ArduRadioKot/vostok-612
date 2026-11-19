"""City-scoped scenario ranking for the VOSTOK-612 research pipeline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .economics import cashflow
from .provenance import cost_profile
from .regional_design import number, positive_number, read_csv, read_inputs, region_options, site_checks, write_csv
from .site_context import read_history_context
from .site_imputation import CITY_METHODS, read_history, scenario_site_scores

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "regional_demo"
CRITERIA = ("cost", "power_fit", "site_rubric")
KEY = ("site_id", "reactor", "modules")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    demand_mw: float
    tariff_rub_per_kwh: float
    cost_profile: str
    capex_factor: float
    opex_factor: float
    weights: tuple[float, float, float]


def read_scenarios(path: Path, region_id: str, city: str) -> list[Scenario]:
    scenarios = []
    seen = set()
    for row in read_csv(path):
        if not (row.get("region_id") or "").strip() or not (row.get("city") or "").strip():
            raise ValueError("Each scenario requires region_id and city")
        if row["region_id"].strip() != region_id or row["city"].strip().casefold() != city.strip().casefold():
            continue
        name = (row.get("scenario_id") or "").strip()
        if not name or name in seen:
            raise ValueError("scenario_id must be unique and nonempty")
        seen.add(name)
        weights = tuple(positive_number(row.get(f"w_{k}"), f"{name}.w_{k}", allow_zero=True)
                        for k in CRITERIA)
        if not math.isclose(sum(weights), 1, abs_tol=1e-8):
            raise ValueError(f"Scenario {name}: weights must sum to one")
        profile = (row.get("cost_profile") or "").strip()
        if profile not in {"legacy_2026", "bigstone_2026"}:
            raise ValueError(f"Scenario {name}: unknown cost profile {profile!r}")
        scenarios.append(Scenario(
            name, positive_number(row.get("demand_mw"), f"{name}.demand_mw"),
            positive_number(row.get("tariff_rub_per_kwh"), f"{name}.tariff_rub_per_kwh"),
            profile,
            positive_number(row.get("capex_factor", 1), f"{name}.capex_factor"),
            positive_number(row.get("opex_factor", 1), f"{name}.opex_factor"),
            weights,
        ))
    if not scenarios:
        raise ValueError("No scenarios match the requested region and city")
    return sorted(scenarios, key=lambda scenario: scenario.scenario_id)


def read_cost_estimates(path: Path, sites: list[dict], reactors: list[dict], max_modules: int):
    """Read full-configuration costs for the selected city."""
    site_ids = {site["site_id"] for site in sites}
    technologies = {reactor["reactor"] for reactor in reactors}
    estimates = {}
    for row in read_csv(path):
        if not {"site_id", "reactor", "modules", "capex_bln", "cash_opex_bln_year",
                "source_reference"}.issubset(row):
            raise ValueError("Invalid explicit cost file header")
        if row["site_id"] not in site_ids:
            continue
        modules = positive_number(row["modules"], "cost-estimate modules")
        if not modules.is_integer() or modules > max_modules or row["reactor"] not in technologies:
            raise ValueError("Cost estimate has unknown reactor or module count")
        key = (row["site_id"], row["reactor"], int(modules))
        if key in estimates or not row["source_reference"].strip():
            raise ValueError("Cost estimates require unique keys and a source_reference")
        estimates[key] = (
            positive_number(row["capex_bln"], "capex_bln"),
            positive_number(row["cash_opex_bln_year"], "cash_opex_bln_year", allow_zero=True),
            row["source_reference"].strip(),
        )
    return estimates


def local_sites(sites: list[dict], city: str, site_id: str | None = None) -> list[dict]:
    city = city.strip().casefold()
    if not city or any(not (s.get("city") or "").strip() for s in sites):
        raise ValueError("Provide --city and a nonempty city column for every site")
    matching = [s for s in sites if s["city"].strip().casefold() == city
                and (site_id is None or s["site_id"] == site_id)]
    if not matching:
        raise ValueError("No declared candidate sites match the requested city/site")
    return sorted(matching, key=lambda site: site["site_id"])


def city_prechecks(region, site, region_audit, site_audit):
    """Only declared analytical constraints gate the scenario shortlist."""
    reasons = [c.code for c in site_audit if c.outcome == "SCENARIO_FAIL"]
    if reasons:
        return "REJECTED", reasons
    unresolved = []
    if (site.get("land_restriction") or "").strip().casefold() != "none":
        unresolved.append("LAND_RESTRICTION_UNRESOLVED")
    if region.get("transport_diameter_hard_scenario_gate", False):
        if any(c.code == "REGIONAL_TRANSPORT_DIAMETER" and c.outcome == "UNKNOWN" for c in region_audit):
            unresolved.append("REGIONAL_TRANSPORT_DIAMETER")
    if region.get("site_transport_hard_scenario_gate", False):
        transport = next(c for c in site_audit if c.code == "SITE_TRANSPORT")
        if transport.outcome == "REVIEW":
            return "REJECTED", [transport.code]
        if transport.outcome == "UNKNOWN":
            unresolved.append(transport.code)
    if unresolved:
        return "NEEDS_HARD_DATA", unresolved
    return "ELIGIBLE_FOR_SCENARIOS", []


def competition_ranks(rows):
    """Equal scenario scores share ranks. A tie can cross the top-k boundary."""
    rank = 0
    previous = None
    ranked = sorted((r for r in rows if r["utility"] is not None),
                    key=lambda r: -r["utility"])
    for i, row in enumerate(ranked, start=1):
        if row["utility"] != previous:
            rank = i
            previous = row["utility"]
        row["rank"] = rank
    if ranked:
        best = ranked[0]["utility"]
        for row in ranked:
            row["utility_loss"] = round(best - row["utility"], 10)


def summarize(candidates, outcomes, n_scenarios, min_feasible):
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[tuple(row[key] for key in KEY)].append(row)
    summary = []
    for candidate in candidates:
        rows = grouped[tuple(candidate[key] for key in KEY)]
        ranked = [row for row in rows if row["rank"] is not None]
        top1 = sum(row["rank"] == 1 for row in ranked)
        top3 = sum(row["rank"] <= 3 for row in ranked)
        feasible = sum(row["passes_scenario_filter"] for row in rows)
        status = candidate["static_status"]
        if status == "ELIGIBLE_FOR_SCENARIOS":
            if len(ranked) / n_scenarios >= min_feasible and ranked:
                status = "SCENARIO_SHORTLIST"
            elif any(row["outcome"] in {"NEEDS_SOFT_DATA", "NEEDS_COST_DATA"} for row in rows):
                status = "NEEDS_RANKING_DATA"
            else:
                status = "BELOW_MIN_SCENARIO_SHARE"
        summary.append({
            **candidate, "status": status, "n_scenarios": n_scenarios,
            "feasible_share": feasible / n_scenarios, "ranked_share": len(ranked) / n_scenarios,
            "p_top1": top1 / n_scenarios, "p_top3": top3 / n_scenarios,
            "mean_loss_when_ranked": float(np.mean([r["utility_loss"] for r in ranked])) if ranked else None,
        })
    return sorted(summary, key=lambda r: (
        r["status"] != "SCENARIO_SHORTLIST", -r["p_top1"], -r["p_top3"], -r["ranked_share"],
        r["site_id"], r["reactor"], r["modules"],
    ))


def analyze_city(region, sites, reactors, scenarios, *, history=None, history_context=None,
                 estimates=None, imputation="median", seed=42, min_feasible=0.8):
    if not scenarios or not 0 < min_feasible <= 1:
        raise ValueError("Provide scenarios and a minimum scenario share in (0, 1]")
    if imputation not in CITY_METHODS:
        raise ValueError(f"Unknown imputation method: {imputation!r}")
    screen = region.get("scenario_screen", {})
    low = positive_number(screen.get("min_fraction_of_min_demand", 0.7), "demand lower factor")
    high = positive_number(screen.get("max_fraction_of_max_demand", 1.3), "demand upper factor")
    if low > high:
        raise ValueError("Demand lower factor must not exceed upper factor")
    hard_site = region.get("site_transport_hard_scenario_gate", False)
    if not isinstance(hard_site, bool):
        raise ValueError("site_transport_hard_scenario_gate must be boolean")
    if region.get("transport_diameter_hard_scenario_gate") and number(region.get("max_transport_diameter_m")) is None:
        raise ValueError("Hard regional transport gate requires a numeric regional limit")
    horizon = positive_number(region.get("scenario_analysis_years", 40), "scenario_analysis_years")
    if not horizon.is_integer() or horizon > 100:
        raise ValueError("scenario_analysis_years must be an integer in 1..100")
    profiles = {name: cost_profile(name) for name in sorted({s.cost_profile for s in scenarios})}
    assumptions = {name: replace(profile[0], max_operation_years=int(horizon))
                   for name, profile in profiles.items()}
    city_bounds = {"min": min(s.demand_mw for s in scenarios), "max": max(s.demand_mw for s in scenarios)}
    city_region = {**region, "demand_mw": city_bounds}
    options, regional_audit, bounds = region_options(city_region, reactors)
    by_option = defaultdict(list)
    for check in regional_audit:
        by_option[check.reactor, check.modules].append(check)

    rubrics, modeling = {}, {}
    use_rubric = any(s.weights[2] > 0 for s in scenarios)
    for site in sorted(sites, key=lambda s: s["site_id"]):
        sid = site["site_id"]
        if use_rubric:
            # Keep one site's draws comparable when another site is masked or removed.
            site_key = int.from_bytes(hashlib.sha256(sid.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(np.random.SeedSequence([seed, site_key]))
            rubrics[sid], modeling[sid] = scenario_site_scores(
                site, history, imputation, len(scenarios), rng, history_context=history_context)
        else:
            rubrics[sid], modeling[sid] = None, {"status": "ZERO_WEIGHT"}
    audit = [asdict(check) for check in regional_audit]
    studies = []
    candidates = []
    powers = {r["reactor"]: positive_number(r["P_el_MW"], "per-module electrical power") for r in reactors}
    for site in sites:
        sid = site["site_id"]
        for option in options:
            reactor, modules = option["reactor"], option["modules"]
            checks = by_option[reactor, modules]
            if option["region_screen_status"] == "SCENARIO_REJECTED":
                state = "REJECTED"
                reasons = [check.code for check in checks if check.outcome == "SCENARIO_FAIL"]
            else:
                local, requests = site_checks(site, option, city_region)
                audit.extend(asdict(check) for check in local)
                state, reasons = city_prechecks(region, site, checks, local)
                for request in requests:
                    studies.append({"site_id": sid, "reactor": reactor, "modules": modules, **asdict(request)})
            candidates.append({
                "site_id": sid, "site_name": site["name"], "reactor": reactor, "modules": modules,
                "electrical_mw": powers[reactor] * modules, "static_status": state,
                "static_reasons": "|".join(reasons),
            })
    outcomes = []
    for i, scenario in enumerate(scenarios):
        rows = []
        for candidate in candidates:
            row = {"scenario_id": scenario.scenario_id, **{key: candidate[key] for key in KEY},
                   "demand_mw": scenario.demand_mw, "passes_scenario_filter": False,
                   "outcome": candidate["static_status"], "rank": None, "utility": None, "utility_loss": None,
                   "lcoe_rub_kwh": None, "npv_bln": None, "site_rubric_0_1": None,
                   "cost_source": None}
            rows.append(row)
            if candidate["static_status"] != "ELIGIBLE_FOR_SCENARIOS":
                continue
            plant = candidate["electrical_mw"]
            if not low * scenario.demand_mw <= plant <= high * scenario.demand_mw:
                row["outcome"] = "DEMAND_WINDOW_FAIL"
                continue
            row["passes_scenario_filter"] = True
            rubric = rubrics[candidate["site_id"]]
            if rubric is None and scenario.weights[2] > 0:
                row["outcome"] = "NEEDS_SOFT_DATA"
                continue
            costs = (None, None, f"scenario_proxy:{scenario.cost_profile}") if estimates is None else estimates.get(
                tuple(candidate[key] for key in KEY))
            if costs is None:
                row["outcome"] = "NEEDS_COST_DATA"
                continue
            result = cashflow(powers[candidate["reactor"]], candidate["modules"],
                              scenario.demand_mw, scenario.tariff_rub_per_kwh, int(horizon),
                              assumptions[scenario.cost_profile], capex_factor=scenario.capex_factor,
                              opex_factor=scenario.opex_factor, capital_cost_bln=costs[0],
                              operating_cost_bln_year=costs[1])
            lcoe = result["lcoe_rub_kwh"]
            cost_utility = scenario.tariff_rub_per_kwh / (scenario.tariff_rub_per_kwh + lcoe)
            fit_utility = min(plant / scenario.demand_mw, scenario.demand_mw / plant)
            rubric_utility = float(rubric[i]) if rubric is not None else None
            reward = sum(w * value for w, value in zip(scenario.weights, (
                cost_utility, fit_utility, rubric_utility if rubric_utility is not None else 0,
            )))
            row.update(outcome="RANKED", utility=round(reward, 10), lcoe_rub_kwh=lcoe,
                       npv_bln=result["npv_bln"], site_rubric_0_1=rubric_utility, cost_source=costs[2])
        competition_ranks(rows)
        outcomes.extend(rows)
    summary = summarize(candidates, outcomes, len(scenarios), min_feasible)
    shortlist = [row for row in summary if row["status"] == "SCENARIO_SHORTLIST"]
    leaders = [row for row in shortlist if row["p_top1"] == shortlist[0]["p_top1"]] if shortlist else []
    report = {
        "region": region["name"], "region_id": region["region_id"], "city": sites[0]["city"],
        "site_count": len(sites), "candidate_count": len(summary), "scenario_count": len(scenarios),
        "scenarios_without_ranked_candidates": len(scenarios) - len({
            row["scenario_id"] for row in outcomes if row["rank"] is not None}),
        "shortlist_count": len(shortlist), "seed": seed, "min_ranked_share": min_feasible,
        "imputation_mode": imputation,
        "screening_envelope_mw": list(bounds), "analysis_horizon_years_assumed": int(horizon),
        "status_counts": dict(Counter(row["status"] for row in summary)),
        "result": ("MODEL_LEADER" if len(leaders) == 1 else "TIED_MODEL_LEADERS"
                   if leaders else "NO_SCENARIO_SHORTLIST"),
        "leaders": leaders, "site_rubric_model": modeling,
        "cost_mode": "SCENARIO_PROXY" if estimates is None else "SUPPLIED_ESTIMATES_ONLY",
        "cost_profiles": {name: profile[1] for name, profile in profiles.items()},
        "scenario_interpretation": "Equal weight per supplied scenario; ordinal rubric estimates are model inputs.",
        "scope": "Preliminary comparison of modeled alternatives. Cost anchors and demand need local validation.",
    }
    return summary, outcomes, audit, studies, report


def add_arguments(parser):
    parser.add_argument("--region", type=Path, default=DEMO / "region.json")
    parser.add_argument("--city", required=True)
    parser.add_argument("--site-id")
    parser.add_argument("--sites", type=Path, default=DEMO / "sites.csv")
    parser.add_argument("--reactors", type=Path, default=ROOT / "data" / "reactors.csv")
    parser.add_argument("--scenarios", type=Path, default=DEMO / "scenarios.csv")
    parser.add_argument("--history", type=Path, default=ROOT / "data" / "sites.csv")
    parser.add_argument("--cost-estimates", type=Path)
    parser.add_argument("--imputation", choices=CITY_METHODS, default="median")
    parser.add_argument("--min-feasible", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)


def run_cli(args, parser):
    try:
        region, sites, reactors = read_inputs(args.region, args.sites, args.reactors)
        sites = local_sites(sites, args.city, args.site_id)
        scenarios = read_scenarios(args.scenarios, region["region_id"], args.city)
        history = read_history(args.history) if args.imputation != "none" else None
        history_context = read_history_context(args.history) if args.imputation == "context-ridge" else None
        estimates = (read_cost_estimates(args.cost_estimates, sites, reactors, region["max_modules"])
                     if args.cost_estimates else None)
        ranking, outcomes, audit, studies, report = analyze_city(
            region, sites, reactors, scenarios, history=history, history_context=history_context,
            estimates=estimates, imputation=args.imputation,
            seed=args.seed, min_feasible=args.min_feasible)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    inputs = {"region": args.region, "sites": args.sites, "reactors": args.reactors, "scenarios": args.scenarios}
    if history is not None:
        inputs["history"] = args.history
    if args.cost_estimates:
        inputs["cost_estimates"] = args.cost_estimates
    report["inputs"] = {name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in inputs.items()}
    report["scenarios"] = [asdict(scenario) for scenario in scenarios]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "ranking.csv", ranking, list(ranking[0]))
    write_csv(args.out / "scenario_outcomes.csv", outcomes, list(outcomes[0]))
    write_csv(args.out / "constraint_audit.csv", audit, list(audit[0]))
    write_csv(args.out / "study_requests.csv", studies,
              ["site_id", "reactor", "modules", "code", "label", "trigger", "classification"])
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("city", "scenario_count", "result", "leaders")},
                     ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run_cli(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
