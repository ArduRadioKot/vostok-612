"""ВОСТОК-612: regional constraints -> configurations -> site studies.

Independent, stdlib-only exploratory extension for the existing nuclear_dss package.
NO regulatory assessment, reactor safety assessment, siting authorization, or design certification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

OUTCOMES = {"SCENARIO_PASS", "SCENARIO_FAIL", "REVIEW", "UNKNOWN"}
MISSING = {"", "нет", "нкт", "none", "null", "nan", "unknown", "?", "n/a", "неизвестно"}
TRUTHY = {"true", "yes", "да", "1"}
FALSEY = {"false", "no", "нет", "0"}
ALLOWED_SOURCE_STATUS = {"illustrative", "documented", "unknown"}


def is_missing(v: Any) -> bool:
    return v is None or str(v).strip().casefold() in MISSING


def number(v: Any) -> float | None:
    if is_missing(v):
        return None
    try:
        x = float(str(v).strip().replace(",", "."))
    except (ValueError, TypeError):
        raise ValueError(f"Expected a numeric value, got {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"Non-finite numeric value: {v!r}")
    return x


def positive_number(v: Any, field: str, *, allow_zero: bool = False) -> float:
    x = number(v)
    if x is None or (x < 0 if allow_zero else x <= 0):
        raise ValueError(f"{field} must be {'nonnegative' if allow_zero else 'positive'}")
    return x


def parse_dimension_diameter(dims: Any) -> float | None:
    """Parse only clear 'height/diameter' entries from v4 catalog.

    This is a parsing convention, NOT confirmation that the number is the
    transport package's effective maximum cross-section.
    """
    if is_missing(dims):
        return None
    s = str(dims).lower().replace(" ", "").replace(",", ".")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)м?/(\d+(?:\.\d+)?)м?", s)
    return float(m.group(2)) if m else None


def parse_footprint_m2(value: Any) -> float | None:
    if is_missing(value):
        return None
    s = str(value).lower().replace(" ", "").replace(",", ".")
    # Ambiguous ranges and 'less than' do not yield a single reliable value.
    if "-" in s or "менее" in s or "<" in s or ">" in s:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:м²|м2)?", s)
    return float(m.group(1)) if m else None


@dataclass(frozen=True)
class Check:
    stage: str
    site_id: str
    reactor: str
    modules: int
    code: str
    outcome: str
    evidence: str
    explanation: str
    action: str = ""

    def __post_init__(self):
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Unexpected check outcome {self.outcome}")


@dataclass(frozen=True)
class Adaptation:
    code: str
    label: str
    trigger: str
    classification: str = "INVESTIGATE_ONLY"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Duplicate columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows: {path}")
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"CSV rows must match the header width: {path}")
    return rows


def read_inputs(region_path: Path, sites_path: Path, reactors_path: Path):
    with region_path.open("r", encoding="utf-8") as f:
        region = json.load(f)
    if not isinstance(region, dict):
        raise ValueError("region JSON must be an object")
    for field in ("region_id", "name", "demand_mw", "max_modules"):
        if field not in region:
            raise ValueError(f"Region missing {field}")
    for field in ("region_id", "name"):
        if not isinstance(region[field], str) or not region[field].strip():
            raise ValueError(f"Region {field} must be a nonempty string")
    dm = region["demand_mw"]
    if not isinstance(dm, dict):
        raise ValueError("demand_mw must be an object with min and max")
    dmin = positive_number(dm.get("min"), "demand_mw.min")
    dmax = positive_number(dm.get("max"), "demand_mw.max")
    if dmin > dmax:
        raise ValueError("min demand > max demand")
    max_modules = region["max_modules"]
    if type(max_modules) is not int or not 1 <= max_modules <= 12:
        raise ValueError("max_modules must be integer in 1..12")
    source_status = region.get("source_status", "unknown")
    if source_status not in ALLOWED_SOURCE_STATUS:
        raise ValueError("Invalid region source_status")
    if source_status == "documented" and is_missing(region.get("source_reference")):
        raise ValueError("Documented region requires source_reference")
    bounds = region.get("scenario_screen", {})
    if not isinstance(bounds, dict):
        raise ValueError("scenario_screen must be an object")
    positive_number(bounds.get("min_fraction_of_min_demand", 0.7), "min_fraction_of_min_demand")
    positive_number(bounds.get("max_fraction_of_max_demand", 1.3), "max_fraction_of_max_demand")
    for opt in ("max_total_capacity_mw", "max_transport_diameter_m"):
        if not is_missing(region.get(opt)):
            positive_number(region[opt], opt)
    if not isinstance(region.get("transport_diameter_hard_scenario_gate", False), bool):
        raise ValueError("transport_diameter_hard_scenario_gate must be boolean")

    sites = read_csv(sites_path)
    reactors = read_csv(reactors_path)
    for col in ("site_id", "region_id", "name", "source_status", "source_reference"):
        if any(col not in s for s in sites):
            raise ValueError(f"sites.csv missing {col}")
    for col in ("reactor", "P_el_MW", "dims", "footprint"):
        if any(col not in r for r in reactors):
            raise ValueError(f"reactors.csv missing {col}")
    ids = [s["site_id"].strip() for s in sites]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("site_id must be nonempty and unique")
    names = [r["reactor"].strip() for r in reactors]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("reactor names must be nonempty and unique")
    for s in sites:
        if s["region_id"] != region["region_id"]:
            raise ValueError(f"Site {s['site_id']} belongs to different region")
        if s["source_status"] not in ALLOWED_SOURCE_STATUS:
            raise ValueError(f"Bad source_status at {s['site_id']}")
        if s["source_status"] == "documented" and is_missing(s["source_reference"]):
            raise ValueError(f"Documented site {s['site_id']} lacks source_reference")
        for field in ("transport_max_diameter_m", "available_land_m2", "local_demand_min_mw", "local_demand_max_mw"):
            if not is_missing(s.get(field)):
                positive_number(s[field], f"{s['site_id']}.{field}")
        if not is_missing(s.get("local_demand_min_mw")) and not is_missing(s.get("local_demand_max_mw")):
            if number(s["local_demand_min_mw"]) > number(s["local_demand_max_mw"]):
                raise ValueError(f"Local demand min > max: {s['site_id']}")
        for axis, limit in (("latitude", 90), ("longitude", 180)):
            if not is_missing(s.get(axis)) and abs(number(s[axis])) > limit:
                raise ValueError(f"Invalid {axis} for {s['site_id']}")
        for field, choices in {
            "water_access": {"yes", "no", "unknown", ""},
            "seismic_hazard": {"low", "moderate", "elevated", "unknown", ""},
            "ground_conditions": {"rock", "mixed", "soft", "permafrost", "unknown", ""},
            "flood_hazard": {"low", "elevated", "unknown", ""},
            "land_restriction": {"none", "reported_exclusion", "unknown", ""},
        }.items():
            if s.get(field, "").strip().lower() not in choices:
                raise ValueError(f"Invalid {field} for site {s['site_id']}")
    for r in reactors:
        positive_number(r["P_el_MW"], f"{r['reactor']}.P_el_MW")
    return region, sites, reactors


def mkcheck(stage, site, reactor, modules, code, result, text, action="", evidence="unknown"):
    return Check(stage, site, reactor, modules, code, result, evidence, text, action)


def region_options(region: dict, reactors: list[dict]):
    """Stage I: broad design-space screen; no siting/safety approval inferred."""
    screen = region.get("scenario_screen", {})
    dmin = number(region["demand_mw"]["min"])
    dmax = number(region["demand_mw"]["max"])
    low = dmin * number(screen.get("min_fraction_of_min_demand", 0.7))
    high = dmax * number(screen.get("max_fraction_of_max_demand", 1.3))
    capacity_limit = number(region.get("max_total_capacity_mw"))
    diameter_limit = number(region.get("max_transport_diameter_m"))
    hard_transport = region.get("transport_diameter_hard_scenario_gate", False)
    results, checks = [], []
    for r in reactors:
        power = number(r["P_el_MW"])
        diameter = parse_dimension_diameter(r.get("dims"))
        footprint = parse_footprint_m2(r.get("footprint"))
        for n in range(1, region["max_modules"] + 1):
            total = n * power
            local = []
            within = low <= total <= high
            local.append(mkcheck("region", "", r["reactor"], n, "POWER_SCENARIO_WINDOW",
                "SCENARIO_PASS" if within else "SCENARIO_FAIL",
                f"Illustrative window [{low:g}, {high:g}] MWe; option={total:g} MWe.",
                "Recheck with measured site demand/load curve.", region.get("source_status", "unknown")))
            if capacity_limit is not None:
                local.append(mkcheck("region", "", r["reactor"], n, "MAX_TOTAL_CAPACITY",
                    "SCENARIO_PASS" if total <= capacity_limit else "SCENARIO_FAIL",
                    f"Scenario ceiling {capacity_limit:g} MWe; option={total:g} MWe.",
                    "Verify load-following, grid stability and exact limit.", region.get("source_status", "unknown")))
            if diameter_limit is not None:
                if diameter is None:
                    local.append(mkcheck("region", "", r["reactor"], n, "REGIONAL_TRANSPORT_DIAMETER",
                        "UNKNOWN", "Reactor package diameter absent or ambiguous in catalog.",
                        "Request approved lifting/shipping package drawings.", "catalog_unknown"))
                else:
                    exceeded = diameter > diameter_limit
                    result = ("SCENARIO_FAIL" if hard_transport else "REVIEW") if exceeded else "SCENARIO_PASS"
                    local.append(mkcheck("region", "", r["reactor"], n, "REGIONAL_TRANSPORT_DIAMETER",
                        result, f"Catalog diameter {diameter:g} m vs illustrative limit {diameter_limit:g} m; actual shipping envelope unverified.",
                        "Study alternative transport routes and actual freight envelope." if exceeded else "Verify bridge/loading/clearance constraints.",
                        "illustrative_catalog"))
            if footprint is None:
                local.append(mkcheck("region", "", r["reactor"], n, "NOMINAL_FOOTPRINT", "UNKNOWN",
                    "Comparable per-module footprint not available from catalog.",
                    "Request layout envelope incl. support infrastructure.", "catalog_unknown"))
            checks.extend(local)
            status = "SCENARIO_REJECTED" if any(c.outcome == "SCENARIO_FAIL" for c in local) else "STAGE1_RETAINED_UNVERIFIED"
            results.append({"reactor":r["reactor"], "modules":n, "electrical_mw":round(total, 3),
                 "reactor_diameter_m_catalog":diameter, "nominal_footprint_m2_catalog":footprint,
                 "region_screen_status":status, "region_review_count":sum(c.outcome == "REVIEW" for c in local),
                 "region_unknown_count":sum(c.outcome == "UNKNOWN" for c in local)})
    return results, checks, (low, high)


ADAPTATIONS = {
 "GEOTECH_STUDY": "Геотехнические изыскания; рассмотреть типы фундаментов, включая свайные (без предписания свай)",
 "SEISMIC_STUDY": "Площадочная оценка сейсмической опасности и динамические расчёты; рассмотреть сейсмоизоляцию",
 "FLOOD_STUDY": "Проверка затопления, паводков и требований к отметкам/защитным системам",
 "HEAT_SINK_STUDY": "Расчёт источника охлаждения; изучить альтернативную конфигурацию теплоотвода",
 "TRANSPORT_STUDY": "Инженерное исследование маршрута тяжёлого/негабаритного груза",
 "LAND_STUDY": "Полный генплан станции и территориально-правовые ограничения",
 "LOAD_STUDY": "Почасовой/сезонный график нагрузок и моделирование энергосистемы",
 "LICENSING_STUDY": "Проверка регуляторного статуса технологии, проекта, разрешений и безопасности",
}


def site_checks(site: dict, option: dict, region: dict):
    """Stage II: tri-state screening with non-prescriptive study flags."""
    name, n = option["reactor"], option["modules"]
    sid = site["site_id"]
    evidence = site["source_status"]
    output, adaptations = [], set()
    def add(code, outcome, explanation, action="", evidence_override=None):
        output.append(mkcheck("site", sid, name, n, code, outcome, explanation,
                              action, evidence if evidence_override is None else evidence_override))
    # Reported exclusions never become regulatory facts just because a CSV says so.
    legal = site.get("land_restriction", "unknown").strip().lower()
    if legal == "reported_exclusion":
        if evidence == "documented":
            add("REPORTED_LAND_EXCLUSION", "SCENARIO_FAIL", "Input asserts a land-use exclusion with a reference, not independently validated.",
                "Review authoritative zoning/permit sources with competent specialists.")
        else:
            add("REPORTED_LAND_EXCLUSION", "REVIEW", "Unverified land exclusion claim in scenario input.",
                "Confirm protection status and legal grounds.")
        adaptations.add("LAND_STUDY")
    elif legal == "none":
        add("LAND_RESTRICTIONS", "REVIEW", "Scenario reports no known restrictions; absence of constraints is not established.",
            "Check land rights, licensing setbacks, ecology and exclusion requirements.")
        adaptations.add("LAND_STUDY")
    else:
        add("LAND_RESTRICTIONS", "UNKNOWN", "Land-use restrictions are unknown.", ADAPTATIONS["LAND_STUDY"])
        adaptations.add("LAND_STUDY")
    # Local demand, where supplied, is a heuristic compatibility window, not engineering feasibility.
    dmin = number(site.get("local_demand_min_mw"))
    dmax = number(site.get("local_demand_max_mw"))
    if dmin is None or dmax is None:
        add("LOCAL_LOAD_PROFILE", "UNKNOWN", "No local minimum/maximum demand envelope.", ADAPTATIONS["LOAD_STUDY"])
        adaptations.add("LOAD_STUDY")
    else:
        min_frac = number(region.get("scenario_screen", {}).get("min_fraction_of_min_demand", 0.7))
        max_frac = number(region.get("scenario_screen", {}).get("max_fraction_of_max_demand", 1.3))
        c = option["electrical_mw"]
        if c < min_frac*dmin or c > max_frac*dmax:
            add("LOCAL_LOAD_PROFILE", "REVIEW", f"Option {c:g} MWe outside illustrative local window [{min_frac*dmin:g}, {max_frac*dmax:g}] MWe.", ADAPTATIONS["LOAD_STUDY"])
            adaptations.add("LOAD_STUDY")
        else:
            add("LOCAL_LOAD_PROFILE", "SCENARIO_PASS", "Within illustrative local demand envelope, without power-system stability analysis.", ADAPTATIONS["LOAD_STUDY"])
    sd = number(site.get("transport_max_diameter_m"))
    rd = option["reactor_diameter_m_catalog"]
    if sd is None or rd is None:
        add("SITE_TRANSPORT", "UNKNOWN", "Route clearance or transport-package dimensions are unknown.", ADAPTATIONS["TRANSPORT_STUDY"])
        adaptations.add("TRANSPORT_STUDY")
    elif rd > sd:
        add("SITE_TRANSPORT", "REVIEW", f"Catalog diameter {rd:g} m exceeds scenario route envelope {sd:g} m.", ADAPTATIONS["TRANSPORT_STUDY"])
        adaptations.add("TRANSPORT_STUDY")
    else:
        add("SITE_TRANSPORT", "SCENARIO_PASS", "Catalog diameter within illustrative route limit; transport engineering still necessary.", ADAPTATIONS["TRANSPORT_STUDY"])
    water = site.get("water_access", "unknown").strip().lower()
    if water == "no":
        add("HEAT_SINK", "REVIEW", "Conventional water access not indicated. No validated alternative heat sink.", ADAPTATIONS["HEAT_SINK_STUDY"])
        adaptations.add("HEAT_SINK_STUDY")
    elif water == "yes":
        add("HEAT_SINK", "REVIEW", "Water access in dataset does not establish adequate heat-sink performance.", ADAPTATIONS["HEAT_SINK_STUDY"])
        adaptations.add("HEAT_SINK_STUDY")
    else:
        add("HEAT_SINK", "UNKNOWN", "Water supply/heat sink unavailable in data.", ADAPTATIONS["HEAT_SINK_STUDY"])
        adaptations.add("HEAT_SINK_STUDY")
    seismic = site.get("seismic_hazard", "unknown").strip().lower()
    if seismic == "elevated":
        add("SEISMIC_HAZARD", "REVIEW", "Scenario labels seismic hazard as elevated. No site-specific hazard analysis.", ADAPTATIONS["SEISMIC_STUDY"])
        adaptations.add("SEISMIC_STUDY")
    elif seismic in {"low", "moderate"}:
        add("SEISMIC_HAZARD", "REVIEW", "Coarse seismic category is not an engineering design basis.", ADAPTATIONS["SEISMIC_STUDY"])
        adaptations.add("SEISMIC_STUDY")
    else:
        add("SEISMIC_HAZARD", "UNKNOWN", "Seismic hazard not characterized.", ADAPTATIONS["SEISMIC_STUDY"])
        adaptations.add("SEISMIC_STUDY")
    ground = site.get("ground_conditions", "unknown").strip().lower()
    if ground in {"soft", "permafrost", "mixed"}:
        add("GROUND_CONDITIONS", "REVIEW", f"Scenario indicates {ground} ground; foundation type not determined.", ADAPTATIONS["GEOTECH_STUDY"])
        adaptations.add("GEOTECH_STUDY")
    elif ground == "rock":
        add("GROUND_CONDITIONS", "REVIEW", "Coarse rock label does not replace geotechnical investigation.", ADAPTATIONS["GEOTECH_STUDY"])
        adaptations.add("GEOTECH_STUDY")
    else:
        add("GROUND_CONDITIONS", "UNKNOWN", "Geotechnical conditions unavailable.", ADAPTATIONS["GEOTECH_STUDY"])
        adaptations.add("GEOTECH_STUDY")
    flood = site.get("flood_hazard", "unknown").strip().lower()
    if flood == "elevated":
        add("FLOOD_HAZARD", "REVIEW", "Coarse flood risk elevated; design-basis event unknown.", ADAPTATIONS["FLOOD_STUDY"])
        adaptations.add("FLOOD_STUDY")
    elif flood == "low":
        add("FLOOD_HAZARD", "REVIEW", "Scenario low flood category has not been verified by assessment.", ADAPTATIONS["FLOOD_STUDY"])
        adaptations.add("FLOOD_STUDY")
    else:
        add("FLOOD_HAZARD", "UNKNOWN", "No flood hazard characterization.", ADAPTATIONS["FLOOD_STUDY"])
        adaptations.add("FLOOD_STUDY")
    area = number(site.get("available_land_m2"))
    footprint = option["nominal_footprint_m2_catalog"]
    if area is None or footprint is None:
        add("PLOT_ENVELOPE", "UNKNOWN", "Site area or comparable reactor catalog footprint unknown.", ADAPTATIONS["LAND_STUDY"])
        adaptations.add("LAND_STUDY")
    elif area < footprint * n:
        add("PLOT_ENVELOPE", "REVIEW", f"Available area {area:g} m2 < nominal {footprint*n:g} m2 (catalog-only, excludes site support facilities).", ADAPTATIONS["LAND_STUDY"])
        adaptations.add("LAND_STUDY")
    else:
        add("PLOT_ENVELOPE", "REVIEW", "Nominal catalog footprint fits, but excludes shared plant footprint and site-specific setbacks.", ADAPTATIONS["LAND_STUDY"])
        adaptations.add("LAND_STUDY")
    add("LICENSING_AND_SAFETY", "UNKNOWN", "No licensing, nuclear safety, emergency planning or external hazards review provided.", ADAPTATIONS["LICENSING_STUDY"], "not_supplied")
    adaptations.add("LICENSING_STUDY")
    return output, [Adaptation(code, ADAPTATIONS[code], sid) for code in sorted(adaptations)]


def build_pipeline(region: dict, sites: list[dict], reactors: list[dict]):
    region_options_list, first_checks, bounds = region_options(region, reactors)
    kept = [x for x in region_options_list if x["region_screen_status"] != "SCENARIO_REJECTED"]
    combined, checks, research = [], list(first_checks), []
    for s in sites:
        for option in kept:
            local, studies = site_checks(s, option, region)
            checks.extend(local)
            failed = any(c.outcome == "SCENARIO_FAIL" for c in local)
            unknown = sum(c.outcome == "UNKNOWN" for c in local)
            review = sum(c.outcome == "REVIEW" for c in local)
            # Comparison index is a triage aid only: not a suitability score.
            priority = round(100 - min(100, review*5 + unknown*8 + option["region_review_count"]*4 + option["region_unknown_count"]*6), 1)
            state = "REPORTED_EXCLUSION_CHECK" if failed else ("REVIEW_REQUIRED" if review else "DATA_GAPS")
            combined.append({"region_id":region["region_id"], "site_id":s["site_id"], "site_name":s["name"],
                             "reactor":option["reactor"], "modules":option["modules"],
                             "electrical_mw":option["electrical_mw"],
                             "catalog_diameter_m":option["reactor_diameter_m_catalog"],
                             "region_screen_status":option["region_screen_status"],
                             "site_screen_status":state, "unresolved_checks":unknown,
                             "review_checks":review, "triage_index_not_suitability":priority,
                             "studies_json":json.dumps([asdict(a) for a in studies], ensure_ascii=False),
                             "data_status":"REQUIRES_VERIFICATION"})
            for ch in local:
                if ch.outcome in {"UNKNOWN", "REVIEW", "SCENARIO_FAIL"} and ch.action:
                    weight = {"LICENSING_AND_SAFETY":10,"REPORTED_LAND_EXCLUSION":10,
                              "HEAT_SINK":9,"SEISMIC_HAZARD":9,"GROUND_CONDITIONS":8,
                              "SITE_TRANSPORT":7,"LOCAL_LOAD_PROFILE":7,"PLOT_ENVELOPE":6,
                              "FLOOD_HAZARD":8,"LAND_RESTRICTIONS":8}.get(ch.code,5)
                    research.append({"site_id":s["site_id"],"reactor":option["reactor"],"modules":option["modules"],
                                     "check_code":ch.code,"outcome":ch.outcome,"priority_heuristic":weight,
                                     "request":ch.action, "evidence":ch.evidence})
    combined.sort(key=lambda x:(x["site_screen_status"] == "REPORTED_EXCLUSION_CHECK", -x["triage_index_not_suitability"], x["site_id"],x["reactor"],x["modules"]))
    research.sort(key=lambda x:(-x["priority_heuristic"], x["site_id"], x["reactor"], x["modules"]))
    report = {
        "pipeline":"vostok-612-regional-design", "version":__version__,
        "region_id":region["region_id"],"region_name":region["name"],
        "region_source_status":region.get("source_status","unknown"),
        "scenario_capacity_window_mw":list(bounds),
        "n_technologies":len(reactors),"n_local_sites":len(sites),
        "n_stage1_options":len(region_options_list),
        "n_stage1_retained":len(kept),"n_site_configuration_combinations":len(combined),
        "site_screen_counts":dict(Counter(x["site_screen_status"] for x in combined)),
        "all_options_unverified":True,
        "trained_ml_model":False,
        "method":"Deterministic interpretable constraint screening; missingness-aware study queue; NO learned ML model.",
        "disclaimer":"Exploratory scenario analysis only. Unknowns are never passed. Region and site data are not validated. No nuclear licensing, geology, safety or construction recommendation. Triage index is NOT site suitability, probability, or validated engineering score.",
        "limitations":["Catalog diameter is parsed from height/diameter text, NOT a certified shipping envelope.",
            "Catalog footprint is not an entire plant site footprint.",
            "No water/heat-sink, seismic, site geology, legal or emergency-planning calculations.",
            "Any conditional engineering measure is a study candidate, NOT a required/final design.",
            "Illustrative and documented source labels are user-supplied and not independently authenticated."],
    }
    return region_options_list, combined, checks, research, report


def write_csv(path: Path, records: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def generate_geojson(sites: list[dict], combinations: list[dict]):
    """Only emit points if coordinates are explicitly supplied in input."""
    by_site = defaultdict(list)
    for row in combinations:
        by_site[row["site_id"]].append(row)
    features = []
    for s in sites:
        lat = number(s.get("latitude"))
        lon = number(s.get("longitude"))
        if lat is None or lon is None:
            continue
        rows = by_site[s["site_id"]]
        features.append({"type":"Feature", "geometry":{"type":"Point","coordinates":[lon,lat]},
            "properties":{"site_id":s["site_id"],"name":s["name"],"source_status":s["source_status"],
                "n_options":len(rows),"n_reported_exclusions":sum(x["site_screen_status"]=="REPORTED_EXCLUSION_CHECK" for x in rows),
                "note":"User-supplied approximate point, not a verified parcel or reactor siting recommendation."}})
    return {"type":"FeatureCollection", "features":features}


def write_outputs(out: Path, sites: list[dict], results):
    s1, combos, checks, queries, report = results
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"region_options.csv",s1,["reactor","modules","electrical_mw","reactor_diameter_m_catalog",
                    "nominal_footprint_m2_catalog","region_screen_status","region_review_count","region_unknown_count"])
    write_csv(out/"site_configuration_options.csv",combos,["region_id","site_id","site_name","reactor","modules",
                    "electrical_mw","catalog_diameter_m","region_screen_status","site_screen_status",
                    "unresolved_checks","review_checks","triage_index_not_suitability","studies_json","data_status"])
    write_csv(out/"constraint_audit.csv",[asdict(c) for c in checks],list(Check.__dataclass_fields__.keys()))
    write_csv(out/"research_queue.csv",queries,["site_id","reactor","modules","check_code","outcome","priority_heuristic","request","evidence"])
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"sites.geojson").write_text(json.dumps(generate_geojson(sites,combos),ensure_ascii=False,indent=2),encoding="utf-8")
    lines = ["# ВОСТОК-612 — региональный предпроектный скрининг", "",
             f"Регион: {report['region_name']} (`{report['region_id']}`); статус источника: `{report['region_source_status']}`.",
             f"Каталог: {report['n_technologies']} технологий; площадок: {report['n_local_sites']}.",
             f"Этап 1: {report['n_stage1_retained']} из {report['n_stage1_options']} вариантов оставлены по **сценарным** ограничениям.",
             f"Этап 2: {report['n_site_configuration_combinations']} сочетаний площадок и конфигураций.",
             "", "**Все варианты требуют верификации. Это не допуск к строительству и не подтверждение ядерной безопасности.**", "",
             "Точные основания решений: `constraint_audit.csv`. Перечень необходимых обследований: `research_queue.csv`.",
             "Показатель `triage_index_not_suitability` — лишь эвристика для очередности исследований, не качество площадки.", "",
             "## Примеры первых вариантов (не рейтинг пригодности)", ""]
    for row in combos[:10]:
        lines.append(f"- `{row['site_id']}` / {row['reactor']} × {row['modules']}: {row['site_screen_status']}; "
                     f"уточнить={row['unresolved_checks']}, исследовать={row['review_checks']}")
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv: list[str] | None = None):
    parser=argparse.ArgumentParser(description="Regional design-space screening (research only; NO nuclear siting approval)")
    parser.add_argument("--root",type=Path,default=Path.cwd(),help="Original vostok-612 checkout, or this extension's demo root")
    parser.add_argument("--region",type=Path,help="JSON region profile")
    parser.add_argument("--sites",type=Path,help="CSV with candidate local sites within the region")
    parser.add_argument("--reactors",type=Path,help="Existing v4 data/reactors.csv; fallback to demo subset")
    parser.add_argument("--out",type=Path,help="Output directory")
    args=parser.parse_args(argv)
    root=args.root.resolve()
    reg=args.region or root/"data/regional_demo/region.json"
    sites_path=args.sites or root/"data/regional_demo/sites.csv"
    reactors_path=args.reactors or (root/"data/reactors.csv" if (root/"data/reactors.csv").exists() else root/"data/regional_demo/reactors_subset.csv")
    out=args.out or root/"results/regional_demo"
    region, sites, reactors = read_inputs(reg, sites_path, reactors_path)
    results=build_pipeline(region,sites,reactors)
    write_outputs(out,sites,results)
    print(json.dumps({"inputs":{"region":str(reg),"sites":str(sites_path),"reactors":str(reactors_path)},
                      "out":str(out),"report":results[-1]},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
