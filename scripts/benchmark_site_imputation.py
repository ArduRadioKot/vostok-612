"""Compare recovery of hidden rubric cells and their original manual ranks."""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from nuclear_dss.data import FEATURES
from nuclear_dss.regional_design import parse_dimension_diameter
from nuclear_dss.site_imputation import METHODS, cross_validate_imputation, site_scores

MANUAL_FINALISTS = ("СВБР-100", "ARC-100", "SEALER-55", "OTRERA 300")


def replay_manual_screen(path: Path) -> list[dict]:
    """Replay only the numeric filters documented for the four finalists."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        catalog = {row["reactor"]: row for row in csv.DictReader(stream)}
    results = []
    for name in MANUAL_FINALISTS:
        row = catalog[name]
        power = float(row["P_el_MW"])
        diameter = parse_dimension_diameter(row["dims"])
        results.append({
            "reactor": name, "one_module_power_mw": power, "catalog_diameter_m": diameter,
            "passes_50_to_150_mw": 50 <= power <= 150,
            "passes_4_5_m": None if diameter is None else diameter <= 4.5,
        })
    return results


def verify_manual_workbook(csv_path: Path, workbook_path: Path) -> dict:
    from openpyxl import load_workbook

    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        history = {row["site"].strip(): tuple(float(row[name]) for name in (*FEATURES, "expert_score"))
                   for row in csv.DictReader(stream)}
    book = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        data = book.worksheets[0].iter_rows(min_row=2, max_col=8, values_only=True)
        manual = {str(name).strip(): tuple(values) for name, *values in data
                  if name and all(isinstance(value, (int, float)) for value in values)}
    finally:
        book.close()
    if history != manual:
        raise ValueError("The CSV does not match all sites and six scores from the manual workbook")
    return {"file": workbook_path.name, "matching_sites": len(manual), "fields_checked_per_site": 7}


def benchmark(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    names = [row["site"].strip() for row in raw]
    rubrics = np.stack([site_scores(row) for row in raw])
    manual = np.array([float(row["expert_score"]) for row in raw])
    if not all(names) or len(set(name.casefold() for name in names)) != len(raw) or len(raw) < 4:
        raise ValueError("At least four uniquely named sites are required")
    if not np.array_equal(rubrics.sum(axis=1), manual):
        raise ValueError("The manual totals must match the complete six-criterion rubric")
    rank = 1 + (manual[None, :] > manual[:, None]).sum(axis=1)
    exact = [{"site": name, "manual_total": float(score), "manual_rank": int(place)}
             for name, score, place in zip(names, manual, rank)]
    summaries = []
    for missing_count in (1, 2):
        by_method = {name: [] for name in (*METHODS, "auto", "gpr")}
        for mask in combinations(range(len(FEATURES)), missing_count):
            observed = np.ones(len(FEATURES), dtype=bool)
            observed[list(mask)] = False
            errors, _ = cross_validate_imputation(rubrics, observed, nested=True, include_gpr=True)
            for method, residuals in errors.items():
                recovered = manual - residuals.sum(axis=1)
                greater = manual[None, :] > recovered[:, None] + 1e-10
                np.fill_diagonal(greater, False)
                recovered_rank = 1 + greater.sum(axis=1)
                by_method[method].append((
                    np.abs(residuals).mean(axis=1),
                    np.abs(recovered_rank - rank),
                    (recovered_rank <= 3) == (rank <= 3),
                ))
        trials = len(next(iter(by_method.values()))) * len(raw)
        summaries.append({
            "missing_criteria": missing_count, "site_mask_trials": trials,
            "metrics": {method: dict(zip(("cell_mae", "rank_mae", "top3_agreement"),
                                         np.asarray(values).mean(axis=(0, 2)).tolist()))
                        for method, values in by_method.items()},
        })
    return {"n_sites": len(raw), "manual_rank": sorted(exact, key=lambda r: r["manual_rank"]),
            "masked_validation": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/sites.csv"))
    parser.add_argument("--reactors", type=Path, default=Path("data/reactors.csv"))
    parser.add_argument("--manual-workbook", type=Path)
    args = parser.parse_args()
    report = benchmark(args.history)
    if args.manual_workbook:
        report["manual_workbook"] = verify_manual_workbook(args.history, args.manual_workbook)
    report["manual_finalist_screen"] = replay_manual_screen(args.reactors)
    print(json.dumps(report, indent=2, ensure_ascii=False))
