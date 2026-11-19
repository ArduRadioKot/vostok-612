"""Regression tests for city-scoped screening, ranking, and soft imputation."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from nuclear_dss.city_pipeline import (
    DEMO, ROOT, analyze_city, competition_ranks, local_sites, read_cost_estimates, read_scenarios, summarize,
)
from nuclear_dss.data import FEATURES
from nuclear_dss.economics import cashflow
from nuclear_dss.regional_design import read_inputs
from nuclear_dss.site_context import read_history_context, site_context
from nuclear_dss.site_imputation import cross_validate_imputation, predict_with_context, read_history, scenario_site_scores


class CityPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.region, cls.sites, cls.reactors = read_inputs(
            DEMO / "region.json", DEMO / "sites.csv", ROOT / "data" / "reactors.csv")
        cls.history = read_history(ROOT / "data" / "sites.csv")
        cls.city = "Учебный Северный"
        cls.selected = local_sites(cls.sites, cls.city)
        cls.scenarios = read_scenarios(DEMO / "scenarios.csv", cls.region["region_id"], cls.city)

    def test_scopes_sites_and_scenarios_to_requested_city(self):
        self.assertEqual({s["site_id"] for s in self.selected}, {"S1", "S2", "S3"})
        self.assertEqual(len(self.scenarios), 9)
        self.assertEqual(local_sites(self.sites, self.city, "S2")[0]["site_id"], "S2")
        with self.assertRaisesRegex(ValueError, "No declared candidate"):
            local_sites(self.sites, self.city, "S4")
        with self.assertRaisesRegex(ValueError, "No scenarios match"):
            read_scenarios(DEMO / "scenarios.csv", "Wrong region", self.city)

    def test_competition_ranking_preserves_ties_and_excludes_nonranked(self):
        rows = [{"utility": v, "rank": None} for v in [0.5, None, 0.8, 0.8, 0.4]]
        competition_ranks(rows)
        self.assertEqual([r["rank"] for r in rows], [3, None, 1, 1, 4])
        self.assertEqual(rows[0]["utility_loss"], 0.3)

    def test_top_k_denominator_includes_failed_and_empty_scenarios(self):
        candidate = dict(site_id="site", reactor="reactor", modules=1,
                         static_status="ELIGIBLE_FOR_SCENARIOS")
        rows = [
            {**candidate, "rank": rank, "passes_scenario_filter": rank is not None,
             "utility_loss": 0.05, "outcome": "RANKED" if rank is not None else "DEMAND_WINDOW_FAIL"}
            for rank in [1, 3, None, None]
        ]
        result = summarize([candidate], rows, 4, 0.5)[0]
        self.assertEqual((result["p_top1"], result["p_top3"], result["feasible_share"]), (0.25, 0.5, 0.5))
        self.assertEqual(result["status"], "SCENARIO_SHORTLIST")

    def test_missing_and_documented_exclusion_cannot_win(self):
        site = {**self.selected[0], "source_status": "documented", "source_reference": "test",
                "land_restriction": "reported_exclusion"}
        ranking, outcomes, _, _, report = analyze_city(
            self.region, [site, self.selected[2]], self.reactors, self.scenarios, history=self.history)
        self.assertTrue(all(row["rank"] is None for row in outcomes))
        self.assertIn("NEEDS_HARD_DATA", {r["static_status"] for r in ranking})
        self.assertIn("REPORTED_LAND_EXCLUSION", {r["static_reasons"] for r in ranking})
        self.assertEqual(report["scenarios_without_ranked_candidates"], len(self.scenarios))
        self.assertEqual(report["leaders"], [])

    def test_explicit_transport_gate_blocks_unknown_route(self):
        region = {**self.region, "site_transport_hard_scenario_gate": True}
        site = {**self.selected[0], "transport_max_diameter_m": ""}
        ranking, _, _, _, report = analyze_city(region, [site], self.reactors, self.scenarios,
                                                history=self.history)
        self.assertIn("SITE_TRANSPORT", {r["static_reasons"] for r in ranking})
        self.assertFalse(report["leaders"])

    def test_static_region_power_limit_applies_in_every_scenario(self):
        region = {**self.region, "max_total_capacity_mw": 50}
        _, outcomes, _, _, report = analyze_city(region, self.selected, self.reactors, self.scenarios,
                                                  history=self.history)
        self.assertFalse(any(r["passes_scenario_filter"] for r in outcomes))
        self.assertFalse(report["leaders"])

    def test_soft_estimates_only_in_rank_and_reproduce_seed(self):
        a = analyze_city(self.region, self.selected, self.reactors, self.scenarios, history=self.history)
        b = analyze_city(self.region, self.selected, self.reactors, self.scenarios, history=self.history)
        self.assertEqual(a, b)
        without = analyze_city(self.region, self.selected, self.reactors, self.scenarios, imputation="none")
        self.assertEqual([r["passes_scenario_filter"] for r in a[1]],
                         [r["passes_scenario_filter"] for r in without[1]])
        self.assertTrue(any(r["outcome"] == "NEEDS_SOFT_DATA" and r["site_id"] == "S2" for r in without[1]))
        self.assertTrue(any(r["outcome"] == "RANKED" and r["site_id"] == "S2" for r in a[1]))
        self.assertTrue(all(r["p_top1"] <= r["p_top3"] <= r["ranked_share"] <= r["feasible_share"] <= 1
                            for r in a[0]))
        self.assertEqual(a[4]["site_rubric_model"]["S2"]["missing"], ["economy"])
        self.assertEqual(a[4]["imputation_mode"], "median")
        self.assertEqual(a[4]["site_rubric_model"]["S2"]["selected_method"], "median")
        self.assertEqual(a[4]["site_rubric_model"]["S2"]["n_history_sites"], len(self.history))
        for scenario in self.scenarios:
            scores = {r["site_rubric_0_1"] for r in a[1] if
                      r["scenario_id"] == scenario.scenario_id and r["site_id"] == "S2" and r["rank"] is not None}
            self.assertLessEqual(len(scores), 1)

    def test_imputation_does_not_train_on_linked_target_site(self):
        name, scores = next(iter(self.history.items()))
        site = {**self.selected[0], "historical_site": name.upper(), **dict(zip(FEATURES, scores))}
        site[FEATURES[0]] = ""
        estimates, audit = scenario_site_scores(site, self.history, "bayesian-ridge", 3, np.random.default_rng(3))
        self.assertEqual(audit["n_history_sites"], len(self.history) - 1)
        self.assertTrue(((0 <= estimates) & (estimates <= 1)).all())
        self.assertEqual(audit["missing"], [FEATURES[0]])

    def test_nested_selector_does_not_use_outer_hidden_cells(self):
        rubrics = np.array(list(self.history.values())[:8])
        observed = np.array([True, False, True, False, True, True])
        changed = rubrics.copy()
        changed[0, ~observed] = 6 - rubrics[0, ~observed]
        before, _ = cross_validate_imputation(rubrics, observed, nested=True)
        after, _ = cross_validate_imputation(changed, observed, nested=True)
        np.testing.assert_allclose(rubrics[0, ~observed] - before["auto"][0],
                                   changed[0, ~observed] - after["auto"][0])

    def test_gpr_prediction_uses_only_observed_cells_in_held_out_site(self):
        rubrics = np.array(list(self.history.values())[:8])
        observed = np.array([True, False, True, False, True, True])
        changed = rubrics.copy()
        changed[0, ~observed] = 6 - rubrics[0, ~observed]
        before, _ = cross_validate_imputation(rubrics, observed, nested=False, include_gpr=True)
        after, _ = cross_validate_imputation(changed, observed, nested=False, include_gpr=True)
        predictions = rubrics[:, ~observed] - before["gpr"]
        self.assertEqual(predictions.shape, (8, 2))
        self.assertTrue(((1 <= predictions) & (predictions <= 5)).all())
        np.testing.assert_allclose(predictions[0], changed[0, ~observed] - after["gpr"][0])

    def test_context_features_enter_the_model_only_with_training_support(self):
        train = np.array([[3., 1.], [3., 2.], [3., 4.], [3., 5.]])
        context = [site_context({"population": n, "admin_region": "East"}) for n in (10, 20, 40, 50)]
        queries = np.array([[3., np.nan], [3., np.nan]])
        extra = [site_context({"population": 45}), site_context({"admin_region": "New"})]
        ridge, enriched, used = predict_with_context(train, queries, np.array([True, False]), context, extra)
        self.assertGreater(enriched[0, 0], ridge[0, 0])
        self.assertEqual(enriched[1, 0], ridge[1, 0])
        self.assertEqual(used, [{"population"}, set()])

    def test_context_uses_history_provenance_and_falls_back_for_unseen_regions(self):
        context = read_history_context(ROOT / "data" / "sites.csv")
        self.assertTrue(all("admin_region" in row for row in context.values()))
        site = {**self.selected[1], "admin_region": "Чукотский автономный округ"}
        _, model = scenario_site_scores(site, self.history, "context-ridge", 3, np.random.default_rng(3),
                                        history_context=context)
        self.assertEqual(model["context_features_used"], ["admin_region"])
        self.assertEqual(model["selected_method"], "context-ridge")
        _, fallback = scenario_site_scores({**site, "admin_region": "New"}, self.history,
                                           "context-ridge", 3, np.random.default_rng(3), history_context=context)
        self.assertEqual(fallback["selected_method"], "median")
        self.assertEqual(fallback["context_features_used"], [])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.csv"
            header = "site,population,context_source_status,context_source_reference\n"
            path.write_text(header + "demo,100,illustrative,\n", encoding="utf-8")
            self.assertEqual(read_history_context(path), {"demo": {}})
            path.write_text(header + "demo,100,documented,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires documented status and a source"):
                read_history_context(path)

    def test_explicit_cost_mode_uses_supplied_costs_and_reports_gaps(self):
        site = self.selected[0]
        estimates = {(site["site_id"], "ALFRED", 1): (90, 5, "synthetic test estimate")}
        ranking, outcomes, _, _, report = analyze_city(
            self.region, [site], self.reactors, self.scenarios, estimates=estimates)
        self.assertEqual(report["cost_mode"], "SUPPLIED_ESTIMATES_ONLY")
        self.assertEqual(len(report["leaders"]), 1)
        self.assertEqual(report["leaders"][0]["reactor"], "ALFRED")
        self.assertEqual({r["cost_source"] for r in outcomes if r["rank"] is not None},
                         {"synthetic test estimate"})
        self.assertTrue(any(r["outcome"] == "NEEDS_COST_DATA" for r in outcomes))
        row = next(r for r in outcomes if r["scenario_id"] == "S01" and r["rank"] is not None)
        independent = cashflow(125, 1, 105, 12, 40, capital_cost_bln=90, operating_cost_bln_year=5)
        self.assertAlmostEqual(row["lcoe_rub_kwh"], independent["lcoe_rub_kwh"])

    def test_rejects_invalid_and_duplicate_scenario_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.csv"
            header, first, *_ = (DEMO / "scenarios.csv").read_text().splitlines()
            path.write_text(f"{header}\n{first}\n{first}\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                read_scenarios(path, self.region["region_id"], self.city)
            path.write_text(f"{header}\n{first.rsplit(',', 1)[0]},0.9\n")
            with self.assertRaisesRegex(ValueError, "sum to one"):
                read_scenarios(path, self.region["region_id"], self.city)

    def test_cost_estimates_require_source_and_unique_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "costs.csv"
            header = "site_id,reactor,modules,capex_bln,cash_opex_bln_year,source_reference"
            row = "S1,ALFRED,1,90,5,Test source"
            path.write_text(f"{header}\n{row}\n")
            self.assertEqual(read_cost_estimates(path, self.selected, self.reactors, 3),
                             {("S1", "ALFRED", 1): (90, 5, "Test source")})
            with path.open("a") as stream:
                stream.write(row + "\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                read_cost_estimates(path, self.selected, self.reactors, 3)

    def test_cli_writes_outcomes_and_input_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            subprocess.run([sys.executable, "-B", "-m", "nuclear_dss", "city", "--city", self.city,
                            "--out", str(out)], cwd=ROOT, check=True, capture_output=True, text=True)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["scenario_count"], 9)
            self.assertEqual(len(report["inputs"]["scenarios"]["sha256"]), 64)
            self.assertEqual(len(report["leaders"]), 2)
            self.assertTrue((out / "scenario_outcomes.csv").is_file())

    def test_cli_uses_numeric_context_from_documented_training_rows(self):
        import csv

        history = ROOT / "data" / "sites.csv"
        with tempfile.TemporaryDirectory() as tmp, history.open(encoding="utf-8-sig", newline="") as stream:
            tmp = Path(tmp)
            reader = csv.DictReader(stream)
            context_fields = ["population", "context_source_status", "context_source_reference"]
            with (tmp / "history.csv").open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=[*reader.fieldnames, *context_fields])
                writer.writeheader()
                for index, row in enumerate(reader, start=1):
                    writer.writerow({**row, "population": index * 1000, "context_source_status": "documented",
                                     "context_source_reference": "synthetic test fixture"})
            header = list(self.sites[0])
            with (tmp / "sites.csv").open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=[*header, "population"])
                writer.writeheader()
                writer.writerows([{**site, "population": 5000} for site in self.sites])
            subprocess.run([sys.executable, "-B", "-m", "nuclear_dss", "city", "--city", self.city,
                            "--history", str(tmp / "history.csv"), "--sites", str(tmp / "sites.csv"),
                            "--imputation", "context-ridge", "--out", str(tmp / "out")],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            report = json.loads((tmp / "out" / "report.json").read_text())
            self.assertEqual(report["site_rubric_model"]["S2"]["context_features_used"], ["population"])

    def test_cli_runs_gpr_for_one_missing_rubric(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            subprocess.run([sys.executable, "-B", "-m", "nuclear_dss", "city", "--city", self.city,
                            "--imputation", "gpr", "--out", str(out)],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            report = json.loads((out / "report.json").read_text())
            model = report["site_rubric_model"]["S2"]
            self.assertEqual(model["selected_method"], "gpr")
            self.assertEqual(model["validation"], "leave_one_site_out")
            self.assertEqual(list(model["estimate"]), ["economy"])
            self.assertAlmostEqual(model["held_out_site_mae"], model["fixed_method_mae"]["gpr"])
            self.assertEqual(report["site_rubric_model"]["S1"]["status"], "OBSERVED")


if __name__ == "__main__":
    unittest.main()
