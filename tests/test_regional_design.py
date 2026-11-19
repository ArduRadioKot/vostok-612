"""Regression checks for the additive region/site pipeline; run with stdlib unittest."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from copy import deepcopy

from nuclear_dss.regional_design import (
    is_missing, parse_dimension_diameter, parse_footprint_m2,
    read_inputs, region_options, site_checks, build_pipeline, write_outputs,
    generate_geojson,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "regional_demo"


class RegionalDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.region, cls.sites, cls.reactors = read_inputs(
            BASE / "region.json", BASE / "sites.csv", BASE / "reactors_subset.csv"
        )

    def test_catalog_dims_are_parsed_as_second_dimension(self):
        self.assertEqual(parse_dimension_diameter("6,53м/4,4м"), 4.4)
        self.assertEqual(parse_dimension_diameter("24м/3,5м"), 3.5)
        self.assertIsNone(parse_dimension_diameter("нет"))
        self.assertIsNone(parse_dimension_diameter("2-3м"))

    def test_ambiguous_footprint_is_not_invented(self):
        self.assertIsNone(parse_footprint_m2("менее 65000 м²"))
        self.assertIsNone(parse_footprint_m2("1800-27000 м²"))
        self.assertEqual(parse_footprint_m2("34000м²"), 34000)

    def test_region_retains_multiple_variants(self):
        variants, checks, window = region_options(self.region, self.reactors)
        retained = {(r['reactor'], r['modules']) for r in variants if r['region_screen_status'] != 'SCENARIO_REJECTED'}
        self.assertIn(('СВБР-100', 1), retained)
        self.assertIn(('СВБР-100', 2), retained)
        self.assertIn(('SEALER-55', 2), retained)
        self.assertNotIn(('4S', 1), retained)  # outside synthetic power scenario
        self.assertTrue(checks)

    def test_unknown_diameter_is_not_made_up_or_excluded(self):
        variants, checks, _ = region_options(self.region, self.reactors)
        r = next(x for x in variants if x['reactor'] == 'Aurora' and x['modules'] == 1)
        self.assertIsNone(r['reactor_diameter_m_catalog'])
        self.assertEqual(r['region_screen_status'], 'STAGE1_RETAINED_UNVERIFIED')
        self.assertTrue(any(c.reactor == 'Aurora' and c.code == 'REGIONAL_TRANSPORT_DIAMETER'
                            and c.outcome == 'UNKNOWN' for c in checks))

    def test_soft_soil_creates_study_not_mandatory_piles(self):
        opts, _, _ = region_options(self.region, self.reactors)
        opt = next(x for x in opts if x['reactor'] == 'СВБР-100' and x['modules'] == 1)
        checks, adaptations = site_checks(self.sites[1], opt, self.region)
        self.assertTrue(any(a.code == 'GEOTECH_STUDY' and 'свайные' in a.label for a in adaptations))
        self.assertTrue(any(c.code == 'GROUND_CONDITIONS' and c.outcome == 'REVIEW' for c in checks))
        self.assertFalse(any('сваи обязательны' in a.label.lower() for a in adaptations))

    def test_illustrative_land_exclusion_is_not_confirmed(self):
        opts, _, _ = region_options(self.region, self.reactors)
        opt = next(x for x in opts if x['reactor'] == 'СВБР-100' and x['modules'] == 1)
        checks, _ = site_checks(self.sites[3], opt, self.region)
        self.assertTrue(any(c.code == 'REPORTED_LAND_EXCLUSION' and c.outcome == 'REVIEW' for c in checks))
        documented = dict(self.sites[3], source_status='documented', source_reference='source:demo:example')
        checks2, _ = site_checks(documented, opt, self.region)
        self.assertTrue(any(c.code == 'REPORTED_LAND_EXCLUSION' and c.outcome == 'SCENARIO_FAIL' for c in checks2))

    def test_report_never_claims_training_or_approval(self):
        result = build_pipeline(self.region, self.sites, self.reactors)
        report = result[-1]
        self.assertFalse(report['trained_ml_model'])
        self.assertTrue(report['all_options_unverified'])
        self.assertEqual(report['n_local_sites'], 4)
        self.assertGreater(report['n_site_configuration_combinations'], 0)
        self.assertTrue(all(x['data_status'] == 'REQUIRES_VERIFICATION' for x in result[1]))

    def test_writes_machine_readable_artifacts(self):
        result = build_pipeline(self.region, self.sites, self.reactors)
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            write_outputs(out, self.sites, result)
            for file in ('report.json', 'region_options.csv', 'site_configuration_options.csv',
                         'constraint_audit.csv', 'research_queue.csv', 'sites.geojson', 'SUMMARY.md'):
                self.assertTrue((out / file).exists(), file)
            geojson = json.loads((out / 'sites.geojson').read_text(encoding='utf-8'))
            self.assertEqual(geojson['features'], []) # no fictional coordinates
            with (out / 'constraint_audit.csv').open(encoding='utf-8-sig') as f:
                self.assertTrue(any(c['outcome'] == 'UNKNOWN' for c in csv.DictReader(f)))

    def test_known_geo_coordinates_work_but_only_if_provided(self):
        s = dict(self.sites[0], latitude='3.4', longitude='5.6')
        geo = generate_geojson([s], [])
        self.assertEqual(geo['features'][0]['geometry']['coordinates'], [5.6, 3.4])

    def test_wrong_region_and_invalid_evidence_fails_validation(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            with (BASE/'region.json').open(encoding='utf-8') as f:
                r = json.load(f)
            r['source_status'] = 'documented'
            r['source_reference'] = ''
            (d/'region.json').write_text(json.dumps(r), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source_reference'):
                read_inputs(d/'region.json', BASE/'sites.csv', BASE/'reactors_subset.csv')

    def test_absolute_transport_gate_is_explicit_and_only_scenario(self):
        region = deepcopy(self.region)
        region['transport_diameter_hard_scenario_gate'] = True
        opts, checks, _ = region_options(region, self.reactors)
        self.assertEqual(next(x for x in opts if x['reactor'] == 'ARC-100' and x['modules'] == 1)['region_screen_status'],
                         'SCENARIO_REJECTED')
        self.assertTrue(any(c.reactor == 'ARC-100' and c.outcome == 'SCENARIO_FAIL' and
                            c.code == 'REGIONAL_TRANSPORT_DIAMETER' for c in checks))


if __name__ == '__main__':
    unittest.main()
