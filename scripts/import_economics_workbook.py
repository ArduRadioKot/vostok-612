"""Extract a traceable single-case economics snapshot from the user-provided workbook.

Usage: python scripts/import_economics_workbook.py '/path/to/Большой камень вся экономика.xlsx'
Requires artifact_tool ONLY for the XLSX import. No arbitrary cost item inference.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from artifact_tool import Blob, SpreadsheetFile


def parse_billion(value: object) -> float:
    text = str(value).strip().replace('\u00a0', ' ')
    m = re.search(r'[-+]?\d+(?:[.,]\d+)?', text)
    if not m:
        raise ValueError(f'Expected an amount in billion RUB, got {value!r}')
    return float(m.group(0).replace(',', '.'))


def import_workbook(path: Path, output: Path) -> dict:
    workbook = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    sheet = workbook.worksheets.get_item_at(0)
    matrix = sheet.get_range('A1:H10').values
    capex = [dict(source_cell=f'C{row+1}', name=str(matrix[row][0]).strip(),
                  value_bln_rub=parse_billion(matrix[row][2])) for row in range(2, 9)]
    opex = [dict(source_cell=f'H{row+1}', name=str(matrix[row][5]).strip().replace('\n', ' '),
                 value_bln_rub_year=parse_billion(matrix[row][7])) for row in range(2, 7)]
    capex_total = parse_billion(matrix[9][2])
    opex_total = parse_billion(matrix[7][7])
    if abs(sum(x['value_bln_rub'] for x in capex) - capex_total) > 1e-8:
        raise ValueError('CAPEX line items do not match source total')
    if abs(sum(x['value_bln_rub_year'] for x in opex) - opex_total) > 1e-8:
        raise ValueError('OPEX line items do not match source total')
    depreciation = [x for x in opex if 'амортизац' in x['name'].casefold()]
    if len(depreciation) != 1:
        raise ValueError('Cannot uniquely identify non-cash depreciation')
    result = {
        'source_file': path.name, 'source_sheet': sheet.name,
        'data_status': 'unverified_project_estimate',
        'applicability': {'site': 'Большой камень (Приморский край)', 'reactor': 'СВБР-100',
                          'modules': 1, 'power_mw': 100},
        'capex_items': capex, 'opex_items': opex,
        'capex_total_bln_rub': capex_total,
        'gross_opex_bln_rub_year': opex_total,
        'noncash_depreciation_bln_rub_year': depreciation[0]['value_bln_rub_year'],
        'cash_opex_bln_rub_year': round(opex_total-depreciation[0]['value_bln_rub_year'], 6),
        'quality_flags': [
            'Workbook is a project estimate, not a verified contract or plant operating data.',
            'Source column B5 says "порт Корсаков", but description and project site say Большой Камень; resolve location discrepancy.',
            'Line items include nuclear fuel-cycle treatment; confirm cost allocation and whether its investment is in CAPEX.',
            'Do not include noncash depreciation in cash OPEX when CAPEX is already counted.',
            'Amounts apply to the specified pilot case only; scaling to other sites/reactors is hypothetical.'
        ]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'economics_bigstone_2026.json')
    args=parser.parse_args()
    print(json.dumps(import_workbook(args.workbook, args.output), ensure_ascii=False, indent=2))
