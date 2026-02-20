from __future__ import annotations

from typing import Any, Dict, List, Tuple


RowKey = Tuple[str, str, str]


def row_key(row: Dict[str, Any]) -> RowKey:
    return (str(row["dataset"]), str(row["example_id"]), str(row["candidate_name"]))


def overlay_rows(
    base_rows: List[Dict[str, Any]],
    patch_rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    merged: Dict[RowKey, Dict[str, Any]] = {row_key(row): row for row in base_rows}
    replaced = 0
    for row in patch_rows:
        key = row_key(row)
        replaced += 1 if key in merged else 0
        merged[key] = row
    ordered_keys = sorted(merged.keys())
    return [merged[key] for key in ordered_keys], replaced
