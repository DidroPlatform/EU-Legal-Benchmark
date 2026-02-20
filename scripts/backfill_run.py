from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import yaml

import run as run_module
from src.config import BenchmarkConfig
from src.io.json_io import read_json, read_jsonl
from src.runtime.bootstrap import load_dotenv_if_available


TargetKey = Tuple[str, str, str]


def collect_backfill_targets(
    base_outputs_dir: Path,
    *,
    include_failed_generation: bool,
    include_parse_errors: bool,
    include_empty_responses: bool,
) -> Set[TargetKey]:
    targets: Set[TargetKey] = set()

    if include_failed_generation:
        summary = read_json(base_outputs_dir / "summary.json")
        for item in summary.get("failed_items", []):
            dataset = item.get("dataset")
            example_id = item.get("example_id")
            candidate_name = item.get("candidate_name")
            if isinstance(dataset, str) and isinstance(example_id, str) and isinstance(candidate_name, str):
                targets.add((dataset, example_id, candidate_name))

    if include_parse_errors:
        for row in read_jsonl(base_outputs_dir / "judgments.jsonl"):
            if row.get("parse_error"):
                targets.add((row["dataset"], row["example_id"], row["candidate_name"]))

    if include_empty_responses:
        for row in read_jsonl(base_outputs_dir / "responses.jsonl"):
            text = row.get("response_text")
            if text is None or not str(text).strip():
                targets.add((row["dataset"], row["example_id"], row["candidate_name"]))

    return targets


def _targets_by_dataset(targets: Iterable[TargetKey]) -> Dict[str, Set[str]]:
    by_dataset: Dict[str, Set[str]] = defaultdict(set)
    for dataset, example_id, _ in targets:
        by_dataset[dataset].add(example_id)
    return by_dataset


def _target_candidates(targets: Iterable[TargetKey]) -> Set[str]:
    return {candidate_name for _, _, candidate_name in targets}


def create_filtered_datasets(
    *,
    run_config: Dict[str, Any],
    dataset_to_example_ids: Dict[str, Set[str]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_dataset_configs: List[Dict[str, Any]] = []

    for dataset_cfg in run_config.get("datasets", []):
        dataset_name = dataset_cfg.get("name")
        source_path = dataset_cfg.get("path")
        if not isinstance(dataset_name, str) or not isinstance(source_path, str):
            continue
        wanted_ids = dataset_to_example_ids.get(dataset_name, set())
        if not wanted_ids:
            continue

        source = Path(source_path)
        out_path = output_dir / f"{dataset_name}.jsonl"
        selected = 0
        with open(source, "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get("id")
                if row_id in wanted_ids:
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    selected += 1

        if selected == 0:
            continue
        new_cfg = dict(dataset_cfg)
        new_cfg["path"] = str(out_path)
        new_cfg["enabled"] = True
        new_cfg["limit"] = None
        filtered_dataset_configs.append(new_cfg)

    return filtered_dataset_configs


def build_backfill_config_dict(
    *,
    source_config_path: Path,
    base_run_config: Dict[str, Any],
    selected_candidates: Sequence[str],
    filtered_datasets: Sequence[Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    with open(source_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    candidates = [
        c for c in cfg.get("candidates", []) if isinstance(c, dict) and c.get("name") in set(selected_candidates)
    ]
    cfg["candidates"] = candidates
    cfg["data"] = cfg.get("data", {})
    cfg["data"]["datasets"] = list(filtered_datasets)
    cfg["run"] = cfg.get("run", {})
    cfg["run"]["run_id"] = run_id
    cfg["run"]["runs_root"] = base_run_config.get("run", {}).get("runs_root", cfg["run"].get("runs_root", "data/runs"))
    cfg["run"]["output_dir"] = base_run_config.get("run", {}).get("output_dir", cfg["run"].get("output_dir", "outputs"))
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted benchmark backfill for a prior run.")
    parser.add_argument("--config", default="config.yaml", help="Primary benchmark config path.")
    parser.add_argument("--base-run-id", required=True, help="Existing run ID to inspect and repair.")
    parser.add_argument("--include-failed-generation", action="store_true")
    parser.add_argument("--include-parse-errors", action="store_true")
    parser.add_argument("--include-empty-responses", action="store_true")
    parser.add_argument("--progress", choices=["log", "off"], default="log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.include_failed_generation or args.include_parse_errors or args.include_empty_responses):
        raise SystemExit("No backfill selectors provided. Set at least one --include-* flag.")

    base_run_dir = Path("data/runs") / args.base_run_id
    base_outputs = base_run_dir / "outputs"
    base_run_config = read_json(base_outputs / "run_config.json")
    targets = collect_backfill_targets(
        base_outputs,
        include_failed_generation=args.include_failed_generation,
        include_parse_errors=args.include_parse_errors,
        include_empty_responses=args.include_empty_responses,
    )
    if not targets:
        print("No affected items found for the requested selectors.")
        return

    dataset_to_example_ids = _targets_by_dataset(targets)
    candidates = sorted(_target_candidates(targets))
    backfill_run_id = f"{args.base_run_id}_backfill_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    with tempfile.TemporaryDirectory(prefix="benchmark-backfill-") as td:
        tmp_dir = Path(td)
        filtered_datasets = create_filtered_datasets(
            run_config=base_run_config,
            dataset_to_example_ids=dataset_to_example_ids,
            output_dir=tmp_dir / "datasets",
        )
        if not filtered_datasets:
            raise SystemExit("No examples matched source datasets. Nothing to run.")

        backfill_cfg_dict = build_backfill_config_dict(
            source_config_path=Path(args.config),
            base_run_config=base_run_config,
            selected_candidates=candidates,
            filtered_datasets=filtered_datasets,
            run_id=backfill_run_id,
        )
        cfg_path = tmp_dir / "backfill_config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(backfill_cfg_dict, f, sort_keys=False)

        load_dotenv_if_available()
        config = BenchmarkConfig.from_yaml(str(cfg_path))
        output_dir = run_module.run(config=config, progress_mode=args.progress)

    print(f"Backfill run completed: {output_dir.parent}")
    print(
        f"Targeted items: {len(targets)} | candidates: {len(candidates)} | "
        f"example_ids: {sum(len(v) for v in dataset_to_example_ids.values())}"
    )


if __name__ == "__main__":
    main()
