#!/usr/bin/env python3
"""Freeze the Phase423 Jacobian-lens observer qualification protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase423_workspace_observer_qualification"
OFFICIAL_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
OFFICIAL_REPOSITORY = "https://github.com/anthropics/jacobian-lens"
PAPER_URL = "https://transformer-circuits.pub/2026/workspace/index.html"
MODELS = ("qwen3", "glm4", "deepseek7b")
EVALUATIONS = (
    "lens-eval-association",
    "lens-eval-multihop",
    "lens-eval-multilingual",
    "lens-eval-typo",
)
FIT_PROMPTS = 100
FIT_CANDIDATES = 500
EVAL_ITEMS_PER_DISTRIBUTION = 48
SOURCE_LAYER_COUNT = 9


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return digest_text(encoded)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def load_fit_prompts() -> list[dict[str, Any]]:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in dataset:
        text = str(record["text"]).strip()
        if len(text) < 600:
            continue
        text_hash = digest_text(text)
        if text_hash in seen:
            continue
        seen.add(text_hash)
        candidates.append((text_hash, text))
        if len(candidates) == FIT_CANDIDATES:
            break
    if len(candidates) != FIT_CANDIDATES:
        raise RuntimeError(
            f"Expected {FIT_CANDIDATES} long WikiText candidates, got {len(candidates)}"
        )
    selected = sorted(candidates, key=lambda item: item[0])[:FIT_PROMPTS]
    rows: list[dict[str, Any]] = []
    for index, (text_hash, text) in enumerate(selected):
        rows.append(
            {
                "prompt_id": f"wikitext_{index:03d}",
                "split": "fit_a" if index % 2 == 0 else "fit_b",
                "source": "Salesforce/wikitext:wikitext-103-raw-v1:train",
                "selection": "sha256_lowest_100_of_first_500_records_ge_600_chars",
                "text_sha256": text_hash,
                "text": text,
            }
        )
    return rows


def load_evaluations(official_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name in EVALUATIONS:
        path = official_root / "data/evaluations" / f"{dataset_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing official evaluation file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = []
        for item in payload["items"]:
            item_hash = digest_json(item)
            candidates.append((item_hash, item))
        selected = sorted(candidates, key=lambda value: value[0])[
            :EVAL_ITEMS_PER_DISTRIBUTION
        ]
        for index, (item_hash, item) in enumerate(selected):
            rows.append(
                {
                    "evaluation_id": f"{dataset_name}_{index:03d}",
                    "dataset": dataset_name,
                    "split": "calibration" if index % 2 == 0 else "holdout",
                    "official_item_sha256": item_hash,
                    "name": item["name"],
                    "prompt": item["prompt"],
                    "target": item.get("target"),
                    "intermediates": item["intermediates"],
                    "readout_position": -1,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("tests/gpt5_temp/phase423_vendor/jacobian-lens"),
    )
    parser.add_argument("--reuse-frozen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = OUT / "phase423_protocol.json"
    fit_path = OUT / "phase423_fit_prompts.jsonl"
    eval_path = OUT / "phase423_evaluation_items.jsonl"
    if args.reuse_frozen and all(path.exists() for path in (protocol_path, fit_path, eval_path)):
        print(protocol_path)
        return

    official_root = args.official_root.resolve()
    fit_rows = load_fit_prompts()
    eval_rows = load_evaluations(official_root)
    protocol = {
        "schema_version": "phase423_workspace_observer_protocol.v1",
        "phase": 423,
        "frozen_at": now(),
        "scientific_role": "observer_qualification",
        "compute_edge_claim_allowed": False,
        "causal_claim_allowed": False,
        "workspace_claim_allowed": False,
        "paper_url": PAPER_URL,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "models_in_execution_order": list(MODELS),
        "fit_contract": {
            "corpus": "Salesforce/wikitext:wikitext-103-raw-v1:train",
            "candidate_records": FIT_CANDIDATES,
            "prompt_count": FIT_PROMPTS,
            "split_counts": {"fit_a": 50, "fit_b": 50},
            "max_seq_len": 64,
            "skip_first": 8,
            "source_layer_count": SOURCE_LAYER_COUNT,
            "source_layer_rule": "nine_even_relative_depths_below_final_target",
            "dtype": "bfloat16",
            "dim_batch_by_model": {"qwen3": 8, "glm4": 4, "deepseek7b": 8},
            "checkpoint_every": 10,
        },
        "evaluation_contract": {
            "datasets": list(EVALUATIONS),
            "items_per_dataset": EVAL_ITEMS_PER_DISTRIBUTION,
            "total_items": len(eval_rows),
            "split_rule": "sha256_sort_then_even_calibration_odd_holdout",
            "candidate_rule": "single_token_raw_space_lower_title_variants",
            "metric_rule": "minimum_full_vocabulary_rank_across_frozen_layers",
        },
        "frozen_gates": {
            "matrix_finite": True,
            "half_matrix_cosine_median_min": 0.80,
            "half_matrix_cosine_layer_min": 0.40,
            "half_relative_difference_median_max": 0.90,
            "eligible_intermediate_fraction_min": 0.80,
            "improved_fraction_each_split_min": 0.55,
            "mrr_ratio_each_split_min": 1.10,
            "pass_at_10_absolute_gain_each_split_min": 0.02,
            "passing_evaluation_datasets_each_split_min": 2,
            "cross_model_qualified_models_min": 2,
        },
        "authorization_rule": (
            "Only observer-qualified models may enter a later workspace-function "
            "audit. No Phase423 output authorizes workspace, compute-edge, causal, "
            "channel, or neuron claims."
        ),
        "fit_prompt_sha256": digest_json(fit_rows),
        "evaluation_sha256": digest_json(eval_rows),
    }
    write_jsonl(fit_path, fit_rows)
    write_jsonl(eval_path, eval_rows)
    write_json(protocol_path, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
