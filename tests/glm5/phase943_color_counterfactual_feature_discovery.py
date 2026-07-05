#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase941_color_feature_neuron_atlas as phase941  # noqa: E402
from phase942_color_counterfactual_closure import build_counterfactual_samples  # noqa: E402


PHASE = 943
RESULT_ROOT = Path("tests/result/phase943_color_counterfactual_feature_discovery")


def phase943_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = build_counterfactual_samples(args)
    for sample in samples:
        sample["phase"] = PHASE
        sample["relation"] = "counterfactual_color_feature_discovery"
    return samples


def rewrite_jsonl_phase(old_path: Path, new_path: Path) -> None:
    rows = []
    if old_path.exists():
        for line in old_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["phase"] = PHASE
            rows.append(row)
    phase941.write_jsonl(new_path, rows)


def rewrite_outputs(out_dir: Path, model_name: str) -> dict[str, str]:
    old_prefix = f"phase941_{model_name}"
    new_prefix = f"phase943_{model_name}"
    summary_old = out_dir / f"{old_prefix}_summary.json"
    summary_new = out_dir / f"{new_prefix}_summary.json"
    if summary_old.exists():
        payload = json.loads(summary_old.read_text(encoding="utf-8"))
        payload["phase"] = PHASE
        payload["title"] = "Counterfactual Color Feature Discovery"
        payload["dataset_type"] = "factorial explicit color x neutral object"
        payload["boundary"] = (
            "This phase discovers channels on counterfactual explicit-color prompts; "
            "compare with Phase 941 typical-object color channels."
        )
        phase941.write_json(summary_new, payload)

    suffixes = [
        "dataset",
        "sample_rows",
        "channel_sample_rows",
        "channel_rows",
        "intervention_rows",
    ]
    for suffix in suffixes:
        rewrite_jsonl_phase(out_dir / f"{old_prefix}_{suffix}.jsonl", out_dir / f"{new_prefix}_{suffix}.jsonl")

    for path in out_dir.glob(f"{old_prefix}_*.json*"):
        path.unlink()

    return {
        "summary": str(summary_new),
        "dataset": str(out_dir / f"{new_prefix}_dataset.jsonl"),
        "channel_rows": str(out_dir / f"{new_prefix}_channel_rows.jsonl"),
        "intervention_rows": str(out_dir / f"{new_prefix}_intervention_rows.jsonl"),
    }


def scan_model(args: argparse.Namespace) -> dict[str, Any]:
    phase941.PHASE = PHASE
    phase941.RESULT_ROOT = RESULT_ROOT
    phase941.build_color_samples = phase943_samples
    payload = phase941.scan_model(args)
    out_dir = RESULT_ROOT / args.round_name
    rewritten = rewrite_outputs(out_dir, args.model)
    print(json.dumps({"phase": PHASE, "model": args.model, "rewritten": rewritten}, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=phase941.MODELS, default="qwen3")
    parser.add_argument("--round-name", default="color_counterfactual_feature_discovery")
    parser.add_argument("--colors", default="")
    parser.add_argument("--objects", type=int, default=6)
    parser.add_argument("--templates-per-object", type=int, default=3)
    parser.add_argument("--domains", default="")
    parser.add_argument("--max-objects-per-color", type=int, default=0)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--topk-blockers", type=int, default=10)
    parser.add_argument("--keep-top-channels-per-sample", type=int, default=128)
    parser.add_argument("--keep-channel-rows", type=int, default=20000)
    parser.add_argument("--keep-channel-sample-rows", type=int, default=50000)
    parser.add_argument("--summary-top-channels", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-interventions", action="store_true")
    parser.add_argument("--intervention-top-channels", type=int, default=4)
    parser.add_argument("--max-intervention-specs", type=int, default=24)
    parser.add_argument("--max-intervention-samples-per-spec", type=int, default=4)
    parser.add_argument("--intervention-factors", default="0.0,2.0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scan_model(args)


if __name__ == "__main__":
    main()
