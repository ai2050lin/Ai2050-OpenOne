#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase266_multi_family_baseline_behavior_readout_scan as p266  # noqa: E402

PHASE = 293
SCHEMA_VERSION = "2.20.0"
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase293_expanded_queue_behavior_readout_runner"
MODELS = ["qwen3", "glm4", "deepseek7b"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def case_from_queue(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "family_id": row["family_id"],
        "mode_id": row.get("mode_id") or "phase292_queue",
        "variant_id": row["variant_id"],
        "variant_type": "phase291_expanded",
        "path_schema_id": f"phase291:path_schema:{row['family_id']}:{row.get('mode_id') or 'queue'}:{row['variant_id']}",
        "target": row["target"],
        "target_aliases": [row["target"]],
        "expected_pattern": row.get("expected_pattern") or "short",
        "output_protocol": row.get("expected_pattern") or "short",
        "boundary_type": "phase291_expanded",
        "continuation_trigger": row.get("channel_focus") or "unknown",
        "scoring_risk": "medium",
        "prompt": row["prompt"],
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    queue = [r for r in read_jsonl(V2 / "phase292_feature_priority_queue_rows.jsonl") if r.get("model") == args.model]
    queue.sort(key=lambda r: int(r.get("phase292_rank") or 10**9))
    if args.limit:
        queue = queue[: args.limit]
    out_dir = OUT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    model_obj = tokenizer = None
    try:
        attn_impls = args.attn_implementations
        if isinstance(attn_impls, list):
            attn_impls = ",".join(attn_impls)
        model_obj, tokenizer, device, _attn_impl = p266.p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, attn_impls
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, item in enumerate(queue, 1):
            case = case_from_queue(item)
            aliases = [str(case["target"])]
            readout = p266.capture_readout(model_obj, tokenizer, device, str(case["prompt"]), aliases)
            output, stopped, new_tokens = p266.generate_probe(model_obj, tokenizer, device, str(case["prompt"]), args.rollout_tokens)
            cls = p266.classify_output(output, case)
            base = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase293",
                "created_at": now(),
                "model": args.model,
                "case_id": case["case_id"],
                "family_id": case["family_id"],
                "mode_id": case["mode_id"],
                "variant_id": case["variant_id"],
                "path_schema_id": case["path_schema_id"],
                "target": case["target"],
                "expected_pattern": case["expected_pattern"],
                "channel_focus": item.get("channel_focus"),
                "phase292_rank": item.get("phase292_rank"),
                "atlas_completion_v2": item.get("atlas_completion_v2"),
            }
            behavior_rows.append(
                {
                    **base,
                    "behavior_id": f"phase293:behavior:{args.model}:{case['case_id']}",
                    **cls,
                    "model_stop_executed": stopped,
                    "generated_token_count": new_tokens,
                }
            )
            readout_rows.append(
                {
                    **base,
                    "readout_id": f"phase293:readout:{args.model}:{case['case_id']}",
                    **readout,
                }
            )
            if idx % args.log_every == 0:
                print(f"{args.model}: processed {idx}/{len(queue)}", flush=True)
    finally:
        del model_obj
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase293",
        "created_at": now(),
        "model": args.model,
        "rows": len(queue),
        "behavior_rows": len(behavior_rows),
        "readout_rows": len(readout_rows),
        "answer_correct_proxy_rate": mean_safe([1.0 if r.get("answer_correct_proxy") else 0.0 for r in behavior_rows]),
        "pattern_matched_proxy_rate": mean_safe([1.0 if r.get("pattern_matched_proxy") else 0.0 for r in behavior_rows]),
        "model_stop_executed_rate": mean_safe([1.0 if r.get("model_stop_executed") else 0.0 for r in behavior_rows]),
        "top_continue_channel_counts": dict(Counter(str(r.get("top_continue_channel")) for r in readout_rows)),
        "competition_winner_counts": dict(Counter(str(r.get("competition_winner")) for r in readout_rows)),
        "mean_top_continue_vs_stop_margin": mean_safe([safe_float(r.get("top_continue_vs_stop_margin")) for r in readout_rows]),
        "status": "complete",
    }
    write_jsonl(out_dir / f"phase293_{args.model}_expanded_behavior_rows.jsonl", behavior_rows)
    write_jsonl(out_dir / f"phase293_{args.model}_expanded_readout_rows.jsonl", readout_rows)
    write_json(out_dir / f"phase293_{args.model}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rollout-tokens", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--round-name", default="expanded_queue_behavior_readout")
    parser.add_argument("--attn-implementations", nargs="*", default=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()
    run_model(args)


if __name__ == "__main__":
    main()
