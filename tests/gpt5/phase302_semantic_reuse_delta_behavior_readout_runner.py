#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
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


PHASE = "Phase302"
SCHEMA_VERSION = "2.29.0"
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase302_semantic_reuse_delta_behavior_readout"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "semantic_reuse_delta_behavior_readout"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def word_set(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff-]+", normalize(text)))


def contains_alias(output: str, aliases: list[str]) -> tuple[bool, str]:
    low = normalize(output)
    words = word_set(output)
    for alias in aliases:
        a = normalize(alias)
        if not a:
            continue
        if " " in a or "-" in a:
            if a in low:
                return True, alias
        elif a in words or a in low:
            return True, alias
    return False, ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def classify_output(output: str, case: dict[str, Any]) -> dict[str, Any]:
    aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
    hit, matched = contains_alias(output, aliases)
    tokens = output.replace("\n", " ").split()
    too_long = len(tokens) > 18
    has_drift = "answer:" in normalize(output) or "question:" in normalize(output) or too_long
    return {
        "alias_hit": hit,
        "matched_alias": matched,
        "answer_correct_proxy": hit,
        "pattern_matched_proxy": hit and not has_drift,
        "token_count": len(tokens),
        "has_drift_marker": has_drift,
        "output_preview": output[:300],
    }


def load_cases() -> list[dict[str, Any]]:
    return read_jsonl(V2 / "phase301_semantic_full_test_plan_rows.jsonl")


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    cases = [r for r in load_cases() if r.get("model") == args.model]
    cases.sort(key=lambda r: (str(r.get("case_type")), str(r.get("object_id")), str(r.get("prompt_type"))))
    if args.limit:
        cases = cases[: args.limit]
    out_dir = OUT / args.round_name
    behavior_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model_obj = tokenizer = None
    try:
        model_obj, tokenizer, device, _attn_impl = p266.p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, case in enumerate(cases, start=1):
            try:
                aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
                readout = p266.capture_readout(model_obj, tokenizer, device, str(case["prompt"]), aliases)
                output, stopped, new_tokens = p266.generate_probe(model_obj, tokenizer, device, str(case["prompt"]), args.rollout_tokens)
                cls = classify_output(output, case)
                base = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": args.model,
                    "case_id": case.get("case_id"),
                    "case_type": case.get("case_type"),
                    "object_id": case.get("object_id"),
                    "contrast_object_id": case.get("contrast_object_id"),
                    "object_label": case.get("object_label"),
                    "category_id": case.get("category_id"),
                    "subclass_id": case.get("subclass_id"),
                    "attribute_type": case.get("attribute_type"),
                    "prompt_type": case.get("prompt_type"),
                    "semantic_field": case.get("semantic_field"),
                    "target": case.get("target"),
                    "target_aliases": aliases,
                }
                behavior_rows.append(
                    {
                        **base,
                        "behavior_id": f"phase302:behavior:{args.model}:{case.get('case_id')}",
                        **cls,
                        "model_stop_executed": stopped,
                        "generated_token_count": new_tokens,
                    }
                )
                readout_rows.append(
                    {
                        **base,
                        "readout_id": f"phase302:readout:{args.model}:{case.get('case_id')}",
                        **readout,
                    }
                )
                rollout_rows.append(
                    {
                        **base,
                        "rollout_id": f"phase302:rollout:{args.model}:{case.get('case_id')}",
                        "generated_text": output[:500],
                        "generated_token_count": new_tokens,
                        "model_stop_executed": stopped,
                        "answer_correct_proxy": cls["answer_correct_proxy"],
                        "pattern_matched_proxy": cls["pattern_matched_proxy"],
                        "target_rank": readout.get("target_rank"),
                        "target_margin_vs_winner": readout.get("target_margin_vs_winner"),
                        "competition_winner": readout.get("competition_winner"),
                        "top_continue_channel": readout.get("top_continue_channel"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "missing_id": f"phase302:missing:{args.model}:{case.get('case_id')}",
                        "model": args.model,
                        "case_id": case.get("case_id"),
                        "reason": repr(exc),
                    }
                )
            if idx % args.log_every == 0:
                log(f"{args.model}: semantic rows {idx}/{len(cases)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p266.p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    summary = summarize_model(args.model, cases, behavior_rows, readout_rows, rollout_rows, missing_rows)
    write_json(out_dir / f"phase302_{args.model}_summary.json", summary)
    write_jsonl(out_dir / f"phase302_{args.model}_semantic_behavior_rows.jsonl", behavior_rows)
    write_jsonl(out_dir / f"phase302_{args.model}_semantic_readout_rows.jsonl", readout_rows)
    write_jsonl(out_dir / f"phase302_{args.model}_semantic_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase302_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize_model(model: str, cases: list[dict[str, Any]], behavior: list[dict[str, Any]], readout: list[dict[str, Any]], rollout: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "model": model,
        "planned_rows": len(cases),
        "behavior_rows": len(behavior),
        "readout_rows": len(readout),
        "rollout_rows": len(rollout),
        "missing_rows": len(missing),
        "answer_correct_proxy_rate": mean_safe([1.0 if r.get("answer_correct_proxy") else 0.0 for r in behavior]),
        "pattern_matched_proxy_rate": mean_safe([1.0 if r.get("pattern_matched_proxy") else 0.0 for r in behavior]),
        "model_stop_executed_rate": mean_safe([1.0 if r.get("model_stop_executed") else 0.0 for r in behavior]),
        "mean_target_rank": mean_safe([safe_float(r.get("target_rank"), 999999.0) for r in readout]),
        "mean_target_margin_vs_winner": mean_safe([safe_float(r.get("target_margin_vs_winner")) for r in readout]),
        "competition_winner_counts": dict(Counter(str(r.get("competition_winner")) for r in readout)),
        "case_type_counts": dict(Counter(str(r.get("case_type")) for r in behavior)),
        "attribute_type_counts": dict(Counter(str(r.get("attribute_type")) for r in behavior)),
    }


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    summaries = [read_json(out_dir / f"phase302_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    behavior: list[dict[str, Any]] = []
    readout: list[dict[str, Any]] = []
    rollout: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        behavior.extend(read_jsonl(out_dir / f"phase302_{model}_semantic_behavior_rows.jsonl"))
        readout.extend(read_jsonl(out_dir / f"phase302_{model}_semantic_readout_rows.jsonl"))
        rollout.extend(read_jsonl(out_dir / f"phase302_{model}_semantic_rollout_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase302_{model}_missing_rows.jsonl"))
    by_attr = defaultdict(list)
    for row in behavior:
        by_attr[str(row.get("attribute_type"))].append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "round_name": round_name,
        "status": "complete",
        "model_summaries": summaries,
        "behavior_rows": len(behavior),
        "readout_rows": len(readout),
        "rollout_rows": len(rollout),
        "missing_rows": len(missing),
        "answer_correct_proxy_rate": mean_safe([1.0 if r.get("answer_correct_proxy") else 0.0 for r in behavior]),
        "pattern_matched_proxy_rate": mean_safe([1.0 if r.get("pattern_matched_proxy") else 0.0 for r in behavior]),
        "model_stop_executed_rate": mean_safe([1.0 if r.get("model_stop_executed") else 0.0 for r in behavior]),
        "competition_winner_counts": dict(Counter(str(r.get("competition_winner")) for r in readout)),
        "attribute_success_rates": {
            attr: mean_safe([1.0 if r.get("answer_correct_proxy") else 0.0 for r in rows]) for attr, rows in sorted(by_attr.items())
        },
    }
    write_json(out_dir / "phase302_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase302_semantic_behavior_rows.jsonl", behavior)
    write_jsonl(out_dir / "phase302_semantic_readout_rows.jsonl", readout)
    write_jsonl(out_dir / "phase302_semantic_rollout_rows.jsonl", rollout)
    write_jsonl(out_dir / "phase302_missing_rows.jsonl", missing)
    write_json(V2 / "phase302_cross_model_summary.json", payload)
    write_jsonl(V2 / "phase302_semantic_behavior_rows.jsonl", behavior)
    write_jsonl(V2 / "phase302_semantic_readout_rows.jsonl", readout)
    write_jsonl(V2 / "phase302_semantic_rollout_rows.jsonl", rollout)
    write_jsonl(V2 / "phase302_missing_rows.jsonl", missing)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rollout-tokens", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        run_model(args)
        return
    for model in MODELS:
        args.model = model
        run_model(args)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
