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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase266_multi_family_baseline_behavior_readout_scan as p266  # noqa: E402


PHASE = 281
SCHEMA_VERSION = "2.8.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
RESULT_ROOT = ROOT / "tests/result/phase281_candidate_closure_quality_verification"
ROUND_DEFAULT = "candidate_closure_quality_verification"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model")), str(row.get("case_id"))


def candidate_rows(model: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(V2 / "phase280_recalibrated_gap_rows.jsonl"):
        flags = row.get("remaining_gap_flags") or row.get("gap_flags") or {}
        if row.get("model") == model and flags.get("candidate_not_closed"):
            rows.append(row)
    rows.sort(key=lambda r: (-safe_float(r.get("priority_score_after_phase279", r.get("priority_score"))), str(r.get("family_id")), str(r.get("case_id"))))
    return rows


def load_case(row: dict[str, Any]) -> dict[str, Any]:
    detail_ref = V2 / str(row.get("detail_ref", ""))
    detail = read_json(detail_ref)
    case = detail.get("case") or {}
    if case:
        return case
    return {
        "case_id": row.get("case_id"),
        "family_id": row.get("family_id"),
        "mode_id": row.get("mode_id"),
        "variant_id": row.get("variant_id"),
        "prompt": row.get("prompt", ""),
        "target": row.get("target", ""),
        "target_aliases": row.get("target_aliases") or [row.get("target", "")],
        "expected_pattern": row.get("expected_pattern", "short_answer"),
        "output_protocol": row.get("output_protocol", "short"),
        "boundary_type": row.get("boundary_type", ""),
        "continuation_trigger": row.get("continuation_trigger", ""),
        "path_schema_id": row.get("path_schema_id", ""),
        "variant_type": row.get("variant_type", ""),
        "scoring_risk": row.get("scoring_risk", "unknown"),
    }


def repeated_protocol_marker(output: str, case: dict[str, Any]) -> bool:
    protocol = str(case.get("output_protocol") or case.get("expected_pattern") or "")
    if "json" in protocol:
        return output.count("{") > 1 or output.count("```") > 2
    if "list" in protocol:
        return output.count("\n-") > 3 or output.count("\n1") > 1
    return False


def closure_row(model: str, row: dict[str, Any], case: dict[str, Any], readout: dict[str, Any], output: str, stopped: bool, new_tokens: int, rollout_tokens: int) -> dict[str, Any]:
    cls = p266.classify_output(output, case)
    semantic_done = bool(cls.get("answer_correct_proxy"))
    protocol_matched = bool(cls.get("pattern_matched_proxy"))
    stop_wins = str(readout.get("competition_winner")) == "stop"
    continue_suppressed = safe_float(readout.get("top_continue_vs_stop_margin")) <= -0.5
    repeated_marker = repeated_protocol_marker(output, case)
    rollout_stable = bool(protocol_matched and not cls.get("has_drift_marker") and not repeated_marker and (stopped or int(new_tokens) < int(rollout_tokens)))
    four_condition_closed = bool(semantic_done and stop_wins and continue_suppressed and rollout_stable)
    weak_candidate_survived = bool(semantic_done and protocol_matched and stop_wins)
    blockers = []
    if not semantic_done:
        blockers.append("semantic_not_done")
    if not stop_wins:
        blockers.append("stop_not_winner")
    if not continue_suppressed:
        blockers.append("continue_not_suppressed")
    if not rollout_stable:
        blockers.append("rollout_not_stable")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase281",
        "created_at": utc_now(),
        "closure_quality_id": f"phase281:closure_quality:{model}:{row['case_id']}",
        "source_gap_id": row.get("gap_id"),
        "model": model,
        "case_id": row["case_id"],
        "family_id": row.get("family_id"),
        "mode_id": row.get("mode_id"),
        "variant_id": row.get("variant_id"),
        "path_schema_id": case.get("path_schema_id"),
        "target": case.get("target"),
        "expected_pattern": case.get("expected_pattern"),
        "output_protocol": case.get("output_protocol"),
        "continuation_trigger": case.get("continuation_trigger"),
        "semantic_done": semantic_done,
        "protocol_matched": protocol_matched,
        "stop_wins": stop_wins,
        "continue_suppressed": continue_suppressed,
        "rollout_stable": rollout_stable,
        "four_condition_closed": four_condition_closed,
        "weak_candidate_survived": weak_candidate_survived,
        "closure_blockers": blockers,
        "r_stop": readout.get("r_stop"),
        "r_continue": readout.get("r_continue"),
        "stop_continue_margin": readout.get("stop_continue_margin"),
        "top_continue_vs_stop_margin": readout.get("top_continue_vs_stop_margin"),
        "top_continue_channel": readout.get("top_continue_channel"),
        "competition_winner": readout.get("competition_winner"),
        "target_logit": readout.get("target_logit"),
        "target_rank": readout.get("target_rank"),
        "generated_text": output[:500],
        "generated_token_count": new_tokens,
        "model_stop_executed": stopped,
        "has_drift_marker": cls.get("has_drift_marker"),
        "repeated_protocol_marker": repeated_marker,
        "answer_correct_proxy": cls.get("answer_correct_proxy"),
        "pattern_matched_proxy": cls.get("pattern_matched_proxy"),
        "source_scores": row.get("scores"),
        "source_remaining_flags": row.get("remaining_gap_flags") or row.get("gap_flags"),
    }


def update_detail(model: str, case_id: str, row: dict[str, Any]) -> None:
    detail_ref = V2 / "case_details" / f"{model}__{case_id}.json"
    if not detail_ref.exists():
        return
    detail = read_json(detail_ref)
    detail["phase281_closure_quality"] = row
    detail["phase281_updated_at"] = utc_now()
    write_json(detail_ref, detail)


def evaluate_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = candidate_rows(model)
    model_obj = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    try:
        if selected:
            model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(model, args.attn_implementations)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            for idx, gap_row in enumerate(selected, start=1):
                try:
                    case = load_case(gap_row)
                    prompt = str(case["prompt"])
                    aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
                    readout = p266.capture_readout(model_obj, tokenizer, device, prompt, aliases)
                    output, stopped, new_tokens = p266.generate_probe(model_obj, tokenizer, device, prompt, int(args.rollout_tokens))
                    qrow = closure_row(model, gap_row, case, readout, output, stopped, new_tokens, int(args.rollout_tokens))
                    rows.append(qrow)
                    update_detail(model, str(gap_row["case_id"]), qrow)
                except Exception as exc:  # noqa: BLE001
                    missing.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase281",
                            "created_at": utc_now(),
                            "missing_id": f"phase281:missing:{model}:{gap_row.get('case_id')}",
                            "model": model,
                            "case_id": gap_row.get("case_id"),
                            "family_id": gap_row.get("family_id"),
                            "reason": repr(exc),
                        }
                    )
                log(f"{model}: phase281 verified {idx}/{len(selected)} candidate rows")
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    summary = summarize_model(model, selected, rows, missing)
    write_json(out_dir / f"phase281_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase281_{model}_closure_quality_rows.jsonl", rows)
    write_jsonl(out_dir / f"phase281_{model}_missing_rows.jsonl", missing)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize_model(model: str, selected: list[dict[str, Any]], rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase281",
        "title": "candidate closure four-condition quality verification",
        "status": "complete",
        "created_at": utc_now(),
        "model": model,
        "candidate_rows": len(selected),
        "closure_quality_rows": len(rows),
        "missing_rows": len(missing),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in rows)),
        "semantic_done_count": sum(1 for r in rows if r.get("semantic_done")),
        "stop_wins_count": sum(1 for r in rows if r.get("stop_wins")),
        "continue_suppressed_count": sum(1 for r in rows if r.get("continue_suppressed")),
        "rollout_stable_count": sum(1 for r in rows if r.get("rollout_stable")),
        "four_condition_closed_count": sum(1 for r in rows if r.get("four_condition_closed")),
        "weak_candidate_survived_count": sum(1 for r in rows if r.get("weak_candidate_survived")),
        "blocker_counts": dict(Counter(b for r in rows for b in r.get("closure_blockers", []))),
        "mean_stop_continue_margin": mean_safe([safe_float(r.get("stop_continue_margin")) for r in rows]),
    }


def update_v2(round_name: str, payload: dict[str, Any]) -> None:
    out_dir = RESULT_ROOT / round_name
    quality_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        quality_rows.extend(read_jsonl(out_dir / f"phase281_{model}_closure_quality_rows.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase281_{model}_missing_rows.jsonl"))
    write_jsonl(V2 / "phase281_closure_quality_rows.jsonl", quality_rows)
    write_jsonl(V2 / "phase281_missing_rows.jsonl", missing_rows)
    write_json(V2 / "phase281_cross_model_summary.json", payload)
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files["phase281_closure_quality_rows"] = "phase281_closure_quality_rows.jsonl"
    files["phase281_missing_rows"] = "phase281_missing_rows.jsonl"
    files["phase281_cross_model_summary"] = "phase281_cross_model_summary.json"
    files["phase281_report"] = "phase281_report.md"
    manifest["latest_closure_quality_phase"] = "Phase281"
    manifest["phase281_summary"] = payload
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase281_cross_model_summary.json", "phase281_closure_quality_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase281_summary_ref"] = "phase281_cross_model_summary.json"
    client["phase281_closure_quality_ref"] = "phase281_closure_quality_rows.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    schema.setdefault("tables", {})["phase281_closure_quality_rows"] = "candidate rows checked by SemanticDone, StopWins, ContinueSuppressed, and RolloutStable"
    write_json(V2 / "schema.json", schema)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase281_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        rows.extend(read_jsonl(out_dir / f"phase281_{model}_closure_quality_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase281_{model}_missing_rows.jsonl"))
    payload = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase281",
        "title": "candidate closure four-condition quality verification",
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "created_at": utc_now(),
        "round_name": round_name,
        "model_summaries": summaries,
        "closure_quality_rows": len(rows),
        "missing_rows": len(missing),
        "model_counts": dict(Counter(str(r.get("model")) for r in rows)),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in rows)),
        "semantic_done_rate": round(sum(1 for r in rows if r.get("semantic_done")) / len(rows), 6) if rows else 0.0,
        "stop_wins_rate": round(sum(1 for r in rows if r.get("stop_wins")) / len(rows), 6) if rows else 0.0,
        "continue_suppressed_rate": round(sum(1 for r in rows if r.get("continue_suppressed")) / len(rows), 6) if rows else 0.0,
        "rollout_stable_rate": round(sum(1 for r in rows if r.get("rollout_stable")) / len(rows), 6) if rows else 0.0,
        "four_condition_closed_count": sum(1 for r in rows if r.get("four_condition_closed")),
        "weak_candidate_survived_count": sum(1 for r in rows if r.get("weak_candidate_survived")),
        "blocker_counts": dict(Counter(b for r in rows for b in r.get("closure_blockers", []))),
        "mean_stop_continue_margin": mean_safe([safe_float(r.get("stop_continue_margin")) for r in rows]),
    }
    write_json(out_dir / "phase281_cross_model_summary.json", payload)
    update_v2(round_name, payload)
    lines = [
        "# Phase281 Candidate Closure Quality Verification",
        "",
        f"- closure_quality_rows: {payload['closure_quality_rows']}",
        f"- missing_rows: {payload['missing_rows']}",
        f"- four_condition_closed_count: {payload['four_condition_closed_count']}",
        f"- weak_candidate_survived_count: {payload['weak_candidate_survived_count']}",
        f"- blocker_counts: {json.dumps(payload['blocker_counts'], ensure_ascii=False)}",
        "",
        "This phase is strict candidate verification, not a new closure claim.",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "phase281_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase281_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--rollout-tokens", type=int, default=16)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        evaluate_model(args, args.model)
        return
    for model in MODELS:
        evaluate_model(args, model)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
