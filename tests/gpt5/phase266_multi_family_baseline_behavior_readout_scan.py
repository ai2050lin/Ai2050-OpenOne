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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase262_continuation_regime_decomposition_atlas as p262  # noqa: E402


PHASE = 266
SOURCE_PHASE = 265
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase266_multi_family_baseline_behavior_readout_scan")
ROUND_DEFAULT = "multi_family_baseline_behavior_readout_scan"


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


def append_unique_jsonl(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def rate_by(rows: list[dict[str, Any]], group_key: str, bool_key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(bool(row.get(bool_key)))
    return {k: round(sum(v) / len(v), 6) for k, v in grouped.items() if v}


def select_balanced(rows: list[dict[str, Any]], max_per_family: int) -> list[dict[str, Any]]:
    by_family_variant: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_family_variant[str(row["family_id"])][str(row["variant_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for family_id in sorted(by_family_variant):
        buckets = by_family_variant[family_id]
        idx = 0
        family_rows: list[dict[str, Any]] = []
        while len(family_rows) < max_per_family:
            added = False
            for variant_id in sorted(buckets):
                bucket = buckets[variant_id]
                if idx < len(bucket):
                    family_rows.append(bucket[idx])
                    added = True
                    if len(family_rows) >= max_per_family:
                        break
            if not added:
                break
            idx += 1
        selected.extend(family_rows)
    return selected


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def classify_output(output: str, case: dict[str, Any]) -> dict[str, Any]:
    low = normalize(output)
    target = str(case.get("target", "")).strip()
    aliases = [str(x) for x in case.get("target_aliases") or [target]]
    alias_hit = any(normalize(a) in low for a in aliases if a)
    token_count = len(output.replace("\n", " ").split())
    expected = str(case.get("expected_pattern", "short"))
    protocol = str(case.get("output_protocol", "short"))
    has_because = "because" in low or "因为" in output
    has_json = "{" in output and "}" in output
    has_list = "\n-" in output or "\n1" in output or output.strip().startswith("-")
    has_drift = "answer:" in low or "question:" in low or token_count > 24
    if expected == "json" or protocol == "json":
        pattern_matched = has_json and alias_hit
    elif expected == "list" or "list" in protocol:
        pattern_matched = has_list and alias_hit
    elif expected == "explain" or protocol == "explain":
        pattern_matched = alias_hit and has_because
    else:
        pattern_matched = alias_hit and token_count <= 8 and not has_drift
    return {
        "alias_hit": alias_hit,
        "answer_correct_proxy": alias_hit,
        "pattern_matched_proxy": pattern_matched,
        "token_count": token_count,
        "has_because": has_because,
        "has_json": has_json,
        "has_list": has_list,
        "has_drift_marker": has_drift,
        "output_preview": output[:300],
    }


def generate_probe(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int) -> tuple[str, bool, int]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    input_len = int(encoded["input_ids"].shape[1])
    eos_id = tokenizer.eos_token_id
    with torch.inference_mode():
        out = model_obj.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, input_len:].detach().cpu().tolist()
    stopped = bool(eos_id is not None and new_ids and int(new_ids[-1]) == int(eos_id))
    return tokenizer.decode(new_ids, skip_special_tokens=False), stopped, len(new_ids)


def capture_readout(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str, aliases: list[str]) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, return_dict=True)
    logits = out.logits[0, last_pos].detach().float().cpu()
    stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
    cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
    return {**p262.score_channels(logits, stop_ids, cont_ids), **p239.readout_metrics(tokenizer, logits, aliases)}


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cases = read_jsonl(ATLAS_ROOT / "mode_family_case_bank_v3.jsonl")
    cases = select_balanced(all_cases, int(args.max_cases_per_family))
    model_obj = None
    tokenizer = None
    behavior_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, case in enumerate(cases, start=1):
            prompt = str(case["prompt"])
            target = str(case["target"])
            aliases = [str(x) for x in case.get("target_aliases") or [target]]
            readout = capture_readout(model_obj, tokenizer, device, prompt, aliases)
            output, stopped, new_tokens = generate_probe(model_obj, tokenizer, device, prompt, int(args.rollout_tokens))
            cls = classify_output(output, case)
            base = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase266",
                "created_at": utc_now(),
                "model": args.model,
                "case_id": case["case_id"],
                "family_id": case["family_id"],
                "mode_id": case["mode_id"],
                "variant_id": case["variant_id"],
                "variant_type": case["variant_type"],
                "path_schema_id": case["path_schema_id"],
                "target": target,
                "expected_pattern": case["expected_pattern"],
                "output_protocol": case["output_protocol"],
                "boundary_type": case["boundary_type"],
                "continuation_trigger": case["continuation_trigger"],
                "scoring_risk_design": case["scoring_risk"],
            }
            behavior = {
                **base,
                "behavior_id": f"phase266:behavior:{args.model}:{case['case_id']}",
                **cls,
                "model_stop_executed": stopped,
                "generated_token_count": new_tokens,
            }
            behavior_rows.append(behavior)
            readout_rows.append(
                {
                    **base,
                    "readout_id": f"phase266:readout:{args.model}:{case['case_id']}",
                    **readout,
                }
            )
            rollout_rows.append(
                {
                    **base,
                    "rollout_id": f"phase266:rollout:{args.model}:{case['case_id']}",
                    "generated_text": output[:500],
                    "generated_token_count": new_tokens,
                    "model_stop_executed": stopped,
                    "answer_correct_proxy": cls["answer_correct_proxy"],
                    "pattern_matched_proxy": cls["pattern_matched_proxy"],
                    "has_drift_marker": cls["has_drift_marker"],
                    "top_continue_channel": readout.get("top_continue_channel"),
                    "competition_winner": readout.get("competition_winner"),
                }
            )
            calibrated_risk = calibrate_risk(case, cls, readout, stopped)
            quality_rows.append(
                {
                    **base,
                    "quality_id": f"phase266:quality:{args.model}:{case['case_id']}",
                    "scoring_risk_calibrated": calibrated_risk,
                    "risk_changed": calibrated_risk != case["scoring_risk"],
                    "answer_correct_proxy": cls["answer_correct_proxy"],
                    "pattern_matched_proxy": cls["pattern_matched_proxy"],
                    "target_margin_vs_winner": readout.get("target_margin_vs_winner"),
                    "stop_continue_margin": readout.get("stop_continue_margin"),
                }
            )
            observations.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase266",
                    "created_at": utc_now(),
                    "observation_id": f"phase266:obs:{args.model}:{case['case_id']}",
                    "case_id": case["case_id"],
                    "model": args.model,
                    "family_id": case["family_id"],
                    "mode_id": case["mode_id"],
                    "variant_id": case["variant_id"],
                    "level": "multi_family_baseline_scan",
                    "component": str(readout.get("top_continue_channel")),
                    "metric_name": "answer_correct_proxy",
                    "metric_value": 1.0 if cls["answer_correct_proxy"] else 0.0,
                    "metric_unit": "bool",
                    "winner": readout.get("competition_winner"),
                }
            )
            if idx % 36 == 0:
                log(f"{args.model}: scanned {idx}/{len(cases)} cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = make_metrics(args.model, behavior_rows, readout_rows, rollout_rows, quality_rows)
    edges = make_edges(args.model, readout_rows)
    payload = summarize_model(args.model, behavior_rows, readout_rows, rollout_rows, quality_rows, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, payload, behavior_rows, readout_rows, rollout_rows, quality_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def calibrate_risk(case: dict[str, Any], cls: dict[str, Any], readout: dict[str, Any], stopped: bool) -> str:
    risk = str(case.get("scoring_risk", "medium"))
    if not cls["answer_correct_proxy"] or safe_float(readout.get("target_margin_vs_winner")) < -5:
        return "high"
    if not cls["pattern_matched_proxy"] or safe_float(readout.get("stop_continue_margin")) < -6 or cls["has_drift_marker"]:
        return "medium" if risk == "low" else risk
    if stopped and cls["pattern_matched_proxy"]:
        return "low"
    return risk


def make_metrics(model: str, behavior: list[dict[str, Any]], readout: list[dict[str, Any]], rollout: list[dict[str, Any]], quality: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, rate in rate_by(behavior, "family_id", "answer_correct_proxy").items():
        fam_behavior = [r for r in behavior if r["family_id"] == family]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase266",
                "created_at": utc_now(),
                "metric_id": f"phase266:{model}:{family}:answer_correct_proxy_rate",
                "scope": "multi_family_baseline_behavior",
                "model": model,
                "family_id": family,
                "metric_name": "answer_correct_proxy_rate",
                "metric_value": rate,
                "pattern_matched_rate": rate_by(fam_behavior, "family_id", "pattern_matched_proxy").get(family, 0.0),
                "rows": len(fam_behavior),
            }
        )
    for family, margin in mean_by(readout, "family_id", "stop_continue_margin").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase266",
                "created_at": utc_now(),
                "metric_id": f"phase266:{model}:{family}:mean_stop_continue_margin",
                "scope": "multi_family_readout",
                "model": model,
                "family_id": family,
                "metric_name": "mean_stop_continue_margin",
                "metric_value": margin,
                "rows": sum(1 for r in readout if r["family_id"] == family),
            }
        )
    risk_counts = Counter(str(r["scoring_risk_calibrated"]) for r in quality)
    for risk, count in risk_counts.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase266",
                "created_at": utc_now(),
                "metric_id": f"phase266:{model}:risk:{risk}:count",
                "scope": "quality_calibration",
                "model": model,
                "metric_name": "calibrated_risk_count",
                "risk": risk,
                "metric_value": count,
                "rows": len(quality),
            }
        )
    return rows


def make_edges(model: str, readout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    grouped = Counter((r["family_id"], str(r.get("top_continue_channel"))) for r in readout if r.get("competition_winner") == "continue")
    for (family, channel), count in grouped.items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase266",
                "created_at": utc_now(),
                "edge_id": f"phase266:{model}:{family}:{channel}:failure_path",
                "source": f"node:{family}",
                "target": f"node:{channel}",
                "edge_type": "baseline_family_failure_path",
                "model": model,
                "evidence_type": "multi_family_baseline_readout_scan",
                "effect_size": count,
                "status": "baseline_not_internal_trace",
            }
        )
    return edges


def summarize_model(model: str, behavior: list[dict[str, Any]], readout: list[dict[str, Any]], rollout: list[dict[str, Any]], quality: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    winner_counts = Counter(str(r.get("competition_winner")) for r in readout)
    top_channels = Counter(str(r.get("top_continue_channel")) for r in readout if r.get("competition_winner") == "continue")
    risks = Counter(str(r.get("scoring_risk_calibrated")) for r in quality)
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Multi-family baseline behavior and readout scan",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "behavior_rows": len(behavior),
        "readout_rows": len(readout),
        "rollout_rows": len(rollout),
        "quality_rows": len(quality),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "answer_correct_proxy_rate_by_family": rate_by(behavior, "family_id", "answer_correct_proxy"),
        "pattern_matched_proxy_rate_by_family": rate_by(behavior, "family_id", "pattern_matched_proxy"),
        "mean_stop_continue_margin_by_family": mean_by(readout, "family_id", "stop_continue_margin"),
        "competition_winner_counts": dict(winner_counts),
        "top_continue_channel_counts": dict(top_channels.most_common()),
        "calibrated_risk_counts": dict(risks),
        "model_stop_rate": round(sum(1 for r in rollout if r.get("model_stop_executed")) / len(rollout), 6) if rollout else 0.0,
    }


def write_model_outputs(out_dir: Path, model: str, summary: dict[str, Any], behavior: list[dict[str, Any]], readout: list[dict[str, Any]], rollout: list[dict[str, Any]], quality: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase266_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase266_{model}_behavior_rows.jsonl", behavior)
    write_jsonl(out_dir / f"phase266_{model}_readout_rows.jsonl", readout)
    write_jsonl(out_dir / f"phase266_{model}_rollout_probe_rows.jsonl", rollout)
    write_jsonl(out_dir / f"phase266_{model}_quality_calibration_rows.jsonl", quality)
    write_jsonl(out_dir / f"phase266_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase266_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase266_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase266_{model}_missing_rows.jsonl", missing)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase266_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    behavior: list[dict[str, Any]] = []
    readout: list[dict[str, Any]] = []
    rollout: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        behavior.extend(read_jsonl(out_dir / f"phase266_{model}_behavior_rows.jsonl"))
        readout.extend(read_jsonl(out_dir / f"phase266_{model}_readout_rows.jsonl"))
        rollout.extend(read_jsonl(out_dir / f"phase266_{model}_rollout_probe_rows.jsonl"))
        quality.extend(read_jsonl(out_dir / f"phase266_{model}_quality_calibration_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase266_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase266_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase266_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase266_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.88,
        "physical_path_atlas": 0.30,
        "multi_family_case_bank": 0.45,
        "multi_family_baseline_scan": 0.16,
        "state_factor_atlas": 0.37,
        "path_cluster_mining": 0.14,
        "trace_signature_validation": 0.48,
        "readout_competition_trace": 0.79,
        "stepwise_rollout_trace": 0.44,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.67,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Multi-family baseline behavior and readout scan",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "behavior_rows": len(behavior),
        "readout_rows": len(readout),
        "rollout_rows": len(rollout),
        "quality_rows": len(quality),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "answer_correct_proxy_rate_by_family": rate_by(behavior, "family_id", "answer_correct_proxy"),
        "pattern_matched_proxy_rate_by_family": rate_by(behavior, "family_id", "pattern_matched_proxy"),
        "mean_stop_continue_margin_by_family": mean_by(readout, "family_id", "stop_continue_margin"),
        "competition_winner_counts": dict(Counter(str(r.get("competition_winner")) for r in readout)),
        "top_continue_channel_counts": dict(Counter(str(r.get("top_continue_channel")) for r in readout if r.get("competition_winner") == "continue").most_common()),
        "calibrated_risk_counts": dict(Counter(str(r.get("scoring_risk_calibrated")) for r in quality)),
        "model_stop_rate": round(sum(1 for r in rollout if r.get("model_stop_executed")) / len(rollout), 6) if rollout else 0.0,
        "progress": progress,
    }
    write_json(out_dir / "phase266_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase266_behavior_rows.jsonl", behavior)
    write_jsonl(out_dir / "phase266_readout_rows.jsonl", readout)
    write_jsonl(out_dir / "phase266_rollout_probe_rows.jsonl", rollout)
    write_jsonl(out_dir / "phase266_quality_calibration_rows.jsonl", quality)
    write_jsonl(out_dir / "phase266_observations.jsonl", observations)
    write_jsonl(out_dir / "phase266_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase266_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase266_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "phase266_behavior_rows.jsonl", behavior)
    write_jsonl(ATLAS_ROOT / "phase266_readout_rows.jsonl", readout)
    write_jsonl(ATLAS_ROOT / "phase266_rollout_probe_rows.jsonl", rollout)
    write_jsonl(ATLAS_ROOT / "phase266_quality_calibration_rows.jsonl", quality)
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase266", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase266 Multi-Family Baseline Behavior And Readout Scan",
        "",
        f"- status: {payload['status']}",
        f"- behavior_rows: {payload['behavior_rows']}",
        f"- readout_rows: {payload['readout_rows']}",
        f"- rollout_rows: {payload['rollout_rows']}",
        f"- answer_correct_proxy_rate_by_family: {json.dumps(payload['answer_correct_proxy_rate_by_family'], ensure_ascii=False)}",
        f"- pattern_matched_proxy_rate_by_family: {json.dumps(payload['pattern_matched_proxy_rate_by_family'], ensure_ascii=False)}",
        f"- mean_stop_continue_margin_by_family: {json.dumps(payload['mean_stop_continue_margin_by_family'], ensure_ascii=False)}",
        f"- competition_winner_counts: {json.dumps(payload['competition_winner_counts'], ensure_ascii=False)}",
        f"- top_continue_channel_counts: {json.dumps(payload['top_continue_channel_counts'], ensure_ascii=False)}",
        f"- calibrated_risk_counts: {json.dumps(payload['calibrated_risk_counts'], ensure_ascii=False)}",
        f"- model_stop_rate: {payload['model_stop_rate']}",
    ]
    (out_dir / "phase266_multi_family_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-family", type=int, default=36)
    parser.add_argument("--rollout-tokens", type=int, default=12)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        evaluate_model(args)
        return
    for model in MODELS:
        args.model = model
        evaluate_model(args)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
