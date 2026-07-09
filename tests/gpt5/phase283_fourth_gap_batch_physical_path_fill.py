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
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase275_selected_gap_batch_physical_path_fill as p275  # noqa: E402


PHASE = 283
SCHEMA_VERSION = "2.10.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
RESULT_ROOT = ROOT / "tests/result/phase283_fourth_gap_batch_physical_path_fill"
ROUND_DEFAULT = "fourth_gap_batch_physical_path_fill"


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


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["batch_rank"] = row.get("phase282_next_batch_rank") or row.get("phase280_next_batch_rank") or row.get("batch_rank")
    out["missing_dimensions"] = row.get("remaining_dimensions") or row.get("missing_dimensions") or []
    out["gap_flags"] = row.get("remaining_gap_flags") or row.get("gap_flags") or {}
    return out


def select_model_queue(model: str, max_cases: int) -> list[dict[str, Any]]:
    rows = [normalize(r) for r in read_jsonl(V2 / "phase282_recalibrated_gap_rows.jsonl") if r.get("model") == model]
    def physical_needed(r: dict[str, Any]) -> bool:
        dims = set(r.get("missing_dimensions") or [])
        return ("need_component_path" in dims) or ("need_causal_audit" in dims) or ("need_layer_path" in dims)
    rows = [r for r in rows if physical_needed(r) and "candidate_closure_verification" not in set(r.get("missing_dimensions") or [])]
    rows.sort(key=lambda r: (-safe_float(r.get("priority_score_after_phase281")), str(r.get("family_id")), str(r.get("case_id"))))
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        family = str(row.get("family_id"))
        if counts[family] >= 2:
            continue
        selected.append(row)
        counts[family] += 1
        if len(selected) >= max_cases:
            break
    for row in rows:
        if len(selected) >= max_cases:
            break
        if row not in selected:
            selected.append(row)
    return selected[:max_cases]


def retag_rows(rows: list[dict[str, Any]], id_keys: list[str]) -> None:
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["phase_id"] = "Phase283"
        row["created_at"] = utc_now()
        for key in id_keys:
            if key in row and isinstance(row[key], str):
                row[key] = row[key].replace("phase275:", "phase283:").replace("Phase275", "Phase283")


def update_case_detail(model: str, case_id: str, summary: dict[str, Any] | None, causal_rows: list[dict[str, Any]]) -> None:
    detail_ref = V2 / "case_details" / f"{model}__{case_id}.json"
    if not detail_ref.exists():
        return
    detail = read_json(detail_ref)
    detail["phase283_component_summary"] = summary
    detail["phase283_causal_fill_rows"] = causal_rows
    detail["phase283_updated_at"] = utc_now()
    write_json(detail_ref, detail)


def evaluate_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_model_queue(model, int(args.max_cases_per_model))
    model_obj = None
    tokenizer = None
    component_rows: list[dict[str, Any]] = []
    attn_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p275.p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stop_ids = {name: p275.p262.token_ids(tokenizer, texts) for name, texts in p275.p262.STOP_GROUPS.items()}
        cont_ids = {name: p275.p262.token_ids(tokenizer, texts) for name, texts in p275.p262.CONT_GROUPS.items()}
        for idx, queue_row in enumerate(selected, start=1):
            case = {
                "model": model,
                "case_id": queue_row["case_id"],
                "family_id": queue_row["family_id"],
                "mode_id": queue_row["mode_id"],
                "variant_id": queue_row["variant_id"],
                "path_schema_id": queue_row.get("path_schema_id") or f"phase283:{queue_row['family_id']}:{queue_row['mode_id']}:{queue_row['variant_id']}",
                "top_continue_channel_phase266": (queue_row.get("path_signature") or {}).get("readout_winner"),
            }
            try:
                comp, attn, mlp, resid, summary = p275.p268.component_decomposition(model_obj, tokenizer, device, case, stop_ids, cont_ids)
                for rows, keys in [
                    (comp, ["component_physical_path_id"]),
                    (attn, ["attention_contribution_id"]),
                    (mlp, ["mlp_contribution_id"]),
                    (resid, ["residual_accumulation_id"]),
                ]:
                    retag_rows(rows, keys)
                    for row in rows:
                        row["source_phase282_gap_id"] = queue_row.get("gap_id")
                        row["source_phase282_batch_rank"] = queue_row.get("batch_rank")
                summary = {
                    **summary,
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase283",
                    "created_at": utc_now(),
                    "source_phase282_gap_id": queue_row.get("gap_id"),
                    "source_phase282_batch_rank": queue_row.get("batch_rank"),
                    "component_summary_id": f"phase283:summary:{model}:{queue_row['case_id']}",
                }
                crows, rrows = p275.causal_audit(model_obj, tokenizer, device, queue_row, summary, int(args.rollout_tokens))
                retag_rows(crows, ["causal_fill_id"])
                retag_rows(rrows, ["rollout_fill_id"])
                for row in crows + rrows:
                    row["source_phase282_gap_id"] = queue_row.get("gap_id")
                    row["source_phase282_batch_rank"] = queue_row.get("batch_rank")
                component_rows.extend(comp)
                attn_rows.extend(attn)
                mlp_rows.extend(mlp)
                residual_rows.extend(resid)
                summary_rows.append(summary)
                causal_rows.extend(crows)
                rollout_rows.extend(rrows)
                update_case_detail(model, str(queue_row["case_id"]), summary, crows)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase283",
                        "created_at": utc_now(),
                        "missing_id": f"phase283:missing:{model}:{queue_row['case_id']}",
                        "source_phase282_gap_id": queue_row.get("gap_id"),
                        "model": model,
                        "case_id": queue_row["case_id"],
                        "family_id": queue_row["family_id"],
                        "reason": repr(exc),
                    }
                )
            log(f"{model}: phase283 filled {idx}/{len(selected)} selected gap rows")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p275.p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    payload = summarize_model(model, selected, component_rows, summary_rows, causal_rows, rollout_rows, missing_rows)
    write_json(out_dir / f"phase283_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase283_{model}_component_physical_path_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase283_{model}_attention_contribution_rows.jsonl", attn_rows)
    write_jsonl(out_dir / f"phase283_{model}_mlp_contribution_rows.jsonl", mlp_rows)
    write_jsonl(out_dir / f"phase283_{model}_residual_accumulation_rows.jsonl", residual_rows)
    write_jsonl(out_dir / f"phase283_{model}_component_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase283_{model}_causal_fill_rows.jsonl", causal_rows)
    write_jsonl(out_dir / f"phase283_{model}_rollout_fill_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase283_{model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_model(model: str, selected: list[dict[str, Any]], component_rows: list[dict[str, Any]], summaries: list[dict[str, Any]], causal_rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]], missing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    low_side = [r for r in causal_rows if r.get("side_effect_level") == "lower"]
    return {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase283",
        "title": "Phase282 fourth gap batch physical path fill",
        "status": "complete",
        "created_at": utc_now(),
        "model": model,
        "selected_gap_rows": len(selected),
        "component_physical_path_rows": len(component_rows),
        "component_summary_rows": len(summaries),
        "causal_fill_rows": len(causal_rows),
        "rollout_fill_rows": len(rollout_rows),
        "missing_rows": len(missing_rows),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in summaries)),
        "dominant_positive_component_counts": dict(Counter(str(r.get("dominant_positive_component")) for r in summaries)),
        "final_winner_counts": dict(Counter(str(r.get("final_winner")) for r in summaries)),
        "strongest_mlp_layers": dict(Counter(str(r.get("strongest_mlp_layer")) for r in summaries).most_common()),
        "mean_sum_positive_attn_delta": mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in summaries]),
        "mean_sum_positive_mlp_delta": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in summaries]),
        "mean_sum_positive_residual_delta": mean_safe([safe_float(r.get("sum_positive_residual_delta")) for r in summaries]),
        "causal_effect_supported_counts": dict(Counter(str(r.get("causal_effect_supported")) for r in causal_rows)),
        "side_effect_risk_counts": dict(Counter(str(r.get("side_effect_risk")) for r in causal_rows)),
        "low_side_effect_supported_rate": round(sum(1 for r in low_side if r.get("causal_effect_supported")) / len(low_side), 6) if low_side else 0.0,
        "low_side_effect_risk_rate": round(sum(1 for r in low_side if r.get("side_effect_risk")) / len(low_side), 6) if low_side else 0.0,
    }


def update_v2(round_name: str, payload: dict[str, Any]) -> None:
    out_dir = RESULT_ROOT / round_name
    table_names = [
        "component_physical_path_rows",
        "attention_contribution_rows",
        "mlp_contribution_rows",
        "residual_accumulation_rows",
        "component_summary_rows",
        "causal_fill_rows",
        "rollout_fill_rows",
        "missing_rows",
    ]
    for table in table_names:
        rows: list[dict[str, Any]] = []
        for model in MODELS:
            rows.extend(read_jsonl(out_dir / f"phase283_{model}_{table}.jsonl"))
        write_jsonl(V2 / f"phase283_{table}.jsonl", rows)
    write_json(V2 / "phase283_cross_model_summary.json", payload)
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for table in table_names:
        files[f"phase283_{table}"] = f"phase283_{table}.jsonl"
    files["phase283_cross_model_summary"] = "phase283_cross_model_summary.json"
    files["phase283_report"] = "phase283_report.md"
    manifest["latest_fill_phase"] = "Phase283"
    manifest["phase283_summary"] = payload
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase283_cross_model_summary.json", "phase283_component_summary_rows.jsonl", "phase283_causal_fill_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase283_summary_ref"] = "phase283_cross_model_summary.json"
    client["phase283_component_summary_ref"] = "phase283_component_summary_rows.jsonl"
    client["phase283_causal_fill_ref"] = "phase283_causal_fill_rows.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase283_component_summary_rows"] = "Phase282 physical-path queue component path fills"
    tables["phase283_causal_fill_rows"] = "Phase282 physical-path queue low-side-effect and diagnostic causal rows"
    write_json(V2 / "schema.json", schema)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase283_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    component_summary_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        component_summary_rows.extend(read_jsonl(out_dir / f"phase283_{model}_component_summary_rows.jsonl"))
        causal_rows.extend(read_jsonl(out_dir / f"phase283_{model}_causal_fill_rows.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase283_{model}_missing_rows.jsonl"))
    low_side = [r for r in causal_rows if r.get("side_effect_level") == "lower"]
    payload = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase283",
        "title": "Phase282 fourth gap batch physical path fill",
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "created_at": utc_now(),
        "round_name": round_name,
        "model_summaries": summaries,
        "component_summary_rows": len(component_summary_rows),
        "causal_fill_rows": len(causal_rows),
        "missing_rows": len(missing_rows),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in component_summary_rows)),
        "model_counts": dict(Counter(str(r.get("model")) for r in component_summary_rows)),
        "dominant_positive_component_counts": dict(Counter(str(r.get("dominant_positive_component")) for r in component_summary_rows)),
        "final_winner_counts": dict(Counter(str(r.get("final_winner")) for r in component_summary_rows)),
        "causal_effect_supported_counts": dict(Counter(str(r.get("causal_effect_supported")) for r in causal_rows)),
        "side_effect_risk_counts": dict(Counter(str(r.get("side_effect_risk")) for r in causal_rows)),
        "low_side_effect_supported_rate": round(sum(1 for r in low_side if r.get("causal_effect_supported")) / len(low_side), 6) if low_side else 0.0,
        "low_side_effect_risk_rate": round(sum(1 for r in low_side if r.get("side_effect_risk")) / len(low_side), 6) if low_side else 0.0,
        "mean_sum_positive_attn_delta": mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in component_summary_rows]),
        "mean_sum_positive_mlp_delta": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in component_summary_rows]),
        "mean_sum_positive_residual_delta": mean_safe([safe_float(r.get("sum_positive_residual_delta")) for r in component_summary_rows]),
    }
    write_json(out_dir / "phase283_cross_model_summary.json", payload)
    update_v2(round_name, payload)
    lines = [
        "# Phase283 Phase282 Physical-Path Gap Batch Fill",
        "",
        f"- status: {payload['status']}",
        f"- component_summary_rows: {payload['component_summary_rows']}",
        f"- causal_fill_rows: {payload['causal_fill_rows']}",
        f"- missing_rows: {payload['missing_rows']}",
        f"- model_counts: {json.dumps(payload['model_counts'], ensure_ascii=False)}",
        f"- family_counts: {json.dumps(payload['family_counts'], ensure_ascii=False)}",
        f"- dominant_positive_component_counts: {json.dumps(payload['dominant_positive_component_counts'], ensure_ascii=False)}",
        f"- low_side_effect_supported_rate: {payload['low_side_effect_supported_rate']}",
        f"- low_side_effect_risk_rate: {payload['low_side_effect_risk_rate']}",
        "",
        "This phase expands physical distribution coverage. It is not closure.",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "phase283_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase283_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-model", type=int, default=18)
    parser.add_argument("--rollout-tokens", type=int, default=6)
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
