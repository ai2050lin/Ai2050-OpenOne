#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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

import phase268_attention_mlp_continuation_path_attribution as p268  # noqa: E402
import phase269_mlp_continuation_writer_necessity_audit as p269  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase262_continuation_regime_decomposition_atlas as p262  # noqa: E402


PHASE = 275
SCHEMA_VERSION = "2.2.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
RESULT_ROOT = ROOT / "tests/result/phase275_selected_gap_batch_physical_path_fill"
ROUND_DEFAULT = "selected_gap_batch_physical_path_fill"


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


def select_model_queue(model: str, max_cases: int, include_candidate_rows: bool) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(V2 / "phase274_selected_batch_rows.jsonl") if r.get("model") == model]
    if not include_candidate_rows:
        rows = [r for r in rows if r.get("batch_kind") != "candidate_closure_path_fill" or "need_component_path" in r.get("missing_dimensions", []) or "need_causal_audit" in r.get("missing_dimensions", [])]
    rows.sort(key=lambda r: (int(r.get("batch_rank") or 999999), -safe_float(r.get("priority_score")), str(r.get("family_id")), str(r.get("case_id"))))
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    for row in rows:
        if len(selected) >= max_cases:
            break
        family = str(row.get("family_id"))
        if family in used_families and len(rows) >= max_cases:
            continue
        selected.append(row)
        used_families.add(family)
    for row in rows:
        if len(selected) >= max_cases:
            break
        if row not in selected:
            selected.append(row)
    return selected[:max_cases]


def update_case_detail(model: str, case_id: str, component_summary: dict[str, Any] | None, causal_rows: list[dict[str, Any]]) -> None:
    detail_ref = V2 / "case_details" / f"{model}__{case_id}.json"
    if not detail_ref.exists():
        return
    detail = read_json(detail_ref)
    detail["phase275_component_summary"] = component_summary
    detail["phase275_causal_fill_rows"] = causal_rows
    detail["phase275_updated_at"] = utc_now()
    write_json(detail_ref, detail)


def patch_types_for_case(queue_row: dict[str, Any]) -> list[dict[str, Any]]:
    missing = set(queue_row.get("missing_dimensions") or [])
    patches = [{"patch_type": "mlp_half_last_token", "scale": 0.5, "side_effect_level": "lower"}]
    if "candidate_closure_verification" in missing:
        return patches
    patches.append({"patch_type": "mlp_zero_last_token", "scale": 0.0, "side_effect_level": "diagnostic_high"})
    return patches


def causal_audit(
    model_obj: Any,
    tokenizer: Any,
    device: torch.device,
    queue_row: dict[str, Any],
    component_summary: dict[str, Any],
    rollout_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_bank = {r["case_id"]: r for r in p269.read_jsonl(p269.ATLAS_ROOT / "mode_family_case_bank_v3.jsonl")}
    source = case_bank.get(str(queue_row["case_id"]))
    if not source:
        raise ValueError(f"case not found in v1 bank: {queue_row['case_id']}")
    layer_idx = component_summary.get("strongest_mlp_layer")
    if layer_idx is None:
        raise ValueError(f"no strongest_mlp_layer for case: {queue_row['case_id']}")
    aliases = [str(x) for x in source.get("target_aliases") or [source.get("target", "")]]
    prompt = str(source["prompt"])
    base_logits = p269.forward_logits(model_obj, tokenizer, device, prompt)
    base_scores = p269.score_logits(tokenizer, base_logits, aliases)
    base_text, base_new_tokens = p269.generate_text(model_obj, tokenizer, device, prompt, rollout_tokens)
    rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    for patch in patch_types_for_case(queue_row):
        patched_logits = p269.forward_logits(model_obj, tokenizer, device, prompt, int(layer_idx), float(patch["scale"]))
        patched_scores = p269.score_logits(tokenizer, patched_logits, aliases)
        patched_text, patched_new_tokens = p269.generate_text(model_obj, tokenizer, device, prompt, rollout_tokens, int(layer_idx), float(patch["scale"]))
        delta_continue_stop = safe_float(patched_scores.get("continue_stop_margin")) - safe_float(base_scores.get("continue_stop_margin"))
        delta_target = safe_float(patched_scores.get("target_logit")) - safe_float(base_scores.get("target_logit"))
        winner_changed = base_scores.get("tri_winner") != patched_scores.get("tri_winner")
        rollout_changed = base_text != patched_text
        side_effect_risk = bool(winner_changed or rollout_changed)
        base = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase275",
            "created_at": utc_now(),
            "source_gap_id": queue_row.get("gap_id"),
            "source_batch_rank": queue_row.get("batch_rank"),
            "model": queue_row["model"],
            "case_id": queue_row["case_id"],
            "family_id": queue_row["family_id"],
            "mode_id": queue_row["mode_id"],
            "variant_id": queue_row["variant_id"],
            "path_schema_id": source.get("path_schema_id"),
            "target": source.get("target"),
            "strongest_mlp_layer_phase275": int(layer_idx),
            "strongest_mlp_delta_phase275": component_summary.get("strongest_mlp_delta"),
            "patch_type": patch["patch_type"],
            "patch_scale": patch["scale"],
            "side_effect_level": patch["side_effect_level"],
        }
        rows.append(
            {
                **base,
                "causal_fill_id": f"phase275:causal:{queue_row['model']}:{queue_row['case_id']}:L{layer_idx}:{patch['patch_type']}",
                "base_continue_stop_margin": round(safe_float(base_scores.get("continue_stop_margin")), 6),
                "patched_continue_stop_margin": round(safe_float(patched_scores.get("continue_stop_margin")), 6),
                "delta_continue_stop_margin": round(delta_continue_stop, 6),
                "base_winner": base_scores.get("tri_winner"),
                "patched_winner": patched_scores.get("tri_winner"),
                "winner_changed": winner_changed,
                "base_target_logit": base_scores.get("target_logit"),
                "patched_target_logit": patched_scores.get("target_logit"),
                "delta_target_logit": round(delta_target, 6),
                "causal_effect_supported": bool(delta_continue_stop < -0.75 or winner_changed),
                "side_effect_risk": side_effect_risk,
            }
        )
        rollout_rows.append(
            {
                **base,
                "rollout_fill_id": f"phase275:rollout:{queue_row['model']}:{queue_row['case_id']}:L{layer_idx}:{patch['patch_type']}",
                "base_text": base_text[:300],
                "patched_text": patched_text[:300],
                "base_new_tokens": base_new_tokens,
                "patched_new_tokens": patched_new_tokens,
                "rollout_changed": rollout_changed,
            }
        )
    return rows, rollout_rows


def evaluate_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_model_queue(model, int(args.max_cases_per_model), bool(args.include_candidate_rows))
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
        model_obj, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
        cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
        for idx, queue_row in enumerate(selected, start=1):
            case = {
                "model": model,
                "case_id": queue_row["case_id"],
                "family_id": queue_row["family_id"],
                "mode_id": queue_row["mode_id"],
                "variant_id": queue_row["variant_id"],
                "path_schema_id": queue_row.get("path_schema_id") or f"phase275:{queue_row['family_id']}:{queue_row['mode_id']}:{queue_row['variant_id']}",
                "top_continue_channel_phase266": queue_row.get("path_signature", {}).get("readout_winner"),
            }
            try:
                comp, attn, mlp, resid, summary = p268.component_decomposition(model_obj, tokenizer, device, case, stop_ids, cont_ids)
                summary = {
                    **summary,
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase275",
                    "source_gap_id": queue_row.get("gap_id"),
                    "source_batch_rank": queue_row.get("batch_rank"),
                    "component_summary_id": f"phase275:summary:{model}:{queue_row['case_id']}",
                }
                for rows, id_key, prefix in [
                    (comp, "component_physical_path_id", "phase275:component_path"),
                    (attn, "attention_contribution_id", "phase275:attn"),
                    (mlp, "mlp_contribution_id", "phase275:mlp"),
                    (resid, "residual_accumulation_id", "phase275:residual"),
                ]:
                    for row in rows:
                        row["schema_version"] = SCHEMA_VERSION
                        row["phase_id"] = "Phase275"
                        row["source_gap_id"] = queue_row.get("gap_id")
                        row["source_batch_rank"] = queue_row.get("batch_rank")
                        layer = row.get("layer_index")
                        row[id_key] = f"{prefix}:{model}:{queue_row['case_id']}:L{layer}"
                component_rows.extend(comp)
                attn_rows.extend(attn)
                mlp_rows.extend(mlp)
                residual_rows.extend(resid)
                summary_rows.append(summary)
                crows, rrows = causal_audit(model_obj, tokenizer, device, queue_row, summary, int(args.rollout_tokens))
                causal_rows.extend(crows)
                rollout_rows.extend(rrows)
                update_case_detail(model, str(queue_row["case_id"]), summary, crows)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase275",
                        "created_at": utc_now(),
                        "missing_id": f"phase275:missing:{model}:{queue_row['case_id']}",
                        "source_gap_id": queue_row.get("gap_id"),
                        "model": model,
                        "case_id": queue_row["case_id"],
                        "family_id": queue_row["family_id"],
                        "reason": repr(exc),
                    }
                )
            log(f"{model}: filled {idx}/{len(selected)} selected gap rows")
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
            torch.cuda.ipc_collect()
    payload = summarize_model(model, selected, component_rows, summary_rows, causal_rows, rollout_rows, missing_rows)
    write_json(out_dir / f"phase275_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase275_{model}_component_physical_path_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase275_{model}_attention_contribution_rows.jsonl", attn_rows)
    write_jsonl(out_dir / f"phase275_{model}_mlp_contribution_rows.jsonl", mlp_rows)
    write_jsonl(out_dir / f"phase275_{model}_residual_accumulation_rows.jsonl", residual_rows)
    write_jsonl(out_dir / f"phase275_{model}_component_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase275_{model}_causal_fill_rows.jsonl", causal_rows)
    write_jsonl(out_dir / f"phase275_{model}_rollout_fill_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase275_{model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_model(
    model: str,
    selected: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    causal_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    low_side = [r for r in causal_rows if r.get("side_effect_level") == "lower"]
    return {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase275",
        "title": "Selected Phase274 gap batch physical path fill",
        "status": "complete",
        "created_at": utc_now(),
        "model": model,
        "selected_gap_rows": len(selected),
        "component_physical_path_rows": len(component_rows),
        "component_summary_rows": len(summary_rows),
        "causal_fill_rows": len(causal_rows),
        "rollout_fill_rows": len(rollout_rows),
        "missing_rows": len(missing_rows),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in summary_rows)),
        "dominant_positive_component_counts": dict(Counter(str(r.get("dominant_positive_component")) for r in summary_rows)),
        "final_winner_counts": dict(Counter(str(r.get("final_winner")) for r in summary_rows)),
        "strongest_mlp_layers": dict(Counter(str(r.get("strongest_mlp_layer")) for r in summary_rows).most_common()),
        "mean_sum_positive_attn_delta": mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in summary_rows]),
        "mean_sum_positive_mlp_delta": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in summary_rows]),
        "mean_sum_positive_residual_delta": mean_safe([safe_float(r.get("sum_positive_residual_delta")) for r in summary_rows]),
        "causal_effect_supported_counts": dict(Counter(str(r.get("causal_effect_supported")) for r in causal_rows)),
        "side_effect_risk_counts": dict(Counter(str(r.get("side_effect_risk")) for r in causal_rows)),
        "low_side_effect_supported_rate": round(sum(1 for r in low_side if r.get("causal_effect_supported")) / len(low_side), 6) if low_side else 0.0,
        "low_side_effect_risk_rate": round(sum(1 for r in low_side if r.get("side_effect_risk")) / len(low_side), 6) if low_side else 0.0,
    }


def update_v2_files(round_name: str, cross_summary: dict[str, Any]) -> None:
    out_dir = RESULT_ROOT / round_name
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        component_rows.extend(read_jsonl(out_dir / f"phase275_{model}_component_physical_path_rows.jsonl"))
        summary_rows.extend(read_jsonl(out_dir / f"phase275_{model}_component_summary_rows.jsonl"))
        causal_rows.extend(read_jsonl(out_dir / f"phase275_{model}_causal_fill_rows.jsonl"))
        rollout_rows.extend(read_jsonl(out_dir / f"phase275_{model}_rollout_fill_rows.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase275_{model}_missing_rows.jsonl"))
    write_jsonl(V2 / "phase275_component_physical_path_rows.jsonl", component_rows)
    write_jsonl(V2 / "phase275_component_summary_rows.jsonl", summary_rows)
    write_jsonl(V2 / "phase275_causal_fill_rows.jsonl", causal_rows)
    write_jsonl(V2 / "phase275_rollout_fill_rows.jsonl", rollout_rows)
    write_jsonl(V2 / "phase275_missing_rows.jsonl", missing_rows)
    write_json(V2 / "phase275_cross_model_summary.json", cross_summary)

    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files.update(
        {
            "phase275_component_physical_path_rows": "phase275_component_physical_path_rows.jsonl",
            "phase275_component_summary_rows": "phase275_component_summary_rows.jsonl",
            "phase275_causal_fill_rows": "phase275_causal_fill_rows.jsonl",
            "phase275_rollout_fill_rows": "phase275_rollout_fill_rows.jsonl",
            "phase275_missing_rows": "phase275_missing_rows.jsonl",
            "phase275_cross_model_summary": "phase275_cross_model_summary.json",
            "phase275_report": "phase275_report.md",
        }
    )
    manifest["latest_fill_phase"] = "Phase275"
    manifest["phase275_summary"] = cross_summary
    write_json(V2 / "manifest.json", manifest)

    client_index = read_json(V2 / "client_index.json")
    for view in ["fill_results", "causal_fill_audit"]:
        if view not in client_index.setdefault("views", []):
            client_index["views"].append(view)
    for item in ["phase275_cross_model_summary.json", "phase275_component_summary_rows.jsonl", "phase275_causal_fill_rows.jsonl"]:
        if item not in client_index.setdefault("initial_files", []):
            client_index["initial_files"].append(item)
    client_index["phase275_summary_ref"] = "phase275_cross_model_summary.json"
    client_index["phase275_component_summary_ref"] = "phase275_component_summary_rows.jsonl"
    client_index["phase275_causal_fill_ref"] = "phase275_causal_fill_rows.jsonl"
    write_json(V2 / "client_index.json", client_index)

    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase275_component_summary_rows"] = "selected Phase274 queue component path fills"
    tables["phase275_causal_fill_rows"] = "low-side-effect and diagnostic MLP causal audit rows"
    tables["phase275_rollout_fill_rows"] = "short rollout comparison rows for side-effect checks"
    write_json(V2 / "schema.json", schema)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase275_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    component_summary_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for model in MODELS:
        component_summary_rows.extend(read_jsonl(out_dir / f"phase275_{model}_component_summary_rows.jsonl"))
        causal_rows.extend(read_jsonl(out_dir / f"phase275_{model}_causal_fill_rows.jsonl"))
        missing_rows.extend(read_jsonl(out_dir / f"phase275_{model}_missing_rows.jsonl"))
    low_side = [r for r in causal_rows if r.get("side_effect_level") == "lower"]
    payload = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase275",
        "title": "Selected Phase274 gap batch physical path fill",
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
        "progress_estimate": {
            "pattern_family_atlas": 0.52,
            "physical_distribution_puzzle": 0.48,
            "component_path_coverage": 0.32,
            "causal_audit_coverage": 0.21,
            "closure": 0.19,
        },
    }
    write_json(out_dir / "phase275_cross_model_summary.json", payload)
    update_v2_files(round_name, payload)
    write_report(out_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase275 Selected Gap Batch Physical Path Fill",
        "",
        f"- status: {payload['status']}",
        f"- component_summary_rows: {payload['component_summary_rows']}",
        f"- causal_fill_rows: {payload['causal_fill_rows']}",
        f"- missing_rows: {payload['missing_rows']}",
        f"- model_counts: {json.dumps(payload['model_counts'], ensure_ascii=False)}",
        f"- family_counts: {json.dumps(payload['family_counts'], ensure_ascii=False)}",
        f"- dominant_positive_component_counts: {json.dumps(payload['dominant_positive_component_counts'], ensure_ascii=False)}",
        f"- causal_effect_supported_counts: {json.dumps(payload['causal_effect_supported_counts'], ensure_ascii=False)}",
        f"- side_effect_risk_counts: {json.dumps(payload['side_effect_risk_counts'], ensure_ascii=False)}",
        "",
        "This phase consumes Phase274 selected gap rows. It is a physical-path fill batch, not closure.",
    ]
    (out_dir / "phase275_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (V2 / "phase275_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-model", type=int, default=3)
    parser.add_argument("--rollout-tokens", type=int, default=6)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--include-candidate-rows", action="store_true")
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
