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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase268_attention_mlp_continuation_path_attribution as p268  # noqa: E402
import phase262_continuation_regime_decomposition_atlas as p262  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402

PHASE = 296
SCHEMA_VERSION = "2.23.0"
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase296_expanded_component_path_probe"
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


def prompt_index() -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("model")), str(r.get("case_id"))): r for r in read_jsonl(V2 / "phase292_feature_priority_queue_rows.jsonl")}


def select_model_rows(model: str, limit: int) -> list[dict[str, Any]]:
    prompts = prompt_index()
    gaps = [r for r in read_jsonl(V2 / "phase294_expanded_gap_rows.jsonl") if r.get("model") == model]
    gaps = [r for r in gaps if (model, str(r.get("case_id"))) in prompts]
    features = {(str(r.get("family_id")), str(r.get("model"))): r for r in read_jsonl(V2 / "phase295_feature_matrix_v3_rows.jsonl")}
    for row in gaps:
        f = features.get((str(row.get("family_id")), model), {})
        row["_priority"] = safe_float(row.get("priority_score")) + 5.0 * (1.0 - safe_float(f.get("atlas_completion_v3"))) + (2.0 if f.get("next_priority") == "hard_readout_stop_failure" else 0.0)
    gaps.sort(key=lambda r: (-safe_float(r.get("_priority")), str(r.get("family_id")), str(r.get("case_id"))))
    selected: list[dict[str, Any]] = []
    used_families: Counter[str] = Counter()
    for row in gaps:
        if len(selected) >= limit:
            break
        fam = str(row.get("family_id"))
        if used_families[fam] >= max(1, limit // 9):
            continue
        selected.append(row)
        used_families[fam] += 1
    for row in gaps:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    return selected[:limit]


def decompose_prompt(model_obj: Any, tokenizer: Any, device: torch.device, row: dict[str, Any], prompt_row: dict[str, Any], stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt = str(prompt_row["prompt"])
    captured, final_logits, last_pos = p268.capture_components(model_obj, tokenizer, device, prompt)
    final_norm = p268.get_final_norm(model_obj)
    component_rows: list[dict[str, Any]] = []
    attn_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    base = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase296",
        "created_at": now(),
        "model": row["model"],
        "case_id": row["case_id"],
        "family_id": row["family_id"],
        "mode_id": row.get("mode_id"),
        "variant_id": row.get("variant_id"),
        "path_schema_id": prompt_row.get("path_schema_id"),
        "channel_focus": prompt_row.get("channel_focus"),
        "target": prompt_row.get("target"),
    }
    final_readout = p268.margin_from_logits(final_logits, stop_ids, cont_ids)
    layer_summaries: list[dict[str, Any]] = []
    for layer_idx in sorted(captured):
        comp = captured[layer_idx]
        h0 = p268.tensor_at_pos(comp.get("layer_input"), last_pos)
        attn = p268.tensor_at_pos(comp.get("attn_out"), last_pos)
        mlp = p268.tensor_at_pos(comp.get("mlp_out"), last_pos)
        layer_out = p268.tensor_at_pos(comp.get("layer_out"), last_pos)
        if h0 is None or layer_out is None:
            continue
        h0_read = p268.margin_from_logits(p268.project_state(model_obj, final_norm, h0), stop_ids, cont_ids)
        h_attn = h0 + attn if attn is not None else h0
        h_attn_read = p268.margin_from_logits(p268.project_state(model_obj, final_norm, h_attn), stop_ids, cont_ids)
        h_mlp = h_attn + mlp if mlp is not None else h_attn
        h_mlp_read = p268.margin_from_logits(p268.project_state(model_obj, final_norm, h_mlp), stop_ids, cont_ids)
        h_out_read = p268.margin_from_logits(p268.project_state(model_obj, final_norm, layer_out), stop_ids, cont_ids)
        m0 = safe_float(h0_read["continue_stop_margin"])
        ma = safe_float(h_attn_read["continue_stop_margin"])
        mm = safe_float(h_mlp_read["continue_stop_margin"])
        mo = safe_float(h_out_read["continue_stop_margin"])
        crow = {
            **base,
            "component_physical_path_id": f"phase296:component_path:{row['model']}:{row['case_id']}:L{layer_idx}",
            "layer_index": layer_idx,
            "input_continue_stop_margin": round(m0, 6),
            "after_attn_continue_stop_margin": round(ma, 6),
            "after_mlp_continue_stop_margin": round(mm, 6),
            "layer_out_continue_stop_margin": round(mo, 6),
            "delta_attn_continue_stop_margin": round(ma - m0, 6),
            "delta_mlp_continue_stop_margin": round(mm - ma, 6),
            "delta_residual_carry_margin": round(mo - mm, 6),
            "attn_available": attn is not None,
            "mlp_available": mlp is not None,
            "layer_out_winner": h_out_read["competition_winner"],
        }
        component_rows.append(crow)
        layer_summaries.append(crow)
        attn_rows.append({**base, "attention_contribution_id": f"phase296:attn:{row['model']}:{row['case_id']}:L{layer_idx}", "layer_index": layer_idx, "delta_continue_stop_margin": crow["delta_attn_continue_stop_margin"], "component_available": attn is not None})
        mlp_rows.append({**base, "mlp_contribution_id": f"phase296:mlp:{row['model']}:{row['case_id']}:L{layer_idx}", "layer_index": layer_idx, "delta_continue_stop_margin": crow["delta_mlp_continue_stop_margin"], "component_available": mlp is not None})
        residual_rows.append({**base, "residual_accumulation_id": f"phase296:residual:{row['model']}:{row['case_id']}:L{layer_idx}", "layer_index": layer_idx, "delta_continue_stop_margin": crow["delta_residual_carry_margin"]})
    strongest_attn = max(layer_summaries, key=lambda r: safe_float(r["delta_attn_continue_stop_margin"]), default={})
    strongest_mlp = max(layer_summaries, key=lambda r: safe_float(r["delta_mlp_continue_stop_margin"]), default={})
    strongest_resid = max(layer_summaries, key=lambda r: safe_float(r["delta_residual_carry_margin"]), default={})
    summary = {
        **base,
        "component_summary_id": f"phase296:summary:{row['model']}:{row['case_id']}",
        "layers_observed": len(layer_summaries),
        "final_continue_stop_margin": final_readout["continue_stop_margin"],
        "final_winner": final_readout["competition_winner"],
        "sum_positive_attn_delta": round(sum(max(0.0, safe_float(r["delta_attn_continue_stop_margin"])) for r in layer_summaries), 6),
        "sum_positive_mlp_delta": round(sum(max(0.0, safe_float(r["delta_mlp_continue_stop_margin"])) for r in layer_summaries), 6),
        "sum_positive_residual_delta": round(sum(max(0.0, safe_float(r["delta_residual_carry_margin"])) for r in layer_summaries), 6),
        "strongest_attn_layer": strongest_attn.get("layer_index"),
        "strongest_attn_delta": strongest_attn.get("delta_attn_continue_stop_margin"),
        "strongest_mlp_layer": strongest_mlp.get("layer_index"),
        "strongest_mlp_delta": strongest_mlp.get("delta_mlp_continue_stop_margin"),
        "strongest_residual_layer": strongest_resid.get("layer_index"),
        "strongest_residual_delta": strongest_resid.get("delta_residual_carry_margin"),
    }
    positives = {"attention": safe_float(summary["sum_positive_attn_delta"]), "mlp": safe_float(summary["sum_positive_mlp_delta"]), "residual": safe_float(summary["sum_positive_residual_delta"])}
    summary["dominant_positive_component"] = max(positives.items(), key=lambda kv: kv[1])[0]
    return component_rows, attn_rows, mlp_rows, residual_rows, summary


def run_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    out_dir = OUT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = prompt_index()
    selected = select_model_rows(model, args.cases_per_model)
    model_obj = tokenizer = None
    component_rows: list[dict[str, Any]] = []
    attn_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
        cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
        for idx, row in enumerate(selected, 1):
            try:
                prompt_row = prompts[(model, str(row["case_id"]))]
                comp, attn, mlp, resid, summary = decompose_prompt(model_obj, tokenizer, device, row, prompt_row, stop_ids, cont_ids)
                component_rows.extend(comp)
                attn_rows.extend(attn)
                mlp_rows.extend(mlp)
                residual_rows.extend(resid)
                summary_rows.append(summary)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append({"schema_version": SCHEMA_VERSION, "phase_id": "Phase296", "created_at": now(), "model": model, "case_id": row.get("case_id"), "family_id": row.get("family_id"), "reason": repr(exc)})
            print(f"{model}: expanded component traced {idx}/{len(selected)}", flush=True)
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
    payload = summarize(model, selected, summary_rows, missing_rows)
    write_json(out_dir / f"phase296_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase296_{model}_component_physical_path_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase296_{model}_attention_contribution_rows.jsonl", attn_rows)
    write_jsonl(out_dir / f"phase296_{model}_mlp_contribution_rows.jsonl", mlp_rows)
    write_jsonl(out_dir / f"phase296_{model}_residual_accumulation_rows.jsonl", residual_rows)
    write_jsonl(out_dir / f"phase296_{model}_component_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase296_{model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize(model: str, selected: list[dict[str, Any]], summaries: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase296",
        "created_at": now(),
        "model": model,
        "selected_rows": len(selected),
        "component_summary_rows": len(summaries),
        "missing_rows": len(missing),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in summaries)),
        "dominant_positive_component_counts": dict(Counter(str(r.get("dominant_positive_component")) for r in summaries)),
        "final_winner_counts": dict(Counter(str(r.get("final_winner")) for r in summaries)),
        "mean_sum_positive_attn_delta": mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in summaries]),
        "mean_sum_positive_mlp_delta": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in summaries]),
        "mean_sum_positive_residual_delta": mean_safe([safe_float(r.get("sum_positive_residual_delta")) for r in summaries]),
    }


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    component = []
    attn = []
    mlp = []
    resid = []
    summaries = []
    missing = []
    model_summaries = []
    for model in MODELS:
        component.extend(read_jsonl(out_dir / f"phase296_{model}_component_physical_path_rows.jsonl"))
        attn.extend(read_jsonl(out_dir / f"phase296_{model}_attention_contribution_rows.jsonl"))
        mlp.extend(read_jsonl(out_dir / f"phase296_{model}_mlp_contribution_rows.jsonl"))
        resid.extend(read_jsonl(out_dir / f"phase296_{model}_residual_accumulation_rows.jsonl"))
        summaries.extend(read_jsonl(out_dir / f"phase296_{model}_component_summary_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase296_{model}_missing_rows.jsonl"))
        ms = read_json(out_dir / f"phase296_{model}_summary.json")
        if ms:
            model_summaries.append(ms)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase296",
        "created_at": now(),
        "model_summaries": model_summaries,
        "component_physical_path_rows": len(component),
        "attention_contribution_rows": len(attn),
        "mlp_contribution_rows": len(mlp),
        "residual_accumulation_rows": len(resid),
        "component_summary_rows": len(summaries),
        "missing_rows": len(missing),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in summaries)),
        "model_counts": dict(Counter(str(r.get("model")) for r in summaries)),
        "dominant_positive_component_counts": dict(Counter(str(r.get("dominant_positive_component")) for r in summaries)),
        "final_winner_counts": dict(Counter(str(r.get("final_winner")) for r in summaries)),
        "mean_sum_positive_attn_delta": mean_safe([safe_float(r.get("sum_positive_attn_delta")) for r in summaries]),
        "mean_sum_positive_mlp_delta": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in summaries]),
        "mean_sum_positive_residual_delta": mean_safe([safe_float(r.get("sum_positive_residual_delta")) for r in summaries]),
        "progress_estimate": {
            "pattern_family_atlas": 0.76,
            "sample_type_coverage": 0.68,
            "feature_mining": 0.59,
            "physical_distribution_puzzle": 0.69,
            "mechanism_audit": 0.46,
            "closure": 0.21,
        },
    }
    files = {
        "phase296_component_physical_path_rows.jsonl": component,
        "phase296_attention_contribution_rows.jsonl": attn,
        "phase296_mlp_contribution_rows.jsonl": mlp,
        "phase296_residual_accumulation_rows.jsonl": resid,
        "phase296_component_summary_rows.jsonl": summaries,
        "phase296_missing_rows.jsonl": missing,
    }
    for name, rows in files.items():
        write_jsonl(OUT / name, rows)
        write_jsonl(V2 / name, rows)
    write_json(OUT / "phase296_summary.json", payload)
    write_json(V2 / "phase296_summary.json", payload)
    report = "\n".join([
        "# Phase296 Expanded Component Path Probe",
        "",
        f"- component_summary_rows: {payload['component_summary_rows']}",
        f"- missing_rows: {payload['missing_rows']}",
        f"- dominant_positive_component_counts: {json.dumps(payload['dominant_positive_component_counts'], ensure_ascii=False)}",
        f"- final_winner_counts: {json.dumps(payload['final_winner_counts'], ensure_ascii=False)}",
        f"- model_counts: {json.dumps(payload['model_counts'], ensure_ascii=False)}",
    ]) + "\n"
    (OUT / "phase296_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase296_report.md").write_text(report, encoding="utf-8")
    update_v2(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def update_v2(payload: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for name in [
        "phase296_component_physical_path_rows",
        "phase296_attention_contribution_rows",
        "phase296_mlp_contribution_rows",
        "phase296_residual_accumulation_rows",
        "phase296_component_summary_rows",
        "phase296_missing_rows",
    ]:
        files[name] = f"{name}.jsonl"
    files["phase296_summary"] = "phase296_summary.json"
    files["phase296_report"] = "phase296_report.md"
    manifest["latest_expanded_component_phase"] = "Phase296"
    manifest["phase296_summary"] = payload
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase296_summary.json", "phase296_component_summary_rows.jsonl", "phase296_mlp_contribution_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase296_summary_ref"] = "phase296_summary.json"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    schema.setdefault("tables", {})["phase296_component_summary_rows"] = "expanded sample component path summary rows"
    schema.setdefault("tables", {})["phase296_component_physical_path_rows"] = "expanded sample layerwise attention/mlp/residual component path rows"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--cases-per-model", type=int, default=9)
    parser.add_argument("--round-name", default="expanded_component_path_probe")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        collect(args.round_name)
        return
    if args.model:
        run_model(args, args.model)
        return
    for model in MODELS:
        run_model(args, model)
    collect(args.round_name)


if __name__ == "__main__":
    main()
