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

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase262_continuation_regime_decomposition_atlas as p262  # noqa: E402
from model_utils import get_layers  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_final_norm, get_mlp  # noqa: E402


PHASE = 268
SOURCE_PHASE = 267
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE267_ROOT = Path("tests/result/phase267_multifamily_continuation_physical_path_trace/multifamily_continuation_physical_path_trace")
RESULT_ROOT = Path("tests/result/phase268_attention_mlp_continuation_path_attribution")
ROUND_DEFAULT = "attention_mlp_continuation_path_attribution"


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
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def tensor_at_pos(value: torch.Tensor | None, pos: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 3:
        return value[0, pos]
    if value.ndim == 2:
        return value[pos]
    if value.ndim == 1:
        return value
    return None


def project_state(model_obj: Any, final_norm: Any | None, state: torch.Tensor) -> torch.Tensor:
    param = next(model_obj.parameters())
    h = state.detach().clone().to(device=param.device, dtype=param.dtype)
    with torch.no_grad():
        if final_norm is not None:
            h = final_norm(h)
        logits = model_obj.get_output_embeddings()(h)
    return logits.detach().float().cpu()


def group_scores(logits: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    for name, ids in groups.items():
        valid = [int(x) for x in ids if 0 <= int(x) < logits.numel()]
        if not valid:
            out[name] = (-1e30, -1)
            continue
        idx = torch.tensor(valid, dtype=torch.long)
        vals = logits[idx]
        pos = int(torch.argmax(vals).item())
        out[name] = (float(vals[pos].item()), int(idx[pos].item()))
    return out


def margin_from_logits(logits: torch.Tensor, stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]]) -> dict[str, Any]:
    stop_scores = group_scores(logits, stop_ids)
    cont_scores = group_scores(logits, cont_ids)
    best_stop_name, (best_stop, best_stop_id) = max(stop_scores.items(), key=lambda kv: kv[1][0])
    best_cont_name, (best_cont, best_cont_id) = max(cont_scores.items(), key=lambda kv: kv[1][0])
    return {
        "r_stop": round(best_stop, 6),
        "r_stop_name": best_stop_name,
        "r_stop_token_id": best_stop_id,
        "r_continue": round(best_cont, 6),
        "r_continue_name": best_cont_name,
        "r_continue_token_id": best_cont_id,
        "continue_stop_margin": round(best_cont - best_stop, 6),
        "stop_continue_margin": round(best_stop - best_cont, 6),
        "competition_winner": "continue" if best_cont >= best_stop else "stop",
    }


def select_cases(model: str, per_model: int) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(PHASE267_ROOT / "phase267_physical_path_rows.jsonl") if r.get("model") == model]
    rows = [r for r in rows if r.get("stable_continue_from_layer") is not None]
    rows.sort(key=lambda r: (-safe_float(r.get("final_continue_stop_margin")), str(r.get("family_id")), str(r.get("case_id"))))
    selected: list[dict[str, Any]] = []
    used_cases: set[str] = set()
    used_families: set[str] = set()
    used_channels: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        if len(selected) >= per_model or row["case_id"] in used_cases:
            return False
        selected.append(row)
        used_cases.add(str(row["case_id"]))
        used_families.add(str(row["family_id"]))
        used_channels.add(str(row.get("top_continue_channel_phase266")))
        return True

    for row in rows:
        if str(row.get("top_continue_channel_phase266")) not in used_channels:
            add(row)
    for row in rows:
        if str(row.get("family_id")) not in used_families:
            add(row)
    for row in rows:
        add(row)
    return selected[:per_model]


def capture_components(model_obj: Any, tokenizer: Any, device: torch.device, prompt: str) -> tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor, int]:
    layers = get_layers(model_obj)
    captured: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(len(layers))}
    handles = []

    for li, layer in enumerate(layers):
        def layer_pre(_module, inputs, layer_idx=li):
            captured[layer_idx]["layer_input"] = inputs[0].detach().float().cpu()

        def layer_out(_module, _inputs, output, layer_idx=li):
            y = extract_tensor(output)
            if y is not None:
                captured[layer_idx]["layer_out"] = y.detach().float().cpu()

        handles.append(layer.register_forward_pre_hook(layer_pre))
        handles.append(layer.register_forward_hook(layer_out))

        attn = get_attn(layer)
        if attn is not None:
            def attn_out(_module, _inputs, output, layer_idx=li):
                y = extract_tensor(output)
                if y is not None:
                    captured[layer_idx]["attn_out"] = y.detach().float().cpu()

            handles.append(attn.register_forward_hook(attn_out))

        mlp = get_mlp(layer)
        if mlp is not None:
            def mlp_out(_module, _inputs, output, layer_idx=li):
                y = extract_tensor(output)
                if y is not None:
                    captured[layer_idx]["mlp_out"] = y.detach().float().cpu()

            handles.append(mlp.register_forward_hook(mlp_out))

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            out = model_obj(**encoded, use_cache=False, return_dict=True)
        final_logits = out.logits[0, last_pos].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return captured, final_logits, last_pos


def component_decomposition(model_obj: Any, tokenizer: Any, device: torch.device, case: dict[str, Any], stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_bank = {r["case_id"]: r for r in read_jsonl(ATLAS_ROOT / "mode_family_case_bank_v3.jsonl")}
    source = case_bank[str(case["case_id"])]
    captured, final_logits, last_pos = capture_components(model_obj, tokenizer, device, str(source["prompt"]))
    final_norm = get_final_norm(model_obj)
    component_rows: list[dict[str, Any]] = []
    attn_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    base = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase268",
        "created_at": utc_now(),
        "model": case["model"],
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "mode_id": case["mode_id"],
        "variant_id": case["variant_id"],
        "path_schema_id": case["path_schema_id"],
        "top_continue_channel_phase267": case.get("top_continue_channel_phase266"),
    }
    final_readout = margin_from_logits(final_logits, stop_ids, cont_ids)
    layer_summaries: list[dict[str, Any]] = []
    for layer_idx in sorted(captured):
        comp = captured[layer_idx]
        h0 = tensor_at_pos(comp.get("layer_input"), last_pos)
        attn = tensor_at_pos(comp.get("attn_out"), last_pos)
        mlp = tensor_at_pos(comp.get("mlp_out"), last_pos)
        layer_out = tensor_at_pos(comp.get("layer_out"), last_pos)
        if h0 is None or layer_out is None:
            continue
        h0_read = margin_from_logits(project_state(model_obj, final_norm, h0), stop_ids, cont_ids)
        h_attn = h0 + attn if attn is not None else h0
        h_attn_read = margin_from_logits(project_state(model_obj, final_norm, h_attn), stop_ids, cont_ids)
        h_mlp = h_attn + mlp if mlp is not None else h_attn
        h_mlp_read = margin_from_logits(project_state(model_obj, final_norm, h_mlp), stop_ids, cont_ids)
        h_out_read = margin_from_logits(project_state(model_obj, final_norm, layer_out), stop_ids, cont_ids)
        m0 = safe_float(h0_read["continue_stop_margin"])
        ma = safe_float(h_attn_read["continue_stop_margin"])
        mm = safe_float(h_mlp_read["continue_stop_margin"])
        mo = safe_float(h_out_read["continue_stop_margin"])
        row = {
            **base,
            "component_physical_path_id": f"phase268:component_path:{case['model']}:{case['case_id']}:L{layer_idx}",
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
        component_rows.append(row)
        layer_summaries.append(row)
        attn_rows.append(
            {
                **base,
                "attention_contribution_id": f"phase268:attn:{case['model']}:{case['case_id']}:L{layer_idx}",
                "layer_index": layer_idx,
                "delta_continue_stop_margin": row["delta_attn_continue_stop_margin"],
                "before_margin": row["input_continue_stop_margin"],
                "after_margin": row["after_attn_continue_stop_margin"],
                "component_available": attn is not None,
            }
        )
        mlp_rows.append(
            {
                **base,
                "mlp_contribution_id": f"phase268:mlp:{case['model']}:{case['case_id']}:L{layer_idx}",
                "layer_index": layer_idx,
                "delta_continue_stop_margin": row["delta_mlp_continue_stop_margin"],
                "before_margin": row["after_attn_continue_stop_margin"],
                "after_margin": row["after_mlp_continue_stop_margin"],
                "component_available": mlp is not None,
            }
        )
        residual_rows.append(
            {
                **base,
                "residual_accumulation_id": f"phase268:residual:{case['model']}:{case['case_id']}:L{layer_idx}",
                "layer_index": layer_idx,
                "delta_continue_stop_margin": row["delta_residual_carry_margin"],
                "before_margin": row["after_mlp_continue_stop_margin"],
                "after_margin": row["layer_out_continue_stop_margin"],
                "note": "layer_out - (layer_input + attn_out + mlp_out), observational carry estimate",
            }
        )
    strongest_attn = max(layer_summaries, key=lambda r: safe_float(r["delta_attn_continue_stop_margin"]), default={})
    strongest_mlp = max(layer_summaries, key=lambda r: safe_float(r["delta_mlp_continue_stop_margin"]), default={})
    strongest_resid = max(layer_summaries, key=lambda r: safe_float(r["delta_residual_carry_margin"]), default={})
    summary = {
        **base,
        "component_summary_id": f"phase268:summary:{case['model']}:{case['case_id']}",
        "layers_observed": len(layer_summaries),
        "final_continue_stop_margin": final_readout["continue_stop_margin"],
        "final_winner": final_readout["competition_winner"],
        "sum_positive_attn_delta": round(sum(max(0.0, safe_float(r["delta_attn_continue_stop_margin"])) for r in layer_summaries), 6),
        "sum_positive_mlp_delta": round(sum(max(0.0, safe_float(r["delta_mlp_continue_stop_margin"])) for r in layer_summaries), 6),
        "sum_positive_residual_delta": round(sum(max(0.0, safe_float(r["delta_residual_carry_margin"])) for r in layer_summaries), 6),
        "sum_signed_attn_delta": round(sum(safe_float(r["delta_attn_continue_stop_margin"]) for r in layer_summaries), 6),
        "sum_signed_mlp_delta": round(sum(safe_float(r["delta_mlp_continue_stop_margin"]) for r in layer_summaries), 6),
        "sum_signed_residual_delta": round(sum(safe_float(r["delta_residual_carry_margin"]) for r in layer_summaries), 6),
        "strongest_attn_layer": strongest_attn.get("layer_index"),
        "strongest_attn_delta": strongest_attn.get("delta_attn_continue_stop_margin"),
        "strongest_mlp_layer": strongest_mlp.get("layer_index"),
        "strongest_mlp_delta": strongest_mlp.get("delta_mlp_continue_stop_margin"),
        "strongest_residual_layer": strongest_resid.get("layer_index"),
        "strongest_residual_delta": strongest_resid.get("delta_residual_carry_margin"),
    }
    positives = {
        "attention": safe_float(summary["sum_positive_attn_delta"]),
        "mlp": safe_float(summary["sum_positive_mlp_delta"]),
        "residual": safe_float(summary["sum_positive_residual_delta"]),
    }
    summary["dominant_positive_component"] = max(positives.items(), key=lambda kv: kv[1])[0] if positives else "none"
    return component_rows, attn_rows, mlp_rows, residual_rows, summary


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_cases(args.model, int(args.cases_per_model))
    for row in selected:
        row["model"] = args.model
    model_obj = None
    tokenizer = None
    component_rows: list[dict[str, Any]] = []
    attn_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
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
        stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
        cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
        for idx, case in enumerate(selected, start=1):
            try:
                comp, attn, mlp, resid, summary = component_decomposition(model_obj, tokenizer, device, case, stop_ids, cont_ids)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase268",
                        "created_at": utc_now(),
                        "missing_id": f"phase268:missing:{args.model}:{case['case_id']}",
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case["family_id"],
                        "reason": repr(exc),
                    }
                )
                continue
            component_rows.extend(comp)
            attn_rows.extend(attn)
            mlp_rows.extend(mlp)
            residual_rows.extend(resid)
            summary_rows.append(summary)
            observations.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase268",
                    "created_at": utc_now(),
                    "observation_id": f"phase268:obs:{args.model}:{case['case_id']}",
                    "case_id": case["case_id"],
                    "model": args.model,
                    "family_id": case["family_id"],
                    "level": "attention_mlp_component_attribution",
                    "component": summary["dominant_positive_component"],
                    "metric_name": "final_continue_stop_margin",
                    "metric_value": summary["final_continue_stop_margin"],
                    "metric_unit": "logit_margin",
                    "winner": summary["final_winner"],
                }
            )
            log(f"{args.model}: component traced {idx}/{len(selected)} cases")
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
    metrics = make_metrics(args.model, summary_rows)
    edges = make_edges(args.model, summary_rows)
    payload = summarize_model(args.model, selected, component_rows, attn_rows, mlp_rows, residual_rows, summary_rows, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, payload, component_rows, attn_rows, mlp_rows, residual_rows, summary_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_metrics(model: str, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_family[str(row["family_id"])].append(row)
    for family, vals in sorted(by_family.items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase268",
                "created_at": utc_now(),
                "metric_id": f"phase268:{model}:{family}:component_positive_sums",
                "scope": "component_path_attribution",
                "model": model,
                "family_id": family,
                "metric_name": "component_positive_delta_sums",
                "metric_value": mean_safe([safe_float(r["sum_positive_attn_delta"]) for r in vals]),
                "mean_positive_attn_delta": mean_safe([safe_float(r["sum_positive_attn_delta"]) for r in vals]),
                "mean_positive_mlp_delta": mean_safe([safe_float(r["sum_positive_mlp_delta"]) for r in vals]),
                "mean_positive_residual_delta": mean_safe([safe_float(r["sum_positive_residual_delta"]) for r in vals]),
                "rows": len(vals),
            }
        )
    return rows


def make_edges(model: str, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    grouped = Counter((r["family_id"], r["dominant_positive_component"]) for r in summaries)
    for (family, component), count in grouped.items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase268",
                "created_at": utc_now(),
                "edge_id": f"phase268:{model}:{family}:{component}",
                "source": f"node:{family}",
                "target": f"node:{component}",
                "edge_type": "family_to_dominant_continuation_component",
                "model": model,
                "evidence_type": "observational_attention_mlp_margin_decomposition",
                "effect_size": count,
                "status": "observational_not_causal_ablation",
            }
        )
    return edges


def summarize_model(model: str, selected: list[dict[str, Any]], component_rows: list[dict[str, Any]], attn_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Attention/MLP separated continuation path attribution",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "selected_cases": len(selected),
        "component_physical_path_rows": len(component_rows),
        "attention_contribution_rows": len(attn_rows),
        "mlp_contribution_rows": len(mlp_rows),
        "residual_accumulation_rows": len(residual_rows),
        "component_summary_rows": len(summary_rows),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "family_counts": dict(Counter(str(r["family_id"]) for r in summary_rows)),
        "channel_counts": dict(Counter(str(r.get("top_continue_channel_phase267")) for r in summary_rows).most_common()),
        "dominant_positive_component_counts": dict(Counter(str(r.get("dominant_positive_component")) for r in summary_rows)),
        "final_winner_counts": dict(Counter(str(r.get("final_winner")) for r in summary_rows)),
        "mean_final_continue_stop_margin": mean_safe([safe_float(r["final_continue_stop_margin"]) for r in summary_rows]),
        "mean_sum_positive_attn_delta": mean_safe([safe_float(r["sum_positive_attn_delta"]) for r in summary_rows]),
        "mean_sum_positive_mlp_delta": mean_safe([safe_float(r["sum_positive_mlp_delta"]) for r in summary_rows]),
        "mean_sum_positive_residual_delta": mean_safe([safe_float(r["sum_positive_residual_delta"]) for r in summary_rows]),
        "strongest_attn_layers": dict(Counter(str(r.get("strongest_attn_layer")) for r in summary_rows).most_common()),
        "strongest_mlp_layers": dict(Counter(str(r.get("strongest_mlp_layer")) for r in summary_rows).most_common()),
    }


def write_model_outputs(out_dir: Path, model: str, summary: dict[str, Any], component_rows: list[dict[str, Any]], attn_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase268_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase268_{model}_component_physical_path_rows.jsonl", component_rows)
    write_jsonl(out_dir / f"phase268_{model}_attention_contribution_rows.jsonl", attn_rows)
    write_jsonl(out_dir / f"phase268_{model}_mlp_contribution_rows.jsonl", mlp_rows)
    write_jsonl(out_dir / f"phase268_{model}_residual_accumulation_rows.jsonl", residual_rows)
    write_jsonl(out_dir / f"phase268_{model}_component_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase268_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase268_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase268_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase268_{model}_missing_rows.jsonl", missing)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase268_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    component_rows: list[dict[str, Any]] = []
    attn_rows: list[dict[str, Any]] = []
    mlp_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        component_rows.extend(read_jsonl(out_dir / f"phase268_{model}_component_physical_path_rows.jsonl"))
        attn_rows.extend(read_jsonl(out_dir / f"phase268_{model}_attention_contribution_rows.jsonl"))
        mlp_rows.extend(read_jsonl(out_dir / f"phase268_{model}_mlp_contribution_rows.jsonl"))
        residual_rows.extend(read_jsonl(out_dir / f"phase268_{model}_residual_accumulation_rows.jsonl"))
        summary_rows.extend(read_jsonl(out_dir / f"phase268_{model}_component_summary_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase268_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase268_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase268_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase268_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.90,
        "physical_path_atlas": 0.37,
        "multi_family_case_bank": 0.45,
        "multi_family_baseline_scan": 0.18,
        "state_factor_atlas": 0.39,
        "path_cluster_mining": 0.17,
        "trace_signature_validation": 0.52,
        "readout_competition_trace": 0.80,
        "component_path_atlas": 0.16,
        "stepwise_rollout_trace": 0.44,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.69,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Attention/MLP separated continuation path attribution",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "component_physical_path_rows": len(component_rows),
        "attention_contribution_rows": len(attn_rows),
        "mlp_contribution_rows": len(mlp_rows),
        "residual_accumulation_rows": len(residual_rows),
        "component_summary_rows": len(summary_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "family_counts": dict(Counter(str(r["family_id"]) for r in summary_rows)),
        "channel_counts": dict(Counter(str(r.get("top_continue_channel_phase267")) for r in summary_rows).most_common()),
        "dominant_positive_component_counts": dict(Counter(str(r.get("dominant_positive_component")) for r in summary_rows)),
        "final_winner_counts": dict(Counter(str(r.get("final_winner")) for r in summary_rows)),
        "mean_final_continue_stop_margin": mean_safe([safe_float(r["final_continue_stop_margin"]) for r in summary_rows]),
        "mean_sum_positive_attn_delta": mean_safe([safe_float(r["sum_positive_attn_delta"]) for r in summary_rows]),
        "mean_sum_positive_mlp_delta": mean_safe([safe_float(r["sum_positive_mlp_delta"]) for r in summary_rows]),
        "mean_sum_positive_residual_delta": mean_safe([safe_float(r["sum_positive_residual_delta"]) for r in summary_rows]),
        "strongest_attn_layers": dict(Counter(str(r.get("strongest_attn_layer")) for r in summary_rows).most_common()),
        "strongest_mlp_layers": dict(Counter(str(r.get("strongest_mlp_layer")) for r in summary_rows).most_common()),
        "progress": progress,
    }
    write_json(out_dir / "phase268_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase268_component_physical_path_rows.jsonl", component_rows)
    write_jsonl(out_dir / "phase268_attention_contribution_rows.jsonl", attn_rows)
    write_jsonl(out_dir / "phase268_mlp_contribution_rows.jsonl", mlp_rows)
    write_jsonl(out_dir / "phase268_residual_accumulation_rows.jsonl", residual_rows)
    write_jsonl(out_dir / "phase268_component_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / "phase268_observations.jsonl", observations)
    write_jsonl(out_dir / "phase268_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase268_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase268_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "phase268_component_physical_path_rows.jsonl", component_rows)
    write_jsonl(ATLAS_ROOT / "phase268_attention_contribution_rows.jsonl", attn_rows)
    write_jsonl(ATLAS_ROOT / "phase268_mlp_contribution_rows.jsonl", mlp_rows)
    write_jsonl(ATLAS_ROOT / "phase268_residual_accumulation_rows.jsonl", residual_rows)
    write_jsonl(ATLAS_ROOT / "phase268_component_summary_rows.jsonl", summary_rows)
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase268", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase268 Attention/MLP Separated Continuation Path Attribution",
        "",
        f"- status: {payload['status']}",
        f"- component_physical_path_rows: {payload['component_physical_path_rows']}",
        f"- attention_contribution_rows: {payload['attention_contribution_rows']}",
        f"- mlp_contribution_rows: {payload['mlp_contribution_rows']}",
        f"- residual_accumulation_rows: {payload['residual_accumulation_rows']}",
        f"- component_summary_rows: {payload['component_summary_rows']}",
        f"- channel_counts: {json.dumps(payload['channel_counts'], ensure_ascii=False)}",
        f"- dominant_positive_component_counts: {json.dumps(payload['dominant_positive_component_counts'], ensure_ascii=False)}",
        f"- final_winner_counts: {json.dumps(payload['final_winner_counts'], ensure_ascii=False)}",
        f"- mean_final_continue_stop_margin: {payload['mean_final_continue_stop_margin']}",
        f"- mean_sum_positive_attn_delta: {payload['mean_sum_positive_attn_delta']}",
        f"- mean_sum_positive_mlp_delta: {payload['mean_sum_positive_mlp_delta']}",
        f"- mean_sum_positive_residual_delta: {payload['mean_sum_positive_residual_delta']}",
        "",
        "Note: This is observational component attribution, not causal ablation or closure.",
    ]
    (out_dir / "phase268_attention_mlp_continuation_path_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--cases-per-model", type=int, default=6)
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
