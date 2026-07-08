#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
import phase252_shared_subspace_coupled_regime_analysis as p252  # noqa: E402
import phase259_template_semantic_done_disentanglement as p259  # noqa: E402


PHASE = 260
SOURCE_PHASE = 259
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase260_sdone_rstop_bridge_layer_localization")
ROUND_DEFAULT = "sdone_rstop_bridge_layer_localization"

SPECS = {
    "qwen3": {"observe_layers": [12, 20, 26, 29, 31, 33]},
    "glm4": {"observe_layers": [12, 20, 26, 28, 30, 32]},
    "deepseek7b": {"observe_layers": [10, 16, 22, 24, 26, 27]},
}

AXIS_PAIRS = {
    "template_done": ("template_complete_semantic_correct", "template_incomplete_semantic_correct"),
    "semantic_done": ("template_complete_semantic_correct", "template_complete_semantic_wrong"),
    "boundary_done": ("boundary_complete_semantic_correct", "template_incomplete_semantic_correct"),
}


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
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def unit(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n


def dot(vec: torch.Tensor, direction: torch.Tensor) -> float:
    if vec.numel() != direction.numel():
        return 0.0
    return float(torch.dot(vec.float(), unit(direction).float()).item())


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def capture_all_layers(model_obj: Any, tokenizer: Any, device: torch.device, text: str, observe_layers: list[int], aliases: list[str]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden = {}
    for layer in observe_layers:
        if int(layer) + 1 < len(out.hidden_states):
            hidden[int(layer)] = out.hidden_states[int(layer) + 1][0, last_pos].detach().float().cpu()
    logits = out.logits[0, last_pos].detach().float().cpu()
    closure = p252.closure_scores(tokenizer, logits)
    readout = p239.readout_metrics(tokenizer, logits, aliases)
    eos_logit = logits[int(tokenizer.eos_token_id)].item() if tokenizer.eos_token_id is not None else 0.0
    return hidden, {"eos_logit": round(float(eos_logit), 6), **{f"closure_{k}": round(v, 6) for k, v in closure.items()}, **{f"readout_{k}": v for k, v in readout.items()}}


def positive_rate(values: list[float]) -> float:
    return round(sum(1 for x in values if x > 0) / len(values), 6) if values else 0.0


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_rows = p259.load_rows(args.model, int(args.max_cases_per_mode))
    observe_layers = list(SPECS[args.model]["observe_layers"])
    model_obj = None
    tokenizer = None
    vector_rows: list[dict[str, Any]] = []
    case_layer_rows: list[dict[str, Any]] = []
    layer_summary_rows: list[dict[str, Any]] = []
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
        hidden_by_case: dict[tuple[str, str, str], dict[str, dict[int, torch.Tensor]]] = {}
        extra_by_case: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        for idx, row in enumerate(behavior_rows, start=1):
            key = (str(row["case_id"]), str(row["variant_id"]), str(row["mode_id"]))
            aliases = list(row.get("target_aliases") or [])
            hidden_by_case[key] = {}
            extra_by_case[key] = {}
            for condition, text in p259.condition_texts(row).items():
                hidden, extra = capture_all_layers(model_obj, tokenizer, device, text, observe_layers, aliases)
                hidden_by_case[key][condition] = hidden
                extra_by_case[key][condition] = extra
            if idx % 15 == 0:
                log(f"{args.model}: captured {idx}/{len(behavior_rows)} cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        directions: dict[tuple[str, int], torch.Tensor] = {}
        for layer in observe_layers:
            template_deltas = []
            semantic_deltas = []
            boundary_deltas = []
            for hidden in hidden_by_case.values():
                tc = hidden["template_complete_semantic_correct"][layer]
                tw = hidden["template_complete_semantic_wrong"][layer]
                ic = hidden["template_incomplete_semantic_correct"][layer]
                iw = hidden["template_incomplete_semantic_wrong"][layer]
                bc = hidden["boundary_complete_semantic_correct"][layer]
                bw = hidden["boundary_complete_semantic_wrong"][layer]
                template_deltas.extend([tc - ic, tw - iw])
                semantic_deltas.extend([tc - tw, ic - iw, bc - bw])
                boundary_deltas.extend([bc - ic, bw - iw])
            raw = {
                "template_done": torch.stack(template_deltas).mean(dim=0),
                "semantic_done": torch.stack(semantic_deltas).mean(dim=0),
                "boundary_done": torch.stack(boundary_deltas).mean(dim=0),
            }
            for axis, vec in raw.items():
                directions[(axis, layer)] = unit(vec)
                vector_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase260",
                        "created_at": utc_now(),
                        "vector_id": f"phase260:vector:{args.model}:{axis}:L{layer}",
                        "model": args.model,
                        "axis": axis,
                        "layer": layer,
                        "component_cases": len(behavior_rows),
                    }
                )
        for key, hidden in hidden_by_case.items():
            case_id, variant_id, mode_id = key
            for layer in observe_layers:
                for axis, (pos_cond, neg_cond) in AXIS_PAIRS.items():
                    direction = directions[(axis, layer)]
                    pos = hidden[pos_cond][layer]
                    neg = hidden[neg_cond][layer]
                    projection_effect = dot(pos, direction) - dot(neg, direction)
                    pos_extra = extra_by_case[key][pos_cond]
                    neg_extra = extra_by_case[key][neg_cond]
                    closure_effect = safe_float(pos_extra.get("closure_closure_proxy_margin")) - safe_float(neg_extra.get("closure_closure_proxy_margin"))
                    eos_effect = safe_float(pos_extra.get("eos_logit")) - safe_float(neg_extra.get("eos_logit"))
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase260",
                        "created_at": utc_now(),
                        "case_layer_id": f"phase260:case_layer:{args.model}:{case_id}:{variant_id}:{axis}:L{layer}",
                        "model": args.model,
                        "case_id": case_id,
                        "variant_id": variant_id,
                        "mode_id": mode_id,
                        "axis": axis,
                        "layer": layer,
                        "positive_condition": pos_cond,
                        "negative_condition": neg_cond,
                        "projection_effect": round(projection_effect, 6),
                        "closure_proxy_effect": round(closure_effect, 6),
                        "eos_logit_effect": round(eos_effect, 6),
                        "projection_and_closure_positive": bool(projection_effect > 0 and closure_effect > 0),
                        "projection_and_eos_positive": bool(projection_effect > 0 and eos_effect > 0),
                    }
                    case_layer_rows.append(row)
                    observations.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase260",
                            "created_at": utc_now(),
                            "observation_id": row["case_layer_id"],
                            "case_id": case_id,
                            "model": args.model,
                            "family_id": "output_protocol",
                            "mode_id": mode_id,
                            "variant_id": variant_id,
                            "level": "sdone_rstop_bridge_layer",
                            "component": f"{axis}:L{layer}",
                            "metric_name": "closure_proxy_effect",
                            "metric_value": row["closure_proxy_effect"],
                            "metric_unit": "logit",
                        }
                    )
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in case_layer_rows:
            grouped[(str(row["axis"]), int(row["layer"]))].append(row)
        for (axis, layer), rows in grouped.items():
            proj = [safe_float(x["projection_effect"]) for x in rows]
            closure = [safe_float(x["closure_proxy_effect"]) for x in rows]
            eos = [safe_float(x["eos_logit_effect"]) for x in rows]
            bridge_rate = sum(1 for x in rows if x.get("projection_and_closure_positive")) / len(rows)
            eos_bridge_rate = sum(1 for x in rows if x.get("projection_and_eos_positive")) / len(rows)
            summary = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase260",
                "created_at": utc_now(),
                "layer_summary_id": f"phase260:layer_summary:{args.model}:{axis}:L{layer}",
                "model": args.model,
                "axis": axis,
                "layer": layer,
                "rows": len(rows),
                "mean_projection_effect": round(mean(proj), 6),
                "mean_closure_proxy_effect": round(mean(closure), 6),
                "mean_eos_logit_effect": round(mean(eos), 6),
                "projection_positive_rate": positive_rate(proj),
                "closure_positive_rate": positive_rate(closure),
                "eos_positive_rate": positive_rate(eos),
                "projection_closure_bridge_rate": round(bridge_rate, 6),
                "projection_eos_bridge_rate": round(eos_bridge_rate, 6),
                "bridge_candidate": bool(mean(proj) > 0 and mean(closure) > 0 and bridge_rate >= 0.5),
                "eos_bridge_candidate": bool(mean(proj) > 0 and mean(eos) > 0 and eos_bridge_rate >= 0.5),
            }
            layer_summary_rows.append(summary)
            metrics.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase260",
                    "created_at": utc_now(),
                    "metric_id": f"phase260:{args.model}:{axis}:L{layer}:bridge_rate",
                    "scope": "sdone_rstop_bridge_layer",
                    "model": args.model,
                    "axis": axis,
                    "layer": layer,
                    "metric_name": "projection_closure_bridge_rate",
                    "metric_value": summary["projection_closure_bridge_rate"],
                    "rows": len(rows),
                }
            )
            edges.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase260",
                    "created_at": utc_now(),
                    "edge_id": f"phase260:bridge:{args.model}:{axis}:L{layer}",
                    "source": f"node:{axis}:L{layer}",
                    "target": "node:StopReadout",
                    "edge_type": "sdone_rstop_bridge_layer",
                    "model": args.model,
                    "evidence_type": "projection_to_closure_proxy_alignment",
                    "effect_direction": "bridge_candidate" if summary["bridge_candidate"] else "no_bridge",
                    "effect_size": summary["projection_closure_bridge_rate"],
                    "confidence": 0.45 if summary["bridge_candidate"] else 0.26,
                    "supporting_phases": ["Phase259", "Phase260"],
                    "status": "layer_bridge_probe_not_causal_closure",
                }
            )
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "S_done to R_stop bridge layer localization",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "case_count": len(behavior_rows),
        "vector_rows": len(vector_rows),
        "case_layer_rows": len(case_layer_rows),
        "layer_summary_rows": len(layer_summary_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "bridge_candidate_count": sum(1 for x in layer_summary_rows if x.get("bridge_candidate")),
        "eos_bridge_candidate_count": sum(1 for x in layer_summary_rows if x.get("eos_bridge_candidate")),
        "mean_bridge_rate_by_axis": mean_by(layer_summary_rows, "axis", "projection_closure_bridge_rate"),
        "mean_eos_bridge_rate_by_axis": mean_by(layer_summary_rows, "axis", "projection_eos_bridge_rate"),
        "mean_closure_effect_by_axis": mean_by(layer_summary_rows, "axis", "mean_closure_proxy_effect"),
        "mean_eos_effect_by_axis": mean_by(layer_summary_rows, "axis", "mean_eos_logit_effect"),
    }
    write_json(out_dir / f"phase260_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase260_{args.model}_vector_rows.jsonl", vector_rows)
    write_jsonl(out_dir / f"phase260_{args.model}_case_layer_rows.jsonl", case_layer_rows)
    write_jsonl(out_dir / f"phase260_{args.model}_layer_summary_rows.jsonl", layer_summary_rows)
    write_jsonl(out_dir / f"phase260_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase260_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase260_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase260_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase260_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    vectors: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        vectors.extend(read_jsonl(out_dir / f"phase260_{model}_vector_rows.jsonl"))
        cases.extend(read_jsonl(out_dir / f"phase260_{model}_case_layer_rows.jsonl"))
        layers.extend(read_jsonl(out_dir / f"phase260_{model}_layer_summary_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase260_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase260_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase260_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase260_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.83,
        "trace_signature_validation": 0.46,
        "semantic_done_signature": 0.24,
        "done_state_cluster_map": 0.21,
        "template_semantic_disentanglement": 0.19,
        "sdone_rstop_bridge": 0.08,
        "residual_state_signature": 0.55,
        "readout_competition_trace": 0.73,
        "stepwise_rollout_trace": 0.41,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.64,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "S_done to R_stop bridge layer localization",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "vector_rows": len(vectors),
        "case_layer_rows": len(cases),
        "layer_summary_rows": len(layers),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "bridge_candidate_count": sum(1 for x in layers if x.get("bridge_candidate")),
        "eos_bridge_candidate_count": sum(1 for x in layers if x.get("eos_bridge_candidate")),
        "mean_bridge_rate_by_axis": mean_by(layers, "axis", "projection_closure_bridge_rate"),
        "mean_eos_bridge_rate_by_axis": mean_by(layers, "axis", "projection_eos_bridge_rate"),
        "mean_closure_effect_by_axis": mean_by(layers, "axis", "mean_closure_proxy_effect"),
        "mean_eos_effect_by_axis": mean_by(layers, "axis", "mean_eos_logit_effect"),
        "progress": progress,
    }
    write_json(out_dir / "phase260_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase260_vector_rows.jsonl", vectors)
    write_jsonl(out_dir / "phase260_case_layer_rows.jsonl", cases)
    write_jsonl(out_dir / "phase260_layer_summary_rows.jsonl", layers)
    write_jsonl(out_dir / "phase260_observations.jsonl", observations)
    write_jsonl(out_dir / "phase260_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase260_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase260_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase260", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase260 S_done to R_stop Bridge Layer Localization",
        "",
        f"- status: {payload['status']}",
        f"- case_layer_rows: {payload['case_layer_rows']}",
        f"- layer_summary_rows: {payload['layer_summary_rows']}",
        f"- bridge_candidate_count: {payload['bridge_candidate_count']}",
        f"- eos_bridge_candidate_count: {payload['eos_bridge_candidate_count']}",
        f"- mean_bridge_rate_by_axis: {json.dumps(payload['mean_bridge_rate_by_axis'], ensure_ascii=False)}",
        f"- mean_eos_bridge_rate_by_axis: {json.dumps(payload['mean_eos_bridge_rate_by_axis'], ensure_ascii=False)}",
        f"- mean_closure_effect_by_axis: {json.dumps(payload['mean_closure_effect_by_axis'], ensure_ascii=False)}",
        f"- mean_eos_effect_by_axis: {json.dumps(payload['mean_eos_effect_by_axis'], ensure_ascii=False)}",
    ]
    (out_dir / "phase260_sdone_rstop_bridge_layer_localization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-mode", type=int, default=8)
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
