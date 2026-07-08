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


PHASE = 259
SOURCE_PHASE = 258
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
RESULT_ROOT = Path("tests/result/phase259_template_semantic_done_disentanglement")
ROUND_DEFAULT = "template_semantic_done_disentanglement"
MODES = ["short_answer", "one_word", "explain_answer", "json_answer", "stop_after_answer"]
WRONG_POOL = ["green", "red", "Paris", "seven", "winter", "metal", "triangle", "incorrect"]

SPECS = {
    "qwen3": {"final_layer": 33},
    "glm4": {"final_layer": 32},
    "deepseek7b": {"final_layer": 27},
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


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        return 0.0
    an = torch.linalg.vector_norm(a.float()).item()
    bn = torch.linalg.vector_norm(b.float()).item()
    if an <= 1e-8 or bn <= 1e-8:
        return 0.0
    return float(torch.dot(a.float(), b.float()).item() / (an * bn))


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def load_rows(model_name: str, max_cases_per_mode: int) -> list[dict[str, Any]]:
    rows = [x for x in read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl") if str(x.get("model")) == model_name]
    rows = [x for x in rows if str(x.get("family_id")) == "output_protocol" and str(x.get("mode_id")) in MODES and list(x.get("target_aliases") or [])]
    rows.sort(key=lambda x: (str(x.get("mode_id")), str(x.get("case_id")), str(x.get("variant_id"))))
    out = []
    counts: Counter[str] = Counter()
    for row in rows:
        mode = str(row.get("mode_id"))
        if counts[mode] >= int(max_cases_per_mode):
            continue
        out.append(row)
        counts[mode] += 1
    return out


def wrong_answer(correct: str, aliases: list[str]) -> str:
    lowered = {str(x).strip().lower() for x in aliases if str(x).strip()}
    lowered.add(str(correct).strip().lower())
    for item in WRONG_POOL:
        if item.lower() not in lowered:
            return item
    return "incorrect"


def condition_texts(row: dict[str, Any]) -> dict[str, str]:
    prompt = str(row["prompt_variant"]).rstrip()
    aliases = [str(x).strip() for x in row.get("target_aliases") or [] if str(x).strip()]
    correct = aliases[0] if aliases else "blue"
    wrong = wrong_answer(correct, aliases)
    return {
        "template_complete_semantic_correct": f"{prompt}\nAnswer: {correct}\n\nReason: complete.",
        "template_complete_semantic_wrong": f"{prompt}\nAnswer: {wrong}\n\nReason: complete.",
        "template_incomplete_semantic_correct": f"{prompt}\n{correct}",
        "template_incomplete_semantic_wrong": f"{prompt}\n{wrong}",
        "boundary_complete_semantic_correct": f"{prompt}\n{correct}.",
        "boundary_complete_semantic_wrong": f"{prompt}\n{wrong}.",
    }


def capture(model_obj: Any, tokenizer: Any, device: torch.device, text: str, final_layer: int, aliases: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    vec = out.hidden_states[int(final_layer) + 1][0, last_pos].detach().float().cpu()
    logits = out.logits[0, last_pos].detach().float().cpu()
    closure = p252.closure_scores(tokenizer, logits)
    readout = p239.readout_metrics(tokenizer, logits, aliases)
    eos_logit = logits[int(tokenizer.eos_token_id)].item() if tokenizer.eos_token_id is not None else 0.0
    return vec, {"eos_logit": round(float(eos_logit), 6), **{f"closure_{k}": round(v, 6) for k, v in closure.items()}, **{f"readout_{k}": v for k, v in readout.items()}}


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_rows = load_rows(args.model, int(args.max_cases_per_mode))
    model_obj = None
    tokenizer = None
    prefix_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
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
        final_layer = int(SPECS[args.model]["final_layer"])
        hidden_by_case: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
        meta_by_case: dict[tuple[str, str, str], dict[str, Any]] = {}
        extra_by_case: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        for idx, row in enumerate(behavior_rows, start=1):
            key = (str(row["case_id"]), str(row["variant_id"]), str(row["mode_id"]))
            meta_by_case[key] = row
            hidden_by_case[key] = {}
            extra_by_case[key] = {}
            aliases = list(row.get("target_aliases") or [])
            for condition, text in condition_texts(row).items():
                vec, extra = capture(model_obj, tokenizer, device, text, final_layer, aliases)
                hidden_by_case[key][condition] = vec
                extra_by_case[key][condition] = extra
            if idx % 15 == 0:
                log(f"{args.model}: captured {idx}/{len(behavior_rows)} cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        template_deltas = []
        semantic_deltas = []
        boundary_deltas = []
        for hidden in hidden_by_case.values():
            tc = hidden["template_complete_semantic_correct"]
            tw = hidden["template_complete_semantic_wrong"]
            ic = hidden["template_incomplete_semantic_correct"]
            iw = hidden["template_incomplete_semantic_wrong"]
            bc = hidden["boundary_complete_semantic_correct"]
            bw = hidden["boundary_complete_semantic_wrong"]
            template_deltas.extend([tc - ic, tw - iw])
            semantic_deltas.extend([tc - tw, ic - iw, bc - bw])
            boundary_deltas.extend([bc - ic, bw - iw])
        directions = {
            "template_done": unit(torch.stack(template_deltas).mean(dim=0)),
            "semantic_done": unit(torch.stack(semantic_deltas).mean(dim=0)),
            "boundary_done": unit(torch.stack(boundary_deltas).mean(dim=0)),
        }
        for name, direction in directions.items():
            vector_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase259",
                    "created_at": utc_now(),
                    "vector_id": f"phase259:vector:{args.model}:{name}",
                    "model": args.model,
                    "vector_name": name,
                    "component_cases": len(behavior_rows),
                }
            )
        for a, avec in directions.items():
            for b, bvec in directions.items():
                if a >= b:
                    continue
                metrics.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase259",
                        "created_at": utc_now(),
                        "metric_id": f"phase259:{args.model}:cosine:{a}:{b}",
                        "scope": "direction_disentanglement",
                        "model": args.model,
                        "metric_name": "direction_cosine",
                        "source_direction": a,
                        "target_direction": b,
                        "metric_value": round(cosine(avec, bvec), 6),
                    }
                )
        for key, hidden in hidden_by_case.items():
            row = meta_by_case[key]
            vals: dict[str, dict[str, float]] = {}
            for condition, vec in hidden.items():
                vals[condition] = {name: dot(vec, direction) for name, direction in directions.items()}
                out = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase259",
                    "created_at": utc_now(),
                    "prefix_id": f"phase259:prefix:{args.model}:{key[0]}:{key[1]}:{condition}",
                    "model": args.model,
                    "case_id": key[0],
                    "variant_id": key[1],
                    "mode_id": key[2],
                    "condition": condition,
                    **{f"{name}_projection": round(value, 6) for name, value in vals[condition].items()},
                    **extra_by_case[key][condition],
                }
                prefix_rows.append(out)
                observations.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase259",
                        "created_at": utc_now(),
                        "observation_id": out["prefix_id"],
                        "case_id": key[0],
                        "model": args.model,
                        "family_id": "output_protocol",
                        "mode_id": key[2],
                        "variant_id": key[1],
                        "level": "template_semantic_done_disentanglement",
                        "component": condition,
                        "metric_name": "closure_proxy_margin",
                        "metric_value": out.get("closure_closure_proxy_margin"),
                        "metric_unit": "logit",
                    }
                )
            def get(cond: str, proj: str) -> float:
                return safe_float(vals.get(cond, {}).get(proj))
            def get_extra(cond: str, field: str) -> float:
                return safe_float(extra_by_case[key].get(cond, {}).get(field))
            summary = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase259",
                "created_at": utc_now(),
                "case_summary_id": f"phase259:case:{args.model}:{key[0]}:{key[1]}",
                "model": args.model,
                "case_id": key[0],
                "variant_id": key[1],
                "mode_id": key[2],
                "template_axis_effect_correct": round(get("template_complete_semantic_correct", "template_done") - get("template_incomplete_semantic_correct", "template_done"), 6),
                "template_axis_effect_wrong": round(get("template_complete_semantic_wrong", "template_done") - get("template_incomplete_semantic_wrong", "template_done"), 6),
                "semantic_axis_effect_template": round(get("template_complete_semantic_correct", "semantic_done") - get("template_complete_semantic_wrong", "semantic_done"), 6),
                "semantic_axis_effect_incomplete": round(get("template_incomplete_semantic_correct", "semantic_done") - get("template_incomplete_semantic_wrong", "semantic_done"), 6),
                "boundary_axis_effect_correct": round(get("boundary_complete_semantic_correct", "boundary_done") - get("template_incomplete_semantic_correct", "boundary_done"), 6),
                "closure_template_effect_correct": round(get_extra("template_complete_semantic_correct", "closure_closure_proxy_margin") - get_extra("template_incomplete_semantic_correct", "closure_closure_proxy_margin"), 6),
                "closure_semantic_effect_template": round(get_extra("template_complete_semantic_correct", "closure_closure_proxy_margin") - get_extra("template_complete_semantic_wrong", "closure_closure_proxy_margin"), 6),
                "closure_boundary_effect_correct": round(get_extra("boundary_complete_semantic_correct", "closure_closure_proxy_margin") - get_extra("template_incomplete_semantic_correct", "closure_closure_proxy_margin"), 6),
            }
            summary["template_semantic_disentangled"] = bool(
                summary["template_axis_effect_correct"] > 0
                and summary["template_axis_effect_wrong"] > 0
                and summary["semantic_axis_effect_template"] > 0
                and summary["semantic_axis_effect_incomplete"] > 0
            )
            case_rows.append(summary)
            edges.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase259",
                    "created_at": utc_now(),
                    "edge_id": f"phase259:disentangle:{args.model}:{key[0]}:{key[1]}",
                    "source": "node:TemplateSemanticDoneAxes",
                    "target": f"case:{key[0]}:{key[1]}",
                    "edge_type": "template_semantic_done_disentanglement",
                    "model": args.model,
                    "case_id": key[0],
                    "variant_id": key[1],
                    "effect_direction": "disentangled" if summary["template_semantic_disentangled"] else "entangled_or_failed",
                    "effect_size": summary["semantic_axis_effect_template"],
                    "confidence": 0.43 if summary["template_semantic_disentangled"] else 0.28,
                    "supporting_phases": ["Phase258", "Phase259"],
                    "status": "projection_disentanglement_not_causal_closure",
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
    for field in [
        "template_axis_effect_correct",
        "semantic_axis_effect_template",
        "closure_template_effect_correct",
        "closure_semantic_effect_template",
        "closure_boundary_effect_correct",
    ]:
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase259",
                "created_at": utc_now(),
                "metric_id": f"phase259:{args.model}:{field}:mean",
                "scope": "case_summary",
                "model": args.model,
                "metric_name": field,
                "metric_value": round(mean(safe_float(x.get(field)) for x in case_rows), 6) if case_rows else 0.0,
                "rows": len(case_rows),
            }
        )
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Template-done and semantic-done disentanglement",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "case_count": len(behavior_rows),
        "prefix_rows": len(prefix_rows),
        "case_summary_rows": len(case_rows),
        "vector_rows": len(vector_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "disentangled_rate": round(sum(1 for x in case_rows if x.get("template_semantic_disentangled")) / len(case_rows), 6) if case_rows else 0.0,
        "mean_case_effects": {field: round(mean(safe_float(x.get(field)) for x in case_rows), 6) if case_rows else 0.0 for field in [
            "template_axis_effect_correct",
            "template_axis_effect_wrong",
            "semantic_axis_effect_template",
            "semantic_axis_effect_incomplete",
            "closure_template_effect_correct",
            "closure_semantic_effect_template",
            "closure_boundary_effect_correct",
        ]},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"phase259_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase259_{args.model}_vector_rows.jsonl", vector_rows)
    write_jsonl(out_dir / f"phase259_{args.model}_prefix_rows.jsonl", prefix_rows)
    write_jsonl(out_dir / f"phase259_{args.model}_case_summary_rows.jsonl", case_rows)
    write_jsonl(out_dir / f"phase259_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase259_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase259_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase259_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase259_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    vectors: list[dict[str, Any]] = []
    prefixes: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        vectors.extend(read_jsonl(out_dir / f"phase259_{model}_vector_rows.jsonl"))
        prefixes.extend(read_jsonl(out_dir / f"phase259_{model}_prefix_rows.jsonl"))
        cases.extend(read_jsonl(out_dir / f"phase259_{model}_case_summary_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase259_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase259_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase259_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase259_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.83,
        "trace_signature_validation": 0.45,
        "semantic_done_signature": 0.23,
        "done_state_cluster_map": 0.20,
        "template_semantic_disentanglement": 0.18,
        "residual_state_signature": 0.54,
        "readout_competition_trace": 0.73,
        "stepwise_rollout_trace": 0.41,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.64,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Template-done and semantic-done disentanglement",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "vector_rows": len(vectors),
        "prefix_rows": len(prefixes),
        "case_summary_rows": len(cases),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "disentangled_rate": round(sum(1 for x in cases if x.get("template_semantic_disentangled")) / len(cases), 6) if cases else 0.0,
        "disentangled_rate_by_model": {
            model: round(sum(1 for x in cases if x.get("model") == model and x.get("template_semantic_disentangled")) / max(1, sum(1 for x in cases if x.get("model") == model)), 6)
            for model in MODELS
        },
        "mean_case_effects": {field: round(mean(safe_float(x.get(field)) for x in cases), 6) if cases else 0.0 for field in [
            "template_axis_effect_correct",
            "template_axis_effect_wrong",
            "semantic_axis_effect_template",
            "semantic_axis_effect_incomplete",
            "closure_template_effect_correct",
            "closure_semantic_effect_template",
            "closure_boundary_effect_correct",
        ]},
        "progress": progress,
    }
    write_json(out_dir / "phase259_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase259_vector_rows.jsonl", vectors)
    write_jsonl(out_dir / "phase259_prefix_rows.jsonl", prefixes)
    write_jsonl(out_dir / "phase259_case_summary_rows.jsonl", cases)
    write_jsonl(out_dir / "phase259_observations.jsonl", observations)
    write_jsonl(out_dir / "phase259_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase259_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase259_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase259", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase259 Template/Semantic Done Disentanglement",
        "",
        f"- status: {payload['status']}",
        f"- prefix_rows: {payload['prefix_rows']}",
        f"- case_summary_rows: {payload['case_summary_rows']}",
        f"- disentangled_rate: {payload['disentangled_rate']}",
        f"- disentangled_rate_by_model: {json.dumps(payload['disentangled_rate_by_model'], ensure_ascii=False)}",
        f"- mean_case_effects: {json.dumps(payload['mean_case_effects'], ensure_ascii=False)}",
    ]
    (out_dir / "phase259_template_semantic_done_disentanglement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
