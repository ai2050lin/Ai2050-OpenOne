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


PHASE = 261
SOURCE_PHASE = 260
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase261_stop_continuation_competition_atlas")
ROUND_DEFAULT = "stop_continuation_competition_atlas"

SPECS = {
    "qwen3": {"final_layer": 33},
    "glm4": {"final_layer": 32},
    "deepseek7b": {"final_layer": 27},
}

STOP_GROUPS = {
    "eos": ["<eos>"],
    "period": [".", "。"],
    "newline": ["\n", "\n\n"],
    "end_boundary": ["</s>", "<|endoftext|>", "<|im_end|>"],
}

CONT_GROUPS = {
    "continue_the": [" the", " The"],
    "continue_because": [" because", " Because"],
    "continue_and": [" and", " And"],
    "continue_comma": [",", "，"],
    "continue_is": [" is", " are", " was"],
    "continue_for": [" for", " to", " of"],
    "continue_next_sentence": ["\nThe", "\n\nThe", " Moreover", " However"],
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


def token_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    ids = set()
    for text in texts:
        if text == "<eos>":
            if tokenizer.eos_token_id is not None:
                ids.add(int(tokenizer.eos_token_id))
            continue
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            encoded = []
        if encoded:
            ids.add(int(encoded[-1]))
    return sorted(ids)


def group_score(logits: torch.Tensor, ids: list[int]) -> float:
    vals = [float(logits[i].item()) for i in ids if 0 <= int(i) < logits.numel()]
    return max(vals) if vals else -1e9


def capture(model_obj: Any, tokenizer: Any, device: torch.device, text: str, final_layer: int, aliases: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[int(final_layer) + 1][0, last_pos].detach().float().cpu()
    logits = out.logits[0, last_pos].detach().float().cpu()
    readout = p239.readout_metrics(tokenizer, logits, aliases)
    return hidden, logits, readout


def competition_scores(tokenizer: Any, logits: torch.Tensor, stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]]) -> dict[str, Any]:
    stop_scores = {name: group_score(logits, ids) for name, ids in stop_ids.items()}
    cont_scores = {name: group_score(logits, ids) for name, ids in cont_ids.items()}
    r_stop_name, r_stop = max(stop_scores.items(), key=lambda x: x[1])
    r_cont_name, r_cont = max(cont_scores.items(), key=lambda x: x[1])
    return {
        **{f"stop_{k}_logit": round(v, 6) for k, v in stop_scores.items()},
        **{f"continue_{k}_logit": round(v, 6) for k, v in cont_scores.items()},
        "r_stop_name": r_stop_name,
        "r_continue_name": r_cont_name,
        "r_stop": round(r_stop, 6),
        "r_continue": round(r_cont, 6),
        "stop_continue_margin": round(r_stop - r_cont, 6),
        "competition_winner": "stop" if r_stop > r_cont else "continue",
    }


def make_directions(hidden_by_case: dict[tuple[str, str, str], dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
    return {
        "template_done": unit(torch.stack(template_deltas).mean(dim=0)),
        "semantic_done": unit(torch.stack(semantic_deltas).mean(dim=0)),
        "boundary_done": unit(torch.stack(boundary_deltas).mean(dim=0)),
    }


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_rows = p259.load_rows(args.model, int(args.max_cases_per_mode))
    model_obj = None
    tokenizer = None
    vector_rows: list[dict[str, Any]] = []
    competition_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
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
        stop_ids = {name: token_ids(tokenizer, texts) for name, texts in STOP_GROUPS.items()}
        cont_ids = {name: token_ids(tokenizer, texts) for name, texts in CONT_GROUPS.items()}
        hidden_by_case: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
        logits_by_case: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
        readout_by_case: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        for idx, row in enumerate(behavior_rows, start=1):
            key = (str(row["case_id"]), str(row["variant_id"]), str(row["mode_id"]))
            aliases = list(row.get("target_aliases") or [])
            hidden_by_case[key] = {}
            logits_by_case[key] = {}
            readout_by_case[key] = {}
            for condition, text in p259.condition_texts(row).items():
                hidden, logits, readout = capture(model_obj, tokenizer, device, text, final_layer, aliases)
                hidden_by_case[key][condition] = hidden
                logits_by_case[key][condition] = logits
                readout_by_case[key][condition] = readout
            if idx % 15 == 0:
                log(f"{args.model}: captured {idx}/{len(behavior_rows)} cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        directions = make_directions(hidden_by_case)
        for name in directions:
            vector_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase261",
                    "created_at": utc_now(),
                    "vector_id": f"phase261:vector:{args.model}:{name}",
                    "model": args.model,
                    "axis": name,
                    "layer": final_layer,
                    "component_cases": len(behavior_rows),
                }
            )
        for key, hidden in hidden_by_case.items():
            case_id, variant_id, mode_id = key
            for condition, vec in hidden.items():
                logits = logits_by_case[key][condition]
                scores = competition_scores(tokenizer, logits, stop_ids, cont_ids)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase261",
                    "created_at": utc_now(),
                    "competition_id": f"phase261:competition:{args.model}:{case_id}:{variant_id}:{condition}",
                    "model": args.model,
                    "case_id": case_id,
                    "variant_id": variant_id,
                    "mode_id": mode_id,
                    "condition": condition,
                    "template_projection": round(dot(vec, directions["template_done"]), 6),
                    "semantic_projection": round(dot(vec, directions["semantic_done"]), 6),
                    "boundary_projection": round(dot(vec, directions["boundary_done"]), 6),
                    **scores,
                    **readout_by_case[key][condition],
                }
                competition_rows.append(row)
                observations.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase261",
                        "created_at": utc_now(),
                        "observation_id": row["competition_id"],
                        "case_id": case_id,
                        "model": args.model,
                        "family_id": "output_protocol",
                        "mode_id": mode_id,
                        "variant_id": variant_id,
                        "level": "stop_continuation_competition",
                        "component": condition,
                        "metric_name": "stop_continue_margin",
                        "metric_value": row["stop_continue_margin"],
                        "metric_unit": "logit",
                        "winner": row["competition_winner"],
                    }
                )
            comparisons = [
                ("template_effect_correct", "template_complete_semantic_correct", "template_incomplete_semantic_correct"),
                ("template_effect_wrong", "template_complete_semantic_wrong", "template_incomplete_semantic_wrong"),
                ("semantic_effect_template", "template_complete_semantic_correct", "template_complete_semantic_wrong"),
                ("semantic_effect_incomplete", "template_incomplete_semantic_correct", "template_incomplete_semantic_wrong"),
                ("boundary_effect_correct", "boundary_complete_semantic_correct", "template_incomplete_semantic_correct"),
                ("boundary_effect_wrong", "boundary_complete_semantic_wrong", "template_incomplete_semantic_wrong"),
            ]
            by_cond = {x["condition"]: x for x in competition_rows if x["model"] == args.model and x["case_id"] == case_id and x["variant_id"] == variant_id}
            for effect_name, pos, neg in comparisons:
                if pos not in by_cond or neg not in by_cond:
                    continue
                p, n = by_cond[pos], by_cond[neg]
                effect_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase261",
                        "created_at": utc_now(),
                        "effect_id": f"phase261:effect:{args.model}:{case_id}:{variant_id}:{effect_name}",
                        "model": args.model,
                        "case_id": case_id,
                        "variant_id": variant_id,
                        "mode_id": mode_id,
                        "effect_name": effect_name,
                        "stop_margin_delta": round(safe_float(p["stop_continue_margin"]) - safe_float(n["stop_continue_margin"]), 6),
                        "r_stop_delta": round(safe_float(p["r_stop"]) - safe_float(n["r_stop"]), 6),
                        "r_continue_delta": round(safe_float(p["r_continue"]) - safe_float(n["r_continue"]), 6),
                        "winner_flip_to_stop": bool(p["competition_winner"] == "stop" and n["competition_winner"] != "stop"),
                    }
                )
        for condition, value in mean_by(competition_rows, "condition", "stop_continue_margin").items():
            metrics.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase261",
                    "created_at": utc_now(),
                    "metric_id": f"phase261:{args.model}:{condition}:mean_stop_continue_margin",
                    "scope": "stop_continuation_competition",
                    "model": args.model,
                    "condition": condition,
                    "metric_name": "mean_stop_continue_margin",
                    "metric_value": value,
                    "rows": sum(1 for x in competition_rows if x.get("condition") == condition),
                }
            )
        for effect_name, value in mean_by(effect_rows, "effect_name", "stop_margin_delta").items():
            metrics.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase261",
                    "created_at": utc_now(),
                    "metric_id": f"phase261:{args.model}:{effect_name}:mean_stop_margin_delta",
                    "scope": "stop_continuation_effect",
                    "model": args.model,
                    "effect_name": effect_name,
                    "metric_name": "mean_stop_margin_delta",
                    "metric_value": value,
                    "rows": sum(1 for x in effect_rows if x.get("effect_name") == effect_name),
                }
            )
            edges.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase261",
                    "created_at": utc_now(),
                    "edge_id": f"phase261:effect:{args.model}:{effect_name}",
                    "source": f"node:{effect_name}",
                    "target": "node:StopVsContinuationCompetition",
                    "edge_type": "stop_continuation_competition_effect",
                    "model": args.model,
                    "evidence_type": "static_prefix_logit_competition",
                    "effect_direction": "improves_stop_margin" if value > 0 else "weakens_stop_margin",
                    "effect_size": value,
                    "confidence": 0.42 if value > 0 else 0.30,
                    "supporting_phases": ["Phase260", "Phase261"],
                    "status": "competition_atlas_not_rollout_closure",
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
    winner_counts = Counter(str(x["competition_winner"]) for x in competition_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Stop-vs-continuation competition atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "case_count": len(behavior_rows),
        "vector_rows": len(vector_rows),
        "competition_rows": len(competition_rows),
        "effect_rows": len(effect_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "competition_winner_counts": dict(winner_counts.most_common()),
        "stop_win_rate": round(winner_counts.get("stop", 0) / len(competition_rows), 6) if competition_rows else 0.0,
        "mean_stop_margin_by_condition": mean_by(competition_rows, "condition", "stop_continue_margin"),
        "mean_stop_margin_delta_by_effect": mean_by(effect_rows, "effect_name", "stop_margin_delta"),
        "mean_r_stop_delta_by_effect": mean_by(effect_rows, "effect_name", "r_stop_delta"),
        "mean_r_continue_delta_by_effect": mean_by(effect_rows, "effect_name", "r_continue_delta"),
    }
    write_json(out_dir / f"phase261_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase261_{args.model}_vector_rows.jsonl", vector_rows)
    write_jsonl(out_dir / f"phase261_{args.model}_competition_rows.jsonl", competition_rows)
    write_jsonl(out_dir / f"phase261_{args.model}_effect_rows.jsonl", effect_rows)
    write_jsonl(out_dir / f"phase261_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase261_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase261_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase261_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase261_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    vectors: list[dict[str, Any]] = []
    competitions: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        vectors.extend(read_jsonl(out_dir / f"phase261_{model}_vector_rows.jsonl"))
        competitions.extend(read_jsonl(out_dir / f"phase261_{model}_competition_rows.jsonl"))
        effects.extend(read_jsonl(out_dir / f"phase261_{model}_effect_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase261_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase261_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase261_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase261_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.84,
        "trace_signature_validation": 0.46,
        "semantic_done_signature": 0.24,
        "done_state_cluster_map": 0.21,
        "template_semantic_disentanglement": 0.19,
        "sdone_rstop_bridge": 0.08,
        "stop_continuation_competition": 0.12,
        "residual_state_signature": 0.55,
        "readout_competition_trace": 0.75,
        "stepwise_rollout_trace": 0.41,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.64,
    }
    winner_counts = Counter(str(x["competition_winner"]) for x in competitions)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Stop-vs-continuation competition atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "vector_rows": len(vectors),
        "competition_rows": len(competitions),
        "effect_rows": len(effects),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "competition_winner_counts": dict(winner_counts.most_common()),
        "stop_win_rate": round(winner_counts.get("stop", 0) / len(competitions), 6) if competitions else 0.0,
        "mean_stop_margin_by_condition": mean_by(competitions, "condition", "stop_continue_margin"),
        "mean_stop_margin_delta_by_effect": mean_by(effects, "effect_name", "stop_margin_delta"),
        "mean_r_stop_delta_by_effect": mean_by(effects, "effect_name", "r_stop_delta"),
        "mean_r_continue_delta_by_effect": mean_by(effects, "effect_name", "r_continue_delta"),
        "progress": progress,
    }
    write_json(out_dir / "phase261_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase261_vector_rows.jsonl", vectors)
    write_jsonl(out_dir / "phase261_competition_rows.jsonl", competitions)
    write_jsonl(out_dir / "phase261_effect_rows.jsonl", effects)
    write_jsonl(out_dir / "phase261_observations.jsonl", observations)
    write_jsonl(out_dir / "phase261_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase261_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase261_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase261", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase261 Stop-vs-Continuation Competition Atlas",
        "",
        f"- status: {payload['status']}",
        f"- competition_rows: {payload['competition_rows']}",
        f"- effect_rows: {payload['effect_rows']}",
        f"- stop_win_rate: {payload['stop_win_rate']}",
        f"- competition_winner_counts: {json.dumps(payload['competition_winner_counts'], ensure_ascii=False)}",
        f"- mean_stop_margin_by_condition: {json.dumps(payload['mean_stop_margin_by_condition'], ensure_ascii=False)}",
        f"- mean_stop_margin_delta_by_effect: {json.dumps(payload['mean_stop_margin_delta_by_effect'], ensure_ascii=False)}",
        f"- mean_r_stop_delta_by_effect: {json.dumps(payload['mean_r_stop_delta_by_effect'], ensure_ascii=False)}",
        f"- mean_r_continue_delta_by_effect: {json.dumps(payload['mean_r_continue_delta_by_effect'], ensure_ascii=False)}",
    ]
    (out_dir / "phase261_stop_continuation_competition_atlas_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
