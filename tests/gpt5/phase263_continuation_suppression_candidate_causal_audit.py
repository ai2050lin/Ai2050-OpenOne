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
import phase259_template_semantic_done_disentanglement as p259  # noqa: E402
import phase262_continuation_regime_decomposition_atlas as p262  # noqa: E402


PHASE = 263
SOURCE_PHASE = 262
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE262_ROOT = Path("tests/result/phase262_continuation_regime_decomposition_atlas/continuation_regime_decomposition_atlas")
RESULT_ROOT = Path("tests/result/phase263_continuation_suppression_candidate_causal_audit")
ROUND_DEFAULT = "continuation_suppression_candidate_causal_audit"

POLICY_GROUPS = {
    "suppress_explanation": ["continue_because"],
    "suppress_structured": ["continue_list_item", "continue_format", "continue_json_structure"],
    "suppress_natural": ["continue_the"],
    "suppress_boundary_aftereffect": ["continue_next_sentence", "continue_format"],
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


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def unit(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float()
    norm = torch.linalg.vector_norm(vec).item()
    if norm <= 1e-8:
        return torch.zeros_like(vec)
    return vec / norm


def select_candidates(model: str, max_per_policy: int) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(PHASE262_ROOT / f"phase262_{model}_continuation_source_map_rows.jsonl") if r.get("candidate_for_suppression")]
    buckets = {
        "suppress_explanation": [r for r in rows if r.get("top_continue_channel") == "continue_because"],
        "suppress_structured": [r for r in rows if r.get("top_continue_channel") in {"continue_list_item", "continue_format", "continue_json_structure"}],
        "suppress_natural": [r for r in rows if r.get("top_continue_channel") == "continue_the"],
        "suppress_boundary_aftereffect": [r for r in rows if r.get("top_continue_channel") == "continue_next_sentence"],
    }
    selected: dict[str, dict[str, Any]] = {}
    for policy, items in buckets.items():
        items.sort(key=lambda r: safe_float(r.get("top_continue_vs_stop_margin")), reverse=True)
        for row in items[:max_per_policy]:
            row = dict(row)
            row["primary_policy"] = policy
            selected[str(row["source_id"])] = row
    return list(selected.values())


def behavior_lookup(model: str, max_cases_per_mode: int) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = p259.load_rows(model, max_cases_per_mode)
    return {(str(r["case_id"]), str(r["variant_id"]), str(r["mode_id"])): r for r in rows}


def candidate_text(candidate: dict[str, Any], behavior: dict[tuple[str, str, str], dict[str, Any]]) -> tuple[str, list[str]]:
    key = (str(candidate["case_id"]), str(candidate["variant_id"]), str(candidate["mode_id"]))
    base = behavior[key]
    text = p259.condition_texts(base)[str(candidate["condition"])]
    regime_id = str(candidate["regime_id"])
    for regime in p262.REGIMES:
        if regime["regime_id"] == regime_id:
            text = regime["apply"](text)
            break
    return text, list(base.get("target_aliases") or [])


def channel_ids(tokenizer: Any, channel_names: list[str]) -> list[int]:
    ids: set[int] = set()
    for channel in channel_names:
        ids.update(p262.token_ids(tokenizer, p262.CONT_GROUPS.get(channel, [])))
    return sorted(ids)


def stop_ids_flat(tokenizer: Any) -> list[int]:
    ids: set[int] = set()
    for texts in p262.STOP_GROUPS.values():
        ids.update(p262.token_ids(tokenizer, texts))
    return sorted(ids)


def vector_from_ids(output_weight: torch.Tensor, ids: list[int], device: torch.device) -> torch.Tensor:
    valid = [int(i) for i in ids if 0 <= int(i) < output_weight.shape[0]]
    if not valid:
        return torch.zeros(output_weight.shape[1], device=device)
    idx = torch.tensor(valid, dtype=torch.long, device=device)
    return unit(output_weight.index_select(0, idx).mean(dim=0))


def capture_hidden_logits(model_obj: Any, tokenizer: Any, device: torch.device, text: str, aliases: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1][0, last_pos].detach()
    logits = out.logits[0, last_pos].detach().float().cpu()
    readout = p239.readout_metrics(tokenizer, logits, aliases)
    return hidden, logits, readout, encoded


def patched_logits(model_obj: Any, hidden: torch.Tensor, patch_vec: torch.Tensor) -> torch.Tensor:
    patched = (hidden.float() + patch_vec.float()).to(hidden.device)
    out_head = model_obj.get_output_embeddings()
    patched = patched.to(dtype=out_head.weight.dtype)
    with torch.inference_mode():
        logits = out_head(patched.unsqueeze(0)).squeeze(0)
    return logits.detach().float().cpu()


def score_all(tokenizer: Any, logits: torch.Tensor, aliases: list[str]) -> dict[str, Any]:
    stop_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.STOP_GROUPS.items()}
    cont_ids = {name: p262.token_ids(tokenizer, texts) for name, texts in p262.CONT_GROUPS.items()}
    return {**p262.score_channels(logits, stop_ids, cont_ids), **p239.readout_metrics(tokenizer, logits, aliases)}


def policy_channels(policy_id: str, top_channel: str) -> list[str]:
    if policy_id in POLICY_GROUPS:
        return POLICY_GROUPS[policy_id]
    if policy_id in {"suppress_top", "stop_plus_top"}:
        return [top_channel]
    return []


def make_patch(
    policy_id: str,
    top_channel: str,
    lambda_value: float,
    alpha_stop: float,
    output_weight: torch.Tensor,
    tokenizer: Any,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    channels = policy_channels(policy_id, top_channel)
    patch = torch.zeros(output_weight.shape[1], device=device)
    if channels:
        patch = patch - float(lambda_value) * vector_from_ids(output_weight, channel_ids(tokenizer, channels), device)
    if policy_id.startswith("stop_plus"):
        patch = patch + float(alpha_stop) * vector_from_ids(output_weight, stop_ids_flat(tokenizer), device)
    return patch, channels


def rollout_probe(
    model_obj: Any,
    tokenizer: Any,
    device: torch.device,
    text: str,
    aliases: list[str],
    policy_id: str,
    top_channel: str,
    lambda_value: float,
    alpha_stop: float,
    output_weight: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    input_ids = encoded["input_ids"]
    eos_id = tokenizer.eos_token_id
    generated: list[int] = []
    stopped = False
    channels = policy_channels(policy_id, top_channel)
    for _step in range(max_new_tokens):
        with torch.inference_mode():
            out = model_obj(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1][0, -1].detach()
        patch, _channels = make_patch(policy_id, top_channel, lambda_value, alpha_stop, output_weight, tokenizer, device)
        logits = patched_logits(model_obj, hidden, patch)
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=input_ids.dtype, device=device)], dim=1)
        if eos_id is not None and next_id == int(eos_id):
            stopped = True
            break
    text_out = tokenizer.decode(generated, skip_special_tokens=False)
    low = text_out.lower()
    alias_hit = any(str(a).lower() in low for a in aliases if str(a))
    return {
        "generated_text": text_out[:500],
        "generated_token_count": len(generated),
        "model_stop_executed": stopped,
        "alias_hit": alias_hit,
        "has_because": "because" in low or "因为" in text_out,
        "has_list_marker": "\n-" in text_out or "\n1" in text_out or "- " in text_out,
        "policy_channels": channels,
    }


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_candidates(args.model, args.max_candidates_per_policy)
    behavior = behavior_lookup(args.model, args.max_cases_per_mode)
    model_obj = None
    tokenizer = None
    suppression_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    stop_plus_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
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
        output_weight = model_obj.get_output_embeddings().weight.detach().float()
        policies = ["suppress_explanation", "suppress_structured", "suppress_natural", "suppress_boundary_aftereffect", "suppress_top", "stop_plus_top"]
        lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
        for idx, candidate in enumerate(candidates, start=1):
            try:
                text, aliases = candidate_text(candidate, behavior)
            except KeyError:
                missing_rows.append({"phase_id": "Phase263", "model": args.model, "candidate": candidate, "reason": "missing_behavior_row"})
                continue
            hidden, base_logits, base_readout, _encoded = capture_hidden_logits(model_obj, tokenizer, device, text, aliases)
            base_scores = score_all(tokenizer, base_logits, aliases)
            top_channel = str(candidate.get("top_continue_channel") or base_scores.get("top_continue_channel"))
            for policy_id in policies:
                for lambda_value in lambdas:
                    alpha_stop = float(args.alpha_stop) if policy_id.startswith("stop_plus") else 0.0
                    patch, channels = make_patch(policy_id, top_channel, lambda_value, alpha_stop, output_weight, tokenizer, device)
                    logits = patched_logits(model_obj, hidden, patch)
                    patched_scores = score_all(tokenizer, logits, aliases)
                    row_id = f"phase263:suppression:{args.model}:{idx}:{policy_id}:l{lambda_value:g}"
                    target_delta = safe_float(patched_scores.get("target_logit")) - safe_float(base_scores.get("target_logit"))
                    margin_delta = safe_float(patched_scores.get("stop_continue_margin")) - safe_float(base_scores.get("stop_continue_margin"))
                    target_channel_delta = safe_float(patched_scores.get(f"{top_channel}_logit")) - safe_float(base_scores.get(f"{top_channel}_logit"))
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase263",
                        "created_at": utc_now(),
                        "suppression_id": row_id,
                        "model": args.model,
                        "case_id": candidate["case_id"],
                        "variant_id": candidate["variant_id"],
                        "mode_id": candidate["mode_id"],
                        "condition": candidate["condition"],
                        "regime_id": candidate["regime_id"],
                        "source_hypothesis": candidate.get("source_hypothesis"),
                        "top_continue_channel": top_channel,
                        "policy_id": policy_id,
                        "suppressed_channels": channels,
                        "lambda_value": lambda_value,
                        "alpha_stop": alpha_stop,
                        "baseline_stop_continue_margin": base_scores["stop_continue_margin"],
                        "patched_stop_continue_margin": patched_scores["stop_continue_margin"],
                        "stop_margin_delta": round(margin_delta, 6),
                        "baseline_winner": base_scores["competition_winner"],
                        "patched_winner": patched_scores["competition_winner"],
                        "winner_flip_to_stop": bool(patched_scores["competition_winner"] == "stop" and base_scores["competition_winner"] != "stop"),
                        "target_logit_delta": round(target_delta, 6),
                        "target_rank_delta": int(patched_scores["target_rank"]) - int(base_scores["target_rank"]),
                        "target_preserved": bool(target_delta >= -1.0),
                        "top_channel_logit_delta": round(target_channel_delta, 6),
                        "baseline_top_channel_logit": base_scores.get(f"{top_channel}_logit"),
                        "patched_top_channel_logit": patched_scores.get(f"{top_channel}_logit"),
                        "baseline_r_stop": base_scores["r_stop"],
                        "patched_r_stop": patched_scores["r_stop"],
                        "baseline_r_continue": base_scores["r_continue"],
                        "patched_r_continue": patched_scores["r_continue"],
                        "baseline_target_logit": base_scores["target_logit"],
                        "patched_target_logit": patched_scores["target_logit"],
                    }
                    suppression_rows.append(row)
                    effect_rows.append(
                        {
                            **row,
                            "effect_id": row_id.replace(":suppression:", ":effect:"),
                            "effective_suppression": bool(target_channel_delta < -0.25),
                            "useful_margin_repair": bool(margin_delta > 0.5 and target_delta >= -1.0),
                        }
                    )
                    if policy_id.startswith("stop_plus"):
                        stop_plus_rows.append({**row, "stop_plus_id": row_id.replace(":suppression:", ":stop_plus:")})
                    answer_rows.append(
                        {
                            **row,
                            "answer_preservation_id": row_id.replace(":suppression:", ":answer:"),
                            "metric_name": "target_logit_delta",
                            "metric_value": round(target_delta, 6),
                        }
                    )
                    observations.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase263",
                            "created_at": utc_now(),
                            "observation_id": row_id,
                            "case_id": candidate["case_id"],
                            "model": args.model,
                            "family_id": "output_protocol",
                            "mode_id": candidate["mode_id"],
                            "variant_id": candidate["variant_id"],
                            "level": "continuation_suppression_causal_audit",
                            "component": policy_id,
                            "metric_name": "stop_margin_delta",
                            "metric_value": round(margin_delta, 6),
                            "metric_unit": "logit",
                            "winner": patched_scores["competition_winner"],
                            "target_preserved": bool(target_delta >= -1.0),
                        }
                    )
            if idx <= args.rollout_candidates:
                for policy_id, lambda_value in [("no_patch", 0.0), ("suppress_top", float(args.rollout_lambda)), ("stop_plus_top", float(args.rollout_lambda))]:
                    probe = rollout_probe(
                        model_obj,
                        tokenizer,
                        device,
                        text,
                        aliases,
                        policy_id,
                        top_channel,
                        lambda_value,
                        float(args.alpha_stop),
                        output_weight,
                        int(args.rollout_tokens),
                    )
                    rollout_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase263",
                            "created_at": utc_now(),
                            "rollout_id": f"phase263:rollout:{args.model}:{idx}:{policy_id}",
                            "model": args.model,
                            "case_id": candidate["case_id"],
                            "variant_id": candidate["variant_id"],
                            "mode_id": candidate["mode_id"],
                            "condition": candidate["condition"],
                            "regime_id": candidate["regime_id"],
                            "policy_id": policy_id,
                            "lambda_value": lambda_value,
                            "alpha_stop": float(args.alpha_stop) if policy_id.startswith("stop_plus") else 0.0,
                            "top_continue_channel": top_channel,
                            **probe,
                        }
                    )
            if idx % 8 == 0:
                log(f"{args.model}: audited {idx}/{len(candidates)} candidates")
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
    metrics = make_metrics(args.model, suppression_rows, rollout_rows)
    edges = make_edges(args.model, suppression_rows)
    payload = summarize_model(args.model, candidates, suppression_rows, effect_rows, stop_plus_rows, answer_rows, rollout_rows, metrics, edges, missing_rows)
    write_model_outputs(out_dir, args.model, payload, suppression_rows, effect_rows, stop_plus_rows, answer_rows, rollout_rows, observations, metrics, edges, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_metrics(model: str, rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for policy_id, value in mean_by(rows, "policy_id", "stop_margin_delta").items():
        subset = [r for r in rows if r.get("policy_id") == policy_id]
        flip_rate = sum(1 for r in subset if r.get("winner_flip_to_stop")) / len(subset) if subset else 0.0
        preserved_rate = sum(1 for r in subset if r.get("target_preserved")) / len(subset) if subset else 0.0
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase263",
                "created_at": utc_now(),
                "metric_id": f"phase263:{model}:{policy_id}:mean_stop_margin_delta",
                "scope": "continuation_suppression_policy",
                "model": model,
                "policy_id": policy_id,
                "metric_name": "mean_stop_margin_delta",
                "metric_value": value,
                "winner_flip_rate": round(flip_rate, 6),
                "target_preserved_rate": round(preserved_rate, 6),
                "rows": len(subset),
            }
        )
    for policy_id, value in mean_by(rollout_rows, "policy_id", "generated_token_count").items():
        subset = [r for r in rollout_rows if r.get("policy_id") == policy_id]
        stop_rate = sum(1 for r in subset if r.get("model_stop_executed")) / len(subset) if subset else 0.0
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase263",
                "created_at": utc_now(),
                "metric_id": f"phase263:{model}:{policy_id}:rollout_mean_tokens",
                "scope": "rollout_probe",
                "model": model,
                "policy_id": policy_id,
                "metric_name": "rollout_mean_generated_tokens",
                "metric_value": value,
                "model_stop_rate": round(stop_rate, 6),
                "rows": len(subset),
            }
        )
    return metrics


def make_edges(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for policy_id, value in mean_by(rows, "policy_id", "stop_margin_delta").items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase263",
                "created_at": utc_now(),
                "edge_id": f"phase263:{model}:{policy_id}:to_stop_margin",
                "source": f"node:{policy_id}",
                "target": "node:StopVsContinuationCompetition",
                "edge_type": "continuation_suppression_candidate_effect",
                "model": model,
                "evidence_type": "final_hidden_readout_level_intervention",
                "effect_direction": "improves_stop_margin" if value > 0 else "weak_or_negative",
                "effect_size": value,
                "confidence": 0.48 if value > 0 else 0.30,
                "supporting_phases": ["Phase262", "Phase263"],
                "status": "readout_level_causal_audit_not_full_rollout_closure",
            }
        )
    return edges


def summarize_model(
    model: str,
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    stop_plus_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy_id"])].append(row)
    policy_summary = {}
    for policy_id, items in by_policy.items():
        policy_summary[policy_id] = {
            "rows": len(items),
            "mean_stop_margin_delta": round(mean(safe_float(x["stop_margin_delta"]) for x in items), 6),
            "mean_top_channel_logit_delta": round(mean(safe_float(x["top_channel_logit_delta"]) for x in items), 6),
            "winner_flip_rate": round(sum(1 for x in items if x.get("winner_flip_to_stop")) / len(items), 6),
            "target_preserved_rate": round(sum(1 for x in items if x.get("target_preserved")) / len(items), 6),
        }
    rollout_summary = {}
    for policy_id, items in defaultdict(list, {k: [r for r in rollout_rows if r.get("policy_id") == k] for k in set(r.get("policy_id") for r in rollout_rows)}).items():
        rollout_summary[policy_id] = {
            "rows": len(items),
            "mean_generated_tokens": round(mean(safe_float(x["generated_token_count"]) for x in items), 6) if items else 0.0,
            "model_stop_rate": round(sum(1 for x in items if x.get("model_stop_executed")) / len(items), 6) if items else 0.0,
        }
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Continuation suppression candidate causal audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "candidate_count": len(candidates),
        "suppression_rows": len(rows),
        "channel_causal_effect_rows": len(effect_rows),
        "stop_plus_rows": len(stop_plus_rows),
        "answer_preservation_rows": len(answer_rows),
        "rollout_probe_rows": len(rollout_rows),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "policy_summary": policy_summary,
        "rollout_summary": rollout_summary,
    }


def write_model_outputs(out_dir: Path, model: str, payload: dict[str, Any], suppression_rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]], stop_plus_rows: list[dict[str, Any]], answer_rows: list[dict[str, Any]], rollout_rows: list[dict[str, Any]], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]], missing_rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"phase263_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase263_{model}_continuation_suppression_rows.jsonl", suppression_rows)
    write_jsonl(out_dir / f"phase263_{model}_channel_causal_effect_rows.jsonl", effect_rows)
    write_jsonl(out_dir / f"phase263_{model}_stop_plus_suppression_rows.jsonl", stop_plus_rows)
    write_jsonl(out_dir / f"phase263_{model}_answer_preservation_rows.jsonl", answer_rows)
    write_jsonl(out_dir / f"phase263_{model}_rollout_probe_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase263_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase263_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase263_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase263_{model}_missing_rows.jsonl", missing_rows)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase263_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    suppression: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    stop_plus: list[dict[str, Any]] = []
    answer: list[dict[str, Any]] = []
    rollout: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        suppression.extend(read_jsonl(out_dir / f"phase263_{model}_continuation_suppression_rows.jsonl"))
        effects.extend(read_jsonl(out_dir / f"phase263_{model}_channel_causal_effect_rows.jsonl"))
        stop_plus.extend(read_jsonl(out_dir / f"phase263_{model}_stop_plus_suppression_rows.jsonl"))
        answer.extend(read_jsonl(out_dir / f"phase263_{model}_answer_preservation_rows.jsonl"))
        rollout.extend(read_jsonl(out_dir / f"phase263_{model}_rollout_probe_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase263_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase263_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase263_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase263_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.85,
        "trace_signature_validation": 0.46,
        "semantic_done_signature": 0.24,
        "done_state_cluster_map": 0.21,
        "template_semantic_disentanglement": 0.19,
        "sdone_rstop_bridge": 0.08,
        "stop_continuation_competition": 0.20,
        "continuation_regime_decomposition": 0.18,
        "continuation_suppression_causal_audit": 0.10,
        "residual_state_signature": 0.55,
        "readout_competition_trace": 0.77,
        "stepwise_rollout_trace": 0.42,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.65,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Continuation suppression candidate causal audit",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "suppression_rows": len(suppression),
        "channel_causal_effect_rows": len(effects),
        "stop_plus_rows": len(stop_plus),
        "answer_preservation_rows": len(answer),
        "rollout_probe_rows": len(rollout),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "mean_stop_margin_delta_by_policy": mean_by(suppression, "policy_id", "stop_margin_delta"),
        "mean_top_channel_logit_delta_by_policy": mean_by(suppression, "policy_id", "top_channel_logit_delta"),
        "winner_flip_rate_by_policy": rate_by(suppression, "policy_id", "winner_flip_to_stop"),
        "target_preserved_rate_by_policy": rate_by(suppression, "policy_id", "target_preserved"),
        "rollout_mean_tokens_by_policy": mean_by(rollout, "policy_id", "generated_token_count"),
        "rollout_stop_rate_by_policy": rate_by(rollout, "policy_id", "model_stop_executed"),
        "progress": progress,
    }
    write_json(out_dir / "phase263_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase263_continuation_suppression_rows.jsonl", suppression)
    write_jsonl(out_dir / "phase263_channel_causal_effect_rows.jsonl", effects)
    write_jsonl(out_dir / "phase263_stop_plus_suppression_rows.jsonl", stop_plus)
    write_jsonl(out_dir / "phase263_answer_preservation_rows.jsonl", answer)
    write_jsonl(out_dir / "phase263_rollout_probe_rows.jsonl", rollout)
    write_jsonl(out_dir / "phase263_observations.jsonl", observations)
    write_jsonl(out_dir / "phase263_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase263_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase263_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase263", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def rate_by(rows: list[dict[str, Any]], group_key: str, bool_key: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(bool(row.get(bool_key)))
    return {k: round(sum(v) / len(v), 6) for k, v in grouped.items() if v}


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase263 Continuation Suppression Candidate Causal Audit",
        "",
        f"- status: {payload['status']}",
        f"- suppression_rows: {payload['suppression_rows']}",
        f"- channel_causal_effect_rows: {payload['channel_causal_effect_rows']}",
        f"- rollout_probe_rows: {payload['rollout_probe_rows']}",
        f"- mean_stop_margin_delta_by_policy: {json.dumps(payload['mean_stop_margin_delta_by_policy'], ensure_ascii=False)}",
        f"- mean_top_channel_logit_delta_by_policy: {json.dumps(payload['mean_top_channel_logit_delta_by_policy'], ensure_ascii=False)}",
        f"- winner_flip_rate_by_policy: {json.dumps(payload['winner_flip_rate_by_policy'], ensure_ascii=False)}",
        f"- target_preserved_rate_by_policy: {json.dumps(payload['target_preserved_rate_by_policy'], ensure_ascii=False)}",
        f"- rollout_mean_tokens_by_policy: {json.dumps(payload['rollout_mean_tokens_by_policy'], ensure_ascii=False)}",
        f"- rollout_stop_rate_by_policy: {json.dumps(payload['rollout_stop_rate_by_policy'], ensure_ascii=False)}",
    ]
    (out_dir / "phase263_continuation_suppression_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-mode", type=int, default=8)
    parser.add_argument("--max-candidates-per-policy", type=int, default=8)
    parser.add_argument("--lambdas", default="2,4,8,12")
    parser.add_argument("--alpha-stop", type=float, default=4.0)
    parser.add_argument("--rollout-candidates", type=int, default=5)
    parser.add_argument("--rollout-tokens", type=int, default=24)
    parser.add_argument("--rollout-lambda", type=float, default=8.0)
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
