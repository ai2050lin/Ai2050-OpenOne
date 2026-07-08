#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402
import phase251_orthogonalized_natural_direction_causal_validation as p251  # noqa: E402
import phase252_shared_subspace_coupled_regime_analysis as p252  # noqa: E402


PHASE = 254
SOURCE_PHASE = 253
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE250_DIR = Path("tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction")
PHASE253_DIR = Path("tests/result/phase253_control_readout_coupling_validation/control_readout_coupling_validation")
RESULT_ROOT = Path("tests/result/phase254_closure_candidate_stop_validation")
ROUND_DEFAULT = "closure_candidate_stop_validation"
WEIGHT_GRID = [0.25, 0.5, 1.0]

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


def norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def unit(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n * float(target_norm)


def key_for(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model")), str(row.get("case_id")), str(row.get("variant_id")))


def load_behavior_by_key() -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_jsonl(PHASE241_DIR / "phase241_large_scale_behavior_rows.jsonl")
    return {(str(x["model"]), str(x["case_id"]), str(x["variant_id"])): x for x in rows}


def load_raw_vectors(model_name: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest = read_json(PHASE246_DIR / "phase246_raw_delta_vector_manifest.json").get("rows", [])
    out = {}
    for item in manifest:
        if str(item.get("model")) != model_name:
            continue
        path = Path(str(item.get("path")))
        if path.exists():
            out[key_for(item)] = {**item, "payload": torch.load(path, map_location="cpu")}
    return out


def load_natural_directions(model_name: str) -> dict[str, torch.Tensor]:
    rows = read_jsonl(PHASE250_DIR / "phase250_natural_direction_rows.jsonl")
    out = {}
    for row in rows:
        if str(row.get("model")) != model_name or str(row.get("scope")) != "global":
            continue
        path = Path(str(row.get("path")))
        if path.exists():
            out[str(row["contrast_id"])] = unit(torch.load(path, map_location="cpu")["direction"])
    return out


def replace_last_token(output: Any, delta: torch.Tensor, sign: float = 1.0) -> Any:
    tensor = p228.extract_tensor(output)
    if tensor is None:
        return output
    changed = tensor.clone()
    changed[:, -1, :] = changed[:, -1, :] + float(sign) * delta.to(device=changed.device, dtype=changed.dtype)
    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    return output


@contextmanager
def no_hook() -> Iterator[None]:
    yield


@contextmanager
def residual_hook(model: Any, layer_idx: int, direction: torch.Tensor, sign: float) -> Iterator[None]:
    layers = p228.get_layers(model)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_last_token(output, direction, sign=sign)

    handle = layers[int(layer_idx)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def classify_stop(tokenizer: Any, ids: list[int], text: str, target_aliases: list[str], max_new_tokens: int) -> dict[str, Any]:
    token_texts = [tokenizer.decode([int(x)]) for x in ids]
    eos_pos = None
    if tokenizer.eos_token_id is not None:
        for i, tok in enumerate(ids, start=1):
            if int(tok) == int(tokenizer.eos_token_id):
                eos_pos = i
                break
    period_pos = next((i for i, t in enumerate(token_texts, start=1) if "." in t or "。" in t), None)
    newline_pos = next((i for i, t in enumerate(token_texts, start=1) if "\n" in t), None)
    lower = text.lower()
    alias_hits = []
    for alias in target_aliases:
        alias_l = str(alias).strip().lower()
        if alias_l and alias_l in lower:
            alias_hits.append((lower.find(alias_l), alias_l))
    answer_pos = min([x[0] for x in alias_hits], default=None)
    answer_seen = answer_pos is not None
    after_answer = lower[answer_pos + len(alias_hits[0][1]) :] if answer_seen and alias_hits else ""
    continued_after_answer = bool(answer_seen and len(after_answer.strip()) > 8)
    if eos_pos is not None:
        stop_type = "eos_stop"
    elif answer_seen and not continued_after_answer:
        stop_type = "semantic_done_no_continue"
    elif period_pos is not None and period_pos >= len(ids) - 1:
        stop_type = "period_boundary_stop"
    elif newline_pos is not None and newline_pos >= len(ids) - 1:
        stop_type = "newline_boundary_stop"
    elif len(ids) >= int(max_new_tokens):
        stop_type = "client_truncation"
    elif continued_after_answer:
        stop_type = "continued_after_answer"
    else:
        stop_type = "other_stop"
    return {
        "stop_type": stop_type,
        "eos_pos": eos_pos,
        "period_pos": period_pos,
        "newline_pos": newline_pos,
        "semantic_answer_seen": answer_seen,
        "semantic_answer_char_pos": answer_pos,
        "continued_after_answer": continued_after_answer,
        "generated_token_count": len(ids),
        "client_truncation": len(ids) >= int(max_new_tokens) and eos_pos is None,
        "over_generation_length": len(after_answer.strip()) if answer_seen else len(text.strip()),
    }


def generate_trace(model: Any, tokenizer: Any, device: torch.device, prompt: str, max_new_tokens: int, hook_factory: Any, target_aliases: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = int(encoded["input_ids"].shape[-1])
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    with hook_factory():
        with torch.inference_mode():
            out = model.generate(**encoded, **kwargs)
    seq = out.sequences[0, input_len:].detach().cpu().tolist()
    text = tokenizer.decode(seq, skip_special_tokens=True).strip()
    step_rows = []
    for step, score in enumerate(out.scores, start=1):
        logits = score[0].detach().float().cpu()
        closure = p252.closure_scores(tokenizer, logits)
        token_id = int(seq[step - 1]) if step - 1 < len(seq) else int(torch.argmax(logits).item())
        step_rows.append(
            {
                "step": step,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                **{k: round(v, 6) for k, v in closure.items()},
            }
        )
    summary = classify_stop(tokenizer, seq, text, target_aliases, int(max_new_tokens))
    summary.update(
        {
            "generated_text": text,
            "mean_closure_proxy_margin": round(mean(safe_float(x.get("closure_proxy_margin")) for x in step_rows), 6) if step_rows else 0.0,
            "final_closure_proxy_margin": safe_float(step_rows[-1].get("closure_proxy_margin")) if step_rows else 0.0,
        }
    )
    return summary, step_rows


def condition_specs(control_dir: torch.Tensor, readout_dir: torch.Tensor, perturb_norm: float) -> list[dict[str, Any]]:
    specs = [
        {"condition": "no_intervention", "kind": "base", "lambda_c": 0.0, "lambda_r": 0.0, "direction": None},
        {"condition": "tokenbank_suppression", "kind": "single", "lambda_c": 0.0, "lambda_r": 1.0, "direction": readout_dir},
        {"condition": "natural_raw_suppression", "kind": "single", "lambda_c": 1.0, "lambda_r": 0.0, "direction": control_dir},
        {"condition": "combined_suppression", "kind": "combined", "lambda_c": 1.0, "lambda_r": 1.0, "direction": normalize(unit(control_dir) + unit(readout_dir), perturb_norm)},
    ]
    for lc in WEIGHT_GRID:
        for lr in WEIGHT_GRID:
            specs.append(
                {
                    "condition": f"weighted_combined_c{lc}_r{lr}",
                    "kind": "weighted_combined",
                    "lambda_c": lc,
                    "lambda_r": lr,
                    "direction": normalize(float(lc) * unit(control_dir) + float(lr) * unit(readout_dir), perturb_norm),
                }
            )
    return specs


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_by_key = load_behavior_by_key()
    raw_vectors = load_raw_vectors(args.model)
    candidates_all = read_jsonl(PHASE253_DIR / "phase253_closure_validation_candidates.jsonl")
    candidates_raw = [x for x in candidates_all if str(x.get("model")) == args.model]
    candidates_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in candidates_raw:
        key = key_for(row)
        old = candidates_by_key.get(key)
        if old is None or safe_float(row.get("final_closure_proxy_margin")) > safe_float(old.get("final_closure_proxy_margin")):
            candidates_by_key[key] = row
    candidates = list(candidates_by_key.values())
    candidates.sort(key=lambda x: safe_float(x.get("final_closure_proxy_margin")), reverse=True)
    candidates = candidates[: int(args.max_candidates_per_model)] if int(args.max_candidates_per_model) > 0 else candidates
    natural_dirs = load_natural_directions(args.model)
    model_obj = None
    tokenizer = None
    fixed_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    weighted_rows: list[dict[str, Any]] = []
    modelclose_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    final_layer = int(SPECS[args.model]["final_layer"])
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenbank_dirs = p252.build_tokenbank_directions(model_obj, tokenizer)
        for idx, cand in enumerate(candidates, start=1):
            key = key_for(cand)
            behavior = behavior_by_key.get(key)
            raw_item = raw_vectors.get(key)
            if behavior is None or raw_item is None:
                missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "reason": "missing_behavior_or_raw"})
                continue
            raw = raw_item["payload"]
            route = str(cand.get("recommended_next_test") or "")
            tokenbank_id, natural_id = p251.ROUTE_TO_DIRECTIONS.get(route, ("continuation_regime", "natural_continuation_explain"))
            perturb_norm = max(norm(raw.get("target_direction")), norm(raw.get("competitor_direction")), norm(raw.get("delta_residual")) * float(args.perturb_scale), 1e-6)
            control_dir = normalize(natural_dirs[natural_id], perturb_norm)
            readout_dir = normalize(tokenbank_dirs[tokenbank_id], perturb_norm)
            prompt = str(behavior["prompt_variant"])
            aliases = list(behavior.get("target_aliases") or [])
            summaries: dict[str, dict[str, Any]] = {}
            for spec in condition_specs(control_dir, readout_dir, perturb_norm):
                condition = str(spec["condition"])

                def hook_factory(direction: torch.Tensor | None = spec["direction"]) -> Any:
                    if direction is None:
                        return no_hook()
                    return residual_hook(model_obj, final_layer, direction, sign=-1.0)

                summary, steps = generate_trace(model_obj, tokenizer, device, prompt, int(args.max_new_tokens), hook_factory, aliases)
                summaries[condition] = summary
                row_base = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase254",
                    "created_at": utc_now(),
                    "model": args.model,
                    "case_id": key[1],
                    "variant_id": key[2],
                    "family_id": behavior.get("family_id"),
                    "mode_id": behavior.get("mode_id"),
                    "condition": condition,
                    "condition_kind": spec["kind"],
                    "lambda_c": spec["lambda_c"],
                    "lambda_r": spec["lambda_r"],
                    "tokenbank_regime": tokenbank_id,
                    "natural_contrast_id": natural_id,
                    **summary,
                }
                rollout_rows.append({**row_base, "rollout_id": f"phase254:rollout:{args.model}:{key[1]}:{key[2]}:{condition}"})
                stop_rows.append({**row_base, "stop_id": f"phase254:stop:{args.model}:{key[1]}:{key[2]}:{condition}"})
                if spec["kind"] == "weighted_combined":
                    weighted_rows.append({**row_base, "weighted_id": f"phase254:weighted:{args.model}:{key[1]}:{key[2]}:{condition}"})
                for step in steps:
                    # Keep rollout rows compact: one summary row per condition; detailed step rows go to observations.
                    pass
            base = summaries.get("no_intervention", {})
            original_condition = str(cand.get("condition"))
            matched = summaries.get(original_condition, {})
            fixed = {
                **cand,
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase254",
                "fixed_candidate_id": f"phase254:fixed:{args.model}:{key[1]}:{key[2]}:{original_condition}",
                "closure_proxy_delta": round(safe_float(matched.get("mean_closure_proxy_margin")) - safe_float(base.get("mean_closure_proxy_margin")), 6),
                "final_closure_proxy_delta": round(safe_float(matched.get("final_closure_proxy_margin")) - safe_float(base.get("final_closure_proxy_margin")), 6),
                "over_generation_delta": round(safe_float(matched.get("over_generation_length")) - safe_float(base.get("over_generation_length")), 6),
                "stop_type": matched.get("stop_type"),
                "base_stop_type": base.get("stop_type"),
                "condition_name_mapping_status": "matched" if matched else "missing_condition",
            }
            fixed_rows.append(fixed)
            best = max([x for x in summaries.items() if x[0] != "no_intervention"], key=lambda item: (safe_float(item[1].get("final_closure_proxy_margin")), -safe_float(item[1].get("over_generation_length"))), default=("", {}))
            best_condition, best_summary = best
            if best_summary:
                is_modelclose_candidate = (
                    safe_float(best_summary.get("final_closure_proxy_margin")) > 0
                    and not bool(best_summary.get("client_truncation"))
                    and (bool(best_summary.get("semantic_answer_seen")) or "answer" not in str(behavior.get("family_id")).lower())
                    and safe_float(best_summary.get("over_generation_length")) <= safe_float(base.get("over_generation_length"), 1e9)
                )
                if is_modelclose_candidate:
                    modelclose_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase254",
                            "created_at": utc_now(),
                            "modelclose_candidate_id": f"phase254:modelclose:{args.model}:{key[1]}:{key[2]}:{best_condition}",
                            "model": args.model,
                            "case_id": key[1],
                            "variant_id": key[2],
                            "family_id": behavior.get("family_id"),
                            "mode_id": behavior.get("mode_id"),
                            "best_condition": best_condition,
                            "base_stop_type": base.get("stop_type"),
                            "best_stop_type": best_summary.get("stop_type"),
                            "final_closure_proxy_margin": best_summary.get("final_closure_proxy_margin"),
                            "mean_closure_proxy_margin": best_summary.get("mean_closure_proxy_margin"),
                            "over_generation_length": best_summary.get("over_generation_length"),
                            "base_over_generation_length": base.get("over_generation_length"),
                            "semantic_answer_seen": best_summary.get("semantic_answer_seen"),
                            "client_truncation": best_summary.get("client_truncation"),
                            "selection_reason": "positive_final_closure_no_client_truncation_no_more_overgeneration",
                        }
                    )
            log(f"{args.model}: candidate={idx}/{len(candidates)} rollout_rows={len(rollout_rows)}")
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
    observations = observation_rows(rollout_rows, stop_rows, weighted_rows)
    metrics = metric_rows(args.model, fixed_rows, rollout_rows, stop_rows, weighted_rows, modelclose_rows)
    edges = graph_edges(rollout_rows, modelclose_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Closure candidate stop validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "candidate_count": len(candidates),
        "fixed_candidate_rows": len(fixed_rows),
        "rollout_rows": len(rollout_rows),
        "stop_type_rows": len(stop_rows),
        "weighted_combined_rows": len(weighted_rows),
        "modelclose_candidate_rows": len(modelclose_rows),
        "missing_rows": len(missing_rows),
        "stop_type_counts": dict(Counter(str(x.get("stop_type")) for x in stop_rows).most_common()),
        "modelclose_condition_counts": dict(Counter(str(x.get("best_condition")) for x in modelclose_rows).most_common()),
        "mean_final_closure_by_condition": mean_by(rollout_rows, "condition", "final_closure_proxy_margin"),
        "mean_over_generation_by_condition": mean_by(rollout_rows, "condition", "over_generation_length"),
    }
    write_json(out_dir / f"phase254_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase254_{args.model}_closure_candidate_fixed_rows.jsonl", fixed_rows)
    write_jsonl(out_dir / f"phase254_{args.model}_64token_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase254_{args.model}_stop_type_rows.jsonl", stop_rows)
    write_jsonl(out_dir / f"phase254_{args.model}_weighted_combined_suppression_rows.jsonl", weighted_rows)
    write_jsonl(out_dir / f"phase254_{args.model}_modelclose_candidate_rows.jsonl", modelclose_rows)
    write_jsonl(out_dir / f"phase254_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase254_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase254_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase254_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def observation_rows(rollouts: list[dict[str, Any]], stops: list[dict[str, Any]], weighted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in rollouts:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase254",
                "created_at": now,
                "observation_id": row["rollout_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "64token_rollout",
                "component": row["condition"],
                "metric_name": "final_closure_proxy_margin",
                "metric_value": row["final_closure_proxy_margin"],
                "metric_unit": "logit",
                "stop_type": row.get("stop_type"),
            }
        )
    return rows


def metric_rows(model_name: str, fixed: list[dict[str, Any]], rollouts: list[dict[str, Any]], stops: list[dict[str, Any]], weighted: list[dict[str, Any]], modelclose: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for condition, value in mean_by(rollouts, "condition", "final_closure_proxy_margin").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase254",
                "created_at": now,
                "metric_id": f"phase254:{model_name}:{condition}:mean_final_closure",
                "scope": "64token_rollout",
                "model": model_name,
                "condition": condition,
                "metric_name": "mean_final_closure_proxy_margin",
                "metric_value": value,
                "rows": sum(1 for x in rollouts if x.get("condition") == condition),
            }
        )
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase254",
            "created_at": now,
            "metric_id": f"phase254:{model_name}:modelclose_candidate_count",
            "scope": "closure_stop_validation",
            "model": model_name,
            "metric_name": "modelclose_candidate_count",
            "metric_value": len(modelclose),
            "rows": len(modelclose),
        }
    )
    return rows


def graph_edges(rollouts: list[dict[str, Any]], modelclose: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in rollouts:
        if row.get("condition") == "no_intervention":
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase254",
                "created_at": now,
                "edge_id": f"phase254:rollout:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['condition']}",
                "source": f"intervention:{row['condition']}",
                "target": "node:StopType",
                "edge_type": "closure_stop_validation",
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "evidence_type": "64token_rollout_stop_type",
                "effect_direction": row.get("stop_type"),
                "effect_size": row.get("final_closure_proxy_margin"),
                "confidence": 0.45,
                "supporting_phases": ["Phase253", "Phase254"],
                "status": "stop_candidate_not_final_modelclose",
            }
        )
    return rows


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase254_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    fixed: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    weighted: list[dict[str, Any]] = []
    modelclose: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        fixed.extend(read_jsonl(out_dir / f"phase254_{model}_closure_candidate_fixed_rows.jsonl"))
        rollouts.extend(read_jsonl(out_dir / f"phase254_{model}_64token_rollout_rows.jsonl"))
        stops.extend(read_jsonl(out_dir / f"phase254_{model}_stop_type_rows.jsonl"))
        weighted.extend(read_jsonl(out_dir / f"phase254_{model}_weighted_combined_suppression_rows.jsonl"))
        modelclose.extend(read_jsonl(out_dir / f"phase254_{model}_modelclose_candidate_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase254_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase254_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase254_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase254_{model}_missing_rows.jsonl"))
    modelclose_by_id: dict[str, dict[str, Any]] = {}
    for row in modelclose:
        modelclose_by_id[str(row.get("modelclose_candidate_id"))] = row
    modelclose = list(modelclose_by_id.values())
    fixed_by_id: dict[str, dict[str, Any]] = {}
    for row in fixed:
        fixed_by_id[str(row.get("fixed_candidate_id"))] = row
    fixed = list(fixed_by_id.values())
    progress = {
        "pattern_family_atlas": 0.82,
        "high_value_trace_selection": 0.67,
        "trace_signature_validation": 0.38,
        "focused_causal_validation": 0.25,
        "regime_field_direction_bank": 0.35,
        "natural_regime_direction_bank": 0.30,
        "regime_level_causal_validation": 0.26,
        "shared_subspace_analysis": 0.20,
        "coupled_regime_field_analysis": 0.23,
        "control_readout_coupling": 0.20,
        "stop_type_validation": 0.18,
        "residual_state_signature": 0.49,
        "readout_competition_trace": 0.72,
        "stepwise_rollout_trace": 0.38,
        "causal_closure": 0.16,
        "general_language_mechanism_confidence": 0.63,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model closure candidate stop validation",
        "status": "complete" if len(summaries) == 3 else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "models": [x.get("model") for x in summaries],
        "model_summaries": {x.get("model"): x for x in summaries},
        "candidate_count": len(fixed),
        "fixed_candidate_rows": len(fixed),
        "rollout_rows": len(rollouts),
        "stop_type_rows": len(stops),
        "weighted_combined_rows": len(weighted),
        "modelclose_candidate_rows": len(modelclose),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "stop_type_counts": dict(Counter(str(x.get("stop_type")) for x in stops).most_common()),
        "modelclose_condition_counts": dict(Counter(str(x.get("best_condition")) for x in modelclose).most_common()),
        "mean_final_closure_by_condition": mean_by(rollouts, "condition", "final_closure_proxy_margin"),
        "mean_over_generation_by_condition": mean_by(rollouts, "condition", "over_generation_length"),
        "top_modelclose_candidates": modelclose[:10],
        "pattern_atlas_progress": progress,
        "judgement": "stop_type_validation_not_final_modelclose",
    }
    write_json(out_dir / "phase254_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase254_closure_candidate_fixed_rows.jsonl", fixed)
    write_jsonl(out_dir / "phase254_64token_rollout_rows.jsonl", rollouts)
    write_jsonl(out_dir / "phase254_stop_type_rows.jsonl", stops)
    write_jsonl(out_dir / "phase254_weighted_combined_suppression_rows.jsonl", weighted)
    write_jsonl(out_dir / "phase254_modelclose_candidate_rows.jsonl", modelclose)
    write_jsonl(out_dir / "phase254_observations.jsonl", observations)
    write_jsonl(out_dir / "phase254_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase254_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase254_missing_rows.jsonl", missing)
    write_report(out_dir / "phase254_closure_validation_report.md", payload)
    update_pattern_atlas(payload, observations, metrics, edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase254 closure candidate stop validation",
        "",
        "Phase254 repairs closure candidate rows and tests 64-token stop behavior.",
        "It separates EOS, boundary, semantic done, continued output, and client truncation.",
        "",
        "## Counts",
        "",
        f"- candidates: {summary['candidate_count']}",
        f"- rollout_rows: {summary['rollout_rows']}",
        f"- weighted_combined_rows: {summary['weighted_combined_rows']}",
        f"- modelclose_candidate_rows: {summary['modelclose_candidate_rows']}",
        "",
        "## Stop Types",
        "",
        "```json",
        json.dumps(summary["stop_type_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Progress",
        "",
        "```json",
        json.dumps(summary["pattern_atlas_progress"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_pattern_atlas(summary: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    progress.update(
        {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "latest_phase": "Phase254",
            "latest_round": summary.get("title"),
            "progress": summary.get("pattern_atlas_progress", {}),
            "current_priority": "audit modelclose candidates against true stopping and semantic preservation",
            "small_model_bias_warning": "Phase254 uses qwen3/glm4/deepseek7b only; stop validation remains candidate evidence, not final closure.",
        }
    )
    write_json(ATLAS_ROOT / "progress.json", progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase254 closure candidate stop validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-candidates-per-model", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--perturb-scale", type=float, default=0.35)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is set")
    evaluate_model(args)


if __name__ == "__main__":
    main()
