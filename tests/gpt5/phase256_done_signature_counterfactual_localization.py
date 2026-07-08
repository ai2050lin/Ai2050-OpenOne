#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase251_orthogonalized_natural_direction_causal_validation as p251  # noqa: E402
import phase252_shared_subspace_coupled_regime_analysis as p252  # noqa: E402
import phase255_modelclose_internal_stop_trace as p255  # noqa: E402


PHASE = 256
SOURCE_PHASE = 255
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE250_DIR = Path("tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction")
PHASE255_DIR = Path("tests/result/phase255_modelclose_internal_stop_trace/modelclose_internal_stop_trace")
RESULT_ROOT = Path("tests/result/phase256_done_signature_counterfactual_localization")
ROUND_DEFAULT = "done_signature_counterfactual_localization"

SPECS = {
    "qwen3": {"final_layer": 33, "observe_layers": [20, 26, 29, 31, 33]},
    "glm4": {"final_layer": 32, "observe_layers": [20, 26, 28, 30, 32]},
    "deepseek7b": {"final_layer": 27, "observe_layers": [16, 22, 24, 26, 27]},
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


def norm(vec: torch.Tensor | None) -> float:
    if vec is None or not torch.is_tensor(vec):
        return 0.0
    return float(torch.linalg.vector_norm(vec.float()).item())


def normalize(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    n = torch.linalg.vector_norm(vec).item()
    if n <= 1e-8:
        return torch.zeros_like(vec)
    return vec / n * float(target_norm)


def dot(vec: torch.Tensor, direction: torch.Tensor) -> float:
    if vec.numel() != direction.numel():
        return 0.0
    return float(torch.dot(vec.float(), unit(direction).float()).item())


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


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


def make_hook_factory(model: Any, layer_idx: int, direction: torch.Tensor | None) -> Any:
    def factory() -> Any:
        if direction is None:
            return no_hook()
        return residual_hook(model, layer_idx, direction, sign=-1.0)

    return factory


def reconstruct_generated_ids(step_rows: list[dict[str, Any]], model: str, case_id: str, variant_id: str, condition: str) -> list[int]:
    rows = [
        x
        for x in step_rows
        if str(x.get("model")) == model
        and str(x.get("case_id")) == case_id
        and str(x.get("variant_id")) == variant_id
        and str(x.get("condition")) == condition
    ]
    rows.sort(key=lambda x: int(x["step"]))
    return [int(x["token_id"]) for x in rows]


def capture_prefix_hidden(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    generated_ids: list[int],
    condition: str,
    hook_factory: Any,
    observe_layers: list[int],
    target_aliases: list[str],
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[int, dict[str, Any]]]:
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    base_ids = prompt_ids["input_ids"]
    hidden_by_step_layer: dict[tuple[int, int], torch.Tensor] = {}
    logits_by_step: dict[int, dict[str, Any]] = {}
    for next_step in range(1, len(generated_ids) + 1):
        prefix = generated_ids[: next_step - 1]
        if prefix:
            prefix_ids = torch.tensor([prefix], device=device, dtype=base_ids.dtype)
            input_ids = torch.cat([base_ids, prefix_ids], dim=1)
        else:
            input_ids = base_ids
        attention_mask = torch.ones_like(input_ids)
        last_pos = int(input_ids.shape[-1]) - 1
        with hook_factory():
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
        logits = out.logits[0, last_pos].detach().float().cpu()
        closure = p252.closure_scores(tokenizer, logits)
        readout = p239.readout_metrics(tokenizer, logits, target_aliases)
        eos_logit = logits[int(tokenizer.eos_token_id)].item() if tokenizer.eos_token_id is not None else 0.0
        logits_by_step[next_step] = {
            "condition": condition,
            "next_step": next_step,
            "next_token_id": int(generated_ids[next_step - 1]),
            "next_token_text": tokenizer.decode([int(generated_ids[next_step - 1])]),
            "eos_logit": round(float(eos_logit), 6),
            **{f"closure_{k}": round(v, 6) for k, v in closure.items()},
            **{f"readout_{k}": v for k, v in readout.items()},
        }
        for layer in observe_layers:
            if int(layer) + 1 < len(out.hidden_states):
                hidden_by_step_layer[(next_step, int(layer))] = out.hidden_states[int(layer) + 1][0, last_pos].detach().float().cpu()
    return hidden_by_step_layer, logits_by_step


def make_done_direction(
    hidden_maps: dict[str, dict[tuple[int, int], torch.Tensor]],
    stop_by_condition: dict[str, dict[str, Any]],
    final_layer: int,
) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
    vectors = []
    rows = []
    for condition, stop in stop_by_condition.items():
        eos = stop.get("eos_pos")
        ans = stop.get("answer_first_step")
        if stop.get("stop_type") != "eos_stop" or eos is None or ans is None:
            continue
        h_eos = hidden_maps.get(condition, {}).get((int(eos), final_layer))
        h_ans = hidden_maps.get(condition, {}).get((int(ans), final_layer))
        if h_eos is None or h_ans is None:
            continue
        delta = h_eos - h_ans
        vectors.append(delta)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase256",
                "created_at": utc_now(),
                "done_vector_component_id": f"phase256:done_component:{condition}",
                "condition": condition,
                "answer_step": int(ans),
                "eos_step": int(eos),
                "delta_norm": round(norm(delta), 6),
                "construction": "hidden_at_eos_prefix_minus_hidden_at_answer_prefix",
            }
        )
    if not vectors:
        return None, rows
    return unit(torch.stack(vectors).mean(dim=0)), rows


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_seed_rows = [x for x in read_jsonl(PHASE255_DIR / "phase255_stop_trace_rows.jsonl") if str(x.get("model")) == args.model]
    step_seed_rows = [x for x in read_jsonl(PHASE255_DIR / "phase255_generation_step_rows.jsonl") if str(x.get("model")) == args.model]
    if not stop_seed_rows:
        payload = {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "title": "Done signature counterfactual localization",
            "status": "complete_no_phase255_stop_seed",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "seed_stop_rows": 0,
            "done_signature_rows": 0,
            "counterfactual_rows": 0,
            "missing_rows": 0,
        }
        write_json(out_dir / f"phase256_{args.model}_summary.json", payload)
        for name in ["done_vector_component_rows", "done_signature_rows", "counterfactual_rows", "observations", "metrics", "graph_edges", "missing_rows"]:
            write_jsonl(out_dir / f"phase256_{args.model}_{name}.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    behavior_by_key = load_behavior_by_key()
    raw_vectors = load_raw_vectors(args.model)
    natural_dirs = load_natural_directions(args.model)
    token_signature_rows: list[dict[str, Any]] = []
    done_component_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model_obj = None
    tokenizer = None
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenbank_dirs = p252.build_tokenbank_directions(model_obj, tokenizer)
        final_layer = int(SPECS[args.model]["final_layer"])
        observe_layers = list(SPECS[args.model]["observe_layers"])
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in stop_seed_rows:
            grouped[key_for(row)].append(row)
        for key, seed_rows in grouped.items():
            behavior = behavior_by_key.get(key)
            raw_item = raw_vectors.get(key)
            if behavior is None or raw_item is None:
                missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "reason": "missing_behavior_or_raw"})
                continue
            raw = raw_item["payload"]
            tokenbank_id, natural_id = p251.ROUTE_TO_DIRECTIONS.get("", ("continuation_regime", "natural_continuation_explain"))
            perturb_norm = max(norm(raw.get("target_direction")), norm(raw.get("competitor_direction")), norm(raw.get("delta_residual")) * float(args.perturb_scale), 1e-6)
            control_dir = normalize(natural_dirs[natural_id], perturb_norm)
            readout_dir = normalize(tokenbank_dirs[tokenbank_id], perturb_norm)
            prompt = str(behavior["prompt_variant"])
            aliases = list(behavior.get("target_aliases") or [])
            stop_by_condition = {str(x["condition"]): x for x in seed_rows}
            hidden_maps: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
            logits_maps: dict[str, dict[int, dict[str, Any]]] = {}
            for stop in seed_rows:
                condition = str(stop["condition"])
                generated_ids = reconstruct_generated_ids(step_seed_rows, args.model, key[1], key[2], condition)
                if not generated_ids:
                    missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "condition": condition, "reason": "missing_generated_ids"})
                    continue
                direction = p255.condition_direction(condition, control_dir, readout_dir, perturb_norm)
                hook_factory = make_hook_factory(model_obj, final_layer, direction)
                hidden_map, logits_map = capture_prefix_hidden(
                    model_obj, tokenizer, device, prompt, generated_ids, condition, hook_factory, observe_layers, aliases
                )
                hidden_maps[condition] = hidden_map
                logits_maps[condition] = logits_map
            done_dir, component_rows = make_done_direction(hidden_maps, stop_by_condition, final_layer)
            done_component_rows.extend(
                {
                    **row,
                    "model": args.model,
                    "case_id": key[1],
                    "variant_id": key[2],
                    "family_id": behavior.get("family_id"),
                    "mode_id": behavior.get("mode_id"),
                }
                for row in component_rows
            )
            if done_dir is None:
                missing_rows.append({"model": args.model, "case_id": key[1], "variant_id": key[2], "reason": "done_direction_unavailable"})
                continue
            for condition, hidden_map in hidden_maps.items():
                stop = stop_by_condition.get(condition, {})
                answer_step = stop.get("answer_first_step")
                eos_step = stop.get("eos_pos")
                for (next_step, layer), vec in hidden_map.items():
                    logits_row = logits_maps.get(condition, {}).get(next_step, {})
                    phase_marker = "other"
                    if answer_step is not None and int(next_step) == int(answer_step):
                        phase_marker = "answer_first_step"
                    if eos_step is not None and int(next_step) == int(eos_step):
                        phase_marker = "eos_step"
                    if eos_step is not None and int(next_step) == int(eos_step) - 1:
                        phase_marker = "pre_eos_step"
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase256",
                        "created_at": utc_now(),
                        "done_signature_id": f"phase256:done_sig:{args.model}:{key[1]}:{key[2]}:{condition}:step{next_step}:L{layer}",
                        "model": args.model,
                        "case_id": key[1],
                        "variant_id": key[2],
                        "family_id": behavior.get("family_id"),
                        "mode_id": behavior.get("mode_id"),
                        "condition": condition,
                        "next_step": int(next_step),
                        "layer": int(layer),
                        "phase_marker": phase_marker,
                        "stop_type": stop.get("stop_type"),
                        "answer_first_step": answer_step,
                        "eos_pos": eos_step,
                        "done_projection": round(dot(vec, done_dir), 6),
                        "control_projection": round(dot(vec, control_dir), 6),
                        "readout_projection": round(dot(vec, readout_dir), 6),
                        **logits_row,
                    }
                    token_signature_rows.append(row)
            final_rows = [x for x in token_signature_rows if x["model"] == args.model and x["case_id"] == key[1] and x["variant_id"] == key[2] and int(x["layer"]) == final_layer]
            for condition, stop in stop_by_condition.items():
                rows = [x for x in final_rows if x["condition"] == condition]
                if not rows:
                    continue
                ans = stop.get("answer_first_step")
                eos = stop.get("eos_pos")
                ans_val = mean([safe_float(x["done_projection"]) for x in rows if ans is not None and int(x["next_step"]) == int(ans)] or [0.0])
                pre_eos_val = mean([safe_float(x["done_projection"]) for x in rows if eos is not None and int(x["next_step"]) == int(eos) - 1] or [0.0])
                eos_val = mean([safe_float(x["done_projection"]) for x in rows if eos is not None and int(x["next_step"]) == int(eos)] or [0.0])
                late_val = mean([safe_float(x["done_projection"]) for x in rows[-3:]] or [0.0])
                counterfactual_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase256",
                        "created_at": utc_now(),
                        "counterfactual_id": f"phase256:counterfactual:{args.model}:{key[1]}:{key[2]}:{condition}",
                        "model": args.model,
                        "case_id": key[1],
                        "variant_id": key[2],
                        "family_id": behavior.get("family_id"),
                        "mode_id": behavior.get("mode_id"),
                        "condition": condition,
                        "stop_type": stop.get("stop_type"),
                        "answer_done_projection": round(ans_val, 6),
                        "pre_eos_done_projection": round(pre_eos_val, 6),
                        "eos_done_projection": round(eos_val, 6),
                        "late_done_projection": round(late_val, 6),
                        "done_gain_answer_to_eos": round(eos_val - ans_val, 6) if eos is not None and ans is not None else None,
                        "done_gain_answer_to_late": round(late_val - ans_val, 6) if ans is not None else None,
                        "interpretation": "eos_aligned_done_growth" if stop.get("stop_type") == "eos_stop" and eos_val > ans_val else "no_eos_or_no_growth",
                    }
                )
            log(f"{args.model}: case={key[1]} conditions={len(seed_rows)} done_rows={len(token_signature_rows)}")
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
    observations = make_observations(token_signature_rows, counterfactual_rows)
    metrics = make_metrics(args.model, token_signature_rows, counterfactual_rows)
    edges = make_edges(counterfactual_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Done signature counterfactual localization",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "seed_stop_rows": len(stop_seed_rows),
        "done_vector_component_rows": len(done_component_rows),
        "done_signature_rows": len(token_signature_rows),
        "counterfactual_rows": len(counterfactual_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "interpretation_counts": dict(Counter(str(x.get("interpretation")) for x in counterfactual_rows).most_common()),
        "mean_done_projection_by_condition": mean_by([x for x in token_signature_rows if x.get("layer") == SPECS[args.model]["final_layer"]], "condition", "done_projection"),
        "mean_done_gain_answer_to_eos_by_condition": mean_by([x for x in counterfactual_rows if x.get("done_gain_answer_to_eos") is not None], "condition", "done_gain_answer_to_eos"),
    }
    write_json(out_dir / f"phase256_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase256_{args.model}_done_vector_component_rows.jsonl", done_component_rows)
    write_jsonl(out_dir / f"phase256_{args.model}_done_signature_rows.jsonl", token_signature_rows)
    write_jsonl(out_dir / f"phase256_{args.model}_counterfactual_rows.jsonl", counterfactual_rows)
    write_jsonl(out_dir / f"phase256_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase256_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase256_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase256_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def make_observations(signature_rows: list[dict[str, Any]], counterfactual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in signature_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase256",
                "created_at": now,
                "observation_id": row["done_signature_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "done_signature_projection",
                "component": f"{row['condition']}:L{row['layer']}",
                "metric_name": "done_projection",
                "metric_value": row["done_projection"],
                "metric_unit": "projection",
            }
        )
    for row in counterfactual_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase256",
                "created_at": now,
                "observation_id": row["counterfactual_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "done_counterfactual_summary",
                "component": row["condition"],
                "metric_name": "done_gain_answer_to_eos",
                "metric_value": row.get("done_gain_answer_to_eos"),
                "metric_unit": "projection",
            }
        )
    return rows


def make_metrics(model_name: str, signature_rows: list[dict[str, Any]], counterfactual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    final_layer = SPECS[model_name]["final_layer"]
    final_rows = [x for x in signature_rows if int(x.get("layer", -1)) == final_layer]
    for condition, value in mean_by(final_rows, "condition", "done_projection").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase256",
                "created_at": now,
                "metric_id": f"phase256:{model_name}:{condition}:mean_done_projection",
                "scope": "done_signature_projection",
                "model": model_name,
                "condition": condition,
                "metric_name": "mean_done_projection",
                "metric_value": value,
                "rows": sum(1 for x in final_rows if x.get("condition") == condition),
            }
        )
    for condition, value in mean_by([x for x in counterfactual_rows if x.get("done_gain_answer_to_eos") is not None], "condition", "done_gain_answer_to_eos").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase256",
                "created_at": now,
                "metric_id": f"phase256:{model_name}:{condition}:done_gain_answer_to_eos",
                "scope": "done_counterfactual_summary",
                "model": model_name,
                "condition": condition,
                "metric_name": "done_gain_answer_to_eos",
                "metric_value": value,
                "rows": sum(1 for x in counterfactual_rows if x.get("condition") == condition),
            }
        )
    return rows


def make_edges(counterfactual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in counterfactual_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase256",
                "created_at": now,
                "edge_id": f"phase256:done:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['condition']}",
                "source": "node:SemanticDoneSignature",
                "target": "node:ModelCloseExecution",
                "edge_type": "done_signature_counterfactual_localization",
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "evidence_type": "answer_to_eos_done_projection_gain",
                "effect_direction": row["interpretation"],
                "effect_size": row.get("done_gain_answer_to_eos"),
                "confidence": 0.48 if row["interpretation"] == "eos_aligned_done_growth" else 0.30,
                "supporting_phases": ["Phase255", "Phase256"],
                "status": "local_signature_not_general_done_state",
            }
        )
    return rows


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase256_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    components: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        components.extend(read_jsonl(out_dir / f"phase256_{model}_done_vector_component_rows.jsonl"))
        signatures.extend(read_jsonl(out_dir / f"phase256_{model}_done_signature_rows.jsonl"))
        counterfactuals.extend(read_jsonl(out_dir / f"phase256_{model}_counterfactual_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase256_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase256_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase256_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase256_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.82,
        "high_value_trace_selection": 0.68,
        "trace_signature_validation": 0.42,
        "focused_causal_validation": 0.25,
        "regime_field_direction_bank": 0.35,
        "natural_regime_direction_bank": 0.30,
        "regime_level_causal_validation": 0.26,
        "shared_subspace_analysis": 0.20,
        "coupled_regime_field_analysis": 0.23,
        "control_readout_coupling": 0.21,
        "stop_type_validation": 0.20,
        "semantic_done_signature": 0.12,
        "residual_state_signature": 0.51,
        "readout_competition_trace": 0.73,
        "stepwise_rollout_trace": 0.41,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.63,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Done signature counterfactual localization",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "done_vector_component_rows": len(components),
        "done_signature_rows": len(signatures),
        "counterfactual_rows": len(counterfactuals),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "interpretation_counts": dict(Counter(str(x.get("interpretation")) for x in counterfactuals).most_common()),
        "mean_done_projection_by_condition": mean_by([x for x in signatures if int(x.get("layer", -1)) == SPECS.get(str(x.get("model")), {}).get("final_layer", -999)], "condition", "done_projection"),
        "mean_done_gain_answer_to_eos_by_condition": mean_by([x for x in counterfactuals if x.get("done_gain_answer_to_eos") is not None], "condition", "done_gain_answer_to_eos"),
        "progress": progress,
    }
    write_json(out_dir / "phase256_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase256_done_vector_component_rows.jsonl", components)
    write_jsonl(out_dir / "phase256_done_signature_rows.jsonl", signatures)
    write_jsonl(out_dir / "phase256_counterfactual_rows.jsonl", counterfactuals)
    write_jsonl(out_dir / "phase256_observations.jsonl", observations)
    write_jsonl(out_dir / "phase256_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase256_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase256_missing_rows.jsonl", missing)
    write_report(out_dir, payload, counterfactuals)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase256", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any], counterfactual_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase256 Done Signature Counterfactual Localization",
        "",
        f"- status: {payload['status']}",
        f"- done_vector_component_rows: {payload['done_vector_component_rows']}",
        f"- done_signature_rows: {payload['done_signature_rows']}",
        f"- counterfactual_rows: {payload['counterfactual_rows']}",
        f"- interpretation_counts: {json.dumps(payload['interpretation_counts'], ensure_ascii=False)}",
        "",
        "## Counterfactual Summary",
    ]
    for row in counterfactual_rows:
        lines.append(
            f"- {row['model']} {row['condition']}: stop={row['stop_type']}, "
            f"answer_done={row['answer_done_projection']}, pre_eos_done={row['pre_eos_done_projection']}, "
            f"eos_done={row['eos_done_projection']}, late_done={row['late_done_projection']}, "
            f"gain_answer_to_eos={row['done_gain_answer_to_eos']}, interpretation={row['interpretation']}"
        )
    (out_dir / "phase256_done_signature_counterfactual_localization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--perturb-scale", type=float, default=0.5)
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
