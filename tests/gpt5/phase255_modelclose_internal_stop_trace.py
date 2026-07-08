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
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase251_orthogonalized_natural_direction_causal_validation as p251  # noqa: E402
import phase252_shared_subspace_coupled_regime_analysis as p252  # noqa: E402
import phase254_closure_candidate_stop_validation as p254  # noqa: E402


PHASE = 255
SOURCE_PHASE = 254
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_DIR = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark/large_scale_pattern_atlas_benchmark")
PHASE246_DIR = Path("tests/result/phase246_focused_causal_validation/focused_causal_validation")
PHASE250_DIR = Path("tests/result/phase250_natural_regime_direction_extraction/natural_regime_direction_extraction")
PHASE254_DIR = Path("tests/result/phase254_closure_candidate_stop_validation/closure_candidate_stop_validation")
RESULT_ROOT = Path("tests/result/phase255_modelclose_internal_stop_trace")
ROUND_DEFAULT = "modelclose_internal_stop_trace"

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


def parse_weighted_condition(condition: str) -> tuple[float, float]:
    match = re.search(r"c([0-9.]+)_r([0-9.]+)", condition)
    if not match:
        return 1.0, 1.0
    return float(match.group(1)), float(match.group(2))


def condition_direction(condition: str, control_dir: torch.Tensor, readout_dir: torch.Tensor, perturb_norm: float) -> torch.Tensor | None:
    if condition == "no_intervention":
        return None
    if condition == "tokenbank_suppression":
        return readout_dir
    if condition == "natural_raw_suppression":
        return control_dir
    if condition == "combined_suppression":
        return normalize(unit(control_dir) + unit(readout_dir), perturb_norm)
    if condition.startswith("weighted_combined"):
        lc, lr = parse_weighted_condition(condition)
        return normalize(lc * unit(control_dir) + lr * unit(readout_dir), perturb_norm)
    return None


def make_hook_factory(model: Any, layer_idx: int, direction: torch.Tensor | None) -> Any:
    def factory() -> Any:
        if direction is None:
            return no_hook()
        return residual_hook(model, layer_idx, direction, sign=-1.0)

    return factory


def target_answer_step(tokenizer: Any, generated_ids: list[int], target_aliases: list[str]) -> int | None:
    for step in range(1, len(generated_ids) + 1):
        text = tokenizer.decode(generated_ids[:step], skip_special_tokens=True).lower()
        for alias in target_aliases:
            alias_l = str(alias).strip().lower()
            if alias_l and alias_l in text:
                return step
    return None


def generate_once(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    condition: str,
    hook_factory: Any,
    target_aliases: list[str],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
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
    generated_ids = out.sequences[0, input_len:].detach().cpu().tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    step_rows = []
    for step, score in enumerate(out.scores, start=1):
        logits = score[0].detach().float().cpu()
        token_id = int(generated_ids[step - 1]) if step - 1 < len(generated_ids) else int(torch.argmax(logits).item())
        closure = p252.closure_scores(tokenizer, logits)
        readout = p239.readout_metrics(tokenizer, logits, target_aliases)
        eos_logit = logits[int(tokenizer.eos_token_id)].item() if tokenizer.eos_token_id is not None else 0.0
        step_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase255",
                "created_at": utc_now(),
                "condition": condition,
                "step": step,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "eos_logit": round(float(eos_logit), 6),
                **{f"closure_{k}": round(v, 6) for k, v in closure.items()},
                **{f"readout_{k}": v for k, v in readout.items()},
            }
        )
    stop = p254.classify_stop(tokenizer, generated_ids, text, target_aliases, int(max_new_tokens))
    answer_step = target_answer_step(tokenizer, generated_ids, target_aliases)
    summary = {
        "condition": condition,
        "generated_text": text,
        "generated_token_count": len(generated_ids),
        "answer_first_step": answer_step,
        "mean_closure_proxy_margin": round(mean(safe_float(x.get("closure_closure_proxy_margin")) for x in step_rows), 6) if step_rows else 0.0,
        "final_closure_proxy_margin": safe_float(step_rows[-1].get("closure_closure_proxy_margin")) if step_rows else 0.0,
        **stop,
    }
    return summary, step_rows, generated_ids


def capture_prefix_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    generated_ids: list[int],
    condition: str,
    hook_factory: Any,
    observe_layers: list[int],
    control_dir: torch.Tensor,
    readout_dir: torch.Tensor,
    target_aliases: list[str],
) -> list[dict[str, Any]]:
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    base_ids = prompt_ids["input_ids"]
    rows = []
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
        token_id = int(generated_ids[next_step - 1])
        for layer in observe_layers:
            if int(layer) + 1 >= len(out.hidden_states):
                continue
            vec = out.hidden_states[int(layer) + 1][0, last_pos].detach().float().cpu()
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase255",
                    "created_at": utc_now(),
                    "condition": condition,
                    "next_step": next_step,
                    "layer": int(layer),
                    "next_token_id": token_id,
                    "next_token_text": tokenizer.decode([token_id]),
                    "eos_logit": round(float(eos_logit), 6),
                    "control_projection": round(dot(vec, control_dir), 6),
                    "readout_projection": round(dot(vec, readout_dir), 6),
                    **{f"closure_{k}": round(v, 6) for k, v in closure.items()},
                    **{f"readout_{k}": v for k, v in readout.items()},
                }
            )
    return rows


def make_observations(stop_rows: list[dict[str, Any]], step_rows: list[dict[str, Any]], prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    now = utc_now()
    for row in stop_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase255",
                "created_at": now,
                "observation_id": row["stop_trace_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "actual_stop_validation",
                "component": row["condition"],
                "metric_name": "stop_type",
                "metric_value": row["stop_type"],
                "metric_unit": "category",
            }
        )
    for row in step_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase255",
                "created_at": now,
                "observation_id": row["step_trace_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "generation_step_trace",
                "component": row["condition"],
                "metric_name": "closure_proxy_margin",
                "metric_value": row["closure_closure_proxy_margin"],
                "metric_unit": "logit",
            }
        )
    for row in prefix_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase255",
                "created_at": now,
                "observation_id": row["prefix_trace_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row["variant_id"],
                "level": "prefix_hidden_projection",
                "component": f"{row['condition']}:L{row['layer']}",
                "metric_name": "readout_projection",
                "metric_value": row["readout_projection"],
                "metric_unit": "projection",
            }
        )
    return rows


def make_metrics(model_name: str, stop_rows: list[dict[str, Any]], step_rows: list[dict[str, Any]], prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    now = utc_now()
    for condition, value in mean_by(stop_rows, "condition", "final_closure_proxy_margin").items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase255",
                "created_at": now,
                "metric_id": f"phase255:{model_name}:{condition}:final_closure_proxy_margin",
                "scope": "modelclose_internal_stop_trace",
                "model": model_name,
                "condition": condition,
                "metric_name": "final_closure_proxy_margin",
                "metric_value": value,
                "rows": sum(1 for x in stop_rows if x.get("condition") == condition),
            }
        )
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase255",
            "created_at": now,
            "metric_id": f"phase255:{model_name}:eos_stop_count",
            "scope": "modelclose_internal_stop_trace",
            "model": model_name,
            "metric_name": "eos_stop_count",
            "metric_value": sum(1 for x in stop_rows if x.get("stop_type") == "eos_stop"),
            "rows": len(stop_rows),
        }
    )
    if prefix_rows:
        final_layer = max(int(x["layer"]) for x in prefix_rows)
        final_rows = [x for x in prefix_rows if int(x["layer"]) == final_layer]
        for condition, value in mean_by(final_rows, "condition", "readout_projection").items():
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase255",
                    "created_at": now,
                    "metric_id": f"phase255:{model_name}:{condition}:mean_final_layer_readout_projection",
                    "scope": "prefix_hidden_projection",
                    "model": model_name,
                    "condition": condition,
                    "metric_name": "mean_final_layer_readout_projection",
                    "metric_value": value,
                    "rows": sum(1 for x in final_rows if x.get("condition") == condition),
                }
            )
    return rows


def make_edges(stop_rows: list[dict[str, Any]], prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    now = utc_now()
    for row in stop_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase255",
                "created_at": now,
                "edge_id": f"phase255:stop:{row['model']}:{row['case_id']}:{row['variant_id']}:{row['condition']}",
                "source": f"intervention:{row['condition']}",
                "target": "node:ModelCloseExecution",
                "edge_type": "internal_stop_trace",
                "model": row["model"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "evidence_type": "actual_stop_type",
                "effect_direction": row["stop_type"],
                "effect_size": row["final_closure_proxy_margin"],
                "confidence": 0.50 if row["stop_type"] == "eos_stop" else 0.35,
                "supporting_phases": ["Phase253", "Phase254", "Phase255"],
                "status": "single_candidate_trace_not_general_closure",
            }
        )
    return rows


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    modelclose = [x for x in read_jsonl(PHASE254_DIR / "phase254_modelclose_candidate_rows.jsonl") if str(x.get("model")) == args.model]
    modelclose = modelclose[: int(args.max_candidates_per_model)] if int(args.max_candidates_per_model) > 0 else modelclose
    behavior_by_key = load_behavior_by_key()
    raw_vectors = load_raw_vectors(args.model)
    natural_dirs = load_natural_directions(args.model)
    stop_rows: list[dict[str, Any]] = []
    generation_step_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model_obj = None
    tokenizer = None
    if not modelclose:
        payload = {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "title": "ModelClose internal stop trace",
            "status": "complete_no_modelclose_candidates",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "modelclose_candidate_count": 0,
            "stop_trace_rows": 0,
            "generation_step_rows": 0,
            "prefix_projection_rows": 0,
            "missing_rows": 0,
        }
        write_json(out_dir / f"phase255_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase255_{args.model}_stop_trace_rows.jsonl", [])
        write_jsonl(out_dir / f"phase255_{args.model}_generation_step_rows.jsonl", [])
        write_jsonl(out_dir / f"phase255_{args.model}_prefix_projection_rows.jsonl", [])
        write_jsonl(out_dir / f"phase255_{args.model}_observations.jsonl", [])
        write_jsonl(out_dir / f"phase255_{args.model}_metrics.jsonl", make_metrics(args.model, [], [], []))
        write_jsonl(out_dir / f"phase255_{args.model}_graph_edges.jsonl", [])
        write_jsonl(out_dir / f"phase255_{args.model}_missing_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenbank_dirs = p252.build_tokenbank_directions(model_obj, tokenizer)
        final_layer = int(SPECS[args.model]["final_layer"])
        observe_layers = list(SPECS[args.model]["observe_layers"])
        for idx, cand in enumerate(modelclose, start=1):
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
            best_condition = str(cand.get("best_condition") or "weighted_combined_c0.25_r1.0")
            trace_conditions = ["no_intervention", "tokenbank_suppression", "natural_raw_suppression", "combined_suppression", best_condition]
            seen = set()
            trace_conditions = [x for x in trace_conditions if not (x in seen or seen.add(x))]
            for condition in trace_conditions:
                direction = condition_direction(condition, control_dir, readout_dir, perturb_norm)
                hook_factory = make_hook_factory(model_obj, final_layer, direction)
                summary, steps, generated_ids = generate_once(model_obj, tokenizer, device, prompt, condition, hook_factory, aliases, int(args.max_new_tokens))
                stop_row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase255",
                    "created_at": utc_now(),
                    "stop_trace_id": f"phase255:stop:{args.model}:{key[1]}:{key[2]}:{condition}",
                    "model": args.model,
                    "case_id": key[1],
                    "variant_id": key[2],
                    "family_id": behavior.get("family_id"),
                    "mode_id": behavior.get("mode_id"),
                    "condition": condition,
                    "tokenbank_regime": tokenbank_id,
                    "natural_contrast_id": natural_id,
                    "source_modelclose_condition": best_condition,
                    **summary,
                }
                stop_rows.append(stop_row)
                for step_row in steps:
                    generation_step_rows.append(
                        {
                            **step_row,
                            "step_trace_id": f"phase255:step:{args.model}:{key[1]}:{key[2]}:{condition}:{step_row['step']}",
                            "model": args.model,
                            "case_id": key[1],
                            "variant_id": key[2],
                            "family_id": behavior.get("family_id"),
                            "mode_id": behavior.get("mode_id"),
                        }
                    )
                trace_rows = capture_prefix_trace(
                    model_obj,
                    tokenizer,
                    device,
                    prompt,
                    generated_ids,
                    condition,
                    hook_factory,
                    observe_layers,
                    control_dir,
                    readout_dir,
                    aliases,
                )
                for row in trace_rows:
                    prefix_rows.append(
                        {
                            **row,
                            "prefix_trace_id": f"phase255:prefix:{args.model}:{key[1]}:{key[2]}:{condition}:step{row['next_step']}:L{row['layer']}",
                            "model": args.model,
                            "case_id": key[1],
                            "variant_id": key[2],
                            "family_id": behavior.get("family_id"),
                            "mode_id": behavior.get("mode_id"),
                        }
                    )
            log(f"{args.model}: modelclose_candidate={idx}/{len(modelclose)} stop_rows={len(stop_rows)} prefix_rows={len(prefix_rows)}")
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
    observations = make_observations(stop_rows, generation_step_rows, prefix_rows)
    metrics = make_metrics(args.model, stop_rows, generation_step_rows, prefix_rows)
    edges = make_edges(stop_rows, prefix_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "ModelClose internal stop trace",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "modelclose_candidate_count": len(modelclose),
        "stop_trace_rows": len(stop_rows),
        "generation_step_rows": len(generation_step_rows),
        "prefix_projection_rows": len(prefix_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing_rows),
        "stop_type_counts": dict(Counter(str(x.get("stop_type")) for x in stop_rows).most_common()),
        "eos_conditions": [x["condition"] for x in stop_rows if x.get("stop_type") == "eos_stop"],
        "mean_final_closure_by_condition": mean_by(stop_rows, "condition", "final_closure_proxy_margin"),
        "mean_over_generation_by_condition": mean_by(stop_rows, "condition", "over_generation_length"),
    }
    write_json(out_dir / f"phase255_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase255_{args.model}_stop_trace_rows.jsonl", stop_rows)
    write_jsonl(out_dir / f"phase255_{args.model}_generation_step_rows.jsonl", generation_step_rows)
    write_jsonl(out_dir / f"phase255_{args.model}_prefix_projection_rows.jsonl", prefix_rows)
    write_jsonl(out_dir / f"phase255_{args.model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase255_{args.model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase255_{args.model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase255_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase255_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    stop_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        stop_rows.extend(read_jsonl(out_dir / f"phase255_{model}_stop_trace_rows.jsonl"))
        generation_rows.extend(read_jsonl(out_dir / f"phase255_{model}_generation_step_rows.jsonl"))
        prefix_rows.extend(read_jsonl(out_dir / f"phase255_{model}_prefix_projection_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase255_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase255_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase255_{model}_graph_edges.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase255_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.82,
        "high_value_trace_selection": 0.68,
        "trace_signature_validation": 0.40,
        "focused_causal_validation": 0.25,
        "regime_field_direction_bank": 0.35,
        "natural_regime_direction_bank": 0.30,
        "regime_level_causal_validation": 0.26,
        "shared_subspace_analysis": 0.20,
        "coupled_regime_field_analysis": 0.23,
        "control_readout_coupling": 0.21,
        "stop_type_validation": 0.20,
        "residual_state_signature": 0.50,
        "readout_competition_trace": 0.73,
        "stepwise_rollout_trace": 0.40,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.63,
    }
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "ModelClose internal stop trace",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "candidate_models": sorted({x["model"] for x in stop_rows}),
        "stop_trace_rows": len(stop_rows),
        "generation_step_rows": len(generation_rows),
        "prefix_projection_rows": len(prefix_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "missing_rows": len(missing),
        "stop_type_counts": dict(Counter(str(x.get("stop_type")) for x in stop_rows).most_common()),
        "eos_stop_conditions": [f"{x['model']}:{x['condition']}" for x in stop_rows if x.get("stop_type") == "eos_stop"],
        "mean_final_closure_by_condition": mean_by(stop_rows, "condition", "final_closure_proxy_margin"),
        "mean_over_generation_by_condition": mean_by(stop_rows, "condition", "over_generation_length"),
        "progress": progress,
    }
    write_json(out_dir / "phase255_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase255_stop_trace_rows.jsonl", stop_rows)
    write_jsonl(out_dir / "phase255_generation_step_rows.jsonl", generation_rows)
    write_jsonl(out_dir / "phase255_prefix_projection_rows.jsonl", prefix_rows)
    write_jsonl(out_dir / "phase255_observations.jsonl", observations)
    write_jsonl(out_dir / "phase255_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase255_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase255_missing_rows.jsonl", missing)
    write_report(out_dir, payload, stop_rows, prefix_rows)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase255", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any], stop_rows: list[dict[str, Any]], prefix_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase255 ModelClose Internal Stop Trace",
        "",
        f"- status: {payload['status']}",
        f"- stop_trace_rows: {payload['stop_trace_rows']}",
        f"- generation_step_rows: {payload['generation_step_rows']}",
        f"- prefix_projection_rows: {payload['prefix_projection_rows']}",
        f"- stop_type_counts: {json.dumps(payload['stop_type_counts'], ensure_ascii=False)}",
        f"- eos_stop_conditions: {json.dumps(payload['eos_stop_conditions'], ensure_ascii=False)}",
        "",
        "## Stop Trace",
    ]
    for row in stop_rows:
        lines.append(
            f"- {row['model']} {row['condition']}: stop={row['stop_type']}, "
            f"tokens={row['generated_token_count']}, answer_step={row.get('answer_first_step')}, "
            f"eos_pos={row.get('eos_pos')}, final_closure={row.get('final_closure_proxy_margin')}, "
            f"over_generation={row.get('over_generation_length')}"
        )
    final_layer_rows = [x for x in prefix_rows if int(x.get("layer", -1)) == max([int(y.get("layer", -1)) for y in prefix_rows], default=-1)]
    if final_layer_rows:
        lines.extend(["", "## Final Layer Prefix Projection Averages"])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in final_layer_rows:
            grouped[str(row["condition"])].append(row)
        for condition, rows in grouped.items():
            lines.append(
                f"- {condition}: readout_projection_mean={round(mean(safe_float(x['readout_projection']) for x in rows), 6)}, "
                f"control_projection_mean={round(mean(safe_float(x['control_projection']) for x in rows), 6)}, "
                f"closure_proxy_mean={round(mean(safe_float(x['closure_closure_proxy_margin']) for x in rows), 6)}"
            )
    (out_dir / "phase255_modelclose_internal_stop_trace_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-candidates-per-model", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
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
