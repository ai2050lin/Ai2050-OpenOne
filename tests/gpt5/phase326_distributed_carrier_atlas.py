#!/usr/bin/env python3
"""Phase326 distributed carrier-set atlas and registered heldout audit.

This probe deliberately separates four evidence levels:
1. natural component observation;
2. frozen carrier-set selection on discovery cases;
3. set-level necessity ablation on calibration/heldout cases;
4. cross-model replication without claiming single-unit causality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase326_distributed_carrier_case_bank as case_bank  # noqa: E402
from hf_probe_env import encode, get_layers, load_probe_model, release_loaded  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402


PHASE = "Phase326"
SCHEMA_VERSION = "5.0.0"
ROUND_DEFAULT = "distributed_carrier_atlas"
MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT = ROOT / "tests/gpt5/result/phase326_distributed_carrier_atlas"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
FRONTEND_V2 = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
MLP_GROUPS = 32
SET_PER_ROLE = 2
ROLES = ("source", "query", "last")
CONDITIONS = (
    "baseline",
    "single_attention_zero",
    "attention_set_zero",
    "single_mlp_zero",
    "mlp_set_zero",
    "joint_set_zero",
    "matched_random_joint_zero",
    "wrong_layer_joint_zero",
)
CONFIRMATION_CONDITIONS = (
    "baseline",
    "single_attention_zero",
    "single_mlp_zero",
    "joint_set_zero",
    "matched_random_joint_zero",
    "wrong_layer_joint_zero",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_down_proj(layer: Any) -> Any | None:
    mlp = getattr(layer, "mlp", None)
    return getattr(mlp, "down_proj", None) if mlp is not None else None


def group_ranges(width: int) -> list[tuple[int, int]]:
    return [(round(i * width / MLP_GROUPS), round((i + 1) * width / MLP_GROUPS)) for i in range(MLP_GROUPS)]


def candidate_layers(n_layers: int) -> list[int]:
    fractions = (0.20, 0.40, 0.60, 0.80, 1.0)
    return sorted({max(0, min(n_layers - 1, round((n_layers - 1) * value))) for value in fractions})


def answer_token_id(tokenizer: Any, text: str) -> int:
    for candidate in (" " + text, text):
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if ids:
            return int(ids[0])
    raise ValueError(f"Cannot tokenize answer {text!r}")


def locate_fragment_span(tokenizer: Any, prompt: str, fragment: str, seq_len: int) -> tuple[int, int]:
    char_start = prompt.index(fragment)
    char_end = char_start + len(fragment)
    prefix_ids = tokenizer(prompt[:char_start], add_special_tokens=True)["input_ids"]
    end_ids = tokenizer(prompt[:char_end], add_special_tokens=True)["input_ids"]
    # Slow BPE/SentencePiece tokenizers may merge the leading space with the
    # first fragment token.  Prefix length therefore points one token past the
    # actual contextual start.
    start = min(seq_len - 1, max(0, len(prefix_ids) - 1))
    end = min(seq_len - 1, max(start, len(end_ids) - 1))
    return start, end


def role_spans(tokenizer: Any, prompt: str, case: dict[str, Any], seq_len: int) -> dict[str, tuple[int, int]]:
    source_fragments = case["source_fragments"]
    first = prompt.index(source_fragments[0])
    last_fragment = source_fragments[-1]
    last = prompt.index(last_fragment, first) + len(last_fragment)
    source_group = prompt[first:last]
    source = locate_fragment_span(tokenizer, prompt, source_group, seq_len)
    query = locate_fragment_span(tokenizer, prompt, case["query_fragment"], seq_len)
    return {"source": source, "query": query, "last": (seq_len - 1, seq_len - 1)}


def role_positions(span: tuple[int, int]) -> list[int]:
    return list(range(span[0], span[1] + 1))


def pool_role(tensor: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
    start, end = span
    return tensor[0, start : end + 1].detach().float().mean(dim=0).cpu()


def get_readout_weight(model_obj: Any) -> torch.Tensor:
    output = model_obj.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise TypeError("Model has no accessible output embedding weight")
    return output.weight


def readout_direction(model_obj: Any, target_id: int, distractor_ids: list[int]) -> torch.Tensor:
    weight = get_readout_weight(model_obj)
    target = weight[target_id]
    distractor = weight[torch.tensor(distractor_ids, device=weight.device)].mean(dim=0)
    direction = target - distractor
    return direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)


def output_metrics(logits: torch.Tensor, target_id: int, distractor_ids: list[int]) -> dict[str, Any]:
    target_logit = float(logits[target_id].item())
    distractor_logits = [float(logits[index].item()) for index in distractor_ids]
    best_distractor = max(distractor_logits)
    all_ids = [target_id, *distractor_ids]
    winner_id = max(all_ids, key=lambda index: float(logits[index].item()))
    return {
        "target_logit": round(target_logit, 6),
        "best_distractor_logit": round(best_distractor, 6),
        "target_margin": round(target_logit - best_distractor, 6),
        "candidate_winner_is_target": winner_id == target_id,
        "candidate_winner_token_id": int(winner_id),
    }


def js_divergence(base_logits: torch.Tensor, changed_logits: torch.Tensor) -> float:
    p = torch.softmax(base_logits.float(), dim=-1).clamp_min(1e-12)
    q = torch.softmax(changed_logits.float(), dim=-1).clamp_min(1e-12)
    m = 0.5 * (p + q)
    value = 0.5 * ((p * (p.log() - m.log())).sum() + (q * (q.log() - m.log())).sum())
    return round(float(value.item()), 8)


@torch.inference_mode()
def capture_natural(
    loaded: Any,
    case: dict[str, Any],
    layers: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model_obj = loaded.model
    encoded = encode(loaded, case["prompt"])
    seq_len = int(encoded["attention_mask"].sum().item())
    spans = role_spans(loaded.tokenizer, case["prompt"], case, seq_len)
    attention: dict[tuple[int, str], torch.Tensor] = {}
    products: dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    for layer_idx in layers:
        layer = get_layers(model_obj)[layer_idx]
        o_proj, _heads, _head_dim = head_meta(model_obj, layer_idx)

        def attn_pre(_module: Any, inputs: tuple[Any, ...], layer_idx: int = layer_idx) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                for role, span in spans.items():
                    attention[(layer_idx, role)] = pool_role(inputs[0], span)

        handles.append(o_proj.register_forward_pre_hook(attn_pre))
        down_proj = get_down_proj(layer)
        if down_proj is not None:

            def mlp_pre(_module: Any, inputs: tuple[Any, ...], layer_idx: int = layer_idx) -> None:
                if inputs and torch.is_tensor(inputs[0]):
                    for role, span in spans.items():
                        products[(layer_idx, role)] = pool_role(inputs[0], span)

            handles.append(down_proj.register_forward_pre_hook(mlp_pre))
    try:
        with torch.inference_mode():
            output = model_obj(**encoded, use_cache=False, return_dict=True)
        logits = output.logits[0, seq_len - 1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    target_id = answer_token_id(loaded.tokenizer, case["target"])
    distractor_ids = [answer_token_id(loaded.tokenizer, value) for value in case["distractors"]]
    direction = readout_direction(model_obj, target_id, distractor_ids)
    observations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for layer_idx in layers:
        layer = get_layers(model_obj)[layer_idx]
        o_proj, n_heads, head_dim = head_meta(model_obj, layer_idx)
        attn_input_direction = torch.mv(o_proj.weight.detach().T, direction.to(o_proj.weight.device, dtype=o_proj.weight.dtype)).float().cpu()
        down_proj = get_down_proj(layer)
        mlp_input_direction = None
        if down_proj is not None:
            mlp_input_direction = torch.mv(
                down_proj.weight.detach().T,
                direction.to(down_proj.weight.device, dtype=down_proj.weight.dtype),
            ).float().cpu()
        for role in ROLES:
            attn_vector = attention[(layer_idx, role)]
            attn_scores = (attn_vector.view(n_heads, head_dim) * attn_input_direction.view(n_heads, head_dim)).sum(dim=1)
            observed_attn = list(range(n_heads)) if case["split"] == "discovery" else torch.topk(attn_scores, k=min(3, n_heads)).indices.tolist()
            for head in observed_attn:
                start, end = head * head_dim, (head + 1) * head_dim
                observations.append(candidate_row(case, loaded.key, "attention_head_input", layer_idx, role, head, start, end, attn_vector[start:end], float(attn_scores[head].item())))
            summaries.append(component_summary_row(case, loaded.key, "attention_head_input", layer_idx, role, attn_vector, attn_scores))
            if mlp_input_direction is None or (layer_idx, role) not in products:
                continue
            product = products[(layer_idx, role)]
            ranges = group_ranges(product.numel())
            mlp_scores = torch.tensor([
                float((product[start:end] * mlp_input_direction[start:end]).sum().item()) for start, end in ranges
            ])
            observed_mlp = list(range(len(ranges))) if case["split"] == "discovery" else torch.topk(mlp_scores, k=min(3, len(ranges))).indices.tolist()
            for group in observed_mlp:
                start, end = ranges[group]
                observations.append(candidate_row(case, loaded.key, "mlp_product_group", layer_idx, role, group, start, end, product[start:end], float(mlp_scores[group].item())))
            summaries.append(component_summary_row(case, loaded.key, "mlp_product_group", layer_idx, role, product, mlp_scores))

    base = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": loaded.key,
        "case_id": case["case_id"],
        "base_case_id": case["base_case_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "split": case["split"],
        "template_id": case["template_id"],
        "target": case["target"],
        "target_token_id": target_id,
        "distractor_token_ids": distractor_ids,
        "sequence_length": seq_len,
        "role_spans": {key: list(value) for key, value in spans.items()},
        "source_token_count": spans["source"][1] - spans["source"][0] + 1,
        "target_absent_from_prompt": case["target"].lower() not in case["prompt"].lower(),
        **output_metrics(logits, target_id, distractor_ids),
    }
    return base, observations, summaries


def candidate_row(
    case: dict[str, Any], model: str, component: str, layer: int, role: str,
    index: int, start: int, end: int, vector: torch.Tensor, score: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "case_id": case["case_id"],
        "base_case_id": case["base_case_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "split": case["split"],
        "template_id": case["template_id"],
        "component_type": component,
        "component_layer": layer,
        "position_role": role,
        "component_index": index,
        "component_start": start,
        "component_end": end,
        "activation_norm": round(float(torch.linalg.vector_norm(vector).item()), 6),
        "approx_target_readout_contribution": round(score, 6),
        "evidence_level": "L3_observational_candidate",
        "single_unit_causal": False,
    }


def component_summary_row(
    case: dict[str, Any], model: str, component: str, layer: int, role: str,
    vector: torch.Tensor, scores: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "split": case["split"],
        "template_id": case["template_id"],
        "component_type": component,
        "component_layer": layer,
        "position_role": role,
        "activation_norm": round(float(torch.linalg.vector_norm(vector).item()), 6),
        "max_target_readout_contribution": round(float(scores.max().item()), 6),
        "positive_component_fraction": round(float((scores > 0).float().mean().item()), 6),
    }


def discovery_stats(observations: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row["split"] != "discovery":
            continue
        key = (
            row["family_id"], row["mechanism_id"], row["component_type"], row["component_layer"],
            row["position_role"], row["component_index"], row["component_start"], row["component_end"],
        )
        grouped[key].append(row)
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, rows in grouped.items():
        scores = [float(row["approx_target_readout_contribution"]) for row in rows]
        templates = {row["template_id"] for row in rows}
        base_cases = {row["base_case_id"] for row in rows}
        result[key] = {
            "mean_score": safe_mean(scores),
            "positive_rate": round(sum(value > 0 for value in scores) / len(scores), 6),
            "observation_count": len(scores),
            "independent_case_count": len(base_cases),
            "template_count": len(templates),
            "rank_score": round(max(0.0, mean(scores)) * (sum(value > 0 for value in scores) / len(scores)), 6),
        }
    return result


def select_carrier_sets(model: str, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats = discovery_stats(observations)
    mechanisms = sorted({(row["family_id"], row["mechanism_id"]) for row in observations})
    selections: list[dict[str, Any]] = []
    for family, mechanism in mechanisms:
        for component in ("attention_head_input", "mlp_product_group"):
            selected: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            for role in ROLES:
                candidates = [
                    (key, values) for key, values in stats.items()
                    if key[0] == family and key[1] == mechanism and key[2] == component and key[4] == role
                ]
                candidates.sort(key=lambda item: (item[1]["rank_score"], item[1]["independent_case_count"]), reverse=True)
                selected.extend(candidates[:SET_PER_ROLE])
            selected.sort(key=lambda item: item[1]["rank_score"], reverse=True)
            for rank, (key, values) in enumerate(selected):
                selections.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model,
                    "family_id": family,
                    "mechanism_id": mechanism,
                    "component_type": component,
                    "component_layer": key[3],
                    "position_role": key[4],
                    "component_index": key[5],
                    "component_start": key[6],
                    "component_end": key[7],
                    "set_rank": rank,
                    "is_single_baseline": rank == 0,
                    **values,
                    "selection_split": "discovery_only",
                    "evidence_level": "L3_frozen_carrier_candidate",
                    "single_unit_causal": False,
                })
    return selections


def selections_for(rows: list[dict[str, Any]], case: dict[str, Any], component: str | None = None) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["family_id"] == case["family_id"]
        and row["mechanism_id"] == case["mechanism_id"]
        and (component is None or row["component_type"] == component)
    ]


def randomize_specs(model_obj: Any, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for spec in specs:
        row = dict(spec)
        layer = int(spec["component_layer"])
        if spec["component_type"] == "attention_head_input":
            _proj, heads, head_dim = head_meta(model_obj, layer)
            index = (int(spec["component_index"]) + heads // 2 + 1) % heads
            row.update(component_index=index, component_start=index * head_dim, component_end=(index + 1) * head_dim)
        else:
            width = get_down_proj(get_layers(model_obj)[layer]).in_features
            ranges = group_ranges(width)
            index = (int(spec["component_index"]) + 17) % len(ranges)
            row.update(component_index=index, component_start=ranges[index][0], component_end=ranges[index][1])
        row["control_transform"] = "deterministic_matched_index_shift"
        result.append(row)
    return result


def wrong_layer_specs(model_obj: Any, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n_layers = len(get_layers(model_obj))
    result = []
    for spec in specs:
        row = dict(spec)
        old = int(spec["component_layer"])
        layer = (old + max(1, n_layers // 5)) % n_layers
        row["component_layer"] = layer
        if spec["component_type"] == "attention_head_input":
            _proj, heads, head_dim = head_meta(model_obj, layer)
            index = int(spec["component_index"]) % heads
            row.update(component_index=index, component_start=index * head_dim, component_end=(index + 1) * head_dim)
        else:
            width = get_down_proj(get_layers(model_obj)[layer]).in_features
            ranges = group_ranges(width)
            index = int(spec["component_index"]) % len(ranges)
            row.update(component_index=index, component_start=ranges[index][0], component_end=ranges[index][1])
        row["control_transform"] = "wrong_layer_shift"
        result.append(row)
    return result


def condition_specs(model_obj: Any, selections: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    attention = selections_for(selections, case, "attention_head_input")
    mlp = selections_for(selections, case, "mlp_product_group")
    joint = [*attention, *mlp]
    return {
        "baseline": [],
        "single_attention_zero": attention[:1],
        "attention_set_zero": attention,
        "single_mlp_zero": mlp[:1],
        "mlp_set_zero": mlp,
        "joint_set_zero": joint,
        "matched_random_joint_zero": randomize_specs(model_obj, joint),
        "wrong_layer_joint_zero": wrong_layer_specs(model_obj, joint),
    }


def run_intervention(
    loaded: Any,
    case: dict[str, Any],
    specs: list[dict[str, Any]],
    capture_specs: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    model_obj = loaded.model
    encoded = encode(loaded, case["prompt"])
    seq_len = int(encoded["attention_mask"].sum().item())
    spans = role_spans(loaded.tokenizer, case["prompt"], case, seq_len)
    by_module: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        by_module[(str(spec["component_type"]), int(spec["component_layer"]))].append(spec)
    capture_by_module: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for spec in capture_specs or []:
        capture_by_module[(str(spec["component_type"]), int(spec["component_layer"]))].append(spec)
    energies: dict[str, list[float]] = defaultdict(list)
    handles = []
    keys = set(by_module) | set(capture_by_module)
    for component, layer_idx in keys:
        layer = get_layers(model_obj)[layer_idx]
        module = head_meta(model_obj, layer_idx)[0] if component == "attention_head_input" else get_down_proj(layer)
        mutate = by_module.get((component, layer_idx), [])
        observe = capture_by_module.get((component, layer_idx), [])

        def pre_hook(
            _module: Any, inputs: tuple[Any, ...], mutate: list[dict[str, Any]] = mutate,
            observe: list[dict[str, Any]] = observe, component: str = component,
        ) -> tuple[Any, ...] | None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            tensor = inputs[0]
            for spec in observe:
                positions = role_positions(spans[str(spec["position_role"])])
                start, end = int(spec["component_start"]), int(spec["component_end"])
                value = tensor[0, positions, start:end].detach().float()
                normalized = float(torch.linalg.vector_norm(value).item() / math.sqrt(max(1, value.numel())))
                energies[component].append(normalized)
            if not mutate:
                return None
            changed = tensor.clone()
            for spec in mutate:
                positions = role_positions(spans[str(spec["position_role"])])
                changed[0, positions, int(spec["component_start"]):int(spec["component_end"])] = 0
            return (changed, *inputs[1:])

        handles.append(module.register_forward_pre_hook(pre_hook))
    try:
        with torch.inference_mode():
            output = model_obj(**encoded, use_cache=False, return_dict=True)
        logits = output.logits[0, seq_len - 1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    energy_summary = {
        "attention_energy": safe_mean(energies.get("attention_head_input", [])),
        "mlp_energy": safe_mean(energies.get("mlp_product_group", [])),
        "joint_energy": safe_mean([*energies.get("attention_head_input", []), *energies.get("mlp_product_group", [])]),
    }
    return logits, energy_summary


def intervention_case_filter(case: dict[str, Any]) -> bool:
    return (case["split"] == "calibration" and case["template_id"] == "template_a") or (
        case["split"] == "heldout" and case["template_id"] == "template_c"
    )


def run_model(model_key: str, round_name: str, max_cases: int = 0) -> dict[str, Any]:
    all_cases = case_bank.build_cases()
    if max_cases:
        all_cases = all_cases[:max_cases]
    out = OUT / round_name
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        model_obj = loaded.model
        layers = candidate_layers(len(get_layers(model_obj)))
        natural_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        for number, case in enumerate(all_cases, start=1):
            base, observations, summaries = capture_natural(loaded, case, layers)
            natural_rows.append(base)
            candidate_rows.extend(observations)
            component_rows.extend(summaries)
            if number % 24 == 0 or number == len(all_cases):
                print(f"[{model_key}] natural {number}/{len(all_cases)}", flush=True)
        selections = select_carrier_sets(model_key, candidate_rows)
        write_jsonl(out / f"phase326_{model_key}_natural_rows.jsonl", natural_rows)
        write_jsonl(out / f"phase326_{model_key}_component_summary_rows.jsonl", component_rows)
        write_jsonl(out / f"phase326_{model_key}_candidate_observations.jsonl", candidate_rows)
        write_jsonl(out / f"phase326_{model_key}_carrier_sets.jsonl", selections)

        intervention_cases = [case for case in all_cases if intervention_case_filter(case)]
        intervention_rows: list[dict[str, Any]] = []
        for number, case in enumerate(intervention_cases, start=1):
            specs_by_condition = condition_specs(model_obj, selections, case)
            joint_specs = specs_by_condition["joint_set_zero"]
            base_logits, energy = run_intervention(loaded, case, [], capture_specs=joint_specs)
            target_id = answer_token_id(loaded.tokenizer, case["target"])
            distractor_ids = [answer_token_id(loaded.tokenizer, value) for value in case["distractors"]]
            base_metrics = output_metrics(base_logits, target_id, distractor_ids)
            for condition in CONDITIONS:
                if condition == "baseline":
                    logits, condition_energy = base_logits, energy
                else:
                    logits, condition_energy = run_intervention(loaded, case, specs_by_condition[condition])
                metrics = output_metrics(logits, target_id, distractor_ids)
                intervention_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model_key,
                    "case_id": case["case_id"],
                    "base_case_id": case["base_case_id"],
                    "family_id": case["family_id"],
                    "mechanism_id": case["mechanism_id"],
                    "split": case["split"],
                    "template_id": case["template_id"],
                    "condition": condition,
                    "selected_component_count": len(specs_by_condition[condition]),
                    "baseline_candidate_winner_is_target": base_metrics["candidate_winner_is_target"],
                    "baseline_target_margin": base_metrics["target_margin"],
                    **metrics,
                    "target_margin_drop": round(float(base_metrics["target_margin"] - metrics["target_margin"]), 6),
                    "js_divergence_from_baseline": 0.0 if condition == "baseline" else js_divergence(base_logits, logits),
                    **condition_energy,
                    "causal_scope": "distributed_component_set" if condition != "baseline" else "none",
                    "single_unit_causal": False,
                })
            if number % 8 == 0 or number == len(intervention_cases):
                print(f"[{model_key}] intervention {number}/{len(intervention_cases)}", flush=True)

        gate_rows = natural_gate_controls(loaded, all_cases, selections, intervention_rows)
        registered = registered_heldout_rows(model_key, intervention_rows, gate_rows)
        summary = model_summary(model_key, loaded, layers, natural_rows, selections, intervention_rows, gate_rows, registered)
        write_jsonl(out / f"phase326_{model_key}_intervention_rows.jsonl", intervention_rows)
        write_jsonl(out / f"phase326_{model_key}_natural_gate_rows.jsonl", gate_rows)
        write_jsonl(out / f"phase326_{model_key}_registered_heldout.jsonl", registered)
        write_json(out / f"phase326_{model_key}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_confirmation_model(model_key: str, round_name: str) -> dict[str, Any]:
    out = OUT / round_name
    selections = read_jsonl(out / f"phase326_{model_key}_carrier_sets.jsonl")
    if not selections:
        raise FileNotFoundError(f"Missing frozen discovery selections for {model_key}")
    cases = case_bank.build_confirmation_cases()
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        rows: list[dict[str, Any]] = []
        for number, case in enumerate(cases, start=1):
            specs_by_condition = condition_specs(loaded.model, selections, case)
            base_logits, _energy = run_intervention(loaded, case, [])
            target_id = answer_token_id(loaded.tokenizer, case["target"])
            distractor_ids = [answer_token_id(loaded.tokenizer, value) for value in case["distractors"]]
            base_metrics = output_metrics(base_logits, target_id, distractor_ids)
            for condition in CONFIRMATION_CONDITIONS:
                logits = base_logits if condition == "baseline" else run_intervention(
                    loaded, case, specs_by_condition[condition]
                )[0]
                metrics = output_metrics(logits, target_id, distractor_ids)
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model_key,
                    "case_id": case["case_id"],
                    "base_case_id": case["base_case_id"],
                    "family_id": case["family_id"],
                    "mechanism_id": case["mechanism_id"],
                    "split": "expanded_confirmation",
                    "template_id": case["template_id"],
                    "condition": condition,
                    "baseline_candidate_winner_is_target": base_metrics["candidate_winner_is_target"],
                    "baseline_target_margin": base_metrics["target_margin"],
                    **metrics,
                    "target_margin_drop": round(float(base_metrics["target_margin"] - metrics["target_margin"]), 6),
                    "js_divergence_from_baseline": 0.0 if condition == "baseline" else js_divergence(base_logits, logits),
                    "selection_frozen_from": "phase326_discovery_only",
                    "confirmation_updates_selection": False,
                    "causal_scope": "distributed_component_set" if condition != "baseline" else "none",
                    "single_unit_causal": False,
                })
            if number % 16 == 0 or number == len(cases):
                print(f"[{model_key}] expanded confirmation {number}/{len(cases)}", flush=True)
        audits = expanded_confirmation_audits(model_key, rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model_key,
            "independent_object_count": 64,
            "prompt_case_count": len(cases),
            "intervention_row_count": len(rows),
            "knowledge_answer_leak_count": sum(
                case["target"].lower() in case["prompt"].lower() for case in cases
            ),
            "mechanism_pass_count": sum(row["expanded_confirmation_pass"] for row in audits),
            "mechanism_audits": audits,
            "single_unit_causal_count": 0,
            "l5_promoted_count": 0,
        }
        write_jsonl(out / f"phase326_{model_key}_expanded_confirmation_rows.jsonl", rows)
        write_jsonl(out / f"phase326_{model_key}_expanded_confirmation_audits.jsonl", audits)
        write_json(out / f"phase326_{model_key}_expanded_confirmation_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def expanded_confirmation_audits(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    mechanisms = sorted({row["mechanism_id"] for row in rows})
    for mechanism in mechanisms:
        mechanism_rows = [row for row in rows if row["mechanism_id"] == mechanism]
        baseline = [row for row in mechanism_rows if row["condition"] == "baseline"]
        eligible_ids = {row["case_id"] for row in baseline if row["candidate_winner_is_target"]}
        eligible_objects = {row["base_case_id"] for row in baseline if row["candidate_winner_is_target"]}

        def values(condition: str, field: str, template: str | None = None) -> list[float]:
            return [
                float(row[field]) for row in mechanism_rows
                if row["condition"] == condition and row["case_id"] in eligible_ids
                and (template is None or row["template_id"] == template)
            ]

        joint = values("joint_set_zero", "target_margin_drop")
        random = values("matched_random_joint_zero", "target_margin_drop")
        wrong = values("wrong_layer_joint_zero", "target_margin_drop")
        single_attn = values("single_attention_zero", "target_margin_drop")
        single_mlp = values("single_mlp_zero", "target_margin_drop")
        joint_mean = safe_mean(joint)
        random_mean = safe_mean(random)
        wrong_mean = safe_mean(wrong)
        best_single = max(safe_mean(single_attn), safe_mean(single_mlp))
        specificity = round(joint_mean - max(random_mean, wrong_mean), 6)
        gain = round(joint_mean - best_single, 6)
        consistency = round(sum(value > 0 for value in joint) / len(joint), 6) if joint else 0.0
        template_audits = []
        for template in case_bank.CONFIRMATION_TEMPLATES:
            t_joint = values("joint_set_zero", "target_margin_drop", template)
            t_random = values("matched_random_joint_zero", "target_margin_drop", template)
            t_wrong = values("wrong_layer_joint_zero", "target_margin_drop", template)
            t_specificity = round(safe_mean(t_joint) - max(safe_mean(t_random), safe_mean(t_wrong)), 6)
            t_consistency = round(sum(value > 0 for value in t_joint) / len(t_joint), 6) if t_joint else 0.0
            template_audits.append({
                "template_id": template,
                "eligible_count": len(t_joint),
                "joint_margin_drop": safe_mean(t_joint),
                "matched_control_specificity": t_specificity,
                "positive_effect_consistency": t_consistency,
                "template_pass": len(t_joint) >= 10 and t_specificity > 0.0 and t_consistency >= 0.65,
            })
        passed = (
            len(eligible_objects) >= 12
            and joint_mean > 0.03
            and specificity > 0.02
            and gain > 0.01
            and consistency >= 0.70
            and all(row["template_pass"] for row in template_audits)
        )
        result.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "split": "expanded_confirmation",
            "registered_prompt_count": len(baseline),
            "registered_independent_object_count": len({row["base_case_id"] for row in baseline}),
            "baseline_eligible_prompt_count": len(eligible_ids),
            "baseline_eligible_independent_object_count": len(eligible_objects),
            "baseline_accuracy": round(len(eligible_ids) / max(1, len(baseline)), 6),
            "joint_margin_drop": joint_mean,
            "matched_random_margin_drop": random_mean,
            "wrong_layer_margin_drop": wrong_mean,
            "best_single_margin_drop": best_single,
            "matched_control_specificity": specificity,
            "distributed_gain_over_single": gain,
            "positive_effect_consistency": consistency,
            "template_audits": template_audits,
            "expanded_confirmation_pass": passed,
            "selection_frozen": True,
            "single_unit_causal": False,
            "l5_promoted": False,
        })
    return result


def collect_confirmation(round_name: str) -> dict[str, Any]:
    out = OUT / round_name
    summaries = [read_json(out / f"phase326_{model}_expanded_confirmation_summary.json") for model in MODELS]
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for model in MODELS:
        rows.extend(read_jsonl(out / f"phase326_{model}_expanded_confirmation_rows.jsonl"))
        audits.extend(read_jsonl(out / f"phase326_{model}_expanded_confirmation_audits.jsonl"))
    initial_heldout = {
        (row["model"], row["family_id"], row["mechanism_id"]): bool(row["set_necessity_pass"])
        for row in read_jsonl(out / "phase326_registered_heldout.jsonl")
        if row["split"] == "heldout"
    }
    for row in audits:
        row["initial_heldout_pass"] = initial_heldout.get(
            (row["model"], row["family_id"], row["mechanism_id"]), False
        )
        row["strict_confirmation_pass"] = bool(
            row["expanded_confirmation_pass"] and row["initial_heldout_pass"]
        )
    mechanism_results = []
    for mechanism in sorted({row["mechanism_id"] for row in audits}):
        current = [row for row in audits if row["mechanism_id"] == mechanism]
        expanded_pass_models = [row["model"] for row in current if row["expanded_confirmation_pass"]]
        pass_models = [row["model"] for row in current if row["strict_confirmation_pass"]]
        mechanism_results.append({
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "expanded_only_pass_models": expanded_pass_models,
            "expanded_confirmation_pass_models": pass_models,
            "cross_model_expanded_confirmation_replicated": len(pass_models) >= 2,
            "single_unit_causal": False,
            "l5_promoted": False,
        })
    cross = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "models": MODELS,
        "independent_object_model_cases": 64 * len(MODELS),
        "prompt_model_cases": 128 * len(MODELS),
        "intervention_row_count": len(rows),
        "frozen_selection": True,
        "mechanism_results": mechanism_results,
        "cross_model_replicated_mechanism_count": sum(
            row["cross_model_expanded_confirmation_replicated"] for row in mechanism_results
        ),
        "single_unit_causal_count": 0,
        "l5_promoted_count": 0,
        "model_summaries": summaries,
    }
    write_jsonl(out / "phase326_expanded_confirmation_rows.jsonl", rows)
    write_jsonl(out / "phase326_expanded_confirmation_audits.jsonl", audits)
    write_json(out / "phase326_expanded_confirmation_cross_model.json", cross)
    main_cross_path = out / "phase326_cross_model_summary.json"
    main_cross = read_json(main_cross_path)
    main_cross["expanded_confirmation"] = cross
    confirm_by_mechanism = {row["mechanism_id"]: row for row in mechanism_results}
    for row in main_cross["mechanism_results"]:
        if row["family_id"] == "content_knowledge":
            row.update(confirm_by_mechanism.get(row["mechanism_id"], {}))
    write_json(main_cross_path, main_cross)
    update_nodes_with_confirmation(out, audits)
    publish(out, main_cross)
    print(json.dumps(cross, ensure_ascii=False, indent=2))
    return cross


def update_nodes_with_confirmation(out: Path, audits: list[dict[str, Any]]) -> None:
    audit_map = {(row["model"], row["family_id"], row["mechanism_id"]): row for row in audits}
    path = out / "phase326_atlas_nodes.jsonl"
    nodes = read_jsonl(path)
    for node in nodes:
        audit = audit_map.get((node["model"], node["family_id"], node["mechanism_id"]))
        node["expanded_confirmation_pass"] = bool(audit and audit["strict_confirmation_pass"])
        if node["expanded_confirmation_pass"]:
            node["evidence_level"] = "L4_expanded_set_member"
    write_jsonl(path, nodes)


def natural_gate_controls(
    loaded: Any,
    all_cases: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    heldout = [case for case in all_cases if case["split"] == "heldout" and case["template_id"] == "template_c"]
    by_family: dict[str, list[str]] = defaultdict(list)
    for case in heldout:
        if case["mechanism_id"] not in by_family[case["family_id"]]:
            by_family[case["family_id"]].append(case["mechanism_id"])
    baseline_energy = {
        row["case_id"]: float(row["joint_energy"])
        for row in intervention_rows if row["condition"] == "baseline" and row["split"] == "heldout"
    }
    result = []
    for case in heldout:
        mechanisms = by_family[case["family_id"]]
        index = mechanisms.index(case["mechanism_id"])
        control_mechanism = mechanisms[(index + 1) % len(mechanisms)]
        control = next(
            row for row in heldout
            if row["family_id"] == case["family_id"]
            and row["mechanism_id"] == control_mechanism
            and row["independent_item_index"] == case["independent_item_index"]
        )
        own_specs = selections_for(selections, case)
        _logits, control_energy = run_intervention(loaded, control, [], capture_specs=own_specs)
        positive = baseline_energy.get(case["case_id"], 0.0)
        control_value = float(control_energy["joint_energy"])
        result.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": loaded.key,
            "family_id": case["family_id"],
            "mechanism_id": case["mechanism_id"],
            "case_id": case["case_id"],
            "control_case_id": control["case_id"],
            "control_mechanism_id": control_mechanism,
            "positive_joint_energy": round(positive, 6),
            "control_joint_energy": round(control_value, 6),
            "energy_ratio": round(positive / max(control_value, 1e-8), 6),
            "positive_exceeds_control": positive > control_value,
            "evidence_level": "L3_observational_gate_contrast",
            "natural_gate_causal": False,
        })
    return result


def registered_heldout_rows(
    model: str,
    intervention_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mechanisms = sorted({(row["family_id"], row["mechanism_id"]) for row in intervention_rows})
    gate_by_mechanism: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        gate_by_mechanism[(row["family_id"], row["mechanism_id"])].append(row)
    result = []
    for family, mechanism in mechanisms:
        for split in ("calibration", "heldout"):
            rows = [row for row in intervention_rows if row["family_id"] == family and row["mechanism_id"] == mechanism and row["split"] == split]
            by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
            eligible = [row for row in by_condition["baseline"] if row["candidate_winner_is_target"]]
            eligible_ids = {row["case_id"] for row in eligible}

            def values(condition: str, field: str) -> list[float]:
                return [float(row[field]) for row in by_condition[condition] if row["case_id"] in eligible_ids]

            joint = values("joint_set_zero", "target_margin_drop")
            random = values("matched_random_joint_zero", "target_margin_drop")
            wrong = values("wrong_layer_joint_zero", "target_margin_drop")
            single_attn = values("single_attention_zero", "target_margin_drop")
            single_mlp = values("single_mlp_zero", "target_margin_drop")
            joint_mean = safe_mean(joint)
            random_mean = safe_mean(random)
            wrong_mean = safe_mean(wrong)
            best_single = max(safe_mean(single_attn), safe_mean(single_mlp))
            specificity = round(joint_mean - max(random_mean, wrong_mean), 6)
            distributed_gain = round(joint_mean - best_single, 6)
            consistency = round(sum(value > 0 for value in joint) / len(joint), 6) if joint else 0.0
            js_joint = safe_mean(values("joint_set_zero", "js_divergence_from_baseline"))
            js_random = safe_mean(values("matched_random_joint_zero", "js_divergence_from_baseline"))
            enough = len(eligible_ids) >= 3
            pass_set_necessity = enough and joint_mean > 0.03 and specificity > 0.02 and distributed_gain > 0.01 and consistency >= 0.75
            gates = gate_by_mechanism[(family, mechanism)] if split == "heldout" else []
            gate_ratio = safe_mean([float(row["energy_ratio"]) for row in gates])
            gate_consistency = round(sum(row["positive_exceeds_control"] for row in gates) / len(gates), 6) if gates else 0.0
            natural_gate_pass = split == "heldout" and gate_ratio > 1.2 and gate_consistency >= 0.75
            result.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": model,
                "family_id": family,
                "mechanism_id": mechanism,
                "split": split,
                "registered_case_count": len(by_condition["baseline"]),
                "baseline_eligible_count": len(eligible_ids),
                "baseline_accuracy": round(len(eligible_ids) / max(1, len(by_condition["baseline"])), 6),
                "joint_margin_drop": joint_mean,
                "matched_random_margin_drop": random_mean,
                "wrong_layer_margin_drop": wrong_mean,
                "best_single_margin_drop": best_single,
                "matched_control_specificity": specificity,
                "distributed_gain_over_single": distributed_gain,
                "positive_effect_consistency": consistency,
                "joint_js_divergence": js_joint,
                "random_js_divergence": js_random,
                "set_necessity_pass": pass_set_necessity,
                "natural_gate_energy_ratio": gate_ratio,
                "natural_gate_consistency": gate_consistency,
                "natural_gate_observational_pass": natural_gate_pass,
                "evidence_level": "L4_set_necessity_candidate" if pass_set_necessity else "L3_or_negative",
                "single_unit_causal": False,
                "l5_promoted": False,
            })
    return result


def model_summary(
    model: str, loaded: Any, layers: list[int], natural: list[dict[str, Any]], selections: list[dict[str, Any]],
    interventions: list[dict[str, Any]], gates: list[dict[str, Any]], registered: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "model_name_or_path": str(getattr(loaded.model.config, "_name_or_path", "")),
        "model_revision": str(getattr(loaded.model.config, "_commit_hash", None) or "local_unknown"),
        "candidate_layers": layers,
        "natural_case_count": len(natural),
        "natural_baseline_accuracy": round(sum(row["candidate_winner_is_target"] for row in natural) / max(1, len(natural)), 6),
        "knowledge_answer_leak_count": sum(
            not row["target_absent_from_prompt"] for row in natural if row["family_id"] == "content_knowledge"
        ),
        "multi_token_reasoning_case_count": sum(
            row["family_id"] == "reasoning_constraint" and row["source_token_count"] > 1 for row in natural
        ),
        "carrier_component_count": len(selections),
        "attention_carrier_count": sum(row["component_type"] == "attention_head_input" for row in selections),
        "mlp_group_carrier_count": sum(row["component_type"] == "mlp_product_group" for row in selections),
        "intervention_row_count": len(interventions),
        "natural_gate_control_count": len(gates),
        "registered_row_count": len(registered),
        "heldout_set_necessity_pass_count": sum(row["split"] == "heldout" and row["set_necessity_pass"] for row in registered),
        "heldout_natural_gate_observational_pass_count": sum(
            row["split"] == "heldout" and row["natural_gate_observational_pass"] for row in registered
        ),
        "single_unit_causal_count": 0,
        "l5_promoted_count": 0,
    }


def collect(round_name: str) -> dict[str, Any]:
    out = OUT / round_name
    summaries = [read_json(out / f"phase326_{model}_summary.json") for model in MODELS]
    all_selections: list[dict[str, Any]] = []
    all_interventions: list[dict[str, Any]] = []
    all_registered: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    all_natural: list[dict[str, Any]] = []
    all_component_summaries: list[dict[str, Any]] = []
    for model in MODELS:
        all_selections.extend(read_jsonl(out / f"phase326_{model}_carrier_sets.jsonl"))
        all_interventions.extend(read_jsonl(out / f"phase326_{model}_intervention_rows.jsonl"))
        all_registered.extend(read_jsonl(out / f"phase326_{model}_registered_heldout.jsonl"))
        all_gates.extend(read_jsonl(out / f"phase326_{model}_natural_gate_rows.jsonl"))
        all_natural.extend(read_jsonl(out / f"phase326_{model}_natural_rows.jsonl"))
        all_component_summaries.extend(read_jsonl(out / f"phase326_{model}_component_summary_rows.jsonl"))

    mechanism_rows = []
    mechanisms = sorted({(row["family_id"], row["mechanism_id"]) for row in all_registered})
    for family, mechanism in mechanisms:
        heldout = [row for row in all_registered if row["family_id"] == family and row["mechanism_id"] == mechanism and row["split"] == "heldout"]
        pass_models = [row["model"] for row in heldout if row["set_necessity_pass"]]
        gate_models = [row["model"] for row in heldout if row["natural_gate_observational_pass"]]
        replicated = len(pass_models) >= 2
        mechanism_rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "family_id": family,
            "mechanism_id": mechanism,
            "models_tested": MODELS,
            "set_necessity_pass_models": pass_models,
            "natural_gate_observational_pass_models": gate_models,
            "cross_model_set_necessity_replicated": replicated,
            "l5_promoted": False,
            "l5_blocker": "No natural sufficiency intervention and no causal natural gate, even when set necessity replicates.",
        })

    nodes, edges = atlas_rows(all_selections, all_registered)
    cross = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "models": MODELS,
        "registered_prompt_cases": len(all_natural),
        "registered_independent_model_cases": 96 * len(MODELS),
        "families_tested": 2,
        "mechanisms_tested": 8,
        "carrier_component_count": len(all_selections),
        "intervention_row_count": len(all_interventions),
        "natural_gate_control_count": len(all_gates),
        "heldout_set_necessity_pass_count": sum(row["split"] == "heldout" and row["set_necessity_pass"] for row in all_registered),
        "cross_model_replicated_mechanism_count": sum(row["cross_model_set_necessity_replicated"] for row in mechanism_rows),
        "single_unit_causal_count": 0,
        "l5_promoted_count": 0,
        "model_summaries": summaries,
        "mechanism_results": mechanism_rows,
        "evidence_statement": "Phase326 maps frozen distributed component-set candidates and tests set-level necessity; it does not identify causal single neurons or natural sufficiency.",
    }
    write_jsonl(out / "phase326_carrier_sets.jsonl", all_selections)
    write_jsonl(out / "phase326_intervention_rows.jsonl", all_interventions)
    write_jsonl(out / "phase326_registered_heldout.jsonl", all_registered)
    write_jsonl(out / "phase326_natural_gate_rows.jsonl", all_gates)
    write_jsonl(out / "phase326_component_summary_rows.jsonl", all_component_summaries)
    write_jsonl(out / "phase326_atlas_nodes.jsonl", nodes)
    write_jsonl(out / "phase326_atlas_edges.jsonl", edges)
    write_json(out / "phase326_cross_model_summary.json", cross)
    publish(out, cross)
    print(json.dumps(cross, ensure_ascii=False, indent=2))
    return cross


def atlas_rows(selections: list[dict[str, Any]], registered: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    heldout = {
        (row["model"], row["family_id"], row["mechanism_id"]): row
        for row in registered if row["split"] == "heldout"
    }
    nodes, edges = [], []
    for row in selections:
        identity = "|".join(str(row[key]) for key in (
            "model", "family_id", "mechanism_id", "component_type", "component_layer", "position_role", "component_index"
        ))
        node_id = "p326_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        audit = heldout.get((row["model"], row["family_id"], row["mechanism_id"]), {})
        nodes.append({
            "schema_version": "pattern_family_neuron_atlas.v1",
            "node_id": node_id,
            "family_id": row["family_id"],
            "mechanism_id": row["mechanism_id"],
            "model": row["model"],
            "model_revision": "local_unknown",
            "layer": row["component_layer"],
            "component": "attention" if row["component_type"] == "attention_head_input" else "mlp",
            "unit_kind": "attention_head" if row["component_type"] == "attention_head_input" else "mlp_product_group",
            "unit_index": row["component_index"],
            "position_role": row["position_role"],
            "display_priority": row["rank_score"],
            "discovery_independent_cases": row["independent_case_count"],
            "discovery_templates": row["template_count"],
            "heldout_set_necessity_pass": bool(audit.get("set_necessity_pass", False)),
            "evidence_level": "L4_set_member" if audit.get("set_necessity_pass") else "L3_candidate",
            "causal_scope": "distributed_component_set",
            "single_unit_causal": False,
            "source_file": "phase326_carrier_sets.jsonl",
        })
    by_path: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_path[(node["model"], node["family_id"], node["mechanism_id"])].append(node)
    for key, path_nodes in by_path.items():
        ordered = sorted(path_nodes, key=lambda row: (row["layer"], ROLES.index(row["position_role"]), row["unit_kind"], row["unit_index"]))
        for source, target in zip(ordered, ordered[1:]):
            edge_id = "p326e_" + hashlib.sha1(f"{source['node_id']}|{target['node_id']}".encode("utf-8")).hexdigest()[:16]
            edges.append({
                "schema_version": "pattern_family_neuron_atlas.v1",
                "edge_id": edge_id,
                "source": source["node_id"],
                "target": target["node_id"],
                "family_id": key[1],
                "mechanism_id": key[2],
                "model": key[0],
                "edge_type": "observed_distributed_carrier_sequence",
                "causal": False,
                "causal_scope": "set_level_only",
            })
    return nodes, edges


def publish(out: Path, cross: dict[str, Any]) -> None:
    files = (
        "phase326_carrier_sets.jsonl", "phase326_intervention_rows.jsonl", "phase326_registered_heldout.jsonl",
        "phase326_natural_gate_rows.jsonl", "phase326_component_summary_rows.jsonl", "phase326_atlas_nodes.jsonl",
        "phase326_atlas_edges.jsonl", "phase326_cross_model_summary.json",
    )
    optional_files = (
        "phase326_expanded_confirmation_rows.jsonl",
        "phase326_expanded_confirmation_audits.jsonl",
        "phase326_expanded_confirmation_cross_model.json",
        "phase326_token_span_validation.json",
    )
    for destination in (V2, FRONTEND_V2):
        destination.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copy2(out / name, destination / name)
        for name in optional_files:
            if (out / name).exists():
                shutil.copy2(out / name, destination / name)
        manifest_path = destination / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"schema_version": "pattern_family_atlas.v2"}
        manifest["updated_at"] = now()
        manifest["phase326"] = {
            "status": "available",
            "families": ["content_knowledge", "reasoning_constraint"],
            "mechanisms": 8,
            "registered_prompt_cases": cross["registered_prompt_cases"],
            "carrier_components": cross["carrier_component_count"],
            "single_unit_causal_count": 0,
            "l5_promoted_count": 0,
            "files": list(files),
        }
        if (out / "phase326_expanded_confirmation_cross_model.json").exists():
            manifest["phase326"]["expanded_confirmation"] = {
                "status": "available",
                "files": list(optional_files),
            }
        write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--collect-confirmation", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if args.collect_confirmation:
        collect_confirmation(args.round)
    elif args.collect:
        collect(args.round)
    elif args.confirm and args.model:
        run_confirmation_model(args.model, args.round)
    elif args.model:
        run_model(args.model, args.round, args.max_cases)
    else:
        raise SystemExit("Use --model MODEL, --confirm, --collect, or --collect-confirmation")


if __name__ == "__main__":
    main()
