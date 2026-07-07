#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase204_global_trajectory_stop_execution_atlas as p204  # noqa: E402
import phase209_pattern_running_contrast_atlas as p209  # noqa: E402
import phase212_switchpoint_causal_validation as p212  # noqa: E402
import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402
from model_utils import get_layers  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402


PHASE = 219
SOURCE_PHASE = 218
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase219_state_write_mlp_causal_validation")


PATTERN_SPECS = {
    "qwen3": [
        {"spec_id": "qwen3_explain_state_write", "pattern_id": "answer_explain", "layers": [11, 29, 31, 33]},
        {"spec_id": "qwen3_repeat_state_write", "pattern_id": "answer_repeat", "layers": [29, 31, 32, 33]},
    ],
    "glm4": [
        {"spec_id": "glm4_repeat_state_write", "pattern_id": "answer_repeat", "layers": [12, 28, 29, 30]},
        {"spec_id": "glm4_explain_competition_state_write", "pattern_id": "answer_explain", "layers": [12, 28, 29, 30]},
    ],
    "deepseek7b": [
        {"spec_id": "deepseek7b_explain_state_write", "pattern_id": "answer_explain", "layers": [24, 25, 26, 27]},
    ],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def module_output_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def replace_module_output(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple) and output:
        return (hidden, *output[1:])
    return output


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def select_rows(rows: list[dict[str, Any]], pattern_id: str, max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success = [row for row in rows if row.get("pattern_id") == pattern_id and row.get("pattern_match")]
    drift = [row for row in rows if row.get("pattern_id") == pattern_id and not row.get("pattern_match")]
    return success[: int(max_rows)], drift[: int(max_rows)]


def prefix_for_step(row: dict[str, Any], step: int) -> str:
    emitted = row.get("emitted_tokens") or []
    prefix_tokens = emitted[: max(0, int(step) - 1)]
    return str(row.get("prompt") or "") + "".join(str(tok) for tok in prefix_tokens)


def hidden_at_text(model, tokenizer, device: torch.device, text: str, layer_idx: int) -> torch.Tensor:
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    vec = result.hidden_states[int(layer_idx) + 1][0, last_pos].detach().float().cpu()
    del result, input_ids, attention_mask
    return vec


def mean_hidden(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict[str, Any]],
    layer_idx: int,
    step: int,
) -> torch.Tensor | None:
    vectors = []
    for row in rows:
        try:
            vectors.append(hidden_at_text(model, tokenizer, device, prefix_for_step(row, step), int(layer_idx)))
        except Exception as exc:
            log(f"skip hidden {row.get('trajectory_id')}: {exc}")
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def build_direction_vectors(
    model,
    tokenizer,
    device: torch.device,
    success_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    layers: list[int],
    max_steps: int,
) -> dict[int, dict[int, torch.Tensor]]:
    out: dict[int, dict[int, torch.Tensor]] = {}
    for step in range(1, int(max_steps) + 1):
        out[int(step)] = {}
        for layer_idx in layers:
            s = mean_hidden(model, tokenizer, device, success_rows, int(layer_idx), int(step))
            d = mean_hidden(model, tokenizer, device, drift_rows, int(layer_idx), int(step))
            if s is not None and d is not None:
                vec = s - d
                norm = torch.linalg.vector_norm(vec).item()
                if norm > 0:
                    out[int(step)][int(layer_idx)] = vec
    return out


def parse_condition(condition: str) -> tuple[str, int | None]:
    if condition == "none":
        return "none", None
    m = re.match(r"^(resid_add|resid_sub|mlp_zero|attn_zero)_L(\d+)$", condition)
    if not m:
        raise ValueError(f"unknown condition: {condition}")
    return m.group(1), int(m.group(2))


def install_condition_hooks(
    model,
    condition: str,
    step: int,
    direction_vectors: dict[int, dict[int, torch.Tensor]],
    scale: float,
):
    kind, layer_idx = parse_condition(condition)
    if kind == "none" or layer_idx is None:
        return []
    layers = get_layers(model)
    layer = layers[int(layer_idx)]
    dtype = next(model.parameters()).dtype
    handles = []

    if kind in {"resid_add", "resid_sub"}:
        vec = direction_vectors.get(int(step), {}).get(int(layer_idx))
        if vec is None:
            return []
        sign = 1.0 if kind == "resid_add" else -1.0
        patch_vec = (sign * float(scale) * vec).to(device=next(model.parameters()).device, dtype=dtype)

        def layer_hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
            hidden = module_output_tensor(output)
            if hidden is None:
                return output
            y = hidden.clone()
            y[:, -1, :] = y[:, -1, :] + patch_vec
            return replace_module_output(output, y)

        handles.append(layer.register_forward_hook(layer_hook))
        return handles

    if kind == "mlp_zero":
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return []

        def mlp_hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
            hidden = module_output_tensor(output)
            if hidden is None:
                return output
            y = hidden.clone()
            y[:, -1, :] = 0
            return replace_module_output(output, y)

        handles.append(mlp.register_forward_hook(mlp_hook))
        return handles

    if kind == "attn_zero":
        attn = get_attention_module(layer)

        def attn_hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
            hidden = module_output_tensor(output)
            if hidden is None:
                return output
            y = hidden.clone()
            y[:, -1, :] = 0
            return replace_module_output(output, y)

        handles.append(attn.register_forward_hook(attn_hook))
        return handles

    return handles


def forward_logits_condition(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    condition: str,
    step: int,
    direction_vectors: dict[int, dict[int, torch.Tensor]],
    scale: float,
) -> torch.Tensor:
    handles = install_condition_hooks(model, condition, int(step), direction_vectors, float(scale))
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = result.logits[0, last_pos].detach().float().cpu()
        del result
    finally:
        for handle in handles:
            handle.remove()
        del input_ids, attention_mask
    return logits


def generate_condition(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    row: dict[str, Any],
    condition: str,
    direction_vectors: dict[int, dict[int, torch.Tensor]],
    max_steps: int,
    scale: float,
) -> dict[str, Any]:
    generated = ""
    emitted_ids: list[int] = []
    emitted_tokens: list[str] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    token_rows = []
    for step in range(1, int(max_steps) + 1):
        logits = forward_logits_condition(
            model,
            tokenizer,
            device,
            str(row.get("prompt") or "") + generated,
            condition,
            int(step),
            direction_vectors,
            float(scale),
        )
        metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
        next_id = int(metrics["top_token_id"])
        next_text = str(metrics.get("top_token") or p212.token_text(tokenizer, next_id))
        emitted_ids.append(next_id)
        emitted_tokens.append(next_text)
        token_rows.append(
            {
                "step": int(step),
                "top_token": next_text,
                "target_rank": metrics.get("target_rank"),
                "prose_margin": metrics.get("prose_margin"),
                "echo_margin": metrics.get("echo_margin"),
                "stop_margin": metrics.get("stop_margin"),
            }
        )
        generated += next_text
        if next_id in eos_ids:
            break
    expected = p209.expected_output_pattern(str(row.get("pattern_id")))
    classification = p209.classify_pattern(generated, row, emitted_ids, eos_ids)
    return {
        "generated": generated,
        "emitted_ids": emitted_ids,
        "emitted_tokens": emitted_tokens,
        "steps_generated": len(emitted_ids),
        "expected_output_pattern": expected,
        "pattern_match": classification.get("output_pattern") == expected,
        "pattern_drift": classification.get("output_pattern") != expected,
        "failure_mode": "match" if classification.get("output_pattern") == expected else classification.get("output_pattern"),
        "token_rows": token_rows,
        **classification,
    }


def capture_write_vectors(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    layers_to_scan: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    layers = get_layers(model)
    captured: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    handles = []

    for layer_idx in layers_to_scan:
        layer = layers[int(layer_idx)]
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:

            def make_mlp_hook(li: int):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                    hidden = module_output_tensor(output)
                    if hidden is not None:
                        captured[int(li)]["mlp"] = hidden[0, -1, :].detach().float().cpu()
                    return None

                return hook

            handles.append(mlp.register_forward_hook(make_mlp_hook(int(layer_idx))))

        attn = get_attention_module(layer)

        def make_attn_hook(li: int):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
                hidden = module_output_tensor(output)
                if hidden is not None:
                    captured[int(li)]["attn"] = hidden[0, -1, :].detach().float().cpu()
                return None

            return hook

        handles.append(attn.register_forward_hook(make_attn_hook(int(layer_idx))))

    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        for layer_idx in layers_to_scan:
            hidden_idx = min(int(layer_idx) + 1, len(result.hidden_states) - 1)
            captured[int(layer_idx)]["resid"] = result.hidden_states[hidden_idx][0, last_pos].detach().float().cpu()
        del result
    finally:
        for handle in handles:
            handle.remove()
        del input_ids, attention_mask
    return captured


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    if torch.linalg.vector_norm(a).item() == 0 or torch.linalg.vector_norm(b).item() == 0:
        return None
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def write_score_rows(
    model,
    tokenizer,
    device: torch.device,
    model_name: str,
    spec: dict[str, Any],
    rows_by_group: list[tuple[str, list[dict[str, Any]]]],
    direction_vectors: dict[int, dict[int, torch.Tensor]],
    max_rows: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    out = []
    layers = [int(x) for x in spec["layers"]]
    for source_group, rows in rows_by_group:
        for row in rows[: int(max_rows)]:
            for step in range(1, int(max_steps) + 1):
                captured = capture_write_vectors(model, tokenizer, device, prefix_for_step(row, int(step)), layers)
                for layer_idx in layers:
                    direction = direction_vectors.get(int(step), {}).get(int(layer_idx))
                    for module_name in ["resid", "mlp", "attn"]:
                        vec = captured.get(int(layer_idx), {}).get(module_name)
                        out.append(
                            {
                                "phase": PHASE,
                                "source_phase": SOURCE_PHASE,
                                "row_kind": "phase219_write_score_row",
                                "model": model_name,
                                "spec_id": spec["spec_id"],
                                "pattern_id": spec["pattern_id"],
                                "source_group": source_group,
                                "trajectory_id": row.get("trajectory_id"),
                                "step": int(step),
                                "layer_idx": int(layer_idx),
                                "module": module_name,
                                "write_score_cosine": cosine(vec, direction),
                                "module_norm": None if vec is None else float(torch.linalg.vector_norm(vec).item()),
                                "direction_norm": None if direction is None else float(torch.linalg.vector_norm(direction).item()),
                            }
                        )
    return out


def summarize_rollouts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["spec_id", "source_group", "condition"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, items in buckets.items():
        rec = {name: value for name, value in zip(keys, key)}
        rec.update(
            {
                "rows": len(items),
                "pattern_match": sum(1 for item in items if item.get("pattern_match")),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "output_patterns": dict(Counter(str(item.get("output_pattern")) for item in items).most_common()),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
            }
        )
        out.append(rec)
    out.sort(key=lambda row: (str(row.get("spec_id")), str(row.get("source_group")), str(row.get("condition"))))
    return out


def effect_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        buckets[str(row.get("spec_id"))][(str(row.get("source_group")), str(row.get("condition")))] = row
    out = []
    for spec_id, by in buckets.items():
        success_none = by.get(("success_repro", "none"), {})
        drift_none = by.get(("drift_repro", "none"), {})
        conditions = sorted({key[1] for key in by if key[1] != "none"})
        for condition in conditions:
            success_patch = by.get(("success_repro", condition), {})
            drift_patch = by.get(("drift_repro", condition), {})
            out.append(
                {
                    "spec_id": spec_id,
                    "condition": condition,
                    "success_rows": success_none.get("rows", 0),
                    "drift_rows": drift_none.get("rows", 0),
                    "success_base_match": finite_int(success_none.get("pattern_match")),
                    "success_patch_match": finite_int(success_patch.get("pattern_match")),
                    "drift_base_match": finite_int(drift_none.get("pattern_match")),
                    "drift_patch_match": finite_int(drift_patch.get("pattern_match")),
                    "damage_match_loss": finite_int(success_none.get("pattern_match")) - finite_int(success_patch.get("pattern_match")),
                    "repair_match_gain": finite_int(drift_patch.get("pattern_match")) - finite_int(drift_none.get("pattern_match")),
                    "success_base_outputs": success_none.get("output_patterns", {}),
                    "success_patch_outputs": success_patch.get("output_patterns", {}),
                    "drift_base_outputs": drift_none.get("output_patterns", {}),
                    "drift_patch_outputs": drift_patch.get("output_patterns", {}),
                }
            )
    out.sort(key=lambda row: abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0)), reverse=True)
    return out


def summarize_write_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["spec_id", "source_group", "layer_idx", "module"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, items in buckets.items():
        vals = [float(item["write_score_cosine"]) for item in items if item.get("write_score_cosine") is not None]
        norms = [float(item["module_norm"]) for item in items if item.get("module_norm") is not None]
        rec = {name: value for name, value in zip(keys, key)}
        rec.update(
            {
                "rows": len(items),
                "valid_scores": len(vals),
                "write_score_mean": sum(vals) / len(vals) if vals else None,
                "write_score_abs_mean": sum(abs(v) for v in vals) / len(vals) if vals else None,
                "module_norm_mean": sum(norms) / len(norms) if norms else None,
            }
        )
        out.append(rec)
    out.sort(
        key=lambda row: (
            str(row.get("spec_id")),
            str(row.get("source_group")),
            -float(row.get("write_score_abs_mean") or 0.0),
        )
    )
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    filter_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    write_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        rows = load_rows(args.model, args.phase210_round)
        for spec in PATTERN_SPECS[args.model]:
            success_rows, drift_rows = select_rows(rows, str(spec["pattern_id"]), int(args.max_filter_rows))
            kept_success: list[dict[str, Any]] = []
            kept_drift: list[dict[str, Any]] = []
            empty_dirs: dict[int, dict[int, torch.Tensor]] = {}
            for source_group, source_rows, target_list in [
                ("success", success_rows, kept_success),
                ("drift", drift_rows, kept_drift),
            ]:
                for row in source_rows:
                    result = generate_condition(
                        model,
                        tokenizer,
                        device,
                        groups,
                        row,
                        "none",
                        empty_dirs,
                        int(args.max_steps),
                        float(args.direction_scale),
                    )
                    reproducible = bool(result.get("pattern_match")) if source_group == "success" else not bool(result.get("pattern_match"))
                    filter_rows.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase219_baseline_filter_row",
                            "model": args.model,
                            "spec_id": spec["spec_id"],
                            "pattern_id": spec["pattern_id"],
                            "source_group": source_group,
                            "trajectory_id": row.get("trajectory_id"),
                            "reproducible": reproducible,
                            "output_pattern": result.get("output_pattern"),
                            "pattern_match": result.get("pattern_match"),
                        }
                    )
                    if reproducible:
                        target_list.append(row)
            kept_success = kept_success[: int(args.max_direction_rows)]
            kept_drift = kept_drift[: int(args.max_direction_rows)]
            if not kept_success or not kept_drift:
                log(f"{args.model}|{spec['spec_id']}: insufficient reproducible success={len(kept_success)} drift={len(kept_drift)}")
                continue
            direction_vectors = build_direction_vectors(
                model,
                tokenizer,
                device,
                kept_success,
                kept_drift,
                [int(x) for x in spec["layers"]],
                int(args.max_steps),
            )
            eval_success = kept_success[: int(args.max_eval_rows)]
            eval_drift = kept_drift[: int(args.max_eval_rows)]
            write_rows.extend(
                write_score_rows(
                    model,
                    tokenizer,
                    device,
                    args.model,
                    spec,
                    [("success_repro", eval_success), ("drift_repro", eval_drift)],
                    direction_vectors,
                    int(args.max_write_rows),
                    int(args.max_write_steps),
                )
            )
            conditions = ["none"]
            for layer_idx in spec["layers"]:
                conditions.extend(
                    [
                        f"resid_add_L{int(layer_idx)}",
                        f"resid_sub_L{int(layer_idx)}",
                        f"mlp_zero_L{int(layer_idx)}",
                        f"attn_zero_L{int(layer_idx)}",
                    ]
                )
            for source_group, eval_rows in [("success_repro", eval_success), ("drift_repro", eval_drift)]:
                for row in eval_rows:
                    for condition in conditions:
                        result = generate_condition(
                            model,
                            tokenizer,
                            device,
                            groups,
                            row,
                            condition,
                            direction_vectors,
                            int(args.max_steps),
                            float(args.direction_scale),
                        )
                        rollout_rows.append(
                            {
                                "phase": PHASE,
                                "source_phase": SOURCE_PHASE,
                                "row_kind": "phase219_state_write_rollout_row",
                                "model": args.model,
                                "spec_id": spec["spec_id"],
                                "pattern_id": spec["pattern_id"],
                                "layers": spec["layers"],
                                "source_group": source_group,
                                "condition": condition,
                                "trajectory_id": row.get("trajectory_id"),
                                "target_label": row.get("target_label"),
                                "object": row.get("object"),
                                **result,
                            }
                        )
            log(f"{args.model}|{spec['spec_id']}: kept success={len(eval_success)} drift={len(eval_drift)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary_rows = summarize_rollouts(rollout_rows)
    effects = effect_rows(summary_rows)
    write_summary_rows = summarize_write_scores(write_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "State Write and MLP Causal Validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "spec_count": len(PATTERN_SPECS[args.model]),
        "filter_rows": len(filter_rows),
        "reproducible_success_rows": sum(1 for row in filter_rows if row.get("source_group") == "success" and row.get("reproducible")),
        "reproducible_drift_rows": sum(1 for row in filter_rows if row.get("source_group") == "drift" and row.get("reproducible")),
        "rollout_rows": len(rollout_rows),
        "write_score_rows": len(write_rows),
        "summary_rows": len(summary_rows),
        "effect_rows": effects,
        "write_summary_rows": write_summary_rows,
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effects),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effects),
    }
    write_json(out_dir / f"phase219_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase219_{args.model}_filter_rows.jsonl", filter_rows)
    write_jsonl(out_dir / f"phase219_{args.model}_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase219_{args.model}_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase219_{args.model}_effect_rows.jsonl", effects)
    write_jsonl(out_dir / f"phase219_{args.model}_write_score_rows.jsonl", write_rows)
    write_jsonl(out_dir / f"phase219_{args.model}_write_summary_rows.jsonl", write_summary_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "filter_rows": len(filter_rows),
                "rollout_rows": len(rollout_rows),
                "write_score_rows": len(write_rows),
                "damage": payload["total_damage_match_loss"],
                "repair": payload["total_repair_match_gain"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase219_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    effects = []
    write_summaries = []
    for model in MODELS:
        effects.extend(p214.iter_jsonl(out_dir / f"phase219_{model}_effect_rows.jsonl") or [])
        write_summaries.extend(p214.iter_jsonl(out_dir / f"phase219_{model}_write_summary_rows.jsonl") or [])
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model State Write and MLP Causal Validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "spec_count": sum(int(summary.get("spec_count") or 0) for summary in summaries),
        "filter_rows": sum(int(summary.get("filter_rows") or 0) for summary in summaries),
        "reproducible_success_rows": sum(int(summary.get("reproducible_success_rows") or 0) for summary in summaries),
        "reproducible_drift_rows": sum(int(summary.get("reproducible_drift_rows") or 0) for summary in summaries),
        "rollout_rows": sum(int(summary.get("rollout_rows") or 0) for summary in summaries),
        "write_score_rows": sum(int(summary.get("write_score_rows") or 0) for summary in summaries),
        "effect_rows": len(effects),
        "write_summary_rows": len(write_summaries),
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effects),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effects),
        "top_effect_rows": sorted(
            effects,
            key=lambda row: abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0)),
            reverse=True,
        )[:40],
        "top_write_summary_rows": sorted(
            write_summaries,
            key=lambda row: float(row.get("write_score_abs_mean") or 0.0),
            reverse=True,
        )[:40],
    }
    write_json(out_dir / "phase219_cross_model_summary.json", payload)
    lines = ["# Phase 219 state write and MLP causal validation", ""]
    for key in [
        "spec_count",
        "filter_rows",
        "reproducible_success_rows",
        "reproducible_drift_rows",
        "rollout_rows",
        "write_score_rows",
        "total_damage_match_loss",
        "total_repair_match_gain",
    ]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(["", "| spec | condition | success | drift | damage | repair | success outputs | drift outputs |", "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in payload["top_effect_rows"][:30]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('condition')} | {row.get('success_rows')} | {row.get('drift_rows')} | "
            f"{row.get('damage_match_loss')} | {row.get('repair_match_gain')} | {row.get('success_patch_outputs')} | {row.get('drift_patch_outputs')} |"
        )
    lines.extend(["", "## Top write scores", "", "| spec | group | layer | module | rows | cosine | abs cosine | norm |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["top_write_summary_rows"][:30]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('layer_idx')} | {row.get('module')} | "
            f"{row.get('rows')} | {row.get('write_score_mean')} | {row.get('write_score_abs_mean')} | {row.get('module_norm_mean')} |"
        )
    (out_dir / "phase219_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "specs": payload["spec_count"],
                "rollout_rows": payload["rollout_rows"],
                "write_score_rows": payload["write_score_rows"],
                "damage": payload["total_damage_match_loss"],
                "repair": payload["total_repair_match_gain"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase219 state write and MLP causal validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="state_write_mlp_causal_validation")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=8)
    parser.add_argument("--max-direction-rows", type=int, default=6)
    parser.add_argument("--max-eval-rows", type=int, default=3)
    parser.add_argument("--max-write-rows", type=int, default=2)
    parser.add_argument("--max-write-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--direction-scale", type=float, default=1.0)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    args = parser.parse_args()
    if not args.summarize and not args.model:
        parser.error("--model is required unless --summarize is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
