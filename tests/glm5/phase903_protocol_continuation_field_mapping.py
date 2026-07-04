#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase900_protocol_stop_gate_discovery as p900  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
import phase902_protocol_continuation_suppressor_search as p902  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 903
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase903_protocol_continuation_field_mapping")
PHASE899_ROOT = Path("tests/result/phase899_domain_axis_rollout_protocol_audit")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def median(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def selected_phase899_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE899_ROOT / args.phase899_round / f"phase899_{model_name}_rollout_rows.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("is_source_candidate") and row.get("rollout_answer_class") and row.get("protocol_drift")
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("eval_domain")),
            str(row.get("source_subset_key")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("edit_mode")),
        )
    )
    max_rows = int(args.max_rows_per_model)
    return rows[:max_rows] if max_rows > 0 else rows


def parse_gears(subset_key: str) -> list[dict[str, Any]]:
    gears = []
    for part in str(subset_key or "").split("+"):
        if part.startswith("L") and "C" in part:
            gear = p862.parse_gear_key(part)
            if gear is not None:
                gears.append(gear)
    return gears


def first_token_ids(tokenizer, phrases: list[str]) -> list[int]:
    return p901.first_token_ids(tokenizer, phrases)


def unique_ids(ids: list[int]) -> list[int]:
    out = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id >= 0 and token_id not in out:
            out.append(token_id)
    return out


def protocol_category_groups(tokenizer) -> dict[str, list[int]]:
    eos = []
    if tokenizer.eos_token_id is not None:
        eos.append(int(tokenizer.eos_token_id))
    groups = {
        "eos": eos,
        "period": first_token_ids(tokenizer, [".", " .", ".\n"]),
        "newline": first_token_ids(tokenizer, ["\n", "\n\n"]),
        "comma": first_token_ids(tokenizer, [",", " ,"]),
        "field_word": first_token_ids(
            tokenizer,
            [
                "Category",
                " Category",
                "Item",
                " Item",
                "Class",
                " Class",
                "Subclass",
                " Subclass",
                "Answer",
                " Answer",
            ],
        ),
        "explanation": first_token_ids(
            tokenizer,
            ["The", " The", "I", " I", "Okay", " Okay", "Please", " Please", "This", " This"],
        ),
        "list_word": first_token_ids(tokenizer, [" or", "or", "1", " 1", "2", " 2"]),
    }
    for key, values in list(groups.items()):
        groups[key] = unique_ids(values)
    groups["stop"] = unique_ids(groups["eos"] + groups["period"])
    groups["protocol"] = unique_ids(
        groups["newline"] + groups["comma"] + groups["field_word"] + groups["explanation"] + groups["list_word"]
    )
    return groups


def decode_token(tokenizer, token_id: int | None) -> str | None:
    return p901.decode_token(tokenizer, token_id)


def best_for_ids(logits: torch.Tensor, ids: list[int]) -> dict[str, Any]:
    return p901.best_for_ids(logits, ids)


def category_best(logits: torch.Tensor, groups: dict[str, list[int]], categories: list[str]) -> dict[str, Any]:
    best: dict[str, Any] = {"category": None, "best_id": None, "best_logit": None, "rank": None}
    for category in categories:
        item = best_for_ids(logits, groups.get(category) or [])
        if item.get("best_logit") is None:
            continue
        if best["best_logit"] is None or float(item["best_logit"]) > float(best["best_logit"]):
            best = {"category": category, **item}
    return best


def category_for_token(token_id: int | None, groups: dict[str, list[int]]) -> str:
    if token_id is None:
        return "none"
    token_id = int(token_id)
    for category in ["eos", "period", "newline", "comma", "field_word", "explanation", "list_word"]:
        if token_id in set(int(x) for x in groups.get(category) or []):
            return category
    return "other"


def state_metrics(tokenizer, logits: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, Any]:
    top_id = int(torch.argmax(logits).item())
    top_logit = float(logits[top_id].item())
    protocol_best = category_best(logits, groups, ["newline", "comma", "field_word", "explanation", "list_word"])
    stop_best = category_best(logits, groups, ["eos", "period"])
    payload: dict[str, Any] = {
        "next_top_id": top_id,
        "next_top_token": decode_token(tokenizer, top_id),
        "next_top_category": category_for_token(top_id, groups),
        "next_top_logit": top_logit,
        "protocol_best_category": protocol_best.get("category"),
        "protocol_best_id": protocol_best.get("best_id"),
        "protocol_best_token": decode_token(tokenizer, protocol_best.get("best_id")),
        "protocol_best_logit": protocol_best.get("best_logit"),
        "protocol_rank": protocol_best.get("rank"),
        "stop_best_category": stop_best.get("category"),
        "stop_best_id": stop_best.get("best_id"),
        "stop_best_token": decode_token(tokenizer, stop_best.get("best_id")),
        "stop_best_logit": stop_best.get("best_logit"),
        "stop_rank": stop_best.get("rank"),
        "protocol_margin_vs_top": None
        if protocol_best.get("best_logit") is None
        else float(protocol_best["best_logit"] - top_logit),
        "stop_margin_vs_top": None if stop_best.get("best_logit") is None else float(stop_best["best_logit"] - top_logit),
    }
    for category in ["newline", "comma", "field_word", "explanation", "list_word", "period", "eos"]:
        item = best_for_ids(logits, groups.get(category) or [])
        payload[f"{category}_best_id"] = item.get("best_id")
        payload[f"{category}_best_token"] = decode_token(tokenizer, item.get("best_id"))
        payload[f"{category}_best_logit"] = item.get("best_logit")
        payload[f"{category}_rank"] = item.get("rank")
    payload["protocol_top1"] = bool(payload.get("protocol_rank") == 1)
    payload["stop_top10"] = bool(payload.get("stop_rank") is not None and int(payload["stop_rank"]) <= 10)
    payload["stop_top1"] = bool(payload.get("stop_rank") == 1)
    return payload


def logits_plain(model, device: torch.device, current_ids: list[int]) -> torch.Tensor:
    input_ids = torch.tensor([current_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()


def zero_last_token_output(output):
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            return output
        patched = output[0].clone()
        if patched.ndim >= 3:
            patched[:, -1, :] = 0
        elif patched.ndim >= 2:
            patched[:, :] = 0
        return (patched, *output[1:])
    if torch.is_tensor(output):
        patched = output.clone()
        if patched.ndim >= 3:
            patched[:, -1, :] = 0
        elif patched.ndim >= 2:
            patched[:, :] = 0
        return patched
    return output


def component_module(model, layer_idx: int, component_kind: str):
    layers = get_layers(model)
    if not (0 <= int(layer_idx) < len(layers)):
        return None
    layer = layers[int(layer_idx)]
    if component_kind == "attention":
        return getattr(layer, "self_attn", None)
    if component_kind == "mlp":
        return getattr(layer, "mlp", None)
    return None


def logits_with_component_zero(
    model,
    device: torch.device,
    current_ids: list[int],
    layer_idx: int,
    component_kind: str,
) -> torch.Tensor | None:
    module = component_module(model, layer_idx, component_kind)
    if module is None:
        return None
    handle = module.register_forward_hook(lambda _module, _inputs, output: zero_last_token_output(output))
    try:
        return logits_plain(model, device, current_ids)
    finally:
        handle.remove()


def layer_indices(model, stride: int) -> list[int]:
    n_layers = len(get_layers(model))
    stride = max(1, int(stride))
    indices = list(range(0, n_layers, stride))
    if n_layers - 1 not in indices:
        indices.append(n_layers - 1)
    return sorted(set(indices))


def make_state_row(
    source_row: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    answer_seen: bool,
    prompt_metrics: dict[str, Any],
    answer_metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "phase": PHASE,
        "row_kind": "phase903_state_prior_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "canonical_answer": source_row.get("canonical_answer"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "prefix_ids": prefix_ids,
        "prefix_text": prefix_text,
        "answer_prefix_seen": answer_seen,
    }
    for key, value in prompt_metrics.items():
        row[f"prompt_{key}"] = value
    for key, value in answer_metrics.items():
        row[f"answer_{key}"] = value
    if prompt_metrics.get("protocol_rank") is not None and answer_metrics.get("protocol_rank") is not None:
        row["protocol_rank_delta_answer_minus_prompt"] = int(answer_metrics["protocol_rank"]) - int(prompt_metrics["protocol_rank"])
    else:
        row["protocol_rank_delta_answer_minus_prompt"] = None
    return row


def make_component_row(
    tokenizer,
    source_row: dict[str, Any],
    layer_idx: int,
    component_kind: str,
    baseline_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    patched_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
) -> dict[str, Any]:
    base_protocol_id = baseline_metrics.get("protocol_best_id")
    protocol_delta = None
    if base_protocol_id is not None:
        protocol_delta = float(patched_logits[int(base_protocol_id)].item() - baseline_logits[int(base_protocol_id)].item())
    base_stop_id = baseline_metrics.get("stop_best_id")
    stop_delta = None
    if base_stop_id is not None:
        stop_delta = float(patched_logits[int(base_stop_id)].item() - baseline_logits[int(base_stop_id)].item())
    stop_rank_delta = None
    if baseline_metrics.get("stop_rank") is not None and patched_metrics.get("stop_rank") is not None:
        stop_rank_delta = int(patched_metrics["stop_rank"]) - int(baseline_metrics["stop_rank"])
    base_next_id = baseline_metrics.get("next_top_id")
    patched_next_id = patched_metrics.get("next_top_id")
    category_transition = f"{baseline_metrics.get('next_top_category')}->{patched_metrics.get('next_top_category')}"
    protocol_category_transition = f"{baseline_metrics.get('protocol_best_category')}->{patched_metrics.get('protocol_best_category')}"
    return {
        "phase": PHASE,
        "row_kind": "phase903_component_zero_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "layer_idx": int(layer_idx),
        "component_kind": component_kind,
        "baseline_next_top_id": base_next_id,
        "baseline_next_top_token": decode_token(tokenizer, base_next_id),
        "baseline_next_top_category": baseline_metrics.get("next_top_category"),
        "patched_next_top_id": patched_next_id,
        "patched_next_top_token": decode_token(tokenizer, patched_next_id),
        "patched_next_top_category": patched_metrics.get("next_top_category"),
        "baseline_protocol_best_id": base_protocol_id,
        "baseline_protocol_best_token": decode_token(tokenizer, base_protocol_id),
        "baseline_protocol_best_category": baseline_metrics.get("protocol_best_category"),
        "patched_protocol_best_id": patched_metrics.get("protocol_best_id"),
        "patched_protocol_best_token": decode_token(tokenizer, patched_metrics.get("protocol_best_id")),
        "patched_protocol_best_category": patched_metrics.get("protocol_best_category"),
        "baseline_protocol_rank": baseline_metrics.get("protocol_rank"),
        "patched_protocol_rank": patched_metrics.get("protocol_rank"),
        "baseline_stop_rank": baseline_metrics.get("stop_rank"),
        "patched_stop_rank": patched_metrics.get("stop_rank"),
        "baseline_stop_best_category": baseline_metrics.get("stop_best_category"),
        "patched_stop_best_category": patched_metrics.get("stop_best_category"),
        "baseline_protocol_logit_delta": protocol_delta,
        "baseline_stop_logit_delta": stop_delta,
        "stop_rank_delta": stop_rank_delta,
        "protocol_logit_reduced": bool(protocol_delta is not None and protocol_delta < 0),
        "protocol_logit_reduced_strong": bool(protocol_delta is not None and protocol_delta <= -0.5),
        "protocol_rank1_removed": bool(baseline_metrics.get("protocol_rank") == 1 and patched_metrics.get("protocol_rank") != 1),
        "next_top_changed": bool(base_next_id is not None and patched_next_id is not None and int(base_next_id) != int(patched_next_id)),
        "stop_rank_improved": bool(stop_rank_delta is not None and stop_rank_delta < 0),
        "category_transition": category_transition,
        "protocol_category_transition": protocol_category_transition,
    }


def summarize_state_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row.get("protocol_rank_delta_answer_minus_prompt") for row in rows if row.get("protocol_rank_delta_answer_minus_prompt") is not None]
    return {
        "rows": len(rows),
        "answer_prefix_seen": sum(1 for row in rows if row.get("answer_prefix_seen")),
        "prompt_next_top_categories": dict(sorted(Counter(str(row.get("prompt_next_top_category")) for row in rows).items())),
        "answer_next_top_categories": dict(sorted(Counter(str(row.get("answer_next_top_category")) for row in rows).items())),
        "prompt_protocol_best_categories": dict(sorted(Counter(str(row.get("prompt_protocol_best_category")) for row in rows).items())),
        "answer_protocol_best_categories": dict(sorted(Counter(str(row.get("answer_protocol_best_category")) for row in rows).items())),
        "answer_protocol_top1": sum(1 for row in rows if row.get("answer_protocol_top1")),
        "answer_stop_top1": sum(1 for row in rows if row.get("answer_stop_top1")),
        "answer_stop_top10": sum(1 for row in rows if row.get("answer_stop_top10")),
        "mean_protocol_rank_delta_answer_minus_prompt": mean([float(value) for value in deltas]),
        "median_protocol_rank_delta_answer_minus_prompt": median([float(value) for value in deltas]),
    }


def summarize_component_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row.get("baseline_protocol_logit_delta") for row in rows if row.get("baseline_protocol_logit_delta") is not None]
    return {
        "rows": len(rows),
        "protocol_logit_reduced": sum(1 for row in rows if row.get("protocol_logit_reduced")),
        "protocol_logit_reduced_strong": sum(1 for row in rows if row.get("protocol_logit_reduced_strong")),
        "protocol_rank1_removed": sum(1 for row in rows if row.get("protocol_rank1_removed")),
        "next_top_changed": sum(1 for row in rows if row.get("next_top_changed")),
        "stop_rank_improved": sum(1 for row in rows if row.get("stop_rank_improved")),
        "mean_protocol_logit_delta": mean([float(value) for value in deltas]),
        "median_protocol_logit_delta": median([float(value) for value in deltas]),
        "baseline_next_top_categories": dict(sorted(Counter(str(row.get("baseline_next_top_category")) for row in rows).items())),
        "patched_next_top_categories": dict(sorted(Counter(str(row.get("patched_next_top_category")) for row in rows).items())),
        "category_transitions": dict(sorted(Counter(str(row.get("category_transition")) for row in rows if row.get("next_top_changed")).items())),
        "protocol_category_transitions": dict(
            sorted(Counter(str(row.get("protocol_category_transition")) for row in rows if row.get("protocol_category_transition")).items())
        ),
    }


def top_component_summaries(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("model")),
                int(row.get("layer_idx")),
                str(row.get("component_kind")),
                str(row.get("baseline_protocol_best_category")),
            )
        ].append(row)
    out = []
    for (model_name, layer_idx, component_kind, category), vals in grouped.items():
        summary = summarize_component_rows(vals)
        out.append(
            {
                "model": model_name,
                "layer_idx": layer_idx,
                "component_kind": component_kind,
                "baseline_protocol_best_category": category,
                **summary,
            }
        )
    out.sort(
        key=lambda row: (
            row.get("protocol_logit_reduced_strong") or 0,
            row.get("protocol_rank1_removed") or 0,
            -(row.get("mean_protocol_logit_delta") or 0.0),
            row.get("protocol_logit_reduced") or 0,
        ),
        reverse=True,
    )
    return out[: int(limit)]


def summarize_model(
    model_name: str,
    state_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    attn_impl: str | None,
    sampled_layers: list[int],
) -> dict[str, Any]:
    by_kind = defaultdict(list)
    by_category = defaultdict(list)
    for row in component_rows:
        by_kind[str(row.get("component_kind"))].append(row)
        by_category[str(row.get("baseline_protocol_best_category"))].append(row)
    overall_components = summarize_component_rows(component_rows)
    if overall_components["protocol_rank1_removed"] > 0 or overall_components["protocol_logit_reduced_strong"] > 0:
        evidence_label = "protocol_field_has_layer_component_sources"
    elif overall_components["protocol_logit_reduced"] > 0:
        evidence_label = "protocol_field_has_weak_component_sources"
    else:
        evidence_label = "protocol_field_not_mapped_by_component_zero"
    return {
        "phase": PHASE,
        "title": "Protocol Continuation Field Mapping",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": len(selected_rows),
        "sampled_layers": sampled_layers,
        "state_summary": summarize_state_rows(state_rows),
        "component_summary": overall_components,
        "component_kind_summaries": {kind: summarize_component_rows(vals) for kind, vals in sorted(by_kind.items())},
        "protocol_category_summaries": {category: summarize_component_rows(vals) for category, vals in sorted(by_category.items())},
        "top_components": top_component_summaries(component_rows, 20),
        "evidence_label": evidence_label,
        "boundary": (
            "Phase903 maps prompt/answer protocol priors and layer-level attention/MLP component sources for protocol tokens. "
            "It is a field-mapping audit, not a clean protocol closure intervention."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = selected_phase899_rows(args.model, args)
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
        }
        p846.write_json(out_dir / f"phase903_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase903_{args.model}_state_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase903_{args.model}_component_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    state_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    attn_impl = None
    sampled_layers: list[int] = []
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = protocol_category_groups(tokenizer)
        sampled_layers = layer_indices(model, int(args.layer_stride))
        component_kinds = [part.strip() for part in str(args.component_kinds).split(",") if part.strip()]
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(selected_rows, 1):
            case = case_map.get(str(source_row.get("case_id")))
            if not case:
                continue
            prompt_key = (str(source_row.get("case_id")), str(source_row.get("prompt_variant")))
            if prompt_key not in prompt_cache:
                prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
                prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
            prompt_ids = prompt_cache[prompt_key]
            gears = parse_gears(str(source_row.get("source_subset_key")))
            source_mode = str(source_row.get("edit_mode"))
            _prefix_logits, prefix_ids, prefix_text, answer_seen = p901.logits_after_answer_prefix(
                model,
                tokenizer,
                device,
                prompt_ids,
                gears,
                source_mode,
                case,
                int(args.max_prefix_tokens),
                float(args.scale_up_factor),
            )
            current_ids = [int(x) for x in prompt_ids] + [int(x) for x in prefix_ids]
            prompt_logits = logits_plain(model, device, prompt_ids)
            answer_logits = logits_plain(model, device, current_ids)
            prompt_metrics = state_metrics(tokenizer, prompt_logits, groups)
            answer_metrics = state_metrics(tokenizer, answer_logits, groups)
            state_rows.append(make_state_row(source_row, prefix_ids, prefix_text, answer_seen, prompt_metrics, answer_metrics))
            for layer_idx in sampled_layers:
                for component_kind in component_kinds:
                    patched_logits = logits_with_component_zero(model, device, current_ids, int(layer_idx), component_kind)
                    if patched_logits is None:
                        continue
                    patched_metrics = state_metrics(tokenizer, patched_logits, groups)
                    component_rows.append(
                        make_component_row(
                            tokenizer,
                            source_row,
                            int(layer_idx),
                            component_kind,
                            answer_metrics,
                            patched_metrics,
                            patched_logits,
                            answer_logits,
                        )
                    )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(
                    f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} "
                    f"state_rows={len(state_rows)} component_rows={len(component_rows)}"
                )
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, state_rows, component_rows, selected_rows, attn_impl, sampled_layers)
    p846.write_json(out_dir / f"phase903_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase903_{args.model}_state_rows.jsonl", state_rows)
    p846.write_jsonl(out_dir / f"phase903_{args.model}_component_rows.jsonl", component_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "state_summary": payload["state_summary"],
                "component_summary": payload["component_summary"],
                "evidence_label": payload["evidence_label"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def merge_counter_dicts(rows: list[dict[str, int]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for key, value in (row or {}).items():
            counter[str(key)] += int(value)
    return dict(sorted(counter.items()))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 903 protocol continuation field mapping",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## State Priors", ""])
    lines.append("| model | rows | answer top categories | answer protocol categories | protocol top1 | stop top1 | stop top10 | median rank delta |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for summary in payload.get("model_summaries") or []:
        state = summary.get("state_summary") or {}
        lines.append(
            "| {model} | {rows} | `{answer_top}` | `{answer_protocol}` | {protocol_top1} | {stop_top1} | {stop_top10} | {delta} |".format(
                model=summary.get("model"),
                rows=state.get("rows"),
                answer_top=json.dumps(state.get("answer_next_top_categories") or {}, ensure_ascii=False),
                answer_protocol=json.dumps(state.get("answer_protocol_best_categories") or {}, ensure_ascii=False),
                protocol_top1=state.get("answer_protocol_top1"),
                stop_top1=state.get("answer_stop_top1"),
                stop_top10=state.get("answer_stop_top10"),
                delta=state.get("median_protocol_rank_delta_answer_minus_prompt"),
            )
        )
    lines.extend(["", "## Component Summaries", ""])
    lines.append("| model | component rows | reduced | strong reduced | rank1 removed | top changed | stop improved | evidence |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        comp = summary.get("component_summary") or {}
        lines.append(
            "| {model} | {rows} | {reduced} | {strong} | {removed} | {changed} | {stop} | {evidence} |".format(
                model=summary.get("model"),
                rows=comp.get("rows"),
                reduced=comp.get("protocol_logit_reduced"),
                strong=comp.get("protocol_logit_reduced_strong"),
                removed=comp.get("protocol_rank1_removed"),
                changed=comp.get("next_top_changed"),
                stop=comp.get("stop_rank_improved"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Components", ""])
    lines.append("| model | layer | kind | category | rows | strong | removed | mean delta | top changed | stop improved |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_components") or []:
        lines.append(
            "| {model} | {layer_idx} | {component_kind} | {baseline_protocol_best_category} | {rows} | "
            "{protocol_logit_reduced_strong} | {protocol_rank1_removed} | {mean_protocol_logit_delta} | "
            "{next_top_changed} | {stop_rank_improved} |".format(**row)
        )
    lines.extend(["", "## Substitution Graph", ""])
    lines.append("```json")
    lines.append(json.dumps(payload.get("category_transitions") or {}, ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    state_summaries = []
    component_summaries = []
    top_components = []
    evidence = Counter()
    for model_name in MODELS:
        summary_path = out_dir / f"phase903_{model_name}_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        state_summaries.append(summary.get("state_summary") or {})
        component_summaries.append(summary.get("component_summary") or {})
        for row in summary.get("top_components") or []:
            top_components.append(row)
    scalar = Counter()
    for summary in summaries:
        scalar["selected_answer_drift_rows"] += int(summary.get("selected_answer_drift_rows") or 0)
        scalar["state_rows"] += int((summary.get("state_summary") or {}).get("rows") or 0)
        component = summary.get("component_summary") or {}
        for key in [
            "rows",
            "protocol_logit_reduced",
            "protocol_logit_reduced_strong",
            "protocol_rank1_removed",
            "next_top_changed",
            "stop_rank_improved",
        ]:
            scalar[f"component_{key}"] += int(component.get(key) or 0)
    top_components.sort(
        key=lambda row: (
            row.get("protocol_logit_reduced_strong") or 0,
            row.get("protocol_rank1_removed") or 0,
            -(row.get("mean_protocol_logit_delta") or 0.0),
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "state_answer_next_top_categories": merge_counter_dicts(
            [summary.get("answer_next_top_categories") or {} for summary in state_summaries]
        ),
        "state_answer_protocol_best_categories": merge_counter_dicts(
            [summary.get("answer_protocol_best_categories") or {} for summary in state_summaries]
        ),
        "category_transitions": merge_counter_dicts(
            [summary.get("category_transitions") or {} for summary in component_summaries]
        ),
        "protocol_category_transitions": merge_counter_dicts(
            [summary.get("protocol_category_transitions") or {} for summary in component_summaries]
        ),
        "model_summaries": summaries,
        "top_components": top_components[:40],
    }
    p846.write_json(out_dir / "phase903_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase903_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="protocol_continuation_field_mapping")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--component-kinds", default="attention,mlp")
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
