#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase834_blocker_aware_internal_route_boundary_predictor as p834  # noqa: E402
import phase837_global_gear_response_atlas_pilot as p837  # noqa: E402
import phase838_gear_response_decomposition_prediction as p838  # noqa: E402
import phase842_negative_mlp_gear_channel_decomposition as p842  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 843
SOURCE_842 = Path("tests/result/phase842_negative_mlp_gear_channel_decomposition/confirm")
RESULT_ROOT = Path("tests/result/phase843_core_channel_natural_route_validation")
TARGET_CLASS = "target_equivalent"


def log(msg: str) -> None:
    p837.log(msg)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    return p838.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    return int(ids[0]) if ids else None


def rank_of(logits: torch.Tensor, token_id: int | None) -> int | None:
    if token_id is None:
        return None
    if token_id < 0 or token_id >= int(logits.numel()):
        return None
    score = float(logits[int(token_id)].item())
    return int((logits > score).sum().item()) + 1


def logit_of(logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None:
        return None
    if token_id < 0 or token_id >= int(logits.numel()):
        return None
    return float(logits[int(token_id)].item())


def core_candidates(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    summary_path = SOURCE_842 / f"phase842_{model_name}_summary.json"
    if not summary_path.exists():
        return []
    summary = read_json(summary_path)
    records = []
    for rec in summary.get("channel_records") or []:
        if int(rec.get("single_target_rows", 0)) <= 0:
            continue
        if int(rec.get("leave_one_out_loss_rows", 0)) <= 0:
            continue
        if int(rec.get("flip_one_loss_rows", 0)) <= 0:
            continue
        records.append(dict(rec))
    records.sort(
        key=lambda r: (
            int(r.get("leave_one_out_loss_rows", 0)),
            int(r.get("flip_one_loss_rows", 0)),
            int(r.get("single_target_rows", 0)),
            -abs(finite(r.get("mean_delta_quality_vs_full"))),
        ),
        reverse=True,
    )
    if int(args.max_core_channels) > 0:
        records = records[: int(args.max_core_channels)]
    return records


def channel_group_source(model_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    labels = p842.negative_role_labels(model_name, argparse.Namespace(max_negative_components=1))
    if not labels:
        return None, None, None
    label = labels[0]
    groups = p842.p839.load_phase837_groups(model_name)
    group = groups.get(label)
    if not group:
        return label, None, None
    source = p842.p837.source_row_for_group(
        model_name,
        group,
        argparse.Namespace(
            source_round="confirm",
            max_source_rows=0,
            component_kinds="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group",
        ),
    )
    return {"label": label}, group, source


def synthetic_case(object_name: str) -> dict[str, Any]:
    return {
        "case_id": f"p843_{object_name}_geometric_shape",
        "object": object_name,
        "question": f"Which category best describes a {object_name}?",
        "answer": "geometric shape",
        "contrast_answer": "living thing",
        "distractors": ["hand tool", "public transport", "musical instrument", "warm color"],
        "synthetic_case": True,
    }


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cmap = p828.p825.case_map()
    out: list[dict[str, Any]] = []
    if args.case_scope in {"candidate", "candidate_plus_geometry", "candidate_plus_controls"}:
        case = cmap.get("p816_triangle_geometric_shape")
        if case:
            out.append({**case, "synthetic_case": False})
    if args.case_scope in {"geometry", "candidate_plus_geometry", "candidate_plus_controls"}:
        existing = {str(case.get("object")) for case in out}
        for obj in parse_csv(args.geometry_objects):
            if obj and obj not in existing:
                out.append(synthetic_case(obj))
    if args.case_scope == "candidate_plus_controls":
        for case_id in parse_csv(args.control_case_ids):
            case = cmap.get(case_id)
            if case:
                out.append({**case, "synthetic_case": False, "control_case": True})
    if 0 < int(args.max_cases) < len(out):
        out = out[: int(args.max_cases)]
    return out


def prompt_for_case(case: dict[str, Any], variant: str) -> str:
    if not case.get("synthetic_case"):
        return p828.p825.natural_prompt(case, variant)
    if variant == "natural_category":
        return (
            "Give a concise category phrase.\n"
            "Do not list choices or explain.\n"
            f"Item: {case['object']}\n"
            "Category:"
        )
    if variant == "natural_question":
        return (
            "Answer with only a short category phrase.\n"
            f"{case['question']}\n"
            "Category:"
        )
    if variant == "object_only":
        return (
            "Write the best short category phrase for the item.\n"
            f"Item: {case['object']}\n"
            "Phrase:"
        )
    if variant in {"no_choices", "exact_choices"}:
        return (
            f"Question: {case['question']}\n"
            "Answer with the best short category phrase.\n"
            "Answer:"
        )
    raise ValueError(f"unknown prompt variant: {variant}")


def classify_output(case: dict[str, Any], text: str, standards: list[dict[str, Any]]) -> dict[str, Any]:
    cleaned = p828.p825.clean_generated(text)
    if not case.get("synthetic_case") and not case.get("control_case"):
        lookup = p828.p820.standard_lookup(standards)
        return p828.p825.boundary_for(lookup, case["case_id"], cleaned)
    low = cleaned.lower().strip()
    obj = str(case.get("object", "")).lower()
    ans = str(case.get("answer", "")).lower()
    if low.startswith(ans) or low.startswith("polygon") or low.startswith("shape"):
        cls = TARGET_CLASS
        rank = 4
    elif obj and low.startswith(obj):
        cls = "object_echo"
        rank = 1
    elif not low:
        cls = "unknown_other"
        rank = 0
    else:
        cls = "unknown_other"
        rank = 0
    return {
        "final_boundary_class": cls,
        "boundary_rank": rank,
        "protocol_valid": True,
        "cleaned": cleaned,
    }


def install_channel_edit(model, layer_idx: int, channel_id: int, mode: str):
    layer = get_layers(model)[int(layer_idx)]

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        patched = inputs[0].clone()
        idx = int(channel_id)
        if idx < 0 or idx >= int(patched.shape[-1]):
            return inputs
        if mode == "zero":
            patched[:, -1, idx] = 0
        elif mode == "flip":
            patched[:, -1, idx] = -patched[:, -1, idx]
        elif mode == "half":
            patched[:, -1, idx] = patched[:, -1, idx] * 0.5
        else:
            raise ValueError(f"unknown channel edit mode: {mode}")
        return (patched, *inputs[1:])

    return layer.mlp.down_proj.register_forward_pre_hook(hook)


def first_logits_with_edit(
    model,
    device: torch.device,
    prompt_ids: list[int],
    layer_idx: int,
    channel_id: int,
    mode: str | None,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handle = None
    if mode and mode != "original":
        handle = install_channel_edit(model, layer_idx, channel_id, mode)
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        if handle is not None:
            handle.remove()


def greedy_with_edit(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    max_new_tokens: int,
    layer_idx: int,
    channel_id: int,
    mode: str | None,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handle = None
        if step == 0 and mode and mode != "original":
            handle = install_channel_edit(model, layer_idx, channel_id, mode)
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        finally:
            if handle is not None:
                handle.remove()
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def token_scores(tokenizer, logits: torch.Tensor, case: dict[str, Any], baseline_id: int | None) -> dict[str, Any]:
    target_id = first_token_id(tokenizer, str(case.get("answer", "")))
    polygon_id = first_token_id(tokenizer, "polygon")
    object_id = first_token_id(tokenizer, str(case.get("object", "")))
    target_ids = [x for x in [target_id, polygon_id] if x is not None]
    best_target_logit = max([logit_of(logits, x) for x in target_ids if logit_of(logits, x) is not None] or [None])
    target_ranks = [rank_of(logits, x) for x in target_ids]
    target_ranks = [x for x in target_ranks if x is not None]
    top = torch.topk(logits, k=min(5, int(logits.numel())))
    return {
        "target_token_id": target_id,
        "target_token": tokenizer.decode([int(target_id)]) if target_id is not None else None,
        "polygon_token_id": polygon_id,
        "polygon_token": tokenizer.decode([int(polygon_id)]) if polygon_id is not None else None,
        "object_token_id": object_id,
        "object_token": tokenizer.decode([int(object_id)]) if object_id is not None else None,
        "baseline_token_id": baseline_id,
        "baseline_token": tokenizer.decode([int(baseline_id)]) if baseline_id is not None else None,
        "best_target_rank": min(target_ranks) if target_ranks else None,
        "object_rank": rank_of(logits, object_id),
        "baseline_rank": rank_of(logits, baseline_id),
        "best_target_logit": best_target_logit,
        "object_logit": logit_of(logits, object_id),
        "target_minus_object_logit": (
            None
            if best_target_logit is None or logit_of(logits, object_id) is None
            else float(best_target_logit - logit_of(logits, object_id))
        ),
        "top_token_ids": [int(x) for x in top.indices.tolist()],
        "top_tokens": [tokenizer.decode([int(x)]) for x in top.indices.tolist()],
        "top_logits": [float(x) for x in top.values.tolist()],
    }


def readout_coefficients(model, tokenizer, case: dict[str, Any], layer_idx: int, channel_id: int, baseline_id: int | None) -> dict[str, Any]:
    weight = p828.p823.lm_head_weight(model).detach().float().cpu()
    layer = get_layers(model)[int(layer_idx)]
    if not hasattr(layer.mlp, "down_proj"):
        return {}
    col = layer.mlp.down_proj.weight.detach().float().cpu()[:, int(channel_id)]
    target_id = first_token_id(tokenizer, str(case.get("answer", "")))
    polygon_id = first_token_id(tokenizer, "polygon")
    object_id = first_token_id(tokenizer, str(case.get("object", "")))

    def coeff(a: int | None, b: int | None) -> float | None:
        if a is None or b is None:
            return None
        if a < 0 or b < 0 or a >= int(weight.shape[0]) or b >= int(weight.shape[0]):
            return None
        return float(torch.dot(weight[int(a)] - weight[int(b)], col).item())

    return {
        "readout_coeff_target_minus_object": coeff(target_id, object_id),
        "readout_coeff_polygon_minus_object": coeff(polygon_id, object_id),
        "readout_coeff_target_minus_baseline": coeff(target_id, baseline_id),
        "readout_coeff_polygon_minus_baseline": coeff(polygon_id, baseline_id),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    core = core_candidates(args.model, args)
    source_meta, group, source = channel_group_source(args.model)
    cases = selected_cases(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "core_candidates": core,
                    "group": group,
                    "source": bool(source),
                    "cases": [case["case_id"] for case in cases],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {}
    if not core or not group or source is None or not cases:
        summary = summarize_rows([], args, None, core, cases, skipped=True)
        summary["skip_reason"] = "no Phase 842 core channel candidate" if not core else "missing group/source/cases"
        p828.write_jsonl(out_dir / f"phase843_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase843_{args.model}_summary.json", summary)
        print(
            json.dumps(
                {"model": args.model, "round": args.round_name, "rows": 0, "skipped_model_load": True, "skip_reason": summary["skip_reason"]},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return summary

    core_rec = core[0]
    layer_idx = int(group["layer_idx"])
    channel_id = int(core_rec["channel_id"])
    channel_local_index = int(core_rec["channel_local_index"])
    log(f"{args.model}/{args.round_name}: core channel L{layer_idx} local={channel_local_index} global={channel_id} cases={len(cases)}")

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    standards = p828.p820.standard_rows()
    rows: list[dict[str, Any]] = []
    modes = parse_csv(args.edit_modes)
    try:
        for case_idx, case in enumerate(cases, 1):
            for variant in parse_csv(args.prompt_variants):
                prompt = prompt_for_case(case, variant)
                prompt_ids = p828.p823.encode_prompt(tokenizer, prompt)
                natural_state = p828.p822.capture_component_state(model, tokenizer, device, prompt, layer_idx)
                mlp_down = natural_state.get("mlp_down_input")
                channel_activation = None
                if mlp_down is not None and 0 <= channel_id < int(mlp_down.numel()):
                    channel_activation = float(mlp_down[int(channel_id)].item())
                original_text, original_ids = greedy_with_edit(
                    model, tokenizer, device, prompt_ids, int(args.max_new_tokens), layer_idx, channel_id, "original"
                )
                original_boundary = classify_output(case, original_text, standards)
                baseline_id = int(original_ids[0]) if original_ids else None
                coeffs = readout_coefficients(model, tokenizer, case, layer_idx, channel_id, baseline_id)
                for mode in modes:
                    logits = first_logits_with_edit(model, device, prompt_ids, layer_idx, channel_id, mode)
                    generated, token_ids = greedy_with_edit(
                        model, tokenizer, device, prompt_ids, int(args.max_new_tokens), layer_idx, channel_id, mode
                    )
                    boundary = classify_output(case, generated, standards)
                    scores = token_scores(tokenizer, logits, case, baseline_id)
                    row = {
                        "row_kind": "phase843_core_channel_natural_route_validation",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "case_id": case["case_id"],
                        "object": case.get("object"),
                        "target_answer": case.get("answer"),
                        "synthetic_case": bool(case.get("synthetic_case")),
                        "control_case": bool(case.get("control_case")),
                        "prompt_variant": variant,
                        "prompt": prompt,
                        "layer_idx": layer_idx,
                        "channel_local_index": channel_local_index,
                        "channel_id": channel_id,
                        "edit_mode": mode,
                        "channel_activation_original": channel_activation,
                        "channel_activation_abs": None if channel_activation is None else abs(channel_activation),
                        "channel_activation_sign": 0
                        if channel_activation is None or channel_activation == 0
                        else (1 if channel_activation > 0 else -1),
                        "original_generated": p828.p825.clean_generated(original_text),
                        "original_boundary_class": original_boundary.get("final_boundary_class"),
                        "original_boundary_rank": int(original_boundary.get("boundary_rank", 0)),
                        "original_target_transition": original_boundary.get("final_boundary_class") == TARGET_CLASS,
                        "generated": p828.p825.clean_generated(generated),
                        "token_ids": token_ids,
                        "boundary_class": boundary.get("final_boundary_class"),
                        "boundary_rank": int(boundary.get("boundary_rank", 0)),
                        "target_transition": boundary.get("final_boundary_class") == TARGET_CLASS,
                        "target_lost_vs_original": bool(
                            original_boundary.get("final_boundary_class") == TARGET_CLASS
                            and boundary.get("final_boundary_class") != TARGET_CLASS
                        ),
                        "target_gained_vs_original": bool(
                            original_boundary.get("final_boundary_class") != TARGET_CLASS
                            and boundary.get("final_boundary_class") == TARGET_CLASS
                        ),
                        "delta_boundary_rank_vs_original": int(boundary.get("boundary_rank", 0))
                        - int(original_boundary.get("boundary_rank", 0)),
                        **coeffs,
                        **scores,
                    }
                    act = row.get("channel_activation_original")
                    for key in [
                        "readout_coeff_target_minus_object",
                        "readout_coeff_polygon_minus_object",
                        "readout_coeff_target_minus_baseline",
                        "readout_coeff_polygon_minus_baseline",
                    ]:
                        row[f"activation_times_{key}"] = None if act is None or row.get(key) is None else float(act) * finite(row.get(key))
                    rows.append(row)
            if case_idx % int(args.log_every) == 0 or case_idx == len(cases):
                log(f"{args.model}: evaluated cases {case_idx}/{len(cases)} rows={len(rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, args, attn_impl, core, cases, skipped=False)
    p828.write_jsonl(out_dir / f"phase843_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase843_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "rows": summary["n_rows"],
                "original_target_rows": summary["original_target_rows"],
                "target_lost_vs_original_rows": summary["target_lost_vs_original_rows"],
                "target_gained_vs_original_rows": summary["target_gained_vs_original_rows"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def avg(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def summarize_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    attn_impl: str | None,
    core: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    skipped: bool,
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_rows = [row for row in rows if row.get("edit_mode") == "original"]
    for row in rows:
        by_mode[str(row.get("edit_mode"))].append(row)
        by_object[str(row.get("object"))].append(row)

    def compact(vals: list[dict[str, Any]]) -> dict[str, Any]:
        activations = [finite(row.get("channel_activation_original")) for row in vals if row.get("channel_activation_original") is not None]
        return {
            "n": len(vals),
            "target_rows": sum(1 for row in vals if row.get("target_transition")),
            "target_lost_vs_original_rows": sum(1 for row in vals if row.get("target_lost_vs_original")),
            "target_gained_vs_original_rows": sum(1 for row in vals if row.get("target_gained_vs_original")),
            "object_echo_rows": sum(1 for row in vals if row.get("boundary_class") == "object_echo"),
            "unknown_other_rows": sum(1 for row in vals if row.get("boundary_class") == "unknown_other"),
            "mean_channel_activation": avg(activations),
            "mean_abs_channel_activation": avg([abs(x) for x in activations]),
            "mean_target_minus_object_logit": avg(
                [finite(row.get("target_minus_object_logit")) for row in vals if row.get("target_minus_object_logit") is not None]
            ),
            "classes": dict(Counter(str(row.get("boundary_class")) for row in vals)),
        }

    top_rows = sorted(
        rows,
        key=lambda row: (
            int(bool(row.get("target_lost_vs_original"))),
            int(bool(row.get("target_gained_vs_original"))),
            abs(finite(row.get("target_minus_object_logit"))),
            abs(finite(row.get("channel_activation_original"))),
        ),
        reverse=True,
    )[:80]
    return {
        "phase": PHASE,
        "title": "Core Channel Natural Route Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "skipped_model_load": skipped,
        "n_rows": len(rows),
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "core_candidates": core,
        "prompt_variants": parse_csv(args.prompt_variants),
        "edit_modes": parse_csv(args.edit_modes),
        "original_target_rows": sum(1 for row in original_rows if row.get("target_transition")),
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "target_lost_vs_original_rows": sum(1 for row in rows if row.get("target_lost_vs_original")),
        "target_gained_vs_original_rows": sum(1 for row in rows if row.get("target_gained_vs_original")),
        "mode_summary": {mode: compact(vals) for mode, vals in sorted(by_mode.items())},
        "object_summary": {obj: compact(vals) for obj, vals in sorted(by_object.items())},
        "top_rows": [
            {
                "case_id": row.get("case_id"),
                "object": row.get("object"),
                "prompt_variant": row.get("prompt_variant"),
                "edit_mode": row.get("edit_mode"),
                "channel_activation_original": row.get("channel_activation_original"),
                "generated": row.get("generated"),
                "boundary_class": row.get("boundary_class"),
                "original_generated": row.get("original_generated"),
                "original_boundary_class": row.get("original_boundary_class"),
                "target_lost_vs_original": bool(row.get("target_lost_vs_original")),
                "target_gained_vs_original": bool(row.get("target_gained_vs_original")),
                "target_minus_object_logit": row.get("target_minus_object_logit"),
                "best_target_rank": row.get("best_target_rank"),
                "object_rank": row.get("object_rank"),
                "top_tokens": row.get("top_tokens"),
            }
            for row in top_rows
        ],
        "boundary": (
            "This phase audits natural activation and first-step natural channel edits for the Phase 842 core channel. "
            "It still does not prove global geometry reuse or full token closure."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 843 Core Channel Natural Route Validation ({payload['round']})",
        "",
        "- Source: Phase 842 core channel candidate.",
        "- Boundary: natural activation + first-step channel edit; not global closure.",
        "",
        "## Model Summary",
        "",
        "| model | skipped | rows | cases | original target | target | lost vs original | gained vs original |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {int(bool(data.get('skipped_model_load')))} | {data.get('n_rows', 0)} | "
            f"{data.get('n_cases', 0)} | {data.get('original_target_rows', 0)} | {data.get('target_rows', 0)} | "
            f"{data.get('target_lost_vs_original_rows', 0)} | {data.get('target_gained_vs_original_rows', 0)} |"
        )
    lines += ["", "## Mode Summary", ""]
    lines += ["| model | mode | n | target | lost | gained | object_echo | unknown | mean act | mean target-object | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for mode, row in (data.get("mode_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{mode}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {row.get('target_gained_vs_original_rows', 0)} | "
                f"{row.get('object_echo_rows', 0)} | {row.get('unknown_other_rows', 0)} | "
                f"{fmt(row.get('mean_channel_activation'))} | {fmt(row.get('mean_target_minus_object_logit'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Object Summary", ""]
    lines += ["| model | object | n | target | lost | gained | mean act | mean abs act | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for obj, row in (data.get("object_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{obj}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {row.get('target_gained_vs_original_rows', 0)} | "
                f"{fmt(row.get('mean_channel_activation'))} | {fmt(row.get('mean_abs_channel_activation'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Top Rows", ""]
    lines += ["| model | object | prompt | mode | act | class | output | orig class | lost | gained | target-object | target rank | object rank | top tokens |"]
    lines += ["|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("top_rows") or []:
            output = str(row.get("generated") or "").replace("|", "/")[:48]
            top_tokens = json.dumps(row.get("top_tokens") or [], ensure_ascii=False).replace("|", "/")[:80]
            lines.append(
                f"| {model_name} | `{row.get('object')}` | `{row.get('prompt_variant')}` | `{row.get('edit_mode')}` | "
                f"{fmt(row.get('channel_activation_original'))} | `{row.get('boundary_class')}` | {output} | "
                f"`{row.get('original_boundary_class')}` | {int(bool(row.get('target_lost_vs_original')))} | "
                f"{int(bool(row.get('target_gained_vs_original')))} | {fmt(row.get('target_minus_object_logit'))} | "
                f"{row.get('best_target_rank')} | {row.get('object_rank')} | `{top_tokens}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_summaries": {},
        "models": [],
    }
    for model_name in p828.MODELS:
        path = out_dir / f"phase843_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase843_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase843_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p828.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-core-channels", type=int, default=1)
    parser.add_argument("--case-scope", choices=["candidate", "geometry", "candidate_plus_geometry", "candidate_plus_controls"], default="candidate")
    parser.add_argument("--geometry-objects", default="triangle,square,rectangle,circle,polygon")
    parser.add_argument("--control-case-ids", default="p816_cat_living_thing,p816_hammer_hand_tool")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--prompt-variants", default="natural_question")
    parser.add_argument("--edit-modes", default="original,zero,flip")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_only:
        print(json.dumps(summarize_round(args.round_name), ensure_ascii=False, indent=2), flush=True)
        return
    eval_model(args)


if __name__ == "__main__":
    main()
