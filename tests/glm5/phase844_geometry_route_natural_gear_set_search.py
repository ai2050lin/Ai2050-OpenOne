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
import phase837_global_gear_response_atlas_pilot as p837  # noqa: E402
import phase838_gear_response_decomposition_prediction as p838  # noqa: E402
import phase843_core_channel_natural_route_validation as p843  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 844
RESULT_ROOT = Path("tests/result/phase844_geometry_route_natural_gear_set_search")
TARGET_CLASS = "target_equivalent"


def log(msg: str) -> None:
    p837.log(msg)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x) for x in parse_csv(text)]


def finite(value: Any, default: float = 0.0) -> float:
    return p838.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_case(object_name: str) -> dict[str, Any]:
    return {
        "case_id": f"p844_{object_name}_geometric_shape",
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
    if args.include_seed_triangle:
        case = cmap.get("p816_triangle_geometric_shape")
        if case:
            out.append({**case, "synthetic_case": False})
    existing = {str(case.get("object")) for case in out}
    for obj in parse_csv(args.geometry_objects):
        if obj and obj not in existing:
            out.append(synthetic_case(obj))
            existing.add(obj)
    if 0 < int(args.max_cases) < len(out):
        out = out[: int(args.max_cases)]
    return out


def prompt_for_case(case: dict[str, Any], variant: str) -> str:
    return p843.prompt_for_case(case, variant)


def classify_output(case: dict[str, Any], text: str, standards: list[dict[str, Any]]) -> dict[str, Any]:
    return p843.classify_output(case, text, standards)


def first_token_id(tokenizer, text: str) -> int | None:
    return p843.first_token_id(tokenizer, text)


def rank_of(logits: torch.Tensor, token_id: int | None) -> int | None:
    return p843.rank_of(logits, token_id)


def logit_of(logits: torch.Tensor, token_id: int | None) -> float | None:
    return p843.logit_of(logits, token_id)


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]


def capture_mlp_down_inputs(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    ids = encode_prompt(tokenizer, prompt)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                captured[int(layer_idx)] = inputs[0][0, answer_pos].detach().float().cpu()

        return hook

    for layer_idx in layer_indices:
        if 0 <= int(layer_idx) < len(layers) and hasattr(layers[int(layer_idx)].mlp, "down_proj"):
            handles.append(layers[int(layer_idx)].mlp.down_proj.register_forward_pre_hook(make_hook(int(layer_idx))))
    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def readout_vectors_for_case(model, tokenizer, case: dict[str, Any]) -> dict[str, Any]:
    weight = p828.p823.lm_head_weight(model).detach().float().cpu()
    target_id = first_token_id(tokenizer, str(case.get("answer", "")))
    polygon_id = first_token_id(tokenizer, "polygon")
    object_id = first_token_id(tokenizer, str(case.get("object", "")))

    def vec(a: int | None, b: int | None) -> torch.Tensor | None:
        if a is None or b is None:
            return None
        if a < 0 or b < 0 or a >= int(weight.shape[0]) or b >= int(weight.shape[0]):
            return None
        return (weight[int(a)] - weight[int(b)]).float()

    return {
        "target_id": target_id,
        "polygon_id": polygon_id,
        "object_id": object_id,
        "target_minus_object": vec(target_id, object_id),
        "polygon_minus_object": vec(polygon_id, object_id),
    }


def down_coeffs(layer, readouts: dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not hasattr(layer.mlp, "down_proj"):
        return None, None
    w = layer.mlp.down_proj.weight.detach().float().cpu()
    tmo = readouts.get("target_minus_object")
    pmo = readouts.get("polygon_minus_object")
    target_coeff = None if tmo is None else (tmo @ w).float()
    polygon_coeff = None if pmo is None else (pmo @ w).float()
    return target_coeff, polygon_coeff


def add_metric(bucket: dict[str, Any], activation: float, target_coeff: float | None, polygon_coeff: float | None) -> None:
    coeffs = [abs(x) for x in [target_coeff, polygon_coeff] if x is not None and math.isfinite(float(x))]
    best_abs_coeff = max(coeffs) if coeffs else 0.0
    signed_coeff = 0.0
    if polygon_coeff is not None and math.isfinite(float(polygon_coeff)):
        signed_coeff = float(polygon_coeff)
    elif target_coeff is not None and math.isfinite(float(target_coeff)):
        signed_coeff = float(target_coeff)
    support = float(activation) * float(signed_coeff)
    bucket["n"] += 1
    bucket["activation_sum"] += float(activation)
    bucket["abs_activation_sum"] += abs(float(activation))
    bucket["neg_count"] += 1 if activation < 0 else 0
    bucket["pos_count"] += 1 if activation > 0 else 0
    bucket["abs_coeff_sum"] += float(best_abs_coeff)
    bucket["signed_support_sum"] += float(support)
    bucket["abs_support_sum"] += abs(float(activation)) * float(best_abs_coeff)
    bucket["max_abs_support"] = max(float(bucket.get("max_abs_support", 0.0)), abs(float(activation)) * float(best_abs_coeff))


def collect_candidate_gears(
    model,
    tokenizer,
    device: torch.device,
    cases: list[dict[str, Any]],
    prompt_variants: list[str],
    layer_indices: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layers = get_layers(model)
    layer_indices = [int(i) for i in layer_indices if 0 <= int(i) < len(layers)]
    buckets: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "activation_sum": 0.0,
            "abs_activation_sum": 0.0,
            "neg_count": 0,
            "pos_count": 0,
            "abs_coeff_sum": 0.0,
            "signed_support_sum": 0.0,
            "abs_support_sum": 0.0,
            "max_abs_support": 0.0,
        }
    )
    sample_rows: list[dict[str, Any]] = []
    coeff_cache: dict[tuple[str, int], tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    for case in cases:
        readouts = readout_vectors_for_case(model, tokenizer, case)
        for layer_idx in layer_indices:
            layer = layers[int(layer_idx)]
            coeff_cache[(case["case_id"], int(layer_idx))] = down_coeffs(layer, readouts)
        for variant in prompt_variants:
            prompt = prompt_for_case(case, variant)
            captured = capture_mlp_down_inputs(model, tokenizer, device, prompt, layer_indices)
            for layer_idx, acts in captured.items():
                target_coeff, polygon_coeff = coeff_cache[(case["case_id"], int(layer_idx))]
                if target_coeff is None and polygon_coeff is None:
                    continue
                coeff_abs = torch.zeros_like(acts)
                if target_coeff is not None:
                    coeff_abs = torch.maximum(coeff_abs, target_coeff.abs())
                if polygon_coeff is not None:
                    coeff_abs = torch.maximum(coeff_abs, polygon_coeff.abs())
                score = acts.abs() * coeff_abs
                k = min(int(args.per_sample_topk), int(score.numel()))
                if k <= 0:
                    continue
                vals, ids = torch.topk(score, k)
                for val, channel in zip(vals.tolist(), ids.tolist(), strict=False):
                    c = int(channel)
                    tc = None if target_coeff is None else float(target_coeff[c].item())
                    pc = None if polygon_coeff is None else float(polygon_coeff[c].item())
                    activation = float(acts[c].item())
                    add_metric(buckets[(int(layer_idx), c)], activation, tc, pc)
                    sample_rows.append(
                        {
                            "case_id": case["case_id"],
                            "object": case.get("object"),
                            "prompt_variant": variant,
                            "layer_idx": int(layer_idx),
                            "channel_id": c,
                            "activation": activation,
                            "target_coeff": tc,
                            "polygon_coeff": pc,
                            "abs_support": float(val),
                        }
                    )
    gears: list[dict[str, Any]] = []
    for (layer_idx, channel_id), b in buckets.items():
        n = int(b["n"])
        if n < int(args.min_candidate_hits):
            continue
        neg_ratio = float(b["neg_count"]) / max(1, n)
        pos_ratio = float(b["pos_count"]) / max(1, n)
        sign_consistency = max(neg_ratio, pos_ratio)
        mean_abs_support = float(b["abs_support_sum"]) / max(1, n)
        mean_signed_support = float(b["signed_support_sum"]) / max(1, n)
        mean_abs_activation = float(b["abs_activation_sum"]) / max(1, n)
        mean_activation = float(b["activation_sum"]) / max(1, n)
        mean_abs_coeff = float(b["abs_coeff_sum"]) / max(1, n)
        gears.append(
            {
                "layer_idx": int(layer_idx),
                "channel_id": int(channel_id),
                "n": n,
                "mean_activation": mean_activation,
                "mean_abs_activation": mean_abs_activation,
                "neg_ratio": neg_ratio,
                "pos_ratio": pos_ratio,
                "sign_consistency": sign_consistency,
                "mean_abs_coeff": mean_abs_coeff,
                "mean_signed_support": mean_signed_support,
                "mean_abs_support": mean_abs_support,
                "max_abs_support": float(b.get("max_abs_support", 0.0)),
                "gear_score": mean_abs_support * (0.5 + 0.5 * sign_consistency) * math.log1p(n),
            }
        )
    gears.sort(key=lambda r: (finite(r.get("gear_score")), finite(r.get("mean_abs_support"))), reverse=True)
    if int(args.max_gears) > 0:
        gears = gears[: int(args.max_gears)]
    return gears, sample_rows


def install_multi_gear_edit(model, gears: list[dict[str, Any]], mode: str) -> list[Any]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for gear in gears:
        by_layer[int(gear["layer_idx"])].append(int(gear["channel_id"]))
    handles = []
    layers = get_layers(model)

    def make_hook(channels: list[int]):
        idx_cpu = torch.tensor([int(x) for x in channels], dtype=torch.long)

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]) or idx_cpu.numel() == 0:
                return inputs
            patched = inputs[0].clone()
            idx = idx_cpu.to(device=patched.device)
            valid = idx[(idx >= 0) & (idx < int(patched.shape[-1]))]
            if valid.numel() == 0:
                return inputs
            if mode == "zero":
                patched[:, -1, valid] = 0
            elif mode == "flip":
                patched[:, -1, valid] = -patched[:, -1, valid]
            elif mode == "half":
                patched[:, -1, valid] = patched[:, -1, valid] * 0.5
            else:
                raise ValueError(f"unknown gear edit mode: {mode}")
            return (patched, *inputs[1:])

        return hook

    for layer_idx, channels in by_layer.items():
        if 0 <= int(layer_idx) < len(layers) and hasattr(layers[int(layer_idx)].mlp, "down_proj"):
            handles.append(layers[int(layer_idx)].mlp.down_proj.register_forward_pre_hook(make_hook(channels)))
    return handles


def first_logits_with_gears(
    model,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    if mode != "original" and gears:
        handles = install_multi_gear_edit(model, gears, mode)
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def greedy_with_gears(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    mode: str,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles = []
        if step == 0 and mode != "original" and gears:
            handles = install_multi_gear_edit(model, gears, mode)
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        finally:
            for handle in handles:
                handle.remove()
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def gear_subsets(gears: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [{"subset_name": "original", "mode": "original", "gears": []}]
    sizes = []
    for size in parse_int_csv(args.subset_sizes):
        if size > 0 and size <= len(gears):
            sizes.append(size)
    seen: set[tuple[int, str]] = set()
    for size in sizes:
        selected = gears[:size]
        for mode in parse_csv(args.edit_modes):
            if mode == "original":
                continue
            key = (size, mode)
            if key in seen:
                continue
            seen.add(key)
            specs.append({"subset_name": f"top{size}_{mode}", "mode": mode, "gears": selected})
    return specs


def token_scores(tokenizer, logits: torch.Tensor, case: dict[str, Any], baseline_id: int | None) -> dict[str, Any]:
    return p843.token_scores(tokenizer, logits, case, baseline_id)


def eval_rows(
    model,
    tokenizer,
    device: torch.device,
    cases: list[dict[str, Any]],
    prompt_variants: list[str],
    gears: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    standards = p828.p820.standard_rows()
    rows: list[dict[str, Any]] = []
    specs = gear_subsets(gears, args)
    for case_idx, case in enumerate(cases, 1):
        for variant in prompt_variants:
            prompt = prompt_for_case(case, variant)
            prompt_ids = encode_prompt(tokenizer, prompt)
            original_text, original_ids = greedy_with_gears(
                model, tokenizer, device, prompt_ids, [], "original", int(args.max_new_tokens)
            )
            original_boundary = classify_output(case, original_text, standards)
            baseline_id = int(original_ids[0]) if original_ids else None
            for spec in specs:
                logits = first_logits_with_gears(model, device, prompt_ids, spec["gears"], spec["mode"])
                generated, token_ids = greedy_with_gears(
                    model,
                    tokenizer,
                    device,
                    prompt_ids,
                    spec["gears"],
                    spec["mode"],
                    int(args.max_new_tokens),
                )
                boundary = classify_output(case, generated, standards)
                scores = token_scores(tokenizer, logits, case, baseline_id)
                rows.append(
                    {
                        "row_kind": "phase844_geometry_route_natural_gear_set_search",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "case_id": case["case_id"],
                        "object": case.get("object"),
                        "target_answer": case.get("answer"),
                        "synthetic_case": bool(case.get("synthetic_case")),
                        "prompt_variant": variant,
                        "prompt": prompt,
                        "subset_name": spec["subset_name"],
                        "edit_mode": spec["mode"],
                        "gear_count": len(spec["gears"]),
                        "gear_keys": [f"L{g['layer_idx']}C{g['channel_id']}" for g in spec["gears"]],
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
                        **scores,
                    }
                )
        if case_idx % int(args.log_every) == 0 or case_idx == len(cases):
            log(f"{args.model}: evaluated cases {case_idx}/{len(cases)} rows={len(rows)}")
    return rows


def avg(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def compact(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "target_rows": sum(1 for row in vals if row.get("target_transition")),
        "target_lost_vs_original_rows": sum(1 for row in vals if row.get("target_lost_vs_original")),
        "target_gained_vs_original_rows": sum(1 for row in vals if row.get("target_gained_vs_original")),
        "object_echo_rows": sum(1 for row in vals if row.get("boundary_class") == "object_echo"),
        "unknown_other_rows": sum(1 for row in vals if row.get("boundary_class") == "unknown_other"),
        "mean_target_minus_object_logit": avg(
            [finite(row.get("target_minus_object_logit")) for row in vals if row.get("target_minus_object_logit") is not None]
        ),
        "classes": dict(Counter(str(row.get("boundary_class")) for row in vals)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    gears: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    attn_impl: str | None,
) -> dict[str, Any]:
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_rows = [row for row in rows if row.get("subset_name") == "original"]
    for row in rows:
        by_subset[str(row.get("subset_name"))].append(row)
        by_object[str(row.get("object"))].append(row)
    top_rows = sorted(
        rows,
        key=lambda row: (
            int(bool(row.get("target_lost_vs_original"))),
            int(bool(row.get("target_gained_vs_original"))),
            abs(finite(row.get("target_minus_object_logit"))),
        ),
        reverse=True,
    )[:80]
    return {
        "phase": PHASE,
        "title": "Geometry Route Natural Gear Set Search",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "prompt_variants": parse_csv(args.prompt_variants),
        "layer_indices": [int(x) for x in getattr(args, "_valid_layers", parse_int_csv(args.layers))],
        "n_candidate_sample_rows": len(sample_rows),
        "n_gears": len(gears),
        "top_gears": gears[: int(args.report_top_gears)],
        "n_rows": len(rows),
        "original_target_rows": sum(1 for row in original_rows if row.get("target_transition")),
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "target_lost_vs_original_rows": sum(1 for row in rows if row.get("target_lost_vs_original")),
        "target_gained_vs_original_rows": sum(1 for row in rows if row.get("target_gained_vs_original")),
        "subset_summary": {k: compact(v) for k, v in sorted(by_subset.items())},
        "object_summary": {k: compact(v) for k, v in sorted(by_object.items())},
        "top_rows": [
            {
                "case_id": row.get("case_id"),
                "object": row.get("object"),
                "prompt_variant": row.get("prompt_variant"),
                "subset_name": row.get("subset_name"),
                "edit_mode": row.get("edit_mode"),
                "gear_count": row.get("gear_count"),
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
                "gear_keys": row.get("gear_keys"),
            }
            for row in top_rows
        ],
        "boundary": (
            "This phase searches natural high-response/readout-coupled MLP gear sets for geometry route cases. "
            "It is a gear-set atlas probe, not a proof of global token closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(args)
    if args.dry_run:
        print(
            json.dumps(
                {"model": args.model, "cases": [case["case_id"] for case in cases], "layers": parse_int_csv(args.layers)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return {}
    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_variants = parse_csv(args.prompt_variants)
    requested_layers = parse_int_csv(args.layers)
    n_layers = len(get_layers(model))
    layer_indices = [int(i) for i in requested_layers if 0 <= int(i) < n_layers]
    if len(layer_indices) != len(requested_layers):
        log(f"{args.model}/{args.round_name}: filtered layers {requested_layers} -> {layer_indices} for n_layers={n_layers}")
    setattr(args, "_valid_layers", layer_indices)
    try:
        gears, sample_rows = collect_candidate_gears(model, tokenizer, device, cases, prompt_variants, layer_indices, args)
        log(f"{args.model}/{args.round_name}: candidate_gears={len(gears)} sample_rows={len(sample_rows)}")
        rows = eval_rows(model, tokenizer, device, cases, prompt_variants, gears, args)
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, gears, sample_rows, cases, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase844_{args.model}_candidate_samples.jsonl", sample_rows[: int(args.max_saved_samples)])
    p828.write_jsonl(out_dir / f"phase844_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase844_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "gears": summary["n_gears"],
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


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 844 Geometry Route Natural Gear Set Search ({payload['round']})",
        "",
        "- Search: natural MLP down-input channel activation x readout-coupling over geometry cases.",
        "- Boundary: gear-set atlas probe; not global closure.",
        "",
        "## Model Summary",
        "",
        "| model | gears | rows | cases | original target | target | lost | gained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {data.get('n_gears', 0)} | {data.get('n_rows', 0)} | {data.get('n_cases', 0)} | "
            f"{data.get('original_target_rows', 0)} | {data.get('target_rows', 0)} | "
            f"{data.get('target_lost_vs_original_rows', 0)} | {data.get('target_gained_vs_original_rows', 0)} |"
        )
    lines += ["", "## Top Gears", ""]
    lines += ["| model | rank | layer | channel | hits | mean act | neg ratio | mean abs support | gear score |"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for idx, gear in enumerate(data.get("top_gears") or [], 1):
            lines.append(
                f"| {model_name} | {idx} | {gear.get('layer_idx')} | {gear.get('channel_id')} | {gear.get('n')} | "
                f"{fmt(gear.get('mean_activation'))} | {fmt(gear.get('neg_ratio'))} | "
                f"{fmt(gear.get('mean_abs_support'))} | {fmt(gear.get('gear_score'))} |"
            )
    lines += ["", "## Subset Summary", ""]
    lines += ["| model | subset | n | target | lost | gained | object_echo | unknown | mean target-object | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for subset, row in (data.get("subset_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{subset}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {row.get('target_gained_vs_original_rows', 0)} | "
                f"{row.get('object_echo_rows', 0)} | {row.get('unknown_other_rows', 0)} | "
                f"{fmt(row.get('mean_target_minus_object_logit'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Object Summary", ""]
    lines += ["| model | object | n | target | lost | gained | object_echo | unknown | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for obj, row in (data.get("object_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{obj}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {row.get('target_gained_vs_original_rows', 0)} | "
                f"{row.get('object_echo_rows', 0)} | {row.get('unknown_other_rows', 0)} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Top Rows", ""]
    lines += ["| model | object | prompt | subset | mode | gears | class | output | orig class | lost | gained | target-object | top tokens |"]
    lines += ["|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("top_rows") or []:
            output = str(row.get("generated") or "").replace("|", "/")[:48]
            top_tokens = json.dumps(row.get("top_tokens") or [], ensure_ascii=False).replace("|", "/")[:80]
            lines.append(
                f"| {model_name} | `{row.get('object')}` | `{row.get('prompt_variant')}` | `{row.get('subset_name')}` | "
                f"`{row.get('edit_mode')}` | {row.get('gear_count')} | `{row.get('boundary_class')}` | {output} | "
                f"`{row.get('original_boundary_class')}` | {int(bool(row.get('target_lost_vs_original')))} | "
                f"{int(bool(row.get('target_gained_vs_original')))} | {fmt(row.get('target_minus_object_logit'))} | "
                f"`{top_tokens}` |"
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
        path = out_dir / f"phase844_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase844_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase844_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p828.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-seed-triangle", action="store_true")
    parser.add_argument("--geometry-objects", default="triangle,square,rectangle,circle")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category")
    parser.add_argument("--layers", default="26,27,28")
    parser.add_argument("--per-sample-topk", type=int, default=64)
    parser.add_argument("--min-candidate-hits", type=int, default=2)
    parser.add_argument("--max-gears", type=int, default=8)
    parser.add_argument("--subset-sizes", default="1,4,8")
    parser.add_argument("--edit-modes", default="zero,flip")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--report-top-gears", type=int, default=12)
    parser.add_argument("--max-saved-samples", type=int, default=2000)
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
