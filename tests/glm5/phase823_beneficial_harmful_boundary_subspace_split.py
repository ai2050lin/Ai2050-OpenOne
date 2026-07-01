#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase796_global_competitor_token_identity_audit as p796  # noqa: E402
import phase816_multi_token_answer_span_rollout_closure as p816  # noqa: E402
import phase820_answer_boundary_standard_v1 as p820  # noqa: E402
import phase821_boundary_standard_guided_causal_localization as p821  # noqa: E402
import phase822_boundary_transition_head_mlp_decomposition as p822  # noqa: E402
from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight, tensor_from_output  # noqa: E402


PHASE = 823
SOURCE_822 = Path("tests/result/phase822_boundary_transition_head_mlp_decomposition")
RESULT_ROOT = Path("tests/result/phase823_beneficial_harmful_boundary_subspace_split")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    vals: list[int] = []
    for item in parse_csv(text):
        vals.append(int(item))
    return sorted(set(x for x in vals if x > 0))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def case_map() -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in p816.CASES}


def clean_generated(text: str) -> str:
    return p816.clean_generated(text)


def boundary_for(lookup: dict[tuple[str, str], dict[str, Any]], case_id: str, phrase: Any) -> dict[str, Any]:
    std = p820.class_for_phrase(lookup, case_id, clean_generated(str(phrase or "")))
    cls = str(std.get("final_boundary_class") or "unknown_other")
    out = dict(std)
    out["boundary_rank"] = int(p821.BOUNDARY_RANK.get(cls, 0))
    return out


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]


def first_token_id(tokenizer, text: str) -> int | None:
    ids = encode_prompt(tokenizer, text)
    return int(ids[0]) if ids else None


def select_source_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_822 / args.source_round / f"phase822_{model_name}_rows.jsonl")
    allowed_kinds = set(parse_csv(args.component_kinds))
    useful = []
    for row in rows:
        if allowed_kinds and str(row.get("component_kind")) not in allowed_kinds:
            continue
        if row.get("target_transition") or row.get("improved_boundary") or row.get("degraded_boundary"):
            useful.append(row)

    # Keep the signal-bearing rows while avoiding one case flooding the run.
    useful.sort(
        key=lambda r: (
            bool(r.get("target_transition")),
            bool(r.get("improved_boundary")),
            bool(r.get("degraded_boundary")),
            finite(r.get("delta_boundary_rank")),
            str(r.get("case_id")),
            str(r.get("component_label")),
        ),
        reverse=True,
    )
    if int(args.max_source_rows) <= 0:
        return useful
    return useful[: int(args.max_source_rows)]


def component_spec_from_row(row: dict[str, Any]) -> dict[str, Any]:
    spec = {
        "component_kind": row.get("component_kind"),
        "component_label": row.get("component_label"),
    }
    if row.get("component_kind") == "attention_head":
        spec.update(
            {
                "head_id": int(row["head_id"]),
                "head_dim": int(row["head_dim"]),
                "num_heads": int(row["num_heads"]),
            }
        )
    if row.get("component_kind") == "mlp_channel_group":
        spec.update(
            {
                "channel_ids": [int(x) for x in row.get("channel_ids") or []],
                "channel_group_size": int(row.get("channel_group_size") or 0),
            }
        )
    return spec


def component_vector(state: dict[str, Any], spec: dict[str, Any]) -> torch.Tensor | None:
    kind = spec["component_kind"]
    if kind == "layer_residual":
        return state.get("layer_output")
    if kind == "attention_output":
        return state.get("attn_output")
    if kind == "mlp_output":
        return state.get("mlp_output")
    if kind == "attention_head":
        vec = state.get("attn_o_input")
        if vec is None:
            return None
        start = int(spec["head_id"]) * int(spec["head_dim"])
        end = start + int(spec["head_dim"])
        return vec[start:end]
    if kind == "mlp_channel_group":
        vec = state.get("mlp_down_input")
        ids = [int(x) for x in spec.get("channel_ids") or []]
        if vec is None or not ids:
            return None
        return vec[torch.tensor(ids, dtype=torch.long)]
    return None


def effective_readout_direction(
    model,
    tokenizer,
    case: dict[str, Any],
    baseline_token_ids: list[int],
    layer_idx: int,
    spec: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    target_id = first_token_id(tokenizer, str(case["answer"]))
    baseline_id = int(baseline_token_ids[0]) if baseline_token_ids else None
    if target_id is None:
        return None, {"target_token_id": None, "baseline_token_id": baseline_id}
    weight = lm_head_weight(model)
    if baseline_id is not None and 0 <= baseline_id < int(weight.shape[0]):
        readout = weight[int(target_id)] - weight[int(baseline_id)]
    else:
        readout = weight[int(target_id)]

    kind = spec["component_kind"]
    meta = {
        "target_token_id": int(target_id),
        "target_token": tokenizer.decode([int(target_id)]),
        "baseline_token_id": baseline_id,
        "baseline_token": tokenizer.decode([int(baseline_id)]) if baseline_id is not None else None,
    }
    if kind in {"layer_residual", "attention_output", "mlp_output"}:
        return readout.float().cpu(), meta
    layer = get_layers(model)[int(layer_idx)]
    if kind == "attention_head":
        if not hasattr(layer.self_attn, "o_proj"):
            return None, meta
        start = int(spec["head_id"]) * int(spec["head_dim"])
        end = start + int(spec["head_dim"])
        # Linear output y = x @ W.T; local readout coefficient is W[:, local].T @ readout.
        w = layer.self_attn.o_proj.weight.detach().float().cpu()[:, start:end]
        return (w.T @ readout.float().cpu()).float(), meta
    if kind == "mlp_channel_group":
        if not hasattr(layer.mlp, "down_proj"):
            return None, meta
        ids = [int(x) for x in spec.get("channel_ids") or []]
        if not ids:
            return None, meta
        w = layer.mlp.down_proj.weight.detach().float().cpu()[:, ids]
        return (w.T @ readout.float().cpu()).float(), meta
    return None, meta


def selected_indices_for_mode(
    mode: str,
    budget: int,
    signed_scores: torch.Tensor,
    seed_parts: tuple[Any, ...],
) -> tuple[list[int], dict[str, Any]]:
    n = int(signed_scores.numel())
    if n <= 0:
        return [], {"candidate_count": 0}
    k = min(int(budget), n)
    positive = torch.nonzero(signed_scores > 0, as_tuple=False).flatten().tolist()
    negative = torch.nonzero(signed_scores < 0, as_tuple=False).flatten().tolist()
    if mode == "all":
        idxs = list(range(n))
    elif mode == "positive_topk":
        ranked = sorted(positive, key=lambda i: float(signed_scores[int(i)].item()), reverse=True)
        idxs = ranked[: min(k, len(ranked))]
    elif mode == "negative_topk":
        ranked = sorted(negative, key=lambda i: float(-signed_scores[int(i)].item()), reverse=True)
        idxs = ranked[: min(k, len(ranked))]
    elif mode == "abs_topk":
        ranked = sorted(range(n), key=lambda i: abs(float(signed_scores[int(i)].item())), reverse=True)
        idxs = ranked[:k]
    elif mode == "random_topk":
        rng = random.Random(stable_seed(*seed_parts))
        idxs = list(range(n))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:k])
    else:
        raise ValueError(f"unknown subspace mode: {mode}")
    return [int(x) for x in idxs], {
        "candidate_count": n,
        "positive_dim_count": len(positive),
        "negative_dim_count": len(negative),
        "budget": int(budget),
    }


def patch_selected_output_hook(rec_vec: torch.Tensor, donor_vec: torch.Tensor, selected: list[int], alpha: float = 1.0):
    idx_cpu = torch.tensor([int(x) for x in selected], dtype=torch.long)

    def hook(_module, _inputs, output):
        tensor = tensor_from_output(output)
        if tensor is None:
            return output
        patched = tensor.clone()
        if idx_cpu.numel() > 0:
            idx = idx_cpu.to(device=patched.device)
            rec = rec_vec.to(device=patched.device, dtype=patched.dtype)
            donor = donor_vec.to(device=patched.device, dtype=patched.dtype)
            patched[:, -1, idx] = rec[idx] + float(alpha) * (donor[idx] - rec[idx])
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook


def patch_selected_head_pre_hook(
    rec_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    selected: list[int],
    spec: dict[str, Any],
    alpha: float = 1.0,
):
    idx_local = torch.tensor([int(x) for x in selected], dtype=torch.long)
    start = int(spec["head_id"]) * int(spec["head_dim"])

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        patched = inputs[0].clone()
        if idx_local.numel() > 0:
            local = idx_local.to(device=patched.device)
            global_idx = local + int(start)
            rec = rec_vec.to(device=patched.device, dtype=patched.dtype)
            donor = donor_vec.to(device=patched.device, dtype=patched.dtype)
            patched[:, -1, global_idx] = rec[local] + float(alpha) * (donor[local] - rec[local])
        return (patched, *inputs[1:])

    return hook


def patch_selected_mlp_channels_pre_hook(
    rec_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    selected: list[int],
    spec: dict[str, Any],
    alpha: float = 1.0,
):
    channel_ids = [int(x) for x in spec.get("channel_ids") or []]
    selected_channels = [channel_ids[int(i)] for i in selected if 0 <= int(i) < len(channel_ids)]
    idx_local = torch.tensor([int(i) for i in selected if 0 <= int(i) < len(channel_ids)], dtype=torch.long)
    idx_channels = torch.tensor(selected_channels, dtype=torch.long)

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        patched = inputs[0].clone()
        if idx_channels.numel() > 0:
            global_idx = idx_channels.to(device=patched.device)
            local = idx_local.to(device=patched.device)
            rec = rec_vec.to(device=patched.device, dtype=patched.dtype)
            donor = donor_vec.to(device=patched.device, dtype=patched.dtype)
            patched[:, -1, global_idx] = rec[local] + float(alpha) * (donor[local] - rec[local])
        return (patched, *inputs[1:])

    return hook


def install_subspace_patch(
    model,
    layer_idx: int,
    spec: dict[str, Any],
    recipient_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    selected: list[int],
    alpha: float,
):
    layer = get_layers(model)[int(layer_idx)]
    kind = spec["component_kind"]
    if kind == "layer_residual":
        return layer.register_forward_hook(patch_selected_output_hook(recipient_vec, donor_vec, selected, alpha))
    if kind == "attention_output":
        return layer.self_attn.register_forward_hook(patch_selected_output_hook(recipient_vec, donor_vec, selected, alpha))
    if kind == "mlp_output":
        return layer.mlp.register_forward_hook(patch_selected_output_hook(recipient_vec, donor_vec, selected, alpha))
    if kind == "attention_head":
        return layer.self_attn.o_proj.register_forward_pre_hook(
            patch_selected_head_pre_hook(recipient_vec, donor_vec, selected, spec, alpha)
        )
    if kind == "mlp_channel_group":
        return layer.mlp.down_proj.register_forward_pre_hook(
            patch_selected_mlp_channels_pre_hook(recipient_vec, donor_vec, selected, spec, alpha)
        )
    raise ValueError(f"unknown component kind: {kind}")


def greedy_generate_with_subspace_patch(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    max_new_tokens: int,
    layer_idx: int | None = None,
    spec: dict[str, Any] | None = None,
    recipient_vec: torch.Tensor | None = None,
    donor_vec: torch.Tensor | None = None,
    selected: list[int] | None = None,
    alpha: float = 1.0,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handle = None
        if (
            step == 0
            and layer_idx is not None
            and spec is not None
            and recipient_vec is not None
            and donor_vec is not None
            and selected is not None
        ):
            handle = install_subspace_patch(model, int(layer_idx), spec, recipient_vec, donor_vec, selected, alpha)
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


def audit_source_component(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    source_row: dict[str, Any],
    standards: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p820.standard_lookup(standards)
    layer_idx = int(source_row["layer_idx"])
    spec = component_spec_from_row(source_row)
    recipient_prompt = p816.build_prompt(case, args.recipient_prompt)
    donor_prompt = p816.build_prompt(case, args.donor_prompt)
    recipient_ids = encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = boundary_for(lookup, case["case_id"], baseline_text)
    recipient_state = p822.capture_component_state(model, tokenizer, device, recipient_prompt, layer_idx)
    donor_state = p822.capture_component_state(model, tokenizer, device, donor_prompt, layer_idx)
    recipient_vec = component_vector(recipient_state, spec)
    donor_vec = component_vector(donor_state, spec)
    effective_dir, readout_meta = effective_readout_direction(model, tokenizer, case, baseline_ids, layer_idx, spec)
    if recipient_vec is None or donor_vec is None or effective_dir is None:
        return []
    recipient_vec = recipient_vec.float().cpu()
    donor_vec = donor_vec.float().cpu()
    effective_dir = effective_dir.float().cpu()
    n = min(int(recipient_vec.numel()), int(donor_vec.numel()), int(effective_dir.numel()))
    if n <= 0:
        return []
    recipient_vec = recipient_vec[:n]
    donor_vec = donor_vec[:n]
    effective_dir = effective_dir[:n]
    delta = donor_vec - recipient_vec
    signed_scores = delta * effective_dir
    budgets = parse_int_csv(args.budgets)
    if not budgets:
        budgets = [n]
    rows: list[dict[str, Any]] = []
    for mode in parse_csv(args.subspace_modes):
        mode_budgets = [n] if mode == "all" else budgets
        for budget in mode_budgets:
            selected, sel_meta = selected_indices_for_mode(
                mode,
                int(budget),
                signed_scores,
                (args.model, case["case_id"], layer_idx, spec.get("component_label"), mode, budget),
            )
            if not selected:
                continue
            patched_text, patched_ids = greedy_generate_with_subspace_patch(
                model,
                tokenizer,
                device,
                recipient_ids,
                args.max_new_tokens,
                layer_idx,
                spec,
                recipient_vec,
                donor_vec,
                selected,
                args.alpha,
            )
            patched_boundary = boundary_for(lookup, case["case_id"], patched_text)
            delta_rank = int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"])
            selected_scores = signed_scores[torch.tensor(selected, dtype=torch.long)]
            row = {
                "row_kind": "phase823_beneficial_harmful_boundary_subspace_split",
                "phase": PHASE,
                "source_phase": 822,
                "model": args.model,
                "round": args.round_name,
                "source_round": args.source_round,
                "case_id": case["case_id"],
                "object": case["object"],
                "target_answer": case["answer"],
                "layer_idx": layer_idx,
                "component_kind": spec.get("component_kind"),
                "component_label": spec.get("component_label"),
                "head_id": spec.get("head_id"),
                "channel_group_size": spec.get("channel_group_size"),
                "source_phase822_role": source_row.get("role_label"),
                "source_phase822_boundary_class": source_row.get("patched_boundary_class"),
                "source_phase822_delta_rank": source_row.get("delta_boundary_rank"),
                "source_phase822_target_transition": bool(source_row.get("target_transition")),
                "subspace_mode": mode,
                "budget": int(budget),
                "n_component_dims": n,
                "n_selected": len(selected),
                "selected_indices": selected[: int(args.save_indices_limit)],
                "selected_signed_score_sum": float(selected_scores.sum().item()),
                "selected_positive_score_sum": float(torch.clamp(selected_scores, min=0.0).sum().item()),
                "selected_negative_abs_score_sum": float(torch.clamp(-selected_scores, min=0.0).sum().item()),
                "total_positive_dim_count": int((signed_scores > 0).sum().item()),
                "total_negative_dim_count": int((signed_scores < 0).sum().item()),
                "total_positive_score_sum": float(torch.clamp(signed_scores, min=0.0).sum().item()),
                "total_negative_abs_score_sum": float(torch.clamp(-signed_scores, min=0.0).sum().item()),
                "selection_meta": sel_meta,
                "readout_meta": readout_meta,
                "baseline_generated": clean_generated(baseline_text),
                "baseline_token_ids": baseline_ids,
                "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
                "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
                "patched_generated": clean_generated(patched_text),
                "patched_token_ids": patched_ids,
                "patched_boundary_class": patched_boundary.get("final_boundary_class"),
                "patched_boundary_rank": int(patched_boundary["boundary_rank"]),
                "delta_boundary_rank": delta_rank,
                "improved_boundary": delta_rank > 0,
                "degraded_boundary": delta_rank < 0,
                "target_transition": patched_boundary.get("final_boundary_class") == "target_equivalent",
                "protocol_repaired": (
                    not bool(baseline_boundary.get("protocol_valid")) and bool(patched_boundary.get("protocol_valid"))
                ),
            }
            rows.append(row)
    return rows


def pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("subspace_mode") not in {"positive_topk", "negative_topk"}:
            continue
        key = (
            row.get("model"),
            row.get("case_id"),
            row.get("layer_idx"),
            row.get("component_kind"),
            row.get("component_label"),
            row.get("budget"),
        )
        by_key[key][str(row.get("subspace_mode"))] = row
    pairs = []
    for key, vals in by_key.items():
        pos = vals.get("positive_topk")
        neg = vals.get("negative_topk")
        if not pos or not neg:
            continue
        pairs.append(
            {
                "key": key,
                "positive_rank": pos.get("patched_boundary_rank"),
                "negative_rank": neg.get("patched_boundary_rank"),
                "positive_class": pos.get("patched_boundary_class"),
                "negative_class": neg.get("patched_boundary_class"),
                "positive_target": bool(pos.get("target_transition")),
                "negative_target": bool(neg.get("target_transition")),
                "positive_better": finite(pos.get("patched_boundary_rank")) > finite(neg.get("patched_boundary_rank")),
            }
        )
    return {
        "paired_count": len(pairs),
        "positive_better_pairs": sum(1 for row in pairs if row["positive_better"]),
        "positive_target_pairs": sum(1 for row in pairs if row["positive_target"]),
        "negative_target_pairs": sum(1 for row in pairs if row["negative_target"]),
        "pairs": pairs[:30],
    }


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None = None) -> dict[str, Any]:
    by_mode = defaultdict(list)
    by_kind_mode = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("subspace_mode"))].append(row)
        by_kind_mode[(str(row.get("component_kind")), str(row.get("subspace_mode")))].append(row)
    def compact(vals: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(vals),
            "improved_rows": sum(1 for row in vals if row.get("improved_boundary")),
            "degraded_rows": sum(1 for row in vals if row.get("degraded_boundary")),
            "target_transition_rows": sum(1 for row in vals if row.get("target_transition")),
            "protocol_repaired_rows": sum(1 for row in vals if row.get("protocol_repaired")),
            "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in vals) / len(vals) if vals else None,
            "patched_classes": dict(Counter(row.get("patched_boundary_class") for row in vals)),
        }
    mode_summary = {mode: compact(vals) for mode, vals in sorted(by_mode.items())}
    kind_mode_summary = {f"{kind}/{mode}": compact(vals) for (kind, mode), vals in sorted(by_kind_mode.items())}
    by_case = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id"))].append(row)
    best_cases = {}
    for case_id, vals in by_case.items():
        best = max(vals, key=lambda r: (bool(r.get("target_transition")), finite(r.get("delta_boundary_rank")), finite(r.get("patched_boundary_rank"))))
        best_cases[case_id] = {
            "baseline_class": vals[0].get("baseline_boundary_class"),
            "best_mode": best.get("subspace_mode"),
            "best_budget": best.get("budget"),
            "best_component_kind": best.get("component_kind"),
            "best_component_label": best.get("component_label"),
            "best_patched_class": best.get("patched_boundary_class"),
            "best_delta_boundary_rank": best.get("delta_boundary_rank"),
            "best_patched_generated": best.get("patched_generated"),
            "any_positive_target": any(row.get("subspace_mode") == "positive_topk" and row.get("target_transition") for row in vals),
            "any_negative_degraded": any(row.get("subspace_mode") == "negative_topk" and row.get("degraded_boundary") for row in vals),
        }
    return {
        "phase": PHASE,
        "title": "Beneficial / Harmful Boundary-Transition Subspace Split",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len(best_cases),
        "n_source_components": len({(row.get("case_id"), row.get("layer_idx"), row.get("component_kind"), row.get("component_label")) for row in rows}),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "protocol_repaired_rows": sum(1 for row in rows if row.get("protocol_repaired")),
        "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in rows) / len(rows) if rows else None,
        "mode_summary": mode_summary,
        "component_mode_summary": kind_mode_summary,
        "pair_summary": pair_summary(rows),
        "by_case": best_cases,
        "boundary": "This phase splits donor-recipient component differences from Phase 822 into readout-positive and readout-negative subspaces and tests whether partial subspace patches preserve, improve, or harm boundary transitions.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p820.standard_rows()
    selected = select_source_rows(args.model, args)
    cmap = case_map()
    log(f"{args.model}/{args.round_name}: selected Phase 822 source components={len(selected)}")
    if args.dry_run:
        payload = {
            "model": args.model,
            "selected": [
                {
                    "case_id": row.get("case_id"),
                    "layer_idx": row.get("layer_idx"),
                    "component_kind": row.get("component_kind"),
                    "component_label": row.get("component_label"),
                    "patched_class": row.get("patched_boundary_class"),
                    "delta": row.get("delta_boundary_rank"),
                }
                for row in selected
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict[str, Any]] = []
    try:
        for idx, source_row in enumerate(selected, 1):
            case = cmap.get(str(source_row.get("case_id")))
            if not case:
                continue
            rows.extend(audit_source_component(model, tokenizer, device, case, source_row, standards, args))
            if idx % int(args.log_every) == 0 or idx == len(selected):
                log(f"{args.model}: split {idx}/{len(selected)} source components; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase823_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase823_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "source_components": summary["n_source_components"],
                "rows": summary["n_rows"],
                "target_transition_rows": summary["target_transition_rows"],
                "improved_rows": summary["improved_rows"],
                "degraded_rows": summary["degraded_rows"],
                "positive_better_pairs": summary["pair_summary"]["positive_better_pairs"],
                "paired_count": summary["pair_summary"]["paired_count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 823 Beneficial / Harmful Boundary-Transition Subspace Split ({payload['round']})",
        "",
        "- Boundary: Phase 820 answer-boundary standard v1.",
        "- Source: Phase 822 successful/improved/degraded components.",
        "- Intervention: split donor-recipient component deltas into readout-positive and readout-negative dimensions, then patch selected subspaces.",
        "",
        "## Model Summary",
        "",
        "| model | source comps | rows | improved | target | degraded | mean delta | positive better pairs | paired |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        pair = data.get("pair_summary") or {}
        lines.append(
            f"| {model_name} | {data.get('n_source_components')} | {data.get('n_rows')} | "
            f"{data.get('improved_rows')} | {data.get('target_transition_rows')} | {data.get('degraded_rows')} | "
            f"{finite(data.get('mean_delta_boundary_rank')):.3f} | {pair.get('positive_better_pairs')} | {pair.get('paired_count')} |"
        )
    lines += ["", "## Mode Summary", ""]
    lines += [
        "| model | mode | n | improved | target | degraded | mean delta | classes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for mode, row in sorted((data.get("mode_summary") or {}).items()):
            lines.append(
                f"| {model_name} | `{mode}` | {row.get('n')} | {row.get('improved_rows')} | "
                f"{row.get('target_transition_rows')} | {row.get('degraded_rows')} | "
                f"{finite(row.get('mean_delta_boundary_rank')):.3f} | "
                f"`{json.dumps(row.get('patched_classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Component / Mode Summary", ""]
    lines += [
        "| model | component/mode | n | improved | target | degraded | mean delta | classes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for key, row in sorted((data.get("component_mode_summary") or {}).items()):
            lines.append(
                f"| {model_name} | `{key}` | {row.get('n')} | {row.get('improved_rows')} | "
                f"{row.get('target_transition_rows')} | {row.get('degraded_rows')} | "
                f"{finite(row.get('mean_delta_boundary_rank')):.3f} | "
                f"`{json.dumps(row.get('patched_classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Best Case Rows", ""]
    lines += [
        "| model | case | baseline | best mode | budget | component | class | delta | generated |",
        "|---|---|---|---|---:|---|---|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for case_id, row in sorted((data.get("by_case") or {}).items()):
            lines.append(
                f"| {model_name} | {case_id} | `{row.get('baseline_class')}` | `{row.get('best_mode')}` | "
                f"{row.get('best_budget')} | `{row.get('best_component_kind')}/{row.get('best_component_label')}` | "
                f"`{row.get('best_patched_class')}` | {row.get('best_delta_boundary_rank')} | `{row.get('best_patched_generated')}` |"
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
    for model_name in MODELS:
        path = out_dir / f"phase823_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase823_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase823_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--donor-prompt", default="exact_choices")
    parser.add_argument("--max-source-rows", type=int, default=4, help="0 means all useful source rows.")
    parser.add_argument("--component-kinds", default="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group")
    parser.add_argument("--subspace-modes", default="all,positive_topk,negative_topk,abs_topk,random_topk")
    parser.add_argument("--budgets", default="16,64")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--save-indices-limit", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    run_model(args)


if __name__ == "__main__":
    main()
