#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase780_surface_form_component_localization import tensor_from_output  # noqa: E402
from phase786_head_mlp_source_audit import infer_num_heads  # noqa: E402


PHASE = 822
SOURCE_821 = Path("tests/result/phase821_boundary_standard_guided_causal_localization")
RESULT_ROOT = Path("tests/result/phase822_boundary_transition_head_mlp_decomposition")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    vals = []
    for part in parse_csv(text):
        vals.append(int(part))
    return vals


def finite(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


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


def tensor_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]


def select_source_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_821 / args.source_round / f"phase821_{model_name}_rows.jsonl")
    if not rows:
        return []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if args.prefer_target and not row.get("target_transition") and any(
            r.get("target_transition") for r in rows if r.get("case_id") == row.get("case_id")
        ):
            continue
        if not row.get("target_transition") and not row.get("improved_boundary"):
            continue
        by_case[str(row.get("case_id"))].append(row)
    selected = []
    for case_id, vals in by_case.items():
        vals.sort(
            key=lambda r: (
                bool(r.get("target_transition")),
                finite(r.get("delta_boundary_rank")),
                finite(r.get("patched_boundary_rank")),
            ),
            reverse=True,
        )
        selected.append(vals[0])
    selected.sort(
        key=lambda r: (
            bool(r.get("target_transition")),
            finite(r.get("delta_boundary_rank")),
            str(r.get("case_id")),
        ),
        reverse=True,
    )
    return selected[: int(args.max_cases)]


def capture_component_state(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    layer_idx: int,
) -> dict[str, Any]:
    ids = encode_prompt(tokenizer, prompt)
    answer_pos = len(ids) - 1
    layer = get_layers(model)[int(layer_idx)]
    state: dict[str, Any] = {
        "prompt_ids": ids,
        "answer_pos": answer_pos,
        "layer_output": None,
        "attn_output": None,
        "mlp_output": None,
        "attn_o_input": None,
        "mlp_down_input": None,
    }
    handles = []

    def layer_hook(_module, _inputs, output):
        tensor = tensor_from_output(output)
        if tensor is not None:
            state["layer_output"] = tensor[0, answer_pos].detach().float().cpu()

    def attn_hook(_module, _inputs, output):
        tensor = tensor_from_output(output)
        if tensor is not None:
            state["attn_output"] = tensor[0, answer_pos].detach().float().cpu()

    def mlp_hook(_module, _inputs, output):
        tensor = tensor_from_output(output)
        if tensor is not None:
            state["mlp_output"] = tensor[0, answer_pos].detach().float().cpu()

    def o_pre_hook(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            state["attn_o_input"] = inputs[0][0, answer_pos].detach().float().cpu()

    def down_pre_hook(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            state["mlp_down_input"] = inputs[0][0, answer_pos].detach().float().cpu()

    handles.append(layer.register_forward_hook(layer_hook))
    handles.append(layer.self_attn.register_forward_hook(attn_hook))
    if hasattr(layer.self_attn, "o_proj"):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(o_pre_hook))
    handles.append(layer.mlp.register_forward_hook(mlp_hook))
    if hasattr(layer.mlp, "down_proj"):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(down_pre_hook))
    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return state


def replace_last_vector_hook(donor_vec: torch.Tensor):
    def hook(_module, _inputs, output):
        tensor = tensor_from_output(output)
        if tensor is None:
            return output
        patched = tensor.clone()
        vec = donor_vec.to(device=patched.device, dtype=patched.dtype)
        patched[:, -1, :] = vec
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook


def replace_head_pre_hook(donor_o_input: torch.Tensor, head_id: int, head_dim: int):
    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        patched = inputs[0].clone()
        start = int(head_id) * int(head_dim)
        end = start + int(head_dim)
        donor = donor_o_input.to(device=patched.device, dtype=patched.dtype)
        patched[:, -1, start:end] = donor[start:end]
        return (patched, *inputs[1:])

    return hook


def replace_mlp_channels_pre_hook(donor_down_input: torch.Tensor, channel_ids: list[int]):
    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]) or not channel_ids:
            return inputs
        patched = inputs[0].clone()
        donor = donor_down_input.to(device=patched.device, dtype=patched.dtype)
        idx = torch.tensor([int(x) for x in channel_ids], dtype=torch.long, device=patched.device)
        patched[:, -1, idx] = donor[idx]
        return (patched, *inputs[1:])

    return hook


def install_patch(model, layer_idx: int, spec: dict[str, Any], donor_state: dict[str, Any]):
    layer = get_layers(model)[int(layer_idx)]
    kind = spec["component_kind"]
    if kind == "layer_residual":
        return layer.register_forward_hook(replace_last_vector_hook(donor_state["layer_output"]))
    if kind == "attention_output":
        return layer.self_attn.register_forward_hook(replace_last_vector_hook(donor_state["attn_output"]))
    if kind == "mlp_output":
        return layer.mlp.register_forward_hook(replace_last_vector_hook(donor_state["mlp_output"]))
    if kind == "attention_head":
        return layer.self_attn.o_proj.register_forward_pre_hook(
            replace_head_pre_hook(donor_state["attn_o_input"], int(spec["head_id"]), int(spec["head_dim"]))
        )
    if kind == "mlp_channel_group":
        return layer.mlp.down_proj.register_forward_pre_hook(
            replace_mlp_channels_pre_hook(donor_state["mlp_down_input"], [int(x) for x in spec["channel_ids"]])
        )
    raise ValueError(f"unknown component kind: {kind}")


def greedy_generate_with_component_patch(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    max_new_tokens: int,
    layer_idx: int | None = None,
    patch_spec: dict[str, Any] | None = None,
    donor_state: dict[str, Any] | None = None,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handle = None
        if step == 0 and layer_idx is not None and patch_spec is not None and donor_state is not None:
            handle = install_patch(model, int(layer_idx), patch_spec, donor_state)
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


def top_channel_groups(recipient_state: dict[str, Any], donor_state: dict[str, Any], sizes: list[int]) -> list[dict[str, Any]]:
    rec = recipient_state.get("mlp_down_input")
    donor = donor_state.get("mlp_down_input")
    if rec is None or donor is None or rec.numel() == 0 or donor.numel() == 0:
        return []
    diff = (donor - rec).float().abs()
    out = []
    seen: set[tuple[int, ...]] = set()
    for size in sizes:
        k = min(int(size), int(diff.numel()))
        if k <= 0:
            continue
        vals, ids = torch.topk(diff, k)
        channel_ids = [int(x) for x in ids.tolist()]
        key = tuple(sorted(channel_ids))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "component_kind": "mlp_channel_group",
                "component_label": f"mlp_topdiff_{k}",
                "channel_group_size": k,
                "channel_ids": channel_ids,
                "channel_abs_diff_sum": float(vals.sum().item()),
            }
        )
    return out


def component_specs(model, layer_idx: int, recipient_state: dict[str, Any], donor_state: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    layer = get_layers(model)[int(layer_idx)]
    specs: list[dict[str, Any]] = []
    if donor_state.get("layer_output") is not None:
        specs.append({"component_kind": "layer_residual", "component_label": "whole_layer_residual"})
    if donor_state.get("attn_output") is not None:
        specs.append({"component_kind": "attention_output", "component_label": "whole_attention_output"})
    if donor_state.get("mlp_output") is not None:
        specs.append({"component_kind": "mlp_output", "component_label": "whole_mlp_output"})
    if donor_state.get("attn_o_input") is not None and hasattr(layer.self_attn, "o_proj"):
        n_heads = infer_num_heads(model, layer.self_attn)
        in_features = int(donor_state["attn_o_input"].numel())
        if n_heads and in_features % int(n_heads) == 0:
            head_dim = in_features // int(n_heads)
            max_heads = int(args.max_heads)
            head_ids = list(range(int(n_heads)))
            if max_heads > 0:
                # Cheap first pass: use donor-recipient input-difference norm to pick the largest heads.
                rec = recipient_state.get("attn_o_input")
                donor = donor_state.get("attn_o_input")
                if rec is not None and donor is not None:
                    scores = []
                    for hid in head_ids:
                        start = hid * head_dim
                        end = start + head_dim
                        scores.append((float((donor[start:end] - rec[start:end]).float().norm().item()), hid))
                    head_ids = [hid for _score, hid in sorted(scores, reverse=True)[:max_heads]]
                else:
                    head_ids = head_ids[:max_heads]
            for head_id in head_ids:
                specs.append(
                    {
                        "component_kind": "attention_head",
                        "component_label": f"head_{head_id}",
                        "head_id": int(head_id),
                        "num_heads": int(n_heads),
                        "head_dim": int(head_dim),
                    }
                )
    specs.extend(top_channel_groups(recipient_state, donor_state, parse_int_csv(args.mlp_channel_groups)))
    return specs


def role_label(row: dict[str, Any]) -> str:
    before = str(row.get("baseline_boundary_class"))
    after = str(row.get("patched_boundary_class"))
    if row.get("target_transition"):
        if before in {"close_near_miss", "broad_near_miss", "unknown_other"}:
            return "category_writer_or_refiner"
        if before in {"format_echo", "object_echo", "format_with_target"}:
            return "protocol_plus_category_repair"
        return "target_writer"
    if row.get("delta_boundary_rank", 0) > 0:
        if before == "format_echo" and after == "generic_blocker":
            return "protocol_verbalizer_not_answer_writer"
        if before == "object_echo":
            return "object_echo_suppressor_partial"
        return "partial_boundary_improver"
    if row.get("delta_boundary_rank", 0) < 0:
        return "harmful_mixer"
    return "neutral"


def audit_source_row(
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
    recipient_prompt = p816.build_prompt(case, args.recipient_prompt)
    donor_prompt = p816.build_prompt(case, args.donor_prompt)
    recipient_ids = encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = greedy_generate_with_component_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = boundary_for(lookup, case["case_id"], baseline_text)
    recipient_state = capture_component_state(model, tokenizer, device, recipient_prompt, layer_idx)
    donor_state = capture_component_state(model, tokenizer, device, donor_prompt, layer_idx)
    specs = component_specs(model, layer_idx, recipient_state, donor_state, args)
    rows = []
    for spec in specs:
        patched_text, patched_ids = greedy_generate_with_component_patch(
            model,
            tokenizer,
            device,
            recipient_ids,
            args.max_new_tokens,
            layer_idx,
            spec,
            donor_state,
        )
        patched_boundary = boundary_for(lookup, case["case_id"], patched_text)
        delta_rank = int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"])
        row = {
            "row_kind": "phase822_boundary_transition_head_mlp_decomposition",
            "phase": PHASE,
            "source_phase": 821,
            "model": args.model,
            "round": args.round_name,
            "source_round": args.source_round,
            "case_id": case["case_id"],
            "object": case["object"],
            "target_answer": case["answer"],
            "layer_idx": layer_idx,
            "recipient_prompt_variant": args.recipient_prompt,
            "donor_prompt_variant": args.donor_prompt,
            "source_component_label": f"L{layer_idx}",
            "source_phase821_best_class": source_row.get("patched_boundary_class"),
            "source_phase821_delta_rank": source_row.get("delta_boundary_rank"),
            "source_phase821_target_transition": bool(source_row.get("target_transition")),
            "component_kind": spec.get("component_kind"),
            "component_label": spec.get("component_label"),
            "head_id": spec.get("head_id"),
            "num_heads": spec.get("num_heads"),
            "head_dim": spec.get("head_dim"),
            "channel_group_size": spec.get("channel_group_size"),
            "channel_ids": spec.get("channel_ids"),
            "channel_abs_diff_sum": spec.get("channel_abs_diff_sum"),
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
        row["role_label"] = role_label(row)
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None = None) -> dict[str, Any]:
    by_kind = defaultdict(list)
    by_role = Counter()
    for row in rows:
        by_kind[str(row.get("component_kind"))].append(row)
        by_role[str(row.get("role_label"))] += 1
    kind_summary = {}
    for kind, vals in by_kind.items():
        kind_summary[kind] = {
            "n": len(vals),
            "improved_rows": sum(1 for row in vals if row.get("improved_boundary")),
            "degraded_rows": sum(1 for row in vals if row.get("degraded_boundary")),
            "target_transition_rows": sum(1 for row in vals if row.get("target_transition")),
            "protocol_repaired_rows": sum(1 for row in vals if row.get("protocol_repaired")),
            "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in vals) / len(vals),
            "patched_classes": dict(Counter(row.get("patched_boundary_class") for row in vals)),
            "roles": dict(Counter(row.get("role_label") for row in vals)),
        }
    by_case = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id"))].append(row)
    case_summary = {}
    for case_id, vals in by_case.items():
        best = max(vals, key=lambda r: (bool(r.get("target_transition")), finite(r.get("delta_boundary_rank")), finite(r.get("patched_boundary_rank"))))
        case_summary[case_id] = {
            "baseline_class": vals[0].get("baseline_boundary_class"),
            "best_component_kind": best.get("component_kind"),
            "best_component_label": best.get("component_label"),
            "best_delta_boundary_rank": best.get("delta_boundary_rank"),
            "best_patched_class": best.get("patched_boundary_class"),
            "best_patched_generated": best.get("patched_generated"),
            "best_role": best.get("role_label"),
            "any_target_transition": any(row.get("target_transition") for row in vals),
            "any_improved": any(row.get("improved_boundary") for row in vals),
        }
    return {
        "phase": PHASE,
        "title": "Boundary-Transition Head / MLP Decomposition",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len(case_summary),
        "component_kind_summary": kind_summary,
        "role_counts": dict(by_role),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "protocol_repaired_rows": sum(1 for row in rows if row.get("protocol_repaired")),
        "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in rows) / len(rows) if rows else None,
        "by_case": case_summary,
        "boundary": "This phase decomposes Phase 821 successful or improved layer-level boundary transitions into attention, MLP, head, and MLP channel-group interventions.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p820.standard_rows()
    selected = select_source_rows(args.model, args)
    cmap = case_map()
    log(f"{args.model}/{args.round_name}: selected source rows={len(selected)}")
    if args.dry_run:
        payload = {
            "model": args.model,
            "selected": [
                {
                    "case_id": row.get("case_id"),
                    "layer_idx": row.get("layer_idx"),
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
            rows.extend(audit_source_row(model, tokenizer, device, case, source_row, standards, args))
            if idx % int(args.log_every) == 0 or idx == len(selected):
                log(f"{args.model}: decomposed {idx}/{len(selected)} source rows; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase822_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase822_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "cases": summary["n_cases"],
                "rows": summary["n_rows"],
                "target_transition_rows": summary["target_transition_rows"],
                "improved_rows": summary["improved_rows"],
                "mean_delta_boundary_rank": summary["mean_delta_boundary_rank"],
                "roles": summary["role_counts"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 822 Boundary-Transition Head / MLP Decomposition ({payload['round']})",
        "",
        "- Boundary: Phase 820 answer-boundary standard v1, with Phase 821 source rows.",
        "- Intervention: decompose a successful/improved layer residual transition into whole attention, whole MLP, attention-head o-proj slices, and MLP top-difference channel groups.",
        "",
        "## Model Summary",
        "",
        "| model | cases | rows | improved | target transitions | protocol repairs | degraded | mean delta | roles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_cases')} | {data.get('n_rows')} | {data.get('improved_rows')} | "
            f"{data.get('target_transition_rows')} | {data.get('protocol_repaired_rows')} | "
            f"{data.get('degraded_rows')} | {finite(data.get('mean_delta_boundary_rank')):.3f} | "
            f"`{json.dumps(data.get('role_counts') or {}, ensure_ascii=False)}` |"
        )
    lines += ["", "## Component Kind Summary", ""]
    lines += [
        "| model | component kind | n | improved | target | protocol | degraded | mean delta | patched classes | roles |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for kind, row in sorted((data.get("component_kind_summary") or {}).items()):
            lines.append(
                f"| {model_name} | {kind} | {row.get('n')} | {row.get('improved_rows')} | "
                f"{row.get('target_transition_rows')} | {row.get('protocol_repaired_rows')} | "
                f"{row.get('degraded_rows')} | {finite(row.get('mean_delta_boundary_rank')):.3f} | "
                f"`{json.dumps(row.get('patched_classes') or {}, ensure_ascii=False)}` | "
                f"`{json.dumps(row.get('roles') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Best Case Components", ""]
    lines += [
        "| model | case | baseline | best kind | best component | best class | delta | generated | role |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for case_id, row in sorted((data.get("by_case") or {}).items()):
            lines.append(
                f"| {model_name} | {case_id} | `{row.get('baseline_class')}` | `{row.get('best_component_kind')}` | "
                f"`{row.get('best_component_label')}` | `{row.get('best_patched_class')}` | "
                f"{row.get('best_delta_boundary_rank')} | `{row.get('best_patched_generated')}` | `{row.get('best_role')}` |"
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
        path = out_dir / f"phase822_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase822_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase822_cross_model_summary.md", payload)
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
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--max-heads", type=int, default=8, help="0 means all heads; positive selects heads with largest donor-recipient o_proj input difference.")
    parser.add_argument("--mlp-channel-groups", default="1,8")
    parser.add_argument("--prefer-target", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
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
