#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
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
import phase820_answer_boundary_standard_v1 as p820  # noqa: E402
import phase822_boundary_transition_head_mlp_decomposition as p822  # noqa: E402
import phase823_beneficial_harmful_boundary_subspace_split as p823  # noqa: E402
import phase825_candidate_pool_expansion_natural_route_validation as p825  # noqa: E402
import phase827_exact_natural_consistency_constrained_search as p827  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402


PHASE = 828
SOURCE_827 = Path("tests/result/phase827_exact_natural_consistency_constrained_search")
RESULT_ROOT = Path("tests/result/phase828_cross_component_consistency_fiber_composition")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def component_id(group: dict[str, Any]) -> tuple[Any, ...]:
    return (
        group["case_id"],
        int(group["layer_idx"]),
        group["component_kind"],
        group["component_label"],
        int(group["budget"]),
    )


def compact_component_label(group: dict[str, Any]) -> str:
    return f"L{group['layer_idx']}:{group['component_kind']}:{group['component_label']}:B{group['budget']}"


def summarize_group(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    first = rows[0]
    selected = first.get("selected_indices_full") or first.get("selected_indices") or []
    exact_target = any(row.get("donor_variant") == "exact_choices" and row.get("target_transition") for row in rows)
    natural_target_count = sum(1 for row in rows if row.get("donor_variant") != "exact_choices" and row.get("target_transition"))
    target_total = sum(1 for row in rows if row.get("target_transition"))
    improved_total = sum(1 for row in rows if row.get("improved_boundary"))
    degraded = sum(1 for row in rows if row.get("degraded_boundary"))
    delta_sum = sum(finite(row.get("delta_boundary_rank")) for row in rows)
    return {
        "case_id": first["case_id"],
        "object": first.get("object"),
        "target_answer": first.get("target_answer"),
        "layer_idx": int(first["layer_idx"]),
        "component_kind": first["component_kind"],
        "component_label": first["component_label"],
        "head_id": first.get("head_id"),
        "channel_group_size": first.get("channel_group_size"),
        "budget": int(first["budget"]),
        "selected_indices": [int(x) for x in selected],
        "single_exact_target": bool(exact_target),
        "single_natural_target_count": int(natural_target_count),
        "single_exact_plus_multi": bool(exact_target and natural_target_count >= int(args.min_natural_targets)),
        "single_target_total": int(target_total),
        "single_improved_total": int(improved_total),
        "single_degraded": int(degraded),
        "single_delta_sum": float(delta_sum),
        "single_donor_classes": {str(row.get("donor_variant")): row.get("patched_boundary_class") for row in rows},
        "single_donor_text": {str(row.get("donor_variant")): row.get("patched_generated") for row in rows},
    }


def load_component_groups(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_827 / args.source_round / f"phase827_{model_name}_rows.jsonl")
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("component_kind")) not in set(parse_csv(args.component_kinds)):
            continue
        if int(row.get("budget", 0)) not in {int(x) for x in parse_csv(args.budgets)}:
            continue
        key = (
            row.get("case_id"),
            int(row.get("layer_idx")),
            row.get("component_kind"),
            row.get("component_label"),
            int(row.get("budget")),
        )
        by_key[key].append(row)
    groups = [summarize_group(vals, args) for vals in by_key.values() if vals]
    if not args.include_weak_groups:
        groups = [
            group
            for group in groups
            if group["single_target_total"] > 0
            or group["single_improved_total"] > 0
            or group["single_exact_target"]
            or group["single_natural_target_count"] > 0
        ]
    groups.sort(
        key=lambda g: (
            int(g["single_exact_plus_multi"]),
            int(g["single_exact_target"]),
            int(g["single_natural_target_count"]),
            int(g["single_target_total"]),
            -int(g["single_degraded"]),
            finite(g["single_delta_sum"]),
            str(g["case_id"]),
            compact_component_label(g),
        ),
        reverse=True,
    )
    if int(args.max_component_groups) > 0:
        groups = groups[: int(args.max_component_groups)]
    return groups


def build_pair_specs(groups: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_case_budget: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_case_budget[(str(group["case_id"]), int(group["budget"]))].append(group)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for vals in by_case_budget.values():
        for a, b in itertools.combinations(vals, 2):
            if (int(a["layer_idx"]), a["component_kind"], a["component_label"]) == (
                int(b["layer_idx"]),
                b["component_kind"],
                b["component_label"],
            ):
                continue
            pairs.append((a, b))

    def pair_score(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int, int, int, int, float]:
        a, b = pair
        exact_plus = int(a["single_exact_plus_multi"]) + int(b["single_exact_plus_multi"])
        exact = int(a["single_exact_target"]) + int(b["single_exact_target"])
        natural = int(a["single_natural_target_count"]) + int(b["single_natural_target_count"])
        target = int(a["single_target_total"]) + int(b["single_target_total"])
        degraded = int(a["single_degraded"]) + int(b["single_degraded"])
        delta = finite(a["single_delta_sum"]) + finite(b["single_delta_sum"])
        return (exact_plus, exact, natural, target, -degraded, delta)

    pairs.sort(key=pair_score, reverse=True)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    return pairs


def component_spec_from_group(group: dict[str, Any]) -> dict[str, Any]:
    spec = {
        "component_kind": group["component_kind"],
        "component_label": group["component_label"],
        "head_id": group.get("head_id"),
        "channel_group_size": group.get("channel_group_size"),
    }
    if group["component_kind"] == "attention_head":
        label = str(group["component_label"])
        head_id = int(group.get("head_id") if group.get("head_id") is not None else label.split("_")[-1])
        spec["head_id"] = head_id
    return spec


def spec_from_source_row(group: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    spec = p823.component_spec_from_row(source_row)
    if group["component_kind"] == "attention_head" and "head_dim" not in spec:
        spec.update(component_spec_from_group(group))
    return spec


def install_multi_patch(model, patch_items: list[dict[str, Any]], alpha: float) -> list[Any]:
    handles = []
    for item in patch_items:
        handles.append(
            p823.install_subspace_patch(
                model,
                int(item["layer_idx"]),
                item["spec"],
                item["recipient_vec"],
                item["donor_vec"],
                item["selected_indices"],
                float(alpha),
            )
        )
    return handles


def greedy_generate_with_multi_patch(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    patch_items: list[dict[str, Any]],
    max_new_tokens: int,
    alpha: float,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles: list[Any] = []
        if step == 0 and patch_items:
            handles = install_multi_patch(model, patch_items, alpha)
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


def group_from_phase822(model_name: str, group: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    source_rows = p825.select_source_rows(model_name, args)
    for row in source_rows:
        if (
            row.get("case_id") == group["case_id"]
            and int(row.get("layer_idx")) == int(group["layer_idx"])
            and row.get("component_kind") == group["component_kind"]
            and row.get("component_label") == group["component_label"]
        ):
            return row
    return None


def eval_pair(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    pair: tuple[dict[str, Any], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p820.standard_lookup(standards)
    recipient_prompt = p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p823.greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = p825.boundary_for(lookup, case["case_id"], baseline_text)
    rows: list[dict[str, Any]] = []
    comp_data = []
    for group in pair:
        source_row = group_from_phase822(args.model, group, args)
        if source_row is None:
            return []
        spec = spec_from_source_row(group, source_row)
        recipient_state = p822.capture_component_state(model, tokenizer, device, recipient_prompt, int(group["layer_idx"]))
        recipient_vec = p823.component_vector(recipient_state, spec)
        if recipient_vec is None:
            return []
        comp_data.append(
            {
                "group": group,
                "source_row": source_row,
                "spec": spec,
                "recipient_vec": recipient_vec.float().cpu(),
                "selected_indices": [int(x) for x in group["selected_indices"]],
            }
        )

    for donor_variant in parse_csv(args.search_donor_prompts):
        patch_items = []
        for item in comp_data:
            donor_prompt = p825.natural_prompt(case, donor_variant)
            donor_state = p822.capture_component_state(model, tokenizer, device, donor_prompt, int(item["group"]["layer_idx"]))
            donor_vec = p823.component_vector(donor_state, item["spec"])
            if donor_vec is None:
                patch_items = []
                break
            patch_items.append(
                {
                    "layer_idx": int(item["group"]["layer_idx"]),
                    "spec": item["spec"],
                    "recipient_vec": item["recipient_vec"],
                    "donor_vec": donor_vec.float().cpu(),
                    "selected_indices": item["selected_indices"],
                }
            )
        if not patch_items:
            continue
        patched_text, patched_ids = greedy_generate_with_multi_patch(
            model,
            tokenizer,
            device,
            recipient_ids,
            patch_items,
            args.max_new_tokens,
            args.alpha,
        )
        patched_boundary = p825.boundary_for(lookup, case["case_id"], patched_text)
        delta_rank = int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"])
        a, b = pair
        rows.append(
            {
                "row_kind": "phase828_cross_component_consistency_fiber_composition",
                "phase": PHASE,
                "model": args.model,
                "round": args.round_name,
                "source_round": args.source_round,
                "case_id": case["case_id"],
                "object": case["object"],
                "target_answer": case["answer"],
                "pair_label": f"{compact_component_label(a)} + {compact_component_label(b)}",
                "component_a": a,
                "component_b": b,
                "budget": int(a["budget"]),
                "donor_variant": donor_variant,
                "validation_mode": "cross_component_composition",
                "baseline_generated": p825.clean_generated(baseline_text),
                "baseline_token_ids": baseline_ids,
                "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
                "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
                "patched_generated": p825.clean_generated(patched_text),
                "patched_token_ids": patched_ids,
                "patched_boundary_class": patched_boundary.get("final_boundary_class"),
                "patched_boundary_rank": int(patched_boundary["boundary_rank"]),
                "delta_boundary_rank": delta_rank,
                "improved_boundary": delta_rank > 0,
                "degraded_boundary": delta_rank < 0,
                "target_transition": patched_boundary.get("final_boundary_class") == "target_equivalent",
                "single_a_exact_plus_multi": bool(a["single_exact_plus_multi"]),
                "single_b_exact_plus_multi": bool(b["single_exact_plus_multi"]),
                "single_a_natural_target_count": int(a["single_natural_target_count"]),
                "single_b_natural_target_count": int(b["single_natural_target_count"]),
                "n_components": 2,
            }
        )
    return rows


def pair_counts(rows: list[dict[str, Any]], min_natural_targets: int) -> dict[str, Any]:
    exact_target = any(row.get("donor_variant") == "exact_choices" and row.get("target_transition") for row in rows)
    natural_target_count = sum(1 for row in rows if row.get("donor_variant") != "exact_choices" and row.get("target_transition"))
    degraded = sum(1 for row in rows if row.get("degraded_boundary"))
    target_total = sum(1 for row in rows if row.get("target_transition"))
    return {
        "exact_target": bool(exact_target),
        "natural_target_count": int(natural_target_count),
        "exact_plus_any": bool(exact_target and natural_target_count >= 1),
        "exact_plus_multi": bool(exact_target and natural_target_count >= int(min_natural_targets)),
        "degraded": int(degraded),
        "target_total": int(target_total),
    }


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in rows) / len(rows) if rows else None,
        "patched_classes": dict(Counter(row.get("patched_boundary_class") for row in rows)),
    }


def summarize_rows(rows: list[dict[str, Any]], groups: list[dict[str, Any]], pairs: list[tuple[dict[str, Any], dict[str, Any]]], args: argparse.Namespace, attn_impl: str | None = None) -> dict[str, Any]:
    by_donor = defaultdict(list)
    by_pair = defaultdict(list)
    for row in rows:
        by_donor[str(row.get("donor_variant"))].append(row)
        by_pair[str(row.get("pair_label"))].append(row)
    pair_records = []
    composition_new_exact_multi = 0
    composition_preserve_exact_multi = 0
    pair_exact_multi = 0
    for label, vals in by_pair.items():
        counts = pair_counts(vals, int(args.min_natural_targets))
        a = vals[0]["component_a"]
        b = vals[0]["component_b"]
        single_had_exact_multi = bool(a["single_exact_plus_multi"] or b["single_exact_plus_multi"])
        if counts["exact_plus_multi"]:
            pair_exact_multi += 1
            if single_had_exact_multi:
                composition_preserve_exact_multi += 1
            else:
                composition_new_exact_multi += 1
        pair_records.append(
            {
                "pair_label": label,
                **counts,
                "single_had_exact_multi": single_had_exact_multi,
                "single_a_natural_target_count": int(a["single_natural_target_count"]),
                "single_b_natural_target_count": int(b["single_natural_target_count"]),
            }
        )
    natural_rows = [row for row in rows if row.get("donor_variant") != "exact_choices"]
    exact_rows = [row for row in rows if row.get("donor_variant") == "exact_choices"]
    return {
        "phase": PHASE,
        "title": "Cross-Component Consistency Fiber Composition",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_component_groups": len(groups),
        "n_pairs": len(pair_records),
        "exact_target_rows": sum(1 for row in exact_rows if row.get("target_transition")),
        "natural_target_rows": sum(1 for row in natural_rows if row.get("target_transition")),
        "natural_degraded_rows": sum(1 for row in natural_rows if row.get("degraded_boundary")),
        "pair_exact_plus_multi": pair_exact_multi,
        "composition_new_exact_multi": composition_new_exact_multi,
        "composition_preserve_exact_multi": composition_preserve_exact_multi,
        "donor_summary": {donor: compact(vals) for donor, vals in sorted(by_donor.items())},
        "pair_records": pair_records[:120],
        "selected_component_groups": [
            {
                "label": compact_component_label(group),
                "case_id": group["case_id"],
                "single_exact_plus_multi": group["single_exact_plus_multi"],
                "single_exact_target": group["single_exact_target"],
                "single_natural_target_count": group["single_natural_target_count"],
                "single_degraded": group["single_degraded"],
            }
            for group in groups
        ],
        "boundary": "This phase composes Phase 827 single-component fibers and tests whether consistency survives or improves under simultaneous multi-component patching.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p820.standard_rows()
    groups = load_component_groups(args.model, args)
    pairs = build_pair_specs(groups, args)
    cmap = p825.case_map()
    log(f"{args.model}/{args.round_name}: groups={len(groups)} pairs={len(pairs)}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "groups": [compact_component_label(group) for group in groups],
                    "pairs": [[compact_component_label(a), compact_component_label(b)] for a, b in pairs],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"groups": groups, "pairs": pairs}
    if not pairs:
        summary = summarize_rows([], groups, pairs, args, attn_impl=None)
        summary["skipped_model_load"] = True
        summary["skip_reason"] = "no eligible same-case cross-component pairs"
        write_jsonl(out_dir / f"phase828_{args.model}_rows.jsonl", [])
        write_json(out_dir / f"phase828_{args.model}_summary.json", summary)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "round": args.round_name,
                    "groups": summary["n_component_groups"],
                    "pairs": summary["n_pairs"],
                    "skipped_model_load": True,
                    "skip_reason": summary["skip_reason"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return summary
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict[str, Any]] = []
    try:
        for idx, pair in enumerate(pairs, 1):
            case = cmap.get(str(pair[0]["case_id"]))
            if not case:
                continue
            rows.extend(eval_pair(model, tokenizer, device, standards, case, pair, args))
            if idx % int(args.log_every) == 0 or idx == len(pairs):
                log(f"{args.model}: evaluated {idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, groups, pairs, args, attn_impl)
    write_jsonl(out_dir / f"phase828_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase828_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "groups": summary["n_component_groups"],
                "pairs": summary["n_pairs"],
                "exact_target_rows": summary["exact_target_rows"],
                "natural_target_rows": summary["natural_target_rows"],
                "natural_degraded_rows": summary["natural_degraded_rows"],
                "pair_exact_plus_multi": summary["pair_exact_plus_multi"],
                "composition_new_exact_multi": summary["composition_new_exact_multi"],
                "composition_preserve_exact_multi": summary["composition_preserve_exact_multi"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 828 Cross-Component Consistency Fiber Composition ({payload['round']})",
        "",
        "- Source: Phase 827 selected subspaces.",
        "- Objective: simultaneous two-component patching, then exact/natural consistency audit.",
        "",
        "## Model Summary",
        "",
        "| model | groups | pairs | exact target | natural target | natural degraded | pair exact+multi | new exact+multi | preserve exact+multi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_component_groups')} | {data.get('n_pairs')} | "
            f"{data.get('exact_target_rows')} | {data.get('natural_target_rows')} | "
            f"{data.get('natural_degraded_rows')} | {data.get('pair_exact_plus_multi')} | "
            f"{data.get('composition_new_exact_multi')} | {data.get('composition_preserve_exact_multi')} |"
        )
    lines += ["", "## Donor Summary", ""]
    lines += [
        "| model | donor | n | improved | target | degraded | mean delta | classes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for donor, row in sorted((data.get("donor_summary") or {}).items()):
            lines.append(
                f"| {model_name} | `{donor}` | {row.get('n')} | {row.get('improved_rows')} | "
                f"{row.get('target_transition_rows')} | {row.get('degraded_rows')} | "
                f"{finite(row.get('mean_delta_boundary_rank')):.3f} | "
                f"`{json.dumps(row.get('patched_classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Pair Records", ""]
    lines += [
        "| model | pair | exact | natural count | exact+multi | degraded | single had exact+multi |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for rec in data.get("pair_records") or []:
            lines.append(
                f"| {model_name} | `{rec.get('pair_label')}` | {int(bool(rec.get('exact_target')))} | "
                f"{rec.get('natural_target_count')} | {int(bool(rec.get('exact_plus_multi')))} | "
                f"{rec.get('degraded')} | {int(bool(rec.get('single_had_exact_multi')))} |"
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
        path = out_dir / f"phase828_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase828_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase828_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--search-donor-prompts", default="exact_choices,natural_category,natural_question,object_only")
    parser.add_argument("--min-natural-targets", type=int, default=2)
    parser.add_argument("--budgets", default="16,32")
    parser.add_argument("--max-component-groups", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--include-weak-groups", action="store_true")
    parser.add_argument("--component-kinds", default="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group")
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
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
