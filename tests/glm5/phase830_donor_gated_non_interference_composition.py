#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase829_non_interference_constrained_component_composition as p829  # noqa: E402


PHASE = 830
RESULT_ROOT = Path("tests/result/phase830_donor_gated_non_interference_composition")


def log(msg: str) -> None:
    p829.log(msg)


def donor_gate_accept(group: dict[str, Any], donor_variant: str, args: argparse.Namespace) -> bool:
    cls = str((group.get("single_donor_classes") or {}).get(donor_variant))
    if args.gating_mode == "target_only":
        return cls == "target_equivalent"
    if args.gating_mode == "not_unknown":
        return cls not in {"unknown_other", "object_echo", "format_echo", "None", ""}
    if args.gating_mode == "all_safe":
        return int(group.get("single_degraded", 0)) <= int(args.max_single_degraded)
    raise ValueError(f"unknown gating_mode: {args.gating_mode}")


def prepare_component_data(
    model,
    tokenizer,
    device: torch.device,
    recipient_prompt: str,
    pair: tuple[dict[str, Any], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    comp_data = []
    for group in pair:
        source_row = p828.group_from_phase822(args.model, group, args)
        if source_row is None:
            return []
        spec = p828.spec_from_source_row(group, source_row)
        recipient_state = p828.p822.capture_component_state(
            model, tokenizer, device, recipient_prompt, int(group["layer_idx"])
        )
        recipient_vec = p828.p823.component_vector(recipient_state, spec)
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
    return comp_data


def eval_pair(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    pair: tuple[dict[str, Any], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p828.p820.standard_lookup(standards)
    recipient_prompt = p828.p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p828.p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p828.p823.greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = p828.p825.boundary_for(lookup, case["case_id"], baseline_text)
    comp_data = prepare_component_data(model, tokenizer, device, recipient_prompt, pair, args)
    if not comp_data:
        return []

    rows: list[dict[str, Any]] = []
    a, b = pair
    for donor_variant in p828.parse_csv(args.search_donor_prompts):
        patch_items = []
        active_labels = []
        for item in comp_data:
            group = item["group"]
            if not donor_gate_accept(group, donor_variant, args):
                continue
            donor_prompt = p828.p825.natural_prompt(case, donor_variant)
            donor_state = p828.p822.capture_component_state(
                model, tokenizer, device, donor_prompt, int(group["layer_idx"])
            )
            donor_vec = p828.p823.component_vector(donor_state, item["spec"])
            if donor_vec is None:
                continue
            patch_items.append(
                {
                    "layer_idx": int(group["layer_idx"]),
                    "spec": item["spec"],
                    "recipient_vec": item["recipient_vec"],
                    "donor_vec": donor_vec.float().cpu(),
                    "selected_indices": item["selected_indices"],
                }
            )
            active_labels.append(p828.compact_component_label(group))

        patched_text, patched_ids = p828.greedy_generate_with_multi_patch(
            model,
            tokenizer,
            device,
            recipient_ids,
            patch_items,
            args.max_new_tokens,
            args.alpha,
        )
        patched_boundary = p828.p825.boundary_for(lookup, case["case_id"], patched_text)
        delta_rank = int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"])
        rows.append(
            {
                "row_kind": "phase830_donor_gated_non_interference_composition",
                "phase": PHASE,
                "model": args.model,
                "round": args.round_name,
                "source_round": args.source_round,
                "case_id": case["case_id"],
                "object": case["object"],
                "target_answer": case["answer"],
                "pair_label": f"{p828.compact_component_label(a)} + {p828.compact_component_label(b)}",
                "component_a": a,
                "component_b": b,
                "budget": int(a["budget"]),
                "donor_variant": donor_variant,
                "validation_mode": "donor_gated_non_interference_composition",
                "gating_mode": args.gating_mode,
                "active_component_labels": active_labels,
                "n_active_components": len(active_labels),
                "baseline_generated": p828.p825.clean_generated(baseline_text),
                "baseline_token_ids": baseline_ids,
                "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
                "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
                "patched_generated": p828.p825.clean_generated(patched_text),
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


def summarize_rows(
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
    attn_impl: str | None = None,
) -> dict[str, Any]:
    summary = p829.summarize_rows(rows, groups, pairs, diagnostics, args, attn_impl)
    summary["phase"] = PHASE
    summary["title"] = "Donor-Gated Non-Interference Composition"
    summary["boundary"] = (
        "This phase tests whether donor-specific gating can remove the remaining "
        "interference found after Phase 829 global safety filtering."
    )
    summary["gating_mode"] = args.gating_mode
    active_counts = defaultdict(int)
    for row in rows:
        active_counts[str(row.get("n_active_components"))] += 1
    summary["active_component_count_distribution"] = dict(active_counts)
    return summary


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p828.p820.standard_rows()
    groups = p828.load_component_groups(args.model, args)
    pairs, diagnostics = p829.build_constrained_pairs(groups, args)
    cmap = p828.p825.case_map()
    log(
        f"{args.model}/{args.round_name}: groups={len(groups)} "
        f"eligible={diagnostics['eligible_component_groups']} pairs={len(pairs)} gating={args.gating_mode}"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "diagnostics": diagnostics,
                    "pairs": [[p828.compact_component_label(a), p828.compact_component_label(b)] for a, b in pairs],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"groups": groups, "pairs": pairs, "diagnostics": diagnostics}
    if not pairs:
        summary = summarize_rows([], groups, pairs, diagnostics, args, attn_impl=None)
        summary["skipped_model_load"] = True
        summary["skip_reason"] = "no eligible donor-gated same-case cross-component pairs"
        p828.write_jsonl(out_dir / f"phase830_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase830_{args.model}_summary.json", summary)
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

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(
        args.model, args.attn_implementations
    )
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
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, groups, pairs, diagnostics, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase830_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase830_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "pairs": summary["n_pairs"],
                "exact_target_rows": summary["exact_target_rows"],
                "natural_target_rows": summary["natural_target_rows"],
                "natural_degraded_rows": summary["natural_degraded_rows"],
                "natural_category_degraded_rows": summary["natural_category_degraded_rows"],
                "pair_exact_plus_multi": summary["pair_exact_plus_multi"],
                "composition_new_exact_multi": summary["composition_new_exact_multi"],
                "composition_preserve_exact_multi": summary["composition_preserve_exact_multi"],
                "active_component_count_distribution": summary["active_component_count_distribution"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 830 Donor-Gated Non-Interference Composition ({payload['round']})",
        "",
        "- Source: Phase 829 selected non-interfering pairs.",
        "- Objective: activate only donor-compatible components inside each pair.",
        "",
        "## Model Summary",
        "",
        "| model | pairs | exact target | natural target | natural degraded | natural_category degraded | pair exact+multi | new exact+multi | preserve exact+multi | active components |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_pairs')} | {data.get('exact_target_rows')} | "
            f"{data.get('natural_target_rows')} | {data.get('natural_degraded_rows')} | "
            f"{data.get('natural_category_degraded_rows')} | {data.get('pair_exact_plus_multi')} | "
            f"{data.get('composition_new_exact_multi')} | {data.get('composition_preserve_exact_multi')} | "
            f"`{json.dumps(data.get('active_component_count_distribution') or {}, ensure_ascii=False)}` |"
        )

    lines += ["", "## Donor Summary", ""]
    lines += [
        "| model | donor | n | improved | target | degraded | mean delta | classes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for donor, row in sorted((data.get("donor_summary") or {}).items()):
            lines.append(
                f"| {model_name} | `{donor}` | {row.get('n')} | {row.get('improved_rows')} | "
                f"{row.get('target_transition_rows')} | {row.get('degraded_rows')} | "
                f"{p829.finite(row.get('mean_delta_boundary_rank')):.3f} | "
                f"`{json.dumps(row.get('patched_classes') or {}, ensure_ascii=False)}` |"
            )

    lines += ["", "## Pair Records", ""]
    lines += [
        "| model | case | pair | exact | natural count | exact+multi | degraded | single had exact+multi |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for rec in data.get("pair_records") or []:
            lines.append(
                f"| {model_name} | `{rec.get('case_id')}` | `{rec.get('pair_label')}` | "
                f"{int(bool(rec.get('exact_target')))} | {rec.get('natural_target_count')} | "
                f"{int(bool(rec.get('exact_plus_multi')))} | {rec.get('degraded')} | "
                f"{int(bool(rec.get('single_had_exact_multi')))} |"
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
        path = out_dir / f"phase830_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase830_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase830_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p829.build_parser()
    parser.add_argument("--gating-mode", default="target_only", choices=["target_only", "not_unknown", "all_safe"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(
            json.dumps(
                {"round": args.round_name, "status": payload["status"], "models": payload["models"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    run_model(args)


if __name__ == "__main__":
    main()
