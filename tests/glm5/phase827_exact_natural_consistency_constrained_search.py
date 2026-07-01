#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
import phase824_boundary_objective_sparse_subspace_search as p824  # noqa: E402
import phase825_candidate_pool_expansion_natural_route_validation as p825  # noqa: E402
import phase826_multi_donor_natural_objective_sparse_search as p826  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402


PHASE = 827
RESULT_ROOT = Path("tests/result/phase827_exact_natural_consistency_constrained_search")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return sorted({int(x) for x in parse_csv(text) if int(x) > 0})


def finite(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def donor_counts(rows: list[dict[str, Any]], min_natural_targets: int) -> dict[str, Any]:
    exact_rows = [row for row in rows if row.get("donor_variant") == "exact_choices"]
    natural_rows = [row for row in rows if row.get("donor_variant") != "exact_choices"]
    exact_target = any(row.get("target_transition") for row in exact_rows)
    natural_target_count = sum(1 for row in natural_rows if row.get("target_transition"))
    degraded = sum(1 for row in rows if row.get("degraded_boundary"))
    target_total = sum(1 for row in rows if row.get("target_transition"))
    improved_total = sum(1 for row in rows if row.get("improved_boundary"))
    delta_sum = sum(finite(row.get("delta_boundary_rank")) for row in rows)
    return {
        "exact_target": exact_target,
        "natural_target_count": natural_target_count,
        "exact_plus_multi_natural": bool(exact_target and natural_target_count >= min_natural_targets),
        "multi_natural": bool(natural_target_count >= min_natural_targets),
        "degraded": degraded,
        "target_total": target_total,
        "improved_total": improved_total,
        "delta_sum": delta_sum,
    }


def aggregate_key(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, int, int, int, int, int, float]:
    counts = donor_counts(rows, int(args.min_natural_targets))
    return (
        int(counts["exact_plus_multi_natural"]),
        int(counts["multi_natural"]),
        int(counts["exact_target"]),
        -int(counts["degraded"]),
        int(counts["target_total"]),
        int(counts["improved_total"]),
        float(counts["delta_sum"]),
    )


def make_row(
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    spec: dict[str, Any],
    donor_variant: str,
    baseline_boundary: dict[str, Any],
    baseline_text: str,
    baseline_ids: list[int],
    budget: int,
    selected: list[int],
    meta: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    selected_full = sorted({int(x) for x in selected})
    return {
        "row_kind": "phase827_exact_natural_consistency_constrained_search",
        "phase": PHASE,
        "source_phase": 822,
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "case_id": case["case_id"],
        "object": case["object"],
        "target_answer": case["answer"],
        "layer_idx": int(source_row["layer_idx"]),
        "component_kind": spec.get("component_kind"),
        "component_label": spec.get("component_label"),
        "head_id": spec.get("head_id"),
        "channel_group_size": spec.get("channel_group_size"),
        "source_phase822_role": source_row.get("role_label"),
        "source_phase822_boundary_class": source_row.get("patched_boundary_class"),
        "source_phase822_delta_rank": source_row.get("delta_boundary_rank"),
        "validation_mode": "exact_natural_consistency_constrained",
        "donor_variant": donor_variant,
        "budget": int(budget),
        "n_selected": len(selected_full),
        "selected_indices": selected_full[: int(args.save_indices_limit)],
        "selected_indices_full": selected_full,
        "baseline_generated": p825.clean_generated(baseline_text),
        "baseline_token_ids": baseline_ids,
        "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
        "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
        "meta": meta,
        **result,
    }


def eval_selected_for_donors(
    model,
    tokenizer,
    device: torch.device,
    lookup: dict[tuple[str, str], dict[str, Any]],
    case: dict[str, Any],
    source_row: dict[str, Any],
    spec: dict[str, Any],
    recipient_ids: list[int],
    baseline_boundary: dict[str, Any],
    baseline_text: str,
    baseline_ids: list[int],
    recipient_vec: torch.Tensor,
    donor_vecs: dict[str, torch.Tensor],
    selected: list[int],
    budget: int,
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for donor_variant, donor_vec in donor_vecs.items():
        result = p824.eval_patch(
            model,
            tokenizer,
            device,
            lookup,
            case,
            recipient_ids,
            baseline_boundary,
            int(source_row["layer_idx"]),
            spec,
            recipient_vec,
            donor_vec,
            selected,
            args,
        )
        rows.append(
            make_row(
                args,
                case,
                source_row,
                spec,
                donor_variant,
                baseline_boundary,
                baseline_text,
                baseline_ids,
                budget,
                selected,
                meta,
                result,
            )
        )
    return rows


def greedy_consistency(
    model,
    tokenizer,
    device: torch.device,
    lookup: dict[tuple[str, str], dict[str, Any]],
    case: dict[str, Any],
    source_row: dict[str, Any],
    spec: dict[str, Any],
    recipient_ids: list[int],
    baseline_boundary: dict[str, Any],
    baseline_text: str,
    baseline_ids: list[int],
    recipient_vec: torch.Tensor,
    donor_vecs: dict[str, torch.Tensor],
    groups: list[list[int]],
    budgets: list[int],
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_budget = max(budgets) if budgets else 0
    selected: list[int] = []
    remaining = [list(group) for group in groups]
    seen_budgets: set[int] = set()
    steps = 0
    while remaining and len(selected) < max_budget and steps < int(args.greedy_steps):
        best = None
        for idx, group in enumerate(remaining):
            candidate = sorted(set(selected + group))
            eval_rows = eval_selected_for_donors(
                model,
                tokenizer,
                device,
                lookup,
                case,
                source_row,
                spec,
                recipient_ids,
                baseline_boundary,
                baseline_text,
                baseline_ids,
                recipient_vec,
                donor_vecs,
                candidate,
                min(len(candidate), max_budget),
                args,
                {**meta, "candidate_group_count": len(groups), "greedy_candidate_eval": True},
            )
            key = aggregate_key(eval_rows, args)
            if best is None or key > best[0]:
                best = (key, idx, candidate)
        if best is None:
            break
        _key, idx, selected = best
        remaining.pop(idx)
        steps += 1
        for budget in budgets:
            if budget in seen_budgets:
                continue
            if len(selected) >= min(int(budget), max_budget):
                seen_budgets.add(budget)
                rows.extend(
                    eval_selected_for_donors(
                        model,
                        tokenizer,
                        device,
                        lookup,
                        case,
                        source_row,
                        spec,
                        recipient_ids,
                        baseline_boundary,
                        baseline_text,
                        baseline_ids,
                        recipient_vec,
                        donor_vecs,
                        selected[: int(budget)],
                        int(budget),
                        args,
                        {**meta, "greedy_steps": steps, "candidate_group_count": len(groups)},
                    )
                )
    for budget in budgets:
        if selected and budget not in seen_budgets:
            rows.extend(
                eval_selected_for_donors(
                    model,
                    tokenizer,
                    device,
                    lookup,
                    case,
                    source_row,
                    spec,
                    recipient_ids,
                    baseline_boundary,
                    baseline_text,
                    baseline_ids,
                    recipient_vec,
                    donor_vecs,
                    selected[: int(budget)],
                    int(budget),
                    args,
                    {**meta, "greedy_steps": steps, "candidate_group_count": len(groups)},
                )
            )
    return rows


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
    spec = p823.component_spec_from_row(source_row)
    recipient_prompt = p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p823.greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = p825.boundary_for(lookup, case["case_id"], baseline_text)
    recipient_state = p822.capture_component_state(model, tokenizer, device, recipient_prompt, layer_idx)
    recipient_vec = p823.component_vector(recipient_state, spec)
    effective_dir, readout_meta = p823.effective_readout_direction(model, tokenizer, case, baseline_ids, layer_idx, spec)
    if recipient_vec is None or effective_dir is None:
        return []
    recipient_vec = recipient_vec.float().cpu()
    effective_dir = effective_dir.float().cpu()
    n = min(int(recipient_vec.numel()), int(effective_dir.numel()))
    if n <= 0:
        return []
    recipient_vec = recipient_vec[:n]
    effective_dir = effective_dir[:n]
    donor_vecs: dict[str, torch.Tensor] = {}
    for donor_variant in parse_csv(args.search_donor_prompts):
        donor_prompt = p825.natural_prompt(case, donor_variant)
        donor_state = p822.capture_component_state(model, tokenizer, device, donor_prompt, layer_idx)
        donor_vec = p823.component_vector(donor_state, spec)
        if donor_vec is None:
            continue
        donor_vecs[donor_variant] = donor_vec.float().cpu()[:n]
    if not donor_vecs:
        return []
    budgets = [min(int(budget), n) for budget in parse_int_csv(args.budgets)]
    budgets = sorted({budget for budget in budgets if budget > 0})
    if not budgets:
        budgets = [min(n, int(args.search_group_size))]
    pool, pool_meta = p826.union_candidate_pool(args, source_row, recipient_vec, donor_vecs, effective_dir)
    groups = p824.chunked(pool, args.search_group_size)
    rows = greedy_consistency(
        model,
        tokenizer,
        device,
        lookup,
        case,
        source_row,
        spec,
        recipient_ids,
        baseline_boundary,
        baseline_text,
        baseline_ids,
        recipient_vec,
        donor_vecs,
        groups,
        budgets,
        args,
        {**pool_meta, "readout_meta": readout_meta},
    )
    for row in rows:
        row["n_component_dims"] = n
        row["expanded_candidate_pool"] = pool[: int(args.save_indices_limit)]
        row["expanded_candidate_pool_size"] = len(pool)
        row["search_donor_prompts"] = list(donor_vecs.keys())
        row["min_natural_targets"] = int(args.min_natural_targets)
    return rows


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in rows) / len(rows) if rows else None,
        "patched_classes": dict(Counter(row.get("patched_boundary_class") for row in rows)),
    }


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None = None) -> dict[str, Any]:
    by_donor = defaultdict(list)
    by_budget = defaultdict(list)
    by_component = defaultdict(list)
    for row in rows:
        by_donor[str(row.get("donor_variant"))].append(row)
        by_budget[str(row.get("budget"))].append(row)
        by_component[(str(row.get("component_kind")), str(row.get("donor_variant")))].append(row)
    exact_rows = [row for row in rows if row.get("donor_variant") == "exact_choices"]
    natural_rows = [row for row in rows if row.get("donor_variant") != "exact_choices"]
    by_pair = defaultdict(list)
    for row in rows:
        key = (
            row.get("case_id"),
            row.get("layer_idx"),
            row.get("component_kind"),
            row.get("component_label"),
            row.get("budget"),
        )
        by_pair[key].append(row)
    exact_plus_any = 0
    exact_plus_multi = 0
    multi_natural = 0
    consistency_records = []
    for key, vals in by_pair.items():
        counts = donor_counts(vals, int(args.min_natural_targets))
        if counts["multi_natural"]:
            multi_natural += 1
        if counts["exact_target"] and counts["natural_target_count"] >= 1:
            exact_plus_any += 1
        if counts["exact_plus_multi_natural"]:
            exact_plus_multi += 1
        if counts["exact_target"] or counts["natural_target_count"]:
            consistency_records.append({"key": key, **counts})
    return {
        "phase": PHASE,
        "title": "Exact-Natural Consistency Constrained Search",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "min_natural_targets": int(args.min_natural_targets),
        "n_rows": len(rows),
        "n_source_components": len(
            {
                (row.get("case_id"), row.get("layer_idx"), row.get("component_kind"), row.get("component_label"))
                for row in rows
            }
        ),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "exact_target_rows": sum(1 for row in exact_rows if row.get("target_transition")),
        "natural_target_rows": sum(1 for row in natural_rows if row.get("target_transition")),
        "natural_improved_rows": sum(1 for row in natural_rows if row.get("improved_boundary")),
        "natural_degraded_rows": sum(1 for row in natural_rows if row.get("degraded_boundary")),
        "multi_natural_target_pairs": multi_natural,
        "exact_plus_any_natural_target_pairs": exact_plus_any,
        "exact_plus_multi_natural_target_pairs": exact_plus_multi,
        "donor_summary": {donor: compact(vals) for donor, vals in sorted(by_donor.items())},
        "budget_summary": {budget: compact(vals) for budget, vals in sorted(by_budget.items())},
        "component_donor_summary": {f"{kind}/{donor}": compact(vals) for (kind, donor), vals in sorted(by_component.items())},
        "consistency_records": consistency_records[:80],
        "boundary": "This phase prioritizes exact target plus multi-natural target consistency before natural-only gains.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p820.standard_rows()
    selected = p825.select_source_rows(args.model, args)
    cmap = p825.case_map()
    log(f"{args.model}/{args.round_name}: selected Phase 822 source components={len(selected)}")
    if args.dry_run:
        payload = {"model": args.model, "selected": selected}
        print(json.dumps({"model": args.model, "n_selected": len(selected)}, ensure_ascii=False, indent=2))
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
                log(f"{args.model}: searched {idx}/{len(selected)} source components; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase827_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase827_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "source_components": summary["n_source_components"],
                "rows": summary["n_rows"],
                "exact_target_rows": summary["exact_target_rows"],
                "natural_target_rows": summary["natural_target_rows"],
                "natural_degraded_rows": summary["natural_degraded_rows"],
                "exact_plus_any_natural_target_pairs": summary["exact_plus_any_natural_target_pairs"],
                "exact_plus_multi_natural_target_pairs": summary["exact_plus_multi_natural_target_pairs"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 827 Exact-Natural Consistency Constrained Search ({payload['round']})",
        "",
        "- Source: Phase 822 signal-bearing components.",
        "- Objective: exact+multi-natural consistency before natural-only target gains.",
        "",
        "## Model Summary",
        "",
        "| model | source comps | rows | exact target | natural target | natural degraded | exact+any natural | exact+multi natural | multi-natural |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_source_components')} | {data.get('n_rows')} | "
            f"{data.get('exact_target_rows')} | {data.get('natural_target_rows')} | "
            f"{data.get('natural_degraded_rows')} | {data.get('exact_plus_any_natural_target_pairs')} | "
            f"{data.get('exact_plus_multi_natural_target_pairs')} | {data.get('multi_natural_target_pairs')} |"
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
        path = out_dir / f"phase827_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase827_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase827_cross_model_summary.md", payload)
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
    parser.add_argument("--max-source-rows", type=int, default=4, help="0 means all useful source rows.")
    parser.add_argument("--component-kinds", default="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group")
    parser.add_argument("--budgets", default="16,32")
    parser.add_argument("--base-candidate-pool", type=int, default=18)
    parser.add_argument("--expanded-candidate-pool", type=int, default=40)
    parser.add_argument("--extra-top-delta", type=int, default=16)
    parser.add_argument("--extra-top-signed", type=int, default=16)
    parser.add_argument("--extra-random", type=int, default=16)
    parser.add_argument("--search-group-size", type=int, default=4)
    parser.add_argument("--greedy-steps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--save-indices-limit", type=int, default=256)
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
