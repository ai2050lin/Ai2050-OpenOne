#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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
import phase823_beneficial_harmful_boundary_subspace_split as p823  # noqa: E402
import phase824_boundary_objective_sparse_subspace_search as p824  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402


PHASE = 825
SOURCE_822 = Path("tests/result/phase822_boundary_transition_head_mlp_decomposition")
SOURCE_823 = Path("tests/result/phase823_beneficial_harmful_boundary_subspace_split")
SOURCE_824 = Path("tests/result/phase824_boundary_objective_sparse_subspace_search")
RESULT_ROOT = Path("tests/result/phase825_candidate_pool_expansion_natural_route_validation")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return sorted({int(x) for x in parse_csv(text) if int(x) > 0})


def finite(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val


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


def natural_prompt(case: dict[str, Any], variant: str) -> str:
    if variant == "exact_choices" or variant == "no_choices":
        return p816.build_prompt(case, variant)
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
    raise ValueError(f"unknown donor variant: {variant}")


def source_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("case_id"), int(row.get("layer_idx")), row.get("component_kind"), row.get("component_label"))


def select_source_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_822 / args.source_round / f"phase822_{model_name}_rows.jsonl")
    allowed_kinds = set(parse_csv(args.component_kinds))
    useful = []
    for row in rows:
        if allowed_kinds and str(row.get("component_kind")) not in allowed_kinds:
            continue
        if row.get("target_transition") or row.get("improved_boundary") or row.get("degraded_boundary"):
            useful.append(row)
    useful.sort(
        key=lambda r: (
            bool(r.get("target_transition")),
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


def prior_success_indices(model_name: str, key: tuple[Any, ...], args: argparse.Namespace) -> tuple[list[int], dict[str, int]]:
    indices: list[int] = []
    counts = Counter()
    for phase, root, prefix in (
        (823, SOURCE_823, "phase823"),
        (824, SOURCE_824, "phase824"),
    ):
        path = root / args.source_round / f"{prefix}_{model_name}_rows.jsonl"
        for row in read_jsonl(path):
            if source_key(row) != key:
                continue
            if not (row.get("target_transition") or row.get("improved_boundary")):
                continue
            selected = [int(x) for x in row.get("selected_indices") or []]
            if not selected:
                continue
            indices.extend(selected)
            counts[f"phase{phase}_{row.get('subspace_mode') or row.get('search_mode')}"] += len(selected)
    out = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out, dict(counts)


def build_pool(
    model_name: str,
    key: tuple[Any, ...],
    delta: torch.Tensor,
    signed_scores: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    base = p824.candidate_pool(delta, signed_scores, args.base_candidate_pool, (model_name, *key, "base"))
    prior, prior_counts = prior_success_indices(model_name, key, args)
    n = int(delta.numel())
    delta_abs = delta.abs()
    signed_abs = signed_scores.abs()
    top_delta = [int(x) for x in torch.topk(delta_abs, min(n, int(args.extra_top_delta))).indices.tolist()]
    top_signed = [int(x) for x in torch.topk(signed_abs, min(n, int(args.extra_top_signed))).indices.tolist()]
    rng = random.Random(p823.stable_seed(model_name, *key, "phase825_hard_random"))
    random_ids = list(range(n))
    rng.shuffle(random_ids)
    values = []
    values.extend(prior)
    values.extend(base)
    values.extend(top_delta)
    values.extend(top_signed)
    values.extend(random_ids[: int(args.extra_random)])
    values.extend(base)
    out = []
    seen = set()
    for idx in values:
        if 0 <= int(idx) < n and int(idx) not in seen:
            out.append(int(idx))
            seen.add(int(idx))
        if len(out) >= int(args.expanded_candidate_pool):
            break
    return out, {
        "base_pool_size": len(base),
        "prior_pool_size": len(prior),
        "prior_counts": prior_counts,
        "expanded_pool_size": len(out),
        "top_delta_added": len(top_delta),
        "top_signed_added": len(top_signed),
    }


def objective_key(row: dict[str, Any]) -> tuple[int, float, float, int]:
    return p824.objective_key(row)


def make_eval_row(
    args: argparse.Namespace,
    model_name: str,
    case: dict[str, Any],
    source_row: dict[str, Any],
    spec: dict[str, Any],
    donor_variant: str,
    baseline_boundary: dict[str, Any],
    baseline_text: str,
    baseline_ids: list[int],
    mode: str,
    selected: list[int],
    budget: int,
    meta: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    selected_full = sorted({int(x) for x in selected})
    return {
        "row_kind": "phase825_candidate_pool_expansion_natural_route_validation",
        "phase": PHASE,
        "source_phase": 824,
        "model": model_name,
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
        "donor_variant": donor_variant,
        "validation_mode": mode,
        "budget": int(budget),
        "n_selected": len(selected_full),
        "selected_indices": selected_full[: int(args.save_indices_limit)],
        "selected_indices_full": selected_full,
        "meta": meta,
        "baseline_generated": clean_generated(baseline_text),
        "baseline_token_ids": baseline_ids,
        "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
        "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
        **result,
    }


def greedy_from_groups(
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
    donor_vec: torch.Tensor,
    groups: list[list[int]],
    budgets: list[int],
    args: argparse.Namespace,
    donor_variant: str,
    mode: str,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    layer_idx = int(source_row["layer_idx"])
    rows = []
    max_budget = max(budgets) if budgets else 0
    selected: list[int] = []
    remaining = [list(g) for g in groups]
    seen_budgets = set()
    steps = 0
    while remaining and len(selected) < max_budget and steps < int(args.greedy_steps):
        best = None
        for idx, group in enumerate(remaining):
            candidate = sorted(set(selected + group))
            result = p824.eval_patch(
                model,
                tokenizer,
                device,
                lookup,
                case,
                recipient_ids,
                baseline_boundary,
                layer_idx,
                spec,
                recipient_vec,
                donor_vec,
                candidate,
                args,
            )
            key = objective_key(result)
            if best is None or key > best[0]:
                best = (key, idx, candidate, result)
        if best is None:
            break
        _key, idx, selected, result = best
        remaining.pop(idx)
        steps += 1
        for budget in budgets:
            if budget in seen_budgets:
                continue
            if len(selected) >= min(int(budget), max_budget):
                seen_budgets.add(budget)
                rows.append(
                    make_eval_row(
                        args,
                        args.model,
                        case,
                        source_row,
                        spec,
                        donor_variant,
                        baseline_boundary,
                        baseline_text,
                        baseline_ids,
                        mode,
                        selected[: int(budget)],
                        int(budget),
                        {**meta, "greedy_steps": steps, "candidate_groups": len(groups)},
                        result,
                    )
                )
    for budget in budgets:
        if selected and budget not in seen_budgets:
            result = p824.eval_patch(
                model,
                tokenizer,
                device,
                lookup,
                case,
                recipient_ids,
                baseline_boundary,
                layer_idx,
                spec,
                recipient_vec,
                donor_vec,
                selected[: int(budget)],
                args,
            )
            rows.append(
                make_eval_row(
                    args,
                    args.model,
                    case,
                    source_row,
                    spec,
                    donor_variant,
                    baseline_boundary,
                    baseline_text,
                    baseline_ids,
                    mode,
                    selected[: int(budget)],
                    int(budget),
                    {**meta, "greedy_steps": steps, "candidate_groups": len(groups)},
                    result,
                )
            )
    return rows


def evaluate_fixed_indices(
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
    donor_vec: torch.Tensor,
    selected_by_budget: dict[int, list[int]],
    args: argparse.Namespace,
    donor_variant: str,
    mode: str,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for budget, selected in sorted(selected_by_budget.items()):
        if not selected:
            continue
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
            make_eval_row(
                args,
                args.model,
                case,
                source_row,
                spec,
                donor_variant,
                baseline_boundary,
                baseline_text,
                baseline_ids,
                mode,
                selected,
                budget,
                meta,
                result,
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
    recipient_prompt = natural_prompt(case, args.recipient_prompt)
    exact_donor_prompt = natural_prompt(case, args.search_donor_prompt)
    recipient_ids = p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p823.greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = boundary_for(lookup, case["case_id"], baseline_text)
    recipient_state = p822.capture_component_state(model, tokenizer, device, recipient_prompt, layer_idx)
    exact_donor_state = p822.capture_component_state(model, tokenizer, device, exact_donor_prompt, layer_idx)
    recipient_vec = p823.component_vector(recipient_state, spec)
    exact_donor_vec = p823.component_vector(exact_donor_state, spec)
    effective_dir, readout_meta = p823.effective_readout_direction(model, tokenizer, case, baseline_ids, layer_idx, spec)
    if recipient_vec is None or exact_donor_vec is None or effective_dir is None:
        return []
    recipient_vec = recipient_vec.float().cpu()
    exact_donor_vec = exact_donor_vec.float().cpu()
    effective_dir = effective_dir.float().cpu()
    n = min(int(recipient_vec.numel()), int(exact_donor_vec.numel()), int(effective_dir.numel()))
    if n <= 0:
        return []
    recipient_vec = recipient_vec[:n]
    exact_donor_vec = exact_donor_vec[:n]
    effective_dir = effective_dir[:n]
    delta = exact_donor_vec - recipient_vec
    signed_scores = delta * effective_dir
    budgets = [min(int(b), n) for b in parse_int_csv(args.budgets)]
    budgets = sorted({b for b in budgets if b > 0})
    if not budgets:
        budgets = [min(n, int(args.search_group_size))]

    key = source_key(source_row)
    pool, pool_meta = build_pool(args.model, key, delta, signed_scores, args)
    groups = p824.chunked(pool, args.search_group_size)
    rows = []
    exact_rows = greedy_from_groups(
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
        exact_donor_vec,
        groups,
        budgets,
        args,
        args.search_donor_prompt,
        "expanded_greedy_exact",
        {**pool_meta, "readout_meta": readout_meta},
    )
    rows.extend(exact_rows)
    best_by_budget: dict[int, list[int]] = {}
    for budget in budgets:
        candidates = [row for row in exact_rows if int(row.get("budget", -1)) == int(budget)]
        if not candidates:
            continue
        best = max(candidates, key=objective_key)
        best_by_budget[int(budget)] = [int(x) for x in (best.get("selected_indices_full") or best.get("selected_indices") or [])]

    for donor_variant in parse_csv(args.validation_donor_prompts):
        donor_prompt = natural_prompt(case, donor_variant)
        donor_state = p822.capture_component_state(model, tokenizer, device, donor_prompt, layer_idx)
        donor_vec = p823.component_vector(donor_state, spec)
        if donor_vec is None:
            continue
        donor_vec = donor_vec.float().cpu()[:n]
        rows.extend(
            evaluate_fixed_indices(
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
                donor_vec,
                best_by_budget,
                args,
                donor_variant,
                "natural_route_validation",
                {**pool_meta, "selected_from": args.search_donor_prompt},
            )
        )
    for row in rows:
        row["n_component_dims"] = n
        row["expanded_candidate_pool"] = pool[: int(args.save_indices_limit)]
        row["expanded_candidate_pool_size"] = len(pool)
        row["candidate_group_count"] = len(groups)
    return rows


def compact(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "improved_rows": sum(1 for row in vals if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in vals if row.get("degraded_boundary")),
        "target_transition_rows": sum(1 for row in vals if row.get("target_transition")),
        "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in vals) / len(vals) if vals else None,
        "patched_classes": dict(Counter(row.get("patched_boundary_class") for row in vals)),
    }


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None = None) -> dict[str, Any]:
    by_mode = defaultdict(list)
    by_donor = defaultdict(list)
    by_component = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("validation_mode"))].append(row)
        by_donor[str(row.get("donor_variant"))].append(row)
        by_component[(str(row.get("component_kind")), str(row.get("validation_mode")))].append(row)
    exact_rows = [r for r in rows if r.get("validation_mode") == "expanded_greedy_exact"]
    natural_rows = [r for r in rows if r.get("validation_mode") == "natural_route_validation"]
    by_key = defaultdict(dict)
    for row in rows:
        key = (
            row.get("case_id"),
            row.get("layer_idx"),
            row.get("component_kind"),
            row.get("component_label"),
            row.get("budget"),
        )
        by_key[key][str(row.get("donor_variant")) + "/" + str(row.get("validation_mode"))] = row
    transfer_pairs = []
    for key, vals in by_key.items():
        exact = vals.get(str(args.search_donor_prompt) + "/expanded_greedy_exact")
        if not exact:
            continue
        for label, row in vals.items():
            if not label.endswith("/natural_route_validation"):
                continue
            transfer_pairs.append(
                {
                    "key": key,
                    "donor": row.get("donor_variant"),
                    "exact_class": exact.get("patched_boundary_class"),
                    "natural_class": row.get("patched_boundary_class"),
                    "exact_rank": exact.get("patched_boundary_rank"),
                    "natural_rank": row.get("patched_boundary_rank"),
                    "exact_target": bool(exact.get("target_transition")),
                    "natural_target": bool(row.get("target_transition")),
                    "natural_preserves_target": bool(exact.get("target_transition") and row.get("target_transition")),
                    "natural_improves": bool(row.get("improved_boundary")),
                }
            )
    return {
        "phase": PHASE,
        "title": "Candidate-Pool Expansion and Natural-Route Validation",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_source_components": len({source_key(row) for row in rows}),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "mode_summary": {mode: compact(vals) for mode, vals in sorted(by_mode.items())},
        "donor_summary": {mode: compact(vals) for mode, vals in sorted(by_donor.items())},
        "component_mode_summary": {f"{kind}/{mode}": compact(vals) for (kind, mode), vals in sorted(by_component.items())},
        "exact_target_rows": sum(1 for row in exact_rows if row.get("target_transition")),
        "natural_target_rows": sum(1 for row in natural_rows if row.get("target_transition")),
        "natural_improved_rows": sum(1 for row in natural_rows if row.get("improved_boundary")),
        "transfer_pair_count": len(transfer_pairs),
        "natural_preserves_target_pairs": sum(1 for row in transfer_pairs if row["natural_preserves_target"]),
        "natural_improves_pairs": sum(1 for row in transfer_pairs if row["natural_improves"]),
        "transfer_pairs": transfer_pairs[:80],
        "boundary": "This phase expands sparse candidate pools with prior successful indices and validates whether exact-choice sparse routes transfer to no-choice natural donor prompts.",
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
                    "role": row.get("role_label"),
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
                log(f"{args.model}: validated {idx}/{len(selected)} source components; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase825_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase825_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "source_components": summary["n_source_components"],
                "rows": summary["n_rows"],
                "exact_target_rows": summary["exact_target_rows"],
                "natural_target_rows": summary["natural_target_rows"],
                "natural_improved_rows": summary["natural_improved_rows"],
                "natural_preserves_target_pairs": summary["natural_preserves_target_pairs"],
                "transfer_pair_count": summary["transfer_pair_count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 825 Candidate-Pool Expansion and Natural-Route Validation ({payload['round']})",
        "",
        "- Source: Phase 822 signal-bearing components plus Phase 823/824 successful sparse indices.",
        "- Search donor: exact_choices.",
        "- Validation donors: no-choice natural prompt rewrites without explicit choices.",
        "",
        "## Model Summary",
        "",
        "| model | source comps | rows | exact target | natural target | natural improved | preserve target | transfer pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_source_components')} | {data.get('n_rows')} | "
            f"{data.get('exact_target_rows')} | {data.get('natural_target_rows')} | "
            f"{data.get('natural_improved_rows')} | {data.get('natural_preserves_target_pairs')} | "
            f"{data.get('transfer_pair_count')} |"
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
        path = out_dir / f"phase825_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase825_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase825_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--search-donor-prompt", default="exact_choices")
    parser.add_argument("--validation-donor-prompts", default="natural_category,natural_question,object_only")
    parser.add_argument("--max-source-rows", type=int, default=4, help="0 means all useful source rows.")
    parser.add_argument("--component-kinds", default="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group")
    parser.add_argument("--budgets", default="16,64")
    parser.add_argument("--base-candidate-pool", type=int, default=24)
    parser.add_argument("--expanded-candidate-pool", type=int, default=48)
    parser.add_argument("--extra-top-delta", type=int, default=16)
    parser.add_argument("--extra-top-signed", type=int, default=16)
    parser.add_argument("--extra-random", type=int, default=16)
    parser.add_argument("--search-group-size", type=int, default=4)
    parser.add_argument("--greedy-steps", type=int, default=8)
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
