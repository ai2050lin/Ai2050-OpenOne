#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
import phase823_beneficial_harmful_boundary_subspace_split as p823  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402


PHASE = 824
SOURCE_822 = Path("tests/result/phase822_boundary_transition_head_mlp_decomposition")
RESULT_ROOT = Path("tests/result/phase824_boundary_objective_sparse_subspace_search")


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
    return val if math.isfinite(val) else default


def clean_generated(text: str) -> str:
    return p816.clean_generated(text)


def case_map() -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in p816.CASES}


def boundary_for(lookup: dict[tuple[str, str], dict[str, Any]], case_id: str, phrase: Any) -> dict[str, Any]:
    std = p820.class_for_phrase(lookup, case_id, clean_generated(str(phrase or "")))
    cls = str(std.get("final_boundary_class") or "unknown_other")
    out = dict(std)
    out["boundary_rank"] = int(p821.BOUNDARY_RANK.get(cls, 0))
    return out


def select_source_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_822 / args.source_round / f"phase822_{model_name}_rows.jsonl")
    allowed_kinds = set(parse_csv(args.component_kinds))
    useful = []
    for row in rows:
        if allowed_kinds and str(row.get("component_kind")) not in allowed_kinds:
            continue
        if row.get("target_transition") or row.get("improved_boundary") or row.get("degraded_boundary"):
            useful.append(row)

    targets = [r for r in useful if r.get("target_transition")]
    degraded = [r for r in useful if r.get("degraded_boundary")]
    partial = [r for r in useful if r.get("improved_boundary") and not r.get("target_transition")]
    for vals in (targets, degraded, partial):
        vals.sort(
            key=lambda r: (
                abs(finite(r.get("delta_boundary_rank"))),
                finite(r.get("patched_boundary_rank")),
                str(r.get("case_id")),
                str(r.get("component_label")),
            ),
            reverse=True,
        )
    if int(args.max_source_rows) <= 0:
        return targets + degraded + partial
    selected: list[dict[str, Any]] = []
    buckets = [targets, degraded, partial]
    while len(selected) < int(args.max_source_rows) and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < int(args.max_source_rows):
                selected.append(bucket.pop(0))
    return selected


def candidate_pool(
    delta: torch.Tensor,
    signed_scores: torch.Tensor,
    max_pool: int,
    seed_parts: tuple[Any, ...],
) -> list[int]:
    n = int(delta.numel())
    if n <= 0:
        return []
    cap = min(int(max_pool), n)
    signed_abs = signed_scores.abs()
    delta_abs = delta.abs()
    n_signed = max(1, cap // 2)
    n_delta = max(1, cap // 4)
    vals: list[int] = []
    vals.extend(int(i) for i in torch.topk(signed_abs, min(n_signed, n)).indices.tolist())
    vals.extend(int(i) for i in torch.topk(delta_abs, min(n_delta, n)).indices.tolist())
    rng = random.Random(p823.stable_seed(*seed_parts, "candidate_random"))
    random_ids = list(range(n))
    rng.shuffle(random_ids)
    vals.extend(random_ids[: max(0, cap - len(dict.fromkeys(vals)))])
    # Stable fill by signed abs if duplicates reduced the pool.
    vals.extend(int(i) for i in torch.topk(signed_abs, min(n, cap * 2)).indices.tolist())
    out: list[int] = []
    seen: set[int] = set()
    for idx in vals:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
        if len(out) >= cap:
            break
    return out


def chunked(indices: list[int], group_size: int) -> list[list[int]]:
    size = max(1, int(group_size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def selected_indices_for_reference(
    mode: str,
    budget: int,
    signed_scores: torch.Tensor,
    seed_parts: tuple[Any, ...],
) -> list[int]:
    selected, _meta = p823.selected_indices_for_mode(mode, budget, signed_scores, seed_parts)
    return selected


def eval_patch(
    model,
    tokenizer,
    device: torch.device,
    lookup: dict[tuple[str, str], dict[str, Any]],
    case: dict[str, Any],
    recipient_ids: list[int],
    baseline_boundary: dict[str, Any],
    layer_idx: int,
    spec: dict[str, Any],
    recipient_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    selected: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    patched_text, patched_ids = p823.greedy_generate_with_subspace_patch(
        model,
        tokenizer,
        device,
        recipient_ids,
        args.max_new_tokens,
        layer_idx,
        spec,
        recipient_vec,
        donor_vec,
        sorted({int(x) for x in selected}),
        args.alpha,
    )
    patched_boundary = boundary_for(lookup, case["case_id"], patched_text)
    delta_rank = int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"])
    return {
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


def objective_key(row: dict[str, Any]) -> tuple[int, float, float, int]:
    return (
        1 if row.get("target_transition") else 0,
        finite(row.get("delta_boundary_rank")),
        finite(row.get("patched_boundary_rank")),
        1 if row.get("protocol_repaired") else 0,
    )


def make_row(
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    spec: dict[str, Any],
    baseline_boundary: dict[str, Any],
    baseline_text: str,
    baseline_ids: list[int],
    mode: str,
    selected: list[int],
    budget: int,
    candidate_meta: dict[str, Any],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "row_kind": "phase824_boundary_objective_sparse_subspace_search",
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
        "source_phase822_target_transition": bool(source_row.get("target_transition")),
        "search_mode": mode,
        "budget": int(budget),
        "n_selected": len(selected),
        "selected_indices": sorted({int(x) for x in selected})[: int(args.save_indices_limit)],
        "candidate_meta": candidate_meta,
        "baseline_generated": clean_generated(baseline_text),
        "baseline_token_ids": baseline_ids,
        "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
        "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
        **eval_result,
    }


def greedy_search_rows(
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
    candidate_groups: list[list[int]],
    budgets: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_idx = int(source_row["layer_idx"])
    probe_rows: list[dict[str, Any]] = []
    for group_idx, group in enumerate(candidate_groups):
        result = eval_patch(
            model, tokenizer, device, lookup, case, recipient_ids, baseline_boundary, layer_idx, spec, recipient_vec, donor_vec, group, args
        )
        probe_rows.append(
            make_row(
                args,
                case,
                source_row,
                spec,
                baseline_boundary,
                baseline_text,
                baseline_ids,
                "candidate_probe",
                group,
                len(group),
                {"group_idx": group_idx, "group_size": len(group)},
                result,
            )
        )

    rows: list[dict[str, Any]] = []
    max_budget = max(budgets) if budgets else 0
    remaining = [list(g) for g in candidate_groups]
    selected: list[int] = []
    seen_budgets: set[int] = set()
    steps = 0
    while remaining and len(selected) < max_budget and steps < int(args.greedy_steps):
        best: tuple[tuple[int, float, float, int], int, list[int], dict[str, Any]] | None = None
        for idx, group in enumerate(remaining):
            candidate_selected = sorted(set(selected + group))
            result = eval_patch(
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
                candidate_selected,
                args,
            )
            key = objective_key(result)
            if best is None or key > best[0]:
                best = (key, idx, candidate_selected, result)
        if best is None:
            break
        _key, best_idx, best_selected, best_result = best
        selected = best_selected
        remaining.pop(best_idx)
        steps += 1
        for budget in budgets:
            if budget in seen_budgets:
                continue
            if len(selected) >= min(int(budget), max_budget):
                seen_budgets.add(budget)
                rows.append(
                    make_row(
                        args,
                        case,
                        source_row,
                        spec,
                        baseline_boundary,
                        baseline_text,
                        baseline_ids,
                        "causal_greedy",
                        selected[: int(budget)],
                        int(budget),
                        {"greedy_steps": steps, "candidate_groups": len(candidate_groups), "search_group_size": args.search_group_size},
                        best_result,
                    )
                )
    if selected:
        for budget in budgets:
            if budget not in seen_budgets:
                result = eval_patch(
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
                    make_row(
                        args,
                        case,
                        source_row,
                        spec,
                        baseline_boundary,
                        baseline_text,
                        baseline_ids,
                        "causal_greedy",
                        selected[: int(budget)],
                        int(budget),
                        {"greedy_steps": steps, "candidate_groups": len(candidate_groups), "search_group_size": args.search_group_size},
                        result,
                    )
                )
    return rows, probe_rows


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
    recipient_prompt = p816.build_prompt(case, args.recipient_prompt)
    donor_prompt = p816.build_prompt(case, args.donor_prompt)
    recipient_ids = p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p823.greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = boundary_for(lookup, case["case_id"], baseline_text)
    recipient_state = p822.capture_component_state(model, tokenizer, device, recipient_prompt, layer_idx)
    donor_state = p822.capture_component_state(model, tokenizer, device, donor_prompt, layer_idx)
    recipient_vec = p823.component_vector(recipient_state, spec)
    donor_vec = p823.component_vector(donor_state, spec)
    effective_dir, readout_meta = p823.effective_readout_direction(model, tokenizer, case, baseline_ids, layer_idx, spec)
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
    budgets = [min(int(b), n) for b in parse_int_csv(args.budgets)]
    budgets = sorted({b for b in budgets if b > 0})
    if not budgets:
        budgets = [min(n, int(args.search_group_size))]
    pool = candidate_pool(
        delta,
        signed_scores,
        args.candidate_pool,
        (args.model, case["case_id"], layer_idx, spec.get("component_label"), args.round_name),
    )
    groups = chunked(pool, args.search_group_size)
    rows: list[dict[str, Any]] = []

    # Reference rows: keep the previous proxy methods as explicit controls.
    for mode in parse_csv(args.reference_modes):
        mode_budgets = [n] if mode == "all" else budgets
        for budget in mode_budgets:
            selected = selected_indices_for_reference(
                mode,
                int(budget),
                signed_scores,
                (args.model, case["case_id"], layer_idx, spec.get("component_label"), mode, budget, "phase824"),
            )
            if not selected:
                continue
            result = eval_patch(
                model, tokenizer, device, lookup, case, recipient_ids, baseline_boundary, layer_idx, spec, recipient_vec, donor_vec, selected, args
            )
            rows.append(
                make_row(
                    args,
                    case,
                    source_row,
                    spec,
                    baseline_boundary,
                    baseline_text,
                    baseline_ids,
                    mode,
                    selected,
                    int(budget),
                    {
                        "n_component_dims": n,
                        "candidate_pool_size": len(pool),
                        "readout_meta": readout_meta,
                    },
                    result,
                )
            )

    greedy_rows, probe_rows = greedy_search_rows(
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
        groups,
        budgets,
        args,
    )
    if args.save_probe_rows:
        rows.extend(probe_rows)
    rows.extend(greedy_rows)
    for row in rows:
        row["n_component_dims"] = n
        row["candidate_pool"] = pool[: int(args.save_indices_limit)]
        row["candidate_pool_size"] = len(pool)
        row["candidate_group_count"] = len(groups)
    return rows


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


def comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("search_mode") == "candidate_probe":
            continue
        key = (
            row.get("model"),
            row.get("case_id"),
            row.get("layer_idx"),
            row.get("component_kind"),
            row.get("component_label"),
            row.get("budget"),
        )
        by_key[key][str(row.get("search_mode"))] = row
    pairs = []
    for key, vals in by_key.items():
        greedy = vals.get("causal_greedy")
        if not greedy:
            continue
        best_control = None
        for mode in ("positive_topk", "abs_topk", "random_topk"):
            row = vals.get(mode)
            if row is None:
                continue
            if best_control is None or objective_key(row) > objective_key(best_control):
                best_control = row
        if best_control is None:
            continue
        pairs.append(
            {
                "key": key,
                "greedy_class": greedy.get("patched_boundary_class"),
                "control_class": best_control.get("patched_boundary_class"),
                "control_mode": best_control.get("search_mode"),
                "greedy_rank": greedy.get("patched_boundary_rank"),
                "control_rank": best_control.get("patched_boundary_rank"),
                "greedy_target": bool(greedy.get("target_transition")),
                "control_target": bool(best_control.get("target_transition")),
                "greedy_better": objective_key(greedy) > objective_key(best_control),
                "greedy_equal_or_better": objective_key(greedy) >= objective_key(best_control),
            }
        )
    return {
        "paired_count": len(pairs),
        "greedy_better_pairs": sum(1 for row in pairs if row["greedy_better"]),
        "greedy_equal_or_better_pairs": sum(1 for row in pairs if row["greedy_equal_or_better"]),
        "greedy_target_pairs": sum(1 for row in pairs if row["greedy_target"]),
        "control_target_pairs": sum(1 for row in pairs if row["control_target"]),
        "pairs": pairs[:40],
    }


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None = None) -> dict[str, Any]:
    by_mode = defaultdict(list)
    by_component_mode = defaultdict(list)
    by_source = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("search_mode"))].append(row)
        by_component_mode[(str(row.get("component_kind")), str(row.get("search_mode")))].append(row)
        by_source[(row.get("case_id"), row.get("layer_idx"), row.get("component_kind"), row.get("component_label"))].append(row)
    best_sources = {}
    for key, vals in by_source.items():
        best = max(vals, key=objective_key)
        best_sources["|".join(str(x) for x in key)] = {
            "baseline_class": vals[0].get("baseline_boundary_class"),
            "best_mode": best.get("search_mode"),
            "best_budget": best.get("budget"),
            "best_class": best.get("patched_boundary_class"),
            "best_delta": best.get("delta_boundary_rank"),
            "best_generated": best.get("patched_generated"),
        }
    return {
        "phase": PHASE,
        "title": "Boundary-Objective Sparse Subspace Search",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_source_components": len(by_source),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "mean_delta_boundary_rank": sum(finite(row.get("delta_boundary_rank")) for row in rows) / len(rows) if rows else None,
        "mode_summary": {mode: compact(vals) for mode, vals in sorted(by_mode.items())},
        "component_mode_summary": {f"{kind}/{mode}": compact(vals) for (kind, mode), vals in sorted(by_component_mode.items())},
        "comparison_summary": comparison_summary(rows),
        "best_sources": best_sources,
        "boundary": "This phase directly scores sparse component subspaces by answer-boundary transition instead of relying only on target first-token readout sign.",
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
                log(f"{args.model}: searched {idx}/{len(selected)} source components; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl)
    write_jsonl(out_dir / f"phase824_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase824_{args.model}_summary.json", summary)
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
                "greedy_better_pairs": summary["comparison_summary"]["greedy_better_pairs"],
                "greedy_equal_or_better_pairs": summary["comparison_summary"]["greedy_equal_or_better_pairs"],
                "paired_count": summary["comparison_summary"]["paired_count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 824 Boundary-Objective Sparse Subspace Search ({payload['round']})",
        "",
        "- Boundary: Phase 820 answer-boundary standard v1.",
        "- Source: Phase 822 signal-bearing components.",
        "- Search: candidate chunks are scored by actual generated boundary class; greedy subsets are compared with readout-positive, abs, random, and all-component controls.",
        "",
        "## Model Summary",
        "",
        "| model | source comps | rows | improved | target | degraded | mean delta | greedy better | greedy >= control | paired |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        comp = data.get("comparison_summary") or {}
        lines.append(
            f"| {model_name} | {data.get('n_source_components')} | {data.get('n_rows')} | "
            f"{data.get('improved_rows')} | {data.get('target_transition_rows')} | {data.get('degraded_rows')} | "
            f"{finite(data.get('mean_delta_boundary_rank')):.3f} | {comp.get('greedy_better_pairs')} | "
            f"{comp.get('greedy_equal_or_better_pairs')} | {comp.get('paired_count')} |"
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
    lines += ["", "## Best Source Components", ""]
    lines += [
        "| model | source | baseline | best mode | budget | class | delta | generated |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for source, row in sorted((data.get("best_sources") or {}).items()):
            lines.append(
                f"| {model_name} | `{source}` | `{row.get('baseline_class')}` | `{row.get('best_mode')}` | "
                f"{row.get('best_budget')} | `{row.get('best_class')}` | {row.get('best_delta')} | `{row.get('best_generated')}` |"
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
        path = out_dir / f"phase824_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase824_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase824_cross_model_summary.md", payload)
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
    parser.add_argument("--reference-modes", default="all,positive_topk,abs_topk,random_topk")
    parser.add_argument("--budgets", default="16,64")
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--search-group-size", type=int, default=4)
    parser.add_argument("--greedy-steps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--save-probe-rows", action="store_true")
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
