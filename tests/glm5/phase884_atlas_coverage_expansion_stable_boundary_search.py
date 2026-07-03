#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
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
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase881_dominant_gear_robustness_and_repair as p881  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 884
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase884_atlas_coverage_expansion_stable_boundary_search")
PHASE883_EDGES = Path(
    "tests/result/phase883_atlas_score_table_and_evidence_calibration/phase875_882_calibration/"
    "phase883_atlas_score_edges.jsonl"
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def clipped(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, float(value)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path) if path.exists() else []


def parse_mode(candidate_key: str) -> str:
    text = str(candidate_key)
    if ":" not in text:
        return "flip"
    return text.rsplit(":", 1)[-1]


def parse_gear_set(key: str) -> list[dict[str, Any]]:
    gears: list[dict[str, Any]] = []
    for part in str(key or "").split("+"):
        gear = p862.parse_gear_key(part.strip())
        if gear is not None:
            gears.append(gear)
    return gears


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def selected_cases(domains: list[str], max_cases_per_domain: int) -> list[dict[str, Any]]:
    wanted = set(domains)
    rows = [dict(case) for case in p856.base_cases() if not wanted or str(case.get("domain")) in wanted]
    if int(max_cases_per_domain) <= 0:
        return rows
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for case in rows:
        domain = str(case.get("domain"))
        if counts[domain] >= int(max_cases_per_domain):
            continue
        out.append(case)
        counts[domain] += 1
    return out


def candidate_allowed(edge: dict[str, Any], args: argparse.Namespace) -> bool:
    if str(edge.get("model")) != str(args.model):
        return False
    if finite(edge.get("atlas_score")) < float(args.min_source_atlas_score):
        return False
    if str(edge.get("control_type")) == "same_layer_random":
        return False
    excluded = set(parse_csv(args.exclude_labels))
    if str(edge.get("evidence_label")) in excluded:
        return False
    gears = parse_gear_set(str(edge.get("gear_key") or ""))
    if not gears:
        return False
    return True


def select_phase883_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    edges = [row for row in read_jsonl(PHASE883_EDGES) if candidate_allowed(row, args)]
    edges.sort(key=lambda row: (finite(row.get("atlas_score")), finite(row.get("closure_rate"))), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        candidate_key = str(edge.get("candidate_key"))
        gear_set_key = str(edge.get("gear_key"))
        unique = (gear_set_key, parse_mode(candidate_key))
        if unique in seen:
            continue
        seen.add(unique)
        gears = parse_gear_set(gear_set_key)
        out.append(
            {
                "candidate_source": "phase883_positive_atlas_edge",
                "source_phase": edge.get("phase_source"),
                "source_evidence_label": edge.get("evidence_label"),
                "source_atlas_score": finite(edge.get("atlas_score")),
                "source_candidate_key": candidate_key,
                "candidate_key": candidate_key,
                "parent_candidate_key": candidate_key,
                "gear_key": gear_set_key,
                "gear_keys": [gear_key(gear) for gear in gears],
                "gears": gears,
                "edit_mode": parse_mode(candidate_key),
                "discovery_domain": edge.get("discovery_domain") or edge.get("domain") or edge.get("eval_domain"),
                "source_eval_domain": edge.get("eval_domain"),
                "source_control_type": edge.get("control_type"),
            }
        )
        if len(out) >= int(args.max_candidates_per_model):
            break
    return out


def same_layer_random_gears(model, model_name: str, gears: list[dict[str, Any]], seed: int, salt: str) -> list[dict[str, Any]]:
    rnd = random.Random(f"{PHASE}:{seed}:{model_name}:{salt}")
    layers = get_layers(model)
    out: list[dict[str, Any]] = []
    for gear in gears:
        layer_idx = int(gear["layer_idx"])
        width = int(layers[layer_idx].mlp.down_proj.weight.shape[1])
        channel_id = int(gear["channel_id"])
        if width <= 1:
            random_channel = channel_id
        else:
            random_channel = rnd.randrange(width)
            if random_channel == channel_id:
                random_channel = (random_channel + 1) % width
        out.append({"layer_idx": layer_idx, "channel_id": random_channel})
    return out


def make_specs(model, model_name: str, candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    include_random = "same_layer_random" in set(parse_csv(args.controls))
    include_subsets = bool(args.include_subsets)
    for candidate in candidates:
        specs.append(
            {
                **candidate,
                "condition_type": "full",
                "control_type": "none",
                "candidate_key": candidate["candidate_key"],
                "gears": candidate["gears"],
                "gear_keys": candidate["gear_keys"],
            }
        )
        if include_subsets and len(candidate["gears"]) > 1:
            for idx, gear in enumerate(candidate["gears"]):
                key = f"{gear_key(gear)}:subset{idx}:{candidate['edit_mode']}"
                specs.append(
                    {
                        **candidate,
                        "condition_type": "proper_subset",
                        "control_type": "proper_subset",
                        "candidate_key": key,
                        "gears": [gear],
                        "gear_keys": [gear_key(gear)],
                    }
                )
        if include_random:
            random_gears = same_layer_random_gears(model, model_name, candidate["gears"], int(args.seed), candidate["candidate_key"])
            random_keys = [gear_key(gear) for gear in random_gears]
            specs.append(
                {
                    **candidate,
                    "condition_type": "same_layer_random",
                    "control_type": "same_layer_random",
                    "candidate_key": "+".join(random_keys) + f":random:{candidate['edit_mode']}",
                    "gears": random_gears,
                    "gear_keys": random_keys,
                }
            )
    return specs


def top_token_compact(tokens: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "token_id": item.get("token_id"),
            "token": item.get("token"),
            "role": item.get("role"),
            "logit": item.get("logit"),
            "gap_vs_threshold": item.get("gap_vs_threshold"),
        }
        for item in (tokens or [])[: int(limit)]
    ]


def boundary_closed(row: dict[str, Any]) -> bool:
    return row.get("class_blocker_count") == 0 and row.get("class_best_rank") == 1


def answer_like(row: dict[str, Any]) -> bool:
    return bool(row.get("rollout_answer_class") or row.get("rollout_clear_answer_class") or row.get("rollout_strict_canonical"))


def eval_condition(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    prompt_variant: str,
    gears: list[dict[str, Any]],
    mode: str,
    args: argparse.Namespace,
    original_logits: torch.Tensor | None,
    original_blockers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    prompt = p856.prompt_for_case(case, prompt_variant)
    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
    token_sets = p856.token_sets(tokenizer, case)
    effective_mode = "original" if not gears else mode
    logits = p862.first_logits_with_scaled_gears(model, device, prompt_ids, gears, effective_mode, float(args.scale_up_factor))
    first = p856.first_token_metrics(tokenizer, logits, token_sets, int(args.topk_tokens))
    blocker = p862.p854.blocker_metrics(tokenizer, logits, token_sets, int(args.topk_blockers))
    generated, token_ids = p862.greedy_with_scaled_gears(
        model,
        tokenizer,
        device,
        prompt_ids,
        gears,
        effective_mode,
        int(args.max_new_tokens),
        float(args.scale_up_factor),
    )
    rollout = p856.classify_rollout(generated, case)
    deltas = {}
    if original_logits is not None and original_blockers is not None:
        deltas = p862.original_blocker_deltas(logits, original_logits, original_blockers, int(args.topk_blockers))
    row = {
        "prompt": prompt,
        "token_ids": token_ids,
        "generated_clean": p856.clean_text(generated),
        **first,
        **{f"blocker_{key}": value for key, value in blocker.items()},
        **rollout,
        **deltas,
    }
    row["class_boundary_closed"] = boundary_closed(row)
    row["answer_like"] = answer_like(row)
    row["class_top_blockers_compact"] = top_token_compact(row.get("blocker_class_top_blockers") or [])
    row["top_tokens_compact"] = top_token_compact(row.get("blocker_top_tokens") or row.get("top_tokens") or [])
    return row


def make_row(
    model_name: str,
    spec: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    base: dict[str, Any],
    intervened: dict[str, Any],
) -> dict[str, Any]:
    closure_from_open = bool((not base.get("class_boundary_closed")) and intervened.get("class_boundary_closed"))
    answer_gain = bool((not base.get("answer_like")) and intervened.get("answer_like"))
    is_same_domain = str(spec.get("discovery_domain")) == str(case.get("domain"))
    original_blocker_delta = intervened.get("original_blocker_delta_mean")
    return {
        "phase": PHASE,
        "row_kind": "phase884_atlas_coverage_stable_boundary_row",
        "model": model_name,
        "candidate_source": spec.get("candidate_source"),
        "source_phase": spec.get("source_phase"),
        "source_evidence_label": spec.get("source_evidence_label"),
        "source_atlas_score": spec.get("source_atlas_score"),
        "source_candidate_key": spec.get("source_candidate_key"),
        "parent_candidate_key": spec.get("parent_candidate_key"),
        "candidate_key": spec.get("candidate_key"),
        "gear_key": "+".join(spec.get("gear_keys") or []),
        "gear_keys": spec.get("gear_keys"),
        "gear_count": len(spec.get("gears") or []),
        "condition_type": spec.get("condition_type"),
        "control_type": spec.get("control_type"),
        "edit_mode": spec.get("edit_mode"),
        "discovery_domain": spec.get("discovery_domain"),
        "source_eval_domain": spec.get("source_eval_domain"),
        "eval_domain": case.get("domain"),
        "is_same_domain": is_same_domain,
        "case_id": case.get("case_id"),
        "object": case.get("object"),
        "canonical_answer": case.get("canonical_answer"),
        "overlap_kind": case.get("overlap_kind"),
        "prompt_variant": prompt_variant,
        "base_generated_clean": base.get("generated_clean"),
        "intervened_generated_clean": intervened.get("generated_clean"),
        "base_rollout_label": base.get("rollout_label"),
        "intervened_rollout_label": intervened.get("rollout_label"),
        "base_class_boundary_closed": base.get("class_boundary_closed"),
        "intervened_class_boundary_closed": intervened.get("class_boundary_closed"),
        "closure_from_open": closure_from_open,
        "answer_gain": answer_gain,
        "domain_specific_closure": bool(closure_from_open and is_same_domain),
        "cross_domain_closure": bool(closure_from_open and not is_same_domain),
        "base_answer_like": base.get("answer_like"),
        "intervened_answer_like": intervened.get("answer_like"),
        "base_class_blocker_count": base.get("class_blocker_count"),
        "intervened_class_blocker_count": intervened.get("class_blocker_count"),
        "base_class_rank": base.get("class_best_rank"),
        "intervened_class_rank": intervened.get("class_best_rank"),
        "base_class_logit": base.get("class_best_logit"),
        "intervened_class_logit": intervened.get("class_best_logit"),
        "class_logit_delta": None
        if base.get("class_best_logit") is None or intervened.get("class_best_logit") is None
        else finite(intervened.get("class_best_logit")) - finite(base.get("class_best_logit")),
        "blocker_reduction": None
        if base.get("class_blocker_count") is None or intervened.get("class_blocker_count") is None
        else finite(base.get("class_blocker_count")) - finite(intervened.get("class_blocker_count")),
        "rank_improvement": None
        if base.get("class_best_rank") is None or intervened.get("class_best_rank") is None
        else finite(base.get("class_best_rank")) - finite(intervened.get("class_best_rank")),
        "original_blocker_delta_mean": original_blocker_delta,
        "clean_like_closure": bool(closure_from_open and original_blocker_delta is not None and finite(original_blocker_delta) < 0),
        "nonclean_like_closure": bool(closure_from_open and original_blocker_delta is not None and finite(original_blocker_delta) >= 0),
        "base_top_blockers": base.get("class_top_blockers_compact"),
        "intervened_top_blockers": intervened.get("class_top_blockers_compact"),
        "base_top_tokens": base.get("top_tokens_compact"),
        "intervened_top_tokens": intervened.get("top_tokens_compact"),
    }


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, vals in buckets.items():
        item = {key: value for key, value in zip(keys, key_values, strict=False)}
        n = len(vals)
        item.update(
            {
                "n": n,
                "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
                "answer_gain": sum(1 for row in vals if row.get("answer_gain")),
                "domain_specific_closure": sum(1 for row in vals if row.get("domain_specific_closure")),
                "cross_domain_closure": sum(1 for row in vals if row.get("cross_domain_closure")),
                "intervened_boundary_closed": sum(1 for row in vals if row.get("intervened_class_boundary_closed")),
                "clean_like_closure": sum(1 for row in vals if row.get("clean_like_closure")),
                "nonclean_like_closure": sum(1 for row in vals if row.get("nonclean_like_closure")),
                "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in vals if row.get("blocker_reduction") is not None]) or 0.0,
                "mean_rank_improvement": mean([finite(row.get("rank_improvement")) for row in vals if row.get("rank_improvement") is not None]) or 0.0,
                "mean_class_logit_delta": mean([finite(row.get("class_logit_delta")) for row in vals if row.get("class_logit_delta") is not None]) or 0.0,
                "coverage_domains": sorted(set(str(row.get("eval_domain")) for row in vals)),
                "closure_domains": sorted(set(str(row.get("eval_domain")) for row in vals if row.get("closure_from_open"))),
                "closure_objects": sorted(set(str(row.get("object")) for row in vals if row.get("closure_from_open"))),
                "closure_prompts": sorted(set(str(row.get("prompt_variant")) for row in vals if row.get("closure_from_open"))),
                "source_atlas_score": mean([finite(row.get("source_atlas_score")) for row in vals if row.get("source_atlas_score") is not None]) or 0.0,
            }
        )
        out.append(item)
    return out


def add_subset_minimality(groups: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    subset_closed: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        if row.get("condition_type") != "proper_subset" or not row.get("closure_from_open"):
            continue
        subset_closed.add(
            (
                str(row.get("model")),
                str(row.get("parent_candidate_key")),
                str(row.get("case_id")),
                str(row.get("prompt_variant")),
                str(row.get("edit_mode")),
            )
        )
    full_total: Counter[tuple[str, str]] = Counter()
    full_with_subset: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.get("condition_type") != "full" or not row.get("closure_from_open"):
            continue
        group_key = (str(row.get("model")), str(row.get("candidate_key")))
        full_total[group_key] += 1
        if (
            str(row.get("model")),
            str(row.get("parent_candidate_key")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("edit_mode")),
        ) in subset_closed:
            full_with_subset[group_key] += 1
    for item in groups:
        group_key = (str(item.get("model")), str(item.get("candidate_key")))
        total = int(full_total.get(group_key, 0))
        fail = int(full_with_subset.get(group_key, 0))
        item["proper_subset_also_closed"] = fail
        item["full_closure_for_minimality"] = total
        item["proper_subset_failure_rate"] = fail / total if total else 0.0


def phase884_score(row: dict[str, Any]) -> float:
    n = max(1, int(row.get("n") or 0))
    closure = finite(row.get("closure_from_open")) / n
    answer = finite(row.get("answer_gain")) / n
    domain = finite(row.get("domain_specific_closure")) / n
    cross = finite(row.get("cross_domain_closure")) / n
    false_closed = max(0.0, finite(row.get("intervened_boundary_closed")) - finite(row.get("closure_from_open"))) / n
    blocker = clipped(finite(row.get("mean_blocker_reduction")) / 10.0)
    control = 1.0 if row.get("control_type") == "same_layer_random" else 0.0
    subset_fail = finite(row.get("proper_subset_failure_rate"))
    stable_prompt_bonus = 1.0 if len(row.get("closure_prompts") or []) >= 2 else 0.0
    stable_object_bonus = 1.0 if len(row.get("closure_objects") or []) >= 2 else 0.0
    return (
        4.0 * domain
        + 2.0 * closure
        + 2.0 * answer
        + blocker
        + stable_prompt_bonus
        + stable_object_bonus
        - 3.0 * cross
        - false_closed
        - 1.5 * control
        - 4.0 * subset_fail
    )


def evidence_label(row: dict[str, Any]) -> str:
    n = max(1, int(row.get("n") or 0))
    closure = int(row.get("closure_from_open") or 0)
    answer = int(row.get("answer_gain") or 0)
    domain = int(row.get("domain_specific_closure") or 0)
    cross = int(row.get("cross_domain_closure") or 0)
    subset_fail = finite(row.get("proper_subset_failure_rate"))
    blocker = finite(row.get("mean_blocker_reduction"))
    if row.get("control_type") == "same_layer_random":
        return "same_layer_random_control"
    if subset_fail > 0:
        return "proper_subset_not_minimal"
    if domain >= 2 and answer >= 1 and cross <= domain and len(row.get("closure_prompts") or []) >= 2:
        return "stable_boundary_candidate"
    if domain > 0 and answer > 0:
        return "repair_candidate"
    if cross > domain and cross > 0:
        return "cross_domain_side_effect"
    if closure > 0 and answer > 0:
        return "non_specific_repair_candidate"
    if blocker > 1.0 and answer == 0:
        return "weak_modulator"
    if n and closure == 0 and answer == 0:
        return "candidate_source_no_repair"
    return "mixed_or_low_support"


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = group_rows(rows, ["model", "control_type", "condition_type", "candidate_key"])
    add_subset_minimality(groups, rows)
    for row in groups:
        row["phase884_score"] = phase884_score(row)
        row["delta_vs_source_atlas"] = finite(row.get("phase884_score")) - finite(row.get("source_atlas_score"))
        row["evidence_label"] = evidence_label(row)
    groups.sort(key=lambda row: finite(row.get("phase884_score")), reverse=True)
    return {
        "n": len(rows),
        "models": dict(Counter(str(row.get("model")) for row in rows)),
        "eval_domains": dict(Counter(str(row.get("eval_domain")) for row in rows)),
        "condition_types": dict(Counter(str(row.get("condition_type")) for row in rows)),
        "closure_from_open": sum(1 for row in rows if row.get("closure_from_open")),
        "answer_gain": sum(1 for row in rows if row.get("answer_gain")),
        "domain_specific_closure": sum(1 for row in rows if row.get("domain_specific_closure")),
        "cross_domain_closure": sum(1 for row in rows if row.get("cross_domain_closure")),
        "intervened_boundary_closed": sum(1 for row in rows if row.get("intervened_class_boundary_closed")),
        "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in rows if row.get("blocker_reduction") is not None]),
        "candidate_groups": groups,
        "evidence_label_counts": dict(Counter(str(row.get("evidence_label")) for row in groups)),
        "positive_phase884_score_edges": sum(1 for row in groups if finite(row.get("phase884_score")) > 0),
        "negative_phase884_score_edges": sum(1 for row in groups if finite(row.get("phase884_score")) < 0),
        "top_score_edges": groups[:20],
        "bottom_score_edges": sorted(groups, key=lambda row: finite(row.get("phase884_score")))[:20],
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_phase883_candidates(args)
    cases = selected_cases(parse_csv(args.eval_domains), int(args.max_cases_per_domain))
    prompt_variants = parse_csv(args.prompt_variants)
    if args.dry_run or not candidates:
        payload = {
            "phase": PHASE,
            "title": "Atlas Coverage Expansion and Stable Boundary Search",
            "model": args.model,
            "round": args.round_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "no_phase883_candidates" if not candidates else "dry_run",
            "selected_candidates": candidates,
            "case_count": len(cases),
            "prompt_variants": prompt_variants,
        }
        p846.write_json(out_dir / f"phase884_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase884_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        specs = make_specs(model, args.model, candidates, args)
        n_layers = len(get_layers(model))
        specs = [
            spec
            for spec in specs
            if all(0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0 for gear in spec.get("gears") or [])
        ]
        base_cache: dict[tuple[str, str], tuple[dict[str, Any], torch.Tensor, list[dict[str, Any]]]] = {}
        for case_idx, case in enumerate(cases, 1):
            for prompt_variant in prompt_variants:
                base_key = (str(case["case_id"]), str(prompt_variant))
                if base_key not in base_cache:
                    prompt = p856.prompt_for_case(case, prompt_variant)
                    prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
                    token_sets = p856.token_sets(tokenizer, case)
                    original_logits = p862.first_logits_with_scaled_gears(
                        model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                    )
                    original_blockers = p862.p854.blocker_metrics(
                        tokenizer, original_logits, token_sets, int(args.topk_blockers)
                    ).get("class_top_blockers") or []
                    base = eval_condition(
                        model,
                        tokenizer,
                        device,
                        case,
                        prompt_variant,
                        [],
                        "original",
                        args,
                        None,
                        None,
                    )
                    base_cache[base_key] = (base, original_logits, original_blockers)
                base, original_logits, original_blockers = base_cache[base_key]
                for spec in specs:
                    intervened = eval_condition(
                        model,
                        tokenizer,
                        device,
                        case,
                        prompt_variant,
                        spec.get("gears") or [],
                        str(spec.get("edit_mode")),
                        args,
                        original_logits,
                        original_blockers,
                    )
                    rows.append(make_row(args.model, spec, case, prompt_variant, base, intervened))
            log(f"{args.model}/{args.round_name}: case={case_idx}/{len(cases)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "phase": PHASE,
        "title": "Atlas Coverage Expansion and Stable Boundary Search",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "selected_candidates": candidates,
        "eval_domains": parse_csv(args.eval_domains),
        "case_count": len(cases),
        "prompt_variants": prompt_variants,
        "summary": group_summary(rows),
        "boundary": (
            "Phase884 expands atlas coverage from Phase883 positive non-side-effect edges. "
            "It searches stable boundary evidence, not full language closure."
        ),
    }
    p846.write_json(out_dir / f"phase884_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase884_{args.model}_rows.jsonl", rows)
    print(json.dumps({k: payload[k] for k in ("phase", "model", "status", "case_count", "summary")}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "title": "Atlas Coverage Expansion and Stable Boundary Search",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
        "all_rows": [],
    }
    all_groups: list[dict[str, Any]] = []
    for model_name in MODELS:
        summary_path = out_dir / f"phase884_{model_name}_summary.json"
        rows_path = out_dir / f"phase884_{model_name}_rows.jsonl"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = summary
            all_groups.extend((summary.get("summary") or {}).get("candidate_groups") or [])
        if rows_path.exists():
            payload["all_rows"].extend(read_jsonl(rows_path))
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["overall_summary"] = group_summary(payload["all_rows"])
    all_groups.sort(key=lambda row: finite(row.get("phase884_score")), reverse=True)
    payload["cross_model_candidate_groups"] = all_groups
    payload["cross_model_evidence_label_counts"] = dict(Counter(str(row.get("evidence_label")) for row in all_groups))
    payload["boundary"] = (
        "Phase884 is atlas coverage expansion. Stable boundary candidates require repeated prompt/object support, "
        "low side effect, and no proper-subset failure. Closure remains a future validation target."
    )
    p846.write_json(out_dir / "phase884_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase884_cross_model_summary.md", payload)
    print(json.dumps({k: payload[k] for k in ("phase", "round", "status", "cross_model_evidence_label_counts")}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 884 Atlas Coverage Expansion ({payload['round']})",
        "",
        "- Boundary: atlas coverage and stable boundary search; not language closure.",
        "- Source: positive non-side-effect Phase883 edges, plus same-layer random controls and proper-subset audit.",
        "",
        "## Models",
        "",
        "| model | status | cases | rows | closure from open | answer gain | domain-specific | cross-domain | labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        s = summary.get("summary") or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('case_count', 0)} | "
            f"{s.get('n', 0)} | {s.get('closure_from_open', 0)} | {s.get('answer_gain', 0)} | "
            f"{s.get('domain_specific_closure', 0)} | {s.get('cross_domain_closure', 0)} | "
            f"`{s.get('evidence_label_counts', {})}` |"
        )
    lines += [
        "",
        "## Overall",
        "",
        f"- Summary: `{ {k: v for k, v in (payload.get('overall_summary') or {}).items() if k != 'candidate_groups'} }`",
        f"- Cross-model evidence labels: `{payload.get('cross_model_evidence_label_counts', {})}`",
        "",
        "## Top Candidate Groups",
        "",
        "| score | delta | label | model | condition | candidate | n | closure | answer | same-domain | cross-domain | subset fail |",
        "|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (payload.get("cross_model_candidate_groups") or [])[:40]:
        lines.append(
            f"| {finite(row.get('phase884_score')):.3f} | {finite(row.get('delta_vs_source_atlas')):.3f} | "
            f"{row.get('evidence_label')} | {row.get('model')} | {row.get('condition_type')} | "
            f"`{row.get('candidate_key')}` | {row.get('n', 0)} | {row.get('closure_from_open', 0)} | "
            f"{row.get('answer_gain', 0)} | {row.get('domain_specific_closure', 0)} | "
            f"{row.get('cross_domain_closure', 0)} | {finite(row.get('proper_subset_failure_rate')):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase884 atlas coverage expansion and stable boundary search.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--round-name", default="coverage_stable_boundary")
    parser.add_argument("--eval-domains", default="geometry,animal,tool,color,material,abstract,plant,object")
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,object_only,classification")
    parser.add_argument("--max-cases-per-domain", type=int, default=4)
    parser.add_argument("--max-candidates-per-model", type=int, default=3)
    parser.add_argument("--min-source-atlas-score", type=float, default=0.0)
    parser.add_argument(
        "--exclude-labels",
        default="candidate_source_no_repair,cross_domain_side_effect,observed_pair_not_minimal,same_layer_random_control",
    )
    parser.add_argument("--controls", default="same_layer_random")
    parser.add_argument("--include-subsets", action="store_true")
    parser.add_argument("--seed", type=int, default=884)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--topk-blockers", type=int, default=20)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        summarize_round(args)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
