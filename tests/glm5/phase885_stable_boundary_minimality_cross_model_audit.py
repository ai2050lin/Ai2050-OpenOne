#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import re
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
import phase884_atlas_coverage_expansion_stable_boundary_search as p884  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 885
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase885_stable_boundary_minimality_cross_model_audit")
PHASE884_SUMMARY = Path(
    "tests/result/phase884_atlas_coverage_expansion_stable_boundary_search/coverage_stable_boundary/"
    "phase884_cross_model_summary.json"
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def clean_candidate_body(candidate_key: str) -> str:
    text = str(candidate_key or "")
    if text.rsplit(":", 1)[-1] in {"flip", "zero"}:
        text = text.rsplit(":", 1)[0]
    parts = []
    for part in text.split("+"):
        match = re.search(r"L\d+C\d+", part)
        if match:
            parts.append(match.group(0))
    return "+".join(parts)


def parse_mode(candidate_key: str) -> str:
    tail = str(candidate_key or "").rsplit(":", 1)[-1]
    return tail if tail in {"flip", "zero"} else "flip"


def parse_gears_from_candidate_key(candidate_key: str) -> list[dict[str, Any]]:
    gears: list[dict[str, Any]] = []
    for part in clean_candidate_body(candidate_key).split("+"):
        if not part:
            continue
        gear = p862.parse_gear_key(part)
        if gear is not None:
            gears.append(gear)
    return gears


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def extended_cases() -> list[dict[str, Any]]:
    rows = [dict(case, split_source="phase856_base") for case in p856.base_cases()]
    extra = [
        ("material", "ceramic", ["material", "substance", "matter"], "holdout_member_not_alias"),
        ("material", "cotton", ["material", "substance", "matter"], "holdout_member_not_alias"),
        ("material", "rubber", ["material", "substance", "matter"], "holdout_member_not_alias"),
        ("material", "stone", ["material", "substance", "matter"], "holdout_member_not_alias"),
        ("material", "copper", ["material", "substance", "matter", "metal"], "holdout_member_not_alias"),
        ("material", "paper", ["material", "substance", "matter"], "holdout_member_not_alias"),
        ("animal", "horse", ["animal", "living thing", "living creature", "creature", "mammal"], "holdout_member_not_alias"),
        ("animal", "cow", ["animal", "living thing", "living creature", "creature", "mammal"], "holdout_member_not_alias"),
        ("animal", "shark", ["animal", "living thing", "living creature", "creature"], "holdout_member_not_alias"),
        ("animal", "lizard", ["animal", "living thing", "living creature", "creature"], "holdout_member_not_alias"),
        ("animal", "insect", ["animal", "living thing", "living creature", "creature"], "holdout_member_not_alias"),
        ("animal", "whale", ["animal", "living thing", "living creature", "creature", "mammal"], "holdout_member_not_alias"),
        ("color", "purple", ["color", "hue", "colour"], "holdout_attribute_value_not_alias"),
        ("color", "orange", ["color", "hue", "colour"], "holdout_attribute_value_not_alias"),
        ("color", "black", ["color", "hue", "colour"], "holdout_attribute_value_not_alias"),
        ("color", "white", ["color", "hue", "colour"], "holdout_attribute_value_not_alias"),
        ("color", "brown", ["color", "hue", "colour"], "holdout_attribute_value_not_alias"),
        ("color", "violet", ["color", "hue", "colour"], "holdout_attribute_value_not_alias"),
    ]
    start = len(rows) + 1
    for idx, (domain, obj, aliases, overlap_kind) in enumerate(extra, start):
        rows.append(
            {
                "case_id": f"p885_{idx:03d}_{domain}_{obj}",
                "domain": domain,
                "object": obj,
                "answer_aliases": aliases,
                "canonical_answer": aliases[0],
                "overlap_kind": overlap_kind,
                "split_source": "phase885_extended_holdout",
            }
        )
    return rows


def prompt_for_case(case: dict[str, Any], variant: str) -> str:
    obj = case["object"]
    if variant in {"natural_question", "natural_category", "object_only", "classification"}:
        return p856.prompt_for_case(case, variant)
    if variant == "question_plain":
        return f"What kind of thing is a {obj}?\nAnswer with a short category phrase:"
    if variant == "type_of_completion":
        return f"Complete with the best short category phrase.\nA {obj} is a type of"
    raise ValueError(f"unknown prompt variant: {variant}")


def stable_phase884_groups() -> list[dict[str, Any]]:
    payload = read_json(PHASE884_SUMMARY)
    return list(payload.get("cross_model_candidate_groups") or [])


def candidate_from_group(group: dict[str, Any], source_kind: str) -> dict[str, Any] | None:
    candidate_key = str(group.get("candidate_key") or "")
    gears = parse_gears_from_candidate_key(candidate_key)
    if not gears:
        return None
    closure_domains = [str(x) for x in group.get("closure_domains") or []]
    discovery_domain = closure_domains[0] if closure_domains else str(group.get("discovery_domain") or "")
    return {
        "candidate_source": source_kind,
        "source_phase": 884,
        "source_evidence_label": group.get("evidence_label"),
        "source_phase884_score": finite(group.get("phase884_score")),
        "source_atlas_score": finite(group.get("source_atlas_score")),
        "source_candidate_key": candidate_key,
        "parent_candidate_key": candidate_key,
        "candidate_key": candidate_key,
        "gear_keys": [gear_key(gear) for gear in gears],
        "gears": gears,
        "edit_mode": parse_mode(candidate_key),
        "discovery_domain": discovery_domain,
        "source_closure_objects": [str(x) for x in group.get("closure_objects") or []],
        "source_closure_prompts": [str(x) for x in group.get("closure_prompts") or []],
        "source_clean_like_closure": int(group.get("clean_like_closure") or 0),
        "source_nonclean_like_closure": int(group.get("nonclean_like_closure") or 0),
    }


def select_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    groups = stable_phase884_groups()
    selected: list[dict[str, Any]] = []
    for group in groups:
        if str(group.get("model")) != str(args.model):
            continue
        if str(group.get("evidence_label")) != "stable_boundary_candidate":
            continue
        item = candidate_from_group(group, "phase884_stable_boundary_candidate")
        if item is not None:
            selected.append(item)
    if args.negative_anchor_policy in {"no_stable", "all_models"} and (
        args.negative_anchor_policy == "all_models" or not selected
    ):
        anchors = [
            group
            for group in groups
            if str(group.get("model")) == str(args.model)
            and str(group.get("evidence_label")) == "candidate_source_no_repair"
        ]
        anchors.sort(key=lambda row: finite(row.get("source_atlas_score")), reverse=True)
        for group in anchors[: int(args.max_negative_anchors)]:
            item = candidate_from_group(group, "phase884_negative_no_repair_anchor")
            if item is not None:
                selected.append(item)
    return selected[: int(args.max_candidates_per_model)]


def same_layer_random_gears(model, model_name: str, gears: list[dict[str, Any]], seed: int, salt: str) -> list[dict[str, Any]]:
    return p884.same_layer_random_gears(model, model_name, gears, seed, salt)


def neighbor_gears(model, gears: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    layers = get_layers(model)
    out = []
    for gear in gears:
        layer_idx = int(gear["layer_idx"])
        width = int(layers[layer_idx].mlp.down_proj.weight.shape[1])
        channel_id = int(gear["channel_id"])
        out.append({"layer_idx": layer_idx, "channel_id": (channel_id + int(offset)) % max(1, width)})
    return out


def opposite_mode(mode: str) -> str:
    return "zero" if str(mode) == "flip" else "flip"


def make_specs(model, model_name: str, candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    controls = set(parse_csv(args.controls))
    specs: list[dict[str, Any]] = []
    for candidate in candidates:
        base = {
            **candidate,
            "condition_type": "candidate",
            "control_type": "none",
            "candidate_key": candidate["candidate_key"],
            "gears": candidate["gears"],
            "gear_keys": candidate["gear_keys"],
            "edit_mode": candidate["edit_mode"],
        }
        specs.append(base)
        if "same_layer_random" in controls:
            random_gears = same_layer_random_gears(
                model, model_name, candidate["gears"], int(args.seed), candidate["candidate_key"]
            )
            random_keys = [gear_key(gear) for gear in random_gears]
            specs.append(
                {
                    **candidate,
                    "condition_type": "control",
                    "control_type": "same_layer_random",
                    "candidate_key": "+".join(random_keys) + f":random:{candidate['edit_mode']}",
                    "gears": random_gears,
                    "gear_keys": random_keys,
                    "edit_mode": candidate["edit_mode"],
                }
            )
        if "neighbor_channel" in controls:
            neighbor = neighbor_gears(model, candidate["gears"], int(args.neighbor_offset))
            neighbor_keys = [gear_key(gear) for gear in neighbor]
            specs.append(
                {
                    **candidate,
                    "condition_type": "control",
                    "control_type": "neighbor_channel",
                    "candidate_key": "+".join(neighbor_keys) + f":neighbor:{candidate['edit_mode']}",
                    "gears": neighbor,
                    "gear_keys": neighbor_keys,
                    "edit_mode": candidate["edit_mode"],
                }
            )
        if "opposite_mode" in controls:
            mode = opposite_mode(candidate["edit_mode"])
            specs.append(
                {
                    **candidate,
                    "condition_type": "control",
                    "control_type": "opposite_mode",
                    "candidate_key": "+".join(candidate["gear_keys"]) + f":opposite:{mode}",
                    "gears": candidate["gears"],
                    "gear_keys": candidate["gear_keys"],
                    "edit_mode": mode,
                }
            )
    return specs


def cases_for_candidate(candidate: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    target_domain = str(candidate.get("discovery_domain"))
    closure_objects = set(str(x) for x in candidate.get("source_closure_objects") or [])
    domains = set(parse_csv(args.eval_domains))
    all_cases = [case for case in extended_cases() if not domains or str(case["domain"]) in domains]
    target_cases = [case for case in all_cases if str(case["domain"]) == target_domain]
    cross_cases = [case for case in all_cases if str(case["domain"]) != target_domain]
    target_cases.sort(
        key=lambda case: (
            0 if str(case["object"]) in closure_objects else 1,
            0 if str(case.get("split_source")) == "phase856_base" else 1,
            str(case["case_id"]),
        )
    )
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for case in target_cases:
        if counts[target_domain] >= int(args.max_same_domain_cases):
            continue
        out.append(dict(case))
        counts[target_domain] += 1
    cross_counts: Counter[str] = Counter()
    for case in cross_cases:
        domain = str(case["domain"])
        if cross_counts[domain] >= int(args.max_cross_cases_per_domain):
            continue
        out.append(dict(case))
        cross_counts[domain] += 1
    return out


def case_split(candidate: dict[str, Any], case: dict[str, Any]) -> str:
    if str(case["domain"]) != str(candidate.get("discovery_domain")):
        return "cross_domain_control"
    if str(case["object"]) in set(str(x) for x in candidate.get("source_closure_objects") or []):
        return "source_supported_object"
    return "same_domain_holdout"


def boundary_closed(row: dict[str, Any]) -> bool:
    return row.get("class_blocker_count") == 0 and row.get("class_best_rank") == 1


def answer_like(row: dict[str, Any]) -> bool:
    return bool(row.get("rollout_answer_class") or row.get("rollout_clear_answer_class") or row.get("rollout_strict_canonical"))


def top_token_compact(tokens: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return p884.top_token_compact(tokens, limit)


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
    prompt = prompt_for_case(case, prompt_variant)
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
    split: str,
    base: dict[str, Any],
    intervened: dict[str, Any],
) -> dict[str, Any]:
    closure_from_open = bool((not base.get("class_boundary_closed")) and intervened.get("class_boundary_closed"))
    answer_gain = bool((not base.get("answer_like")) and intervened.get("answer_like"))
    is_same_domain = str(spec.get("discovery_domain")) == str(case.get("domain"))
    original_blocker_delta = intervened.get("original_blocker_delta_mean")
    return {
        "phase": PHASE,
        "row_kind": "phase885_stable_boundary_holdout_row",
        "model": model_name,
        "candidate_source": spec.get("candidate_source"),
        "source_phase": spec.get("source_phase"),
        "source_evidence_label": spec.get("source_evidence_label"),
        "source_phase884_score": spec.get("source_phase884_score"),
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
        "source_closure_objects": spec.get("source_closure_objects"),
        "source_closure_prompts": spec.get("source_closure_prompts"),
        "source_clean_like_closure": spec.get("source_clean_like_closure"),
        "source_nonclean_like_closure": spec.get("source_nonclean_like_closure"),
        "eval_domain": case.get("domain"),
        "case_split": split,
        "is_same_domain": is_same_domain,
        "case_id": case.get("case_id"),
        "object": case.get("object"),
        "canonical_answer": case.get("canonical_answer"),
        "overlap_kind": case.get("overlap_kind"),
        "split_source": case.get("split_source"),
        "prompt_variant": prompt_variant,
        "base_generated_clean": base.get("generated_clean"),
        "intervened_generated_clean": intervened.get("generated_clean"),
        "base_rollout_label": base.get("rollout_label"),
        "intervened_rollout_label": intervened.get("rollout_label"),
        "base_class_boundary_closed": base.get("class_boundary_closed"),
        "intervened_class_boundary_closed": intervened.get("class_boundary_closed"),
        "closure_from_open": closure_from_open,
        "answer_gain": answer_gain,
        "same_domain_closure": bool(closure_from_open and is_same_domain),
        "holdout_closure": bool(closure_from_open and split == "same_domain_holdout"),
        "source_object_closure": bool(closure_from_open and split == "source_supported_object"),
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
    out = []
    for key_values, vals in buckets.items():
        item = {key: value for key, value in zip(keys, key_values, strict=False)}
        n = len(vals)
        item.update(
            {
                "n": n,
                "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
                "answer_gain": sum(1 for row in vals if row.get("answer_gain")),
                "same_domain_closure": sum(1 for row in vals if row.get("same_domain_closure")),
                "holdout_closure": sum(1 for row in vals if row.get("holdout_closure")),
                "source_object_closure": sum(1 for row in vals if row.get("source_object_closure")),
                "cross_domain_closure": sum(1 for row in vals if row.get("cross_domain_closure")),
                "clean_like_closure": sum(1 for row in vals if row.get("clean_like_closure")),
                "nonclean_like_closure": sum(1 for row in vals if row.get("nonclean_like_closure")),
                "intervened_boundary_closed": sum(1 for row in vals if row.get("intervened_class_boundary_closed")),
                "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in vals if row.get("blocker_reduction") is not None]) or 0.0,
                "mean_rank_improvement": mean([finite(row.get("rank_improvement")) for row in vals if row.get("rank_improvement") is not None]) or 0.0,
                "mean_class_logit_delta": mean([finite(row.get("class_logit_delta")) for row in vals if row.get("class_logit_delta") is not None]) or 0.0,
                "coverage_domains": sorted(set(str(row.get("eval_domain")) for row in vals)),
                "closure_domains": sorted(set(str(row.get("eval_domain")) for row in vals if row.get("closure_from_open"))),
                "holdout_objects": sorted(set(str(row.get("object")) for row in vals if row.get("holdout_closure"))),
                "closure_objects": sorted(set(str(row.get("object")) for row in vals if row.get("closure_from_open"))),
                "closure_prompts": sorted(set(str(row.get("prompt_variant")) for row in vals if row.get("closure_from_open"))),
                "source_phase884_score": mean([finite(row.get("source_phase884_score")) for row in vals if row.get("source_phase884_score") is not None]) or 0.0,
                "source_atlas_score": mean([finite(row.get("source_atlas_score")) for row in vals if row.get("source_atlas_score") is not None]) or 0.0,
            }
        )
        out.append(item)
    return out


def add_control_leakage(groups: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    control_total: Counter[tuple[str, str]] = Counter()
    control_closed: Counter[tuple[str, str]] = Counter()
    control_answer: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.get("condition_type") != "control":
            continue
        key = (str(row.get("model")), str(row.get("parent_candidate_key")))
        control_total[key] += 1
        if row.get("closure_from_open"):
            control_closed[key] += 1
        if row.get("answer_gain"):
            control_answer[key] += 1
    for item in groups:
        key = (str(item.get("model")), str(item.get("parent_candidate_key")))
        total = int(control_total.get(key, 0))
        closed = int(control_closed.get(key, 0))
        answer = int(control_answer.get(key, 0))
        item["control_total"] = total
        item["control_closure_from_open"] = closed
        item["control_answer_gain"] = answer
        item["control_leak_rate"] = closed / total if total else 0.0


def phase885_score(row: dict[str, Any]) -> float:
    n = max(1, int(row.get("n") or 0))
    closure = finite(row.get("closure_from_open")) / n
    answer = finite(row.get("answer_gain")) / n
    same = finite(row.get("same_domain_closure")) / n
    holdout = finite(row.get("holdout_closure")) / n
    cross = finite(row.get("cross_domain_closure")) / n
    clean = finite(row.get("clean_like_closure")) / n
    nonclean = finite(row.get("nonclean_like_closure")) / n
    false_closed = max(0.0, finite(row.get("intervened_boundary_closed")) - finite(row.get("closure_from_open"))) / n
    control = finite(row.get("control_leak_rate"))
    blocker = clipped(finite(row.get("mean_blocker_reduction")) / 10.0)
    prompt_bonus = 1.0 if len(row.get("closure_prompts") or []) >= 2 else 0.0
    object_bonus = 1.0 if len(row.get("holdout_objects") or []) >= 2 else 0.0
    return (
        5.0 * holdout
        + 2.0 * same
        + 2.0 * closure
        + 2.0 * answer
        + 1.5 * clean
        + blocker
        + prompt_bonus
        + object_bonus
        - 3.0 * cross
        - 2.0 * nonclean
        - false_closed
        - 2.0 * control
    )


def evidence_label(row: dict[str, Any]) -> str:
    if row.get("condition_type") == "control":
        if int(row.get("closure_from_open") or 0) > 0:
            return "control_reproduces_boundary"
        return "control_negative"
    holdout = int(row.get("holdout_closure") or 0)
    answer = int(row.get("answer_gain") or 0)
    cross = int(row.get("cross_domain_closure") or 0)
    clean = int(row.get("clean_like_closure") or 0)
    nonclean = int(row.get("nonclean_like_closure") or 0)
    control = int(row.get("control_closure_from_open") or 0)
    if control > 0:
        return "control_leakage_not_minimal"
    if cross > 0 and cross >= holdout:
        return "cross_domain_side_effect"
    if holdout >= 2 and answer >= 2 and clean >= nonclean:
        return "holdout_clean_stable_boundary"
    if holdout >= 2 and answer >= 1:
        return "holdout_mixed_stable_boundary"
    if holdout > 0 and answer > 0:
        return "weak_holdout_repair"
    if int(row.get("same_domain_closure") or 0) > 0:
        return "source_or_seen_only_boundary"
    if str(row.get("source_evidence_label")) == "candidate_source_no_repair":
        return "negative_anchor_no_repair"
    return "no_holdout_boundary"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = group_rows(rows, ["model", "parent_candidate_key", "condition_type", "control_type", "candidate_key"])
    add_control_leakage(groups, rows)
    for row in groups:
        row["phase885_score"] = phase885_score(row)
        row["delta_vs_phase884"] = finite(row.get("phase885_score")) - finite(row.get("source_phase884_score"))
        row["evidence_label"] = evidence_label(row)
    groups.sort(key=lambda row: finite(row.get("phase885_score")), reverse=True)
    candidate_groups = [row for row in groups if row.get("condition_type") == "candidate"]
    return {
        "n": len(rows),
        "models": dict(Counter(str(row.get("model")) for row in rows)),
        "eval_domains": dict(Counter(str(row.get("eval_domain")) for row in rows)),
        "case_splits": dict(Counter(str(row.get("case_split")) for row in rows)),
        "condition_types": dict(Counter(str(row.get("condition_type")) for row in rows)),
        "closure_from_open": sum(1 for row in rows if row.get("closure_from_open")),
        "answer_gain": sum(1 for row in rows if row.get("answer_gain")),
        "same_domain_closure": sum(1 for row in rows if row.get("same_domain_closure")),
        "holdout_closure": sum(1 for row in rows if row.get("holdout_closure")),
        "source_object_closure": sum(1 for row in rows if row.get("source_object_closure")),
        "cross_domain_closure": sum(1 for row in rows if row.get("cross_domain_closure")),
        "clean_like_closure": sum(1 for row in rows if row.get("clean_like_closure")),
        "nonclean_like_closure": sum(1 for row in rows if row.get("nonclean_like_closure")),
        "candidate_groups": groups,
        "candidate_only_groups": candidate_groups,
        "evidence_label_counts": dict(Counter(str(row.get("evidence_label")) for row in groups)),
        "candidate_evidence_label_counts": dict(Counter(str(row.get("evidence_label")) for row in candidate_groups)),
        "positive_phase885_score_edges": sum(1 for row in candidate_groups if finite(row.get("phase885_score")) > 0),
        "top_score_edges": groups[:20],
        "bottom_score_edges": sorted(groups, key=lambda row: finite(row.get("phase885_score")))[:20],
    }


def structural_audit(candidate_groups: list[dict[str, Any]]) -> dict[str, Any]:
    stable_labels = {"holdout_clean_stable_boundary", "holdout_mixed_stable_boundary", "weak_holdout_repair"}
    domain_to_models: dict[str, set[str]] = defaultdict(set)
    domain_to_edges: dict[str, list[str]] = defaultdict(list)
    for row in candidate_groups:
        if str(row.get("evidence_label")) not in stable_labels:
            continue
        domains = row.get("closure_domains") or [row.get("discovery_domain")]
        for domain in domains:
            domain_to_models[str(domain)].add(str(row.get("model")))
            domain_to_edges[str(domain)].append(f"{row.get('model')}:{row.get('parent_candidate_key')}")
    return {
        "domains_with_holdout_or_weak_repair": {
            domain: sorted(models) for domain, models in sorted(domain_to_models.items())
        },
        "domain_edges": {domain: sorted(edges) for domain, edges in sorted(domain_to_edges.items())},
        "cross_model_isomorphic_domain_candidates": sorted(
            domain for domain, models in domain_to_models.items() if len(models) >= 2
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_candidates(args)
    prompt_variants = parse_csv(args.prompt_variants)
    if args.dry_run or not candidates:
        payload = {
            "phase": PHASE,
            "title": "Stable Boundary Minimality and Cross-Model Structural Audit",
            "model": args.model,
            "round": args.round_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "no_candidates" if not candidates else "dry_run",
            "selected_candidates": candidates,
            "prompt_variants": prompt_variants,
        }
        p846.write_json(out_dir / f"phase885_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase885_{args.model}_rows.jsonl", [])
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
        total_steps = sum(len(cases_for_candidate(candidate, args)) for candidate in candidates)
        step = 0
        for candidate in candidates:
            candidate_cases = cases_for_candidate(candidate, args)
            candidate_specs = [spec for spec in specs if spec.get("parent_candidate_key") == candidate.get("parent_candidate_key")]
            for case in candidate_cases:
                split = case_split(candidate, case)
                for prompt_variant in prompt_variants:
                    base_key = (str(case["case_id"]), str(prompt_variant))
                    if base_key not in base_cache:
                        prompt = prompt_for_case(case, prompt_variant)
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
                    for spec in candidate_specs:
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
                        rows.append(make_row(args.model, spec, case, prompt_variant, split, base, intervened))
                step += 1
                log(f"{args.model}/{args.round_name}: candidate={candidate['candidate_key']} case_step={step}/{total_steps} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows)
    payload = {
        "phase": PHASE,
        "title": "Stable Boundary Minimality and Cross-Model Structural Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete",
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "selected_candidates": candidates,
        "prompt_variants": prompt_variants,
        "eval_domains": parse_csv(args.eval_domains),
        "summary": summary,
        "structural_audit": structural_audit(summary.get("candidate_only_groups") or []),
        "boundary": (
            "Phase885 audits Phase884 stable boundary candidates under holdout objects/prompts, "
            "single-gear controls, clean-like separation, and cross-model structural coverage. "
            "It is still atlas construction, not language closure."
        ),
    }
    p846.write_json(out_dir / f"phase885_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase885_{args.model}_rows.jsonl", rows)
    print(json.dumps({k: payload[k] for k in ("phase", "model", "status", "summary", "structural_audit")}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "title": "Stable Boundary Minimality and Cross-Model Structural Audit",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
        "all_rows": [],
    }
    all_groups: list[dict[str, Any]] = []
    for model_name in MODELS:
        summary_path = out_dir / f"phase885_{model_name}_summary.json"
        rows_path = out_dir / f"phase885_{model_name}_rows.jsonl"
        if summary_path.exists():
            summary = read_json(summary_path)
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = summary
            all_groups.extend((summary.get("summary") or {}).get("candidate_only_groups") or [])
        if rows_path.exists():
            payload["all_rows"].extend(p884.read_jsonl(rows_path))
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["overall_summary"] = summarize_rows(payload["all_rows"])
    all_groups.sort(key=lambda row: finite(row.get("phase885_score")), reverse=True)
    payload["cross_model_candidate_groups"] = all_groups
    payload["cross_model_structural_audit"] = structural_audit(all_groups)
    payload["cross_model_evidence_label_counts"] = dict(Counter(str(row.get("evidence_label")) for row in all_groups))
    payload["boundary"] = (
        "Phase885 checks whether Phase884 stable boundary candidates survive holdout and control audits. "
        "Cross-model isomorphism is recorded only if the same domain has holdout-compatible candidates in at least two models."
    )
    p846.write_json(out_dir / "phase885_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase885_cross_model_summary.md", payload)
    print(json.dumps({k: payload[k] for k in ("phase", "round", "status", "cross_model_evidence_label_counts", "cross_model_structural_audit")}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 885 Stable Boundary Audit ({payload['round']})",
        "",
        "- Boundary: holdout and control audit for Phase884 stable boundary candidates; not language closure.",
        "- Controls: same-layer random, neighbor channel, opposite edit mode.",
        "- Holdout: new same-domain objects plus additional prompt variants.",
        "",
        "## Models",
        "",
        "| model | status | rows | closure | holdout | answer | clean | nonclean | cross | candidate labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        s = summary.get("summary") or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {s.get('n', 0)} | "
            f"{s.get('closure_from_open', 0)} | {s.get('holdout_closure', 0)} | {s.get('answer_gain', 0)} | "
            f"{s.get('clean_like_closure', 0)} | {s.get('nonclean_like_closure', 0)} | "
            f"{s.get('cross_domain_closure', 0)} | `{s.get('candidate_evidence_label_counts', {})}` |"
        )
    lines += [
        "",
        "## Overall",
        "",
        f"- Summary: `{ {k: v for k, v in (payload.get('overall_summary') or {}).items() if k != 'candidate_groups' and k != 'candidate_only_groups'} }`",
        f"- Cross-model candidate labels: `{payload.get('cross_model_evidence_label_counts', {})}`",
        f"- Structural audit: `{payload.get('cross_model_structural_audit', {})}`",
        "",
        "## Candidate Groups",
        "",
        "| score | delta | label | model | parent | candidate | holdout | answer | clean | nonclean | cross | controls | objects | prompts |",
        "|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in (payload.get("cross_model_candidate_groups") or [])[:40]:
        lines.append(
            f"| {finite(row.get('phase885_score')):.3f} | {finite(row.get('delta_vs_phase884')):.3f} | "
            f"{row.get('evidence_label')} | {row.get('model')} | `{row.get('parent_candidate_key')}` | "
            f"`{row.get('candidate_key')}` | {row.get('holdout_closure', 0)} | {row.get('answer_gain', 0)} | "
            f"{row.get('clean_like_closure', 0)} | {row.get('nonclean_like_closure', 0)} | "
            f"{row.get('cross_domain_closure', 0)} | {row.get('control_closure_from_open', 0)} | "
            f"`{row.get('holdout_objects', [])}` | `{row.get('closure_prompts', [])}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase885 stable boundary minimality and cross-model structural audit.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--round-name", default="holdout_minimality_cross_model")
    parser.add_argument("--eval-domains", default="geometry,animal,tool,color,material,abstract,plant,object")
    parser.add_argument(
        "--prompt-variants",
        default="natural_question,natural_category,object_only,classification,question_plain,type_of_completion",
    )
    parser.add_argument("--max-same-domain-cases", type=int, default=10)
    parser.add_argument("--max-cross-cases-per-domain", type=int, default=1)
    parser.add_argument("--max-candidates-per-model", type=int, default=4)
    parser.add_argument("--negative-anchor-policy", choices=["none", "no_stable", "all_models"], default="no_stable")
    parser.add_argument("--max-negative-anchors", type=int, default=2)
    parser.add_argument("--controls", default="same_layer_random,neighbor_channel,opposite_mode")
    parser.add_argument("--neighbor-offset", type=int, default=1)
    parser.add_argument("--seed", type=int, default=885)
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
