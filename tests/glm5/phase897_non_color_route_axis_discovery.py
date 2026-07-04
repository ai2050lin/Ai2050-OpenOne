#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
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
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase894_weak_no_single_closure_rollout_probe as p894  # noqa: E402
import phase895_no_single_minimality_head_pathway_split as p895  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 897
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase897_non_color_route_axis_discovery")
PHASE884_SUMMARY = Path(
    "tests/result/phase884_atlas_coverage_expansion_stable_boundary_search/coverage_stable_boundary/"
    "phase884_cross_model_summary.json"
)
PHASE896_ROOT = Path("tests/result/phase896_cross_domain_pair_search_long_rollout/cross_domain_pair_search_long_rollout")
NON_COLOR_DOMAINS = ["animal", "material", "tool", "plant", "abstract", "object", "geometry"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return p846.mean(values)


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def parse_gear_key(text: str) -> dict[str, Any] | None:
    return p862.parse_gear_key(str(text))


def clean_candidate_body(candidate_key: str) -> str:
    return p885.clean_candidate_body(candidate_key)


def gear_keys_from_subset_key(subset_key: str) -> list[str]:
    return [part for part in str(subset_key or "").split("+") if part.startswith("L") and "C" in part]


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    domains = set(parse_csv(args.domains))
    max_per_domain = int(args.max_cases_per_domain)
    rows = [dict(case) for case in p885.extended_cases() if str(case.get("domain")) in domains]
    rows.sort(key=lambda case: (str(case.get("domain")), str(case.get("split_source", "phase856_base")), str(case.get("object"))))
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for case in rows:
        domain = str(case.get("domain"))
        if max_per_domain > 0 and counts[domain] >= max_per_domain:
            continue
        counts[domain] += 1
        for prompt_variant in parse_csv(args.prompt_variants):
            item = dict(case)
            item["prompt_variant"] = prompt_variant
            item["case_split"] = case.get("split_source", "phase856_base")
            out.append(item)
    return out


def history_candidates(model_name: str) -> dict[str, list[dict[str, Any]]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phase884 = read_json(PHASE884_SUMMARY)
    for group in phase884.get("cross_model_candidate_groups") or []:
        if str(group.get("model")) != model_name:
            continue
        label = str(group.get("evidence_label") or "")
        if label not in {"stable_boundary_candidate", "repair_candidate", "cross_domain_side_effect"}:
            continue
        domains = [str(item) for item in group.get("closure_domains") or [] if str(item) != "color"]
        if not domains:
            continue
        for gear_text in clean_candidate_body(str(group.get("candidate_key") or "")).split("+"):
            gear = parse_gear_key(gear_text)
            if gear is None:
                continue
            for domain in domains:
                by_domain[domain].append(
                    {
                        "source": "history_phase884",
                        "source_label": label,
                        "domain": domain,
                        "layer_idx": int(gear["layer_idx"]),
                        "channel_id": int(gear["channel_id"]),
                        "gear_key": gear_key(gear),
                        "specificity_score": None,
                        "mean_abs_domain": None,
                        "mean_abs_other": None,
                    }
                )
    for row in read_json(PHASE896_ROOT / f"phase896_{model_name}_summary.json").get("pair_domain_groups") or []:
        domain = str(row.get("domain") or "")
        if domain == "color" or domain not in NON_COLOR_DOMAINS:
            continue
        if int(row.get("closure_from_open") or 0) <= 0:
            continue
        for gear_text in gear_keys_from_subset_key(str(row.get("subset_key") or "")):
            gear = parse_gear_key(gear_text)
            if gear is None:
                continue
            by_domain[domain].append(
                {
                    "source": "history_phase896_pair_domain",
                    "source_label": "pair_domain_closure_or_lift",
                    "domain": domain,
                    "layer_idx": int(gear["layer_idx"]),
                    "channel_id": int(gear["channel_id"]),
                    "gear_key": gear_key(gear),
                    "specificity_score": None,
                    "mean_abs_domain": None,
                    "mean_abs_other": None,
                }
            )
    phase896 = read_json(PHASE896_ROOT / f"phase896_{model_name}_summary.json")
    focus_key = str(phase896.get("focus_subset_key") or "")
    for row in phase896.get("domain_groups") or []:
        domain = str(row.get("domain") or "")
        if domain == "color" or domain not in NON_COLOR_DOMAINS:
            continue
        if int(row.get("focus_closure_from_open") or 0) <= 0:
            continue
        for gear_text in gear_keys_from_subset_key(focus_key):
            gear = parse_gear_key(gear_text)
            if gear is None:
                continue
            by_domain[domain].append(
                {
                    "source": "history_phase896_focus",
                    "source_label": "focus_closure",
                    "domain": domain,
                    "layer_idx": int(gear["layer_idx"]),
                    "channel_id": int(gear["channel_id"]),
                    "gear_key": gear_key(gear),
                    "specificity_score": None,
                    "mean_abs_domain": None,
                    "mean_abs_other": None,
                }
            )
    return by_domain


def candidate_layers(n_layers: int, args: argparse.Namespace, history: dict[str, list[dict[str, Any]]]) -> list[int]:
    layers = set(range(max(0, int(n_layers) - int(args.candidate_layer_window)), int(n_layers)))
    for items in history.values():
        for item in items:
            layers.add(int(item["layer_idx"]))
    return sorted(layer for layer in layers if 0 <= layer < int(n_layers))


def first_logits_with_activation_capture(
    model,
    device: torch.device,
    prompt_ids: list[int],
    layers_to_capture: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    layers = get_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                tensor = inputs[0]
                if tensor.ndim >= 3:
                    captured[int(layer_idx)] = tensor[0, -1, :].detach().float().cpu()
                elif tensor.ndim >= 2:
                    captured[int(layer_idx)] = tensor[0, :].detach().float().cpu()
            return inputs

        return hook

    for layer_idx in layers_to_capture:
        if 0 <= int(layer_idx) < len(layers) and hasattr(layers[int(layer_idx)].mlp, "down_proj"):
            handles.append(layers[int(layer_idx)].mlp.down_proj.register_forward_pre_hook(make_hook(int(layer_idx))))
    try:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def update_activation_stats(
    stats: dict[str, dict[int, torch.Tensor]],
    counts: Counter[str],
    domain: str,
    captured: dict[int, torch.Tensor],
) -> None:
    counts[domain] += 1
    for layer_idx, vec in captured.items():
        abs_vec = vec.abs().to(dtype=torch.float32)
        if layer_idx not in stats[domain]:
            stats[domain][layer_idx] = torch.zeros_like(abs_vec)
        stats[domain][layer_idx] += abs_vec


def discovered_candidates(
    model_name: str,
    stats: dict[str, dict[int, torch.Tensor]],
    counts: Counter[str],
    history: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    domains = parse_csv(args.domains)
    max_axes = int(args.max_candidate_axes_per_domain)
    per_layer_topk = int(args.per_layer_discovery_topk)
    all_layers = sorted({layer for by_layer in stats.values() for layer in by_layer})
    total_by_layer: dict[int, torch.Tensor] = {}
    total_count = sum(counts.values())
    for layer_idx in all_layers:
        total: torch.Tensor | None = None
        for domain in domains:
            tensor = stats.get(domain, {}).get(layer_idx)
            if tensor is None:
                continue
            total = tensor.clone() if total is None else total + tensor
        if total is not None:
            total_by_layer[layer_idx] = total

    out: list[dict[str, Any]] = []
    for domain in domains:
        domain_items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in history.get(domain, []):
            if item["gear_key"] in seen:
                continue
            seen.add(item["gear_key"])
            copied = dict(item)
            copied["model"] = model_name
            copied["rank_in_domain"] = len(domain_items) + 1
            domain_items.append(copied)
        candidates: list[dict[str, Any]] = []
        domain_count = int(counts.get(domain) or 0)
        if domain_count > 0:
            for layer_idx in all_layers:
                dom_sum = stats.get(domain, {}).get(layer_idx)
                total = total_by_layer.get(layer_idx)
                if dom_sum is None or total is None:
                    continue
                other_count = max(0, int(total_count) - domain_count)
                dom_mean = dom_sum / float(max(1, domain_count))
                other_mean = (total - dom_sum) / float(max(1, other_count)) if other_count else torch.zeros_like(dom_mean)
                score = dom_mean - other_mean
                k = min(per_layer_topk, int(score.numel()))
                if k <= 0:
                    continue
                values, indices = torch.topk(score, k=k)
                for value, channel_id in zip(values.tolist(), indices.tolist(), strict=False):
                    gear = {"layer_idx": int(layer_idx), "channel_id": int(channel_id)}
                    key = gear_key(gear)
                    candidates.append(
                        {
                            "source": "activation_domain_specificity",
                            "source_label": "late_mlp_abs_activation_domain_minus_other",
                            "domain": domain,
                            "layer_idx": int(layer_idx),
                            "channel_id": int(channel_id),
                            "gear_key": key,
                            "specificity_score": float(value),
                            "mean_abs_domain": float(dom_mean[int(channel_id)].item()),
                            "mean_abs_other": float(other_mean[int(channel_id)].item()),
                            "model": model_name,
                        }
                    )
        candidates.sort(
            key=lambda row: (
                finite(row.get("specificity_score"), -999999.0),
                finite(row.get("mean_abs_domain")),
            ),
            reverse=True,
        )
        for item in candidates:
            if len(domain_items) >= max_axes:
                break
            if item["gear_key"] in seen:
                continue
            if finite(item.get("specificity_score"), -999999.0) <= 0 and domain_items:
                continue
            seen.add(item["gear_key"])
            copied = dict(item)
            copied["rank_in_domain"] = len(domain_items) + 1
            domain_items.append(copied)
        for item in domain_items[:max_axes]:
            out.append(item)
    return out


def specs_for_domain(domain: str, candidates: list[dict[str, Any]], max_subset_size: int) -> list[dict[str, Any]]:
    domain_candidates = [row for row in candidates if str(row.get("domain")) == domain]
    gears = []
    seen: set[str] = set()
    for row in domain_candidates:
        gear = {"layer_idx": int(row["layer_idx"]), "channel_id": int(row["channel_id"])}
        key = gear_key(gear)
        if key in seen:
            continue
        seen.add(key)
        gears.append(gear)
    specs: list[dict[str, Any]] = []
    for size in range(1, min(int(max_subset_size), len(gears)) + 1):
        for combo in itertools.combinations(gears, size):
            keys = [gear_key(gear) for gear in combo]
            specs.append(
                {
                    "subset_key": "+".join(keys),
                    "subset_size": size,
                    "subset_relation": "single_axis" if size == 1 else "domain_pair",
                    "gear_keys": keys,
                    "gears": [dict(gear) for gear in combo],
                }
            )
    return specs


def make_search_row(
    model_name: str,
    condition: dict[str, Any],
    spec: dict[str, Any],
    base_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "row_kind": "phase897_domain_axis_search_row",
        "model": model_name,
        "case_id": condition.get("case_id"),
        "case_split": condition.get("case_split"),
        "eval_domain": condition.get("domain"),
        "object": condition.get("object"),
        "prompt_variant": condition.get("prompt_variant"),
        "edit_mode": condition.get("edit_mode"),
        "subset_key": spec.get("subset_key"),
        "subset_size": spec.get("subset_size"),
        "subset_relation": spec.get("subset_relation"),
        "gear_keys": spec.get("gear_keys"),
        "base_boundary_closed": bool(base_metrics.get("class_boundary_closed")),
        "boundary_closed": bool(metrics.get("class_boundary_closed")),
        "closure_from_open": bool((not base_metrics.get("class_boundary_closed")) and metrics.get("class_boundary_closed")),
        "target_lift": p895.target_lift(base_metrics, metrics),
        "base_class_rank": base_metrics.get("class_best_rank"),
        "class_rank": metrics.get("class_best_rank"),
        "base_full_class_blocker_count": base_metrics.get("full_class_blocker_count"),
        "full_class_blocker_count": metrics.get("full_class_blocker_count"),
        "full_blocker_reduction": p895.blocker_reduction(base_metrics, metrics),
        "full_top_blocker_token": metrics.get("full_class_top_blocker_token"),
        "full_top_blocker_role": metrics.get("full_class_top_blocker_role"),
        "full_top_blocker_role_counts": metrics.get("full_class_top_blocker_role_counts"),
        "class_minus_object_logit": metrics.get("full_class_minus_object_logit"),
    }


def add_condition_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("model")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        singles = [row for row in vals if int(row.get("subset_size") or 0) == 1]
        pairs = [row for row in vals if int(row.get("subset_size") or 0) == 2]
        single_closure = {str(row.get("subset_key")): bool(row.get("closure_from_open")) for row in singles}
        pair_closure_keys = [str(row.get("subset_key")) for row in pairs if row.get("closure_from_open")]
        no_single_pair_keys = []
        for row in pairs:
            components = [str(item) for item in row.get("gear_keys") or []]
            row["any_component_single_closure"] = any(single_closure.get(item, False) for item in components)
            row["no_single_pair_closure"] = bool(row.get("closure_from_open") and not row["any_component_single_closure"])
            if row["no_single_pair_closure"]:
                no_single_pair_keys.append(str(row.get("subset_key")))
        known_minimal_pair_keys = [
            key2 for key2 in no_single_pair_keys if not any(other != key2 for other in pair_closure_keys)
        ]
        for row in vals:
            row["condition_any_single_axis_closure"] = any(single_closure.values())
            row["condition_pair_closure_keys"] = sorted(pair_closure_keys)
            row["condition_no_single_pair_keys"] = sorted(no_single_pair_keys)
            row["condition_known_axis_minimal_pair_keys"] = sorted(known_minimal_pair_keys)
            if int(row.get("subset_size") or 0) == 2:
                row["known_axis_minimal_pair_closure"] = str(row.get("subset_key")) in known_minimal_pair_keys
        out.append(
            {
                "phase": PHASE,
                "row_kind": "phase897_condition_summary",
                "model": key[0],
                "case_id": key[1],
                "prompt_variant": key[2],
                "edit_mode": key[3],
                "eval_domain": vals[0].get("eval_domain") if vals else None,
                "object": vals[0].get("object") if vals else None,
                "any_single_axis_closure": any(single_closure.values()),
                "single_closure_keys": sorted([name for name, closed in single_closure.items() if closed]),
                "pair_closure_keys": sorted(pair_closure_keys),
                "no_single_pair_keys": sorted(no_single_pair_keys),
                "known_axis_minimal_pair_keys": sorted(known_minimal_pair_keys),
            }
        )
    return out


def summarize_model(
    model_name: str,
    conditions: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]],
    attn_impl: str | None,
    layers_to_capture: list[int],
) -> dict[str, Any]:
    by_domain_conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        by_domain_conditions[str(row.get("eval_domain"))].append(row)
    by_domain_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_domain_candidates[str(row.get("domain"))].append(row)
    by_pair_domain: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_single_domain: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in search_rows:
        if int(row.get("subset_size") or 0) == 2:
            by_pair_domain[(str(row.get("eval_domain")), str(row.get("subset_key")))].append(row)
        elif int(row.get("subset_size") or 0) == 1:
            by_single_domain[(str(row.get("eval_domain")), str(row.get("subset_key")))].append(row)

    domain_groups = []
    for domain in sorted(set(by_domain_conditions) | set(by_domain_candidates)):
        vals = by_domain_conditions.get(domain, [])
        domain_groups.append(
            {
                "model": model_name,
                "domain": domain,
                "candidate_u_size": len(by_domain_candidates.get(domain, [])),
                "conditions": len(vals),
                "single_axis_closure_conditions": sum(1 for row in vals if row.get("any_single_axis_closure")),
                "pair_closure_conditions": sum(1 for row in vals if row.get("pair_closure_keys")),
                "no_single_pair_conditions": sum(1 for row in vals if row.get("no_single_pair_keys")),
                "known_axis_minimal_pair_conditions": sum(1 for row in vals if row.get("known_axis_minimal_pair_keys")),
                "single_keys": counter_values(Counter(key for row in vals for key in (row.get("single_closure_keys") or []))),
                "pair_keys": counter_values(Counter(key for row in vals for key in (row.get("known_axis_minimal_pair_keys") or []))),
                "candidate_sources": counter_values(Counter(str(row.get("source")) for row in by_domain_candidates.get(domain, []))),
            }
        )
    domain_groups.sort(
        key=lambda row: (
            row.get("known_axis_minimal_pair_conditions") or 0,
            row.get("no_single_pair_conditions") or 0,
            row.get("single_axis_closure_conditions") or 0,
        ),
        reverse=True,
    )

    single_groups = []
    for (domain, subset), vals in by_single_domain.items():
        single_groups.append(
            {
                "model": model_name,
                "domain": domain,
                "subset_key": subset,
                "rows": len(vals),
                "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
                "mean_target_lift": mean([finite(row.get("target_lift")) for row in vals if row.get("target_lift") is not None]) or 0.0,
                "mean_blocker_reduction": mean(
                    [finite(row.get("full_blocker_reduction")) for row in vals if row.get("full_blocker_reduction") is not None]
                )
                or 0.0,
            }
        )
    single_groups.sort(key=lambda row: (row.get("closure_from_open") or 0, row.get("mean_target_lift") or 0.0), reverse=True)

    pair_groups = []
    for (domain, subset), vals in by_pair_domain.items():
        pair_groups.append(
            {
                "model": model_name,
                "domain": domain,
                "subset_key": subset,
                "rows": len(vals),
                "closure_from_open": sum(1 for row in vals if row.get("closure_from_open")),
                "no_single_pair_closure": sum(1 for row in vals if row.get("no_single_pair_closure")),
                "known_axis_minimal_pair_closure": sum(1 for row in vals if row.get("known_axis_minimal_pair_closure")),
                "mean_target_lift": mean([finite(row.get("target_lift")) for row in vals if row.get("target_lift") is not None]) or 0.0,
                "mean_blocker_reduction": mean(
                    [finite(row.get("full_blocker_reduction")) for row in vals if row.get("full_blocker_reduction") is not None]
                )
                or 0.0,
            }
        )
    pair_groups.sort(
        key=lambda row: (
            row.get("known_axis_minimal_pair_closure") or 0,
            row.get("no_single_pair_closure") or 0,
            row.get("closure_from_open") or 0,
            row.get("mean_target_lift") or 0.0,
        ),
        reverse=True,
    )

    overall = {
        "candidate_axes": len(candidate_rows),
        "history_candidate_axes": sum(1 for row in candidate_rows if str(row.get("source")).startswith("history")),
        "activation_candidate_axes": sum(1 for row in candidate_rows if row.get("source") == "activation_domain_specificity"),
        "selected_conditions": len(conditions),
        "search_rows": len(search_rows),
        "condition_rows": len(condition_rows),
        "single_axis_closure_conditions": sum(1 for row in condition_rows if row.get("any_single_axis_closure")),
        "pair_closure_conditions": sum(1 for row in condition_rows if row.get("pair_closure_keys")),
        "no_single_pair_conditions": sum(1 for row in condition_rows if row.get("no_single_pair_keys")),
        "known_axis_minimal_pair_conditions": sum(1 for row in condition_rows if row.get("known_axis_minimal_pair_keys")),
    }
    if overall["known_axis_minimal_pair_conditions"]:
        evidence_label = "domain_specific_known_axis_minimal_pair_candidates"
    elif overall["no_single_pair_conditions"]:
        evidence_label = "domain_specific_no_single_pair_candidates"
    elif overall["single_axis_closure_conditions"]:
        evidence_label = "domain_specific_single_axis_routes"
    else:
        evidence_label = "candidate_axes_without_boundary_closure"

    return {
        "phase": PHASE,
        "title": "Non-Color Route Axis Discovery and Domain-Specific Pair Search",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "layers_to_capture": layers_to_capture,
        "domains": parse_csv(",".join(NON_COLOR_DOMAINS)),
        "overall": overall,
        "domain_groups": domain_groups,
        "single_domain_groups": single_groups,
        "pair_domain_groups": pair_groups,
        "evidence_label": evidence_label,
        "boundary": (
            "Phase897 discovers candidate axes by late-MLP domain-specific activation plus prior atlas history, "
            "then tests domain-local single/pair interventions. It is candidate_U discovery, not global closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions_base = selected_cases(args)
    conditions = []
    for item in conditions_base:
        for mode in parse_csv(args.edit_modes):
            copied = dict(item)
            copied["edit_mode"] = mode
            conditions.append(copied)
    history = history_candidates(args.model)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run",
            "selected_conditions": len(conditions),
            "history_candidates": {domain: [row["gear_key"] for row in rows] for domain, rows in history.items()},
        }
        p846.write_json(out_dir / f"phase897_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase897_{args.model}_candidate_axes.jsonl", [])
        p846.write_jsonl(out_dir / f"phase897_{args.model}_search_rows.jsonl", [])
        p846.write_jsonl(out_dir / f"phase897_{args.model}_condition_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    candidate_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    base_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
    activation_stats: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    activation_counts: Counter[str] = Counter()
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        layers_to_capture = candidate_layers(n_layers, args, history)
        for idx, condition in enumerate(conditions_base, 1):
            prompt = p885.prompt_for_case(condition, str(condition.get("prompt_variant")))
            prompt_ids = p862.p844.encode_prompt(tokenizer, prompt)
            token_sets = p856.token_sets(tokenizer, condition)
            logits, captured = first_logits_with_activation_capture(model, device, prompt_ids, layers_to_capture)
            metrics = p895.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens), int(args.topk_blockers))
            base_cache[(str(condition.get("case_id")), str(condition.get("prompt_variant")))] = (
                metrics,
                token_sets,
                prompt_ids,
            )
            update_activation_stats(activation_stats, activation_counts, str(condition.get("domain")), captured)
            if idx % max(1, int(args.log_every)) == 0 or idx == len(conditions_base):
                log(f"{args.model}/{args.round_name}: discovery_base={idx}/{len(conditions_base)}")

        candidate_rows = discovered_candidates(args.model, activation_stats, activation_counts, history, args)
        specs_by_domain = {
            domain: specs_for_domain(domain, candidate_rows, int(args.max_subset_size)) for domain in parse_csv(args.domains)
        }

        for idx, condition in enumerate(conditions, 1):
            domain = str(condition.get("domain"))
            base_metrics, token_sets, prompt_ids = base_cache[(str(condition.get("case_id")), str(condition.get("prompt_variant")))]
            for spec in specs_by_domain.get(domain, []):
                logits = p862.first_logits_with_scaled_gears(
                    model,
                    device,
                    prompt_ids,
                    spec["gears"],
                    str(condition.get("edit_mode")),
                    float(args.scale_up_factor),
                )
                metrics = p895.metrics_for_logits(tokenizer, logits, token_sets, int(args.topk_tokens), int(args.topk_blockers))
                search_rows.append(make_search_row(args.model, condition, spec, base_metrics, metrics))
            if idx % max(1, int(args.log_every)) == 0 or idx == len(conditions):
                log(f"{args.model}/{args.round_name}: search_condition={idx}/{len(conditions)} rows={len(search_rows)}")
        condition_rows = add_condition_fields(search_rows)
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(
        args.model,
        conditions,
        candidate_rows,
        search_rows,
        condition_rows,
        attn_impl,
        candidate_layers(999, args, history) if model is None else [],
    )
    # Preserve the actual layer list when model has already been released.
    if search_rows:
        used_layers = sorted({int(row["layer_idx"]) for row in candidate_rows})
        payload["layers_to_capture"] = used_layers
    p846.write_json(out_dir / f"phase897_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase897_{args.model}_candidate_axes.jsonl", candidate_rows)
    p846.write_jsonl(out_dir / f"phase897_{args.model}_search_rows.jsonl", search_rows)
    p846.write_jsonl(out_dir / f"phase897_{args.model}_condition_rows.jsonl", condition_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 897 non-color route axis discovery",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Domain groups", ""])
    lines.append(
        "| model | domain | U size | conditions | single closure | pair closure | no-single pair | known minimal pair | single keys | pair keys |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload.get("domain_groups") or []:
        lines.append(
            "| {model} | {domain} | {candidate_u_size} | {conditions} | {single_axis_closure_conditions} | "
            "{pair_closure_conditions} | {no_single_pair_conditions} | {known_axis_minimal_pair_conditions} | "
            "{single_keys} | {pair_keys} |".format(**row)
        )
    lines.extend(["", "## Top single axes", ""])
    lines.append("| model | domain | subset | rows | closure | mean lift | mean blocker reduction |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in (payload.get("single_domain_groups") or [])[:40]:
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | {row.get('subset_key')} | {row.get('rows')} | "
            f"{row.get('closure_from_open')} | {finite(row.get('mean_target_lift')):.3f} | "
            f"{finite(row.get('mean_blocker_reduction')):.3f} |"
        )
    lines.extend(["", "## Top pair axes", ""])
    lines.append("| model | domain | subset | rows | closure | no-single | known minimal | mean lift | mean blocker reduction |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in (payload.get("pair_domain_groups") or [])[:40]:
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | {row.get('subset_key')} | {row.get('rows')} | "
            f"{row.get('closure_from_open')} | {row.get('no_single_pair_closure')} | "
            f"{row.get('known_axis_minimal_pair_closure')} | {finite(row.get('mean_target_lift')):.3f} | "
            f"{finite(row.get('mean_blocker_reduction')):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase897_{model_name}_summary.json"
        if path.exists():
            summaries.append(read_json(path))
    overall: Counter[str] = Counter()
    domain_groups: list[dict[str, Any]] = []
    single_groups: list[dict[str, Any]] = []
    pair_groups: list[dict[str, Any]] = []
    for summary in summaries:
        for key, value in (summary.get("overall") or {}).items():
            if isinstance(value, int):
                overall[key] += value
        domain_groups.extend(summary.get("domain_groups") or [])
        single_groups.extend(summary.get("single_domain_groups") or [])
        pair_groups.extend(summary.get("pair_domain_groups") or [])
    domain_groups.sort(
        key=lambda row: (
            row.get("known_axis_minimal_pair_conditions") or 0,
            row.get("no_single_pair_conditions") or 0,
            row.get("single_axis_closure_conditions") or 0,
        ),
        reverse=True,
    )
    single_groups.sort(key=lambda row: (row.get("closure_from_open") or 0, row.get("mean_target_lift") or 0.0), reverse=True)
    pair_groups.sort(
        key=lambda row: (
            row.get("known_axis_minimal_pair_closure") or 0,
            row.get("no_single_pair_closure") or 0,
            row.get("closure_from_open") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall": {key: int(value) for key, value in sorted(overall.items())},
        "domain_groups": domain_groups,
        "single_domain_groups": single_groups,
        "pair_domain_groups": pair_groups,
        "evidence_label_counts": counter_values(Counter(str(summary.get("evidence_label")) for summary in summaries)),
    }
    p846.write_json(out_dir / "phase897_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase897_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="non_color_axis_discovery")
    parser.add_argument("--domains", default="animal,material,tool,plant,abstract,object,geometry")
    parser.add_argument("--prompt-variants", default="natural_question,classification")
    parser.add_argument("--edit-modes", default="flip,zero")
    parser.add_argument("--max-cases-per-domain", type=int, default=6)
    parser.add_argument("--candidate-layer-window", type=int, default=6)
    parser.add_argument("--max-candidate-axes-per-domain", type=int, default=4)
    parser.add_argument("--per-layer-discovery-topk", type=int, default=16)
    parser.add_argument("--max-subset-size", type=int, default=2)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--topk-tokens", type=int, default=30)
    parser.add_argument("--topk-blockers", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=24)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
