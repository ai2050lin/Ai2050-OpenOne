#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase854_full_vocab_blocker_min_cut_validation as p854  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 858
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase858_cross_domain_independent_gear_isomorphism_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(part) for part in parse_csv(text)]


def gear_key(gear: dict[str, Any]) -> str:
    return p854.gear_key(gear)


def gear_from_key(text: str) -> dict[str, Any] | None:
    return p854.gear_from_key(text)


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    domains = set(parse_csv(args.domains))
    rows = [dict(case) for case in p856.base_cases() if not domains or str(case.get("domain")) in domains]
    limit = int(args.max_cases_per_domain)
    if limit > 0:
        out: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for case in rows:
            domain = str(case.get("domain"))
            if counts[domain] >= limit:
                continue
            out.append(case)
            counts[domain] += 1
        rows = out
    if int(args.max_cases) > 0:
        rows = rows[: int(args.max_cases)]
    return rows


def token_id_for_class(tokenizer, case: dict[str, Any], sets: dict[str, Any]) -> int | None:
    ids = sets.get("strict_target_ids") or sets.get("class_target_ids") or []
    if ids:
        return int(ids[0])
    return p856.first_token_id(tokenizer, str(case.get("canonical_answer") or ""))


def token_id_for_object(tokenizer, case: dict[str, Any], sets: dict[str, Any]) -> int | None:
    ids = sets.get("object_ids") or []
    if ids:
        return int(ids[0])
    return p856.first_token_id(tokenizer, str(case.get("object") or ""))


def lm_head_weight(model) -> torch.Tensor:
    return p844.p828.p823.lm_head_weight(model).detach().float().cpu()


def class_object_coeff(
    model,
    tokenizer,
    case: dict[str, Any],
    sets: dict[str, Any],
    layer_idx: int,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    class_id = token_id_for_class(tokenizer, case, sets)
    object_id = token_id_for_object(tokenizer, case, sets)
    if class_id is None or object_id is None:
        return None
    if class_id < 0 or object_id < 0 or class_id >= int(weight.shape[0]) or object_id >= int(weight.shape[0]):
        return None
    layers = get_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        return None
    layer = layers[layer_idx]
    if not hasattr(layer.mlp, "down_proj"):
        return None
    w_down = layer.mlp.down_proj.weight.detach().float().cpu()
    return (weight[int(class_id)] - weight[int(object_id)]) @ w_down


def bucket_template() -> dict[str, Any]:
    return {
        "n": 0,
        "support_sum": 0.0,
        "abs_support_sum": 0.0,
        "positive_support_sum": 0.0,
        "negative_support_sum": 0.0,
        "activation_abs_sum": 0.0,
        "coeff_abs_sum": 0.0,
        "pos_count": 0,
        "neg_count": 0,
        "objects": Counter(),
        "prompts": Counter(),
    }


def add_support(bucket: dict[str, Any], activation: torch.Tensor, coeff: torch.Tensor, case: dict[str, Any], prompt: str) -> None:
    support = (activation.float().cpu() * coeff.float().cpu()).float()
    abs_support = torch.abs(support)
    bucket["n"] += 1
    bucket["support_sum"] += float(support.sum().item())
    bucket["abs_support_sum"] += float(abs_support.sum().item())
    bucket["positive_support_sum"] += float(torch.clamp(support, min=0).sum().item())
    bucket["negative_support_sum"] += float(torch.clamp(support, max=0).sum().item())
    bucket["activation_abs_sum"] += float(torch.abs(activation.float().cpu()).sum().item())
    bucket["coeff_abs_sum"] += float(torch.abs(coeff.float().cpu()).sum().item())
    bucket["pos_count"] += int((support > 0).sum().item())
    bucket["neg_count"] += int((support < 0).sum().item())
    bucket["objects"][str(case.get("object"))] += 1
    bucket["prompts"][prompt] += 1


def discover_domain_candidates(
    model,
    tokenizer,
    device: torch.device,
    cases: list[dict[str, Any]],
    prompt_variants: list[str],
    layer_indices: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    layers = get_layers(model)
    valid_layers = [idx for idx in layer_indices if 0 <= idx < len(layers)]
    weight = lm_head_weight(model)
    buckets: dict[tuple[int, int], dict[str, Any]] = defaultdict(bucket_template)
    coeff_cache: dict[tuple[str, int], torch.Tensor | None] = {}

    for case_idx, case in enumerate(cases, 1):
        sets = p856.token_sets(tokenizer, case)
        for layer_idx in valid_layers:
            coeff_cache[(str(case.get("case_id")), layer_idx)] = class_object_coeff(
                model, tokenizer, case, sets, layer_idx, weight
            )
        for prompt_variant in prompt_variants:
            prompt = p856.prompt_for_case(case, prompt_variant)
            captured = p844.capture_mlp_down_inputs(model, tokenizer, device, prompt, valid_layers)
            for layer_idx, activation in captured.items():
                coeff = coeff_cache.get((str(case.get("case_id")), int(layer_idx)))
                if coeff is None:
                    continue
                if activation.numel() != coeff.numel():
                    continue
                add_support(buckets[(int(layer_idx), -1)], activation, coeff, case, prompt_variant)
                for channel_id in torch.topk(torch.abs(activation * coeff), k=min(int(args.per_prompt_top_channels), activation.numel())).indices.tolist():
                    value = bucket_template()
                    support = float((activation[int(channel_id)].float().cpu() * coeff[int(channel_id)].float().cpu()).item())
                    value["n"] = 1
                    value["support_sum"] = support
                    value["abs_support_sum"] = abs(support)
                    value["positive_support_sum"] = max(support, 0.0)
                    value["negative_support_sum"] = min(support, 0.0)
                    value["activation_abs_sum"] = abs(float(activation[int(channel_id)].item()))
                    value["coeff_abs_sum"] = abs(float(coeff[int(channel_id)].item()))
                    value["pos_count"] = 1 if support > 0 else 0
                    value["neg_count"] = 1 if support < 0 else 0
                    value["objects"][str(case.get("object"))] += 1
                    value["prompts"][prompt_variant] += 1
                    key = (int(layer_idx), int(channel_id))
                    agg = buckets[key]
                    for name in [
                        "n",
                        "support_sum",
                        "abs_support_sum",
                        "positive_support_sum",
                        "negative_support_sum",
                        "activation_abs_sum",
                        "coeff_abs_sum",
                        "pos_count",
                        "neg_count",
                    ]:
                        agg[name] += value[name]
                    agg["objects"].update(value["objects"])
                    agg["prompts"].update(value["prompts"])
        if case_idx % max(1, int(args.log_every)) == 0 or case_idx == len(cases):
            log(f"{args.model}/{args.round_name}: discovered support for {case_idx}/{len(cases)} cases")

    rows: list[dict[str, Any]] = []
    for (layer_idx, channel_id), bucket in buckets.items():
        if channel_id < 0:
            continue
        n = int(bucket["n"])
        if n < int(args.min_candidate_support):
            continue
        support_sum = finite(bucket["support_sum"])
        abs_support_sum = finite(bucket["abs_support_sum"])
        signed_mean = support_sum / n if n else 0.0
        abs_mean = abs_support_sum / n if n else 0.0
        polarity = "positive_supporter" if signed_mean >= 0 else "negative_blocker"
        rows.append(
            {
                "row_kind": "phase858_domain_candidate_gear",
                "phase": PHASE,
                "model": args.model,
                "round": args.round_name,
                "domain": str(cases[0].get("domain")) if cases else "",
                "gear_key": f"L{layer_idx}C{channel_id}",
                "layer_idx": int(layer_idx),
                "channel_id": int(channel_id),
                "n": n,
                "support_sum": support_sum,
                "abs_support_sum": abs_support_sum,
                "support_mean": signed_mean,
                "abs_support_mean": abs_mean,
                "positive_support_sum": finite(bucket["positive_support_sum"]),
                "negative_support_sum": finite(bucket["negative_support_sum"]),
                "activation_abs_mean": finite(bucket["activation_abs_sum"]) / n if n else 0.0,
                "coeff_abs_mean": finite(bucket["coeff_abs_sum"]) / n if n else 0.0,
                "pos_count": int(bucket["pos_count"]),
                "neg_count": int(bucket["neg_count"]),
                "polarity": polarity,
                "object_count": dict(bucket["objects"]),
                "prompt_count": dict(bucket["prompts"]),
            }
        )
    rows.sort(key=lambda row: (row["abs_support_mean"], abs(row["support_mean"]), row["n"]), reverse=True)
    return rows


def candidate_specs(domain: str, candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    positive = [row for row in candidates if row.get("polarity") == "positive_supporter"]
    negative = [row for row in candidates if row.get("polarity") == "negative_blocker"]
    positive.sort(key=lambda row: (finite(row.get("support_mean")), finite(row.get("abs_support_mean"))), reverse=True)
    negative.sort(key=lambda row: (finite(row.get("support_mean")), -finite(row.get("abs_support_mean"))))

    specs: list[dict[str, Any]] = [
        {"condition_type": "original", "candidate_key": "original", "candidate_role": "baseline", "mode": "original", "gears": []}
    ]
    seen: set[str] = set()

    def add_single(row: dict[str, Any], mode: str, role: str) -> None:
        key = str(row.get("gear_key"))
        dedupe = f"{key}:{mode}:{role}"
        if dedupe in seen:
            return
        gear = gear_from_key(key)
        if gear is None:
            return
        specs.append(
            {
                "condition_type": f"{role}_single_{mode}",
                "candidate_key": dedupe,
                "candidate_role": role,
                "mode": mode,
                "gears": [gear],
            }
        )
        seen.add(dedupe)

    for row in negative[: int(args.max_negative_gears)]:
        add_single(row, "zero", "negative_blocker")
        add_single(row, "flip", "negative_blocker")
    for row in positive[: int(args.max_positive_gears)]:
        add_single(row, "zero", "positive_supporter")

    def add_combo(rows: list[dict[str, Any]], mode: str, role: str) -> None:
        keys = [str(row.get("gear_key")) for row in rows[: int(args.max_combo_gears)]]
        gears = [gear_from_key(key) for key in keys]
        gears = [gear for gear in gears if gear is not None]
        if len(gears) < 2:
            return
        combo = "+".join(gear_key(gear) for gear in gears)
        dedupe = f"{combo}:{mode}:{role}"
        if dedupe in seen:
            return
        specs.append(
            {
                "condition_type": f"{role}_combo_{mode}",
                "candidate_key": dedupe,
                "candidate_role": role,
                "mode": mode,
                "gears": gears,
            }
        )
        seen.add(dedupe)

    add_combo(negative, "zero", "negative_blocker")
    add_combo(negative, "flip", "negative_blocker")
    add_combo(positive, "zero", "positive_supporter")
    for spec in specs:
        spec["domain"] = domain
    return specs


def row_pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("domain")), str(row.get("case_id")), str(row.get("prompt_variant")))


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rollout = [bool(row.get("rollout_answer_class")) for row in rows]
    first = [bool(row.get("first_token_answer_class")) for row in rows]
    return {
        "n": len(rows),
        "first_token_answer_class": sum(first),
        "first_token_clear_answer_class": sum(1 for row in rows if row.get("first_token_clear_answer_class")),
        "rollout_answer_class": sum(rollout),
        "rollout_clear_answer_class": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "rollout_identity_class_overlap": sum(1 for row in rows if row.get("rollout_identity_class_overlap")),
        "rollout_object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "rollout_other_or_format": sum(1 for row in rows if row.get("rollout_other_or_format")),
        "rollout_labels": dict(Counter(str(row.get("rollout_label")) for row in rows)),
        "mean_class_blocker_count": mean(
            [finite(row.get("class_blocker_count")) for row in rows if row.get("class_blocker_count") is not None]
        ),
        "mean_class_minus_object_logit": mean(
            [finite(row.get("class_minus_object_logit")) for row in rows if row.get("class_minus_object_logit") is not None]
        ),
    }


def pair_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    originals: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("condition_type") == "original":
            originals[row_pair_key(row)] = row
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row.get("condition_type") == "original":
            continue
        base = originals.get(row_pair_key(row))
        if base:
            grouped[str(row.get("candidate_key"))].append((base, row))

    out: list[dict[str, Any]] = []
    for key, pairs in grouped.items():
        first_gain = sum(1 for base, row in pairs if not base.get("first_token_answer_class") and row.get("first_token_answer_class"))
        first_loss = sum(1 for base, row in pairs if base.get("first_token_answer_class") and not row.get("first_token_answer_class"))
        rollout_gain = sum(1 for base, row in pairs if not base.get("rollout_answer_class") and row.get("rollout_answer_class"))
        rollout_loss = sum(1 for base, row in pairs if base.get("rollout_answer_class") and not row.get("rollout_answer_class"))
        clear_gain = sum(
            1 for base, row in pairs if not base.get("rollout_clear_answer_class") and row.get("rollout_clear_answer_class")
        )
        clear_loss = sum(
            1 for base, row in pairs if base.get("rollout_clear_answer_class") and not row.get("rollout_clear_answer_class")
        )
        echo_reduced = sum(1 for base, row in pairs if base.get("rollout_object_echo") and not row.get("rollout_object_echo"))
        echo_induced = sum(1 for base, row in pairs if not base.get("rollout_object_echo") and row.get("rollout_object_echo"))
        blocker_reduction = [
            finite(base.get("class_blocker_count")) - finite(row.get("class_blocker_count"))
            for base, row in pairs
            if base.get("class_blocker_count") is not None and row.get("class_blocker_count") is not None
        ]
        margin_gain = [
            finite(row.get("class_minus_object_logit")) - finite(base.get("class_minus_object_logit"))
            for base, row in pairs
            if base.get("class_minus_object_logit") is not None and row.get("class_minus_object_logit") is not None
        ]
        first = pairs[0][1]
        score = (
            3.0 * clear_gain
            + 2.0 * rollout_gain
            + 1.0 * first_gain
            + 1.0 * echo_reduced
            + 0.15 * (mean(blocker_reduction) or 0.0)
            + 0.15 * (mean(margin_gain) or 0.0)
            - 3.0 * clear_loss
            - 2.0 * rollout_loss
            - 1.0 * first_loss
            - 1.0 * echo_induced
        )
        out.append(
            {
                "candidate_key": key,
                "condition_type": first.get("condition_type"),
                "candidate_role": first.get("candidate_role"),
                "domain": first.get("domain"),
                "mode": first.get("edit_mode"),
                "gear_keys": first.get("gear_keys"),
                "n_pairs": len(pairs),
                "first_gain": first_gain,
                "first_loss": first_loss,
                "rollout_gain": rollout_gain,
                "rollout_loss": rollout_loss,
                "clear_rollout_gain": clear_gain,
                "clear_rollout_loss": clear_loss,
                "object_echo_reduced": echo_reduced,
                "object_echo_induced": echo_induced,
                "mean_blocker_reduction": mean(blocker_reduction),
                "mean_class_minus_object_gain": mean(margin_gain),
                "effect_score": score,
            }
        )
    out.sort(key=lambda row: (finite(row.get("effect_score")), row.get("clear_rollout_gain", 0)), reverse=True)
    return out


def isomorphism_audit(effects: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        by_domain[str(row.get("domain"))].append(row)
    best_by_domain: dict[str, dict[str, Any]] = {}
    for domain, rows in by_domain.items():
        best_by_domain[domain] = sorted(rows, key=lambda row: finite(row.get("effect_score")), reverse=True)[0]
    gear_domain: dict[str, set[str]] = defaultdict(set)
    layer_domain: dict[str, set[str]] = defaultdict(set)
    domains_with_positive_effect = 0
    domains_with_clear_gain = 0
    for domain, row in best_by_domain.items():
        if finite(row.get("effect_score")) > 0:
            domains_with_positive_effect += 1
        if int(row.get("clear_rollout_gain") or 0) > 0:
            domains_with_clear_gain += 1
        for key in row.get("gear_keys") or []:
            gear_domain[str(key)].add(domain)
            parsed = p854.parse_gear_key(str(key))
            if parsed:
                layer_domain[f"L{parsed[0]}"].add(domain)
    shared_gears = {gear: sorted(domains) for gear, domains in gear_domain.items() if len(domains) >= 2}
    shared_layers = {layer: sorted(domains) for layer, domains in layer_domain.items() if len(domains) >= 2}
    return {
        "n_domains": len(best_by_domain),
        "domains_with_positive_effect": domains_with_positive_effect,
        "domains_with_clear_gain": domains_with_clear_gain,
        "best_by_domain": best_by_domain,
        "shared_best_gears": shared_gears,
        "shared_best_layers": shared_layers,
        "exact_gear_isomorphism_count": len(shared_gears),
        "layer_isomorphism_count": len(shared_layers),
        "boundary": (
            "Exact shared gear keys across domains are strong isomorphism evidence. "
            "Shared layers without shared channels are only weak shape evidence."
        ),
    }


def summarize(args: argparse.Namespace, attn_impl: str | None, cases: list[dict[str, Any]], candidates: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain"))].append(row)
        by_condition[str(row.get("condition_type"))].append(row)
    effects = pair_effects(rows)
    return {
        "phase": PHASE,
        "title": "Cross-Domain Independent Gear Discovery and Isomorphism Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "domains": sorted({str(case.get("domain")) for case in cases}),
        "n_cases": len(cases),
        "prompt_variants": parse_csv(args.prompt_variants),
        "layer_indices": parse_int_csv(args.layer_indices),
        "n_candidate_gears": len(candidates),
        "n_rows": len(rows),
        "overall": compact(rows),
        "domain_summary": {domain: compact(group) for domain, group in sorted(by_domain.items())},
        "condition_summary": {condition: compact(group) for condition, group in sorted(by_condition.items())},
        "top_candidate_gears": candidates[:80],
        "candidate_effects": effects,
        "top_effects": effects[:40],
        "isomorphism_audit": isomorphism_audit(effects),
        "boundary": (
            "This phase independently discovers domain-local MLP gear candidates through class-vs-object readout support "
            "and validates them by zero/flip interventions. It is an atlas discovery/isomorphism audit, not language "
            "encoding closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(args)
    prompt_variants = parse_csv(args.prompt_variants)
    layer_indices = parse_int_csv(args.layer_indices)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_domain[str(case.get("domain"))].append(case)

    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "domains": sorted(by_domain),
            "cases": cases,
            "prompt_variants": prompt_variants,
            "layer_indices": layer_indices,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        valid_layers = [idx for idx in layer_indices if 0 <= idx < n_layers]

        specs_by_domain: dict[str, list[dict[str, Any]]] = {}
        for domain, domain_cases in sorted(by_domain.items()):
            domain_candidates = discover_domain_candidates(
                model, tokenizer, device, domain_cases, prompt_variants, valid_layers, args
            )
            candidate_rows.extend(domain_candidates)
            specs_by_domain[domain] = candidate_specs(domain, domain_candidates, args)
            log(
                f"{args.model}/{args.round_name}: domain={domain} candidates={len(domain_candidates)} "
                f"specs={len(specs_by_domain[domain])}"
            )

        for domain_idx, (domain, domain_cases) in enumerate(sorted(by_domain.items()), 1):
            specs = specs_by_domain.get(domain) or []
            for case in domain_cases:
                sets = p856.token_sets(tokenizer, case)
                for prompt_variant in prompt_variants:
                    prompt = p856.prompt_for_case(case, prompt_variant)
                    prompt_ids = p844.encode_prompt(tokenizer, prompt)
                    for spec in specs:
                        valid_gears = [
                            gear
                            for gear in spec.get("gears", [])
                            if 0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0
                        ]
                        logits = p844.first_logits_with_gears(model, device, prompt_ids, valid_gears, str(spec["mode"]))
                        first = p856.first_token_metrics(tokenizer, logits, sets, int(args.topk_tokens))
                        generated, token_ids = p844.greedy_with_gears(
                            model,
                            tokenizer,
                            device,
                            prompt_ids,
                            valid_gears,
                            str(spec["mode"]),
                            int(args.max_new_tokens),
                        )
                        rollout = p856.classify_rollout(generated, case)
                        rows.append(
                            {
                                "row_kind": "phase858_cross_domain_independent_gear_isomorphism_audit",
                                "phase": PHASE,
                                "model": args.model,
                                "round": args.round_name,
                                "domain": domain,
                                "case_id": case["case_id"],
                                "object": case["object"],
                                "canonical_answer": case["canonical_answer"],
                                "answer_aliases": case["answer_aliases"],
                                "overlap_kind": case["overlap_kind"],
                                "prompt_variant": prompt_variant,
                                "prompt": prompt,
                                "condition_type": spec["condition_type"],
                                "candidate_key": spec["candidate_key"],
                                "candidate_role": spec["candidate_role"],
                                "edit_mode": spec["mode"],
                                "gear_count": len(valid_gears),
                                "gear_keys": [gear_key(gear) for gear in valid_gears],
                                "token_ids": token_ids,
                                **sets,
                                **first,
                                **rollout,
                            }
                        )
            log(f"{args.model}/{args.round_name}: evaluated domain {domain_idx}/{len(by_domain)} rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(args, attn_impl, cases, candidate_rows, rows)
    p846.write_jsonl(out_dir / f"phase858_{args.model}_candidate_gears.jsonl", candidate_rows)
    p846.write_jsonl(out_dir / f"phase858_{args.model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase858_{args.model}_summary.json", summary)
    iso = summary["isomorphism_audit"]
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "candidate_gears": len(candidate_rows),
                "rows": len(rows),
                "domains_with_positive_effect": iso["domains_with_positive_effect"],
                "domains_with_clear_gain": iso["domains_with_clear_gain"],
                "exact_gear_isomorphism_count": iso["exact_gear_isomorphism_count"],
                "layer_isomorphism_count": iso["layer_isomorphism_count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 858 Cross-Domain Independent Gear Discovery and Isomorphism Audit ({payload['round']})",
        "",
        "- Source: independent domain-local class-vs-object readout support scan.",
        "- Boundary: gear atlas discovery and isomorphism audit, not language closure.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | domains | candidates | rows | positive domains | clear-gain domains | shared best gears | shared best layers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        iso = data.get("isomorphism_audit") or {}
        lines.append(
            f"| {model_name} | {iso.get('n_domains', 0)} | {data.get('n_candidate_gears', 0)} | "
            f"{data.get('n_rows', 0)} | {iso.get('domains_with_positive_effect', 0)} | "
            f"{iso.get('domains_with_clear_gain', 0)} | {iso.get('exact_gear_isomorphism_count', 0)} | "
            f"{iso.get('layer_isomorphism_count', 0)} |"
        )
    lines += [
        "",
        "## Best Domain Effects",
        "",
        "| model | domain | role | mode | gears | pairs | score | first gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | margin gain |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        best = ((data.get("isomorphism_audit") or {}).get("best_by_domain") or {})
        for domain, row in sorted(best.items()):
            lines.append(
                f"| {model_name} | `{domain}` | `{row.get('candidate_role')}` | `{row.get('mode')}` | "
                f"`{'+'.join(row.get('gear_keys') or [])}` | {row.get('n_pairs', 0)} | "
                f"{fmt(row.get('effect_score'))} | {row.get('first_gain', 0)}/{row.get('first_loss', 0)} | "
                f"{row.get('rollout_gain', 0)}/{row.get('rollout_loss', 0)} | "
                f"{row.get('clear_rollout_gain', 0)}/{row.get('clear_rollout_loss', 0)} | "
                f"{row.get('object_echo_reduced', 0)}/{row.get('object_echo_induced', 0)} | "
                f"{fmt(row.get('mean_blocker_reduction'))} | {fmt(row.get('mean_class_minus_object_gain'))} |"
            )
    lines += [
        "",
        "## Isomorphism Notes",
        "",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        iso = data.get("isomorphism_audit") or {}
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(f"- shared best gears: `{json.dumps(iso.get('shared_best_gears') or {}, ensure_ascii=False)}`")
        lines.append(f"- shared best layers: `{json.dumps(iso.get('shared_best_layers') or {}, ensure_ascii=False)}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase858_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = p846.read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase858_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase858_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--domains", default="geometry,animal,tool,color,material,abstract")
    parser.add_argument("--max-cases-per-domain", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category")
    parser.add_argument("--layer-indices", default="24,26,27,28,29,30,31,32")
    parser.add_argument("--per-prompt-top-channels", type=int, default=96)
    parser.add_argument("--min-candidate-support", type=int, default=1)
    parser.add_argument("--max-negative-gears", type=int, default=2)
    parser.add_argument("--max-positive-gears", type=int, default=1)
    parser.add_argument("--max-combo-gears", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "round": args.round_name,
                    "status": payload.get("status"),
                    "models": payload.get("models"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
