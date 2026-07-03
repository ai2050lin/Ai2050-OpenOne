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
import phase876_nonclean_route_causal_validation as p876  # noqa: E402
import phase881_dominant_gear_robustness_and_repair as p881  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 882
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase882_domain_conditioned_dominant_gear_discovery")


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


def selected_cases(domains: list[str], max_cases_per_domain: int) -> list[dict[str, Any]]:
    wanted = set(domains)
    cases = [dict(case) for case in p876.validation_cases() if not wanted or str(case.get("domain")) in wanted]
    if int(max_cases_per_domain) <= 0:
        return cases
    counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for case in cases:
        domain = str(case.get("domain"))
        if counts[domain] >= int(max_cases_per_domain):
            continue
        out.append(case)
        counts[domain] += 1
    return out


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]


def capture_mlp_down_inputs(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    ids = encode_prompt(tokenizer, prompt)
    if not ids:
        return {}
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                captured[int(layer_idx)] = inputs[0][0, answer_pos].detach().float().cpu()

        return hook

    for layer_idx in layer_indices:
        if 0 <= int(layer_idx) < len(layers) and hasattr(layers[int(layer_idx)].mlp, "down_proj"):
            handles.append(layers[int(layer_idx)].mlp.down_proj.register_forward_pre_hook(make_hook(int(layer_idx))))
    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def average_lm_head_vector(weight: torch.Tensor, token_ids: list[int]) -> torch.Tensor | None:
    valid = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < int(weight.shape[0])]
    if not valid:
        return None
    return weight[valid].float().mean(dim=0)


def readout_vector_for_case(model, tokenizer, case: dict[str, Any]) -> torch.Tensor | None:
    weight = p862.p844.p828.p823.lm_head_weight(model).detach().float().cpu()
    sets = p856.token_sets(tokenizer, case)
    class_vec = average_lm_head_vector(weight, list(sets.get("class_target_ids") or []))
    object_vec = average_lm_head_vector(weight, list(sets.get("object_ids") or []))
    if class_vec is None:
        return None
    if object_vec is None:
        return class_vec
    return (class_vec - object_vec).float()


def down_coeff_for_vector(layer, readout_vec: torch.Tensor | None) -> torch.Tensor | None:
    if readout_vec is None or not hasattr(layer.mlp, "down_proj"):
        return None
    weight = layer.mlp.down_proj.weight.detach().float().cpu()
    return (readout_vec @ weight).float()


def add_candidate_metric(bucket: dict[str, Any], activation: float, coeff: float) -> None:
    support = float(activation) * float(coeff)
    bucket["n"] += 1
    bucket["activation_sum"] += float(activation)
    bucket["abs_activation_sum"] += abs(float(activation))
    bucket["coeff_sum"] += float(coeff)
    bucket["abs_coeff_sum"] += abs(float(coeff))
    bucket["signed_support_sum"] += support
    bucket["abs_support_sum"] += abs(support)
    bucket["neg_support_count"] += 1 if support < 0 else 0
    bucket["pos_support_count"] += 1 if support > 0 else 0
    bucket["max_abs_support"] = max(float(bucket.get("max_abs_support", 0.0)), abs(support))


def discover_domain_candidates(
    model,
    tokenizer,
    device: torch.device,
    model_name: str,
    domain: str,
    cases: list[dict[str, Any]],
    prompt_variants: list[str],
    layer_indices: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layers = get_layers(model)
    layer_indices = [int(idx) for idx in layer_indices if 0 <= int(idx) < len(layers)]
    buckets: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "activation_sum": 0.0,
            "abs_activation_sum": 0.0,
            "coeff_sum": 0.0,
            "abs_coeff_sum": 0.0,
            "signed_support_sum": 0.0,
            "abs_support_sum": 0.0,
            "neg_support_count": 0,
            "pos_support_count": 0,
            "max_abs_support": 0.0,
        }
    )
    sample_rows: list[dict[str, Any]] = []
    coeff_cache: dict[tuple[str, int], torch.Tensor | None] = {}
    domain_cases = [case for case in cases if str(case.get("domain")) == str(domain)]
    for case in domain_cases:
        readout_vec = readout_vector_for_case(model, tokenizer, case)
        for layer_idx in layer_indices:
            coeff_cache[(str(case["case_id"]), int(layer_idx))] = down_coeff_for_vector(layers[int(layer_idx)], readout_vec)
        for prompt_variant in prompt_variants:
            prompt = p876.validation_prompt(case, prompt_variant)
            captured = capture_mlp_down_inputs(model, tokenizer, device, prompt, layer_indices)
            for layer_idx, acts in captured.items():
                coeff = coeff_cache.get((str(case["case_id"]), int(layer_idx)))
                if coeff is None:
                    continue
                score = acts.abs() * coeff.abs()
                topk = min(int(args.per_sample_topk), int(score.numel()))
                if topk <= 0:
                    continue
                vals, ids = torch.topk(score, topk)
                for value, channel in zip(vals.tolist(), ids.tolist(), strict=False):
                    channel_id = int(channel)
                    activation = float(acts[channel_id].item())
                    coeff_value = float(coeff[channel_id].item())
                    add_candidate_metric(buckets[(int(layer_idx), channel_id)], activation, coeff_value)
                    sample_rows.append(
                        {
                            "phase": PHASE,
                            "model": model_name,
                            "discovery_domain": domain,
                            "case_id": case["case_id"],
                            "object": case.get("object"),
                            "prompt_variant": prompt_variant,
                            "layer_idx": int(layer_idx),
                            "channel_id": channel_id,
                            "activation": activation,
                            "readout_coeff": coeff_value,
                            "abs_support": float(value),
                        }
                    )
    candidates: list[dict[str, Any]] = []
    for (layer_idx, channel_id), bucket in buckets.items():
        n = int(bucket["n"])
        if n < int(args.min_candidate_hits):
            continue
        sign_consistency = max(
            float(bucket["neg_support_count"]) / max(1, n),
            float(bucket["pos_support_count"]) / max(1, n),
        )
        mean_abs_support = float(bucket["abs_support_sum"]) / max(1, n)
        mean_signed_support = float(bucket["signed_support_sum"]) / max(1, n)
        role = "negative_class_suppressor_candidate" if mean_signed_support < 0 else "positive_class_support_candidate"
        candidate = {
            "phase": PHASE,
            "model": model_name,
            "candidate_source": "phase882_domain_readout_activation_search",
            "control_type": "discovered",
            "discovery_domain": domain,
            "gear_key": f"L{int(layer_idx)}C{int(channel_id)}",
            "gear": {"layer_idx": int(layer_idx), "channel_id": int(channel_id), "gear_key": f"L{int(layer_idx)}C{int(channel_id)}"},
            "layer_idx": int(layer_idx),
            "channel_id": int(channel_id),
            "n": n,
            "mean_activation": float(bucket["activation_sum"]) / max(1, n),
            "mean_abs_activation": float(bucket["abs_activation_sum"]) / max(1, n),
            "mean_readout_coeff": float(bucket["coeff_sum"]) / max(1, n),
            "mean_abs_readout_coeff": float(bucket["abs_coeff_sum"]) / max(1, n),
            "mean_signed_support": mean_signed_support,
            "mean_abs_support": mean_abs_support,
            "max_abs_support": float(bucket["max_abs_support"]),
            "sign_consistency": sign_consistency,
            "candidate_role": role,
            "discovery_score": mean_abs_support * (0.5 + 0.5 * sign_consistency) * max(1.0, float(n) ** 0.5),
        }
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (finite(row.get("discovery_score")), finite(row.get("mean_abs_support")), int(row.get("n") or 0)),
        reverse=True,
    )
    return candidates[: int(args.max_candidates_per_domain)], sample_rows


def same_layer_random_control(model_name: str, candidate: dict[str, Any], width: int, seed: int) -> dict[str, Any]:
    rnd = random.Random(f"{PHASE}:{seed}:{model_name}:{candidate['discovery_domain']}:{candidate['gear_key']}")
    channel_id = int(candidate["channel_id"])
    if width <= 1:
        random_channel = channel_id
    else:
        random_channel = rnd.randrange(width)
        if random_channel == channel_id:
            random_channel = (random_channel + 1) % width
    layer_idx = int(candidate["layer_idx"])
    key = f"L{layer_idx}C{random_channel}"
    out = dict(candidate)
    out.update(
        {
            "candidate_source": "phase882_same_layer_random_control",
            "control_type": "same_layer_random",
            "gear_key": key,
            "gear": {"layer_idx": layer_idx, "channel_id": random_channel, "gear_key": key},
            "channel_id": random_channel,
            "matched_candidate_key": candidate["gear_key"],
            "candidate_role": "same_layer_random_control",
            "discovery_score": None,
        }
    )
    return out


def build_candidates_with_controls(
    model,
    model_name: str,
    discovered: list[dict[str, Any]],
    include_random_controls: bool,
    seed: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = list(discovered)
    if not include_random_controls:
        return candidates
    layers = get_layers(model)
    for cand in discovered:
        layer_idx = int(cand["layer_idx"])
        width = int(layers[layer_idx].mlp.down_proj.weight.shape[1])
        candidates.append(same_layer_random_control(model_name, cand, width, seed))
    return candidates


def eval_candidate_condition(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    prompt_variant: str,
    candidate: dict[str, Any] | None,
    mode: str,
    args: argparse.Namespace,
    original_logits: torch.Tensor | None,
    original_blockers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    gear = None if candidate is None else candidate["gear"]
    return p881.eval_condition(
        model,
        tokenizer,
        device,
        case,
        prompt_variant,
        gear,
        mode,
        float(args.scale_up_factor),
        int(args.max_new_tokens),
        int(args.topk_tokens),
        int(args.topk_blockers),
        original_logits,
        original_blockers,
    )


def make_eval_row(
    model_name: str,
    candidate: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    mode: str,
    base: dict[str, Any],
    intervened: dict[str, Any],
) -> dict[str, Any]:
    row = p881.make_result_row(model_name, {**candidate, "candidate_key": f"{candidate['gear_key']}:{mode}"}, case, prompt_variant, base, intervened)
    row.update(
        {
            "phase": PHASE,
            "row_kind": "phase882_domain_conditioned_dominant_gear_eval",
            "candidate_source": candidate.get("candidate_source"),
            "control_type": candidate.get("control_type"),
            "discovery_domain": candidate.get("discovery_domain"),
            "eval_domain": case.get("domain"),
            "is_same_domain": str(candidate.get("discovery_domain")) == str(case.get("domain")),
            "candidate_role": candidate.get("candidate_role"),
            "discovery_score": candidate.get("discovery_score"),
            "mean_signed_support": candidate.get("mean_signed_support"),
            "mean_abs_support": candidate.get("mean_abs_support"),
            "matched_candidate_key": candidate.get("matched_candidate_key"),
        }
    )
    row["domain_specific_closure"] = bool(row.get("closure_from_open") and row.get("is_same_domain"))
    row["cross_domain_closure"] = bool(row.get("closure_from_open") and not row.get("is_same_domain"))
    return row


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "models": dict(Counter(str(row.get("model")) for row in rows)),
        "discovery_domains": dict(Counter(str(row.get("discovery_domain")) for row in rows)),
        "eval_domains": dict(Counter(str(row.get("eval_domain")) for row in rows)),
        "control_types": dict(Counter(str(row.get("control_type")) for row in rows)),
        "closure_from_open": sum(1 for row in rows if row.get("closure_from_open")),
        "answer_gain": sum(1 for row in rows if row.get("answer_gain")),
        "domain_specific_closure": sum(1 for row in rows if row.get("domain_specific_closure")),
        "cross_domain_closure": sum(1 for row in rows if row.get("cross_domain_closure")),
        "intervened_boundary_closed": sum(1 for row in rows if row.get("intervened_class_boundary_closed")),
        "mean_blocker_reduction": mean([finite(row.get("blocker_reduction")) for row in rows if row.get("blocker_reduction") is not None]),
        "mean_rank_improvement": mean([finite(row.get("rank_improvement")) for row in rows if row.get("rank_improvement") is not None]),
        "mean_class_logit_delta": mean([finite(row.get("class_logit_delta")) for row in rows if row.get("class_logit_delta") is not None]),
    }


def grouped(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(str(row.get(name)) for name in keys)
        buckets[key].append(row)
    return {key: group_summary(items) for key, items in sorted(buckets.items())}


def evidence_label(summary: dict[str, Any]) -> str:
    same = int(summary.get("domain_specific_closure") or 0)
    cross = int(summary.get("cross_domain_closure") or 0)
    answer = int(summary.get("answer_gain") or 0)
    blocker = finite(summary.get("mean_blocker_reduction"))
    if same >= 2 and answer >= 1 and cross <= same:
        return "domain_dominant_candidate"
    if same == 0 and answer == 0 and blocker > 1.0:
        return "weak_modulator"
    if cross > same and cross > 0:
        return "side_effect_or_non_specific"
    if same == 0 and answer == 0:
        return "no_repair"
    return "mixed_or_low_support"


def summarize_model(
    model_name: str,
    discovered: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    status: str,
    args: argparse.Namespace,
    attn_impl: str | None = None,
) -> dict[str, Any]:
    by_candidate = grouped(rows, ["control_type", "discovery_domain", "candidate_key"])
    labeled = []
    for key, info in by_candidate.items():
        parts = key.split("|")
        labeled.append({"key": key, "control_type": parts[0] if parts else None, "evidence_label": evidence_label(info), **info})
    labeled.sort(key=lambda row: (row.get("domain_specific_closure") or 0, row.get("answer_gain") or 0), reverse=True)
    return {
        "phase": PHASE,
        "title": "Domain-Conditioned Dominant Gear Discovery and Control Audit",
        "model": model_name,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "discovery_domains": parse_csv(args.discovery_domains),
        "eval_domains": parse_csv(args.eval_domains),
        "prompt_variants": parse_csv(args.prompt_variants),
        "edit_modes": parse_csv(args.edit_modes),
        "layer_indices": parse_int_csv(args.layers),
        "n_discovered_candidates": len(discovered),
        "n_candidates_with_controls": len(candidates),
        "top_discovered_candidates": discovered,
        "n_candidate_sample_rows": len(sample_rows),
        "n_rows": len(rows),
        "summary": group_summary(rows),
        "by_control_type": grouped(rows, ["control_type"]),
        "by_discovery_eval_domain": grouped(rows, ["control_type", "discovery_domain", "eval_domain"]),
        "by_candidate": by_candidate,
        "candidate_evidence_labels": labeled,
        "boundary": (
            "Candidate discovery uses class-vs-object readout-coupled MLP activations, then tests single-gear edits with "
            "same-layer random controls and cross-domain evaluation. This is atlas construction, not closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    discovery_domains = parse_csv(args.discovery_domains)
    eval_domains = parse_csv(args.eval_domains)
    prompt_variants = parse_csv(args.prompt_variants)
    edit_modes = parse_csv(args.edit_modes)
    discovery_cases = selected_cases(discovery_domains, int(args.max_cases_per_domain))
    eval_cases = selected_cases(eval_domains, int(args.max_cases_per_domain))
    layer_indices = parse_int_csv(args.layers)
    if args.dry_run:
        payload = summarize_model(args.model, [], [], [], [], "dry_run", args)
        payload["discovery_cases"] = [case["case_id"] for case in discovery_cases]
        payload["eval_cases"] = [case["case_id"] for case in eval_cases]
        p846.write_json(out_dir / f"phase882_{args.model}_summary.json", payload)
        return payload

    model = None
    tokenizer = None
    all_sample_rows: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        valid_layers = [int(idx) for idx in layer_indices if 0 <= int(idx) < n_layers]
        setattr(args, "layers", ",".join(str(idx) for idx in valid_layers))
        for domain in discovery_domains:
            domain_candidates, sample_rows = discover_domain_candidates(
                model, tokenizer, device, args.model, domain, discovery_cases, prompt_variants, valid_layers, args
            )
            discovered.extend(domain_candidates)
            all_sample_rows.extend(sample_rows)
            log(f"{args.model}/{args.round_name}: discovered domain={domain} candidates={len(domain_candidates)} samples={len(sample_rows)}")
        include_controls = "same_layer_random" in set(parse_csv(args.controls))
        candidates = build_candidates_with_controls(model, args.model, discovered, include_controls, int(args.seed))
        if not candidates:
            payload = summarize_model(args.model, discovered, candidates, all_sample_rows, rows, "no_candidates", args, attn_impl)
            p846.write_json(out_dir / f"phase882_{args.model}_summary.json", payload)
            p846.write_jsonl(out_dir / f"phase882_{args.model}_candidate_samples.jsonl", all_sample_rows)
            p846.write_jsonl(out_dir / f"phase882_{args.model}_rows.jsonl", [])
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            return payload

        base_cache: dict[tuple[str, str], tuple[dict[str, Any], torch.Tensor, list[dict[str, Any]]]] = {}
        for case_idx, case in enumerate(eval_cases, 1):
            for prompt_variant in prompt_variants:
                base_key = (str(case["case_id"]), str(prompt_variant))
                if base_key not in base_cache:
                    prompt = p876.validation_prompt(case, prompt_variant)
                    prompt_ids = encode_prompt(tokenizer, prompt)
                    sets = p856.token_sets(tokenizer, case)
                    original_logits = p862.first_logits_with_scaled_gears(
                        model, device, prompt_ids, [], "original", float(args.scale_up_factor)
                    )
                    original_blockers = p862.p854.blocker_metrics(
                        tokenizer, original_logits, sets, int(args.topk_blockers)
                    ).get("class_top_blockers") or []
                    base = eval_candidate_condition(
                        model,
                        tokenizer,
                        device,
                        case,
                        prompt_variant,
                        None,
                        "original",
                        args,
                        None,
                        None,
                    )
                    base_cache[base_key] = (base, original_logits, original_blockers)
                base, original_logits, original_blockers = base_cache[base_key]
                for candidate in candidates:
                    for mode in edit_modes:
                        intervened = eval_candidate_condition(
                            model,
                            tokenizer,
                            device,
                            case,
                            prompt_variant,
                            candidate,
                            mode,
                            args,
                            original_logits,
                            original_blockers,
                        )
                        rows.append(make_eval_row(args.model, candidate, case, prompt_variant, mode, base, intervened))
            log(f"{args.model}/{args.round_name}: eval case={case_idx}/{len(eval_cases)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, discovered, candidates, all_sample_rows, rows, "complete", args, attn_impl)
    p846.write_json(out_dir / f"phase882_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase882_{args.model}_candidate_samples.jsonl", all_sample_rows[: int(args.max_saved_samples)])
    p846.write_jsonl(out_dir / f"phase882_{args.model}_rows.jsonl", rows)
    print(json.dumps({k: payload[k] for k in ("phase", "model", "status", "n_discovered_candidates", "n_rows", "summary")}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 882 Domain-Conditioned Dominant Gear Discovery ({payload.get('round')})",
        "",
        "- Boundary: atlas construction and candidate discovery; not closure.",
        "- Discovery: class-vs-object readout-coupled MLP activation.",
        "- Controls: same-layer random controls and cross-domain evaluation.",
        "",
        "## Models",
        "",
        "| model | status | discovered | rows | closure from open | answer gain | domain-specific | cross-domain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        s = summary.get("summary") or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('n_discovered_candidates', 0)} | "
            f"{summary.get('n_rows', 0)} | {s.get('closure_from_open', 0)} | {s.get('answer_gain', 0)} | "
            f"{s.get('domain_specific_closure', 0)} | {s.get('cross_domain_closure', 0)} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        f"- Summary: `{payload.get('overall_summary', {})}`",
        "",
        "## Candidate Evidence Labels",
        "",
        "| model | candidate | label | n | same-domain closure | cross-domain closure | answer gain | control |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("top_candidate_evidence_labels", [])[:40]:
        lines.append(
            f"| {row.get('model')} | `{row.get('key')}` | {row.get('evidence_label')} | {row.get('n', 0)} | "
            f"{row.get('domain_specific_closure', 0)} | {row.get('cross_domain_closure', 0)} | "
            f"{row.get('answer_gain', 0)} | {row.get('control_type')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_root / args.round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "title": "Domain-Conditioned Dominant Gear Discovery and Control Audit",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
        "all_rows": [],
    }
    labels = []
    for model_name in MODELS:
        path = out_dir / f"phase882_{model_name}_summary.json"
        rows_path = out_dir / f"phase882_{model_name}_rows.jsonl"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = summary
            for row in summary.get("candidate_evidence_labels") or []:
                labels.append({"model": model_name, **row})
        if rows_path.exists():
            payload["all_rows"].extend(p846.read_jsonl(rows_path))
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["overall_summary"] = group_summary(payload["all_rows"])
    payload["overall_by_model"] = grouped(payload["all_rows"], ["model"])
    payload["overall_by_control_type"] = grouped(payload["all_rows"], ["control_type"])
    payload["overall_by_discovery_eval_domain"] = grouped(payload["all_rows"], ["model", "control_type", "discovery_domain", "eval_domain"])
    labels.sort(key=lambda row: (row.get("domain_specific_closure") or 0, row.get("answer_gain") or 0), reverse=True)
    payload["top_candidate_evidence_labels"] = labels[:80]
    payload["boundary"] = (
        "Phase 882 searches material/color domain-conditioned candidates and tests specificity with controls. "
        "It does not prove token minimal cut or long rollout closure."
    )
    p846.write_json(out_dir / "phase882_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase882_cross_model_summary.md", payload)
    print(json.dumps({k: payload[k] for k in ("phase", "round", "status", "overall_summary")}, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 882 domain-conditioned dominant gear discovery and control audit.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--round-name", default="material_color_domain_discovery")
    parser.add_argument("--discovery-domains", default="material,color")
    parser.add_argument("--eval-domains", default="animal,material,color")
    parser.add_argument("--prompt-variants", default="nonclean_direct,semantic_pressure,echo_pressure,format_pressure")
    parser.add_argument("--edit-modes", default="flip,half,zero,scale_up")
    parser.add_argument("--layers", default="20,21,22,23,24,25,26,27,28,29,30,31")
    parser.add_argument("--max-cases-per-domain", type=int, default=6)
    parser.add_argument("--per-sample-topk", type=int, default=16)
    parser.add_argument("--min-candidate-hits", type=int, default=2)
    parser.add_argument("--max-candidates-per-domain", type=int, default=3)
    parser.add_argument("--controls", default="same_layer_random")
    parser.add_argument("--seed", type=int, default=882)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--topk-blockers", type=int, default=20)
    parser.add_argument("--max-saved-samples", type=int, default=2000)
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
