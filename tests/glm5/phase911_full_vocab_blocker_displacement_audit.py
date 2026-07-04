#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
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
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
import phase903_protocol_continuation_field_mapping as p903  # noqa: E402
import phase906_eos_action_boundary_test as p906  # noqa: E402
import phase909_l0_attention_source_span_eos_boundary_audit as p909  # noqa: E402
import phase910_prompt_preserving_termination_route_reconstruction as p910  # noqa: E402


PHASE = 911
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase911_full_vocab_blocker_displacement_audit")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def specs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "control_label": "route_only_alpha_1",
            "control_family": "prompt_preserving_route_control",
            "control_kind": "route_only",
            "neural_intervention": True,
            "diagnostic_only": False,
            "alpha": 1.0,
        }
    ]
    for k in [1, 3, 8, 16, 32]:
        out.append(
            {
                "control_label": f"route_logit_mask_blocker_top{k}",
                "control_family": "logit_blocker_mask_diagnostic",
                "control_kind": "logit_mask",
                "neural_intervention": False,
                "diagnostic_only": True,
                "mask_topk_blockers": k,
            }
        )
    for beta in [0.05, 0.10, 0.25, 0.50]:
        out.append(
            {
                "control_label": f"route_plus_unembed_margin_top1_beta_{beta:g}",
                "control_family": "internal_readout_margin_direction",
                "control_kind": "readout_direction",
                "direction_kind": "eos_minus_blocker_top1",
                "neural_intervention": True,
                "diagnostic_only": False,
                "beta": beta,
            }
        )
    for beta in [0.10, 0.25, 0.50]:
        out.append(
            {
                "control_label": f"route_minus_unembed_blocker_top1_beta_{beta:g}",
                "control_family": "internal_readout_blocker_suppression",
                "control_kind": "readout_direction",
                "direction_kind": "minus_blocker_top1",
                "neural_intervention": True,
                "diagnostic_only": False,
                "beta": beta,
            }
        )
        out.append(
            {
                "control_label": f"route_minus_unembed_blocker_top3_beta_{beta:g}",
                "control_family": "internal_readout_blocker_suppression",
                "control_kind": "readout_direction",
                "direction_kind": "minus_blocker_top3_mean",
                "neural_intervention": True,
                "diagnostic_only": False,
                "beta": beta,
            }
        )
        out.append(
            {
                "control_label": f"route_plus_unembed_eos_beta_{beta:g}",
                "control_family": "internal_readout_eos_boost",
                "control_kind": "readout_direction",
                "direction_kind": "eos_boost",
                "neural_intervention": True,
                "diagnostic_only": False,
                "beta": beta,
            }
        )
    return out


def install_l0_output_vector(model, vector: torch.Tensor) -> list[Any]:
    module = p903.component_module(model, 0, "attention")
    if module is None:
        return []

    def hook(_module, _inputs, output):
        tensor = p910.attn_output_tensor(output)
        if tensor is None:
            return output
        patched = tensor.clone()
        local = vector.to(device=patched.device, dtype=patched.dtype)
        if patched.ndim >= 3:
            patched[:, -1, :] += local
        elif patched.ndim >= 2:
            patched[-1, :] += local
        return p910.replace_attn_output(output, patched)

    return [module.register_forward_hook(hook)]


def logits_with_l0_vector(model, device: torch.device, current_ids: list[int], vector: torch.Tensor) -> torch.Tensor | None:
    handles = install_l0_output_vector(model, vector)
    if not handles:
        return None
    try:
        return p903.logits_plain(model, device, current_ids)
    finally:
        for handle in handles:
            handle.remove()


def output_embedding_weight(model) -> torch.Tensor | None:
    try:
        emb = model.get_output_embeddings()
    except Exception:
        emb = None
    if emb is None or not hasattr(emb, "weight"):
        return None
    weight = getattr(emb, "weight")
    if not torch.is_tensor(weight):
        return None
    return weight.detach().float().cpu()


def unit_vector(vector: torch.Tensor) -> torch.Tensor | None:
    norm = torch.linalg.vector_norm(vector.float())
    if float(norm.item()) <= 0:
        return None
    return vector.float() / norm


def readout_direction(
    lm_weight: torch.Tensor | None,
    hidden_dim: int,
    eos_id: int | None,
    blocker_ids: list[int],
    direction_kind: str,
) -> torch.Tensor | None:
    if lm_weight is None or eos_id is None:
        return None
    if lm_weight.ndim != 2 or lm_weight.shape[1] != int(hidden_dim):
        return None
    eos_id = int(eos_id)
    if not (0 <= eos_id < lm_weight.shape[0]):
        return None
    valid_blockers = [int(token_id) for token_id in blocker_ids if 0 <= int(token_id) < lm_weight.shape[0]]
    if direction_kind == "eos_boost":
        return unit_vector(lm_weight[eos_id])
    if not valid_blockers:
        return None
    if direction_kind == "eos_minus_blocker_top1":
        return unit_vector(lm_weight[eos_id] - lm_weight[valid_blockers[0]])
    if direction_kind == "minus_blocker_top1":
        return unit_vector(-lm_weight[valid_blockers[0]])
    if direction_kind == "minus_blocker_top3_mean":
        blockers = lm_weight[valid_blockers[:3]].mean(dim=0)
        return unit_vector(-blockers)
    return None


def top_non_eos_ids(top_rows: list[dict[str, Any]], limit: int) -> list[int]:
    out: list[int] = []
    for row in top_rows:
        if row.get("category") == "eos":
            continue
        token_id = row.get("token_id")
        if token_id is None:
            continue
        token_id = int(token_id)
        if token_id not in out:
            out.append(token_id)
        if len(out) >= int(limit):
            break
    return out


def rank_for_token(logits: torch.Tensor, token_id: int | None) -> int | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    score = logits[int(token_id)]
    return int((logits > score).sum().item()) + 1


def token_logit(logits: torch.Tensor, token_id: int | None) -> float | None:
    if token_id is None or not (0 <= int(token_id) < int(logits.numel())):
        return None
    return float(logits[int(token_id)].item())


def masked_logits(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    out = logits.clone()
    valid = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < int(out.numel())]
    if valid:
        out[torch.tensor(valid, dtype=torch.long, device=out.device)] = -torch.inf
    return out


def full_vocab_blocker(tokenizer, logits: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, Any] | None:
    top_rows = p910.topk_tokens(tokenizer, logits, groups, 64)
    return p910.first_non_eos_top(top_rows)


def eos_margin_vs_blocker(metrics: dict[str, Any], blocker: dict[str, Any] | None) -> float | None:
    if blocker is None or metrics.get("eos_best_logit") is None:
        return None
    return float(metrics["eos_best_logit"]) - float(blocker["logit"])


def margin(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def strict_clean_candidate(tokenizer, case: dict[str, Any], prefix_ids: list[int], eos_top1: bool) -> bool:
    if not eos_top1 or tokenizer.eos_token_id is None:
        return False
    text = tokenizer.decode([int(x) for x in prefix_ids] + [int(tokenizer.eos_token_id)], skip_special_tokens=True)
    return bool(p906.clean_flags(text, case).get("strict_clean_answer_no_protocol"))


def make_row(
    tokenizer,
    groups: dict[str, list[int]],
    source_row: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    prefix_ids: list[int],
    prefix_text: str,
    baseline_metrics: dict[str, Any],
    route_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
    baseline_logits: torch.Tensor,
    route_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    route_blocker: dict[str, Any] | None,
    patched_blocker: dict[str, Any] | None,
    route_top_rows: list[dict[str, Any]],
    patched_top_rows: list[dict[str, Any]],
    route_delta_norm: float | None,
    extra_direction_norm: float | None,
    used_blocker_ids: list[int],
) -> dict[str, Any]:
    eos_id = route_metrics.get("eos_best_id")
    route_eos_logit = route_metrics.get("eos_best_logit")
    patched_eos_logit = patched_metrics.get("eos_best_logit")
    route_blocker_id = route_blocker.get("token_id") if route_blocker else None
    route_blocker_logit = route_blocker.get("logit") if route_blocker else None
    patched_route_blocker_logit = token_logit(patched_logits, route_blocker_id)
    route_margin = eos_margin_vs_blocker(route_metrics, route_blocker)
    patched_margin = eos_margin_vs_blocker(patched_metrics, patched_blocker)
    patched_margin_vs_route_blocker = margin(patched_eos_logit, patched_route_blocker_logit)
    route_margin_vs_route_blocker = margin(route_eos_logit, route_blocker_logit)
    patched_eos_rank = patched_metrics.get("eos_rank")
    eos_top1 = bool(patched_eos_rank == 1)
    route_blocker_rank_after = rank_for_token(patched_logits, route_blocker_id)
    route_blocker_displaced = bool(route_blocker_rank_after is not None and route_blocker_rank_after > 1)
    row = {
        "phase": PHASE,
        "row_kind": "phase911_full_vocab_blocker_displacement_row",
        "model": source_row.get("model"),
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "canonical_answer": source_row.get("canonical_answer"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "prefix_text": prefix_text,
        "control_label": spec.get("control_label"),
        "control_family": spec.get("control_family"),
        "control_kind": spec.get("control_kind"),
        "direction_kind": spec.get("direction_kind"),
        "beta": spec.get("beta"),
        "mask_topk_blockers": spec.get("mask_topk_blockers"),
        "neural_intervention": bool(spec.get("neural_intervention")),
        "diagnostic_only": bool(spec.get("diagnostic_only")),
        "prompt_input_intact": True,
        "prompt_all_zero_used_as_test_control": False,
        "route_alpha": 1.0,
        "route_delta_norm": route_delta_norm,
        "extra_direction_norm": extra_direction_norm,
        "used_blocker_ids": used_blocker_ids,
        "used_blocker_tokens": [p903.decode_token(tokenizer, token_id) for token_id in used_blocker_ids],
        "baseline_eos_rank": baseline_metrics.get("eos_rank"),
        "baseline_next_top_token": baseline_metrics.get("next_top_token"),
        "baseline_next_top_category": baseline_metrics.get("next_top_category"),
        "route_eos_rank": route_metrics.get("eos_rank"),
        "route_eos_logit": route_eos_logit,
        "route_next_top_token": route_metrics.get("next_top_token"),
        "route_next_top_category": route_metrics.get("next_top_category"),
        "route_blocker_id": route_blocker_id,
        "route_blocker_token": route_blocker.get("token") if route_blocker else None,
        "route_blocker_category": route_blocker.get("category") if route_blocker else None,
        "route_blocker_logit": route_blocker_logit,
        "route_blocker_rank_after_patch": route_blocker_rank_after,
        "route_blocker_logit_after_patch": patched_route_blocker_logit,
        "route_blocker_logit_delta": margin(patched_route_blocker_logit, route_blocker_logit),
        "route_blocker_displaced": route_blocker_displaced,
        "route_eos_margin_vs_full_vocab_blocker": route_margin,
        "patched_eos_rank": patched_eos_rank,
        "patched_eos_logit": patched_eos_logit,
        "patched_eos_top1": eos_top1,
        "patched_eos_top5": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 5),
        "patched_eos_top10": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 10),
        "patched_eos_top50": bool(patched_eos_rank is not None and int(patched_eos_rank) <= 50),
        "patched_next_top_token": patched_metrics.get("next_top_token"),
        "patched_next_top_category": patched_metrics.get("next_top_category"),
        "patched_blocker_id": patched_blocker.get("token_id") if patched_blocker else None,
        "patched_blocker_token": patched_blocker.get("token") if patched_blocker else None,
        "patched_blocker_category": patched_blocker.get("category") if patched_blocker else None,
        "patched_blocker_logit": patched_blocker.get("logit") if patched_blocker else None,
        "patched_eos_margin_vs_full_vocab_blocker": patched_margin,
        "patched_eos_margin_nonnegative": bool(patched_margin is not None and patched_margin >= 0),
        "patched_eos_margin_vs_route_blocker": patched_margin_vs_route_blocker,
        "eos_margin_delta_vs_route_blocker": margin(patched_margin_vs_route_blocker, route_margin_vs_route_blocker),
        "eos_margin_delta_vs_full_vocab_blocker": margin(patched_margin, route_margin),
        "eos_logit_delta_vs_route": margin(patched_eos_logit, route_eos_logit),
        "eos_rank_delta_vs_route": None
        if patched_eos_rank is None or route_metrics.get("eos_rank") is None
        else int(patched_eos_rank) - int(route_metrics["eos_rank"]),
        "strict_clean_candidate": strict_clean_candidate(tokenizer, case, prefix_ids, eos_top1),
        "route_top8": route_top_rows[:8],
        "patched_top8": patched_top_rows[:8],
    }
    if eos_id is not None:
        row["eos_id"] = int(eos_id)
        row["eos_token"] = p903.decode_token(tokenizer, int(eos_id))
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    internal_rows = [row for row in rows if row.get("neural_intervention")]
    diagnostic_rows = [row for row in rows if row.get("diagnostic_only")]
    return {
        "rows": len(rows),
        "internal_rows": len(internal_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "route_eos_top10": sum(1 for row in rows if row.get("route_eos_rank") is not None and int(row["route_eos_rank"]) <= 10),
        "route_eos_top50": sum(1 for row in rows if row.get("route_eos_rank") is not None and int(row["route_eos_rank"]) <= 50),
        "patched_eos_top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "patched_eos_top5": sum(1 for row in rows if row.get("patched_eos_top5")),
        "patched_eos_top10": sum(1 for row in rows if row.get("patched_eos_top10")),
        "patched_eos_top50": sum(1 for row in rows if row.get("patched_eos_top50")),
        "internal_eos_top1": sum(1 for row in internal_rows if row.get("patched_eos_top1")),
        "internal_eos_top5": sum(1 for row in internal_rows if row.get("patched_eos_top5")),
        "internal_eos_top10": sum(1 for row in internal_rows if row.get("patched_eos_top10")),
        "internal_eos_top50": sum(1 for row in internal_rows if row.get("patched_eos_top50")),
        "diagnostic_eos_top1": sum(1 for row in diagnostic_rows if row.get("patched_eos_top1")),
        "diagnostic_eos_top5": sum(1 for row in diagnostic_rows if row.get("patched_eos_top5")),
        "diagnostic_eos_top10": sum(1 for row in diagnostic_rows if row.get("patched_eos_top10")),
        "diagnostic_eos_top50": sum(1 for row in diagnostic_rows if row.get("patched_eos_top50")),
        "route_blocker_displaced": sum(1 for row in rows if row.get("route_blocker_displaced")),
        "internal_route_blocker_displaced": sum(1 for row in internal_rows if row.get("route_blocker_displaced")),
        "diagnostic_route_blocker_displaced": sum(1 for row in diagnostic_rows if row.get("route_blocker_displaced")),
        "patched_eos_margin_nonnegative": sum(1 for row in rows if row.get("patched_eos_margin_nonnegative")),
        "internal_eos_margin_nonnegative": sum(1 for row in internal_rows if row.get("patched_eos_margin_nonnegative")),
        "diagnostic_eos_margin_nonnegative": sum(1 for row in diagnostic_rows if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "internal_strict_clean_candidate": sum(1 for row in internal_rows if row.get("strict_clean_candidate")),
        "diagnostic_strict_clean_candidate": sum(1 for row in diagnostic_rows if row.get("strict_clean_candidate")),
        "median_route_eos_margin_vs_blocker": median([row.get("route_eos_margin_vs_full_vocab_blocker") for row in rows]),
        "median_patched_eos_margin_vs_blocker": median([row.get("patched_eos_margin_vs_full_vocab_blocker") for row in rows]),
        "median_eos_margin_delta_vs_blocker": median([row.get("eos_margin_delta_vs_full_vocab_blocker") for row in rows]),
        "median_route_blocker_logit_delta": median([row.get("route_blocker_logit_delta") for row in rows]),
        "route_blocker_categories": dict(sorted(Counter(str(row.get("route_blocker_category")) for row in rows).items())),
        "patched_blocker_categories": dict(sorted(Counter(str(row.get("patched_blocker_category")) for row in rows).items())),
        "route_blocker_tokens_top12": dict(Counter(str(row.get("route_blocker_token")) for row in rows).most_common(12)),
        "patched_blocker_tokens_top12": dict(Counter(str(row.get("patched_blocker_token")) for row in rows).most_common(12)),
    }


def summarize_by_control(rows: list[dict[str, Any]], top_n: int = 80) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("control_label"))].append(row)
    summaries = []
    for label, vals in buckets.items():
        summary = summarize_rows(vals)
        first = vals[0]
        summary.update(
            {
                "control_label": label,
                "control_family": first.get("control_family"),
                "control_kind": first.get("control_kind"),
                "direction_kind": first.get("direction_kind"),
                "beta": first.get("beta"),
                "mask_topk_blockers": first.get("mask_topk_blockers"),
                "neural_intervention": first.get("neural_intervention"),
                "diagnostic_only": first.get("diagnostic_only"),
            }
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            row.get("internal_strict_clean_candidate") or 0,
            row.get("internal_eos_top1") or 0,
            row.get("internal_eos_top5") or 0,
            row.get("internal_eos_top10") or 0,
            row.get("internal_eos_margin_nonnegative") or 0,
            row.get("diagnostic_eos_top1") or 0,
            row.get("diagnostic_eos_top5") or 0,
            row.get("diagnostic_eos_top10") or 0,
            row.get("patched_eos_top50") or 0,
            row.get("median_eos_margin_delta_vs_blocker") or -9999,
        ),
        reverse=True,
    )
    return summaries[:top_n]


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_count: int, attn_impl: str | None) -> dict[str, Any]:
    overall = summarize_rows(rows)
    controls = summarize_by_control(rows)
    if overall["internal_eos_top1"] > 0:
        evidence = "internal_blocker_displacement_reaches_eos_top1"
    elif overall["internal_eos_top5"] > 0:
        evidence = "internal_blocker_displacement_reaches_eos_top5"
    elif overall["internal_eos_margin_nonnegative"] > 0:
        evidence = "internal_blocker_margin_crosses_zero"
    elif overall["diagnostic_eos_top1"] > 0:
        evidence = "logit_mask_diagnostic_shows_narrow_blocker_bottleneck"
    elif overall["diagnostic_eos_top10"] > 0:
        evidence = "logit_mask_diagnostic_shows_multi_blocker_bottleneck"
    elif overall["route_eos_top50"] > 0:
        evidence = "route_near_but_blocker_not_displaced"
    else:
        evidence = "no_route_near_and_no_blocker_displacement"
    return {
        "phase": PHASE,
        "title": "Full-vocabulary Blocker Displacement Audit",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "selected_answer_drift_rows": selected_count,
        "intervention_count": len(specs()),
        "overall": overall,
        "control_summaries": controls,
        "evidence_label": evidence,
        "boundary": (
            "Phase911 fixes the Phase910 prompt-preserving route and tests whether the remaining "
            "full-vocabulary blocker field can be displaced. Logit masks are diagnostic only."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = p906.selected_phase899_rows(args.model, args)
    all_specs = specs()
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
            "specs": all_specs,
        }
        p846.write_json(out_dir / f"phase911_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase911_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
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
        groups = p903.protocol_category_groups(tokenizer)
        lm_weight = output_embedding_weight(model)
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(selected_rows, 1):
            case = case_map.get(str(source_row.get("case_id")))
            if not case:
                continue
            prompt_key = (str(source_row.get("case_id")), str(source_row.get("prompt_variant")))
            if prompt_key not in prompt_cache:
                prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
                prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
            prompt_ids = prompt_cache[prompt_key]
            gears = p903.parse_gears(str(source_row.get("source_subset_key")))
            _prefix_logits, prefix_ids, prefix_text, _answer_seen = p901.logits_after_answer_prefix(
                model,
                tokenizer,
                device,
                prompt_ids,
                gears,
                str(source_row.get("edit_mode")),
                case,
                int(args.max_prefix_tokens),
                float(args.scale_up_factor),
            )
            current_ids = [int(x) for x in prompt_ids] + [int(x) for x in prefix_ids]
            answer_logits = p903.logits_plain(model, device, current_ids)
            answer_metrics = p903.state_metrics(tokenizer, answer_logits, groups)
            period_id = answer_metrics.get("period_best_id") or ((groups.get("period") or [None])[0])
            if period_id is None:
                continue
            period_ids = current_ids + [int(period_id)]
            baseline_logits, base_vec = p910.logits_and_l0_vector(model, device, period_ids)
            baseline_metrics = p903.state_metrics(tokenizer, baseline_logits, groups)
            prompt_zero_handles = p909.install_attention_input_span_scale(model, 0, 0, len(prompt_ids), 0.0)
            _prompt_zero_logits, prompt_zero_vec = p910.logits_and_l0_vector(model, device, period_ids, prompt_zero_handles)
            if base_vec is None or prompt_zero_vec is None:
                continue
            route_delta = prompt_zero_vec - base_vec
            route_delta_norm = float(torch.linalg.vector_norm(route_delta).item())
            if route_delta_norm <= 0:
                continue
            route_logits = logits_with_l0_vector(model, device, period_ids, route_delta)
            if route_logits is None:
                continue
            route_metrics = p903.state_metrics(tokenizer, route_logits, groups)
            route_top_rows = p910.topk_tokens(tokenizer, route_logits, groups, 64)
            route_blocker = p910.first_non_eos_top(route_top_rows)
            blocker_ids = top_non_eos_ids(route_top_rows, 64)
            hidden_dim = int(route_delta.numel())
            for spec in all_specs:
                extra_direction_norm = None
                used_blocker_ids: list[int] = []
                if spec["control_kind"] == "route_only":
                    patched_logits = route_logits
                    used_blocker_ids = []
                elif spec["control_kind"] == "logit_mask":
                    used_blocker_ids = blocker_ids[: int(spec["mask_topk_blockers"])]
                    patched_logits = masked_logits(route_logits, used_blocker_ids)
                elif spec["control_kind"] == "readout_direction":
                    used_blocker_ids = blocker_ids[:3]
                    direction = readout_direction(
                        lm_weight,
                        hidden_dim,
                        route_metrics.get("eos_best_id"),
                        blocker_ids,
                        str(spec["direction_kind"]),
                    )
                    if direction is None:
                        continue
                    beta = float(spec["beta"])
                    extra = direction * (beta * route_delta_norm)
                    extra_direction_norm = float(torch.linalg.vector_norm(extra).item())
                    patched_logits = logits_with_l0_vector(model, device, period_ids, route_delta + extra)
                    if patched_logits is None:
                        continue
                else:
                    continue
                patched_metrics = p903.state_metrics(tokenizer, patched_logits, groups)
                patched_top_rows = p910.topk_tokens(tokenizer, patched_logits, groups, 64)
                patched_blocker = p910.first_non_eos_top(patched_top_rows)
                rows.append(
                    make_row(
                        tokenizer,
                        groups,
                        source_row,
                        case,
                        spec,
                        prefix_ids,
                        prefix_text,
                        baseline_metrics,
                        route_metrics,
                        patched_metrics,
                        baseline_logits,
                        route_logits,
                        patched_logits,
                        route_blocker,
                        patched_blocker,
                        route_top_rows,
                        patched_top_rows,
                        route_delta_norm,
                        extra_direction_norm,
                        used_blocker_ids,
                    )
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, len(selected_rows), attn_impl)
    p846.write_json(out_dir / f"phase911_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase911_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "overall": payload["overall"],
                "evidence_label": payload["evidence_label"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    scalar = Counter()
    evidence = Counter()
    top_controls = []
    for model_name in MODELS:
        path = out_dir / f"phase911_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in [
            "rows",
            "internal_rows",
            "diagnostic_rows",
            "route_eos_top10",
            "route_eos_top50",
            "patched_eos_top1",
            "patched_eos_top5",
            "patched_eos_top10",
            "patched_eos_top50",
            "internal_eos_top1",
            "internal_eos_top5",
            "internal_eos_top10",
            "internal_eos_top50",
            "diagnostic_eos_top1",
            "diagnostic_eos_top5",
            "diagnostic_eos_top10",
            "diagnostic_eos_top50",
            "route_blocker_displaced",
            "internal_route_blocker_displaced",
            "diagnostic_route_blocker_displaced",
            "patched_eos_margin_nonnegative",
            "internal_eos_margin_nonnegative",
            "diagnostic_eos_margin_nonnegative",
            "strict_clean_candidate",
            "internal_strict_clean_candidate",
            "diagnostic_strict_clean_candidate",
        ]:
            scalar[key] += int(overall.get(key) or 0)
        for row in summary.get("control_summaries") or []:
            item = dict(row)
            item["model"] = summary.get("model")
            top_controls.append(item)
    top_controls.sort(
        key=lambda row: (
            row.get("internal_strict_clean_candidate") or 0,
            row.get("internal_eos_top1") or 0,
            row.get("internal_eos_top5") or 0,
            row.get("internal_eos_top10") or 0,
            row.get("internal_eos_margin_nonnegative") or 0,
            row.get("diagnostic_eos_top1") or 0,
            row.get("diagnostic_eos_top5") or 0,
            row.get("diagnostic_eos_top10") or 0,
            row.get("patched_eos_top50") or 0,
            row.get("median_eos_margin_delta_vs_blocker") or -9999,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "top_controls": top_controls[:80],
    }
    p846.write_json(out_dir / "phase911_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase911_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 911 full-vocabulary blocker displacement audit",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Model Summaries", ""])
    lines.append(
        "| model | rows | route top10 | route top50 | internal top1 | internal top5 | internal top10 | diagnostic top1 | diagnostic top10 | internal margin>=0 | strict clean | evidence |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in payload.get("model_summaries") or []:
        overall = summary.get("overall") or {}
        lines.append(
            "| {model} | {rows} | {route10} | {route50} | {itop1} | {itop5} | {itop10} | {dtop1} | {dtop10} | {imargin} | {clean} | {evidence} |".format(
                model=summary.get("model"),
                rows=overall.get("rows"),
                route10=overall.get("route_eos_top10"),
                route50=overall.get("route_eos_top50"),
                itop1=overall.get("internal_eos_top1"),
                itop5=overall.get("internal_eos_top5"),
                itop10=overall.get("internal_eos_top10"),
                dtop1=overall.get("diagnostic_eos_top1"),
                dtop10=overall.get("diagnostic_eos_top10"),
                imargin=overall.get("internal_eos_margin_nonnegative"),
                clean=overall.get("strict_clean_candidate"),
                evidence=summary.get("evidence_label"),
            )
        )
    lines.extend(["", "## Top Controls", ""])
    lines.append(
        "| model | control | family | neural | diagnostic | rows | internal top1 | internal top5 | internal top10 | diagnostic top1 | diagnostic top10 | margin>=0 | median margin delta | route blockers | patched blockers |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload.get("top_controls") or []:
        lines.append(
            "| {model} | {control_label} | {control_family} | {neural_intervention} | {diagnostic_only} | {rows} | {internal_eos_top1} | {internal_eos_top5} | {internal_eos_top10} | {diagnostic_eos_top1} | {diagnostic_eos_top10} | {patched_eos_margin_nonnegative} | {median_eos_margin_delta_vs_blocker} | {route_blocker_tokens_top12} | {patched_blocker_tokens_top12} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="full_vocab_blocker_displacement_audit")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=5)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
