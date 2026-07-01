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

import phase831_internal_gate_predictor_search as p831  # noqa: E402
import phase832_gate_signal_expansion_beyond_readout as p832  # noqa: E402
import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase829_non_interference_constrained_component_composition as p829  # noqa: E402


PHASE = 833
RESULT_ROOT = Path("tests/result/phase833_internal_route_boundary_predictor_search")

TARGET_CLASSES = {"target_equivalent"}


def log(msg: str) -> None:
    p829.log(msg)


def predictor_modes(args: argparse.Namespace) -> list[str]:
    return [x.strip() for x in str(args.predictor_modes).split(",") if x.strip()]


def route_class(group: dict[str, Any], donor_variant: str) -> str:
    return str((group.get("single_donor_classes") or {}).get(donor_variant) or "unknown_other")


def component_kind(group: dict[str, Any]) -> str:
    return str(group.get("component_kind") or "")


def is_non_residual(group: dict[str, Any]) -> bool:
    return component_kind(group) != "layer_residual"


def score_axis(vals: torch.Tensor) -> dict[str, Any]:
    if vals.numel() <= 0:
        return {
            "signed_sum": 0.0,
            "signed_mean": 0.0,
            "positive_sum": 0.0,
            "negative_abs_sum": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_ratio": 0.0,
            "top_abs_signed_score": 0.0,
        }
    pos = torch.clamp(vals, min=0.0)
    neg = torch.clamp(-vals, min=0.0)
    top = vals[torch.argmax(torch.abs(vals))]
    pos_count = int((vals > 0).sum().item())
    neg_count = int((vals < 0).sum().item())
    denom = pos_count + neg_count
    return {
        "signed_sum": float(vals.sum().item()),
        "signed_mean": float(vals.mean().item()),
        "positive_sum": float(pos.sum().item()),
        "negative_abs_sum": float(neg.sum().item()),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_ratio": float(pos_count / denom) if denom else 0.0,
        "top_abs_signed_score": float(top.item()),
    }


def internal_features(
    recipient_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    effective_dir: torch.Tensor,
    selected_indices: list[int],
) -> dict[str, Any]:
    n = min(int(recipient_vec.numel()), int(donor_vec.numel()), int(effective_dir.numel()))
    if n <= 0:
        return {}
    recipient = recipient_vec[:n].float()
    donor = donor_vec[:n].float()
    readout = effective_dir[:n].float()
    delta = donor - recipient
    full_signed = delta * readout
    full = score_axis(full_signed)

    selected = [int(i) for i in selected_indices if 0 <= int(i) < n]
    if selected:
        idx = torch.tensor(selected, dtype=torch.long)
        selected_signed = full_signed[idx]
        sel = score_axis(selected_signed)
        selected_delta = delta[idx]
        selected_recipient = recipient[idx]
        selected_donor = donor[idx]
        selected_readout = readout[idx]
    else:
        idx = torch.tensor([], dtype=torch.long)
        sel = score_axis(torch.empty(0))
        selected_delta = torch.empty(0)
        selected_recipient = torch.empty(0)
        selected_donor = torch.empty(0)
        selected_readout = torch.empty(0)

    selected_pos_energy = float(sel["positive_sum"])
    selected_neg_energy = float(sel["negative_abs_sum"])
    selected_energy = selected_pos_energy + selected_neg_energy
    full_pos_energy = float(full["positive_sum"])
    full_neg_energy = float(full["negative_abs_sum"])
    full_energy = full_pos_energy + full_neg_energy

    out = {
        "n_selected_for_gate": len(selected),
        "selected_signed_sum": float(sel["signed_sum"]),
        "selected_signed_mean": float(sel["signed_mean"]),
        "selected_positive_sum": selected_pos_energy,
        "selected_negative_abs_sum": selected_neg_energy,
        "selected_positive_count": int(sel["positive_count"]),
        "selected_negative_count": int(sel["negative_count"]),
        "selected_positive_ratio": float(sel["positive_ratio"]),
        "selected_top_abs_signed_score": float(sel["top_abs_signed_score"]),
        "selected_energy_ratio": float(selected_pos_energy / selected_energy) if selected_energy else 0.0,
        "selected_delta_norm": float(selected_delta.norm().item()) if selected_delta.numel() else 0.0,
        "selected_recipient_score": float((selected_recipient * selected_readout).sum().item())
        if selected_recipient.numel()
        else 0.0,
        "selected_donor_score": float((selected_donor * selected_readout).sum().item())
        if selected_donor.numel()
        else 0.0,
        "full_signed_sum": float(full["signed_sum"]),
        "full_signed_mean": float(full["signed_mean"]),
        "full_positive_sum": full_pos_energy,
        "full_negative_abs_sum": full_neg_energy,
        "full_positive_count": int(full["positive_count"]),
        "full_negative_count": int(full["negative_count"]),
        "full_positive_ratio": float(full["positive_ratio"]),
        "full_top_abs_signed_score": float(full["top_abs_signed_score"]),
        "full_energy_ratio": float(full_pos_energy / full_energy) if full_energy else 0.0,
        "full_delta_norm": float(delta.norm().item()),
        "full_recipient_score": float((recipient * readout).sum().item()),
        "full_donor_score": float((donor * readout).sum().item()),
        "recipient_norm": float(recipient.norm().item()),
        "donor_norm": float(donor.norm().item()),
    }
    out["selected_donor_minus_recipient_score"] = out["selected_donor_score"] - out["selected_recipient_score"]
    out["full_donor_minus_recipient_score"] = out["full_donor_score"] - out["full_recipient_score"]
    return out


def count_ge(features: dict[str, Any], prefix: str = "selected") -> bool:
    return int(features.get(f"{prefix}_positive_count") or 0) >= int(features.get(f"{prefix}_negative_count") or 0)


def count_gt(features: dict[str, Any], prefix: str = "selected") -> bool:
    return int(features.get(f"{prefix}_positive_count") or 0) > int(features.get(f"{prefix}_negative_count") or 0)


def sum_pos(features: dict[str, Any], prefix: str = "selected") -> bool:
    return float(features.get(f"{prefix}_signed_sum") or 0.0) > 0.0


def donor_gain_pos(features: dict[str, Any], prefix: str = "selected") -> bool:
    return float(features.get(f"{prefix}_donor_minus_recipient_score") or 0.0) > 0.0


def predict_gate(
    features: dict[str, Any],
    group: dict[str, Any],
    donor_variant: str,
    cls: str,
    mode: str,
) -> bool:
    non_residual = is_non_residual(group)
    is_category = donor_variant == "natural_category"
    if mode == "oracle_route_target_only":
        return cls in TARGET_CLASSES
    if mode == "selected_sum_positive":
        return sum_pos(features, "selected")
    if mode == "selected_count_majority":
        return count_gt(features, "selected")
    if mode == "selected_count_nonnegative":
        return count_ge(features, "selected")
    if mode == "selected_energy_ratio_55":
        return float(features.get("selected_energy_ratio") or 0.0) >= 0.55
    if mode == "full_sum_positive":
        return sum_pos(features, "full")
    if mode == "full_count_majority":
        return count_gt(features, "full")
    if mode == "donor_selected_gain_positive":
        return donor_gain_pos(features, "selected")
    if mode == "donor_full_gain_positive":
        return donor_gain_pos(features, "full")
    if mode == "non_residual_only":
        return non_residual
    if mode == "non_residual_count_nonnegative":
        return non_residual and count_ge(features, "selected")
    if mode == "category_nonresidual_else_count_nonnegative":
        if is_category:
            return non_residual and count_ge(features, "selected")
        return count_ge(features, "selected")
    if mode == "category_nonresidual_else_sum_positive":
        if is_category:
            return non_residual and sum_pos(features, "selected")
        return sum_pos(features, "selected")
    if mode == "category_nonresidual_else_gain_positive":
        if is_category:
            return non_residual and donor_gain_pos(features, "selected")
        return donor_gain_pos(features, "selected")
    raise ValueError(f"unknown predictor mode: {mode}")


def eval_pair(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    pair: tuple[dict[str, Any], dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = p828.p820.standard_lookup(standards)
    recipient_prompt = p828.p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p831.p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p831.p823.greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = p828.p825.boundary_for(lookup, case["case_id"], baseline_text)
    comp_data = p831.prepare_component_data(model, tokenizer, device, recipient_prompt, baseline_ids, case, pair, args)
    if not comp_data:
        return [], []

    a, b = pair
    modes = predictor_modes(args)
    decision_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for donor_variant in p828.parse_csv(args.search_donor_prompts):
        donor_prompt = p828.p825.natural_prompt(case, donor_variant)
        patch_items_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
        active_labels_by_mode: dict[str, list[str]] = {mode: [] for mode in modes}
        for item in comp_data:
            group = item["group"]
            cls = route_class(group, donor_variant)
            donor_state = p828.p822.capture_component_state(model, tokenizer, device, donor_prompt, int(group["layer_idx"]))
            donor_vec = p831.p823.component_vector(donor_state, item["spec"])
            if donor_vec is None:
                continue
            donor_vec = donor_vec.float().cpu()
            features = internal_features(
                item["recipient_vec"], donor_vec, item["effective_dir"], item["selected_indices"]
            )
            oracle_gate = cls in TARGET_CLASSES
            label = p828.compact_component_label(group)
            for mode in modes:
                pred = predict_gate(features, group, donor_variant, cls, mode)
                decision_rows.append(
                    {
                        "row_kind": "phase833_internal_route_boundary_predictor_decision",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "case_id": case["case_id"],
                        "object": case["object"],
                        "target_answer": case["answer"],
                        "pair_label": f"{p828.compact_component_label(a)} + {p828.compact_component_label(b)}",
                        "component_label_full": label,
                        "component_group": group,
                        "component_kind": component_kind(group),
                        "donor_variant": donor_variant,
                        "predictor_mode": mode,
                        "uses_behavior_label_as_input": mode == "oracle_route_target_only",
                        "oracle_target_only_gate": bool(oracle_gate),
                        "predicted_gate": bool(pred),
                        "decision_correct_vs_oracle": bool(pred) == bool(oracle_gate),
                        "single_donor_class": cls,
                        "readout_meta": item.get("readout_meta"),
                        **features,
                    }
                )
                if pred:
                    patch_items_by_mode[mode].append(
                        {
                            "layer_idx": int(group["layer_idx"]),
                            "spec": item["spec"],
                            "recipient_vec": item["recipient_vec"],
                            "donor_vec": donor_vec,
                            "selected_indices": item["selected_indices"],
                        }
                    )
                    active_labels_by_mode[mode].append(label)

        for mode in modes:
            patched_text, patched_ids = p828.greedy_generate_with_multi_patch(
                model,
                tokenizer,
                device,
                recipient_ids,
                patch_items_by_mode[mode],
                args.max_new_tokens,
                args.alpha,
            )
            patched_boundary = p828.p825.boundary_for(lookup, case["case_id"], patched_text)
            delta_rank = int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"])
            generation_rows.append(
                {
                    "row_kind": "phase833_internal_route_boundary_predictor_generation",
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
                    "predictor_mode": mode,
                    "validation_mode": "internal_route_boundary_predictor_search",
                    "active_component_labels": active_labels_by_mode[mode],
                    "n_active_components": len(active_labels_by_mode[mode]),
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
    return decision_rows, generation_rows


def summarize_rows(
    decision_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
    attn_impl: str | None = None,
) -> dict[str, Any]:
    base = p831.summarize_rows(decision_rows, generation_rows, groups, pairs, diagnostics, args, attn_impl)
    base["phase"] = PHASE
    base["title"] = "Internal Route-Boundary Predictor Search"
    base["boundary"] = (
        "This phase searches for basic internal or protocol-structural proxies for route boundary. "
        "oracle_route_target_only is included only as a behavioral upper bound."
    )
    return base


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p828.p820.standard_rows()
    groups = p828.load_component_groups(args.model, args)
    pairs, diagnostics = p829.build_constrained_pairs(groups, args)
    cmap = p828.p825.case_map()
    log(f"{args.model}/{args.round_name}: groups={len(groups)} pairs={len(pairs)} modes={predictor_modes(args)}")
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
        summary = summarize_rows([], [], groups, pairs, diagnostics, args, attn_impl=None)
        summary["skipped_model_load"] = True
        summary["skip_reason"] = "no eligible same-case pairs for internal route-boundary prediction"
        p828.write_jsonl(out_dir / f"phase833_{args.model}_decision_rows.jsonl", [])
        p828.write_jsonl(out_dir / f"phase833_{args.model}_generation_rows.jsonl", [])
        p828.write_json(out_dir / f"phase833_{args.model}_summary.json", summary)
        print(json.dumps({"model": args.model, "round": args.round_name, "pairs": 0, "skipped_model_load": True}, ensure_ascii=False, indent=2))
        return summary

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(
        args.model, args.attn_implementations
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decision_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    try:
        for idx, pair in enumerate(pairs, 1):
            case = cmap.get(str(pair[0]["case_id"]))
            if not case:
                continue
            drows, grows = eval_pair(model, tokenizer, device, standards, case, pair, args)
            decision_rows.extend(drows)
            generation_rows.extend(grows)
            if idx % int(args.log_every) == 0 or idx == len(pairs):
                log(
                    f"{args.model}: evaluated {idx}/{len(pairs)} pairs; "
                    f"decisions={len(decision_rows)} generations={len(generation_rows)}"
                )
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(decision_rows, generation_rows, groups, pairs, diagnostics, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase833_{args.model}_decision_rows.jsonl", decision_rows)
    p828.write_jsonl(out_dir / f"phase833_{args.model}_generation_rows.jsonl", generation_rows)
    p828.write_json(out_dir / f"phase833_{args.model}_summary.json", summary)
    printable = {
        "model": args.model,
        "round": args.round_name,
        "pairs": summary["n_pairs"],
        "mode_summaries": {
            mode: {
                "accuracy": data["decision"].get("accuracy_vs_target_only"),
                "pair_exact_plus_multi": data["generation"].get("pair_exact_plus_multi"),
                "natural_target_rows": data["generation"].get("natural_target_rows"),
                "natural_degraded_rows": data["generation"].get("natural_degraded_rows"),
                "natural_category_target_rows": data["generation"].get("natural_category_target_rows"),
                "active": data["generation"].get("active_component_count_distribution"),
            }
            for mode, data in summary["mode_summaries"].items()
        },
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 833 Internal Route-Boundary Predictor Search ({payload['round']})",
        "",
        "- Source: Phase 829 pairs, Phase 831 internal features, and Phase 832 route-boundary upper bound.",
        "- Objective: test whether simple internal/protocol-structural proxies can replace behavioral route labels.",
        "",
        "## Model Summary",
        "",
        "| model | predictor | decision acc | TP | TN | FP | FN | pair exact+multi | natural target | natural degraded | nat_category target | nat_category degraded | active components |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for mode, mode_data in (data.get("mode_summaries") or {}).items():
            dec = mode_data.get("decision") or {}
            gen = mode_data.get("generation") or {}
            acc = dec.get("accuracy_vs_target_only")
            acc_text = "" if acc is None else f"{float(acc):.3f}"
            lines.append(
                f"| {model_name} | `{mode}` | {acc_text} | {dec.get('tp')} | {dec.get('tn')} | "
                f"{dec.get('fp')} | {dec.get('fn')} | {gen.get('pair_exact_plus_multi')} | "
                f"{gen.get('natural_target_rows')} | {gen.get('natural_degraded_rows')} | "
                f"{gen.get('natural_category_target_rows')} | {gen.get('natural_category_degraded_rows')} | "
                f"`{json.dumps(gen.get('active_component_count_distribution') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Boundary", ""]
    lines.append(
        "The non-oracle modes do not use single-component behavioral classes as gate inputs. "
        "If they approach oracle_route_target_only, route boundary has an internal/protocol-structural proxy; "
        "if not, Phase 832's route boundary remains behavioral rather than endogenous."
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
        path = out_dir / f"phase833_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase833_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase833_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p829.build_parser()
    parser.add_argument(
        "--predictor-modes",
        default=(
            "selected_sum_positive,selected_count_majority,selected_count_nonnegative,"
            "selected_energy_ratio_55,full_sum_positive,full_count_majority,"
            "donor_selected_gain_positive,non_residual_count_nonnegative,"
            "category_nonresidual_else_count_nonnegative,category_nonresidual_else_sum_positive,"
            "category_nonresidual_else_gain_positive,oracle_route_target_only"
        ),
    )
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
