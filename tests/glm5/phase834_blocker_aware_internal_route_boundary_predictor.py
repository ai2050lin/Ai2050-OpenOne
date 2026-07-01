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
import phase833_internal_route_boundary_predictor_search as p833  # noqa: E402
import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase829_non_interference_constrained_component_composition as p829  # noqa: E402


PHASE = 834
RESULT_ROOT = Path("tests/result/phase834_blocker_aware_internal_route_boundary_predictor")
TARGET_CLASSES = {"target_equivalent"}


def log(msg: str) -> None:
    p829.log(msg)


def predictor_modes(args: argparse.Namespace) -> list[str]:
    return [x.strip() for x in str(args.predictor_modes).split(",") if x.strip()]


def first_step_logits(
    model,
    device: torch.device,
    prompt_ids: list[int],
    patch_items: list[dict[str, Any]],
    alpha: float,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles: list[Any] = []
    if patch_items:
        handles = p828.install_multi_patch(model, patch_items, alpha)
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits


def rank_profile(logits: torch.Tensor, target_id: int | None, baseline_id: int | None) -> dict[str, Any]:
    if target_id is None or target_id < 0 or target_id >= int(logits.numel()):
        return {
            "target_rank": None,
            "above_target_count": None,
            "target_logit": None,
            "top_logit": float(torch.max(logits).item()) if logits.numel() else None,
            "target_minus_top": None,
            "target_minus_baseline_token": None,
            "top_token_id": int(torch.argmax(logits).item()) if logits.numel() else None,
        }
    target_logit = float(logits[int(target_id)].item())
    top_id = int(torch.argmax(logits).item())
    top_logit = float(logits[top_id].item())
    above = int((logits > target_logit).sum().item())
    baseline_logit = None
    if baseline_id is not None and 0 <= int(baseline_id) < int(logits.numel()):
        baseline_logit = float(logits[int(baseline_id)].item())
    return {
        "target_rank": above + 1,
        "above_target_count": above,
        "target_logit": target_logit,
        "top_logit": top_logit,
        "target_minus_top": target_logit - top_logit,
        "target_minus_baseline_token": None if baseline_logit is None else target_logit - baseline_logit,
        "top_token_id": top_id,
    }


def blocker_features(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    patch_item: dict[str, Any] | None,
    args: argparse.Namespace,
    target_id: int | None,
    baseline_id: int | None,
    baseline_profile: dict[str, Any],
) -> dict[str, Any]:
    patch_items = [patch_item] if patch_item is not None else []
    logits = first_step_logits(model, device, prompt_ids, patch_items, float(args.alpha))
    profile = rank_profile(logits, target_id, baseline_id)
    top_id = profile.get("top_token_id")
    profile["top_token"] = tokenizer.decode([int(top_id)]) if top_id is not None else None
    base_rank = baseline_profile.get("target_rank")
    base_above = baseline_profile.get("above_target_count")
    rank = profile.get("target_rank")
    above = profile.get("above_target_count")
    profile["target_rank_improved"] = (
        bool(rank is not None and base_rank is not None and int(rank) < int(base_rank))
    )
    profile["above_target_decreased"] = (
        bool(above is not None and base_above is not None and int(above) < int(base_above))
    )
    return profile


def base_category_count(features: dict[str, Any], group: dict[str, Any], donor_variant: str) -> bool:
    if donor_variant == "natural_category":
        return p833.is_non_residual(group) and p833.count_ge(features, "selected")
    return p833.count_ge(features, "selected")


def blocker_safe(features: dict[str, Any], rank_limit: int) -> bool:
    rank = features.get("target_rank")
    return bool(rank is not None and int(rank) <= int(rank_limit))


def predict_gate(features: dict[str, Any], group: dict[str, Any], donor_variant: str, cls: str, mode: str) -> bool:
    if mode == "oracle_route_target_only":
        return cls in TARGET_CLASSES
    if mode == "category_nonresidual_else_count_nonnegative":
        return base_category_count(features, group, donor_variant)
    if mode == "category_count_rank_improved":
        return base_category_count(features, group, donor_variant) and bool(features.get("target_rank_improved"))
    if mode == "category_count_above_decreased":
        return base_category_count(features, group, donor_variant) and bool(features.get("above_target_decreased"))
    if mode == "category_count_rank_le50":
        return base_category_count(features, group, donor_variant) and blocker_safe(features, 50)
    if mode == "category_count_rank_le500":
        return base_category_count(features, group, donor_variant) and blocker_safe(features, 500)
    if mode == "category_count_rank_le2000":
        return base_category_count(features, group, donor_variant) and blocker_safe(features, 2000)
    if mode == "nonresidual_count_rank_le500":
        return p833.is_non_residual(group) and p833.count_ge(features, "selected") and blocker_safe(features, 500)
    if mode == "count_rank_improved":
        return p833.count_ge(features, "selected") and bool(features.get("target_rank_improved"))
    if mode == "count_rank_le500":
        return p833.count_ge(features, "selected") and blocker_safe(features, 500)
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

    target_id = comp_data[0]["readout_meta"].get("target_token_id") if comp_data else None
    baseline_id = int(baseline_ids[0]) if baseline_ids else None
    no_patch_logits = first_step_logits(model, device, recipient_ids, [], float(args.alpha))
    no_patch_profile = rank_profile(no_patch_logits, target_id, baseline_id)

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
            cls = p833.route_class(group, donor_variant)
            donor_state = p828.p822.capture_component_state(model, tokenizer, device, donor_prompt, int(group["layer_idx"]))
            donor_vec = p831.p823.component_vector(donor_state, item["spec"])
            if donor_vec is None:
                continue
            donor_vec = donor_vec.float().cpu()
            patch_item = {
                "layer_idx": int(group["layer_idx"]),
                "spec": item["spec"],
                "recipient_vec": item["recipient_vec"],
                "donor_vec": donor_vec,
                "selected_indices": item["selected_indices"],
            }
            features = p833.internal_features(
                item["recipient_vec"], donor_vec, item["effective_dir"], item["selected_indices"]
            )
            features.update(
                blocker_features(
                    model,
                    tokenizer,
                    device,
                    recipient_ids,
                    patch_item,
                    args,
                    target_id,
                    baseline_id,
                    no_patch_profile,
                )
            )
            oracle_gate = cls in TARGET_CLASSES
            label = p828.compact_component_label(group)
            for mode in modes:
                pred = predict_gate(features, group, donor_variant, cls, mode)
                decision_rows.append(
                    {
                        "row_kind": "phase834_blocker_aware_internal_route_boundary_decision",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "case_id": case["case_id"],
                        "object": case["object"],
                        "target_answer": case["answer"],
                        "pair_label": f"{p828.compact_component_label(a)} + {p828.compact_component_label(b)}",
                        "component_label_full": label,
                        "component_group": group,
                        "component_kind": p833.component_kind(group),
                        "donor_variant": donor_variant,
                        "predictor_mode": mode,
                        "uses_behavior_label_as_input": mode == "oracle_route_target_only",
                        "oracle_target_only_gate": bool(oracle_gate),
                        "predicted_gate": bool(pred),
                        "decision_correct_vs_oracle": bool(pred) == bool(oracle_gate),
                        "single_donor_class": cls,
                        "baseline_target_rank": no_patch_profile.get("target_rank"),
                        "baseline_above_target_count": no_patch_profile.get("above_target_count"),
                        "readout_meta": item.get("readout_meta"),
                        **features,
                    }
                )
                if pred:
                    patch_items_by_mode[mode].append(patch_item)
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
                    "row_kind": "phase834_blocker_aware_internal_route_boundary_generation",
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
                    "validation_mode": "blocker_aware_internal_route_boundary_predictor",
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
    base["title"] = "Blocker-Aware Internal Route-Boundary Predictor"
    base["boundary"] = (
        "This phase adds first-step full-vocabulary target-rank and blocker-count signals. "
        "It tests whether blocker awareness can remove the Phase 833 qwen3 degradation."
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
        summary["skip_reason"] = "no eligible same-case pairs for blocker-aware route-boundary prediction"
        p828.write_jsonl(out_dir / f"phase834_{args.model}_decision_rows.jsonl", [])
        p828.write_jsonl(out_dir / f"phase834_{args.model}_generation_rows.jsonl", [])
        p828.write_json(out_dir / f"phase834_{args.model}_summary.json", summary)
        print(json.dumps({"model": args.model, "round": args.round_name, "pairs": 0, "skipped_model_load": True}, ensure_ascii=False, indent=2))
        return summary

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
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
                log(f"{args.model}: evaluated {idx}/{len(pairs)} pairs; decisions={len(decision_rows)} generations={len(generation_rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(decision_rows, generation_rows, groups, pairs, diagnostics, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase834_{args.model}_decision_rows.jsonl", decision_rows)
    p828.write_jsonl(out_dir / f"phase834_{args.model}_generation_rows.jsonl", generation_rows)
    p828.write_json(out_dir / f"phase834_{args.model}_summary.json", summary)
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
        f"# Phase 834 Blocker-Aware Internal Route-Boundary Predictor ({payload['round']})",
        "",
        "- Source: Phase 833 protocol-structure proxy plus first-step full-vocabulary blocker profile.",
        "- Objective: test whether target-rank / above-target signals can remove qwen3 route interference.",
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
        "First-step blocker awareness is still a proxy. It is useful only if it improves natural_category recovery "
        "without adding natural degradation, and only if the same rule works across qwen3 and DS7B."
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
        path = out_dir / f"phase834_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase834_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase834_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p829.build_parser()
    parser.add_argument(
        "--predictor-modes",
        default=(
            "category_nonresidual_else_count_nonnegative,category_count_rank_improved,"
            "category_count_above_decreased,category_count_rank_le50,category_count_rank_le500,"
            "category_count_rank_le2000,nonresidual_count_rank_le500,count_rank_improved,"
            "count_rank_le500,oracle_route_target_only"
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
