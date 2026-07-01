#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase823_beneficial_harmful_boundary_subspace_split as p823  # noqa: E402
import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase829_non_interference_constrained_component_composition as p829  # noqa: E402


PHASE = 831
RESULT_ROOT = Path("tests/result/phase831_internal_gate_predictor_search")


def log(msg: str) -> None:
    p829.log(msg)


def predictor_modes(args: argparse.Namespace) -> list[str]:
    return [x.strip() for x in str(args.predictor_modes).split(",") if x.strip()]


def prepare_component_data(
    model,
    tokenizer,
    device: torch.device,
    recipient_prompt: str,
    baseline_ids: list[int],
    case: dict[str, Any],
    pair: tuple[dict[str, Any], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    comp_data = []
    for group in pair:
        source_row = p828.group_from_phase822(args.model, group, args)
        if source_row is None:
            return []
        spec = p828.spec_from_source_row(group, source_row)
        recipient_state = p828.p822.capture_component_state(model, tokenizer, device, recipient_prompt, int(group["layer_idx"]))
        recipient_vec = p823.component_vector(recipient_state, spec)
        effective_dir, readout_meta = p823.effective_readout_direction(
            model, tokenizer, case, baseline_ids, int(group["layer_idx"]), spec
        )
        if recipient_vec is None or effective_dir is None:
            return []
        comp_data.append(
            {
                "group": group,
                "source_row": source_row,
                "spec": spec,
                "recipient_vec": recipient_vec.float().cpu(),
                "effective_dir": effective_dir.float().cpu(),
                "readout_meta": readout_meta,
                "selected_indices": [int(x) for x in group["selected_indices"]],
            }
        )
    return comp_data


def selected_feature_scores(
    recipient_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    effective_dir: torch.Tensor,
    selected_indices: list[int],
) -> dict[str, Any]:
    n = min(int(recipient_vec.numel()), int(donor_vec.numel()), int(effective_dir.numel()))
    selected = [int(i) for i in selected_indices if 0 <= int(i) < n]
    if not selected:
        return {
            "n_selected_for_gate": 0,
            "selected_signed_sum": 0.0,
            "selected_signed_mean": 0.0,
            "selected_positive_sum": 0.0,
            "selected_negative_abs_sum": 0.0,
            "selected_positive_count": 0,
            "selected_negative_count": 0,
            "top_abs_signed_score": 0.0,
            "delta_norm": 0.0,
        }
    idx = torch.tensor(selected, dtype=torch.long)
    delta = donor_vec[:n] - recipient_vec[:n]
    signed = delta * effective_dir[:n]
    vals = signed[idx]
    pos = torch.clamp(vals, min=0.0)
    neg = torch.clamp(-vals, min=0.0)
    top = vals[torch.argmax(torch.abs(vals))] if vals.numel() else torch.tensor(0.0)
    return {
        "n_selected_for_gate": len(selected),
        "selected_signed_sum": float(vals.sum().item()),
        "selected_signed_mean": float(vals.mean().item()) if vals.numel() else 0.0,
        "selected_positive_sum": float(pos.sum().item()),
        "selected_negative_abs_sum": float(neg.sum().item()),
        "selected_positive_count": int((vals > 0).sum().item()),
        "selected_negative_count": int((vals < 0).sum().item()),
        "top_abs_signed_score": float(top.item()),
        "delta_norm": float(delta[idx].norm().item()),
    }


def predict_gate(features: dict[str, Any], mode: str) -> bool:
    signed_sum = float(features.get("selected_signed_sum") or 0.0)
    pos_count = int(features.get("selected_positive_count") or 0)
    neg_count = int(features.get("selected_negative_count") or 0)
    top_abs = float(features.get("top_abs_signed_score") or 0.0)
    if mode == "signed_sum_positive":
        return signed_sum > 0
    if mode == "positive_count_majority":
        return pos_count > neg_count
    if mode == "top_abs_positive":
        return top_abs > 0
    if mode == "sum_and_count_positive":
        return signed_sum > 0 and pos_count >= neg_count
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
    recipient_ids = p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p823.greedy_generate_with_subspace_patch(
        model, tokenizer, device, recipient_ids, args.max_new_tokens
    )
    baseline_boundary = p828.p825.boundary_for(lookup, case["case_id"], baseline_text)
    comp_data = prepare_component_data(model, tokenizer, device, recipient_prompt, baseline_ids, case, pair, args)
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
            donor_state = p828.p822.capture_component_state(model, tokenizer, device, donor_prompt, int(group["layer_idx"]))
            donor_vec = p823.component_vector(donor_state, item["spec"])
            if donor_vec is None:
                continue
            donor_vec = donor_vec.float().cpu()
            features = selected_feature_scores(
                item["recipient_vec"], donor_vec, item["effective_dir"], item["selected_indices"]
            )
            oracle_gate = str((group.get("single_donor_classes") or {}).get(donor_variant)) == "target_equivalent"
            label = p828.compact_component_label(group)
            for mode in modes:
                pred = predict_gate(features, mode)
                decision_rows.append(
                    {
                        "row_kind": "phase831_internal_gate_predictor_decision",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "case_id": case["case_id"],
                        "object": case["object"],
                        "target_answer": case["answer"],
                        "pair_label": f"{p828.compact_component_label(a)} + {p828.compact_component_label(b)}",
                        "component_label_full": label,
                        "component_group": group,
                        "donor_variant": donor_variant,
                        "predictor_mode": mode,
                        "oracle_target_only_gate": bool(oracle_gate),
                        "predicted_gate": bool(pred),
                        "decision_correct_vs_oracle": bool(pred) == bool(oracle_gate),
                        "single_donor_class": (group.get("single_donor_classes") or {}).get(donor_variant),
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
                    "row_kind": "phase831_internal_gate_predictor_generation",
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
                    "validation_mode": "internal_readout_alignment_gate_prediction",
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


def finite(value: Any, default: float = 0.0) -> float:
    return p829.finite(value, default)


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return p829.compact(rows)


def decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    tp = sum(1 for row in rows if row.get("predicted_gate") and row.get("oracle_target_only_gate"))
    tn = sum(1 for row in rows if not row.get("predicted_gate") and not row.get("oracle_target_only_gate"))
    fp = sum(1 for row in rows if row.get("predicted_gate") and not row.get("oracle_target_only_gate"))
    fn = sum(1 for row in rows if not row.get("predicted_gate") and row.get("oracle_target_only_gate"))
    return {
        "n": n,
        "accuracy_vs_target_only": (tp + tn) / n if n else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "predicted_active": tp + fp,
        "oracle_active": tp + fn,
    }


def generation_summary(rows: list[dict[str, Any]], min_natural_targets: int) -> dict[str, Any]:
    by_pair = defaultdict(list)
    by_donor = defaultdict(list)
    for row in rows:
        by_pair[str(row.get("pair_label"))].append(row)
        by_donor[str(row.get("donor_variant"))].append(row)
    pair_exact_multi = 0
    pair_records = []
    new_exact_multi = 0
    preserve_exact_multi = 0
    for label, vals in by_pair.items():
        counts = p828.pair_counts(vals, min_natural_targets)
        a = vals[0]["component_a"]
        b = vals[0]["component_b"]
        single_had = bool(a["single_exact_plus_multi"] or b["single_exact_plus_multi"])
        if counts["exact_plus_multi"]:
            pair_exact_multi += 1
            if single_had:
                preserve_exact_multi += 1
            else:
                new_exact_multi += 1
        pair_records.append(
            {
                "pair_label": label,
                "case_id": vals[0].get("case_id"),
                **counts,
                "single_had_exact_multi": single_had,
            }
        )
    exact_rows = [r for r in rows if r.get("donor_variant") == "exact_choices"]
    natural_rows = [r for r in rows if r.get("donor_variant") != "exact_choices"]
    natural_category = [r for r in rows if r.get("donor_variant") == "natural_category"]
    active_counts = Counter(str(row.get("n_active_components")) for row in rows)
    return {
        "n_rows": len(rows),
        "exact_target_rows": sum(1 for row in exact_rows if row.get("target_transition")),
        "natural_target_rows": sum(1 for row in natural_rows if row.get("target_transition")),
        "natural_degraded_rows": sum(1 for row in natural_rows if row.get("degraded_boundary")),
        "natural_category_target_rows": sum(1 for row in natural_category if row.get("target_transition")),
        "natural_category_degraded_rows": sum(1 for row in natural_category if row.get("degraded_boundary")),
        "pair_exact_plus_multi": pair_exact_multi,
        "composition_new_exact_multi": new_exact_multi,
        "composition_preserve_exact_multi": preserve_exact_multi,
        "active_component_count_distribution": dict(active_counts),
        "donor_summary": {donor: compact(vals) for donor, vals in sorted(by_donor.items())},
        "pair_records": pair_records[:120],
    }


def summarize_rows(
    decision_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
    attn_impl: str | None = None,
) -> dict[str, Any]:
    decisions_by_mode = defaultdict(list)
    generations_by_mode = defaultdict(list)
    for row in decision_rows:
        decisions_by_mode[str(row.get("predictor_mode"))].append(row)
    for row in generation_rows:
        generations_by_mode[str(row.get("predictor_mode"))].append(row)
    mode_summaries = {}
    for mode in predictor_modes(args):
        mode_summaries[mode] = {
            "decision": decision_summary(decisions_by_mode.get(mode, [])),
            "generation": generation_summary(generations_by_mode.get(mode, []), int(args.min_natural_targets)),
        }
    return {
        "phase": PHASE,
        "title": "Internal Gate Predictor Search",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_component_groups": len(groups),
        "n_pairs": len(pairs),
        "n_decision_rows": len(decision_rows),
        "n_generation_rows": len(generation_rows),
        "diagnostics": diagnostics,
        "predictor_modes": predictor_modes(args),
        "mode_summaries": mode_summaries,
        "boundary": "This phase asks whether a simple internal readout-alignment signal can predict the donor-specific gates used manually in Phase 830.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p828.p820.standard_rows()
    groups = p828.load_component_groups(args.model, args)
    pairs, diagnostics = p829.build_constrained_pairs(groups, args)
    cmap = p828.p825.case_map()
    log(f"{args.model}/{args.round_name}: groups={len(groups)} pairs={len(pairs)} predictors={predictor_modes(args)}")
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
        summary["skip_reason"] = "no eligible same-case pairs for gate prediction"
        p828.write_jsonl(out_dir / f"phase831_{args.model}_decision_rows.jsonl", [])
        p828.write_jsonl(out_dir / f"phase831_{args.model}_generation_rows.jsonl", [])
        p828.write_json(out_dir / f"phase831_{args.model}_summary.json", summary)
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
    p828.write_jsonl(out_dir / f"phase831_{args.model}_decision_rows.jsonl", decision_rows)
    p828.write_jsonl(out_dir / f"phase831_{args.model}_generation_rows.jsonl", generation_rows)
    p828.write_json(out_dir / f"phase831_{args.model}_summary.json", summary)
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
                "active": data["generation"].get("active_component_count_distribution"),
            }
            for mode, data in summary["mode_summaries"].items()
        },
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 831 Internal Gate Predictor Search ({payload['round']})",
        "",
        "- Source: Phase 829 pairs and Phase 830 target-only gates.",
        "- Objective: test whether simple internal readout-alignment signals can predict useful donor gates.",
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
    lines.append("This is not a learned classifier; it is a basic readout-alignment probe. It is useful only if the same signal both approximates the Phase 830 gate and preserves generation quality.")
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
        path = out_dir / f"phase831_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase831_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase831_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p829.build_parser()
    parser.add_argument("--predictor-modes", default="signed_sum_positive,positive_count_majority,top_abs_positive")
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
