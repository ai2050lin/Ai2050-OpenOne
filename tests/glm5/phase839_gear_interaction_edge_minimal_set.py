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
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase834_blocker_aware_internal_route_boundary_predictor as p834  # noqa: E402
import phase837_global_gear_response_atlas_pilot as p837  # noqa: E402
import phase838_gear_response_decomposition_prediction as p838  # noqa: E402


PHASE = 839
RESULT_ROOT = Path("tests/result/phase839_gear_interaction_edge_minimal_set")
SOURCE_837 = Path("tests/result/phase837_global_gear_response_atlas_pilot/confirm")
SOURCE_838 = Path("tests/result/phase838_gear_response_decomposition_prediction")
TARGET_CLASS = "target_equivalent"


def log(msg: str) -> None:
    p837.log(msg)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    return p838.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def component_label(group: dict[str, Any]) -> str:
    return f"{group.get('case_id')}::{p828.compact_component_label(group)}"


def load_phase837_groups(model_name: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(SOURCE_837 / f"phase837_{model_name}_rows.jsonl")
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("component_label_full") or "")
        group = row.get("component_group")
        if label and isinstance(group, dict) and label not in groups:
            groups[label] = group
    return groups


def load_phase838_components(model_name: str) -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_838 / "phase838_component_vectors.jsonl")
    return [row for row in rows if row.get("model") == model_name]


def select_component_labels(model_name: str, args: argparse.Namespace) -> list[str]:
    groups = load_phase837_groups(model_name)
    comps = load_phase838_components(model_name)
    comps = [row for row in comps if row.get("component_label_full") in groups]
    comps.sort(
        key=lambda row: (
            finite(row.get("train_target_quality_score")),
            finite(row.get("mean_target_quality_score")),
            -finite(row.get("mean_echo_risk_score")),
            -finite(row.get("mean_harm_risk_score")),
            str(row.get("component_label_full")),
        ),
        reverse=True,
    )
    labels: list[str] = []
    seen: set[str] = set()
    for row in comps:
        label = str(row.get("component_label_full"))
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
        if len(labels) >= int(args.max_components):
            break
    return labels


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    summary = read_json(SOURCE_838 / "phase838_summary.json")
    cases: list[str] = []
    if args.case_scope == "holdout":
        for model in summary.get("models", []):
            cases.extend(summary["model_summaries"][model].get("holdout_cases") or [])
        case_ids = []
        seen: set[str] = set()
        for case_id in cases:
            if case_id not in seen:
                seen.add(case_id)
                case_ids.append(case_id)
    else:
        case_ids = [str(case["case_id"]) for case in p837.p816.CASES]
    cmap = p828.p825.case_map()
    out = [cmap[case_id] for case_id in case_ids if case_id in cmap]
    if 0 < int(args.max_cases) < len(out):
        out = out[: int(args.max_cases)]
    return out


def combo_specs(labels: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for label in labels:
        combos.append({"combo_kind": "single", "labels": [label]})
    pair_labels = list(itertools.combinations(labels, 2))
    if int(args.max_pairs) > 0:
        pair_labels = pair_labels[: int(args.max_pairs)]
    for pair in pair_labels:
        combos.append({"combo_kind": "pair", "labels": list(pair)})
    if args.include_sets and len(labels) >= 3:
        max_size = min(int(args.max_set_size), len(labels))
        for size in range(3, max_size + 1):
            for combo in itertools.combinations(labels, size):
                combos.append({"combo_kind": f"set{size}", "labels": list(combo)})
    return combos


def prepare_component_data(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    group: dict[str, Any],
    source_row: dict[str, Any],
    recipient_prompt: str,
    baseline_ids: list[int],
) -> dict[str, Any] | None:
    return p837.component_data_for_case(
        model,
        tokenizer,
        device,
        group,
        source_row,
        recipient_prompt,
        case,
        baseline_ids,
    )


def donor_patch_item(
    model,
    tokenizer,
    device: torch.device,
    case: dict[str, Any],
    donor_variant: str,
    group: dict[str, Any],
    comp_data: dict[str, Any],
) -> dict[str, Any] | None:
    donor_prompt = p828.p825.natural_prompt(case, donor_variant)
    donor_state = p828.p822.capture_component_state(model, tokenizer, device, donor_prompt, int(group["layer_idx"]))
    donor_vec = p828.p823.component_vector(donor_state, comp_data["spec"])
    if donor_vec is None:
        return None
    return {
        "layer_idx": int(group["layer_idx"]),
        "spec": comp_data["spec"],
        "recipient_vec": comp_data["recipient_vec"],
        "donor_vec": donor_vec.float().cpu(),
        "selected_indices": comp_data["selected_indices"],
    }


def combo_label(labels: list[str]) -> str:
    return " + ".join(labels)


def eval_case(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    labels: list[str],
    groups: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lookup = p828.p820.standard_lookup(standards)
    recipient_prompt = p828.p825.natural_prompt(case, args.recipient_prompt)
    recipient_ids = p828.p823.encode_prompt(tokenizer, recipient_prompt)
    baseline_text, baseline_ids = p828.greedy_generate_with_multi_patch(
        model, tokenizer, device, recipient_ids, [], int(args.max_new_tokens), float(args.alpha)
    )
    baseline_boundary = p828.p825.boundary_for(lookup, case["case_id"], baseline_text)
    candidates = p837.gear_candidates(tokenizer, case, args)
    baseline_scored = p837.p816.score_candidates(
        model, tokenizer, device, recipient_ids, candidates, int(args.batch_size), int(args.top_k)
    )
    baseline_span = p837.gear_span_profile(baseline_scored)
    target_id = p837.target_first_id(tokenizer, case)
    baseline_id = int(baseline_ids[0]) if baseline_ids else None
    no_patch_logits = p834.first_step_logits(model, device, recipient_ids, [], float(args.alpha))
    no_patch_rank_profile = p834.rank_profile(no_patch_logits, target_id, baseline_id)
    no_patch_rank_profile["top_token"] = (
        tokenizer.decode([int(no_patch_rank_profile["top_token_id"])])
        if no_patch_rank_profile.get("top_token_id") is not None
        else None
    )

    comp_cache: dict[str, dict[str, Any]] = {}
    for label in labels:
        group = groups.get(label)
        source = source_rows.get(label)
        if not group or not source:
            continue
        comp = prepare_component_data(model, tokenizer, device, case, group, source, recipient_prompt, baseline_ids)
        if comp is not None:
            comp_cache[label] = comp

    rows: list[dict[str, Any]] = []
    combos = combo_specs([label for label in labels if label in comp_cache], args)
    for donor_variant in parse_csv(args.search_donor_prompts):
        donor_cache: dict[str, dict[str, Any] | None] = {}
        for label in comp_cache:
            donor_cache[label] = donor_patch_item(
                model,
                tokenizer,
                device,
                case,
                donor_variant,
                groups[label],
                comp_cache[label],
            )

        for combo in combos:
            patch_items = []
            missing = False
            for label in combo["labels"]:
                item = donor_cache.get(label)
                if item is None:
                    missing = True
                    break
                patch_items.append(item)
            if missing or not patch_items:
                continue

            patched_text, patched_ids = p828.greedy_generate_with_multi_patch(
                model,
                tokenizer,
                device,
                recipient_ids,
                patch_items,
                int(args.max_new_tokens),
                float(args.alpha),
            )
            patched_boundary = p828.p825.boundary_for(lookup, case["case_id"], patched_text)
            patch_logits = p834.first_step_logits(model, device, recipient_ids, patch_items, float(args.alpha))
            rank_features = p834.rank_profile(patch_logits, target_id, baseline_id)
            top_id = rank_features.get("top_token_id")
            rank_features["top_token"] = tokenizer.decode([int(top_id)]) if top_id is not None else None
            base_rank = no_patch_rank_profile.get("target_rank")
            base_above = no_patch_rank_profile.get("above_target_count")
            rank = rank_features.get("target_rank")
            above = rank_features.get("above_target_count")
            rank_features["target_rank_improved"] = bool(rank is not None and base_rank is not None and int(rank) < int(base_rank))
            rank_features["above_target_decreased"] = bool(
                above is not None and base_above is not None and int(above) < int(base_above)
            )
            patched_scored = p837.p835.score_candidates_with_first_logits(
                tokenizer, candidates, baseline_scored, patch_logits, int(args.top_k)
            )
            patched_span = p837.gear_span_profile(patched_scored)
            row: dict[str, Any] = {
                "row_kind": "phase839_gear_interaction_edge_minimal_set",
                "phase": PHASE,
                "model": args.model,
                "round": args.round_name,
                "source_phase": "phase838_top_components",
                "case_id": case["case_id"],
                "object": case["object"],
                "target_answer": case["answer"],
                "donor_variant": donor_variant,
                "recipient_prompt": args.recipient_prompt,
                "combo_kind": combo["combo_kind"],
                "combo_labels": combo["labels"],
                "combo_label": combo_label(combo["labels"]),
                "n_components": len(combo["labels"]),
                "baseline_generated": p828.p825.clean_generated(baseline_text),
                "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
                "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
                "baseline_protocol_valid": bool(baseline_boundary.get("protocol_valid")),
                "patched_generated": p828.p825.clean_generated(patched_text),
                "patched_token_ids": patched_ids,
                "patched_boundary_class": patched_boundary.get("final_boundary_class"),
                "patched_boundary_rank": int(patched_boundary["boundary_rank"]),
                "patched_protocol_valid": bool(patched_boundary.get("protocol_valid")),
                "delta_boundary_rank": int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"]),
                "improved_boundary": int(patched_boundary["boundary_rank"]) > int(baseline_boundary["boundary_rank"]),
                "degraded_boundary": int(patched_boundary["boundary_rank"]) < int(baseline_boundary["boundary_rank"]),
                "target_transition": patched_boundary.get("final_boundary_class") == TARGET_CLASS,
                "baseline_span": baseline_span,
                "baseline_rank_profile": no_patch_rank_profile,
                **rank_features,
                **p837.profile_delta("patch", baseline_span, patched_span),
            }
            row["response_type"] = p837.classify_response(row)
            row["response_vector"] = p838.row_vector(row)
            rows.append(row)
    add_interaction_features(rows, args)
    return rows


def combo_key(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(row.get("case_id")),
        str(row.get("donor_variant")),
        tuple(str(x) for x in row.get("combo_labels") or []),
    )


def add_interaction_features(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lookup = {combo_key(row): row for row in rows}
    quality_threshold = float(args.interaction_quality_threshold)
    for row in rows:
        labels = tuple(str(x) for x in row.get("combo_labels") or [])
        vec = row.get("response_vector") or {}
        q = finite(vec.get("target_quality_score"))
        echo = finite(vec.get("echo_risk_score"))
        harm = finite(vec.get("harm_risk_score"))
        singles = [
            lookup.get((str(row.get("case_id")), str(row.get("donor_variant")), (label,)))
            for label in labels
        ]
        singles = [s for s in singles if s is not None]
        single_q = [finite((s.get("response_vector") or {}).get("target_quality_score")) for s in singles]
        single_echo = [finite((s.get("response_vector") or {}).get("echo_risk_score")) for s in singles]
        single_harm = [finite((s.get("response_vector") or {}).get("harm_risk_score")) for s in singles]
        best_single_q = max(single_q) if single_q else None
        min_single_echo = min(single_echo) if single_echo else None
        min_single_harm = min(single_harm) if single_harm else None
        row["best_single_quality"] = best_single_q
        row["min_single_echo_risk"] = min_single_echo
        row["min_single_harm_risk"] = min_single_harm
        row["interaction_quality_gain"] = None if best_single_q is None else q - best_single_q
        row["interaction_echo_gain"] = None if min_single_echo is None else min_single_echo - echo
        row["interaction_harm_gain"] = None if min_single_harm is None else min_single_harm - harm
        row["positive_interaction_edge"] = bool(
            len(labels) > 1
            and row["interaction_quality_gain"] is not None
            and finite(row["interaction_quality_gain"]) > quality_threshold
            and (min_single_echo is None or echo <= min_single_echo + float(args.echo_tolerance))
            and (min_single_harm is None or harm <= min_single_harm + float(args.harm_tolerance))
        )

        proper_subset_quality = []
        if len(labels) > 1:
            for size in range(1, len(labels)):
                for subset in itertools.combinations(labels, size):
                    item = lookup.get((str(row.get("case_id")), str(row.get("donor_variant")), tuple(subset)))
                    if item:
                        proper_subset_quality.append(finite((item.get("response_vector") or {}).get("target_quality_score")))
        best_subset_q = max(proper_subset_quality) if proper_subset_quality else best_single_q
        row["best_proper_subset_quality"] = best_subset_q
        row["minimal_sufficient_candidate"] = bool(
            len(labels) > 1
            and bool(row.get("target_transition"))
            and finite(vec.get("echo_risk_score")) <= float(args.max_minimal_echo_risk)
            and finite(vec.get("harm_risk_score")) <= float(args.max_minimal_harm_risk)
            and best_subset_q is not None
            and q > finite(best_subset_q) + quality_threshold
        )


def summarize_rows(rows: list[dict[str, Any]], labels: list[str], cases: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None) -> dict[str, Any]:
    by_kind = defaultdict(list)
    by_combo = defaultdict(list)
    for row in rows:
        by_kind[str(row.get("combo_kind"))].append(row)
        by_combo[str(row.get("combo_label"))].append(row)

    combo_records = []
    for label, vals in by_combo.items():
        combo_records.append(
            {
                "combo_label": label,
                "combo_kind": vals[0].get("combo_kind"),
                "n_components": vals[0].get("n_components"),
                "n_rows": len(vals),
                "target_rows": sum(1 for row in vals if row.get("target_transition")),
                "object_echo_rows": sum(1 for row in vals if row.get("patched_boundary_class") == "object_echo"),
                "format_echo_rows": sum(1 for row in vals if row.get("patched_boundary_class") == "format_echo"),
                "degraded_rows": sum(1 for row in vals if row.get("degraded_boundary")),
                "positive_interaction_rows": sum(1 for row in vals if row.get("positive_interaction_edge")),
                "minimal_sufficient_rows": sum(1 for row in vals if row.get("minimal_sufficient_candidate")),
                "mean_quality": avg_vec(vals, "target_quality_score"),
                "mean_echo_risk": avg_vec(vals, "echo_risk_score"),
                "mean_harm_risk": avg_vec(vals, "harm_risk_score"),
                "mean_interaction_gain": avg_field(vals, "interaction_quality_gain"),
            }
        )
    combo_records.sort(
        key=lambda item: (
            int(item["positive_interaction_rows"]),
            int(item["minimal_sufficient_rows"]),
            finite(item["mean_interaction_gain"]),
            finite(item["mean_quality"]),
        ),
        reverse=True,
    )
    top_rows = sorted(
        [row for row in rows if row.get("positive_interaction_edge")],
        key=lambda row: finite(row.get("interaction_quality_gain")),
        reverse=True,
    )[:40]
    return {
        "phase": PHASE,
        "title": "Gear Interaction Edge and Minimal Sufficient Set Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_components": len(labels),
        "n_cases": len(cases),
        "component_labels": labels,
        "case_ids": [case["case_id"] for case in cases],
        "donor_variants": parse_csv(args.search_donor_prompts),
        "combo_kind_summary": {kind: compact_rows(vals) for kind, vals in sorted(by_kind.items())},
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "positive_interaction_rows": sum(1 for row in rows if row.get("positive_interaction_edge")),
        "minimal_sufficient_rows": sum(1 for row in rows if row.get("minimal_sufficient_candidate")),
        "mean_quality": avg_vec(rows, "target_quality_score"),
        "mean_echo_risk": avg_vec(rows, "echo_risk_score"),
        "mean_harm_risk": avg_vec(rows, "harm_risk_score"),
        "combo_records": combo_records[:120],
        "top_interaction_rows": [
            {
                "case_id": row.get("case_id"),
                "donor_variant": row.get("donor_variant"),
                "combo_kind": row.get("combo_kind"),
                "combo_label": row.get("combo_label"),
                "patched_boundary_class": row.get("patched_boundary_class"),
                "patched_generated": row.get("patched_generated"),
                "target_quality_score": finite((row.get("response_vector") or {}).get("target_quality_score")),
                "echo_risk_score": finite((row.get("response_vector") or {}).get("echo_risk_score")),
                "harm_risk_score": finite((row.get("response_vector") or {}).get("harm_risk_score")),
                "interaction_quality_gain": finite(row.get("interaction_quality_gain")),
                "interaction_echo_gain": finite(row.get("interaction_echo_gain")),
                "minimal_sufficient_candidate": bool(row.get("minimal_sufficient_candidate")),
            }
            for row in top_rows
        ],
        "boundary": "This phase tests interaction edges among Phase 838 selected gear components. It is still patch intervention evidence, not natural mechanism closure.",
    }


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def avg_vec(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [finite((row.get("response_vector") or {}).get(key)) for row in rows]
    return avg(vals)


def avg_field(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [finite(row.get(key)) for row in rows if row.get(key) is not None]
    return avg(vals)


def compact_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "format_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "format_echo"),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "positive_interaction_rows": sum(1 for row in rows if row.get("positive_interaction_edge")),
        "minimal_sufficient_rows": sum(1 for row in rows if row.get("minimal_sufficient_candidate")),
        "mean_quality": avg_vec(rows, "target_quality_score"),
        "mean_echo_risk": avg_vec(rows, "echo_risk_score"),
        "mean_harm_risk": avg_vec(rows, "harm_risk_score"),
        "classes": dict(Counter(str(row.get("patched_boundary_class")) for row in rows)),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = select_component_labels(args.model, args)
    groups = load_phase837_groups(args.model)
    cases = selected_cases(args)
    source_rows = {}
    for label in labels:
        group = groups.get(label)
        if not group:
            continue
        source = p837.source_row_for_group(args.model, group, args)
        if source is not None:
            source_rows[label] = source
    log(
        f"{args.model}/{args.round_name}: labels={len(labels)} source_rows={len(source_rows)} "
        f"cases={len(cases)} donors={parse_csv(args.search_donor_prompts)}"
    )
    if args.dry_run:
        print(json.dumps({"model": args.model, "labels": labels, "cases": [case["case_id"] for case in cases]}, ensure_ascii=False, indent=2))
        return {"labels": labels, "cases": cases}
    if not labels or not source_rows or not cases:
        summary = summarize_rows([], labels, cases, args, None)
        summary["skipped_model_load"] = True
        summary["skip_reason"] = "missing selected labels/source rows/cases"
        p828.write_jsonl(out_dir / f"phase839_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase839_{args.model}_summary.json", summary)
        return summary

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    standards = p828.p820.standard_rows()
    rows: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, 1):
            rows.extend(eval_case(model, tokenizer, device, standards, case, labels, groups, source_rows, args))
            if idx % int(args.log_every) == 0 or idx == len(cases):
                log(f"{args.model}: evaluated cases {idx}/{len(cases)} rows={len(rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, labels, cases, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase839_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase839_{args.model}_summary.json", summary)
    printable = {
        "model": args.model,
        "round": args.round_name,
        "rows": summary["n_rows"],
        "target_rows": summary["target_rows"],
        "object_echo_rows": summary["object_echo_rows"],
        "format_echo_rows": summary["format_echo_rows"],
        "degraded_rows": summary["degraded_rows"],
        "positive_interaction_rows": summary["positive_interaction_rows"],
        "minimal_sufficient_rows": summary["minimal_sufficient_rows"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 839 Gear Interaction Edge and Minimal Set ({payload['round']})",
        "",
        "- Source: Phase 838 top gear components, tested on held-out cases.",
        "- Boundary: patch-intervention interaction test; not natural mechanism proof.",
        "",
        "## Model Summary",
        "",
        "| model | rows | components | cases | target | object_echo | format_echo | degraded | positive interaction | minimal candidates | mean quality | mean echo risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {data.get('n_rows', 0)} | {data.get('n_components', 0)} | {data.get('n_cases', 0)} | "
            f"{data.get('target_rows', 0)} | {data.get('object_echo_rows', 0)} | {data.get('format_echo_rows', 0)} | "
            f"{data.get('degraded_rows', 0)} | {data.get('positive_interaction_rows', 0)} | {data.get('minimal_sufficient_rows', 0)} | "
            f"{fmt(data.get('mean_quality'))} | {fmt(data.get('mean_echo_risk'))} |"
        )
    lines += ["", "## Combo Kind Summary", ""]
    lines += ["| model | combo kind | n | target | object_echo | format_echo | positive interaction | minimal | mean quality | mean echo risk | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for kind, row in sorted((data.get("combo_kind_summary") or {}).items()):
            lines.append(
                f"| {model_name} | `{kind}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('object_echo_rows', 0)} | {row.get('format_echo_rows', 0)} | "
                f"{row.get('positive_interaction_rows', 0)} | {row.get('minimal_sufficient_rows', 0)} | "
                f"{fmt(row.get('mean_quality'))} | {fmt(row.get('mean_echo_risk'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Top Interaction Rows", ""]
    lines += ["| model | case | donor | kind | combo | class | output | quality | gain | echo risk | echo gain | minimal |"]
    lines += ["|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("top_interaction_rows") or []:
            output = str(row.get("patched_generated") or "").replace("|", "/")[:60]
            lines.append(
                f"| {model_name} | `{row.get('case_id')}` | `{row.get('donor_variant')}` | `{row.get('combo_kind')}` | "
                f"`{row.get('combo_label')}` | `{row.get('patched_boundary_class')}` | {output} | "
                f"{fmt(row.get('target_quality_score'))} | {fmt(row.get('interaction_quality_gain'))} | "
                f"{fmt(row.get('echo_risk_score'))} | {fmt(row.get('interaction_echo_gain'))} | "
                f"{int(bool(row.get('minimal_sufficient_candidate')))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


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
        path = out_dir / f"phase839_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase839_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase839_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p828.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--case-scope", choices=["holdout", "all"], default="holdout")
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--include-sets", action="store_true")
    parser.add_argument("--max-set-size", type=int, default=3)
    parser.add_argument("--search-donor-prompts", default="exact_choices,natural_category")
    parser.add_argument("--budgets", default="16,32")
    parser.add_argument("--component-kinds", default="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group")
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--max-span-candidates", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--interaction-quality-threshold", type=float, default=0.05)
    parser.add_argument("--echo-tolerance", type=float, default=0.05)
    parser.add_argument("--harm-tolerance", type=float, default=0.05)
    parser.add_argument("--max-minimal-echo-risk", type=float, default=0.25)
    parser.add_argument("--max-minimal-harm-risk", type=float, default=0.05)
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
