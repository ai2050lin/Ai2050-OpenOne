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

import phase816_multi_token_answer_span_rollout_closure as p816  # noqa: E402
import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402
import phase829_non_interference_constrained_component_composition as p829  # noqa: E402
import phase834_blocker_aware_internal_route_boundary_predictor as p834  # noqa: E402
import phase835_span_protocol_blocker_profile as p835  # noqa: E402


PHASE = 837
RESULT_ROOT = Path("tests/result/phase837_global_gear_response_atlas_pilot")
TARGET_CLASS = "target_equivalent"


def log(msg: str) -> None:
    p829.log(msg)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if val == val and val not in {float("inf"), float("-inf")} else default


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = list(p816.CASES)
    if int(args.max_cases) > 0 and int(args.max_cases) < len(cases):
        idxs = p816.select_evenly(len(cases), int(args.max_cases))
        cases = [cases[i] for i in idxs]
    return cases


def group_key(group: dict[str, Any]) -> str:
    return f"{group.get('case_id')}::{p828.compact_component_label(group)}"


def gear_candidates(tokenizer, case: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = p816.span_candidates(tokenizer, case, args)
    seen = {(row["candidate_class"], tuple(int(x) for x in row["token_ids"])) for row in out}
    for text in p816.phrase_variants(str(case["object"])):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        key = ("object_echo", tuple(int(x) for x in ids))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "candidate_class": "object_echo",
                "phrase": str(case["object"]),
                "variant_text": text,
                "token_ids": [int(x) for x in ids],
                "span_len": len(ids),
                "normalized_text": p816.norm_text(text),
            }
        )
    return out[: int(args.max_span_candidates)]


def best_score(scored: list[dict[str, Any]], cls: str) -> float | None:
    row = p816.best_by_class(scored, cls)
    return None if not row else p816.finite(row.get("score_mean_logprob"))


def best_rank(scored: list[dict[str, Any]], cls: str) -> int | None:
    row = p816.best_by_class(scored, cls)
    if not row:
        return None
    key = tuple(int(x) for x in row.get("token_ids") or [])
    for idx, item in enumerate(scored, 1):
        if tuple(int(x) for x in item.get("token_ids") or []) == key:
            return idx
    return None


def gear_span_profile(scored: list[dict[str, Any]]) -> dict[str, Any]:
    base = p835.span_profile(scored)
    target = best_score(scored, "target")
    echo = best_score(scored, "object_echo")
    contrast = best_score(scored, "contrast")
    generic = best_score(scored, "generic_blocker")
    base.update(
        {
            "best_object_echo": p835.compact_span(p816.best_by_class(scored, "object_echo")),
            "span_target_margin_vs_echo": None if target is None or echo is None else target - echo,
            "span_echo_cleared": bool(target is not None and echo is not None and target > echo),
            "span_echo_rank": best_rank(scored, "object_echo"),
            "span_target_score": target,
            "span_contrast_score": contrast,
            "span_generic_score": generic,
            "span_echo_score": echo,
            "best_candidate_class": scored[0].get("candidate_class") if scored else None,
            "best_candidate": p835.compact_span(scored[0]) if scored else None,
        }
    )
    return base


def profile_delta(prefix: str, baseline: dict[str, Any], patched: dict[str, Any]) -> dict[str, Any]:
    out = p835.span_delta_features(prefix, baseline, patched)
    b = baseline.get("span_target_margin_vs_echo")
    p = patched.get("span_target_margin_vs_echo")
    out[f"{prefix}_span_target_margin_vs_echo"] = p
    out[f"{prefix}_span_target_margin_vs_echo_baseline"] = b
    out[f"{prefix}_span_target_margin_vs_echo_improved"] = bool(p is not None and b is not None and float(p) > float(b))
    out[f"{prefix}_span_echo_cleared"] = bool(patched.get("span_echo_cleared"))
    out[f"{prefix}_span_echo_rank"] = patched.get("span_echo_rank")
    out[f"{prefix}_best_object_echo"] = patched.get("best_object_echo")
    out[f"{prefix}_best_candidate_class"] = patched.get("best_candidate_class")
    out[f"{prefix}_best_candidate"] = patched.get("best_candidate")
    return out


def target_first_id(tokenizer, case: dict[str, Any]) -> int | None:
    ids = tokenizer.encode(str(case["answer"]), add_special_tokens=False)
    return int(ids[0]) if ids else None


def source_row_for_group(model_name: str, group: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    return p828.group_from_phase822(model_name, group, args)


def component_data_for_case(
    model,
    tokenizer,
    device: torch.device,
    group: dict[str, Any],
    source_row: dict[str, Any],
    recipient_prompt: str,
    case: dict[str, Any],
    baseline_ids: list[int],
) -> dict[str, Any] | None:
    spec = p828.spec_from_source_row(group, source_row)
    recipient_state = p828.p822.capture_component_state(model, tokenizer, device, recipient_prompt, int(group["layer_idx"]))
    recipient_vec = p828.p823.component_vector(recipient_state, spec)
    if recipient_vec is None:
        return None
    effective_dir, readout_meta = p828.p823.effective_readout_direction(
        model, tokenizer, case, baseline_ids, int(group["layer_idx"]), spec
    )
    return {
        "spec": spec,
        "recipient_vec": recipient_vec.float().cpu(),
        "effective_dir": None if effective_dir is None else effective_dir.float().cpu(),
        "readout_meta": readout_meta,
        "selected_indices": [int(x) for x in group.get("selected_indices") or []],
    }


def signed_features(data: dict[str, Any], donor_vec: torch.Tensor) -> dict[str, Any]:
    if data.get("effective_dir") is None:
        return {
            "selected_signed_sum": None,
            "selected_positive_count": None,
            "selected_negative_count": None,
            "delta_norm": None,
        }
    return p835.p833.internal_features(data["recipient_vec"], donor_vec, data["effective_dir"], data["selected_indices"])


def classify_response(row: dict[str, Any]) -> str:
    if row.get("target_transition"):
        return "target_writer_candidate"
    if row.get("degraded_boundary"):
        return "harmful_mixer"
    if row.get("patched_boundary_class") == "object_echo" or row.get("patch_span_target_margin_vs_echo_improved") is False:
        if row.get("patch_span_echo_cleared") is False:
            return "echo_amplifier_or_unsuppressed"
    if row.get("patch_span_contrast_cleared") and not row.get("target_transition"):
        return "contrast_suppressor_candidate"
    if row.get("patch_span_target_margin_vs_echo_improved") and row.get("patch_span_echo_cleared"):
        return "echo_suppressor_candidate"
    if row.get("target_rank_improved") or row.get("above_target_decreased"):
        return "first_token_blocker_reducer"
    if row.get("improved_boundary"):
        return "boundary_improver_non_target"
    return "neutral_or_unresolved"


def eval_case(
    model,
    tokenizer,
    device: torch.device,
    standards: list[dict[str, Any]],
    case: dict[str, Any],
    groups: list[dict[str, Any]],
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
    candidates = gear_candidates(tokenizer, case, args)
    baseline_scored = p816.score_candidates(
        model, tokenizer, device, recipient_ids, candidates, int(args.batch_size), int(args.top_k)
    )
    baseline_span = gear_span_profile(baseline_scored)
    target_id = target_first_id(tokenizer, case)
    baseline_id = int(baseline_ids[0]) if baseline_ids else None
    no_patch_logits = p834.first_step_logits(model, device, recipient_ids, [], float(args.alpha))
    no_patch_rank_profile = p834.rank_profile(no_patch_logits, target_id, baseline_id)
    no_patch_rank_profile["top_token"] = tokenizer.decode([int(no_patch_rank_profile["top_token_id"])]) if no_patch_rank_profile.get("top_token_id") is not None else None

    component_cache: dict[str, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    for group in groups:
        label = group_key(group)
        source_row = source_rows.get(label)
        if source_row is None:
            continue
        if label not in component_cache:
            component_cache[label] = component_data_for_case(
                model, tokenizer, device, group, source_row, recipient_prompt, case, baseline_ids
            )
        comp = component_cache[label]
        if comp is None:
            continue
        for donor_variant in parse_csv(args.search_donor_prompts):
            donor_prompt = p828.p825.natural_prompt(case, donor_variant)
            donor_state = p828.p822.capture_component_state(model, tokenizer, device, donor_prompt, int(group["layer_idx"]))
            donor_vec = p828.p823.component_vector(donor_state, comp["spec"])
            if donor_vec is None:
                continue
            donor_vec = donor_vec.float().cpu()
            patch_item = {
                "layer_idx": int(group["layer_idx"]),
                "spec": comp["spec"],
                "recipient_vec": comp["recipient_vec"],
                "donor_vec": donor_vec,
                "selected_indices": comp["selected_indices"],
            }
            patched_text, patched_ids = p828.greedy_generate_with_multi_patch(
                model,
                tokenizer,
                device,
                recipient_ids,
                [patch_item],
                int(args.max_new_tokens),
                float(args.alpha),
            )
            patched_boundary = p828.p825.boundary_for(lookup, case["case_id"], patched_text)
            patch_logits = p834.first_step_logits(model, device, recipient_ids, [patch_item], float(args.alpha))
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
            patched_scored = p835.score_candidates_with_first_logits(
                tokenizer, candidates, baseline_scored, patch_logits, int(args.top_k)
            )
            patched_span = gear_span_profile(patched_scored)
            row: dict[str, Any] = {
                "row_kind": "phase837_global_gear_response_fingerprint",
                "phase": PHASE,
                "model": args.model,
                "round": args.round_name,
                "source_round": args.source_round,
                "case_id": case["case_id"],
                "object": case["object"],
                "target_answer": case["answer"],
                "component_source_case_id": group.get("case_id"),
                "component_label_full": label,
                "component_group": group,
                "component_kind": group.get("component_kind"),
                "donor_variant": donor_variant,
                "recipient_prompt": args.recipient_prompt,
                "baseline_generated": p828.p825.clean_generated(baseline_text),
                "baseline_boundary_class": baseline_boundary.get("final_boundary_class"),
                "baseline_boundary_rank": int(baseline_boundary["boundary_rank"]),
                "baseline_protocol_valid": bool(baseline_boundary.get("protocol_valid")),
                "patched_generated": p828.p825.clean_generated(patched_text),
                "patched_boundary_class": patched_boundary.get("final_boundary_class"),
                "patched_boundary_rank": int(patched_boundary["boundary_rank"]),
                "patched_protocol_valid": bool(patched_boundary.get("protocol_valid")),
                "delta_boundary_rank": int(patched_boundary["boundary_rank"]) - int(baseline_boundary["boundary_rank"]),
                "improved_boundary": int(patched_boundary["boundary_rank"]) > int(baseline_boundary["boundary_rank"]),
                "degraded_boundary": int(patched_boundary["boundary_rank"]) < int(baseline_boundary["boundary_rank"]),
                "target_transition": patched_boundary.get("final_boundary_class") == TARGET_CLASS,
                "baseline_span": baseline_span,
                "baseline_rank_profile": no_patch_rank_profile,
                "readout_meta": comp.get("readout_meta"),
                **rank_features,
                **signed_features(comp, donor_vec),
                **profile_delta("patch", baseline_span, patched_span),
            }
            row["response_type"] = classify_response(row)
            row["fingerprint"] = {
                "delta_boundary_rank": row["delta_boundary_rank"],
                "target_transition": row["target_transition"],
                "degraded_boundary": row["degraded_boundary"],
                "target_rank_improved": row.get("target_rank_improved"),
                "above_target_decreased": row.get("above_target_decreased"),
                "span_margin_improved": row.get("patch_span_target_margin_vs_non_target_improved"),
                "contrast_cleared": row.get("patch_span_contrast_cleared"),
                "generic_cleared": row.get("patch_span_generic_cleared"),
                "echo_cleared": row.get("patch_span_echo_cleared"),
                "echo_margin_improved": row.get("patch_span_target_margin_vs_echo_improved"),
                "patched_boundary_class": row.get("patched_boundary_class"),
                "best_candidate_class": row.get("patch_best_candidate_class"),
            }
            rows.append(row)
    return rows


def compact_top(counter: Counter, n: int = 12) -> dict[str, int]:
    return dict(counter.most_common(n))


def summarize_rows(rows: list[dict[str, Any]], groups: list[dict[str, Any]], cases: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str | None) -> dict[str, Any]:
    by_component = defaultdict(list)
    for row in rows:
        by_component[str(row.get("component_label_full"))].append(row)
    reuse = []
    for label, vals in by_component.items():
        target_cases = sorted({str(row.get("case_id")) for row in vals if row.get("target_transition")})
        contrast_cases = sorted({str(row.get("case_id")) for row in vals if row.get("patch_span_contrast_cleared")})
        echo_clear_cases = sorted({str(row.get("case_id")) for row in vals if row.get("patch_span_echo_cleared")})
        if target_cases or len(contrast_cases) > 1 or len(echo_clear_cases) > 1:
            reuse.append(
                {
                    "component_label_full": label,
                    "target_case_count": len(target_cases),
                    "target_cases": target_cases[:12],
                    "contrast_cleared_case_count": len(contrast_cases),
                    "echo_cleared_case_count": len(echo_clear_cases),
                    "response_types": compact_top(Counter(str(row.get("response_type")) for row in vals), 8),
                }
            )
    reuse.sort(
        key=lambda x: (
            int(x["target_case_count"]),
            int(x["contrast_cleared_case_count"]),
            int(x["echo_cleared_case_count"]),
            str(x["component_label_full"]),
        ),
        reverse=True,
    )
    return {
        "phase": PHASE,
        "title": "Global Gear Response Atlas Pilot",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_component_groups": len(groups),
        "n_cases": len(cases),
        "donor_variants": parse_csv(args.search_donor_prompts),
        "component_kind_counts": compact_top(Counter(str(group.get("component_kind")) for group in groups), 20),
        "target_transition_rows": sum(1 for row in rows if row.get("target_transition")),
        "improved_rows": sum(1 for row in rows if row.get("improved_boundary")),
        "degraded_rows": sum(1 for row in rows if row.get("degraded_boundary")),
        "object_echo_rows": sum(1 for row in rows if row.get("patched_boundary_class") == "object_echo"),
        "contrast_cleared_rows": sum(1 for row in rows if row.get("patch_span_contrast_cleared")),
        "echo_cleared_rows": sum(1 for row in rows if row.get("patch_span_echo_cleared")),
        "target_rank_improved_rows": sum(1 for row in rows if row.get("target_rank_improved")),
        "response_type_counts": compact_top(Counter(str(row.get("response_type")) for row in rows), 20),
        "patched_boundary_classes": compact_top(Counter(str(row.get("patched_boundary_class")) for row in rows), 20),
        "donor_response_counts": {
            donor: compact_top(Counter(str(row.get("response_type")) for row in rows if row.get("donor_variant") == donor), 12)
            for donor in parse_csv(args.search_donor_prompts)
        },
        "reuse_candidates": reuse[:80],
        "boundary": (
            "This pilot measures standardized component response fingerprints across cases and donors. "
            "It is not yet a global gear decomposition or a natural mechanism proof."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p828.p820.standard_rows()
    groups = p828.load_component_groups(args.model, args)
    cases = select_cases(args)
    source_rows = {}
    for group in groups:
        source = source_row_for_group(args.model, group, args)
        if source is not None:
            source_rows[group_key(group)] = source
    log(f"{args.model}/{args.round_name}: groups={len(groups)} source_rows={len(source_rows)} cases={len(cases)} donors={parse_csv(args.search_donor_prompts)}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "groups": [group_key(group) for group in groups],
                    "cases": [case["case_id"] for case in cases],
                    "source_rows": len(source_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"groups": groups, "cases": cases, "source_rows": source_rows}
    if not groups or not source_rows or not cases:
        summary = summarize_rows([], groups, cases, args, attn_impl=None)
        summary["skipped_model_load"] = True
        summary["skip_reason"] = "missing groups/source_rows/cases for gear response atlas"
        p828.write_jsonl(out_dir / f"phase837_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase837_{args.model}_summary.json", summary)
        print(json.dumps({"model": args.model, "round": args.round_name, "rows": 0, "skipped_model_load": True}, ensure_ascii=False, indent=2))
        return summary

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, 1):
            case_rows = eval_case(model, tokenizer, device, standards, case, groups, source_rows, args)
            rows.extend(case_rows)
            if idx % int(args.log_every) == 0 or idx == len(cases):
                log(f"{args.model}: evaluated cases {idx}/{len(cases)} rows={len(rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, groups, cases, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase837_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase837_{args.model}_summary.json", summary)
    printable = {
        "model": args.model,
        "round": args.round_name,
        "rows": summary["n_rows"],
        "target_transition_rows": summary["target_transition_rows"],
        "degraded_rows": summary["degraded_rows"],
        "object_echo_rows": summary["object_echo_rows"],
        "response_type_counts": summary["response_type_counts"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 837 Global Gear Response Atlas Pilot ({payload['round']})",
        "",
        "- Objective: collect standardized gear-like response fingerprints across components, cases, donors, and output metrics.",
        "- Boundary: pilot atlas only; no final gear decomposition yet.",
        "",
        "## Model Summary",
        "",
        "| model | rows | groups | cases | target | improved | degraded | object_echo | contrast_cleared | echo_cleared | rank_improved | top response types |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {data.get('n_rows', 0)} | {data.get('n_component_groups', 0)} | {data.get('n_cases', 0)} | "
            f"{data.get('target_transition_rows', 0)} | {data.get('improved_rows', 0)} | {data.get('degraded_rows', 0)} | "
            f"{data.get('object_echo_rows', 0)} | {data.get('contrast_cleared_rows', 0)} | {data.get('echo_cleared_rows', 0)} | "
            f"{data.get('target_rank_improved_rows', 0)} | `{json.dumps(data.get('response_type_counts') or {}, ensure_ascii=False)}` |"
        )
    lines += ["", "## Reuse Candidates", ""]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(f"### {model_name}")
        reuse = data.get("reuse_candidates") or []
        if not reuse:
            lines.append("")
            lines.append("No reuse candidates under this pilot setting.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| component | target cases | contrast-cleared cases | echo-cleared cases | response types |")
        lines.append("|---|---:|---:|---:|---|")
        for item in reuse[:12]:
            lines.append(
                f"| `{item.get('component_label_full')}` | {item.get('target_case_count')} | "
                f"{item.get('contrast_cleared_case_count')} | {item.get('echo_cleared_case_count')} | "
                f"`{json.dumps(item.get('response_types') or {}, ensure_ascii=False)}` |"
            )
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
        "model_summaries": {},
        "models": [],
    }
    for model_name in p828.MODELS:
        path = out_dir / f"phase837_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase837_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase837_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p829.build_parser()
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-span-candidates", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
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
