#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase895_no_single_minimality_head_pathway_split as p895  # noqa: E402
import phase898_domain_axis_holdout_validation as p898  # noqa: E402


PHASE = 899
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase899_domain_axis_rollout_protocol_audit")
PHASE898_ROOT = Path("tests/result/phase898_domain_axis_holdout_validation")
PHASE898_ROUND = "domain_axis_holdout_validation"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def counter_values(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def gear_key(gear: dict[str, Any]) -> str:
    return p862.gear_key(gear)


def gear_keys_from_subset(subset_key: str) -> list[str]:
    return [part for part in str(subset_key or "").split("+") if part.startswith("L") and "C" in part]


def parse_gears(subset_key: str) -> list[dict[str, Any]]:
    gears = [p862.parse_gear_key(key) for key in gear_keys_from_subset(subset_key)]
    return [gear for gear in gears if gear is not None]


def condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("edit_mode")))


def source_priority(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("no_single_pair_conditions") or 0),
        int(row.get("source_candidate_closure") or 0),
        int(row.get("single_axis_closure_conditions") or 0),
        int(row.get("conditions") or 0),
    )


def condition_priority(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(bool(row.get("no_single_pair_keys"))),
        int(bool(row.get("single_closure_keys"))),
        str(row.get("case_id")),
        str(row.get("prompt_variant")),
    )


def load_phase898_selection(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = PHASE898_ROOT / args.phase898_round
    summary = read_json(out_dir / f"phase898_{model_name}_summary.json")
    condition_rows = read_jsonl(out_dir / f"phase898_{model_name}_condition_rows.jsonl")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        if row.get("source_candidate_closure"):
            by_source[str(row.get("source_key"))].append(row)
    sources = []
    for source in summary.get("source_summaries") or []:
        key = str(source.get("source_key"))
        selected = list(by_source.get(key) or [])
        if not selected:
            continue
        selected.sort(key=condition_priority, reverse=True)
        if int(args.max_conditions_per_source) > 0:
            selected = selected[: int(args.max_conditions_per_source)]
        source = dict(source)
        source["selected_phase898_conditions"] = selected
        sources.append(source)
    sources.sort(key=source_priority, reverse=True)
    if int(args.max_sources_per_model) > 0:
        sources = sources[: int(args.max_sources_per_model)]
    return sources


def specs_for_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    source_subset_key = str(source.get("subset_key") or source.get("source_subset_key") or "")
    gears = parse_gears(source_subset_key)
    if not gears:
        return []
    specs = [
        {
            "rollout_relation": "base_original",
            "subset_key": "base_original",
            "subset_size": 0,
            "gear_keys": [],
            "gears": [],
            "is_source_candidate": False,
            "is_component": False,
        }
    ]
    if len(gears) == 2:
        for gear in gears:
            specs.append(
                {
                    "rollout_relation": "component_single",
                    "subset_key": gear_key(gear),
                    "subset_size": 1,
                    "gear_keys": [gear_key(gear)],
                    "gears": [gear],
                    "is_source_candidate": False,
                    "is_component": True,
                }
            )
        specs.append(
            {
                "rollout_relation": "source_candidate_pair",
                "subset_key": "+".join(gear_key(gear) for gear in gears),
                "subset_size": 2,
                "gear_keys": [gear_key(gear) for gear in gears],
                "gears": gears,
                "is_source_candidate": True,
                "is_component": False,
            }
        )
    else:
        gear = gears[0]
        specs.append(
            {
                "rollout_relation": "source_candidate_single",
                "subset_key": gear_key(gear),
                "subset_size": 1,
                "gear_keys": [gear_key(gear)],
                "gears": [gear],
                "is_source_candidate": True,
                "is_component": False,
            }
        )
    return specs


def protocol_drift(text: str) -> bool:
    raw = str(text or "").lower()
    norm = p895.p894.normalize(text) if hasattr(p895, "p894") else raw
    raw_markers = [
        "answer:",
        "category:",
        "subclass:",
        "item:",
        "class:",
        "the answer is",
        "category is",
        "please",
        "i need",
        "okay, so",
        "1.",
        "2.",
        "{",
        "}",
        "another example",
        "the category that",
    ]
    norm_markers = [
        "subclass",
        "please",
        "i need",
        "okay so",
        "the answer is",
        "category is",
        "another example",
        "the category that",
    ]
    word_count = len(re.findall(r"[A-Za-z]+", raw))
    list_like = raw.count(",") >= 2 or " or " in raw
    return (
        any(marker in raw for marker in raw_markers)
        or any(marker in norm for marker in norm_markers)
        or word_count > 5
        or list_like
    )


def rollout_flags(text: str, case: dict[str, Any]) -> dict[str, Any]:
    rollout = p856.classify_rollout(text, case)
    drift = protocol_drift(text)
    clear_no_drift = bool(rollout.get("rollout_clear_answer_class")) and not drift
    class_no_echo_no_drift = bool(rollout.get("rollout_answer_class")) and not bool(rollout.get("rollout_object_echo")) and not drift
    return {
        **rollout,
        "protocol_drift": drift,
        "rollout_clear_answer_no_protocol": clear_no_drift,
        "rollout_class_no_echo_no_protocol": class_no_echo_no_drift,
        "rollout_bad_transition": bool(rollout.get("rollout_object_echo") or rollout.get("rollout_other_or_format") or drift),
    }


def make_rollout_row(
    model_name: str,
    source: dict[str, Any],
    condition: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    generated: str,
    generated_ids: list[int],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "row_kind": "phase899_rollout_protocol_audit_row",
        "model": model_name,
        "source_key": source.get("source_key"),
        "source_type": source.get("source_type"),
        "source_subset_key": source.get("subset_key"),
        "eval_domain": source.get("domain"),
        "case_id": condition.get("case_id"),
        "case_split": case.get("split_source", condition.get("case_split")),
        "object": case.get("object"),
        "canonical_answer": case.get("canonical_answer"),
        "answer_aliases": case.get("answer_aliases"),
        "prompt_variant": condition.get("prompt_variant"),
        "edit_mode": condition.get("edit_mode"),
        "phase898_no_single_pair_keys": condition.get("no_single_pair_keys"),
        "phase898_single_closure_keys": condition.get("single_closure_keys"),
        "rollout_relation": spec.get("rollout_relation"),
        "subset_key": spec.get("subset_key"),
        "subset_size": spec.get("subset_size"),
        "gear_keys": spec.get("gear_keys"),
        "is_source_candidate": spec.get("is_source_candidate"),
        "is_component": spec.get("is_component"),
        "generated_text": generated,
        "generated_ids": generated_ids,
        **rollout_flags(generated, case),
    }


def summarize_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "clear_answer": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "clear_answer_no_protocol": sum(1 for row in rows if row.get("rollout_clear_answer_no_protocol")),
        "class_no_echo_no_protocol": sum(1 for row in rows if row.get("rollout_class_no_echo_no_protocol")),
        "object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "identity_overlap": sum(1 for row in rows if row.get("rollout_identity_class_overlap")),
        "other_or_format": sum(1 for row in rows if row.get("rollout_other_or_format")),
        "protocol_drift": sum(1 for row in rows if row.get("protocol_drift")),
        "bad_transition": sum(1 for row in rows if row.get("rollout_bad_transition")),
        "labels": counter_values(Counter(str(row.get("rollout_label")) for row in rows)),
    }


def summarize_model(
    model_name: str,
    sources: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    attn_impl: str | None,
) -> dict[str, Any]:
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_condition: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_relation[str(row.get("rollout_relation"))].append(row)
        by_source[str(row.get("source_key"))].append(row)
        by_source_condition[
            (
                str(row.get("source_key")),
                str(row.get("case_id")),
                str(row.get("prompt_variant")),
                str(row.get("edit_mode")),
            )
        ].append(row)

    source_summaries = []
    for source in sources:
        key = str(source.get("source_key"))
        vals = by_source.get(key, [])
        source_rows = [row for row in vals if row.get("is_source_candidate")]
        base_rows = [row for row in vals if row.get("rollout_relation") == "base_original"]
        component_rows = [row for row in vals if row.get("is_component")]
        clean_gains = 0
        clean_losses = 0
        for cond_rows in by_source_condition.values():
            if not cond_rows or str(cond_rows[0].get("source_key")) != key:
                continue
            base = next((row for row in cond_rows if row.get("rollout_relation") == "base_original"), None)
            src = next((row for row in cond_rows if row.get("is_source_candidate")), None)
            if not base or not src:
                continue
            if src.get("rollout_clear_answer_no_protocol") and not base.get("rollout_clear_answer_no_protocol"):
                clean_gains += 1
            if base.get("rollout_clear_answer_no_protocol") and not src.get("rollout_clear_answer_no_protocol"):
                clean_losses += 1
        source_summaries.append(
            {
                "model": model_name,
                "source_key": key,
                "source_type": source.get("source_type"),
                "domain": source.get("domain"),
                "subset_key": source.get("subset_key"),
                "phase898_source_candidate_closure": int(source.get("source_candidate_closure") or 0),
                "selected_conditions": len(source.get("selected_phase898_conditions") or []),
                "base": summarize_relation(base_rows),
                "source_candidate": summarize_relation(source_rows),
                "components": summarize_relation(component_rows),
                "clean_rollout_gain_vs_base": clean_gains,
                "clean_rollout_loss_vs_base": clean_losses,
            }
        )
    source_summaries.sort(
        key=lambda row: (
            row["source_candidate"].get("clear_answer_no_protocol") or 0,
            row["source_candidate"].get("class_no_echo_no_protocol") or 0,
            row.get("selected_conditions") or 0,
        ),
        reverse=True,
    )

    source_candidate_rows = [row for row in rows if row.get("is_source_candidate")]
    base_rows = by_relation.get("base_original", [])
    component_rows = [row for row in rows if row.get("is_component")]
    overall = {
        "sources": len(sources),
        "selected_conditions": sum(len(source.get("selected_phase898_conditions") or []) for source in sources),
        "rollout_rows": len(rows),
        "base_rows": len(base_rows),
        "component_rows": len(component_rows),
        "source_candidate_rows": len(source_candidate_rows),
        "base": summarize_relation(base_rows),
        "source_candidate": summarize_relation(source_candidate_rows),
        "components": summarize_relation(component_rows),
    }
    if overall["source_candidate"]["clear_answer_no_protocol"]:
        evidence_label = "first_token_axis_has_some_clean_rollout_transfer"
    elif overall["source_candidate"]["answer_class"]:
        evidence_label = "first_token_axis_rollout_partly_answer_class_but_not_clean"
    else:
        evidence_label = "first_token_axis_does_not_rollout_to_answer_route"
    return {
        "phase": PHASE,
        "title": "Domain Axis Rollout And Protocol/Object Echo Audit",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "sources": sources,
        "overall": overall,
        "source_summaries": source_summaries,
        "relation_summaries": {relation: summarize_relation(vals) for relation, vals in sorted(by_relation.items())},
        "evidence_label": evidence_label,
        "boundary": (
            "Phase899 audits whether Phase898 first-token domain-axis closure survives greedy rollout "
            "without object echo or protocol drift. It is a route-quality audit, not global closure."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = load_phase898_selection(args.model, args)
    if args.dry_run or not sources:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "status": "dry_run" if sources else "no_sources",
            "sources": sources,
        }
        p846.write_json(out_dir / f"phase899_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase899_{args.model}_rollout_rows.jsonl", [])
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
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for sidx, source in enumerate(sources, 1):
            specs = specs_for_source(source)
            for cidx, condition in enumerate(source.get("selected_phase898_conditions") or [], 1):
                case = case_map.get(str(condition.get("case_id")))
                if not case:
                    continue
                prompt_key = (str(condition.get("case_id")), str(condition.get("prompt_variant")))
                if prompt_key not in prompt_cache:
                    prompt = p885.prompt_for_case(case, str(condition.get("prompt_variant")))
                    prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
                prompt_ids = prompt_cache[prompt_key]
                for spec in specs:
                    mode = "original" if spec.get("rollout_relation") == "base_original" else str(condition.get("edit_mode"))
                    generated, generated_ids = p862.greedy_with_scaled_gears(
                        model,
                        tokenizer,
                        device,
                        prompt_ids,
                        list(spec.get("gears") or []),
                        mode,
                        int(args.max_new_tokens),
                        float(args.scale_up_factor),
                    )
                    rows.append(make_rollout_row(args.model, source, condition, case, spec, generated, generated_ids))
                if cidx % max(1, int(args.log_every)) == 0 or cidx == len(source.get("selected_phase898_conditions") or []):
                    log(
                        f"{args.model}/{args.round_name}: source={sidx}/{len(sources)} "
                        f"condition={cidx}/{len(source.get('selected_phase898_conditions') or [])} rows={len(rows)}"
                    )
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, sources, rows, attn_impl)
    p846.write_json(out_dir / f"phase899_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase899_{args.model}_rollout_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 899 domain axis rollout and protocol/object echo audit",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Relation summaries", ""])
    lines.append("| relation | rows | clear no protocol | class no echo no protocol | answer class | object echo | protocol drift | bad transition | labels |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for relation, row in (payload.get("relation_summaries") or {}).items():
        lines.append(
            "| {relation} | {rows} | {clear_answer_no_protocol} | {class_no_echo_no_protocol} | {answer_class} | "
            "{object_echo} | {protocol_drift} | {bad_transition} | {labels} |".format(relation=relation, **row)
        )
    lines.extend(["", "## Source summaries", ""])
    lines.append(
        "| model | source | domain | subset | selected | source clean | source class-clean | source echo | source drift | gain | loss | labels |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("source_summaries") or []:
        src = row.get("source_candidate") or {}
        lines.append(
            "| {model} | {source_type} | {domain} | {subset_key} | {selected_conditions} | "
            "{clean} | {class_clean} | {echo} | {drift} | {gain} | {loss} | {labels} |".format(
                clean=src.get("clear_answer_no_protocol"),
                class_clean=src.get("class_no_echo_no_protocol"),
                echo=src.get("object_echo"),
                drift=src.get("protocol_drift"),
                gain=row.get("clean_rollout_gain_vs_base"),
                loss=row.get("clean_rollout_loss_vs_base"),
                labels=src.get("labels"),
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase899_{model_name}_summary.json"
        if path.exists():
            summaries.append(read_json(path))
    relation_summaries: dict[str, Counter[str]] = defaultdict(Counter)
    source_summaries = []
    scalar = Counter()
    evidence_counts = Counter()
    for summary in summaries:
        evidence_counts[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        for key in ["sources", "selected_conditions", "rollout_rows", "base_rows", "component_rows", "source_candidate_rows"]:
            scalar[key] += int(overall.get(key) or 0)
        for relation, rel in (summary.get("relation_summaries") or {}).items():
            for key, value in rel.items():
                if isinstance(value, int):
                    relation_summaries[relation][key] += int(value)
            for label, count in (rel.get("labels") or {}).items():
                relation_summaries[relation][f"label::{label}"] += int(count)
        source_summaries.extend(summary.get("source_summaries") or [])

    relation_payload: dict[str, dict[str, Any]] = {}
    for relation, counter in sorted(relation_summaries.items()):
        labels = {
            key.split("::", 1)[1]: int(value)
            for key, value in sorted(counter.items())
            if key.startswith("label::")
        }
        relation_payload[relation] = {
            key: int(value)
            for key, value in sorted(counter.items())
            if not key.startswith("label::")
        }
        relation_payload[relation]["labels"] = labels
    source_summaries.sort(
        key=lambda row: (
            (row.get("source_candidate") or {}).get("clear_answer_no_protocol") or 0,
            (row.get("source_candidate") or {}).get("class_no_echo_no_protocol") or 0,
            row.get("selected_conditions") or 0,
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
        "relation_summaries": relation_payload,
        "source_summaries": source_summaries,
        "evidence_label_counts": counter_values(evidence_counts),
    }
    p846.write_json(out_dir / "phase899_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase899_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="domain_axis_rollout_protocol_audit")
    parser.add_argument("--phase898-round", default=PHASE898_ROUND)
    parser.add_argument("--max-sources-per-model", type=int, default=8)
    parser.add_argument("--max-conditions-per-source", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=12)
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
