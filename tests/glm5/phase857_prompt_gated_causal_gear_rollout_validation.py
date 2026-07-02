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

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase854_full_vocab_blocker_min_cut_validation as p854  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 857
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase857_prompt_gated_causal_gear_rollout_validation")
PHASE854_ROOT = Path("tests/result/phase854_full_vocab_blocker_min_cut_validation")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def source_row_key(row: dict[str, Any]) -> str:
    return str(row.get("source_row_key") or "")


def gear_key(gear: dict[str, Any]) -> str:
    return p854.gear_key(gear)


def gear_from_key(text: str) -> dict[str, Any] | None:
    return p854.gear_from_key(text)


def phase854_rows_path(round_name: str, model_name: str) -> Path:
    return PHASE854_ROOT / round_name / f"phase854_{model_name}_rows.jsonl"


def phase854_edges_path(round_name: str, model_name: str) -> Path:
    return PHASE854_ROOT / round_name / f"phase854_{model_name}_edge_rows.jsonl"


def load_phase854_rows(model_name: str, round_name: str) -> list[dict[str, Any]]:
    path = phase854_rows_path(round_name, model_name)
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 854 rows: {path}")
    return p846.read_jsonl(path)


def load_phase854_edges(model_name: str, round_name: str) -> list[dict[str, Any]]:
    path = phase854_edges_path(round_name, model_name)
    if not path.exists():
        return []
    return p846.read_jsonl(path)


def necessary_edges_by_source(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edges:
        if row.get("label") != "necessary_blocker_reducer":
            continue
        out[source_row_key(row)].append(row)
    return out


def phase856_case_for_object(obj: str) -> dict[str, Any]:
    obj_norm = str(obj or "").strip().lower()
    for case in p856.base_cases():
        if str(case.get("object")).lower() == obj_norm:
            return dict(case)
    return {
        "case_id": f"p857_unknown_{obj_norm or 'object'}",
        "domain": "unknown",
        "object": obj_norm,
        "answer_aliases": ["geometric shape", "geometric figure", "shape", "figure"],
        "canonical_answer": "geometric shape",
        "overlap_kind": "unknown",
    }


def transfer_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    domains = set(parse_csv(args.target_domains))
    if not domains:
        return []
    rows = [case for case in p856.base_cases() if str(case.get("domain")) in domains]
    limit = int(args.max_target_cases_per_domain)
    if limit > 0:
        out: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for case in rows:
            domain = str(case.get("domain"))
            if counts[domain] >= limit:
                continue
            out.append(dict(case))
            counts[domain] += 1
        rows = out
    if int(args.max_target_cases) > 0:
        rows = rows[: int(args.max_target_cases)]
    return [dict(case) for case in rows]


def select_sources(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    full_rows = [row for row in load_phase854_rows(args.model, args.source_round) if row.get("condition_type") == "full_combo"]
    edge_map = necessary_edges_by_source(load_phase854_edges(args.model, args.source_round))
    allowed_objects = set(parse_csv(args.objects))
    if allowed_objects:
        full_rows = [row for row in full_rows if str(row.get("object") or "").lower() in allowed_objects]

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in full_rows:
        key = source_row_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    def sort_key(row: dict[str, Any]) -> tuple[int, int, float, float, str, str]:
        key = source_row_key(row)
        has_necessary = int(key in edge_map)
        target_group = int(str(row.get("source_group")) == "strong_target_not_exact")
        closed = int(bool(row.get("answer_class_closure")))
        blockers = finite(row.get("class_blocker_count"), 999999.0)
        return (-has_necessary, -target_group, -closed, blockers, str(row.get("object")), key)

    deduped.sort(key=sort_key)
    if int(args.max_sources) > 0:
        deduped = deduped[: int(args.max_sources)]
    return deduped, edge_map


def condition_specs(source: dict[str, Any], edge_map: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    gears = [gear_from_key(str(key)) for key in source.get("gear_keys") or []]
    gears = [gear for gear in gears if gear is not None]
    mode = str(source.get("edit_mode") or source.get("source_edit_mode") or "zero")
    specs: list[dict[str, Any]] = [
        {"condition_type": "original", "candidate_key": None, "mode": "original", "gears": []},
        {"condition_type": "full_combo", "candidate_key": None, "mode": mode, "gears": gears},
    ]
    by_key = {gear_key(gear): gear for gear in gears}
    added = 0
    for edge in edge_map.get(source_row_key(source), []):
        candidate = str(edge.get("candidate_key") or "")
        if candidate not in by_key:
            continue
        remain = [gear for gear in gears if gear_key(gear) != candidate]
        if remain:
            specs.append({"condition_type": "without_necessary", "candidate_key": candidate, "mode": mode, "gears": remain})
            added += 1
        if added >= int(args.max_necessary_conditions_per_source):
            break
    return specs


def row_pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_row_key")),
        str(row.get("case_id")),
        str(row.get("prompt_variant")),
        str(row.get("condition_type")),
    )


def binary_stats(actual: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(1 for a, p in zip(actual, predicted) if a and p)
    tn = sum(1 for a, p in zip(actual, predicted) if (not a) and (not p))
    fp = sum(1 for a, p in zip(actual, predicted) if (not a) and p)
    fn = sum(1 for a, p in zip(actual, predicted) if a and (not p))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = len(actual)
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [bool(row.get("rollout_answer_class")) for row in rows]
    predicted = [bool(row.get("first_token_answer_class")) for row in rows]
    return {
        "n": len(rows),
        "first_token_answer_class": sum(1 for row in rows if row.get("first_token_answer_class")),
        "first_token_clear_answer_class": sum(1 for row in rows if row.get("first_token_clear_answer_class")),
        "rollout_answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "rollout_clear_answer_class": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "rollout_identity_class_overlap": sum(1 for row in rows if row.get("rollout_identity_class_overlap")),
        "rollout_object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "rollout_other_or_format": sum(1 for row in rows if row.get("rollout_other_or_format")),
        "rollout_labels": dict(Counter(str(row.get("rollout_label")) for row in rows)),
        "mean_class_blocker_count": mean([finite(row.get("class_blocker_count")) for row in rows]),
        "mean_class_rank": mean([finite(row.get("class_best_rank"), 999999.0) for row in rows]),
        "mean_class_minus_object_logit": mean([finite(row.get("class_minus_object_logit")) for row in rows]),
        "first_token_predicts_rollout": binary_stats(actual, predicted),
    }


def compare_pairs(rows: list[dict[str, Any]], left_condition: str, right_condition: str) -> dict[str, Any]:
    by_key = {
        (str(row.get("source_row_key")), str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("condition_type"))): row
        for row in rows
    }
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        if row.get("condition_type") != left_condition:
            continue
        key = (str(row.get("source_row_key")), str(row.get("case_id")), str(row.get("prompt_variant")), right_condition)
        other = by_key.get(key)
        if other:
            pairs.append((row, other))

    def gain(metric: str) -> int:
        return sum(1 for left, right in pairs if (not bool(left.get(metric))) and bool(right.get(metric)))

    def loss(metric: str) -> int:
        return sum(1 for left, right in pairs if bool(left.get(metric)) and (not bool(right.get(metric))))

    blocker_deltas = []
    margin_deltas = []
    for left, right in pairs:
        if left.get("class_blocker_count") is not None and right.get("class_blocker_count") is not None:
            blocker_deltas.append(finite(left.get("class_blocker_count")) - finite(right.get("class_blocker_count")))
        if left.get("class_minus_object_logit") is not None and right.get("class_minus_object_logit") is not None:
            margin_deltas.append(finite(right.get("class_minus_object_logit")) - finite(left.get("class_minus_object_logit")))
    return {
        "n_pairs": len(pairs),
        "left_condition": left_condition,
        "right_condition": right_condition,
        "answer_class_gain": gain("first_token_answer_class"),
        "answer_class_loss": loss("first_token_answer_class"),
        "clear_rollout_gain": gain("rollout_clear_answer_class"),
        "clear_rollout_loss": loss("rollout_clear_answer_class"),
        "rollout_class_gain": gain("rollout_answer_class"),
        "rollout_class_loss": loss("rollout_answer_class"),
        "object_echo_reduced": sum(1 for left, right in pairs if bool(left.get("rollout_object_echo")) and not bool(right.get("rollout_object_echo"))),
        "object_echo_induced": sum(1 for left, right in pairs if (not bool(left.get("rollout_object_echo"))) and bool(right.get("rollout_object_echo"))),
        "mean_blocker_reduction": mean(blocker_deltas),
        "mean_class_minus_object_gain": mean(margin_deltas),
    }


def summarize(args: argparse.Namespace, attn_impl: str | None, sources: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition_type"))].append(row)
        by_prompt[str(row.get("prompt_variant"))].append(row)
        by_prompt_condition[f"{row.get('prompt_variant')}::{row.get('condition_type')}"].append(row)
        by_target_domain[str(row.get("domain"))].append(row)
        by_source_group[str(row.get("source_group"))].append(row)
    return {
        "phase": PHASE,
        "title": "Prompt-Gated Causal Gear Rollout Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_round": args.source_round,
        "n_sources": len(sources),
        "source_groups": dict(Counter(str(row.get("source_group")) for row in sources)),
        "n_rows": len(rows),
        "condition_summary": {key: compact(group) for key, group in sorted(by_condition.items())},
        "prompt_summary": {key: compact(group) for key, group in sorted(by_prompt.items())},
        "prompt_condition_summary": {key: compact(group) for key, group in sorted(by_prompt_condition.items())},
        "target_domain_summary": {key: compact(group) for key, group in sorted(by_target_domain.items())},
        "source_group_summary": {key: compact(group) for key, group in sorted(by_source_group.items())},
        "full_vs_original": compare_pairs(rows, "original", "full_combo"),
        "without_necessary_vs_full": compare_pairs(rows, "full_combo", "without_necessary"),
        "top_mismatch_rows": sorted(
            [
                row
                for row in rows
                if bool(row.get("first_token_answer_class")) != bool(row.get("rollout_answer_class"))
                or row.get("rollout_object_echo")
            ],
            key=lambda row: (
                int(bool(row.get("rollout_object_echo"))),
                int(bool(row.get("first_token_answer_class")) != bool(row.get("rollout_answer_class"))),
                -finite(row.get("class_minus_object_logit")),
            ),
            reverse=True,
        )[:80],
        "boundary": (
            "This is a causal replay audit over Phase 854 gear combos under Phase 856 prompt gates. "
            "It tests prompt-gated boundary changes, not a newly discovered global closure mechanism."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sources, edge_map = select_sources(args)
    target_cases = transfer_cases(args)
    prompt_variants = parse_csv(args.prompt_variants)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "sources": [
                {
                    "source_row_key": source_row_key(row),
                    "source_group": row.get("source_group"),
                    "case_id": row.get("case_id"),
                    "object": row.get("object"),
                    "prompt_variant": row.get("prompt_variant"),
                    "gear_keys": row.get("gear_keys"),
                    "edit_mode": row.get("edit_mode"),
                    "answer_class_closure": row.get("answer_class_closure"),
                    "necessary_edges": [edge.get("candidate_key") for edge in edge_map.get(source_row_key(row), [])],
                }
                for row in sources
            ],
            "target_cases": [
                {
                    "case_id": case.get("case_id"),
                    "domain": case.get("domain"),
                    "object": case.get("object"),
                    "canonical_answer": case.get("canonical_answer"),
                }
                for case in target_cases
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        for source_idx, source in enumerate(sources, 1):
            cases = target_cases or [phase856_case_for_object(str(source.get("object") or ""))]
            specs = condition_specs(source, edge_map, args)
            for case in cases:
                tsets = p856.token_sets(tokenizer, case)
                for prompt_variant in prompt_variants:
                    prompt = p856.prompt_for_case(case, prompt_variant)
                    prompt_ids = p844.encode_prompt(tokenizer, prompt)
                    for spec in specs:
                        valid_gears = [
                            gear
                            for gear in spec["gears"]
                            if 0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0
                        ]
                        logits = p844.first_logits_with_gears(model, device, prompt_ids, valid_gears, str(spec["mode"]))
                        first = p856.first_token_metrics(tokenizer, logits, tsets, int(args.topk_tokens))
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
                                "row_kind": "phase857_prompt_gated_causal_gear_rollout_validation",
                                "phase": PHASE,
                                "model": args.model,
                                "round": args.round_name,
                                "source_round": args.source_round,
                                "transfer_mode": bool(target_cases),
                                "source_row_key": source_row_key(source),
                                "source_case_id": source.get("case_id"),
                                "source_object": source.get("object"),
                                "source_prompt_variant": source.get("prompt_variant"),
                                "source_group": source.get("source_group"),
                                "source_answer_class_closure": source.get("answer_class_closure"),
                                "source_class_blocker_count": source.get("class_blocker_count"),
                                "source_combo_key": source.get("source_combo_key"),
                                "source_edit_mode": source.get("edit_mode"),
                                "case_id": case["case_id"],
                                "domain": case.get("domain"),
                                "object": case.get("object"),
                                "canonical_answer": case.get("canonical_answer"),
                                "overlap_kind": case.get("overlap_kind"),
                                "prompt_variant": prompt_variant,
                                "prompt": prompt,
                                "condition_type": spec["condition_type"],
                                "candidate_key": spec["candidate_key"],
                                "edit_mode": spec["mode"],
                                "gear_count": len(valid_gears),
                                "gear_keys": [gear_key(gear) for gear in valid_gears],
                                "token_ids": token_ids,
                                **tsets,
                                **first,
                                **rollout,
                            }
                        )
            if source_idx % max(1, int(args.log_every)) == 0 or source_idx == len(sources):
                log(f"{args.model}/{args.round_name}: prompt-gated sources {source_idx}/{len(sources)} rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(args, attn_impl, sources, rows)
    p846.write_jsonl(out_dir / f"phase857_{args.model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase857_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "sources": len(sources),
                "rows": len(rows),
                "full_vs_original": summary["full_vs_original"],
                "without_necessary_vs_full": summary["without_necessary_vs_full"],
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
        f"# Phase 857 Prompt-Gated Causal Gear Rollout Validation ({payload['round']})",
        "",
        "- Source: Phase 854 full-combo and necessary-blocker-reducer rows.",
        "- Boundary: prompt-gated causal replay, not a new gear search.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | sources | rows | full first | full rollout | full clear | full echo | full vs original clear gain/loss | echo reduced/induced | without necessary clear loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        full = (data.get("condition_summary") or {}).get("full_combo") or {}
        fvo = data.get("full_vs_original") or {}
        wvf = data.get("without_necessary_vs_full") or {}
        lines.append(
            f"| {model_name} | {data.get('n_sources', 0)} | {data.get('n_rows', 0)} | "
            f"{full.get('first_token_answer_class', 0)} | {full.get('rollout_answer_class', 0)} | "
            f"{full.get('rollout_clear_answer_class', 0)} | {full.get('rollout_object_echo', 0)} | "
            f"{fvo.get('clear_rollout_gain', 0)}/{fvo.get('clear_rollout_loss', 0)} | "
            f"{fvo.get('object_echo_reduced', 0)}/{fvo.get('object_echo_induced', 0)} | "
            f"{wvf.get('clear_rollout_loss', 0)} |"
        )
    lines += [
        "",
        "## Prompt / Condition Summary",
        "",
        "| model | prompt::condition | n | first class | rollout class | clear rollout | object echo | class blockers | class-object margin | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for key, stats in sorted((data.get("prompt_condition_summary") or {}).items()):
            pred = stats.get("first_token_predicts_rollout") or {}
            lines.append(
                f"| {model_name} | `{key}` | {stats.get('n', 0)} | "
                f"{stats.get('first_token_answer_class', 0)} | {stats.get('rollout_answer_class', 0)} | "
                f"{stats.get('rollout_clear_answer_class', 0)} | {stats.get('rollout_object_echo', 0)} | "
                f"{fmt(stats.get('mean_class_blocker_count'))} | {fmt(stats.get('mean_class_minus_object_logit'))} | "
                f"{fmt(pred.get('f1'))} |"
            )
    lines += [
        "",
        "## Pairwise Effects",
        "",
        "| model | comparison | pairs | answer gain/loss | rollout gain/loss | clear gain/loss | echo reduced/induced | blocker reduction | class-object gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for key in ["full_vs_original", "without_necessary_vs_full"]:
            stats = data.get(key) or {}
            lines.append(
                f"| {model_name} | `{key}` | {stats.get('n_pairs', 0)} | "
                f"{stats.get('answer_class_gain', 0)}/{stats.get('answer_class_loss', 0)} | "
                f"{stats.get('rollout_class_gain', 0)}/{stats.get('rollout_class_loss', 0)} | "
                f"{stats.get('clear_rollout_gain', 0)}/{stats.get('clear_rollout_loss', 0)} | "
                f"{stats.get('object_echo_reduced', 0)}/{stats.get('object_echo_induced', 0)} | "
                f"{fmt(stats.get('mean_blocker_reduction'))} | {fmt(stats.get('mean_class_minus_object_gain'))} |"
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
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase857_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = p846.read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase857_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase857_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--prompt-variants", default="natural_question,natural_category,object_only")
    parser.add_argument("--objects", default="")
    parser.add_argument("--target-domains", default="")
    parser.add_argument("--max-target-cases-per-domain", type=int, default=1)
    parser.add_argument("--max-target-cases", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=4)
    parser.add_argument("--max-necessary-conditions-per-source", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=2)
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
