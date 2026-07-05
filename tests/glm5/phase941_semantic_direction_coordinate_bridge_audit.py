#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
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

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase939_bilingual_specificity_tightening_audit as p939  # noqa: E402
import phase940_semantic_boundary_bridge_audit as p940  # noqa: E402


PHASE = 941
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase941_semantic_direction_coordinate_bridge_audit")
PHASE940_ROOT = Path("tests/result/phase940_semantic_boundary_bridge_audit")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not vals else float(sum(vals) / len(vals))


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def parse_ints(text: str) -> list[int]:
    out = []
    for part in parse_csv(text):
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return out


def norm(vec: torch.Tensor | None) -> float:
    if vec is None:
        return 0.0
    return float(torch.linalg.vector_norm(vec.float().cpu()).item())


def scale_to(vec: torch.Tensor | None, target_norm: float) -> torch.Tensor | None:
    if vec is None:
        return None
    base = vec.float().cpu()
    base_norm = norm(base)
    if base_norm <= 0 or target_norm <= 0:
        return None
    return base / base_norm * float(target_norm)


def deterministic_generator(salt: str) -> torch.Generator:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) % (2**31 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def topk_direction(direction: torch.Tensor, k: int, largest: bool) -> torch.Tensor | None:
    vec = direction.float().cpu()
    dim = int(vec.numel())
    if dim <= 0:
        return None
    use_k = max(1, min(int(k), dim))
    idx = torch.topk(vec.abs(), use_k, largest=largest).indices
    out = torch.zeros_like(vec)
    out[idx] = vec[idx]
    return out


def random_k_direction(direction: torch.Tensor, k: int, salt: str) -> torch.Tensor | None:
    vec = direction.float().cpu()
    dim = int(vec.numel())
    if dim <= 0:
        return None
    use_k = max(1, min(int(k), dim))
    generator = deterministic_generator(salt)
    perm = torch.randperm(dim, generator=generator)[:use_k]
    out = torch.zeros_like(vec)
    out[perm] = torch.randn(use_k, generator=generator, dtype=torch.float32)
    return scale_to(out, norm(vec))


def selected_positive_pairs(args: argparse.Namespace) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    summary = read_json(PHASE940_ROOT / args.phase940_round / f"phase940_{args.model}_summary.json")
    selected: set[tuple[str, str]] = set()
    rows = []
    for row in summary.get("bridge_rows") or []:
        if row.get("condition") != "specific_direction":
            continue
        if finite(row.get("relation_margin_delta_mean"), 0.0) <= 0:
            continue
        if finite(row.get("boundary_margin_delta_mean"), 0.0) <= 0:
            continue
        if finite(row.get("boundary_bridge_gain_vs_best_control"), -999.0) <= float(args.min_phase940_bridge_gain):
            continue
        relation = str(row.get("relation"))
        pair = str(row.get("language_pair"))
        selected.add((relation, pair))
        rows.append(dict(row))
    if selected:
        return selected, rows
    fallback, phase939_rows = p940.selected_relation_language_pairs(args)
    return fallback, phase939_rows


def filter_specs(
    specs: list[dict[str, Any]],
    selected_pairs: set[tuple[str, str]],
    max_per_pair: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        key = (str(spec.get("relation")), str(spec.get("language_pair")))
        if key not in selected_pairs or spec.get("specific_direction") is None:
            continue
        buckets[key].append(spec)
    out: list[dict[str, Any]] = []
    for key in sorted(buckets):
        items = sorted(
            buckets[key],
            key=lambda spec: (
                str(spec.get("target_label")),
                str(spec.get("train_template")),
                str(spec.get("test_template")),
            ),
        )
        if int(max_per_pair) > 0:
            items = items[: int(max_per_pair)]
        out.extend(items)
    return out


def make_row(
    model_name: str,
    sample: dict[str, Any],
    spec: dict[str, Any],
    condition: str,
    top_k: int | None,
    alpha: float | None,
    direction: torch.Tensor | None,
    full_direction: torch.Tensor | None,
    base: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    row = p940.make_row(model_name, sample, spec, condition, alpha, base, patched)
    row["phase"] = PHASE
    row["row_kind"] = "phase941_semantic_direction_coordinate_bridge_row"
    row["top_k"] = top_k
    row["direction_norm_used"] = norm(direction)
    row["full_specific_direction_norm"] = norm(full_direction)
    row["norm_fraction_vs_full"] = None
    if full_direction is not None and norm(full_direction) > 0:
        row["norm_fraction_vs_full"] = row["direction_norm_used"] / norm(full_direction)
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "target_logit_delta_mean": mean([row.get("target_label_logit_delta") for row in rows]),
        "relation_margin_delta_mean": mean([row.get("target_margin_vs_relation_best_other_delta") for row in rows]),
        "boundary_margin_delta_mean": mean([row.get("target_vs_all_boundary_margin_delta") for row in rows]),
        "period_margin_delta_mean": mean([row.get("target_vs_punctuation_period_margin_delta") for row in rows]),
        "eos_margin_delta_mean": mean([row.get("target_vs_eos_margin_delta") for row in rows]),
        "boundary_logit_delta_mean": mean([row.get("all_boundary_logit_delta") for row in rows]),
        "norm_fraction_mean": mean([row.get("norm_fraction_vs_full") for row in rows]),
        "rank_improved": sum(1 for row in rows if row.get("rank_improved")),
        "new_relation_winner_target": sum(1 for row in rows if row.get("new_relation_winner_target")),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        item = {key: value for key, value in zip(keys, key_tuple)}
        item.update(summarize_rows(items))
        out.append(item)
    out.sort(
        key=lambda row: (
            finite(row.get("boundary_margin_delta_mean"), -999.0),
            finite(row.get("relation_margin_delta_mean"), -999.0),
        ),
        reverse=True,
    )
    return out


def concentration_rows(by_relation_condition: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in by_relation_condition:
        buckets[(row.get("relation"), row.get("language_pair"), row.get("alpha"), row.get("top_k"))].append(row)
    out = []
    for (relation, language_pair, alpha, top_k), rows in buckets.items():
        if top_k is None:
            continue
        by_cond = {row.get("condition"): row for row in rows}
        top_raw = by_cond.get("topk_raw")
        top_same = by_cond.get("topk_same_norm")
        rand = by_cond.get("randomk_same_norm")
        bottom = by_cond.get("bottomk_same_norm")
        full_rows = [
            row
            for row in by_relation_condition
            if row.get("relation") == relation
            and row.get("language_pair") == language_pair
            and row.get("alpha") == alpha
            and row.get("condition") == "full_specific"
            and row.get("top_k") is None
        ]
        full = full_rows[0] if full_rows else None
        if not top_raw or not top_same or not full:
            continue
        control_values = [
            finite(row.get("boundary_margin_delta_mean"), -999.0)
            for row in [rand, bottom]
            if row is not None
        ]
        control_best = max(control_values) if control_values else None
        full_boundary = finite(full.get("boundary_margin_delta_mean"), 0.0)
        raw_boundary = finite(top_raw.get("boundary_margin_delta_mean"), 0.0)
        same_boundary = finite(top_same.get("boundary_margin_delta_mean"), 0.0)
        out.append(
            {
                "relation": relation,
                "language_pair": language_pair,
                "alpha": alpha,
                "top_k": top_k,
                "full_relation_margin_delta_mean": full.get("relation_margin_delta_mean"),
                "full_boundary_margin_delta_mean": full.get("boundary_margin_delta_mean"),
                "topk_raw_relation_margin_delta_mean": top_raw.get("relation_margin_delta_mean"),
                "topk_raw_boundary_margin_delta_mean": top_raw.get("boundary_margin_delta_mean"),
                "topk_same_relation_margin_delta_mean": top_same.get("relation_margin_delta_mean"),
                "topk_same_boundary_margin_delta_mean": top_same.get("boundary_margin_delta_mean"),
                "control_best_boundary_margin_delta_mean": control_best,
                "topk_raw_boundary_fraction_of_full": None
                if full_boundary <= 0
                else float(raw_boundary / full_boundary),
                "topk_same_boundary_gain_vs_control": None
                if control_best is None
                else float(same_boundary - control_best),
                "joint_coordinate_bridge_score": None
                if control_best is None
                else float(min(same_boundary, finite(top_same.get("relation_margin_delta_mean"), -999.0), same_boundary - control_best)),
                "topk_raw_rows": top_raw.get("rows"),
                "topk_same_rows": top_same.get("rows"),
                "full_rows": full.get("rows"),
            }
        )
    out.sort(
        key=lambda row: (
            finite(row.get("joint_coordinate_bridge_score"), -999.0),
            finite(row.get("topk_raw_boundary_fraction_of_full"), -999.0),
            finite(row.get("topk_same_boundary_gain_vs_control"), -999.0),
        ),
        reverse=True,
    )
    return out


def evidence_label(rows: list[dict[str, Any]]) -> str:
    positives = [
        row
        for row in rows
        if finite(row.get("topk_raw_boundary_fraction_of_full"), 0.0) >= 0.25
        and finite(row.get("topk_same_boundary_gain_vs_control"), -999.0) > 0.02
        and finite(row.get("joint_coordinate_bridge_score"), -999.0) > 0.02
    ]
    rels = {str(row.get("relation")) for row in positives}
    if len(rels) >= 2:
        return "coordinate_concentrated_semantic_boundary_bridge_positive"
    if positives:
        return "partial_coordinate_concentrated_semantic_boundary_bridge_positive"
    return "coordinate_concentrated_bridge_weak_or_controlled"


def condition_vectors(direction: torch.Tensor, topks: list[int], spec: dict[str, Any]) -> list[tuple[str, int | None, torch.Tensor | None]]:
    full = direction.float().cpu()
    full_norm = norm(full)
    out: list[tuple[str, int | None, torch.Tensor | None]] = [("full_specific", None, full)]
    for top_k in topks:
        top_raw = topk_direction(full, top_k, largest=True)
        bottom_raw = topk_direction(full, top_k, largest=False)
        salt = "|".join(
            [
                str(spec.get("relation")),
                str(spec.get("target_label")),
                str(spec.get("train_template")),
                str(spec.get("test_template")),
                str(top_k),
                "phase941",
            ]
        )
        out.append(("topk_raw", int(top_k), top_raw))
        out.append(("topk_same_norm", int(top_k), scale_to(top_raw, full_norm)))
        out.append(("bottomk_same_norm", int(top_k), scale_to(bottom_raw, full_norm)))
        out.append(("randomk_same_norm", int(top_k), random_k_direction(full, int(top_k), salt)))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = p939.build_samples(args)
    selected_pairs, selected_phase940_rows = selected_positive_pairs(args)
    hidden_by_relation = p938.phase937_best_hidden(args.model, args.phase937_round)
    if not hidden_by_relation:
        hidden_by_relation = {relation: -1 for relation in parse_csv(args.relations)}
    topks = parse_ints(args.topks)
    dry_payload = {
        "phase": PHASE,
        "title": "Semantic Direction Coordinate Bridge Audit",
        "model": args.model,
        "sample_count": len(samples),
        "selected_relation_language_pairs": sorted([f"{rel}:{pair}" for rel, pair in selected_pairs]),
        "selected_phase940_bridge_rows": selected_phase940_rows,
        "requested_topks": topks,
        "hidden_by_relation": hidden_by_relation,
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase941_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase941_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    rows: list[dict[str, Any]] = []
    direction_specs: list[dict[str, Any]] = []
    selected_specs: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        auto_indices = p940.p937.auto_hidden_indices(model)
        hidden_by_relation = {
            rel: (auto_indices[len(auto_indices) // 2] if int(idx) < 0 else int(idx))
            for rel, idx in hidden_by_relation.items()
        }
        hidden_indices = sorted(set(hidden_by_relation.values()))
        vectors = p938.forward_vectors(model, tokenizer, device, samples, hidden_indices, int(args.batch_size))
        baseline_logits = p938.forward_logits(model, tokenizer, device, samples, int(args.batch_size))
        direction_specs = p939.build_direction_specs(samples, vectors, hidden_by_relation, int(args.min_train_per_label))
        selected_specs = filter_specs(direction_specs, selected_pairs, int(args.max_specs_per_pair))
        if int(args.max_direction_specs) > 0:
            selected_specs = selected_specs[: int(args.max_direction_specs)]
        labels_by_relation = {relation: p938.relation_labels(samples, relation) for relation in hidden_by_relation}
        token_maps = {relation: p938.label_token_map(tokenizer, labels) for relation, labels in labels_by_relation.items()}
        boundary_groups = p940.boundary_token_groups(tokenizer)
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for sample in samples:
            relation = str(sample["relation"])
            baseline_metrics[p938.sample_key(sample)] = p940.boundary_metrics(
                tokenizer,
                baseline_logits[p938.sample_key(sample)],
                str(sample["target_label"]),
                token_maps[relation],
                boundary_groups,
            )

        for spec_idx, spec in enumerate(selected_specs, 1):
            relation = str(spec["relation"])
            label_tokens = token_maps[relation]
            full_direction = spec.get("specific_direction")
            if full_direction is None:
                continue
            for sample in spec["test_samples"]:
                base = baseline_metrics[p938.sample_key(sample)]
                rows.append(make_row(args.model, sample, spec, "baseline", None, None, None, full_direction, base, base))
            for condition, top_k, direction in condition_vectors(full_direction, topks, spec):
                if direction is None or norm(direction) <= 0:
                    continue
                for alpha in [float(x) for x in parse_csv(args.alphas)]:
                    patched_logits = p938.patched_logits_batch(
                        model,
                        tokenizer,
                        device,
                        spec["test_samples"],
                        int(spec["hidden_idx"]),
                        direction,
                        float(alpha),
                        int(args.batch_size),
                    )
                    for sample in spec["test_samples"]:
                        base = baseline_metrics[p938.sample_key(sample)]
                        patched = p940.boundary_metrics(
                            tokenizer,
                            patched_logits[p938.sample_key(sample)],
                            str(sample["target_label"]),
                            label_tokens,
                            boundary_groups,
                        )
                        rows.append(
                            make_row(
                                args.model,
                                sample,
                                spec,
                                condition,
                                top_k,
                                float(alpha),
                                direction,
                                full_direction,
                                base,
                                patched,
                            )
                        )
            if spec_idx % max(1, int(args.log_every)) == 0 or spec_idx == len(selected_specs):
                log(f"{args.model}/{args.round_name}: selected_spec={spec_idx}/{len(selected_specs)} rows={len(rows)}")
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_condition = summarize_by(rows, ["condition", "top_k", "alpha"])
    by_relation_language_condition = summarize_by(rows, ["relation", "language_pair", "condition", "top_k", "alpha"])
    by_language_condition = summarize_by(rows, ["language_pair", "condition", "top_k", "alpha"])
    concentrations = concentration_rows(by_relation_language_condition)
    evidence = evidence_label(concentrations)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "hidden_by_relation": hidden_by_relation,
        "direction_specs_all": len(direction_specs),
        "direction_specs_selected": len(selected_specs),
        "rows": len(rows),
        "overall": summarize_rows(rows),
        "by_condition": by_condition,
        "by_language_condition": by_language_condition,
        "by_relation_language_condition": by_relation_language_condition,
        "concentration_rows": concentrations,
        "evidence_label": evidence,
        "boundary": "residual-coordinate subspace audit for Phase940 semantic directions; not MLP-neuron or natural-generation closure",
    }
    write_json(out_dir / f"phase941_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase941_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": evidence,
                "selected_specs": len(selected_specs),
                "top_concentration_rows": concentrations[:16],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase941_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    condition_rows = []
    concentration_summary = []
    for summary in summaries:
        model_name = summary.get("model")
        for row in summary.get("by_condition") or []:
            item = dict(row)
            item["model"] = model_name
            condition_rows.append(item)
        for row in summary.get("concentration_rows") or []:
            item = dict(row)
            item["model"] = model_name
            concentration_summary.append(item)
    concentration_summary.sort(
        key=lambda row: (
            finite(row.get("joint_coordinate_bridge_score"), -999.0),
            finite(row.get("topk_same_boundary_gain_vs_control"), -999.0),
            finite(row.get("topk_raw_boundary_fraction_of_full"), -999.0),
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "evidence_counts": dict(evidence_counts),
        "model_summaries": summaries,
        "condition_rows": condition_rows,
        "top_concentration_rows": concentration_summary[:220],
    }
    write_json(out_dir / "phase941_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase941_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 941 semantic direction coordinate bridge audit", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Condition Rows", ""]
    lines.append("| model | condition | top_k | alpha | rows | relation margin | boundary margin | period margin | eos margin | norm fraction |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("condition_rows") or []:
        lines.append(
            "| {model} | {condition} | {top_k} | {alpha} | {rows} | {relation_margin_delta_mean} | {boundary_margin_delta_mean} | {period_margin_delta_mean} | {eos_margin_delta_mean} | {norm_fraction_mean} |".format(
                **row
            )
        )
    lines += ["", "## Top Coordinate Rows", ""]
    lines.append("| model | relation | pair | top_k | full boundary | top raw boundary | raw fraction | top same boundary | control best | gain | joint |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_concentration_rows") or []:
        lines.append(
            "| {model} | {relation} | {language_pair} | {top_k} | {full_boundary_margin_delta_mean} | {topk_raw_boundary_margin_delta_mean} | {topk_raw_boundary_fraction_of_full} | {topk_same_boundary_margin_delta_mean} | {control_best_boundary_margin_delta_mean} | {topk_same_boundary_gain_vs_control} | {joint_coordinate_bridge_score} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="semantic_direction_coordinate_bridge_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase940-round", default="semantic_boundary_bridge_audit")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=6)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--min-specific-margin", type=float, default=0.05)
    parser.add_argument("--min-specific-gain", type=float, default=0.05)
    parser.add_argument("--min-phase940-bridge-gain", type=float, default=0.02)
    parser.add_argument("--max-specs-per-pair", type=int, default=8)
    parser.add_argument("--max-direction-specs", type=int, default=0)
    parser.add_argument("--topks", default="64,256")
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=12)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "evidence": payload["evidence_counts"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
