#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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

import phase937_semantic_reuse_difference_state_atlas as p937  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase939_bilingual_specificity_tightening_audit as p939  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402


PHASE = 940
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase940_semantic_boundary_bridge_audit")
PHASE939_ROOT = Path("tests/result/phase939_bilingual_specificity_tightening_audit")

CONTROL_CONDITIONS = {
    "wrong_mean_direction",
    "random_same_norm",
    "negative_target_direction",
    "template_shift_same_norm",
}
TARGET_CONDITIONS = {"specific_direction", "target_direction"}


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


def token_candidates(tokenizer, phrases: list[str]) -> list[int]:
    ids: list[int] = []
    for phrase in phrases:
        try:
            encoded = tokenizer.encode(phrase, add_special_tokens=False)
        except Exception:
            encoded = []
        if encoded:
            token_id = int(encoded[0])
            if token_id not in ids:
                ids.append(token_id)
    return ids


def boundary_token_groups(tokenizer) -> dict[str, list[int]]:
    eos_ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        eos_ids.append(int(eos))
    elif isinstance(eos, list):
        eos_ids.extend(int(x) for x in eos if isinstance(x, int))
    period = token_candidates(tokenizer, [".", " .", "。", " 。"])
    punctuation = token_candidates(
        tokenizer,
        [
            ".",
            " .",
            "。",
            " 。",
            ",",
            " ,",
            "，",
            " ，",
            ":",
            " :",
            "：",
            " ：",
            ";",
            " ;",
            "；",
            " ；",
            "!",
            " !",
            "！",
            " ！",
            "?",
            " ?",
            "？",
            " ？",
        ],
    )
    protocol = token_candidates(tokenizer, ["\n", "\n\n", ":", " :", "：", " ：", "Answer", " Answer", "答案", " 答案"])
    all_boundary = sorted(set(period + punctuation + protocol + eos_ids))
    return {
        "punctuation_period": period,
        "punctuation_general": punctuation,
        "protocol_boundary": protocol,
        "eos": sorted(set(eos_ids)),
        "all_boundary": all_boundary,
    }


def max_token_score(logits: torch.Tensor, token_ids: list[int]) -> tuple[float | None, int | None]:
    valid = [int(x) for x in token_ids if 0 <= int(x) < int(logits.numel())]
    if not valid:
        return None, None
    best_id = max(valid, key=lambda token_id: float(logits[token_id].item()))
    return float(logits[best_id].item()), int(best_id)


def boundary_metrics(
    tokenizer,
    logits: torch.Tensor,
    target_label: str,
    relation_label_tokens: dict[str, list[int]],
    boundary_groups: dict[str, list[int]],
) -> dict[str, Any]:
    rel = p938.target_margin(logits, target_label, relation_label_tokens)
    target_score = rel.get("target_label_logit")
    out = dict(rel)
    for group_name, token_ids in boundary_groups.items():
        score, token_id = max_token_score(logits, token_ids)
        out[f"{group_name}_logit"] = score
        out[f"{group_name}_token_id"] = token_id
        out[f"{group_name}_token"] = None if token_id is None else p901.decode_token(tokenizer, int(token_id))
        out[f"target_vs_{group_name}_margin"] = (
            None if target_score is None or score is None else float(float(target_score) - float(score))
        )
    return out


def selected_relation_language_pairs(args: argparse.Namespace) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    summary_path = PHASE939_ROOT / args.phase939_round / f"phase939_{args.model}_summary.json"
    summary = read_json(summary_path)
    pairs: set[tuple[str, str]] = set()
    selected_rows = []
    for row in summary.get("specificity_rows") or []:
        if row.get("condition") != "specific_direction":
            continue
        margin = finite(row.get("target_margin_delta_mean"), 0.0)
        gain = finite(row.get("specificity_gain_vs_best_control"), -999.0)
        if margin >= float(args.min_specific_margin) and gain >= float(args.min_specific_gain):
            relation = str(row.get("relation"))
            pair = str(row.get("language_pair"))
            pairs.add((relation, pair))
            selected_rows.append(dict(row))
    return pairs, selected_rows


def filter_specs(specs: list[dict[str, Any]], selected_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for spec in specs:
        key = (str(spec.get("relation")), str(spec.get("language_pair")))
        if key in selected_pairs and spec.get("specific_direction") is not None:
            out.append(spec)
    return out


def make_row(
    model_name: str,
    sample: dict[str, Any],
    spec: dict[str, Any],
    condition: str,
    alpha: float | None,
    base: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "phase": PHASE,
        "row_kind": "phase940_semantic_boundary_bridge_row",
        "model": model_name,
        "sample_id": sample.get("sample_id"),
        "domain": sample.get("domain"),
        "object": sample.get("object"),
        "relation": spec.get("relation"),
        "target_label": spec.get("target_label"),
        "prompt_language": sample.get("prompt_language"),
        "train_language": spec.get("train_language"),
        "test_language": spec.get("test_language"),
        "language_pair": spec.get("language_pair"),
        "prompt_template": sample.get("prompt_template"),
        "train_template": spec.get("train_template"),
        "test_template": spec.get("test_template"),
        "hidden_idx": int(spec.get("hidden_idx")),
        "condition": condition,
        "alpha": alpha,
        "direction_norm": spec.get("direction_norm"),
        "specific_direction_norm": spec.get("specific_direction_norm"),
    }
    metric_names = [
        "target_label_logit",
        "target_margin_vs_relation_best_other",
        "target_label_rank",
        "target_vs_punctuation_period_margin",
        "target_vs_punctuation_general_margin",
        "target_vs_protocol_boundary_margin",
        "target_vs_eos_margin",
        "target_vs_all_boundary_margin",
        "punctuation_period_logit",
        "punctuation_general_logit",
        "protocol_boundary_logit",
        "eos_logit",
        "all_boundary_logit",
    ]
    for name in metric_names:
        base_value = base.get(name)
        patched_value = patched.get(name)
        row[f"base_{name}"] = base_value
        row[f"patched_{name}"] = patched_value
        if base_value is not None and patched_value is not None and name != "target_label_rank":
            row[f"{name}_delta"] = float(float(patched_value) - float(base_value))
    base_rank = base.get("target_label_rank")
    patched_rank = patched.get("target_label_rank")
    row["target_rank_delta"] = None if base_rank is None or patched_rank is None else int(patched_rank) - int(base_rank)
    row["rank_improved"] = bool(base_rank is not None and patched_rank is not None and int(patched_rank) < int(base_rank))
    row["base_relation_winner"] = base.get("relation_winner")
    row["patched_relation_winner"] = patched.get("relation_winner")
    row["base_target_is_relation_winner"] = base.get("target_is_relation_winner")
    row["patched_target_is_relation_winner"] = patched.get("target_is_relation_winner")
    row["new_relation_winner_target"] = bool((not base.get("target_is_relation_winner")) and patched.get("target_is_relation_winner"))
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


def bridge_rows(by_relation_language_condition: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in by_relation_language_condition:
        groups[(row.get("relation"), row.get("language_pair"), row.get("alpha"))].append(row)
    out = []
    for (relation, language_pair, alpha), items in groups.items():
        by_cond = {row.get("condition"): row for row in items}
        controls = [by_cond[name] for name in CONTROL_CONDITIONS if name in by_cond]
        control_best = None if not controls else max(finite(row.get("boundary_margin_delta_mean"), -999.0) for row in controls)
        for condition in sorted(TARGET_CONDITIONS):
            row = by_cond.get(condition)
            if not row:
                continue
            boundary_delta = finite(row.get("boundary_margin_delta_mean"), -999.0)
            relation_delta = finite(row.get("relation_margin_delta_mean"), -999.0)
            out.append(
                {
                    "relation": relation,
                    "language_pair": language_pair,
                    "alpha": alpha,
                    "condition": condition,
                    "rows": row.get("rows"),
                    "relation_margin_delta_mean": row.get("relation_margin_delta_mean"),
                    "boundary_margin_delta_mean": row.get("boundary_margin_delta_mean"),
                    "period_margin_delta_mean": row.get("period_margin_delta_mean"),
                    "eos_margin_delta_mean": row.get("eos_margin_delta_mean"),
                    "boundary_logit_delta_mean": row.get("boundary_logit_delta_mean"),
                    "control_best_boundary_margin_delta": control_best,
                    "boundary_bridge_gain_vs_best_control": None if control_best is None else float(boundary_delta - control_best),
                    "joint_bridge_score": None if control_best is None else float(min(boundary_delta, relation_delta, boundary_delta - control_best)),
                    "rank_improved": row.get("rank_improved"),
                    "new_relation_winner_target": row.get("new_relation_winner_target"),
                }
            )
    out.sort(
        key=lambda row: (
            finite(row.get("joint_bridge_score"), -999.0),
            finite(row.get("boundary_bridge_gain_vs_best_control"), -999.0),
            finite(row.get("boundary_margin_delta_mean"), -999.0),
        ),
        reverse=True,
    )
    return out


def evidence_label(bridges: list[dict[str, Any]]) -> str:
    positives = [
        row
        for row in bridges
        if row.get("condition") == "specific_direction"
        and finite(row.get("relation_margin_delta_mean"), 0.0) > 0
        and finite(row.get("boundary_margin_delta_mean"), 0.0) > 0
        and finite(row.get("boundary_bridge_gain_vs_best_control"), -999.0) > 0.02
    ]
    rels = {str(row.get("relation")) for row in positives}
    bilingual = {str(row.get("relation")) for row in positives if str(row.get("language_pair")) in {"en->zh", "zh->en"}}
    if len(rels) >= 2 and bilingual:
        return "semantic_boundary_bridge_positive"
    if positives:
        return "partial_semantic_boundary_bridge_positive"
    return "semantic_boundary_bridge_weak_or_controlled"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = p939.build_samples(args)
    selected_pairs, selected_phase939_rows = selected_relation_language_pairs(args)
    hidden_by_relation = p938.phase937_best_hidden(args.model, args.phase937_round)
    if not hidden_by_relation:
        hidden_by_relation = {relation: -1 for relation in parse_csv(args.relations)}
    dry_payload = {
        "phase": PHASE,
        "title": "Semantic Boundary Bridge Audit",
        "model": args.model,
        "sample_count": len(samples),
        "selected_relation_language_pairs": sorted([f"{r}:{p}" for r, p in selected_pairs]),
        "selected_phase939_rows": selected_phase939_rows,
        "hidden_by_relation": hidden_by_relation,
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase940_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase940_{args.model}_rows.jsonl", [])
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
        auto_indices = p937.auto_hidden_indices(model)
        hidden_by_relation = {
            rel: (auto_indices[len(auto_indices) // 2] if int(idx) < 0 else int(idx))
            for rel, idx in hidden_by_relation.items()
        }
        hidden_indices = sorted(set(hidden_by_relation.values()))
        vectors = p938.forward_vectors(model, tokenizer, device, samples, hidden_indices, int(args.batch_size))
        baseline_logits = p938.forward_logits(model, tokenizer, device, samples, int(args.batch_size))
        direction_specs = p939.build_direction_specs(samples, vectors, hidden_by_relation, int(args.min_train_per_label))
        selected_specs = filter_specs(direction_specs, selected_pairs)
        if int(args.max_direction_specs) > 0:
            selected_specs = selected_specs[: int(args.max_direction_specs)]
        alphas = [float(x) for x in parse_csv(args.alphas)]
        labels_by_relation = {relation: p938.relation_labels(samples, relation) for relation in hidden_by_relation}
        token_maps = {relation: p938.label_token_map(tokenizer, labels) for relation, labels in labels_by_relation.items()}
        boundary_groups = boundary_token_groups(tokenizer)
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for sample in samples:
            relation = str(sample["relation"])
            baseline_metrics[p938.sample_key(sample)] = boundary_metrics(
                tokenizer,
                baseline_logits[p938.sample_key(sample)],
                str(sample["target_label"]),
                token_maps[relation],
                boundary_groups,
            )

        for spec_idx, spec in enumerate(selected_specs, 1):
            relation = str(spec["relation"])
            label_tokens = token_maps[relation]
            for sample in spec["test_samples"]:
                base = baseline_metrics[p938.sample_key(sample)]
                rows.append(make_row(args.model, sample, spec, "baseline", None, base, base))
            conditions = [
                ("specific_direction", spec.get("specific_direction")),
                ("target_direction", spec.get("target_direction")),
                ("random_same_norm", spec.get("random_direction")),
                ("wrong_mean_direction", spec.get("wrong_mean_direction")),
                ("template_shift_same_norm", spec.get("template_shift_same_norm")),
                ("negative_target_direction", -spec["target_direction"]),
            ]
            for condition, direction in conditions:
                if direction is None:
                    continue
                for alpha in alphas:
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
                        patched = boundary_metrics(
                            tokenizer,
                            patched_logits[p938.sample_key(sample)],
                            str(sample["target_label"]),
                            label_tokens,
                            boundary_groups,
                        )
                        rows.append(make_row(args.model, sample, spec, condition, float(alpha), base, patched))
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

    by_condition = summarize_by(rows, ["condition", "alpha"])
    by_relation_language_condition = summarize_by(rows, ["relation", "language_pair", "condition", "alpha"])
    by_language_condition = summarize_by(rows, ["language_pair", "condition", "alpha"])
    bridges = bridge_rows(by_relation_language_condition)
    evidence = evidence_label(bridges)
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
        "bridge_rows": bridges,
        "evidence_label": evidence,
        "boundary": "first-token semantic direction versus punctuation/eos/protocol boundary margins; not Phase936 channel-level gear closure",
    }
    write_json(out_dir / f"phase940_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase940_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": evidence,
                "selected_specs": len(selected_specs),
                "top_bridge_rows": bridges[:16],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase940_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    condition_rows = []
    language_rows = []
    bridge_summary = []
    for summary in summaries:
        model = summary.get("model")
        for row in summary.get("by_condition") or []:
            item = dict(row)
            item["model"] = model
            condition_rows.append(item)
        for row in summary.get("by_language_condition") or []:
            item = dict(row)
            item["model"] = model
            language_rows.append(item)
        for row in summary.get("bridge_rows") or []:
            item = dict(row)
            item["model"] = model
            bridge_summary.append(item)
    bridge_summary.sort(
        key=lambda row: (
            finite(row.get("joint_bridge_score"), -999.0),
            finite(row.get("boundary_bridge_gain_vs_best_control"), -999.0),
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
        "language_condition_rows": language_rows,
        "top_bridge_rows": bridge_summary[:180],
    }
    write_json(out_dir / "phase940_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase940_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 940 semantic boundary bridge audit", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Condition Rows", ""]
    lines.append("| model | condition | alpha | rows | relation margin | boundary margin | period margin | eos margin | boundary logit |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("condition_rows") or []:
        lines.append(
            "| {model} | {condition} | {alpha} | {rows} | {relation_margin_delta_mean} | {boundary_margin_delta_mean} | {period_margin_delta_mean} | {eos_margin_delta_mean} | {boundary_logit_delta_mean} |".format(
                **row
            )
        )
    lines += ["", "## Language Pair Rows", ""]
    lines.append("| model | pair | condition | alpha | rows | relation margin | boundary margin | bridge gain |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("language_condition_rows") or []:
        lines.append(
            "| {model} | {language_pair} | {condition} | {alpha} | {rows} | {relation_margin_delta_mean} | {boundary_margin_delta_mean} | {boundary_bridge_gain_vs_best_control} |".format(
                **{**row, "boundary_bridge_gain_vs_best_control": row.get("boundary_bridge_gain_vs_best_control")}
            )
        )
    lines += ["", "## Top Bridge Rows", ""]
    lines.append("| model | relation | pair | condition | rows | relation margin | boundary margin | control best | bridge gain | joint score |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_bridge_rows") or []:
        lines.append(
            "| {model} | {relation} | {language_pair} | {condition} | {rows} | {relation_margin_delta_mean} | {boundary_margin_delta_mean} | {control_best_boundary_margin_delta} | {boundary_bridge_gain_vs_best_control} | {joint_bridge_score} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="semantic_boundary_bridge_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=6)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--min-specific-margin", type=float, default=0.05)
    parser.add_argument("--min-specific-gain", type=float, default=0.05)
    parser.add_argument("--max-direction-specs", type=int, default=0)
    parser.add_argument("--alphas", default="1.0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
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
