#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402
import phase199_l4_edge_natural_gate_rollout_audit as p199  # noqa: E402
import phase200_protocol_gated_rollout_repair_audit as p200  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402


PHASE = 202
SOURCE_PHASE = 201
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase202_stop_prose_rollout_repair_audit")
PHASE201_ROOT = Path("tests/result/phase201_stop_prose_component_atlas")


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


def mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            vals.append(fval)
    return None if not vals else float(sum(vals) / len(vals))


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def load_phase201_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE201_ROOT / args.phase201_round / f"phase201_{args.model}_summary.json"
    summary = read_json(path)
    rows = [dict(row) for row in summary.get("causal_summary_rows") or []]
    rows = [row for row in rows if str(row.get("condition")) in {"ablate", "boost"}]
    if args.model == "glm4":
        rows = [row for row in rows if finite(row.get("causal_candidate_score"), -999.0) > float(args.min_candidate_score)]
    else:
        rows.sort(key=lambda row: finite(row.get("causal_candidate_score"), -999.0), reverse=True)
    dedup = []
    seen = set()
    for row in sorted(rows, key=lambda r: finite(r.get("causal_candidate_score"), -999.0), reverse=True):
        key = (
            row.get("model"),
            row.get("candidate_type"),
            row.get("relation"),
            row.get("language_pair"),
            row.get("prompt_protocol"),
            int(row.get("layer_idx")),
            int(row.get("channel_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        row["candidate_key"] = (
            f"{row.get('model')}|{row.get('candidate_type')}|{row.get('relation')}|{row.get('language_pair')}|"
            f"{row.get('prompt_protocol')}|L{row.get('layer_idx')}|c{row.get('channel_id')}"
        )
        dedup.append(row)
        if len(dedup) >= int(args.max_candidates):
            break
    return dedup


def post_answer_prompt(sample: dict[str, Any], protocol: str) -> str:
    prompt = p200.protocol_prompt(sample, protocol).rstrip()
    target = str(sample.get("target_label") or "").strip()
    return f"{prompt} {target}".strip()


def continuation_label(generated: str, sample: dict[str, Any]) -> dict[str, Any]:
    raw = str(generated or "")
    stripped = raw.strip()
    lower = stripped.lower()
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", stripped)
    obj = str(sample.get("object") or "").lower()
    target = str(sample.get("target_label") or "").lower()
    prose_markers = ["because", "usually", "which", "that", "what", "answer", "typically", " and ", " it ", " the "]
    raw_prose = any(marker in f" {lower} " for marker in prose_markers)
    raw_echo = bool(obj and obj in lower)
    raw_target_echo = bool(target and len(words) > 1 and target in lower)
    raw_question_drift = any(marker in lower for marker in ["what is", "?", ":", "\n", "。", "，"])
    starts_stop = stripped == "" or stripped.startswith((".", "。", "\n"))
    post_answer_stable = bool(starts_stop and not raw_prose and not raw_echo and not raw_question_drift and len(words) <= 2)
    return {
        "post_answer_starts_stop": starts_stop,
        "post_answer_raw_prose_continuation": raw_prose,
        "post_answer_raw_object_echo": raw_echo,
        "post_answer_raw_target_echo": raw_target_echo,
        "post_answer_raw_question_drift": raw_question_drift,
        "post_answer_word_count": len(words),
        "post_answer_stable": post_answer_stable,
    }


def classify_row(rollout_mode: str, generated: str, sample: dict[str, Any]) -> dict[str, Any]:
    if rollout_mode == "natural":
        rollout = p199.classify_rollout(generated, sample)
        return {
            **rollout,
            "raw_prose_continuation": bool(p199.strict_protocol_drift(generated)),
            "raw_object_echo": bool(rollout.get("rollout_object_echo")),
            "strict_rollout_stable": bool(rollout.get("long_rollout_stable")),
        }
    post = continuation_label(generated, sample)
    return {
        **post,
        "rollout_clear_answer_class": True,
        "protocol_drift": not post["post_answer_stable"],
        "rollout_object_echo": post["post_answer_raw_object_echo"],
        "raw_prose_continuation": post["post_answer_raw_prose_continuation"],
        "raw_object_echo": post["post_answer_raw_object_echo"],
        "strict_rollout_stable": post["post_answer_stable"],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "clear": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "drift": sum(1 for row in rows if row.get("protocol_drift")),
        "echo": sum(1 for row in rows if row.get("raw_object_echo") or row.get("rollout_object_echo")),
        "prose": sum(1 for row in rows if row.get("raw_prose_continuation")),
        "stable": sum(1 for row in rows if row.get("strict_rollout_stable")),
        "post_answer_starts_stop": sum(1 for row in rows if row.get("post_answer_starts_stop")),
        "labels": dict(Counter(str(row.get("rollout_label")) for row in rows if row.get("rollout_label") is not None)),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(summarize_rows(items))
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def build_repair_eval(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["candidate_key", "rollout_mode", "prompt_protocol"]
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in condition_rows:
        grouped[tuple(row.get(key) for key in keys)][str(row.get("condition"))] = row
    out = []
    for key_tuple, by_condition in grouped.items():
        base = by_condition.get("baseline") or {}
        ablate = by_condition.get("ablate") or {}
        boost = by_condition.get("boost") or {}
        for condition, row in [("ablate", ablate), ("boost", boost)]:
            if not row:
                continue
            out.append(
                {
                    **{key: value for key, value in zip(keys, key_tuple)},
                    "condition": condition,
                    "baseline_rows": base.get("rows", 0),
                    "rows": row.get("rows", 0),
                    "stable_delta": int(row.get("stable", 0)) - int(base.get("stable", 0)),
                    "drift_delta": int(row.get("drift", 0)) - int(base.get("drift", 0)),
                    "echo_delta": int(row.get("echo", 0)) - int(base.get("echo", 0)),
                    "prose_delta": int(row.get("prose", 0)) - int(base.get("prose", 0)),
                    "clear_delta": int(row.get("clear", 0)) - int(base.get("clear", 0)),
                    "repair_effective": bool(
                        int(row.get("stable", 0)) > int(base.get("stable", 0))
                        and int(row.get("drift", 0)) < int(base.get("drift", 0))
                        and int(row.get("clear", 0)) >= int(base.get("clear", 0))
                    ),
                }
            )
    out.sort(key=lambda row: (not row.get("repair_effective"), -int(row.get("stable_delta", 0)), int(row.get("drift_delta", 0))))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_phase201_candidates(args)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Stop-Prose Candidate Natural Rollout Repair Audit",
        "model": args.model,
        "selected_candidates": candidates,
        "prompt_protocols": parse_csv(args.prompt_protocols),
        "rollout_modes": parse_csv(args.rollout_modes),
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase202_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    rows: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    meta: dict[str, Any] = {}
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        protocols = parse_csv(args.prompt_protocols)
        rollout_modes = parse_csv(args.rollout_modes)
        for cand in candidates:
            relation = str(cand.get("relation"))
            language_pair = str(cand.get("language_pair"))
            base_samples = holdout_by_pair.get((relation, language_pair)) or []
            if int(args.max_samples_per_candidate) > 0:
                base_samples = base_samples[: int(args.max_samples_per_candidate)]
            layer_idx = int(cand.get("layer_idx"))
            channel_id = int(cand.get("channel_id"))
            for sample in base_samples:
                for protocol in protocols:
                    for rollout_mode in rollout_modes:
                        prompt = p200.protocol_prompt(sample, protocol) if rollout_mode == "natural" else post_answer_prompt(sample, protocol)
                        for condition, factor in [("baseline", None), ("ablate", 0.0), ("boost", float(args.boost_factor))]:
                            generated = p199.generate_with_channel_scale(
                                model,
                                tokenizer,
                                device,
                                prompt,
                                layer_idx,
                                channel_id if factor is not None else None,
                                factor,
                                int(args.max_new_tokens),
                            )
                            classified = classify_row(rollout_mode, generated, {**sample, "prompt": prompt})
                            rows.append(
                                {
                                    "phase": PHASE,
                                    "source_phase": SOURCE_PHASE,
                                    "row_kind": "phase202_stop_prose_rollout_repair_row",
                                    "model": args.model,
                                    "candidate_key": cand.get("candidate_key"),
                                    "candidate_type": cand.get("candidate_type"),
                                    "phase201_condition": cand.get("condition"),
                                    "phase201_score": cand.get("causal_candidate_score"),
                                    "phase201_stop_margin_delta": cand.get("stop_margin_delta_mean"),
                                    "phase201_prose_margin_delta": cand.get("prose_margin_delta_mean"),
                                    "phase201_echo_margin_delta": cand.get("echo_margin_delta_mean"),
                                    "relation": relation,
                                    "language_pair": language_pair,
                                    "prompt_protocol": protocol,
                                    "rollout_mode": rollout_mode,
                                    "layer_idx": layer_idx,
                                    "channel_id": channel_id,
                                    "condition": condition,
                                    "factor": factor,
                                    "sample_id": sample.get("sample_id"),
                                    "object": sample.get("object"),
                                    "target_label": sample.get("target_label"),
                                    "prompt": prompt,
                                    "generated": generated,
                                    **classified,
                                }
                            )
            log(f"{args.model}/{args.round_name}: {cand.get('candidate_key')} samples={len(base_samples)}")
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    condition_rows = summarize_by(
        rows,
        [
            "candidate_key",
            "model",
            "candidate_type",
            "relation",
            "language_pair",
            "prompt_protocol",
            "rollout_mode",
            "layer_idx",
            "channel_id",
            "condition",
        ],
    )
    repair_eval_rows = build_repair_eval(condition_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "rows": len(rows),
        "condition_rows": condition_rows,
        "repair_eval_rows": repair_eval_rows,
        "boundary": "Raw rollout audit for Phase201 stop/prose candidates; L6 only if strict rollout stable improves with drift not increasing.",
    }
    write_json(out_dir / f"phase202_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase202_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "rows": len(rows),
                "top_repair_eval_rows": repair_eval_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase202_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    condition_rows = []
    repair_eval_rows = []
    for summary in summaries:
        condition_rows.extend(dict(row) for row in summary.get("condition_rows") or [])
        repair_eval_rows.extend(dict(row) for row in summary.get("repair_eval_rows") or [])
    payload = {
        "schema_version": "phase202_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "model_summaries": summaries,
        "condition_rows": condition_rows,
        "repair_eval_rows": repair_eval_rows,
    }
    write_json(out_dir / "phase202_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase202_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 202 stop/prose rollout repair audit", ""]
    lines.append("| model | candidate | mode | protocol | condition | stable delta | drift delta | echo delta | prose delta | effective |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("repair_eval_rows") or []:
        lines.append(
            f"| {str(row.get('candidate_key')).split('|')[0]} | {row.get('candidate_key')} | {row.get('rollout_mode')} | "
            f"{row.get('prompt_protocol')} | {row.get('condition')} | {row.get('stable_delta')} | {row.get('drift_delta')} | "
            f"{row.get('echo_delta')} | {row.get('prose_delta')} | {row.get('repair_effective')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="stop_prose_rollout_repair_audit")
    parser.add_argument("--phase201-round", default="stop_prose_component_atlas")
    parser.add_argument("--phase944-round", default="activation_weighted_mlp_channel_causal_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase940-round", default="semantic_boundary_bridge_audit")
    parser.add_argument("--phase943-round", default="consensus_coordinate_component_mapping_audit")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=8)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--min-specific-margin", type=float, default=0.05)
    parser.add_argument("--min-specific-gain", type=float, default=0.05)
    parser.add_argument("--min-phase940-bridge-gain", type=float, default=0.02)
    parser.add_argument("--max-specs-per-pair", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--min-train-specs", type=int, default=4)
    parser.add_argument("--min-holdout-specs", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--min-candidate-score", type=float, default=0.01)
    parser.add_argument("--max-samples-per-candidate", type=int, default=16)
    parser.add_argument("--prompt-protocols", default="plain,short_answer,stop_explicit")
    parser.add_argument("--rollout-modes", default="natural,post_answer")
    parser.add_argument("--boost-factor", type=float, default=1.5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
