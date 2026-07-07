#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402
import phase219_state_write_mlp_causal_validation as p219  # noqa: E402
import phase227_token_trigger_gateup_decomposition as p227  # noqa: E402
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402
import phase229_readout_regime_selection_atlas as p229  # noqa: E402


PHASE = 233
SOURCE_PHASE = 232
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase233_competitor_source_hook_causal_validation")


SPECS = {
    "qwen3": [
        {
            "spec_id": "qwen3_explain_competitor_hook",
            "pattern_id": "answer_explain",
            "source_layers": [29],
            "observe_layers": [29, 31, 33],
            "candidate_variants": ["no_answer_anchor", "because_removed", "repeat_instruction"],
            "candidate_regimes": ["because_reason", "period_stop", "answer_boundary"],
        }
    ],
    "glm4": [
        {
            "spec_id": "glm4_repeat_competitor_hook",
            "pattern_id": "answer_repeat",
            "source_layers": [30],
            "observe_layers": [28, 30, 32],
            "candidate_variants": ["no_answer_anchor", "explain_instruction", "no_instruction"],
            "candidate_regimes": ["for_continuation", "newline_boundary", "because_reason"],
        }
    ],
    "deepseek7b": [
        {
            "spec_id": "deepseek7b_explain_competitor_hook",
            "pattern_id": "answer_explain",
            "source_layers": [24],
            "observe_layers": [24, 26, 27],
            "candidate_variants": ["no_instruction", "short_answer_instruction", "because_removed"],
            "candidate_regimes": ["be_continuation", "the_continuation", "prose", "echo"],
        }
    ],
}

PATCH_COMPONENTS = ["product", "gate_up_pair", "down_out"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def mean_vec(vecs: list[torch.Tensor]) -> torch.Tensor | None:
    if not vecs:
        return None
    return torch.stack(vecs, dim=0).mean(dim=0)


def rank_regimes(metric: dict[str, Any]) -> list[tuple[str, float]]:
    scores = metric.get("regime_scores") or {}
    pairs = [(str(k), finite_float(v)) for k, v in scores.items() if str(k) != "target"]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


def component_patch_spec(component: str, layer_idx: int, alpha: float, delta_map: dict[str, torch.Tensor]) -> dict[str, Any] | None:
    if component == "product" and "product" in delta_map:
        return {"layer_idx": layer_idx, "component": "product", "alpha": alpha, "product_vec": -delta_map["product"]}
    if component == "down_out" and "down_out" in delta_map:
        return {"layer_idx": layer_idx, "component": "down_out", "alpha": alpha, "down_out_vec": -delta_map["down_out"]}
    if component == "gate_up_pair" and "gate" in delta_map and "up" in delta_map:
        return {
            "layer_idx": layer_idx,
            "component": "gate_up_pair",
            "alpha": alpha,
            "gate_vec": -delta_map["gate"],
            "up_vec": -delta_map["up"],
        }
    return None


def collect_delta_rows(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    base_groups: dict[str, list[int]],
    model_name: str,
    spec: dict[str, Any],
    source_group: str,
    rows: list[dict[str, Any]],
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, torch.Tensor]]]:
    raw_rows: list[dict[str, Any]] = []
    delta_bucket: dict[tuple[str, int, int], dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    variants = list(spec["candidate_variants"])
    regimes = list(spec["candidate_regimes"])
    for sample in rows:
        prompt_variants = p227.prompt_variants(str(sample.get("prompt") or ""), str(spec["pattern_id"]))
        for variant in variants:
            if variant not in prompt_variants:
                continue
            for step in range(1, int(max_steps) + 1):
                full_text = p227.prefix_variant(sample, prompt_variants["full"], int(step))
                var_text = p227.prefix_variant(sample, prompt_variants[variant], int(step))
                full_internal, _full_hidden, full_logits = p228.capture_internal(
                    model, tokenizer, device, full_text, spec["source_layers"], spec["observe_layers"]
                )
                var_internal, _var_hidden, var_logits = p228.capture_internal(
                    model, tokenizer, device, var_text, spec["source_layers"], spec["observe_layers"]
                )
                full_metric = p229.regime_metric(tokenizer, full_logits, sample, base_groups)
                var_metric = p229.regime_metric(tokenizer, var_logits, sample, base_groups)
                full_ranked = rank_regimes(full_metric)
                var_ranked = rank_regimes(var_metric)
                for layer_idx in spec["source_layers"]:
                    f_parts = full_internal.get(int(layer_idx), {})
                    v_parts = var_internal.get(int(layer_idx), {})
                    for component in ["gate", "up", "product", "down_out"]:
                        if component in f_parts and component in v_parts and f_parts[component].ndim == 1:
                            delta_bucket[(variant, int(step), int(layer_idx))][component].append(v_parts[component] - f_parts[component])
                for regime in regimes:
                    delta = finite_float((var_metric.get("regime_scores") or {}).get(regime)) - finite_float(
                        (full_metric.get("regime_scores") or {}).get(regime)
                    )
                    target_delta = finite_float(var_metric.get("target_logit")) - finite_float(full_metric.get("target_logit"))
                    raw_rows.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase233_prompt_source_observation_row",
                            "model": model_name,
                            "spec_id": spec["spec_id"],
                            "pattern_id": spec["pattern_id"],
                            "source_group": source_group,
                            "trajectory_id": sample.get("trajectory_id"),
                            "variant": variant,
                            "step": int(step),
                            "regime": regime,
                            "full_winning_regime": full_metric.get("winning_regime"),
                            "variant_winning_regime": var_metric.get("winning_regime"),
                            "full_top_token": full_metric.get("top_token"),
                            "variant_top_token": var_metric.get("top_token"),
                            "full_first_competitor": full_ranked[0][0] if full_ranked else "",
                            "variant_first_competitor": var_ranked[0][0] if var_ranked else "",
                            "variant_second_competitor": var_ranked[1][0] if len(var_ranked) > 1 else "",
                            "target_delta": target_delta,
                            "regime_delta": delta,
                            "competitor_minus_target_delta": delta - target_delta,
                            "winner_switched": full_metric.get("winning_regime") != var_metric.get("winning_regime"),
                            "target_margin_delta": finite_float(var_metric.get("target_margin_vs_winner"))
                            - finite_float(full_metric.get("target_margin_vs_winner")),
                            "variant_target_margin": finite_float(var_metric.get("target_margin_vs_winner")),
                        }
                    )
                del full_internal, var_internal, full_logits, var_logits
    mean_delta: dict[tuple[str, int, int], dict[str, torch.Tensor]] = defaultdict(dict)
    for key, part_map in delta_bucket.items():
        for component, vecs in part_map.items():
            vec = mean_vec(vecs)
            if vec is not None:
                mean_delta[key][component] = vec
    return raw_rows, mean_delta


def run_suppression_rows(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    base_groups: dict[str, list[int]],
    model_name: str,
    spec: dict[str, Any],
    source_group: str,
    rows: list[dict[str, Any]],
    mean_delta: dict[tuple[str, int, int], dict[str, torch.Tensor]],
    max_steps: int,
    alphas: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    variants = list(spec["candidate_variants"])
    regimes = list(spec["candidate_regimes"])
    for sample in rows:
        prompt_variants = p227.prompt_variants(str(sample.get("prompt") or ""), str(spec["pattern_id"]))
        for variant in variants:
            if variant not in prompt_variants:
                continue
            for step in range(1, int(max_steps) + 1):
                var_text = p227.prefix_variant(sample, prompt_variants[variant], int(step))
                var_logits = p228.run_logits(model, tokenizer, device, var_text)
                var_metric = p229.regime_metric(tokenizer, var_logits, sample, base_groups)
                var_ranked = rank_regimes(var_metric)
                for layer_idx in spec["source_layers"]:
                    delta_map = mean_delta.get((variant, int(step), int(layer_idx)), {})
                    for component in PATCH_COMPONENTS:
                        for alpha in alphas:
                            patch_spec = component_patch_spec(component, int(layer_idx), float(alpha), delta_map)
                            if patch_spec is None:
                                continue
                            suppressed_logits = p228.run_logits(model, tokenizer, device, var_text, patch_spec)
                            suppressed_metric = p229.regime_metric(tokenizer, suppressed_logits, sample, base_groups)
                            suppressed_ranked = rank_regimes(suppressed_metric)
                            for regime in regimes:
                                before = finite_float((var_metric.get("regime_scores") or {}).get(regime))
                                after = finite_float((suppressed_metric.get("regime_scores") or {}).get(regime))
                                target_before = finite_float(var_metric.get("target_logit"))
                                target_after = finite_float(suppressed_metric.get("target_logit"))
                                out.append(
                                    {
                                        "phase": PHASE,
                                        "source_phase": SOURCE_PHASE,
                                        "row_kind": "phase233_source_suppression_row",
                                        "model": model_name,
                                        "spec_id": spec["spec_id"],
                                        "pattern_id": spec["pattern_id"],
                                        "source_group": source_group,
                                        "trajectory_id": sample.get("trajectory_id"),
                                        "variant": variant,
                                        "step": int(step),
                                        "source_layer": int(layer_idx),
                                        "component": component,
                                        "alpha": float(alpha),
                                        "regime": regime,
                                        "variant_winning_regime": var_metric.get("winning_regime"),
                                        "suppressed_winning_regime": suppressed_metric.get("winning_regime"),
                                        "variant_top_token": var_metric.get("top_token"),
                                        "suppressed_top_token": suppressed_metric.get("top_token"),
                                        "variant_first_competitor": var_ranked[0][0] if var_ranked else "",
                                        "variant_second_competitor": var_ranked[1][0] if len(var_ranked) > 1 else "",
                                        "suppressed_first_competitor": suppressed_ranked[0][0] if suppressed_ranked else "",
                                        "suppressed_second_competitor": suppressed_ranked[1][0] if len(suppressed_ranked) > 1 else "",
                                        "regime_suppression_delta": after - before,
                                        "target_delta_after_suppression": target_after - target_before,
                                        "competitor_minus_target_suppression_delta": (after - before) - (target_after - target_before),
                                        "target_margin_delta_after_suppression": finite_float(
                                            suppressed_metric.get("target_margin_vs_winner")
                                        )
                                        - finite_float(var_metric.get("target_margin_vs_winner")),
                                        "suppressed_target_margin": finite_float(suppressed_metric.get("target_margin_vs_winner")),
                                        "winner_changed_by_suppression": var_metric.get("winning_regime")
                                        != suppressed_metric.get("winning_regime"),
                                    }
                                )
                            del suppressed_logits
                del var_logits
    return out


def summarize_prompt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("model"), row.get("spec_id"), row.get("source_group"), row.get("variant"), row.get("step"), row.get("regime"))].append(row)
    out = []
    for key, items in buckets.items():
        model, spec_id, group, variant, step, regime = key
        out.append(
            {
                "model": model,
                "spec_id": spec_id,
                "source_group": group,
                "variant": variant,
                "step": int(step),
                "regime": regime,
                "rows": len(items),
                "mean_target_delta": mean(finite_float(x.get("target_delta")) for x in items),
                "mean_regime_delta": mean(finite_float(x.get("regime_delta")) for x in items),
                "mean_competitor_minus_target_delta": mean(finite_float(x.get("competitor_minus_target_delta")) for x in items),
                "winner_switch_rate": sum(1 for x in items if x.get("winner_switched")) / len(items) if items else 0.0,
                "variant_winners": dict(Counter(str(x.get("variant_winning_regime")) for x in items).most_common(6)),
                "variant_top_tokens": dict(Counter(str(x.get("variant_top_token")) for x in items).most_common(6)),
            }
        )
    out.sort(key=lambda row: (float(row.get("winner_switch_rate") or 0.0), float(row.get("mean_competitor_minus_target_delta") or 0.0)), reverse=True)
    return out


def summarize_suppression(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row.get("model"),
                row.get("spec_id"),
                row.get("source_group"),
                row.get("variant"),
                row.get("step"),
                row.get("component"),
                row.get("alpha"),
                row.get("regime"),
            )
        ].append(row)
    out = []
    for key, items in buckets.items():
        model, spec_id, group, variant, step, component, alpha, regime = key
        reduced = [x for x in items if finite_float(x.get("regime_suppression_delta")) < 0.0]
        target_helped = [x for x in items if finite_float(x.get("target_margin_delta_after_suppression")) > 0.0]
        out.append(
            {
                "model": model,
                "spec_id": spec_id,
                "source_group": group,
                "variant": variant,
                "step": int(step),
                "component": component,
                "alpha": float(alpha),
                "regime": regime,
                "rows": len(items),
                "regime_reduction_rate": len(reduced) / len(items) if items else 0.0,
                "target_margin_help_rate": len(target_helped) / len(items) if items else 0.0,
                "winner_changed_rate": sum(1 for x in items if x.get("winner_changed_by_suppression")) / len(items) if items else 0.0,
                "mean_regime_suppression_delta": mean(finite_float(x.get("regime_suppression_delta")) for x in items),
                "mean_target_delta_after_suppression": mean(finite_float(x.get("target_delta_after_suppression")) for x in items),
                "mean_target_margin_delta_after_suppression": mean(
                    finite_float(x.get("target_margin_delta_after_suppression")) for x in items
                ),
                "suppressed_winners": dict(Counter(str(x.get("suppressed_winning_regime")) for x in items).most_common(6)),
                "suppressed_top_tokens": dict(Counter(str(x.get("suppressed_top_token")) for x in items).most_common(6)),
            }
        )
    out.sort(
        key=lambda row: (
            float(row.get("target_margin_help_rate") or 0.0),
            -float(row.get("mean_regime_suppression_delta") or 0.0),
            float(row.get("mean_target_margin_delta_after_suppression") or 0.0),
        ),
        reverse=True,
    )
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    observation_rows: list[dict[str, Any]] = []
    suppression_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_groups = p201.token_groups(tokenizer)
        source_rows_all = load_rows(args.model, args.phase210_round)
        alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]
        for spec in SPECS[args.model]:
            success_rows, drift_rows = p219.select_rows(source_rows_all, str(spec["pattern_id"]), int(args.max_filter_rows))
            eval_pairs = [
                ("success", success_rows[: int(args.max_eval_rows)]),
                ("drift", drift_rows[: int(args.max_eval_rows)]),
            ]
            filter_rows.append(
                {
                    "phase": PHASE,
                    "row_kind": "phase233_source_row_count",
                    "model": args.model,
                    "spec_id": spec["spec_id"],
                    "pattern_id": spec["pattern_id"],
                    "success_rows": len(eval_pairs[0][1]),
                    "drift_rows": len(eval_pairs[1][1]),
                }
            )
            for source_group, rows in eval_pairs:
                if not rows:
                    continue
                obs, mean_delta = collect_delta_rows(
                    model,
                    tokenizer,
                    device,
                    base_groups,
                    args.model,
                    spec,
                    source_group,
                    rows,
                    int(args.max_steps),
                )
                observation_rows.extend(obs)
                suppression_rows.extend(
                    run_suppression_rows(
                        model,
                        tokenizer,
                        device,
                        base_groups,
                        args.model,
                        spec,
                        source_group,
                        rows,
                        mean_delta,
                        int(args.max_steps),
                        alphas,
                    )
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        prompt_summary = summarize_prompt(observation_rows)
        suppression_summary = summarize_suppression(suppression_rows)
        payload = {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "title": "Competitor source hook causal validation",
            "status": "complete",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "filter_rows": filter_rows,
            "observation_rows": len(observation_rows),
            "suppression_rows": len(suppression_rows),
            "top_prompt_summary": prompt_summary[:80],
            "top_suppression_summary": suppression_summary[:100],
        }
        write_json(out_dir / f"phase233_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase233_{args.model}_filter_rows.jsonl", filter_rows)
        write_jsonl(out_dir / f"phase233_{args.model}_observation_rows.jsonl", observation_rows)
        write_jsonl(out_dir / f"phase233_{args.model}_suppression_rows.jsonl", suppression_rows)
        write_jsonl(out_dir / f"phase233_{args.model}_prompt_summary_rows.jsonl", prompt_summary)
        write_jsonl(out_dir / f"phase233_{args.model}_suppression_summary_rows.jsonl", suppression_summary)
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "model": args.model,
                    "status": "complete",
                    "observation_rows": len(observation_rows),
                    "suppression_rows": len(suppression_rows),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return payload
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase233_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    prompt_rows: list[dict[str, Any]] = []
    suppression_rows: list[dict[str, Any]] = []
    for model in MODELS:
        prompt_rows.extend(iter_jsonl(out_dir / f"phase233_{model}_prompt_summary_rows.jsonl"))
        suppression_rows.extend(iter_jsonl(out_dir / f"phase233_{model}_suppression_summary_rows.jsonl"))
    prompt_rows.sort(key=lambda row: (float(row.get("winner_switch_rate") or 0.0), float(row.get("mean_competitor_minus_target_delta") or 0.0)), reverse=True)
    suppression_rows.sort(
        key=lambda row: (
            float(row.get("target_margin_help_rate") or 0.0),
            -float(row.get("mean_regime_suppression_delta") or 0.0),
            float(row.get("mean_target_margin_delta_after_suppression") or 0.0),
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model competitor source hook causal validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "observation_rows": sum(int(x.get("observation_rows") or 0) for x in summaries),
        "suppression_rows": sum(int(x.get("suppression_rows") or 0) for x in summaries),
        "top_prompt_summary": prompt_rows[:120],
        "top_suppression_summary": suppression_rows[:160],
    }
    write_json(out_dir / "phase233_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase233_cross_model_summary.md", payload)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "models": payload["models"],
                "observation_rows": payload["observation_rows"],
                "suppression_rows": payload["suppression_rows"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 233 competitor source hook causal validation", ""]
    for key in ["observation_rows", "suppression_rows"]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(
        [
            "",
            "## Prompt Source Summary",
            "",
            "| model | group | variant | step | regime | rows | switch rate | target delta | regime delta | comp-target | winners | top tokens |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["top_prompt_summary"][:80]:
        lines.append(
            f"| {row.get('model')} | {row.get('source_group')} | {row.get('variant')} | {row.get('step')} | {row.get('regime')} | {row.get('rows')} | "
            f"{finite_float(row.get('winner_switch_rate')):.4f} | {finite_float(row.get('mean_target_delta')):.4f} | "
            f"{finite_float(row.get('mean_regime_delta')):.4f} | {finite_float(row.get('mean_competitor_minus_target_delta')):.4f} | "
            f"{row.get('variant_winners')} | {row.get('variant_top_tokens')} |"
        )
    lines.extend(
        [
            "",
            "## Suppression Summary",
            "",
            "| model | group | variant | step | component | alpha | regime | rows | reduce rate | margin help | winner change | regime delta | target delta | margin delta | winners |",
            "| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["top_suppression_summary"][:100]:
        lines.append(
            f"| {row.get('model')} | {row.get('source_group')} | {row.get('variant')} | {row.get('step')} | {row.get('component')} | "
            f"{finite_float(row.get('alpha')):.2f} | {row.get('regime')} | {row.get('rows')} | "
            f"{finite_float(row.get('regime_reduction_rate')):.4f} | {finite_float(row.get('target_margin_help_rate')):.4f} | "
            f"{finite_float(row.get('winner_changed_rate')):.4f} | {finite_float(row.get('mean_regime_suppression_delta')):.4f} | "
            f"{finite_float(row.get('mean_target_delta_after_suppression')):.4f} | {finite_float(row.get('mean_target_margin_delta_after_suppression')):.4f} | "
            f"{row.get('suppressed_winners')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase233 competitor source hook causal validation")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="competitor_source_hook_causal_validation")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=32)
    parser.add_argument("--max-eval-rows", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    args = parser.parse_args()
    if not args.summarize and not args.model:
        parser.error("--model is required unless --summarize is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize_round(args.round_name)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
