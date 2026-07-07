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

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase204_global_trajectory_stop_execution_atlas as p204  # noqa: E402
import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402
import phase219_state_write_mlp_causal_validation as p219  # noqa: E402
import phase222_statewrite_factor_competition as p222  # noqa: E402
import phase227_token_trigger_gateup_decomposition as p227  # noqa: E402
import phase228_module_tree_gateup_causal_validation as p228  # noqa: E402


PHASE = 229
SOURCE_PHASE = 228
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase229_readout_regime_selection_atlas")


SPECS = {
    "qwen3": [
        {"spec_id": "qwen3_explain_l29_readout_regime", "pattern_id": "answer_explain", "source_layers": [29], "observe_layers": [29, 31, 33]},
    ],
    "glm4": [
        {"spec_id": "glm4_repeat_l30_readout_regime", "pattern_id": "answer_repeat", "source_layers": [30], "observe_layers": [28, 30, 32]},
    ],
    "deepseek7b": [
        {"spec_id": "deepseek7b_explain_l24_readout_regime", "pattern_id": "answer_explain", "source_layers": [24], "observe_layers": [24, 26, 27]},
    ],
}


REGIME_TEXTS = {
    "then_continuation": [" Then", "Then", " then", "then"],
    "the_continuation": [" The", "The", " the", "the"],
    "for_continuation": [" For", "For", " for", "for"],
    "answer_boundary": [" Answer", "Answer", " answer", "answer"],
    "because_reason": [" Because", "Because", " because", "because"],
    "be_continuation": [" be", "be", " is", "is", " are", "are"],
    "comma_repeat": [",", ", "],
    "period_stop": [".", ".\n"],
    "colon_boundary": [":", ": "],
    "newline_boundary": ["\n", "\n\n"],
    "space_boundary": [" "],
}


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


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def token_ids_for_texts(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def regime_groups(tokenizer: Any, sample: dict[str, Any], base_groups: dict[str, list[int]]) -> dict[str, list[int]]:
    groups = {name: token_ids_for_texts(tokenizer, texts) for name, texts in REGIME_TEXTS.items()}
    groups["target"] = p938.first_token_candidates(tokenizer, str(sample.get("target_label") or ""))
    groups["prose"] = list(base_groups.get("prose") or [])
    groups["echo"] = p204.object_ids(tokenizer, sample)
    groups["stop"] = list(base_groups.get("stop") or [])
    return {name: sorted(set(int(x) for x in ids if x is not None)) for name, ids in groups.items() if ids}


def max_group_score(logits: torch.Tensor, ids: list[int]) -> tuple[float, int, str]:
    valid = [int(x) for x in ids if 0 <= int(x) < logits.numel()]
    if not valid:
        return -1e30, -1, ""
    idx = torch.tensor(valid, dtype=torch.long)
    values = logits[idx]
    pos = int(torch.argmax(values).item())
    token_id = int(idx[pos].item())
    return float(values[pos].item()), token_id, ""


def regime_metric(tokenizer: Any, logits: torch.Tensor, sample: dict[str, Any], base_groups: dict[str, list[int]]) -> dict[str, Any]:
    groups = regime_groups(tokenizer, sample, base_groups)
    scores: dict[str, float] = {}
    token_ids: dict[str, int] = {}
    for name, ids in groups.items():
        score, token_id, _ = max_group_score(logits, ids)
        scores[name] = score
        token_ids[name] = token_id
    top_id = int(torch.argmax(logits).item())
    top_token = p204.token_text(tokenizer, top_id)
    target_score = scores.get("target", -1e30)
    target_ids = groups.get("target") or []
    target_score2, target_id, target_rank = p204.max_score(logits, target_ids)
    non_target = {name: score for name, score in scores.items() if name != "target"}
    winning_regime = max(non_target.items(), key=lambda item: item[1])[0] if non_target else "none"
    winning_score = non_target.get(winning_regime, -1e30)
    return {
        "top_token_id": top_id,
        "top_token": top_token,
        "target_token_id": int(target_id),
        "target_token": p204.token_text(tokenizer, int(target_id)) if int(target_id) >= 0 else "",
        "target_logit": float(target_score2),
        "target_rank": int(target_rank),
        "winning_regime": winning_regime,
        "winning_regime_logit": float(winning_score),
        "target_margin_vs_winner": float(target_score - winning_score),
        "regime_scores": scores,
        "regime_token_ids": token_ids,
    }


def masked_vec(vec: torch.Tensor, channels: list[int] | None) -> torch.Tensor:
    if channels is None:
        return vec.clone()
    if not channels:
        return torch.zeros_like(vec)
    if vec.ndim != 1 or max(channels) >= vec.shape[0]:
        return vec.clone()
    out = torch.zeros_like(vec)
    idx = torch.tensor(channels, dtype=torch.long)
    out[idx] = vec[idx]
    return out


def build_patch_specs_for_step(
    selected: dict[str, dict[int, dict[int, list[int]]]],
    success_internal: dict[int, dict[int, dict[str, torch.Tensor]]],
    drift_internal: dict[int, dict[int, dict[str, torch.Tensor]]],
    source_group: str,
    step: int,
    layer_idx: int,
    alphas: list[float],
) -> list[dict[str, Any]]:
    s_map = success_internal.get(int(step), {}).get(int(layer_idx), {})
    d_map = drift_internal.get(int(step), {}).get(int(layer_idx), {})
    if not s_map or not d_map:
        return []
    sign = 1.0 if source_group == "drift" else -1.0
    delta = {name: sign * (s_map[name] - d_map[name]) for name in s_map.keys() & d_map.keys()}
    pos_channels = selected.get("pos", {}).get(int(step), {}).get(int(layer_idx), [])
    scopes = {
        "top16": pos_channels[:16],
        "top64": pos_channels[:64],
        "all": None,
    }
    specs: list[dict[str, Any]] = []
    for scope, channels in scopes.items():
        product = masked_vec(delta["product"], channels) if "product" in delta else None
        gate = masked_vec(delta["gate"], channels) if "gate" in delta else None
        up = masked_vec(delta["up"], channels) if "up" in delta else None
        for alpha in alphas:
            if product is not None:
                specs.append(
                    {
                        "variant": f"patch_product_{scope}_a{alpha:g}",
                        "layer_idx": int(layer_idx),
                        "component": "product",
                        "channel_scope": scope,
                        "alpha": float(alpha),
                        "product_vec": product,
                    }
                )
            if gate is not None and up is not None:
                specs.append(
                    {
                        "variant": f"patch_gate_up_pair_{scope}_a{alpha:g}",
                        "layer_idx": int(layer_idx),
                        "component": "gate_up_pair",
                        "channel_scope": scope,
                        "alpha": float(alpha),
                        "gate_vec": gate,
                        "up_vec": up,
                    }
                )
    return specs


def build_rows(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    base_groups: dict[str, list[int]],
    model_name: str,
    spec: dict[str, Any],
    source_group: str,
    source_rows: list[dict[str, Any]],
    selected: dict[str, dict[int, dict[int, list[int]]]],
    success_internal: dict[int, dict[int, dict[str, torch.Tensor]]],
    drift_internal: dict[int, dict[int, dict[str, torch.Tensor]]],
    max_steps: int,
    alphas: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source_layers = [int(x) for x in spec["source_layers"]]
    for sample in source_rows:
        variants = p227.prompt_variants(str(sample.get("prompt") or ""), str(spec["pattern_id"]))
        for step in range(1, int(max_steps) + 1):
            full_text = p227.prefix_variant(sample, variants["full"], int(step))
            base_logits = p228.run_logits(model, tokenizer, device, full_text)
            base_metric = regime_metric(tokenizer, base_logits, sample, base_groups)
            eval_items: list[tuple[str, str, torch.Tensor, dict[str, Any] | None]] = []
            for variant_name, prompt in variants.items():
                text = p227.prefix_variant(sample, prompt, int(step))
                logits = base_logits if variant_name == "full" else p228.run_logits(model, tokenizer, device, text)
                eval_items.append(("natural", variant_name, logits, None))
            for layer_idx in source_layers:
                for patch_spec in build_patch_specs_for_step(selected, success_internal, drift_internal, source_group, int(step), int(layer_idx), alphas):
                    logits = p228.run_logits(model, tokenizer, device, full_text, patch_spec)
                    eval_items.append(("patch", str(patch_spec["variant"]), logits, patch_spec))
            for intervention_type, variant_name, logits, patch_spec in eval_items:
                metric = regime_metric(tokenizer, logits, sample, base_groups)
                regime_delta = {
                    name: finite_float(metric["regime_scores"].get(name)) - finite_float(base_metric["regime_scores"].get(name))
                    for name in sorted(set(metric["regime_scores"]) | set(base_metric["regime_scores"]))
                }
                out.append(
                    {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase229_readout_regime_row",
                        "model": model_name,
                        "spec_id": spec["spec_id"],
                        "pattern_id": spec["pattern_id"],
                        "source_group": source_group,
                        "trajectory_id": sample.get("trajectory_id"),
                        "step": int(step),
                        "intervention_type": intervention_type,
                        "variant": variant_name,
                        "component": None if patch_spec is None else patch_spec.get("component"),
                        "channel_scope": None if patch_spec is None else patch_spec.get("channel_scope"),
                        "alpha": None if patch_spec is None else patch_spec.get("alpha"),
                        "top_token": metric["top_token"],
                        "base_top_token": base_metric["top_token"],
                        "top_token_changed": metric["top_token_id"] != base_metric["top_token_id"],
                        "target_rank": metric["target_rank"],
                        "base_target_rank": base_metric["target_rank"],
                        "rank_improve": int(base_metric["target_rank"]) - int(metric["target_rank"]),
                        "target_logit_delta": finite_float(metric["target_logit"]) - finite_float(base_metric["target_logit"]),
                        "target_margin_vs_winner": metric["target_margin_vs_winner"],
                        "base_target_margin_vs_winner": base_metric["target_margin_vs_winner"],
                        "margin_delta_vs_winner": finite_float(metric["target_margin_vs_winner"]) - finite_float(base_metric["target_margin_vs_winner"]),
                        "winning_regime": metric["winning_regime"],
                        "base_winning_regime": base_metric["winning_regime"],
                        "winning_regime_changed": metric["winning_regime"] != base_metric["winning_regime"],
                        "regime_delta": regime_delta,
                    }
                )
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row.get("spec_id"),
                row.get("source_group"),
                row.get("intervention_type"),
                row.get("variant"),
                row.get("component"),
                row.get("channel_scope"),
                row.get("alpha"),
                row.get("step"),
            )
        ].append(row)
    out = []
    for key, items in buckets.items():
        spec_id, source_group, intervention_type, variant, component, scope, alpha, step = key
        out.append(
            {
                "spec_id": spec_id,
                "source_group": source_group,
                "intervention_type": intervention_type,
                "variant": variant,
                "component": component,
                "channel_scope": scope,
                "alpha": alpha,
                "step": int(step),
                "rows": len(items),
                "mean_rank_improve": sum(finite_float(x.get("rank_improve")) for x in items) / len(items),
                "mean_target_logit_delta": sum(finite_float(x.get("target_logit_delta")) for x in items) / len(items),
                "mean_margin_delta_vs_winner": sum(finite_float(x.get("margin_delta_vs_winner")) for x in items) / len(items),
                "mean_target_margin_vs_winner": sum(finite_float(x.get("target_margin_vs_winner")) for x in items) / len(items),
                "top_token_changed": sum(1 for x in items if x.get("top_token_changed")),
                "winning_regime_changed": sum(1 for x in items if x.get("winning_regime_changed")),
                "winning_regimes": dict(Counter(str(x.get("winning_regime")) for x in items).most_common(8)),
                "top_tokens": dict(Counter(str(x.get("top_token")) for x in items).most_common(8)),
            }
        )
    out.sort(
        key=lambda row: abs(float(row.get("mean_margin_delta_vs_winner") or 0.0)) + abs(float(row.get("mean_target_logit_delta") or 0.0)),
        reverse=True,
    )
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    regime_rows: list[dict[str, Any]] = []
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
            success_rows = success_rows[: int(args.max_direction_rows)]
            drift_rows = drift_rows[: int(args.max_direction_rows)]
            filter_rows.append(
                {
                    "phase": PHASE,
                    "row_kind": "phase229_source_row_count",
                    "model": args.model,
                    "spec_id": spec["spec_id"],
                    "pattern_id": spec["pattern_id"],
                    "success_rows": len(success_rows),
                    "drift_rows": len(drift_rows),
                }
            )
            if not success_rows or not drift_rows:
                continue
            source_layers = [int(x) for x in spec["source_layers"]]
            observe_layers = [int(x) for x in spec["observe_layers"]]
            all_layers = sorted(set(source_layers + observe_layers))
            residual_dirs = p219.build_direction_vectors(model, tokenizer, device, success_rows, drift_rows, all_layers, int(args.max_steps))
            success_internal = p228.mean_internal(model, tokenizer, device, success_rows, source_layers, observe_layers, int(args.max_steps))
            drift_internal = p228.mean_internal(model, tokenizer, device, drift_rows, source_layers, observe_layers, int(args.max_steps))
            success_z = {step: {layer: part_map["product"] for layer, part_map in layer_map.items() if "product" in part_map} for step, layer_map in success_internal.items()}
            drift_z = {step: {layer: part_map["product"] for layer, part_map in layer_map.items() if "product" in part_map} for step, layer_map in drift_internal.items()}
            score_spec = {"spec_id": spec["spec_id"], "pattern_id": spec["pattern_id"], "layers": source_layers}
            _channel_rows, selected, _z_delta = p222.signed_channel_score_rows(
                model,
                args.model,
                score_spec,
                residual_dirs,
                success_z,
                drift_z,
                int(args.max_steps),
                int(args.top_channels),
            )
            for source_group, rows in [("drift", drift_rows[: int(args.max_eval_rows)]), ("success", success_rows[: int(args.max_eval_rows)])]:
                regime_rows.extend(
                    build_rows(
                        model,
                        tokenizer,
                        device,
                        base_groups,
                        args.model,
                        spec,
                        source_group,
                        rows,
                        selected,
                        success_internal,
                        drift_internal,
                        int(args.max_steps),
                        alphas,
                    )
                )
            log(f"{args.model}|{spec['spec_id']}: regime_rows={len(regime_rows)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary_rows = summarize_rows(regime_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Readout regime selection source atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "filter_rows": filter_rows,
        "regime_rows": len(regime_rows),
        "summary_rows": len(summary_rows),
        "top_summary": summary_rows[:100],
    }
    write_json(out_dir / f"phase229_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase229_{args.model}_filter_rows.jsonl", filter_rows)
    write_jsonl(out_dir / f"phase229_{args.model}_regime_rows.jsonl", regime_rows)
    write_jsonl(out_dir / f"phase229_{args.model}_summary_rows.jsonl", summary_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "regime_rows": len(regime_rows)}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase229_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    summary_rows = []
    for model in MODELS:
        summary_rows.extend(p214.iter_jsonl(out_dir / f"phase229_{model}_summary_rows.jsonl") or [])
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Cross-model readout regime selection source atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [x.get("model") for x in summaries],
        "regime_rows": sum(int(x.get("regime_rows") or 0) for x in summaries),
        "top_summary": sorted(
            summary_rows,
            key=lambda row: abs(float(row.get("mean_margin_delta_vs_winner") or 0.0)) + abs(float(row.get("mean_target_logit_delta") or 0.0)),
            reverse=True,
        )[:140],
    }
    write_json(out_dir / "phase229_cross_model_summary.json", payload)
    lines = ["# Phase 229 readout regime selection source atlas", ""]
    lines.append(f"regime_rows: {payload['regime_rows']}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| spec | group | type | variant | step | rows | rank improve | target logit delta | margin delta | target margin | winner changed | winners | top tokens |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["top_summary"][:90]:
        lines.append(
            f"| {row.get('spec_id')} | {row.get('source_group')} | {row.get('intervention_type')} | {row.get('variant')} | {row.get('step')} | {row.get('rows')} | "
            f"{finite_float(row.get('mean_rank_improve')):.4f} | {finite_float(row.get('mean_target_logit_delta')):.4f} | "
            f"{finite_float(row.get('mean_margin_delta_vs_winner')):.4f} | {finite_float(row.get('mean_target_margin_vs_winner')):.4f} | "
            f"{row.get('winning_regime_changed')} | {row.get('winning_regimes')} | {row.get('top_tokens')} |"
        )
    (out_dir / "phase229_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"phase": PHASE, "status": "complete", "models": payload["models"], "regime_rows": payload["regime_rows"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase229 readout regime selection source atlas")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--round-name", default="readout_regime_selection_atlas")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=16)
    parser.add_argument("--max-direction-rows", type=int, default=6)
    parser.add_argument("--max-eval-rows", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--top-channels", type=int, default=96)
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
