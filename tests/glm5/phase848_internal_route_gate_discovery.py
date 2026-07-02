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
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase847_context_gated_residual_audit as p847  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 848
RESULT_ROOT = Path("tests/result/phase848_internal_route_gate_discovery")
PHASE845_ROOT = Path("tests/result/phase845_geometry_gear_interaction_edge_atlas")
MODELS = p846.MODELS
PREDICTORS = (
    "global_combo",
    "prompt_combo",
    "object_combo",
    "internal_sign_combo",
    "internal_count_combo",
    "internal_strength_combo",
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def gear_key(layer_idx: int, channel_id: int) -> str:
    return f"L{int(layer_idx)}C{int(channel_id)}"


def parse_gear_key(key: str) -> tuple[int, int]:
    text = str(key)
    if not text.startswith("L") or "C" not in text:
        raise ValueError(f"bad gear key: {key}")
    layer, channel = text[1:].split("C", 1)
    return int(layer), int(channel)


def load_phase845_rows(round_name: str, model_name: str) -> list[dict[str, Any]]:
    path = PHASE845_ROOT / round_name / f"phase845_{model_name}_rows.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 845 rows: {path}")
    rows = p846.read_jsonl(path)
    return [
        row
        for row in rows
        if str(row.get("combo_type")) in {"pair", "triplet"} and row.get("interaction_residual") is not None
    ]


def load_phase845_gears(round_name: str, model_name: str) -> list[dict[str, Any]]:
    path = PHASE845_ROOT / round_name / f"phase845_{model_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 845 summary: {path}")
    data = p846.read_json(path)
    gears: list[dict[str, Any]] = []
    for row in data.get("top_gears") or []:
        gear = dict(row)
        gear["gear_key"] = str(gear.get("gear_key") or gear_key(int(gear["layer_idx"]), int(gear["channel_id"])))
        gears.append(gear)
    return gears


def prompt_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("case_id")), str(row.get("prompt_variant")), str(row.get("prompt")))


def activation_sign(value: float, eps: float = 1e-6) -> str:
    if value > eps:
        return "+"
    if value < -eps:
        return "-"
    return "0"


def residual_class(value: float, threshold: float) -> str:
    return p847.residual_class(value, threshold)


def capture_prompt_activations(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict[str, Any]],
    gears: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[str, str, str], dict[str, float]]:
    layer_indices = sorted({int(gear["layer_idx"]) for gear in gears})
    valid_layers = set(range(len(get_layers(model))))
    layer_indices = [layer for layer in layer_indices if layer in valid_layers]
    gear_lookup = {str(gear["gear_key"]): (int(gear["layer_idx"]), int(gear["channel_id"])) for gear in gears}
    unique_prompts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        unique_prompts[prompt_key(row)] = row

    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for idx, (key, row) in enumerate(unique_prompts.items(), 1):
        prompt = str(row.get("prompt"))
        captured = p844.capture_mlp_down_inputs(model, tokenizer, device, prompt, layer_indices)
        acts: dict[str, float] = {}
        for gkey, (layer_idx, channel_id) in gear_lookup.items():
            vec = captured.get(int(layer_idx))
            if vec is None or channel_id < 0 or channel_id >= int(vec.numel()):
                acts[gkey] = 0.0
            else:
                acts[gkey] = float(vec[int(channel_id)].item())
        out[key] = acts
        if idx % max(1, int(args.log_every)) == 0 or idx == len(unique_prompts):
            log(f"{args.model}/{args.round_name}: captured prompt activations {idx}/{len(unique_prompts)}")
    return out


def make_feature_rows(rows: list[dict[str, Any]], prompt_acts: dict[tuple[str, str, str], dict[str, float]]) -> list[dict[str, Any]]:
    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        acts_by_key = prompt_acts.get(prompt_key(row), {})
        gkeys = [str(key) for key in row.get("gear_keys") or p846.split_combo_key(str(row.get("combo_key")))]
        acts = [float(acts_by_key.get(key, 0.0)) for key in gkeys]
        signs = [activation_sign(value) for value in acts]
        abs_values = [abs(value) for value in acts]
        pos_count = sum(1 for sign in signs if sign == "+")
        neg_count = sum(1 for sign in signs if sign == "-")
        zero_count = sum(1 for sign in signs if sign == "0")
        feature_rows.append(
            {
                "row_kind": "phase848_internal_route_gate_feature",
                "phase": PHASE,
                "source_phase": 845,
                "model": row.get("model"),
                "source_round": row.get("round"),
                "case_id": row.get("case_id"),
                "object": row.get("object"),
                "prompt_variant": row.get("prompt_variant"),
                "prompt": row.get("prompt"),
                "combo_type": row.get("combo_type"),
                "edit_mode": row.get("edit_mode"),
                "combo_key": row.get("combo_key"),
                "gear_keys": gkeys,
                "gear_count": len(gkeys),
                "activation_values": acts,
                "activation_signs": signs,
                "sign_pattern": "".join(signs),
                "pos_count": pos_count,
                "neg_count": neg_count,
                "zero_count": zero_count,
                "signed_sum": sum(acts),
                "signed_mean": finite(mean(acts)),
                "abs_sum": sum(abs_values),
                "abs_mean": finite(mean(abs_values)),
                "max_abs": max(abs_values) if abs_values else 0.0,
                "min_abs": min(abs_values) if abs_values else 0.0,
                "original_margin": finite(row.get("original_target_minus_object_logit")),
                "actual_residual": finite(row.get("interaction_residual")),
                "actual_delta": finite(row.get("margin_delta_vs_original")),
                "expected_additive_delta": finite(row.get("expected_additive_delta")),
                "actual_class": residual_class(finite(row.get("interaction_residual")), 0.5),
                "target_transition": bool(row.get("target_transition")),
                "target_gained_vs_original": bool(row.get("target_gained_vs_original")),
                "target_lost_vs_original": bool(row.get("target_lost_vs_original")),
            }
        )
    return feature_rows


def quantile(values: list[float], frac: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(frac * (len(ordered) - 1)))))
    return ordered[idx]


class GateResidualPredictor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.low = quantile([finite(row.get("abs_sum")) for row in rows], 1 / 3)
        self.high = quantile([finite(row.get("abs_sum")) for row in rows], 2 / 3)
        self.tables: dict[str, dict[tuple[Any, ...], float]] = {}
        for name in PREDICTORS:
            self.tables[name] = self._fit(rows, lambda row, predictor=name: self.key(row, predictor))
        self.type_mode = self._fit(rows, lambda row: (row.get("combo_type"), row.get("edit_mode")))
        self.global_mean = finite(mean([finite(row.get("actual_residual")) for row in rows]))

    def strength_bucket(self, value: float) -> str:
        if value <= self.low:
            return "low"
        if value <= self.high:
            return "mid"
        return "high"

    def key(self, row: dict[str, Any], predictor: str) -> tuple[Any, ...]:
        base = (row.get("combo_type"), row.get("edit_mode"), row.get("combo_key"))
        if predictor == "global_combo":
            return base
        if predictor == "prompt_combo":
            return (*base, row.get("prompt_variant"))
        if predictor == "object_combo":
            return (*base, row.get("object"))
        if predictor == "internal_sign_combo":
            return (*base, row.get("sign_pattern"))
        if predictor == "internal_count_combo":
            return (*base, row.get("pos_count"), row.get("neg_count"), row.get("zero_count"))
        if predictor == "internal_strength_combo":
            return (*base, row.get("sign_pattern"), self.strength_bucket(finite(row.get("abs_sum"))))
        raise ValueError(f"unknown predictor: {predictor}")

    def _fit(self, rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[tuple[Any, ...], float]:
        groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(finite(row.get("actual_residual")))
        return {key: finite(mean(values)) for key, values in groups.items()}

    def predict(self, row: dict[str, Any], predictor: str) -> float:
        table = self.tables[predictor]
        key = self.key(row, predictor)
        if key in table:
            return table[key]
        if predictor == "internal_strength_combo":
            sign_key = self.key(row, "internal_sign_combo")
            sign_table = self.tables["internal_sign_combo"]
            if sign_key in sign_table:
                return sign_table[sign_key]
        base_key = self.key(row, "global_combo")
        global_table = self.tables["global_combo"]
        if base_key in global_table:
            return global_table[base_key]
        type_mode_key = (row.get("combo_type"), row.get("edit_mode"))
        if type_mode_key in self.type_mode:
            return self.type_mode[type_mode_key]
        return self.global_mean


def binary_stats(actual: list[bool], predicted: list[bool]) -> dict[str, Any]:
    return p846.binary_stats(actual, predicted)


def eval_predictions(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mae": None, "rmse": None, "sign_accuracy": None, "class_accuracy": None, "strong": binary_stats([], [])}
    errors = [finite(row.get("predicted_residual")) - finite(row.get("actual_residual")) for row in rows]
    sign_hits = [
        (finite(row.get("predicted_residual")) > 0) == (finite(row.get("actual_residual")) > 0)
        for row in rows
        if finite(row.get("predicted_residual")) != 0 or finite(row.get("actual_residual")) != 0
    ]
    class_hits = [
        residual_class(finite(row.get("predicted_residual")), threshold)
        == residual_class(finite(row.get("actual_residual")), threshold)
        for row in rows
    ]
    actual_strong = [abs(finite(row.get("actual_residual"))) >= threshold for row in rows]
    pred_strong = [abs(finite(row.get("predicted_residual"))) >= threshold for row in rows]
    return {
        "n": len(rows),
        "mae": mean([abs(err) for err in errors]),
        "rmse": math.sqrt(mean([err * err for err in errors]) or 0.0),
        "sign_accuracy": p846.safe_div(sum(1 for hit in sign_hits if hit), len(sign_hits)),
        "class_accuracy": p846.safe_div(sum(1 for hit in class_hits if hit), len(class_hits)),
        "strong": binary_stats(actual_strong, pred_strong),
        "actual_classes": dict(Counter(residual_class(finite(row.get("actual_residual")), threshold) for row in rows)),
        "predicted_classes": dict(Counter(residual_class(finite(row.get("predicted_residual")), threshold) for row in rows)),
    }


def split_specs(rows: list[dict[str, Any]], split_types: list[str]) -> list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]]:
    specs: list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    if "in_sample" in split_types:
        specs.append(("in_sample", "all", rows, rows))
    if "object_holdout" in split_types:
        for obj in sorted({str(row.get("object")) for row in rows}):
            train = [row for row in rows if str(row.get("object")) != obj]
            test = [row for row in rows if str(row.get("object")) == obj]
            if train and test:
                specs.append(("object_holdout", obj, train, test))
    if "prompt_holdout" in split_types:
        for prompt in sorted({str(row.get("prompt_variant")) for row in rows}):
            train = [row for row in rows if str(row.get("prompt_variant")) != prompt]
            test = [row for row in rows if str(row.get("prompt_variant")) == prompt]
            if train and test:
                specs.append(("prompt_holdout", prompt, train, test))
    return specs


def evaluate_split(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    split_type: str,
    split_key: str,
    threshold: float,
) -> dict[str, Any]:
    predictor = GateResidualPredictor(train_rows)
    predictions: list[dict[str, Any]] = []
    for predictor_name in PREDICTORS:
        for row in test_rows:
            pred = predictor.predict(row, predictor_name)
            predictions.append(
                {
                    "row_kind": "phase848_internal_route_gate_prediction",
                    "phase": PHASE,
                    "model": row.get("model"),
                    "source_round": row.get("source_round"),
                    "split_type": split_type,
                    "split_key": split_key,
                    "predictor": predictor_name,
                    "case_id": row.get("case_id"),
                    "object": row.get("object"),
                    "prompt_variant": row.get("prompt_variant"),
                    "combo_type": row.get("combo_type"),
                    "edit_mode": row.get("edit_mode"),
                    "combo_key": row.get("combo_key"),
                    "sign_pattern": row.get("sign_pattern"),
                    "pos_count": row.get("pos_count"),
                    "neg_count": row.get("neg_count"),
                    "abs_sum": row.get("abs_sum"),
                    "actual_residual": row.get("actual_residual"),
                    "predicted_residual": pred,
                    "actual_class": residual_class(finite(row.get("actual_residual")), threshold),
                    "predicted_class": residual_class(pred, threshold),
                    "abs_error": abs(pred - finite(row.get("actual_residual"))),
                }
            )
    by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_predictor[str(row["predictor"])].append(row)
    summary = {name: eval_predictions(rows, threshold) for name, rows in sorted(by_predictor.items())}
    global_mae = summary.get("global_combo", {}).get("mae")
    gains = {
        name: None if global_mae is None or stats.get("mae") is None else float(global_mae - stats["mae"])
        for name, stats in summary.items()
        if name != "global_combo"
    }
    return {
        "split_type": split_type,
        "split_key": split_key,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "predictor_summary": summary,
        "mae_gain_vs_global_combo": gains,
        "predictions": predictions,
    }


def aggregate(split_results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in split_results:
        by_split[str(result["split_type"])].append(result)
    out: dict[str, Any] = {}
    for split_type, results in sorted(by_split.items()):
        rows_by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        gains: dict[str, list[float]] = defaultdict(list)
        for result in results:
            for row in result["predictions"]:
                rows_by_predictor[str(row["predictor"])].append(row)
            for name, value in (result.get("mae_gain_vs_global_combo") or {}).items():
                if value is not None:
                    gains[str(name)].append(float(value))
        out[split_type] = {
            "n_splits": len(results),
            "split_keys": [result["split_key"] for result in results],
            "predictor_summary": {
                name: eval_predictions(rows, threshold) for name, rows in sorted(rows_by_predictor.items())
            },
            "mean_mae_gain_vs_global_combo": {name: mean(values) for name, values in sorted(gains.items())},
        }
    return out


def feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "sign_patterns": dict(Counter(str(row.get("sign_pattern")) for row in rows)),
        "neg_count": dict(Counter(int(row.get("neg_count", 0)) for row in rows)),
        "mean_abs_sum": mean([finite(row.get("abs_sum")) for row in rows]),
        "mean_abs_mean": mean([finite(row.get("abs_mean")) for row in rows]),
        "mean_original_margin": mean([finite(row.get("original_margin")) for row in rows]),
        "residual_classes": dict(Counter(str(row.get("actual_class")) for row in rows)),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_phase845_rows(args.phase845_round, args.model)
    gears = load_phase845_gears(args.phase845_round, args.model)[: int(args.top_gears)]
    if args.max_rows and int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "phase845_round": args.phase845_round,
            "rows": len(rows),
            "gears": [gear["gear_key"] for gear in gears],
            "unique_prompts": len({prompt_key(row) for row in rows}),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt_acts = capture_prompt_activations(model, tokenizer, device, rows, gears, args)
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    feature_rows = make_feature_rows(rows, prompt_acts)
    split_types = [part.strip() for part in args.split_types.split(",") if part.strip()]
    split_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for idx, (split_type, split_key, train_rows, test_rows) in enumerate(split_specs(feature_rows, split_types), 1):
        result = evaluate_split(train_rows, test_rows, split_type, split_key, float(args.interaction_threshold))
        split_results.append(result)
        all_predictions.extend(result["predictions"])
        if idx % max(1, int(args.log_every)) == 0 or idx == len(split_results):
            log(f"{args.model}/{args.round_name}: evaluated split {idx} {split_type}:{split_key}")

    summary = {
        "phase": PHASE,
        "title": "Internal Route Gate Discovery",
        "model": args.model,
        "round": args.round_name,
        "phase845_round": args.phase845_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_feature_rows": len(feature_rows),
        "n_predictions": len(all_predictions),
        "n_unique_prompts": len(prompt_acts),
        "top_gears": gears,
        "feature_summary": feature_summary(feature_rows),
        "split_summary": aggregate(split_results, float(args.interaction_threshold)),
        "boundary": (
            "This phase tests whether natural internal activations of geometry gears form a route-gate "
            "signal for Phase 845 interaction residuals. It is a gate-discovery probe, not closure."
        ),
    }
    p846.write_jsonl(out_dir / f"phase848_{args.model}_feature_rows.jsonl", feature_rows)
    p846.write_jsonl(out_dir / f"phase848_{args.model}_predictions.jsonl", all_predictions)
    p846.write_json(out_dir / f"phase848_{args.model}_split_results.json", [{k: v for k, v in r.items() if k != "predictions"} for r in split_results])
    p846.write_json(out_dir / f"phase848_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "feature_rows": len(feature_rows),
                "unique_prompts": len(prompt_acts),
                "feature_summary": summary["feature_summary"],
                "split_summary": {
                    split: {
                        predictor: {
                            "n": stats.get("n"),
                            "mae": stats.get("mae"),
                            "sign_accuracy": stats.get("sign_accuracy"),
                            "strong_f1": (stats.get("strong") or {}).get("f1"),
                        }
                        for predictor, stats in split_row.get("predictor_summary", {}).items()
                    }
                    for split, split_row in summary["split_summary"].items()
                },
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
        f"# Phase 848 Internal Route Gate Discovery ({payload['round']})",
        "",
        "- Source: Phase 845 residual rows plus fresh natural activation captures.",
        "- Method: compare external and internal activation-gate residual predictors.",
        "- Boundary: internal gate discovery probe; not geometry closure.",
        "",
        "## Model Summary",
        "",
        "| model | feature rows | prompts | split | predictor | n | MAE | sign acc | strong F1 | MAE gain vs global |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for split, split_row in (data.get("split_summary") or {}).items():
            gains = split_row.get("mean_mae_gain_vs_global_combo") or {}
            for predictor, stats in (split_row.get("predictor_summary") or {}).items():
                lines.append(
                    f"| {model_name} | {data.get('n_feature_rows', 0)} | {data.get('n_unique_prompts', 0)} | "
                    f"`{split}` | `{predictor}` | {stats.get('n', 0)} | {fmt(stats.get('mae'))} | "
                    f"{fmt(stats.get('sign_accuracy'))} | {fmt((stats.get('strong') or {}).get('f1'))} | "
                    f"{fmt(gains.get(predictor))} |"
                )
    lines += ["", "## Feature Summary", ""]
    lines += ["| model | sign patterns | neg count | residual classes | mean abs sum |"]
    lines += ["|---|---|---|---|---:|"]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        fs = data.get("feature_summary") or {}
        lines.append(
            f"| {model_name} | `{json.dumps(fs.get('sign_patterns') or {}, ensure_ascii=False)}` | "
            f"`{json.dumps(fs.get('neg_count') or {}, ensure_ascii=False)}` | "
            f"`{json.dumps(fs.get('residual_classes') or {}, ensure_ascii=False)}` | "
            f"{fmt(fs.get('mean_abs_sum'))} |"
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
        path = out_dir / f"phase848_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = p846.read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase848_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase848_cross_model_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase845-round", default="smoke")
    parser.add_argument("--top-gears", type=int, default=6)
    parser.add_argument("--split-types", default="in_sample,object_holdout,prompt_holdout")
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)


if __name__ == "__main__":
    main()
