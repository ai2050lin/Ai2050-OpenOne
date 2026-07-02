#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import phase846_geometry_boundary_equation_fitting as p846


PHASE = 847
RESULT_ROOT = Path("tests/result/phase847_context_gated_residual_audit")
PHASE845_ROOT = Path("tests/result/phase845_geometry_gear_interaction_edge_atlas")
PREDICTORS = ("global_combo", "prompt_combo", "object_combo", "object_prompt_combo")


def residual(row: dict[str, Any]) -> float:
    return p846.finite(row.get("interaction_residual"))


def residual_class(value: float, threshold: float) -> str:
    if value >= threshold:
        return "synergy"
    if value <= -threshold:
        return "antagonistic"
    return "additive"


def source_path(round_name: str, model_name: str) -> Path:
    return PHASE845_ROOT / round_name / f"phase845_{model_name}_rows.jsonl"


def residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("combo_type")) in {"pair", "triplet"} and row.get("interaction_residual") is not None
    ]


def key_global(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("combo_type"), row.get("edit_mode"), row.get("combo_key"))


def key_prompt(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("combo_type"), row.get("edit_mode"), row.get("combo_key"), row.get("prompt_variant"))


def key_object(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("combo_type"), row.get("edit_mode"), row.get("combo_key"), row.get("object"))


def key_object_prompt(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("combo_type"),
        row.get("edit_mode"),
        row.get("combo_key"),
        row.get("object"),
        row.get("prompt_variant"),
    )


KEY_FNS: dict[str, Callable[[dict[str, Any]], tuple[Any, ...]]] = {
    "global_combo": key_global,
    "prompt_combo": key_prompt,
    "object_combo": key_object,
    "object_prompt_combo": key_object_prompt,
}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fit_means(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[tuple[Any, ...], float]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(residual(row))
    return {key: p846.finite(mean(values)) for key, values in groups.items()}


class ResidualPredictor:
    def __init__(self, train_rows: list[dict[str, Any]]) -> None:
        self.means = {name: fit_means(train_rows, fn) for name, fn in KEY_FNS.items()}
        self.type_mode = fit_means(train_rows, lambda row: (row.get("combo_type"), row.get("edit_mode")))
        self.type_only = fit_means(train_rows, lambda row: (row.get("combo_type"),))
        self.global_mean = p846.finite(mean([residual(row) for row in train_rows]))

    def predict(self, row: dict[str, Any], predictor: str) -> float:
        key = KEY_FNS[predictor](row)
        if key in self.means[predictor]:
            return self.means[predictor][key]
        global_key = key_global(row)
        if global_key in self.means["global_combo"]:
            return self.means["global_combo"][global_key]
        type_mode_key = (row.get("combo_type"), row.get("edit_mode"))
        if type_mode_key in self.type_mode:
            return self.type_mode[type_mode_key]
        type_key = (row.get("combo_type"),)
        if type_key in self.type_only:
            return self.type_only[type_key]
        return self.global_mean


def eval_rows(prediction_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not prediction_rows:
        return {"n": 0, "mae": None, "rmse": None, "sign_accuracy": None, "class_accuracy": None}
    errors = [p846.finite(row["predicted_residual"]) - p846.finite(row["actual_residual"]) for row in prediction_rows]
    sign_hits = [
        (p846.finite(row["predicted_residual"]) > 0) == (p846.finite(row["actual_residual"]) > 0)
        for row in prediction_rows
        if p846.finite(row["predicted_residual"]) != 0 or p846.finite(row["actual_residual"]) != 0
    ]
    class_hits = [
        residual_class(p846.finite(row["predicted_residual"]), threshold)
        == residual_class(p846.finite(row["actual_residual"]), threshold)
        for row in prediction_rows
    ]
    return {
        "n": len(prediction_rows),
        "mae": mean([abs(err) for err in errors]),
        "rmse": math.sqrt(mean([err * err for err in errors]) or 0.0),
        "sign_accuracy": p846.safe_div(sum(1 for hit in sign_hits if hit), len(sign_hits)),
        "class_accuracy": p846.safe_div(sum(1 for hit in class_hits if hit), len(class_hits)),
        "actual_classes": dict(Counter(residual_class(p846.finite(row["actual_residual"]), threshold) for row in prediction_rows)),
        "predicted_classes": dict(Counter(residual_class(p846.finite(row["predicted_residual"]), threshold) for row in prediction_rows)),
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
    predictor = ResidualPredictor(train_rows)
    predictions: list[dict[str, Any]] = []
    for predictor_name in PREDICTORS:
        for row in test_rows:
            pred = predictor.predict(row, predictor_name)
            actual = residual(row)
            predictions.append(
                {
                    "row_kind": "phase847_context_gated_residual_prediction",
                    "phase": PHASE,
                    "source_phase": 845,
                    "model": row.get("model"),
                    "source_round": row.get("round"),
                    "split_type": split_type,
                    "split_key": split_key,
                    "predictor": predictor_name,
                    "case_id": row.get("case_id"),
                    "object": row.get("object"),
                    "prompt_variant": row.get("prompt_variant"),
                    "combo_type": row.get("combo_type"),
                    "edit_mode": row.get("edit_mode"),
                    "combo_key": row.get("combo_key"),
                    "actual_residual": actual,
                    "predicted_residual": pred,
                    "actual_class": residual_class(actual, threshold),
                    "predicted_class": residual_class(pred, threshold),
                    "abs_error": abs(pred - actual),
                }
            )
    by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_predictor[str(row["predictor"])].append(row)
    summary = {name: eval_rows(rows, threshold) for name, rows in sorted(by_predictor.items())}
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
        by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        gains: dict[str, list[float]] = defaultdict(list)
        for result in results:
            for row in result["predictions"]:
                by_predictor[str(row["predictor"])].append(row)
            for name, value in (result.get("mae_gain_vs_global_combo") or {}).items():
                if value is not None:
                    gains[str(name)].append(float(value))
        out[split_type] = {
            "n_splits": len(results),
            "split_keys": [result["split_key"] for result in results],
            "predictor_summary": {name: eval_rows(rows, threshold) for name, rows in sorted(by_predictor.items())},
            "mean_mae_gain_vs_global_combo": {name: mean(values) for name, values in sorted(gains.items())},
        }
    return out


def context_strength(rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_means = fit_means(rows, key_global)
    prompt_means = fit_means(rows, key_prompt)
    object_means = fit_means(rows, key_object)
    object_prompt_means = fit_means(rows, key_object_prompt)

    def diffs(means: dict[tuple[Any, ...], float], parent_len: int) -> list[float]:
        values: list[float] = []
        for key, value in means.items():
            parent = tuple(key[:parent_len])
            if parent in global_means:
                values.append(abs(value - global_means[parent]))
        return values

    return {
        "prompt_mean_abs_shift_from_global": mean(diffs(prompt_means, 3)),
        "object_mean_abs_shift_from_global": mean(diffs(object_means, 3)),
        "object_prompt_mean_abs_shift_from_global": mean(diffs(object_prompt_means, 3)),
        "n_global_combo_terms": len(global_means),
        "n_prompt_combo_terms": len(prompt_means),
        "n_object_combo_terms": len(object_means),
        "n_object_prompt_combo_terms": len(object_prompt_means),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    source = source_path(args.phase845_round, args.model)
    if not source.exists():
        raise FileNotFoundError(f"missing Phase 845 rows: {source}")
    rows = residual_rows(p846.read_jsonl(source))
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    split_types = [part.strip() for part in args.split_types.split(",") if part.strip()]
    specs = split_specs(rows, split_types)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "source": str(source),
            "residual_rows": len(rows),
            "splits": [{"split_type": a, "split_key": b, "train_rows": len(c), "test_rows": len(d)} for a, b, c, d in specs],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    split_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for idx, (split_type, split_key, train_rows, test_rows) in enumerate(specs, 1):
        result = evaluate_split(train_rows, test_rows, split_type, split_key, float(args.interaction_threshold))
        split_results.append(result)
        all_predictions.extend(result["predictions"])
        if idx % max(1, int(args.log_every)) == 0 or idx == len(specs):
            p846.log(f"{args.model}/{args.round_name}: split {idx}/{len(specs)} {split_type}:{split_key} rows={len(test_rows)}")

    summary = {
        "phase": PHASE,
        "title": "Context Gated Residual Audit",
        "model": args.model,
        "round": args.round_name,
        "source_phase845_round": args.phase845_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "residual_rows": len(rows),
        "objects": sorted({str(row.get("object")) for row in rows}),
        "prompt_variants": sorted({str(row.get("prompt_variant")) for row in rows}),
        "combo_counts": dict(Counter(str(row.get("combo_type")) for row in rows)),
        "interaction_threshold": float(args.interaction_threshold),
        "context_strength": context_strength(rows),
        "split_summary": aggregate(split_results, float(args.interaction_threshold)),
        "n_predictions": len(all_predictions),
        "boundary": (
            "This phase audits whether Phase 845 interaction residuals are better explained by prompt/object "
            "context gates. It is residual transfer diagnostics, not route closure."
        ),
    }
    out_dir = RESULT_ROOT / args.round_name
    p846.write_jsonl(out_dir / f"phase847_{args.model}_predictions.jsonl", all_predictions)
    p846.write_json(out_dir / f"phase847_{args.model}_split_results.json", [{k: v for k, v in r.items() if k != "predictions"} for r in split_results])
    p846.write_json(out_dir / f"phase847_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "residual_rows": len(rows),
                "context_strength": summary["context_strength"],
                "split_summary": {
                    split: {
                        predictor: {
                            "n": row.get("n"),
                            "mae": row.get("mae"),
                            "class_accuracy": row.get("class_accuracy"),
                        }
                        for predictor, row in split_row.get("predictor_summary", {}).items()
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
        f"# Phase 847 Context Gated Residual Audit ({payload['round']})",
        "",
        "- Source: Phase 845 pair/triplet interaction residual rows.",
        "- Method: compare global residual means with prompt/object conditioned residual means.",
        "- Boundary: residual transfer diagnostics; not token closure.",
        "",
        "## Residual Prediction",
        "",
        "| model | split | predictor | n | MAE | RMSE | sign acc | class acc | mean MAE gain vs global |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in p846.MODELS:
        data = payload.get("model_summaries", {}).get(model) or {}
        for split, split_row in (data.get("split_summary") or {}).items():
            gains = split_row.get("mean_mae_gain_vs_global_combo") or {}
            for predictor, stats in (split_row.get("predictor_summary") or {}).items():
                lines.append(
                    f"| {model} | `{split}` | `{predictor}` | {stats.get('n', 0)} | "
                    f"{fmt(stats.get('mae'))} | {fmt(stats.get('rmse'))} | "
                    f"{fmt(stats.get('sign_accuracy'))} | {fmt(stats.get('class_accuracy'))} | "
                    f"{fmt(gains.get(predictor))} |"
                )
    lines += ["", "## Context Strength", ""]
    lines += ["| model | residual rows | prompt shift | object shift | object+prompt shift | terms |"]
    lines += ["|---|---:|---:|---:|---:|---|"]
    for model in p846.MODELS:
        data = payload.get("model_summaries", {}).get(model) or {}
        strength = data.get("context_strength") or {}
        terms = {
            "global": strength.get("n_global_combo_terms"),
            "prompt": strength.get("n_prompt_combo_terms"),
            "object": strength.get("n_object_combo_terms"),
            "object_prompt": strength.get("n_object_prompt_combo_terms"),
        }
        lines.append(
            f"| {model} | {data.get('residual_rows', 0)} | "
            f"{fmt(strength.get('prompt_mean_abs_shift_from_global'))} | "
            f"{fmt(strength.get('object_mean_abs_shift_from_global'))} | "
            f"{fmt(strength.get('object_prompt_mean_abs_shift_from_global'))} | "
            f"`{json.dumps(terms, ensure_ascii=False)}` |"
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
    for model in p846.MODELS:
        path = out_dir / f"phase847_{model}_summary.json"
        if path.exists():
            payload["model_summaries"][model] = p846.read_json(path)
            payload["models"].append(model)
    payload["status"] = "complete" if len(payload["models"]) == len(p846.MODELS) else "partial"
    p846.write_json(out_dir / "phase847_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase847_cross_model_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p846.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase845-round", default="smoke")
    parser.add_argument("--split-types", default="in_sample,object_holdout,prompt_holdout")
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
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
