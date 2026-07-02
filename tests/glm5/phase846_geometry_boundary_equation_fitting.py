#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE = 846
MODELS = ("qwen3", "glm4", "deepseek7b")
RESULT_ROOT = Path("tests/result/phase846_geometry_boundary_equation_fitting")
PHASE845_ROOT = Path("tests/result/phase845_geometry_gear_interaction_edge_atlas")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def finite(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def model_source_path(round_name: str, model_name: str) -> Path:
    return PHASE845_ROOT / round_name / f"phase845_{model_name}_rows.jsonl"


def row_actual_margin(row: dict[str, Any]) -> float:
    if row.get("target_minus_object_logit") is not None:
        return finite(row.get("target_minus_object_logit"))
    return finite(row.get("original_target_minus_object_logit")) + finite(row.get("margin_delta_vs_original"))


def row_original_margin(row: dict[str, Any]) -> float:
    return finite(row.get("original_target_minus_object_logit"))


def row_actual_delta(row: dict[str, Any]) -> float:
    return finite(row.get("margin_delta_vs_original"))


def row_target(row: dict[str, Any]) -> bool:
    return bool(row.get("target_transition"))


def row_original_target(row: dict[str, Any]) -> bool:
    return bool(row.get("original_target_transition"))


def split_combo_key(combo_key: str) -> list[str]:
    if not combo_key or combo_key == "original":
        return []
    return [part for part in str(combo_key).split("+") if part]


def mode_key(row: dict[str, Any]) -> str:
    return str(row.get("edit_mode") or "")


def combo_key(row: dict[str, Any]) -> str:
    return str(row.get("combo_key") or "")


def combo_type(row: dict[str, Any]) -> str:
    return str(row.get("combo_type") or "")


def train_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = [(row_actual_margin(row), row_target(row)) for row in rows if row.get("target_minus_object_logit") is not None]
    if not points:
        return {"threshold": 0.0, "train_f1": 0.0, "train_accuracy": 0.0, "source": "default_empty"}

    margins = sorted({m for m, _ in points})
    candidates: list[float] = []
    candidates.append(margins[0] - 1.0)
    candidates.extend((a + b) / 2.0 for a, b in zip(margins, margins[1:]))
    candidates.append(margins[-1] + 1.0)
    if 0.0 not in candidates:
        candidates.append(0.0)

    best: dict[str, Any] | None = None
    for threshold in candidates:
        stats = binary_stats([target for _, target in points], [m >= threshold for m, _ in points])
        score = (stats["f1"], stats["accuracy"])
        if best is None or score > (best["train_f1"], best["train_accuracy"]):
            best = {
                "threshold": threshold,
                "train_f1": stats["f1"],
                "train_accuracy": stats["accuracy"],
                "source": "grid_margin_target_transition",
            }
    return best or {"threshold": 0.0, "train_f1": 0.0, "train_accuracy": 0.0, "source": "fallback"}


def binary_stats(actual: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(1 for a, p in zip(actual, predicted) if a and p)
    tn = sum(1 for a, p in zip(actual, predicted) if (not a) and (not p))
    fp = sum(1 for a, p in zip(actual, predicted) if (not a) and p)
    fn = sum(1 for a, p in zip(actual, predicted) if a and (not p))
    n = len(actual)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class BoundaryEquation:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.threshold_info = train_threshold(rows)
        self.threshold = float(self.threshold_info["threshold"])
        self.single_means: dict[tuple[str, str], float] = {}
        self.single_mode_means: dict[str, float] = {}
        self.residual_means: dict[tuple[str, str, str], float] = {}
        self.residual_type_mode_means: dict[tuple[str, str], float] = {}
        self.residual_type_means: dict[str, float] = {}
        self._fit(rows)

    def _fit(self, rows: list[dict[str, Any]]) -> None:
        single_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        single_mode_values: dict[str, list[float]] = defaultdict(list)
        residual_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        residual_type_mode_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        residual_type_values: dict[str, list[float]] = defaultdict(list)

        for row in rows:
            ctype = combo_type(row)
            mode = mode_key(row)
            key = combo_key(row)
            if ctype == "single" and row.get("margin_delta_vs_original") is not None:
                delta = row_actual_delta(row)
                single_values[(mode, key)].append(delta)
                single_mode_values[mode].append(delta)
            if ctype in {"pair", "triplet"} and row.get("interaction_residual") is not None:
                residual = finite(row.get("interaction_residual"))
                residual_values[(ctype, mode, key)].append(residual)
                residual_type_mode_values[(ctype, mode)].append(residual)
                residual_type_values[ctype].append(residual)

        self.single_means = {k: finite(mean(v)) for k, v in single_values.items()}
        self.single_mode_means = {k: finite(mean(v)) for k, v in single_mode_values.items()}
        self.residual_means = {k: finite(mean(v)) for k, v in residual_values.items()}
        self.residual_type_mode_means = {k: finite(mean(v)) for k, v in residual_type_mode_values.items()}
        self.residual_type_means = {k: finite(mean(v)) for k, v in residual_type_values.items()}

    def single_delta(self, mode: str, gear_key: str) -> float:
        if (mode, gear_key) in self.single_means:
            return self.single_means[(mode, gear_key)]
        return self.single_mode_means.get(mode, 0.0)

    def additive_delta(self, row: dict[str, Any]) -> float:
        ctype = combo_type(row)
        mode = mode_key(row)
        key = combo_key(row)
        if ctype == "original":
            return 0.0
        if ctype == "single":
            return self.single_delta(mode, key)
        return sum(self.single_delta(mode, gear) for gear in split_combo_key(key))

    def residual_delta(self, row: dict[str, Any]) -> float:
        ctype = combo_type(row)
        if ctype not in {"pair", "triplet"}:
            return 0.0
        mode = mode_key(row)
        key = combo_key(row)
        if (ctype, mode, key) in self.residual_means:
            return self.residual_means[(ctype, mode, key)]
        if (ctype, mode) in self.residual_type_mode_means:
            return self.residual_type_mode_means[(ctype, mode)]
        return self.residual_type_means.get(ctype, 0.0)

    def predict(self, row: dict[str, Any], predictor: str) -> dict[str, Any]:
        additive = self.additive_delta(row)
        residual = self.residual_delta(row) if predictor == "interaction_equation" else 0.0
        predicted_delta = additive + residual
        predicted_margin = row_original_margin(row) + predicted_delta
        predicted_target = predicted_margin >= self.threshold
        return {
            "predictor": predictor,
            "predicted_delta": predicted_delta,
            "predicted_margin": predicted_margin,
            "predicted_target_transition": predicted_target,
            "predicted_target_gained_vs_original": (not row_original_target(row)) and predicted_target,
            "predicted_target_lost_vs_original": row_original_target(row) and (not predicted_target),
            "additive_component": additive,
            "residual_component": residual,
            "threshold": self.threshold,
        }


def eval_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "n": 0,
            "mae_delta": None,
            "rmse_delta": None,
            "sign_accuracy_delta": None,
            "target": binary_stats([], []),
            "gain": binary_stats([], []),
            "lost": binary_stats([], []),
            "boundary_change_accuracy": 0.0,
        }

    abs_errors = [abs(finite(row["predicted_delta"]) - finite(row["actual_delta"])) for row in predictions]
    sq_errors = [(finite(row["predicted_delta"]) - finite(row["actual_delta"])) ** 2 for row in predictions]
    sign_hits = [
        (finite(row["predicted_delta"]) > 0) == (finite(row["actual_delta"]) > 0)
        for row in predictions
        if finite(row["predicted_delta"]) != 0 or finite(row["actual_delta"]) != 0
    ]
    actual_targets = [bool(row["actual_target_transition"]) for row in predictions]
    predicted_targets = [bool(row["predicted_target_transition"]) for row in predictions]
    actual_gain = [bool(row["actual_target_gained_vs_original"]) for row in predictions]
    predicted_gain = [bool(row["predicted_target_gained_vs_original"]) for row in predictions]
    actual_lost = [bool(row["actual_target_lost_vs_original"]) for row in predictions]
    predicted_lost = [bool(row["predicted_target_lost_vs_original"]) for row in predictions]
    boundary_change_hits = [
        (bool(row["actual_target_gained_vs_original"]), bool(row["actual_target_lost_vs_original"]))
        == (bool(row["predicted_target_gained_vs_original"]), bool(row["predicted_target_lost_vs_original"]))
        for row in predictions
    ]
    return {
        "n": len(predictions),
        "mae_delta": mean(abs_errors),
        "rmse_delta": math.sqrt(mean(sq_errors) or 0.0),
        "sign_accuracy_delta": safe_div(sum(1 for hit in sign_hits if hit), len(sign_hits)),
        "target": binary_stats(actual_targets, predicted_targets),
        "gain": binary_stats(actual_gain, predicted_gain),
        "lost": binary_stats(actual_lost, predicted_lost),
        "boundary_change_accuracy": safe_div(sum(1 for hit in boundary_change_hits if hit), len(boundary_change_hits)),
        "actual_target_rows": sum(actual_targets),
        "predicted_target_rows": sum(predicted_targets),
        "actual_gain_rows": sum(actual_gain),
        "predicted_gain_rows": sum(predicted_gain),
        "actual_lost_rows": sum(actual_lost),
        "predicted_lost_rows": sum(predicted_lost),
    }


def build_prediction_row(
    row: dict[str, Any],
    split_type: str,
    split_key: str,
    predictor: str,
    equation: BoundaryEquation,
) -> dict[str, Any]:
    pred = equation.predict(row, predictor)
    return {
        "row_kind": "phase846_geometry_boundary_equation_prediction",
        "phase": PHASE,
        "source_phase": 845,
        "model": row.get("model"),
        "source_round": row.get("round"),
        "split_type": split_type,
        "split_key": split_key,
        "predictor": predictor,
        "case_id": row.get("case_id"),
        "object": row.get("object"),
        "prompt_variant": row.get("prompt_variant"),
        "combo_type": row.get("combo_type"),
        "edit_mode": row.get("edit_mode"),
        "combo_key": row.get("combo_key"),
        "actual_delta": row_actual_delta(row),
        "actual_margin": row_actual_margin(row),
        "actual_target_transition": row_target(row),
        "actual_target_gained_vs_original": bool(row.get("target_gained_vs_original")),
        "actual_target_lost_vs_original": bool(row.get("target_lost_vs_original")),
        "actual_boundary_class": row.get("boundary_class"),
        "original_margin": row_original_margin(row),
        "original_target_transition": row_original_target(row),
        **pred,
    }


def evaluate_split(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    split_type: str,
    split_key: str,
) -> dict[str, Any]:
    equation = BoundaryEquation(train_rows)
    eval_rows = [row for row in test_rows if combo_type(row) != "original"]
    prediction_rows: list[dict[str, Any]] = []
    for predictor in ("additive_only", "interaction_equation"):
        for row in eval_rows:
            prediction_rows.append(build_prediction_row(row, split_type, split_key, predictor, equation))

    by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_combo_predictor: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_predictor[str(row["predictor"])].append(row)
        by_combo_predictor[(str(row["predictor"]), str(row["combo_type"]))].append(row)

    predictor_summary = {predictor: eval_predictions(rows) for predictor, rows in by_predictor.items()}
    combo_summary = {
        f"{predictor}:{ctype}": eval_predictions(rows)
        for (predictor, ctype), rows in sorted(by_combo_predictor.items())
    }
    add_mae = predictor_summary.get("additive_only", {}).get("mae_delta")
    int_mae = predictor_summary.get("interaction_equation", {}).get("mae_delta")
    improvement = None if add_mae is None or int_mae is None else float(add_mae - int_mae)
    return {
        "split_type": split_type,
        "split_key": split_key,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "eval_rows": len(eval_rows),
        "threshold_info": equation.threshold_info,
        "n_single_terms": len(equation.single_means),
        "n_residual_terms": len(equation.residual_means),
        "predictor_summary": predictor_summary,
        "combo_summary": combo_summary,
        "interaction_mae_improvement_over_additive": improvement,
        "predictions": prediction_rows,
    }


def split_specs(rows: list[dict[str, Any]], split_types: list[str]) -> list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]]:
    specs: list[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    if "in_sample" in split_types:
        specs.append(("in_sample", "all", rows, rows))
    if "object_holdout" in split_types:
        for obj in sorted({str(row.get("object")) for row in rows}):
            test = [row for row in rows if str(row.get("object")) == obj]
            train = [row for row in rows if str(row.get("object")) != obj]
            if train and test:
                specs.append(("object_holdout", obj, train, test))
    if "prompt_holdout" in split_types:
        for prompt in sorted({str(row.get("prompt_variant")) for row in rows}):
            test = [row for row in rows if str(row.get("prompt_variant")) == prompt]
            train = [row for row in rows if str(row.get("prompt_variant")) != prompt]
            if train and test:
                specs.append(("prompt_holdout", prompt, train, test))
    return specs


def aggregate_split_summaries(split_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in split_results:
        by_split[str(result["split_type"])].append(result)

    out: dict[str, Any] = {}
    for split_type, results in sorted(by_split.items()):
        rows_by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        improvements: list[float] = []
        for result in results:
            for prediction in result["predictions"]:
                rows_by_predictor[str(prediction["predictor"])].append(prediction)
            improvement = result.get("interaction_mae_improvement_over_additive")
            if improvement is not None:
                improvements.append(float(improvement))
        out[split_type] = {
            "n_splits": len(results),
            "split_keys": [result["split_key"] for result in results],
            "mean_interaction_mae_improvement_over_additive": mean(improvements),
            "predictor_summary": {
                predictor: eval_predictions(rows) for predictor, rows in sorted(rows_by_predictor.items())
            },
        }
    return out


def summarize_model(args: argparse.Namespace, rows: list[dict[str, Any]], split_results: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [prediction for result in split_results for prediction in result["predictions"]]
    by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_predictor[str(row["predictor"])].append(row)
    source_counts = Counter(str(row.get("combo_type")) for row in rows)
    return {
        "phase": PHASE,
        "title": "Geometry Boundary Equation Fitting",
        "model": args.model,
        "round": args.round_name,
        "source_phase845_round": args.phase845_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_rows": len(rows),
        "source_combo_counts": dict(source_counts),
        "source_objects": sorted({str(row.get("object")) for row in rows}),
        "source_prompt_variants": sorted({str(row.get("prompt_variant")) for row in rows}),
        "split_types": args.split_types.split(","),
        "split_summary": aggregate_split_summaries(split_results),
        "overall_predictor_summary": {
            predictor: eval_predictions(rows_) for predictor, rows_ in sorted(by_predictor.items())
        },
        "n_predictions": len(predictions),
        "boundary": (
            "This phase fits a simple route-boundary equation from Phase 845 rows and tests it on "
            "in-sample, object-holdout, and prompt-holdout splits. It does not load models and does "
            "not claim token closure."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    source_path = model_source_path(args.phase845_round, args.model)
    if not source_path.exists():
        raise FileNotFoundError(f"missing Phase 845 rows: {source_path}")
    rows = read_jsonl(source_path)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]
    split_types = [part.strip() for part in args.split_types.split(",") if part.strip()]
    specs = split_specs(rows, split_types)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "source_path": str(source_path),
            "source_rows": len(rows),
            "split_count": len(specs),
            "splits": [{"split_type": s[0], "split_key": s[1], "train_rows": len(s[2]), "test_rows": len(s[3])} for s in specs],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    split_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for idx, (split_type, split_key, train_rows, test_rows) in enumerate(specs, 1):
        result = evaluate_split(train_rows, test_rows, split_type, split_key)
        split_results.append(result)
        all_predictions.extend(result["predictions"])
        if idx % max(1, int(args.log_every)) == 0 or idx == len(specs):
            log(
                f"{args.model}/{args.round_name}: split {idx}/{len(specs)} "
                f"{split_type}:{split_key} eval_rows={result['eval_rows']}"
            )

    summary = summarize_model(args, rows, split_results)
    compact_splits = [{k: v for k, v in result.items() if k != "predictions"} for result in split_results]
    write_jsonl(out_dir / f"phase846_{args.model}_predictions.jsonl", all_predictions)
    write_json(out_dir / f"phase846_{args.model}_split_results.json", compact_splits)
    write_json(out_dir / f"phase846_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "source_rows": summary["source_rows"],
                "predictions": summary["n_predictions"],
                "split_summary": {
                    split: {
                        predictor: {
                            "n": data.get("n"),
                            "mae_delta": data.get("mae_delta"),
                            "target_accuracy": data.get("target", {}).get("accuracy"),
                            "target_f1": data.get("target", {}).get("f1"),
                        }
                        for predictor, data in row.get("predictor_summary", {}).items()
                    }
                    for split, row in summary["split_summary"].items()
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
        f"# Phase 846 Geometry Boundary Equation Fitting ({payload['round']})",
        "",
        "- Source: Phase 845 gear interaction rows.",
        "- Method: compare `additive_only` against `interaction_equation` on in-sample / object-holdout / prompt-holdout splits.",
        "- Boundary: this is route-boundary prediction, not token closure.",
        "",
        "## Model Summary",
        "",
        "| model | source rows | predictions | split | predictor | n | MAE delta | RMSE delta | target acc | target F1 | gain F1 | mean interaction MAE gain |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for split, split_row in (data.get("split_summary") or {}).items():
            improvement = split_row.get("mean_interaction_mae_improvement_over_additive")
            for predictor, pred in (split_row.get("predictor_summary") or {}).items():
                lines.append(
                    f"| {model_name} | {data.get('source_rows', 0)} | {data.get('n_predictions', 0)} | "
                    f"`{split}` | `{predictor}` | {pred.get('n', 0)} | {fmt(pred.get('mae_delta'))} | "
                    f"{fmt(pred.get('rmse_delta'))} | {fmt((pred.get('target') or {}).get('accuracy'))} | "
                    f"{fmt((pred.get('target') or {}).get('f1'))} | {fmt((pred.get('gain') or {}).get('f1'))} | "
                    f"{fmt(improvement)} |"
                )

    lines += ["", "## Source Shapes", ""]
    lines += ["| model | objects | prompts | combo counts |"]
    lines += ["|---|---|---|---|"]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | `{', '.join(data.get('source_objects') or [])}` | "
            f"`{', '.join(data.get('source_prompt_variants') or [])}` | "
            f"`{json.dumps(data.get('source_combo_counts') or {}, ensure_ascii=False)}` |"
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
        "model_summaries": {},
        "models": [],
    }
    for model_name in MODELS:
        path = out_dir / f"phase846_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase846_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase846_cross_model_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase845-round", default="smoke")
    parser.add_argument("--split-types", default="in_sample,object_holdout,prompt_holdout")
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
