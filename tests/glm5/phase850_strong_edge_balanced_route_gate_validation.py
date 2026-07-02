#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase848_internal_route_gate_discovery as p848  # noqa: E402
import phase849_residual_blocker_route_gate_expansion as p849  # noqa: E402


PHASE = 850
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase850_strong_edge_balanced_route_gate_validation")
PHASE849_ROOT = Path("tests/result/phase849_residual_blocker_route_gate_expansion")
PREDICTORS = (
    "global_combo",
    "internal_strength_combo",
    "residual_projection_combo",
    "blocker_field_combo",
    "route_competition_combo",
    "compact_joint_gate_combo",
    "joint_gate_combo",
    "model_default_gate",
    "train_selected_gate",
)
BASE_PREDICTORS = (
    "global_combo",
    "internal_strength_combo",
    "residual_projection_combo",
    "blocker_field_combo",
    "route_competition_combo",
    "compact_joint_gate_combo",
    "joint_gate_combo",
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def source_path(round_name: str, model_name: str) -> Path:
    return PHASE849_ROOT / round_name / f"phase849_{model_name}_feature_rows.jsonl"


def read_rows(round_name: str, model_name: str) -> list[dict[str, Any]]:
    path = source_path(round_name, model_name)
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 849 feature rows: {path}")
    return p846.read_jsonl(path)


def row_id(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(k, ""))
        for k in ["case_id", "object", "prompt_variant", "combo_type", "edit_mode", "combo_key"]
    )


def actual_class(row: dict[str, Any], threshold: float) -> str:
    return p849.residual_class(finite(row.get("actual_residual")), threshold)


def class_is_strong(label: str) -> bool:
    return str(label) in {"synergy", "antagonistic"}


def default_gate_for_model(model_name: str) -> str:
    if model_name == "qwen3":
        return "residual_projection_combo"
    if model_name == "deepseek7b":
        return "blocker_field_combo"
    return "residual_projection_combo"


def split_specs(rows: list[dict[str, Any]], split_types: list[str]):
    return p849.split_specs(rows, split_types)


def binary_stats(actual: list[bool], predicted: list[bool]) -> dict[str, Any]:
    return p846.binary_stats(actual, predicted)


def balanced_accuracy(actual: list[bool], predicted: list[bool]) -> float | None:
    pos = [p for a, p in zip(actual, predicted, strict=False) if a]
    neg = [p for a, p in zip(actual, predicted, strict=False) if not a]
    if not pos or not neg:
        return None
    tpr = sum(1 for p in pos if p) / len(pos)
    tnr = sum(1 for p in neg if not p) / len(neg)
    return (tpr + tnr) / 2.0


def f1_for_label(actual: list[str], predicted: list[str], label: str) -> float:
    a = [x == label for x in actual]
    p = [x == label for x in predicted]
    return binary_stats(a, p)["f1"]


def macro_f1(actual: list[str], predicted: list[str], labels: tuple[str, ...] = ("additive", "synergy", "antagonistic")) -> float | None:
    if not actual:
        return None
    return mean([f1_for_label(actual, predicted, label) for label in labels])


def balanced_subset(rows: list[dict[str, Any]], threshold: float, max_neg_multiplier: int = 1) -> list[dict[str, Any]]:
    positives = [row for row in rows if class_is_strong(actual_class(row, threshold))]
    negatives = [row for row in rows if not class_is_strong(actual_class(row, threshold))]
    if not positives or not negatives:
        return []
    positives = sorted(positives, key=row_id)
    negatives = sorted(negatives, key=row_id)
    max_neg = min(len(negatives), len(positives) * max(1, int(max_neg_multiplier)))
    return positives + negatives[:max_neg]


class StrongGatePredictor:
    def __init__(self, rows: list[dict[str, Any]], model_name: str, threshold: float) -> None:
        self.rows = rows
        self.model_name = model_name
        self.threshold = float(threshold)
        self.mean_predictor = p849.MultiGateResidualPredictor(rows)
        self.class_tables: dict[str, dict[tuple[Any, ...], Counter[str]]] = {}
        for name in BASE_PREDICTORS:
            self.class_tables[name] = self._fit_class_table(name)
        self.type_mode_table: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
        self.global_table: Counter[str] = Counter()
        for row in rows:
            label = actual_class(row, self.threshold)
            self.type_mode_table[(row.get("combo_type"), row.get("edit_mode"))][label] += 1
            self.global_table[label] += 1
        self.thresholds = {name: self._calibrate_threshold(name) for name in BASE_PREDICTORS}
        self.train_scores = {name: self._eval_train_predictor(name) for name in BASE_PREDICTORS}
        self.selected_gate = self._select_gate()

    def resolve_name(self, name: str) -> str:
        if name == "model_default_gate":
            return default_gate_for_model(self.model_name)
        if name == "train_selected_gate":
            return self.selected_gate
        return name

    def _fit_class_table(self, predictor_name: str) -> dict[tuple[Any, ...], Counter[str]]:
        table: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
        for row in self.rows:
            key = self.mean_predictor.key(row, predictor_name)
            table[key][actual_class(row, self.threshold)] += 1
        return table

    def _fallback_counts(self, row: dict[str, Any]) -> Counter[str]:
        type_key = (row.get("combo_type"), row.get("edit_mode"))
        if type_key in self.type_mode_table:
            return self.type_mode_table[type_key]
        return self.global_table

    def class_counts(self, row: dict[str, Any], predictor_name: str) -> Counter[str]:
        name = self.resolve_name(predictor_name)
        order = {
            "joint_gate_combo": [
                "joint_gate_combo",
                "compact_joint_gate_combo",
                "internal_strength_combo",
                "residual_projection_combo",
                "blocker_field_combo",
                "global_combo",
            ],
            "compact_joint_gate_combo": [
                "compact_joint_gate_combo",
                "internal_strength_combo",
                "residual_projection_combo",
                "blocker_field_combo",
                "global_combo",
            ],
            "residual_projection_combo": ["residual_projection_combo", "global_combo"],
            "blocker_field_combo": ["blocker_field_combo", "global_combo"],
            "route_competition_combo": ["route_competition_combo", "global_combo"],
            "internal_strength_combo": ["internal_strength_combo", "global_combo"],
            "global_combo": ["global_combo"],
        }.get(name, [name, "global_combo"])
        for candidate in order:
            table = self.class_tables.get(candidate) or {}
            key = self.mean_predictor.key(row, candidate)
            if key in table:
                return table[key]
        return self._fallback_counts(row)

    def strong_prob(self, row: dict[str, Any], predictor_name: str) -> float:
        counts = self.class_counts(row, predictor_name)
        strong = counts.get("synergy", 0) + counts.get("antagonistic", 0)
        total = sum(counts.values())
        return (strong + 0.5) / (total + 1.0) if total else 0.0

    def direction_label(self, row: dict[str, Any], predictor_name: str) -> str:
        counts = self.class_counts(row, predictor_name)
        if counts.get("synergy", 0) > counts.get("antagonistic", 0):
            return "synergy"
        if counts.get("antagonistic", 0) > counts.get("synergy", 0):
            return "antagonistic"
        residual = self.mean_predictor.predict(row, self.resolve_name(predictor_name))
        if residual >= self.threshold:
            return "synergy"
        if residual <= -self.threshold:
            return "antagonistic"
        return "synergy"

    def predict_class(self, row: dict[str, Any], predictor_name: str) -> str:
        name = self.resolve_name(predictor_name)
        prob = self.strong_prob(row, name)
        cutoff = self.thresholds.get(name, 1.1)
        if prob >= cutoff:
            return self.direction_label(row, name)
        return "additive"

    def _threshold_candidates(self, predictor_name: str) -> list[float]:
        probs = sorted({self.strong_prob(row, predictor_name) for row in self.rows})
        candidates = [0.0, 0.25, 0.5, 0.75, 1.01]
        candidates.extend(probs)
        candidates.extend((a + b) / 2.0 for a, b in zip(probs, probs[1:]))
        return sorted(set(float(x) for x in candidates))

    def _calibrate_threshold(self, predictor_name: str) -> float:
        if not any(class_is_strong(actual_class(row, self.threshold)) for row in self.rows):
            return 1.1
        best_cutoff = 0.5
        best_score = (-1.0, -1.0, -1.0)
        actual = [class_is_strong(actual_class(row, self.threshold)) for row in self.rows]
        for cutoff in self._threshold_candidates(predictor_name):
            pred = [self.strong_prob(row, predictor_name) >= cutoff for row in self.rows]
            stats = binary_stats(actual, pred)
            bal = balanced_accuracy(actual, pred)
            score = (float(stats["f1"]), float(bal if bal is not None else 0.0), float(stats["recall"]))
            if score > best_score:
                best_score = score
                best_cutoff = cutoff
        return best_cutoff

    def _eval_train_predictor(self, predictor_name: str) -> dict[str, Any]:
        rows = self.rows
        predictions = [
            {
                "actual_class": actual_class(row, self.threshold),
                "predicted_class": self.predict_class(row, predictor_name),
            }
            for row in rows
        ]
        return eval_class_predictions(predictions)

    def _select_gate(self) -> str:
        if self.model_name == "qwen3" and "residual_projection_combo" in BASE_PREDICTORS:
            default = "residual_projection_combo"
        elif self.model_name == "deepseek7b" and "blocker_field_combo" in BASE_PREDICTORS:
            default = "blocker_field_combo"
        else:
            default = "residual_projection_combo"
        best = default
        best_score = (-1.0, -1.0, -1.0)
        for name, stats in self.train_scores.items():
            strong = stats.get("strong") or {}
            score = (
                float(strong.get("f1") or 0.0),
                float(stats.get("balanced_accuracy") or 0.0),
                float(strong.get("recall") or 0.0),
            )
            if score > best_score:
                best_score = score
                best = name
        return best


def eval_class_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    actual_labels = [str(row["actual_class"]) for row in predictions]
    predicted_labels = [str(row["predicted_class"]) for row in predictions]
    actual_strong = [class_is_strong(label) for label in actual_labels]
    predicted_strong = [class_is_strong(label) for label in predicted_labels]
    strong = binary_stats(actual_strong, predicted_strong)
    return {
        "n": len(predictions),
        "actual_classes": dict(Counter(actual_labels)),
        "predicted_classes": dict(Counter(predicted_labels)),
        "strong": strong,
        "balanced_accuracy": balanced_accuracy(actual_strong, predicted_strong),
        "macro_f1": macro_f1(actual_labels, predicted_labels),
        "class_accuracy": (
            sum(1 for a, p in zip(actual_labels, predicted_labels, strict=False) if a == p) / len(predictions)
            if predictions
            else None
        ),
    }


def evaluate_split(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    split_type: str,
    split_key: str,
    model_name: str,
    threshold: float,
) -> dict[str, Any]:
    predictor = StrongGatePredictor(train_rows, model_name, threshold)
    prediction_rows: list[dict[str, Any]] = []
    for name in PREDICTORS:
        resolved = predictor.resolve_name(name)
        for row in test_rows:
            actual = actual_class(row, threshold)
            predicted = predictor.predict_class(row, name)
            prediction_rows.append(
                {
                    "row_kind": "phase850_strong_edge_balanced_route_gate_prediction",
                    "phase": PHASE,
                    "model": model_name,
                    "split_type": split_type,
                    "split_key": split_key,
                    "predictor": name,
                    "resolved_predictor": resolved,
                    "selected_gate": predictor.selected_gate,
                    "calibrated_threshold": predictor.thresholds.get(resolved),
                    "case_id": row.get("case_id"),
                    "object": row.get("object"),
                    "prompt_variant": row.get("prompt_variant"),
                    "combo_type": row.get("combo_type"),
                    "edit_mode": row.get("edit_mode"),
                    "combo_key": row.get("combo_key"),
                    "actual_residual": finite(row.get("actual_residual")),
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "actual_strong": class_is_strong(actual),
                    "predicted_strong": class_is_strong(predicted),
                    "strong_prob": predictor.strong_prob(row, name),
                }
            )
    by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_predictor[str(row["predictor"])].append(row)
    raw_summary = {name: eval_class_predictions(rows) for name, rows in sorted(by_predictor.items())}
    balanced_rows = balanced_subset(test_rows, threshold)
    balanced_ids = {row_id(row) for row in balanced_rows}
    balanced_predictions = [
        row for row in prediction_rows if row_id(row) in balanced_ids
    ]
    by_predictor_balanced: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in balanced_predictions:
        by_predictor_balanced[str(row["predictor"])].append(row)
    balanced_summary = {
        name: eval_class_predictions(rows) for name, rows in sorted(by_predictor_balanced.items())
    }
    return {
        "split_type": split_type,
        "split_key": split_key,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "balanced_test_rows": len(balanced_rows),
        "selected_gate": predictor.selected_gate,
        "train_scores": predictor.train_scores,
        "raw_summary": raw_summary,
        "balanced_summary": balanced_summary,
        "predictions": prediction_rows,
    }


def aggregate(split_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in split_results:
        by_split[str(result["split_type"])].append(result)
    out: dict[str, Any] = {}
    for split_type, results in sorted(by_split.items()):
        raw_by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        balanced_by_predictor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        selected = Counter(str(result.get("selected_gate")) for result in results)
        for result in results:
            balanced_ids = set()
            if result.get("balanced_test_rows"):
                test_rows_by_id = {
                    row_id(pred): pred for pred in result.get("predictions", []) if pred.get("predictor") == "global_combo"
                }
                positive_ids = {rid for rid, pred in test_rows_by_id.items() if bool(pred.get("actual_strong"))}
                negative_ids = [rid for rid, pred in sorted(test_rows_by_id.items()) if not bool(pred.get("actual_strong"))]
                balanced_ids = positive_ids | set(negative_ids[: len(positive_ids)])
            for pred in result.get("predictions", []):
                raw_by_predictor[str(pred["predictor"])].append(pred)
                if row_id(pred) in balanced_ids:
                    balanced_by_predictor[str(pred["predictor"])].append(pred)
        out[split_type] = {
            "n_splits": len(results),
            "split_keys": [result["split_key"] for result in results],
            "selected_gate_counts": dict(selected),
            "raw_summary": {
                name: eval_class_predictions(rows) for name, rows in sorted(raw_by_predictor.items())
            },
            "balanced_summary": {
                name: eval_class_predictions(rows) for name, rows in sorted(balanced_by_predictor.items()) if rows
            },
        }
    return out


def feature_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return {
        "n": len(rows),
        "actual_classes": dict(Counter(actual_class(row, threshold) for row in rows)),
        "objects": dict(Counter(str(row.get("object")) for row in rows)),
        "prompt_variants": dict(Counter(str(row.get("prompt_variant")) for row in rows)),
        "strong_rows": sum(1 for row in rows if class_is_strong(actual_class(row, threshold))),
        "mean_abs_residual": mean([abs(finite(row.get("actual_residual"))) for row in rows]),
        "mean_strong_abs_residual": mean(
            [abs(finite(row.get("actual_residual"))) for row in rows if class_is_strong(actual_class(row, threshold))]
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.phase849_round, args.model)
    if args.max_rows and int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    split_types = [part.strip() for part in args.split_types.split(",") if part.strip()]
    specs = split_specs(rows, split_types)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "phase849_round": args.phase849_round,
            "rows": len(rows),
            "feature_summary": feature_summary(rows, float(args.interaction_threshold)),
            "split_count": len(specs),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    split_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for idx, (split_type, split_key, train_rows, test_rows) in enumerate(specs, 1):
        result = evaluate_split(train_rows, test_rows, split_type, split_key, args.model, float(args.interaction_threshold))
        split_results.append(result)
        all_predictions.extend(result["predictions"])
        if idx % max(1, int(args.log_every)) == 0 or idx == len(specs):
            log(
                f"{args.model}/{args.round_name}: evaluated split {idx}/{len(specs)} "
                f"{split_type}:{split_key} selected={result['selected_gate']}"
            )
    compact_splits = [{k: v for k, v in result.items() if k != "predictions"} for result in split_results]
    summary = {
        "phase": PHASE,
        "title": "Strong-edge Balanced Route Gate Validation",
        "model": args.model,
        "round": args.round_name,
        "phase849_round": args.phase849_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_rows": len(rows),
        "n_predictions": len(all_predictions),
        "feature_summary": feature_summary(rows, float(args.interaction_threshold)),
        "split_summary": aggregate(split_results),
        "boundary": (
            "This phase re-evaluates Phase 849 gates with strong-edge-focused calibrated class predictors "
            "and balanced metrics. It does not run new model forward passes and does not claim closure."
        ),
    }
    p846.write_jsonl(out_dir / f"phase850_{args.model}_predictions.jsonl", all_predictions)
    p846.write_json(out_dir / f"phase850_{args.model}_split_results.json", compact_splits)
    p846.write_json(out_dir / f"phase850_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "feature_summary": summary["feature_summary"],
                "split_summary": {
                    split: {
                        "selected_gate_counts": row.get("selected_gate_counts"),
                        "raw": {
                            predictor: {
                                "n": stats.get("n"),
                                "strong_f1": (stats.get("strong") or {}).get("f1"),
                                "strong_recall": (stats.get("strong") or {}).get("recall"),
                                "balanced_accuracy": stats.get("balanced_accuracy"),
                                "macro_f1": stats.get("macro_f1"),
                            }
                            for predictor, stats in (row.get("raw_summary") or {}).items()
                        },
                        "balanced": {
                            predictor: {
                                "n": stats.get("n"),
                                "strong_f1": (stats.get("strong") or {}).get("f1"),
                                "strong_recall": (stats.get("strong") or {}).get("recall"),
                                "balanced_accuracy": stats.get("balanced_accuracy"),
                                "macro_f1": stats.get("macro_f1"),
                            }
                            for predictor, stats in (row.get("balanced_summary") or {}).items()
                        },
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
        f"# Phase 850 Strong-edge Balanced Route Gate Validation ({payload['round']})",
        "",
        "- Source: Phase 849 feature rows.",
        "- Method: calibrated strong-edge class predictors with raw and balanced metrics.",
        "- Boundary: strong-edge validation, not closure.",
        "",
        "## Raw Strong-edge Summary",
        "",
        "| model | rows | strong rows | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        fs = data.get("feature_summary") or {}
        for split, split_row in (data.get("split_summary") or {}).items():
            for predictor, stats in (split_row.get("raw_summary") or {}).items():
                strong = stats.get("strong") or {}
                lines.append(
                    f"| {model_name} | {data.get('source_rows', 0)} | {fs.get('strong_rows', 0)} | "
                    f"`{split}` | `{predictor}` | {stats.get('n', 0)} | {fmt(strong.get('f1'))} | "
                    f"{fmt(strong.get('recall'))} | {fmt(strong.get('precision'))} | "
                    f"{fmt(stats.get('balanced_accuracy'))} | {fmt(stats.get('macro_f1'))} |"
                )
    lines += ["", "## Balanced Subset Summary", ""]
    lines += ["| model | split | predictor | n | strong F1 | recall | precision | balanced acc | macro F1 |"]
    lines += ["|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for split, split_row in (data.get("split_summary") or {}).items():
            for predictor, stats in (split_row.get("balanced_summary") or {}).items():
                strong = stats.get("strong") or {}
                lines.append(
                    f"| {model_name} | `{split}` | `{predictor}` | {stats.get('n', 0)} | "
                    f"{fmt(strong.get('f1'))} | {fmt(strong.get('recall'))} | {fmt(strong.get('precision'))} | "
                    f"{fmt(stats.get('balanced_accuracy'))} | {fmt(stats.get('macro_f1'))} |"
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
        path = out_dir / f"phase850_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = p846.read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase850_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase850_cross_model_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase849-round", default="smoke")
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
