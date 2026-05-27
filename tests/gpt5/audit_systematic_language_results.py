from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from model_registry import REPO_ROOT


def load_results(input_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*_systematic_language.json")):
        if path.name.endswith(".partial.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        results[data["model"]] = data
    if not results:
        raise SystemExit(f"No *_systematic_language.json files found in {input_dir}")
    return results


def best_choice(row: dict[str, Any], score_key: str = "full_logprob") -> dict[str, Any]:
    return max(row["choice_scores"], key=lambda item: float(item[score_key]))


def slim_case(row: dict[str, Any]) -> dict[str, Any]:
    pred = best_choice(row)
    pred_index = int(pred["choice_index"])
    return {
        "case_id": row["case_id"],
        "category": row["category"],
        "prompt": row["prompt"],
        "choices": row["choices"],
        "answer_index": row["answer_index"],
        "answer": row["answer"],
        "predicted_index": pred_index,
        "predicted": row["choices"][pred_index],
        "full_margin": row["full_margin"],
        "mean_margin": row["mean_margin"],
        "first_token_margin": row["first_token_margin"],
        "full_correct": row["full_correct"],
        "mean_correct": row["mean_correct"],
        "first_token_correct": row["first_token_correct"],
        "note": row.get("note", ""),
    }


def bucket_margin(margin: float) -> str:
    if margin <= -2:
        return "<=-2"
    if margin <= 0:
        return "(-2,0]"
    if margin <= 1:
        return "(0,1]"
    if margin <= 3:
        return "(1,3]"
    return ">3"


def summarize_model(data: dict[str, Any], top_k: int, low_margin: float) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["cases"]:
        categories[row["category"]].append(row)

    per_category: dict[str, Any] = {}
    for category, rows in sorted(categories.items()):
        failures = [row for row in rows if not row["full_correct"]]
        low_correct = [
            row for row in rows if row["full_correct"] and 0 < float(row["full_margin"]) <= low_margin
        ]
        high_correct = [row for row in rows if row["full_correct"]]
        first_full_disagreements = [
            row for row in rows if bool(row["first_token_correct"]) != bool(row["full_correct"])
        ]
        mean_full_disagreements = [
            row for row in rows if bool(row["mean_correct"]) != bool(row["full_correct"])
        ]

        predicted_counts: dict[str, int] = defaultdict(int)
        answer_counts: dict[str, int] = defaultdict(int)
        bucket_counts: dict[str, int] = defaultdict(int)
        unique_prompts = {row["prompt"] for row in rows}
        unique_prompt_choices = {(row["prompt"], tuple(row["choices"])) for row in rows}
        for row in rows:
            pred = best_choice(row)
            predicted_counts[row["choices"][int(pred["choice_index"])].strip()] += 1
            answer_counts[row["answer"].strip()] += 1
            bucket_counts[bucket_margin(float(row["full_margin"]))] += 1

        per_category[category] = {
            "n": len(rows),
            "unique_prompts": len(unique_prompts),
            "unique_prompt_choices": len(unique_prompt_choices),
            "duplicate_factor": len(rows) / max(1, len(unique_prompt_choices)),
            "accuracy": (len(rows) - len(failures)) / len(rows),
            "num_failures": len(failures),
            "num_low_margin_correct": len(low_correct),
            "num_first_full_disagreements": len(first_full_disagreements),
            "num_mean_full_disagreements": len(mean_full_disagreements),
            "predicted_counts": dict(sorted(predicted_counts.items())),
            "answer_counts": dict(sorted(answer_counts.items())),
            "margin_buckets": {key: bucket_counts.get(key, 0) for key in ["<=-2", "(-2,0]", "(0,1]", "(1,3]", ">3"]},
            "worst_failures": [slim_case(row) for row in sorted(failures, key=lambda x: float(x["full_margin"]))[:top_k]],
            "low_margin_correct": [
                slim_case(row) for row in sorted(low_correct, key=lambda x: float(x["full_margin"]))[:top_k]
            ],
            "high_margin_correct": [
                slim_case(row)
                for row in sorted(high_correct, key=lambda x: float(x["full_margin"]), reverse=True)[:top_k]
            ],
            "first_full_disagreements": [slim_case(row) for row in first_full_disagreements[:top_k]],
            "mean_full_disagreements": [slim_case(row) for row in mean_full_disagreements[:top_k]],
        }

    return {
        "model": data["model"],
        "class": data["class"],
        "num_cases": data["num_cases"],
        "complete": data.get("complete"),
        "overall": data["aggregate"]["overall"],
        "per_category": per_category,
    }


def cross_model_audit(results: dict[str, dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_case: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for model, data in results.items():
        for row in data["cases"]:
            by_case[(row["category"], row["case_id"])][model] = row

    category_overlap: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "all_correct": 0,
            "all_wrong": 0,
            "mixed": 0,
            "wrong_by_model": defaultdict(int),
            "all_wrong_examples": [],
            "mixed_examples": [],
        }
    )

    for (category, case_id), model_rows in sorted(by_case.items()):
        entry = category_overlap[category]
        entry["n"] += 1
        wrong_models = [model for model, row in sorted(model_rows.items()) if not row["full_correct"]]
        for model in wrong_models:
            entry["wrong_by_model"][model] += 1

        example = {
            "case_id": case_id,
            "category": category,
            "prompt": next(iter(model_rows.values()))["prompt"],
            "choices": next(iter(model_rows.values()))["choices"],
            "answer": next(iter(model_rows.values()))["answer"],
            "wrong_models": wrong_models,
            "margins": {model: row["full_margin"] for model, row in sorted(model_rows.items())},
        }
        if not wrong_models:
            entry["all_correct"] += 1
        elif len(wrong_models) == len(model_rows):
            entry["all_wrong"] += 1
            if len(entry["all_wrong_examples"]) < top_k:
                entry["all_wrong_examples"].append(example)
        else:
            entry["mixed"] += 1
            if len(entry["mixed_examples"]) < top_k:
                entry["mixed_examples"].append(example)

    clean = {}
    for category, entry in sorted(category_overlap.items()):
        clean[category] = {
            "n": entry["n"],
            "all_correct": entry["all_correct"],
            "all_wrong": entry["all_wrong"],
            "mixed": entry["mixed"],
            "wrong_by_model": dict(sorted(entry["wrong_by_model"].items())),
            "all_wrong_examples": entry["all_wrong_examples"],
            "mixed_examples": entry["mixed_examples"],
        }
    return clean


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Systematic Language Audit v1")
    lines.append("")
    lines.append("## Model Overview")
    lines.append("")
    lines.append("| model | complete | n | full acc | mean acc | first acc | first/full disagree |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, data in sorted(audit["models"].items()):
        overall = data["overall"]
        lines.append(
            "| {model} | {complete} | {n} | {full:.2%} | {mean:.2%} | {first:.2%} | {disagree} |".format(
                model=model,
                complete=data["complete"],
                n=data["num_cases"],
                full=overall["full"]["accuracy"],
                mean=overall["mean"]["accuracy"],
                first=overall["first_token"]["accuracy"],
                disagree=overall["first_full_disagreements"],
            )
        )

    lines.append("")
    lines.append("## Category Accuracy")
    lines.append("")
    lines.append("| model | category | n | unique | dup factor | full acc | failures | low-margin correct |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for model, data in sorted(audit["models"].items()):
        for category, cat in data["per_category"].items():
            lines.append(
                f"| {model} | {category} | {cat['n']} | {cat['unique_prompt_choices']} | "
                f"{cat['duplicate_factor']:.1f} | {cat['accuracy']:.2%} | "
                f"{cat['num_failures']} | {cat['num_low_margin_correct']} |"
            )

    lines.append("")
    lines.append("## Cross Model Overlap")
    lines.append("")
    lines.append("| category | n | all correct | all wrong | mixed | wrong by model |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for category, entry in audit["cross_model"].items():
        wrong_by_model = ", ".join(f"{k}:{v}" for k, v in entry["wrong_by_model"].items())
        lines.append(
            f"| {category} | {entry['n']} | {entry['all_correct']} | {entry['all_wrong']} | "
            f"{entry['mixed']} | {wrong_by_model} |"
        )

    lines.append("")
    lines.append("## Worst Failures")
    for model, data in sorted(audit["models"].items()):
        lines.append("")
        lines.append(f"### {model}")
        for category, cat in data["per_category"].items():
            failures = cat["worst_failures"][:5]
            if not failures:
                continue
            lines.append("")
            lines.append(f"#### {category}")
            for row in failures:
                lines.append(
                    f"- `{row['case_id']}` margin={row['full_margin']:.3f}; "
                    f"answer={row['answer']!r}; predicted={row['predicted']!r}; prompt={row['prompt']!r}"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "results" / "gpt5_systematic_language_benchmark_v1"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "gpt5_systematic_language_audit_v1"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--low-margin", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(input_dir)
    audit = {
        "input_dir": str(input_dir),
        "top_k": args.top_k,
        "low_margin": args.low_margin,
        "models": {
            model: summarize_model(data, top_k=args.top_k, low_margin=args.low_margin)
            for model, data in sorted(results.items())
        },
        "cross_model": cross_model_audit(results, top_k=args.top_k),
    }

    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(audit, output_dir / "audit.md")
    print(f"[audit] wrote {output_dir / 'audit.json'}")
    print(f"[audit] wrote {output_dir / 'audit.md'}")


if __name__ == "__main__":
    main()
