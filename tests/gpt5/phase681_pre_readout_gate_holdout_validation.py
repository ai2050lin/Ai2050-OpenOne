#!/usr/bin/env python3
"""
Phase 681: Holdout Validation for Pre-Readout Failure Gates.

Phase 680 enumerated simple pre-readout threshold gates on the same rows that
were evaluated. This script checks whether the best gates survive a deterministic
train/test split. No model forward pass is used.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase680_pre_readout_natural_gate_cross_family_audit import (  # noqa: E402
    OUT_ROOT as PHASE680_ROOT,
    eval_gate,
    make_gates,
    rank_evals,
)


OUT_ROOT = Path("results/glm5_phase681_pre_readout_gate_holdout_validation")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_alternate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = sorted(rows, key=lambda r: r["case_id"])
    train = [r for i, r in enumerate(rows) if i % 2 == 0]
    test = [r for i, r in enumerate(rows) if i % 2 == 1]
    return train, test


def eligible(rows: list[dict], min_fail: int, min_success: int) -> bool:
    failures = sum(1 for r in rows if not r["expected_top1"])
    successes = len(rows) - failures
    return failures >= min_fail and successes >= min_success


def choose_and_eval(train: list[dict], test: list[dict], include_reference: bool) -> dict | None:
    gates = make_gates(train, include_reference=include_reference)
    if not include_reference:
        gates = [g for g in gates if g["kind"] != "near_readout_reference"]
    else:
        gates = [g for g in gates if g["kind"] == "near_readout_reference"]
    ranked = rank_evals([eval_gate(train, g) for g in gates])
    if not ranked:
        return None
    best_train = ranked[0]
    gate = next(g for g in gates if g["name"] == best_train["gate"])
    test_eval = eval_gate(test, gate)
    return {
        "gate": gate["name"],
        "kind": gate["kind"],
        "feature": gate["feature"],
        "train": best_train,
        "test": test_eval,
    }


def validate_model(model: str, min_fail: int, min_success: int) -> dict:
    rows = read_jsonl(PHASE680_ROOT / f"phase680_{model}_pre_readout_rows.jsonl")
    families = sorted({r["family"] for r in rows})
    groups = {"overall": rows}
    groups.update({fam: [r for r in rows if r["family"] == fam] for fam in families})

    validations = []
    for group, group_rows in groups.items():
        train, test = split_alternate(group_rows)
        if not eligible(train, min_fail, min_success) or len(test) == 0:
            continue
        pre = choose_and_eval(train, test, include_reference=False)
        ref = choose_and_eval(train, test, include_reference=True)
        validations.append({
            "group": group,
            "n_train": len(train),
            "n_test": len(test),
            "train_failures": sum(1 for r in train if not r["expected_top1"]),
            "test_failures": sum(1 for r in test if not r["expected_top1"]),
            "pre_readout": pre,
            "near_readout_reference": ref,
        })

    cross_family = []
    for source in families:
        src = [r for r in rows if r["family"] == source]
        train, source_holdout = split_alternate(src)
        if not eligible(train, min_fail, min_success):
            continue
        pre = choose_and_eval(train, source_holdout, include_reference=False)
        if pre is None:
            continue
        gate_name = pre["gate"]
        gates = make_gates(train, include_reference=False)
        gate = next(g for g in gates if g["name"] == gate_name)
        for target in families:
            if target == source:
                continue
            target_rows = [r for r in rows if r["family"] == target]
            cross_family.append({
                "source_family": source,
                "target_family": target,
                "gate": gate_name,
                "kind": gate["kind"],
                "source_train": pre["train"],
                "source_holdout": pre["test"],
                "target_eval": eval_gate(target_rows, gate),
            })

    return {
        "model": model,
        "n_rows": len(rows),
        "validations": validations,
        "cross_family": cross_family,
    }


def write_markdown(payload: dict) -> None:
    lines = [
        "# Phase 681 Pre-Readout Gate Holdout Validation",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | group | train_fail | test_fail | pre gate | train score | test score | test capture | test false_pos | ref gate | ref test score |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for model in payload["models"]:
        for item in model["validations"]:
            pre = item["pre_readout"] or {"gate": "NA", "train": {}, "test": {}}
            ref = item["near_readout_reference"] or {"gate": "NA", "test": {}}
            lines.append(
                f"| {model['model']} | {item['group']} | {item['train_failures']} | {item['test_failures']} | "
                f"{pre.get('gate', 'NA')} | {pre.get('train', {}).get('gate_score', 0.0):.3f} | "
                f"{pre.get('test', {}).get('gate_score', 0.0):.3f} | "
                f"{pre.get('test', {}).get('failure_capture_rate', 0.0):.3f} | "
                f"{pre.get('test', {}).get('success_false_positive_rate', 0.0):.3f} | "
                f"{ref.get('gate', 'NA')} | {ref.get('test', {}).get('gate_score', 0.0):.3f} |"
            )
    lines.extend(["", "## Cross-Family Checks", ""])
    lines.append("| model | source | target | gate | source_holdout_score | target_score | target_capture | target_false_pos |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for model in payload["models"]:
        ranked = sorted(
            model["cross_family"],
            key=lambda r: (
                -r["target_eval"]["gate_score"],
                -r["target_eval"]["failure_capture_rate"],
                r["target_eval"]["success_false_positive_rate"],
            ),
        )[:20]
        for item in ranked:
            te = item["target_eval"]
            lines.append(
                f"| {model['model']} | {item['source_family']} | {item['target_family']} | {item['gate']} | "
                f"{item['source_holdout']['gate_score']:.3f} | {te['gate_score']:.3f} | "
                f"{te['failure_capture_rate']:.3f} | {te['success_false_positive_rate']:.3f} |"
            )
    (OUT_ROOT / "phase681_holdout_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 681,
        "title": "Holdout Validation for Pre-Readout Failure Gates",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [validate_model(model, min_fail=3, min_success=3) for model in MODELS],
    }
    (OUT_ROOT / "phase681_holdout_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
