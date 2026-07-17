#!/usr/bin/env python3
"""Phase446 anti-shortcut behavior qualification.

Runs one model per invocation. No physical traces, no interventions.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase446_antishortcut_static_contract import (  # noqa: E402
    PAIRS_PER_SPLIT,
    TASKS,
    TRANSFORMS,
    Z_TWO_SIDED_95,
    wilson_bounds,
)


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_static_contract" / "phase446_samples.jsonl"
AUDIT_PATH = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_static_contract" / "phase446_static_audit_report.json"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")

SEMANTIC_LCB_MIN = 0.85
OTHER_UCB_MAX = 0.05
CF_LCB_MIN = 0.85
ORBIT_LCB_MIN = 0.80
SHORTCUT_MAX = 0.55
GAIN_MIN = 0.25


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_best_shortcuts() -> dict[str, float]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    out = {}
    for task, baselines in audit["static_baselines"]["reports"].items():
        out[task] = max(item["accuracy"] for item in baselines.values())
    return out


def normalize_generated(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s:：,，.;。!！?？\"'`]+", "", text)
    return text[:1].upper() if text else ""


def classify(expected: str, generated: str) -> str:
    token = normalize_generated(generated)
    if token == expected:
        return "semantic"
    if token in {"A", "B"}:
        return "wrong"
    return "other"


def prompt_for(text: str) -> str:
    return f"{text}\nAnswer:"


def generate_batch(model: Any, tokenizer: Any, device: torch.device, prompts: list[str], max_new_tokens: int) -> list[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    results = []
    for row_index, ids in enumerate(output):
        new_ids = ids[int(input_lengths[row_index]):]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


def behavior_rows(rows: list[dict[str, Any]], limit_per_task: int | None) -> list[dict[str, Any]]:
    selected = []
    counts: Counter[str] = Counter()
    for row in rows:
        if row["split"] != "behavior_discovery":
            continue
        key = f"{row['ability']}/{row['task']}"
        if limit_per_task is not None and counts[key] >= limit_per_task:
            continue
        item = dict(row)
        item["eval_text"] = row["input_text"]
        item["transform"] = "base"
        selected.append(item)
        counts[key] += 1
    return selected


def orbit_rows(rows: list[dict[str, Any]], limit_per_task: int | None) -> list[dict[str, Any]]:
    selected = []
    counts: Counter[str] = Counter()
    for row in rows:
        if row["split"] != "counterfactual_orbit_holdout":
            continue
        key = f"{row['ability']}/{row['task']}"
        if limit_per_task is not None and counts[key] >= limit_per_task:
            continue
        counts[key] += 1
        for variant in row["surface_variants"]:
            item = dict(row)
            item["eval_text"] = variant["text"]
            item["transform"] = variant["transform"]
            selected.append(item)
    return selected


def run_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    model_key: str,
    stage: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        generated = generate_batch(model, tokenizer, device, [prompt_for(row["eval_text"]) for row in batch], max_new_tokens)
        for row, gen in zip(batch, generated, strict=True):
            out.append({
                "model": model_key,
                "stage": stage,
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "pair_role": row["pair_role"],
                "ability": row["ability"],
                "task": row["task"],
                "split": row["split"],
                "transform": row["transform"],
                "canonical_answer": row["canonical_answer"],
                "generated": gen,
                "normalized_generated": normalize_generated(gen),
                "classification": classify(row["canonical_answer"], gen),
            })
        if (start // batch_size) % 20 == 0:
            log(f"{model_key} {stage}: {min(start + len(batch), len(rows))}/{len(rows)}")
    return out


def lcb(k: int, n: int) -> float:
    return wilson_bounds(k, n, Z_TWO_SIDED_95)[0]


def ucb(k: int, n: int) -> float:
    return wilson_bounds(k, n, Z_TWO_SIDED_95)[1]


def summarize_behavior(records: list[dict[str, Any]], shortcuts: dict[str, float]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[f"{row['ability']}/{row['task']}"].append(row)
    out = []
    for task_key, rows in sorted(buckets.items()):
        counts = Counter(row["classification"] for row in rows)
        n = len(rows)
        semantic = counts["semantic"]
        other = counts["other"]
        rate = semantic / n if n else 0.0
        shortcut = shortcuts[task_key]
        out.append({
            "task": task_key,
            "n": n,
            "semantic": semantic,
            "wrong": counts["wrong"],
            "other": other,
            "semantic_rate": rate,
            "semantic_lcb_95": lcb(semantic, n),
            "other_ucb_95": ucb(other, n),
            "best_static_shortcut": shortcut,
            "semantic_gain": rate - shortcut,
            "behavior_pass": lcb(semantic, n) >= SEMANTIC_LCB_MIN and ucb(other, n) <= OTHER_UCB_MAX,
            "shortcut_pass": shortcut <= SHORTCUT_MAX and rate - shortcut >= GAIN_MIN,
        })
    return out


def pair_consistency(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[f"{row['ability']}/{row['task']}"][row["pair_id"]].append(row)
    out = []
    for task_key, pairs in sorted(grouped.items()):
        ok = 0
        for pair_rows in pairs.values():
            roles = {row["pair_role"]: row for row in pair_rows}
            ok += int("base" in roles and "counterfactual" in roles and all(row["classification"] == "semantic" for row in roles.values()))
        n = len(pairs)
        out.append({
            "task": task_key,
            "n_pairs": n,
            "consistent_pairs": ok,
            "counterfactual_lcb_95": lcb(ok, n),
            "counterfactual_pass": lcb(ok, n) >= CF_LCB_MIN,
        })
    return out


def orbit_consistency(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[f"{row['ability']}/{row['task']}"][row["sample_id"]].append(row)
    out = []
    for task_key, samples in sorted(grouped.items()):
        ok = 0
        for sample_rows in samples.values():
            transforms = {row["transform"] for row in sample_rows}
            ok += int(transforms == set(TRANSFORMS) and all(row["classification"] == "semantic" for row in sample_rows))
        n = len(samples)
        out.append({
            "task": task_key,
            "n_samples": n,
            "orbit_consistent_samples": ok,
            "orbit_lcb_95": lcb(ok, n),
            "orbit_pass": lcb(ok, n) >= ORBIT_LCB_MIN,
        })
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_report(model_key: str, behavior: list[dict[str, Any]], orbit: list[dict[str, Any]], shortcuts: dict[str, float]) -> dict[str, Any]:
    behavior_summary = summarize_behavior(behavior, shortcuts)
    cf_summary = pair_consistency(behavior)
    orbit_summary = orbit_consistency(orbit)
    by_task = {}
    for task_key in sorted(shortcuts):
        b = next(row for row in behavior_summary if row["task"] == task_key)
        cf = next(row for row in cf_summary if row["task"] == task_key)
        orb = next(row for row in orbit_summary if row["task"] == task_key)
        by_task[task_key] = {
            "behavior": b,
            "counterfactual": cf,
            "orbit": orb,
            "qualified_for_minimal_physical": (
                b["behavior_pass"] and b["shortcut_pass"] and cf["counterfactual_pass"] and orb["orbit_pass"]
            ),
        }
    return {
        "schema_version": "phase446_antishortcut_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_key,
        "status": "pass" if any(row["qualified_for_minimal_physical"] for row in by_task.values()) else "no_physical_candidate",
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "by_task": by_task,
        "physical_collection_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SAMPLES_PATH)
    shortcuts = load_best_shortcuts()
    model = None
    try:
        model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
        b_rows = behavior_rows(rows, args.limit_per_task)
        behavior = run_generation(model, tokenizer, device, b_rows, args.model, "behavior_discovery", args.batch_size, args.max_new_tokens)
        o_rows = orbit_rows(rows, args.limit_per_task)
        orbit = run_generation(model, tokenizer, device, o_rows, args.model, "counterfactual_orbit_holdout", args.batch_size, args.max_new_tokens)
        report = build_report(args.model, behavior, orbit, shortcuts)
        write_jsonl(OUT_DIR / f"phase446_{args.model}_generations.jsonl", behavior + orbit)
        (OUT_DIR / f"phase446_{args.model}_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(OUT_DIR / f"phase446_{args.model}_summary.json")
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
