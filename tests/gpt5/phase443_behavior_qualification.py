#!/usr/bin/env python3
"""Phase443 behavior qualification for the Phase442 frozen samples.

This script loads exactly one model per run, uses greedy short generation, and
does not collect physical traces or run interventions.
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


SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase442_static_sample_contract" / "phase442_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase443_behavior_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")
ABILITIES = ("knowledge_network", "single_step_reasoning", "syntax_system")

SEMANTIC_MIN = 75
OTHER_MAX = 0
ORBIT_MIN = 72
SURFACE_GAP_MAX = 0.05

ANSWER_POOL = ["red", "blue", "green", "gold"]
BOUNDARY = ["complete", "open"]
SINGULAR_VERBS = ["glows", "turns", "moves", "rests"]
PLURAL_VERBS = ["glow", "turn", "move", "rest"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[\s:：,，.;。!！?？\"'`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_like(value: str) -> str:
    return norm_text(value).strip(".,;:!?，。；：！？\"'`")


def wrong_candidates(row: dict[str, Any]) -> list[str]:
    answer = row["canonical_answer"]
    wrongs = list(row.get("wrong_answers") or [])
    if answer in ANSWER_POOL:
        wrongs.extend(item for item in ANSWER_POOL if item != answer)
    if answer in BOUNDARY:
        wrongs.extend(item for item in BOUNDARY if item != answer)
    if answer in SINGULAR_VERBS:
        wrongs.append(PLURAL_VERBS[SINGULAR_VERBS.index(answer)])
    if answer in PLURAL_VERBS:
        wrongs.append(SINGULAR_VERBS[PLURAL_VERBS.index(answer)])
    role_nodes = row.get("role_nodes", {})
    for key in ("distractor", "entity", "category", "attribute"):
        value = role_nodes.get(key)
        if value and value != answer:
            wrongs.append(value)
    return sorted(set(wrongs))


def classify(row: dict[str, Any], generated: str) -> str:
    out = norm_text(generated)
    aliases = [token_like(alias) for alias in row["answer_aliases"]]
    wrongs = [token_like(item) for item in wrong_candidates(row)]
    for alias in aliases:
        if out == alias or out.startswith(alias + " ") or out.startswith(alias + ".") or alias in out[: max(64, len(alias) + 8)]:
            return "semantic"
    for wrong in wrongs:
        if wrong and (out == wrong or out.startswith(wrong + " ") or wrong in out[: max(64, len(wrong) + 8)]):
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
        new_ids = ids[int(input_lengths[row_index]) :]
        results.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return results


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
    out_rows = []
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [prompt_for(row["eval_text"]) for row in batch]
        generated = generate_batch(model, tokenizer, device, prompts, max_new_tokens)
        for row, gen in zip(batch, generated, strict=True):
            label = classify(row, gen)
            out_rows.append(
                {
                    "model": model_key,
                    "stage": stage,
                    "sample_id": row["sample_id"],
                    "ability": row["ability"],
                    "task": row["task"],
                    "split": row["split"],
                    "transform": row.get("transform", "base"),
                    "canonical_answer": row["canonical_answer"],
                    "generated": gen,
                    "classification": label,
                }
            )
        if (start // batch_size) % 20 == 0:
            log(f"{model_key} {stage}: {min(start + len(batch), len(rows))}/{len(rows)}")
    return out_rows


def discovery_rows(rows: list[dict[str, Any]], limit_per_task: int | None = None) -> list[dict[str, Any]]:
    selected = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row["split"] != "task_discovery":
            continue
        key = (row["ability"], row["task"])
        if limit_per_task is not None and counts[key] >= limit_per_task:
            continue
        item = dict(row)
        item["eval_text"] = row["input_text"]
        item["transform"] = "base"
        selected.append(item)
        counts[key] += 1
    return selected


def orbit_rows(rows: list[dict[str, Any]], selected_tasks: dict[str, str], limit_per_task: int | None = None) -> list[dict[str, Any]]:
    selected = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row["split"] != "surface_orbit_holdout":
            continue
        if selected_tasks.get(row["ability"]) != row["task"]:
            continue
        key = (row["ability"], row["task"])
        if limit_per_task is not None and counts[key] >= limit_per_task:
            continue
        for variant in row["surface_variants"]:
            item = dict(row)
            item["eval_text"] = variant["text"]
            item["transform"] = variant["transform"]
            selected.append(item)
        counts[key] += 1
    return selected


def summarize(records: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in records:
        buckets[tuple(row[key] for key in keys)][row["classification"]] += 1
    out = []
    for key, counts in sorted(buckets.items()):
        total = sum(counts.values())
        item = {name: value for name, value in zip(keys, key, strict=True)}
        item.update(
            {
                "n": total,
                "semantic": counts["semantic"],
                "wrong": counts["wrong"],
                "other": counts["other"],
                "semantic_rate": counts["semantic"] / total if total else 0.0,
                "other_rate": counts["other"] / total if total else 0.0,
                "hard_pass": counts["semantic"] >= SEMANTIC_MIN and counts["other"] <= OTHER_MAX,
            }
        )
        out.append(item)
    return out


def choose_tasks(discovery_summary: list[dict[str, Any]]) -> dict[str, str]:
    selected = {}
    for ability in ABILITIES:
        candidates = [row for row in discovery_summary if row["ability"] == ability and row["hard_pass"]]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (-row["semantic_rate"], row["other"], row["task"]))
        selected[ability] = candidates[0]["task"]
    return selected


def orbit_group_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[(row["ability"], row["task"], row["sample_id"])].append(row)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for (ability, task, _sample_id), rows in by_group.items():
        ok = rows and all(row["classification"] == "semantic" for row in rows)
        counts[(ability, task)]["consistent" if ok else "inconsistent"] += 1
    out = []
    for (ability, task), counter in sorted(counts.items()):
        n = counter["consistent"] + counter["inconsistent"]
        out.append(
            {
                "ability": ability,
                "task": task,
                "n_groups": n,
                "consistent": counter["consistent"],
                "inconsistent": counter["inconsistent"],
                "orbit_group_pass": counter["consistent"] >= ORBIT_MIN,
            }
        )
    return out


def build_report(model_key: str, discovery: list[dict[str, Any]], orbit: list[dict[str, Any]]) -> dict[str, Any]:
    discovery_summary = summarize(discovery, ["ability", "task"])
    selected = choose_tasks(discovery_summary)
    orbit_by_transform = summarize(orbit, ["ability", "task", "transform"])
    orbit_groups = orbit_group_summary(orbit)
    final = {}
    for ability in ABILITIES:
        task = selected.get(ability)
        if not task:
            final[ability] = {"selected_task": None, "status": "no_task_passed_discovery"}
            continue
        task_transforms = [row for row in orbit_by_transform if row["ability"] == ability and row["task"] == task]
        rates = [row["semantic_rate"] for row in task_transforms]
        gap = max(rates) - min(rates) if rates else 1.0
        group = next((row for row in orbit_groups if row["ability"] == ability and row["task"] == task), None)
        final[ability] = {
            "selected_task": task,
            "surface_orbit_max_gap": gap,
            "surface_gap_pass": gap <= SURFACE_GAP_MAX,
            "orbit_group_pass": bool(group and group["orbit_group_pass"]),
            "orbit_group": group,
            "status": "pass" if gap <= SURFACE_GAP_MAX and group and group["orbit_group_pass"] else "fail_surface_orbit",
        }
    return {
        "schema_version": "phase443_behavior_qualification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_key,
        "cuda_used": torch.cuda.is_available(),
        "model_weights_loaded": True,
        "status": "pass" if all(value["status"] == "pass" for value in final.values()) else "fail",
        "discovery_summary": discovery_summary,
        "selected_tasks": selected,
        "surface_orbit_by_transform": orbit_by_transform,
        "surface_orbit_group_summary": orbit_groups,
        "final_by_ability": final,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows()
    model = None
    try:
        model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
        d_rows = discovery_rows(all_rows, args.limit_per_task)
        discovery = run_generation(model, tokenizer, device, d_rows, args.model, "task_discovery", args.batch_size, args.max_new_tokens)
        d_summary = summarize(discovery, ["ability", "task"])
        selected = choose_tasks(d_summary)
        log(f"{args.model} selected_tasks={selected}")
        o_rows = orbit_rows(all_rows, selected, args.limit_per_task)
        orbit = run_generation(model, tokenizer, device, o_rows, args.model, "surface_orbit_holdout", args.batch_size, args.max_new_tokens) if o_rows else []
        report = build_report(args.model, discovery, orbit)
        write_jsonl(OUT_DIR / f"phase443_{args.model}_generations.jsonl", discovery + orbit)
        (OUT_DIR / f"phase443_{args.model}_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(OUT_DIR / f"phase443_{args.model}_summary.json")
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
