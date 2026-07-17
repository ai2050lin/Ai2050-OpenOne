#!/usr/bin/env python3
"""Phase446 anti-shortcut static protocol and sample freeze.

This stage does not load model weights and does not use CUDA. It creates a new
protocol version rather than editing Phase442-444 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from model_registry import MODEL_SPECS


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_static_contract"
PROTOCOL_PATH = OUT_DIR / "phase446_protocol_v1_freeze.json"
SAMPLES_PATH = OUT_DIR / "phase446_samples.jsonl"
CERTS_PATH = OUT_DIR / "phase446_semantic_certificates.jsonl"
AUDIT_PATH = OUT_DIR / "phase446_static_audit_report.json"
TOKEN_PATH = OUT_DIR / "phase446_tokenization_report.json"
MANIFEST_PATH = OUT_DIR / "phase446_artifact_manifest.json"

PHASE444_PATH = ROOT / "tests" / "gpt5" / "result" / "phase444_behavior_boundary_analysis" / "phase444_behavior_boundary_analysis.json"

SPLITS = [
    "interface_calibration",
    "behavior_discovery",
    "counterfactual_orbit_holdout",
    "physical_window_freeze",
    "physical_prediction_holdout",
    "sealed_physical_holdout",
]

SPLIT_CODES = {
    "interface_calibration": "ifc",
    "behavior_discovery": "bdv",
    "counterfactual_orbit_holdout": "coh",
    "physical_window_freeze": "pwf",
    "physical_prediction_holdout": "pph",
    "sealed_physical_holdout": "sph",
}

TASKS = {
    "knowledge_network": "relation_truth_judgment",
    "single_step_reasoning": "conditional_implication_truth",
    "syntax_system": "agent_role_truth",
}

PAIRS_PER_SPLIT = 96
LABELS = {"true": "A", "false": "B"}
Z_TWO_SIDED_95 = 1.96

NAMES = [
    "mira", "naro", "lina", "pavo", "suri", "toma", "vexa", "rilo",
    "dina", "kavo", "zuri", "fena", "malo", "sena", "tavi", "nilo",
]
KINDS = ["dax", "nup", "vorn", "kel", "siv", "marn", "tul", "prax"]
PROPS = ["luminous", "brittle", "hollow", "woven", "silent", "curved", "dense", "nimble"]
PREDS = ["glimmers", "rotates", "softens", "aligns", "settles", "tilts", "folds", "rises"]

TRANSFORMS = [
    "lexical_rewrite",
    "order_rewrite",
    "distance_rewrite",
    "boundary_rewrite",
    "syntax_rewrite",
    "query_rewrite",
]


def stable_hash(*parts: object, n: int = 16) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_bounds(k: int, n: int, z: float = Z_TWO_SIDED_95) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def min_k_for_lcb(n: int, threshold: float) -> int | None:
    for k in range(n + 1):
        if wilson_bounds(k, n)[0] >= threshold:
            return k
    return None


def max_k_for_ucb(n: int, threshold: float) -> int | None:
    valid = [k for k in range(n + 1) if wilson_bounds(k, n)[1] <= threshold]
    return max(valid) if valid else None


def pick(pool: list[str], idx: int, offset: int = 0) -> str:
    return pool[(idx + offset) % len(pool)]


def make_context(ability: str, split: str, pair_index: int) -> dict[str, str]:
    code = f"{ability[:2]}_{SPLIT_CODES[split]}_{pair_index:03d}"
    return {
        "name1": f"{pick(NAMES, pair_index)}_{code}_u",
        "name2": f"{pick(NAMES, pair_index, 5)}_{code}_v",
        "kind1": f"{pick(KINDS, pair_index)}_{code}_k1",
        "kind2": f"{pick(KINDS, pair_index, 3)}_{code}_k2",
        "prop1": f"{pick(PROPS, pair_index)}_{code}_p1",
        "prop2": f"{pick(PROPS, pair_index, 4)}_{code}_p2",
        "pred1": f"{pick(PREDS, pair_index)}_{code}_q1",
        "pred2": f"{pick(PREDS, pair_index, 4)}_{code}_q2",
    }


def label(is_true: bool) -> str:
    return LABELS["true" if is_true else "false"]


def instruction() -> str:
    return "Reply with A if the queried statement is true, and B if it is false."


def render_knowledge(split: str, pair_index: int, is_counterfactual: bool) -> dict[str, Any]:
    ctx = make_context("knowledge_network", split, pair_index)
    use_second = pair_index % 2 == 1
    entity = ctx["name2"] if use_second else ctx["name1"]
    true_prop = ctx["prop2"] if use_second else ctx["prop1"]
    false_prop = ctx["prop1"] if use_second else ctx["prop2"]
    query_prop = false_prop if is_counterfactual else true_prop
    is_true = not is_counterfactual
    facts = (
        f"All members of {ctx['kind1']} have trait {ctx['prop1']}. {ctx['name1']} is a member of {ctx['kind1']}. "
        f"All members of {ctx['kind2']} have trait {ctx['prop2']}. {ctx['name2']} is a member of {ctx['kind2']}."
    )
    query = f"Queried statement: {entity} has trait {query_prop}."
    return sample_row("knowledge_network", split, pair_index, is_counterfactual, facts, query, is_true, {
        "entity": entity,
        "query_property": query_prop,
        "first_candidate": ctx["prop1"],
        "last_candidate": ctx["prop2"],
        "semantic_relation": "entity_trait_truth",
    })


def render_reasoning(split: str, pair_index: int, is_counterfactual: bool) -> dict[str, Any]:
    ctx = make_context("single_step_reasoning", split, pair_index)
    use_second = pair_index % 2 == 1
    subject = ctx["name2"] if use_second else ctx["name1"]
    premise = ctx["prop2"] if use_second else ctx["prop1"]
    true_conclusion = ctx["pred2"] if use_second else ctx["pred1"]
    false_conclusion = ctx["pred1"] if use_second else ctx["pred2"]
    query_conclusion = false_conclusion if is_counterfactual else true_conclusion
    is_true = not is_counterfactual
    facts = (
        f"Rule one: if an item has trait {ctx['prop1']}, then it {ctx['pred1']}. "
        f"Rule two: if an item has trait {ctx['prop2']}, then it {ctx['pred2']}. "
        f"Case fact: {subject} has trait {premise}."
    )
    query = f"Queried statement: {subject} {query_conclusion}."
    return sample_row("single_step_reasoning", split, pair_index, is_counterfactual, facts, query, is_true, {
        "subject": subject,
        "premise": premise,
        "query_conclusion": query_conclusion,
        "first_candidate": ctx["pred1"],
        "last_candidate": ctx["pred2"],
        "semantic_relation": "conditional_conclusion_truth",
    })


def render_syntax(split: str, pair_index: int, is_counterfactual: bool) -> dict[str, Any]:
    ctx = make_context("syntax_system", split, pair_index)
    active = pair_index % 2 == 0
    if active:
        sentence = f"{ctx['name1']} followed {ctx['name2']}."
        agent = ctx["name1"]
        patient = ctx["name2"]
    else:
        sentence = f"{ctx['name1']} was followed by {ctx['name2']}."
        agent = ctx["name2"]
        patient = ctx["name1"]
    query_entity = patient if is_counterfactual else agent
    is_true = not is_counterfactual
    facts = f"Sentence: {sentence}"
    query = f"Queried statement: {query_entity} is the agent of the following event."
    return sample_row("syntax_system", split, pair_index, is_counterfactual, facts, query, is_true, {
        "candidate_entity": query_entity,
        "agent": agent,
        "patient": patient,
        "first_candidate": ctx["name1"],
        "last_candidate": ctx["name2"],
        "voice": "active" if active else "passive",
        "semantic_relation": "agent_role_truth",
    })


def sample_row(
    ability: str,
    split: str,
    pair_index: int,
    is_counterfactual: bool,
    facts: str,
    query: str,
    is_true: bool,
    roles: dict[str, str],
) -> dict[str, Any]:
    task = TASKS[ability]
    cf_role = "counterfactual" if is_counterfactual else "base"
    pair_id = stable_hash(ability, split, pair_index, "pair")
    sample_id = stable_hash(ability, split, pair_index, cf_role)
    text = f"{facts} {query} {instruction()}"
    logic = {
        "ability": ability,
        "task": task,
        "pair_id": pair_id,
        "truth_value": is_true,
        "roles": roles,
    }
    semantic_hash = stable_hash(json.dumps(logic, sort_keys=True), n=20)
    variants = [
        {
            "transform": transform,
            "text": transform_text(facts, query, transform),
            "expected_label": label(is_true),
            "semantic_hash": semantic_hash,
        }
        for transform in TRANSFORMS
    ]
    return {
        "sample_id": sample_id,
        "pair_id": pair_id,
        "pair_role": cf_role,
        "ability": ability,
        "task": task,
        "split": split,
        "pair_index": pair_index,
        "facts_text": facts,
        "query_text": query,
        "input_text": text,
        "canonical_answer": label(is_true),
        "answer_aliases": [label(is_true)],
        "truth_value": is_true,
        "content_type": ability,
        "operation_type": roles["semantic_relation"],
        "output_type": "binary_truth_label",
        "role_nodes": roles,
        "logic_form": logic,
        "semantic_hash": semantic_hash,
        "surface_variants": variants,
        "content_hash": stable_hash(text, json.dumps(logic, sort_keys=True), n=32),
    }


def transform_text(facts: str, query: str, transform: str) -> str:
    if transform == "lexical_rewrite":
        return f"Given these records: {facts} Check this claim: {query.removeprefix('Queried statement: ')} {instruction()}"
    if transform == "order_rewrite":
        return f"{query} Evidence: {facts} {instruction()}"
    if transform == "distance_rewrite":
        return f"{facts} Neutral separator: this sentence carries no task information. {query} {instruction()}"
    if transform == "boundary_rewrite":
        return f"{facts.replace('. ', '; ')} {query} {instruction()}"
    if transform == "syntax_rewrite":
        return f"Use only the following facts to judge the statement. {facts} The statement to judge is: {query.removeprefix('Queried statement: ')} {instruction()}"
    if transform == "query_rewrite":
        return f"{facts} Decide whether this is true: {query.removeprefix('Queried statement: ')} {instruction()}"
    raise ValueError(transform)


def build_samples() -> list[dict[str, Any]]:
    rows = []
    renderers = {
        "knowledge_network": render_knowledge,
        "single_step_reasoning": render_reasoning,
        "syntax_system": render_syntax,
    }
    for ability, renderer in renderers.items():
        for split in SPLITS:
            for pair_index in range(PAIRS_PER_SPLIT):
                rows.append(renderer(split, pair_index, False))
                rows.append(renderer(split, pair_index, True))
    return rows


def build_protocol() -> dict[str, Any]:
    n = PAIRS_PER_SPLIT * 2
    return {
        "schema_version": "phase446_antishortcut_static_contract.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "theory_name": "语言是动态模式网络",
        "method_frame": "条件物理状态图谱",
        "status": "anti_shortcut_static_contract_frozen_no_model_run",
        "tasks": TASKS,
        "splits": SPLITS,
        "pairs_per_split_per_task": PAIRS_PER_SPLIT,
        "samples_per_split_per_task": n,
        "surface_transforms": TRANSFORMS,
        "output_labels": {"A": "true", "B": "false"},
        "stop_scope": "model_task_unit",
        "behavior_gates": {
            "semantic_lcb_95_min": 0.85,
            "semantic_min_successes_per_split": min_k_for_lcb(n, 0.85),
            "other_ucb_95_max": 0.05,
            "other_max_failures_per_split": max_k_for_ucb(n, 0.05),
            "shortcut_accuracy_max": 0.55,
            "semantic_gain_min": 0.25,
            "counterfactual_lcb_95_min": 0.85,
            "orbit_group_lcb_95_min": 0.80,
        },
        "forbidden_before_static_pass": [
            "model_behavior_run",
            "physical_trace_collection",
            "causal_intervention",
            "head_channel_neuron_scan",
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def label_leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    leaks = []
    for row in rows:
        for label_value in ("A", "B"):
            if re.search(rf"\b{label_value}\b", row["facts_text"]):
                leaks.append({"sample_id": row["sample_id"], "label": label_value})
    return {"facts_label_leak_count": len(leaks), "pass": not leaks, "examples": leaks[:10]}


def candidate_non_degenerate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    distinct = 0
    for row in rows:
        first = row["role_nodes"].get("first_candidate")
        last = row["role_nodes"].get("last_candidate")
        if first is None or last is None:
            continue
        checked += 1
        distinct += int(first != last)
    rate = distinct / checked if checked else 0.0
    return {"checked": checked, "distinct": distinct, "distinct_rate": rate, "pass": rate >= 0.95}


def split_disjoint_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row["split"]
        code = SPLIT_CODES[split]
        for value in row["role_nodes"].values():
            if isinstance(value, str) and f"_{code}_" in value:
                by_split[split].add(value)
    overlaps = []
    for idx, left in enumerate(SPLITS):
        for right in SPLITS[idx + 1:]:
            shared = by_split[left] & by_split[right]
            if shared:
                overlaps.append({"left": left, "right": right, "count": len(shared), "examples": sorted(shared)[:5]})
    return {"pass": not overlaps, "overlaps": overlaps}


def baseline_predictions(row: dict[str, Any]) -> dict[str, str]:
    first = row["role_nodes"].get("first_candidate", "")
    last = row["role_nodes"].get("last_candidate", "")
    query = row["query_text"]
    query_mentions_first = first in query
    query_mentions_last = last in query
    return {
        "constant_A": "A",
        "constant_B": "B",
        "first_candidate_positive": "A" if query_mentions_first else "B",
        "last_candidate_positive": "A" if query_mentions_last else "B",
        "first_last_xor": "A" if query_mentions_first and not query_mentions_last else "B",
        "query_length_even": "A" if len(query) % 2 == 0 else "B",
        "pair_index_even": "A" if row["pair_index"] % 2 == 0 else "B",
        "voice_active": "A" if row["role_nodes"].get("voice") == "active" else "B",
    }


def static_baseline_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = {}
    failing = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "behavior_discovery":
            grouped[f"{row['ability']}/{row['task']}"].append(row)
    for task_key, task_rows in sorted(grouped.items()):
        reports[task_key] = {}
        for baseline in baseline_predictions(task_rows[0]):
            correct = sum(1 for row in task_rows if baseline_predictions(row)[baseline] == row["canonical_answer"])
            n = len(task_rows)
            acc = correct / n
            reports[task_key][baseline] = {"n": n, "correct": correct, "accuracy": acc}
            if acc > 0.55:
                failing.append({"task": task_key, "baseline": baseline, "accuracy": acc})
    return {"pass": not failing, "reports": reports, "failing_baselines": failing}


def counterfactual_pair_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    bad = []
    for pair_id, pair_rows in by_pair.items():
        if len(pair_rows) != 2:
            bad.append({"pair_id": pair_id, "reason": "pair_size", "size": len(pair_rows)})
            continue
        labels = {row["canonical_answer"] for row in pair_rows}
        if labels != {"A", "B"}:
            bad.append({"pair_id": pair_id, "reason": "labels", "labels": sorted(labels)})
    return {"checked_pairs": len(by_pair), "bad_pair_count": len(bad), "pass": not bad, "examples": bad[:10]}


def semantic_certificate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    variant_count = 0
    for row in rows:
        for variant in row["surface_variants"]:
            variant_count += 1
            if variant["semantic_hash"] != row["semantic_hash"] or variant["expected_label"] != row["canonical_answer"]:
                failures.append({"sample_id": row["sample_id"], "transform": variant["transform"]})
    return {"checked_variants": variant_count, "failure_count": len(failures), "pass": not failures, "examples": failures[:10]}


def old_phase444_anomaly_summary() -> dict[str, Any]:
    if not PHASE444_PATH.exists():
        return {"available": False}
    data = json.loads(PHASE444_PATH.read_text(encoding="utf-8"))
    baselines = data["qwen3_selected_task_baseline_audit"]["task_baselines"]
    anomalies = []
    for task, task_baselines in baselines.items():
        high = [name for name, item in task_baselines.items() if item["accuracy"] >= 0.95]
        if len(high) >= 2:
            anomalies.append({"task": task, "high_baselines": high})
    return {"available": True, "anomalies": anomalies}


def tokenization_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for model_key in ("qwen3", "glm4", "deepseek7b"):
        spec = MODEL_SPECS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(str(spec.local_dir), trust_remote_code=True, local_files_only=True)
        label_lengths = {label: len(tokenizer.encode(label, add_special_tokens=False)) for label in ("A", "B")}
        prompt_lengths = [len(tokenizer.encode(row["input_text"], add_special_tokens=False)) for row in rows]
        reports.append({
            "model": model_key,
            "label_token_lengths": label_lengths,
            "labels_single_token": all(length == 1 for length in label_lengths.values()),
            "max_prompt_tokens": max(prompt_lengths),
            "prompt_over_1024": sum(length > 1024 for length in prompt_lengths),
        })
    return {
        "pass": all(report["labels_single_token"] and report["prompt_over_1024"] == 0 for report in reports),
        "reports": reports,
    }


def build_audit(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    leakage = label_leakage_audit(rows)
    candidates = candidate_non_degenerate_audit(rows)
    disjoint = split_disjoint_audit(rows)
    baselines = static_baseline_audit(rows)
    pairs = counterfactual_pair_audit(rows)
    certs = semantic_certificate_audit(rows)
    tokenization = tokenization_audit(rows)
    all_pass = all(section["pass"] for section in [leakage, candidates, disjoint, baselines, pairs, certs, tokenization])
    return {
        "schema_version": "phase446_static_audit_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_contract_pass_no_model_run" if all_pass else "static_contract_fail_no_model_run",
        "protocol_schema_version": protocol["schema_version"],
        "old_phase444_anomaly_summary": old_phase444_anomaly_summary(),
        "label_leakage": leakage,
        "candidate_non_degenerate": candidates,
        "split_disjoint": disjoint,
        "static_baselines": baselines,
        "counterfactual_pairs": pairs,
        "semantic_certificates": certs,
        "tokenization": tokenization,
        "authorization": {
            "behavior_run_authorized": all_pass,
            "physical_run_authorized": False,
            "reason": "Phase446 has only completed static anti-shortcut gates.",
        },
    }


def write_manifest(paths: list[Path]) -> None:
    artifacts = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    manifest = {
        "schema_version": "phase446_artifact_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_no_model_run",
        "artifacts": artifacts,
        "joint_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol()
    rows = build_samples()
    certs = [
        {
            "sample_id": row["sample_id"],
            "pair_id": row["pair_id"],
            "canonical_answer": row["canonical_answer"],
            "logic_form": row["logic_form"],
            "semantic_hash": row["semantic_hash"],
            "surface_variant_hashes": {variant["transform"]: variant["semantic_hash"] for variant in row["surface_variants"]},
        }
        for row in rows
    ]
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, rows)
    write_jsonl(CERTS_PATH, certs)
    audit = build_audit(protocol, rows)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TOKEN_PATH.write_text(json.dumps(audit["tokenization"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest([PROTOCOL_PATH, SAMPLES_PATH, CERTS_PATH, AUDIT_PATH, TOKEN_PATH])
    print(AUDIT_PATH)


if __name__ == "__main__":
    main()
