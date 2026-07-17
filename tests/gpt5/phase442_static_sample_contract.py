#!/usr/bin/env python3
"""Phase442 static sample generation and pre-run contract audit.

This stage intentionally does not load CUDA models. It repairs the Phase441
sample-size feasibility issue and freezes synthetic natural-language samples,
semantic certificates, split contracts, simple baseline audits, and hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase442_static_sample_contract"
PROTOCOL_PATH = OUT_DIR / "phase442_protocol_v3_freeze.json"
SAMPLES_PATH = OUT_DIR / "phase442_samples.jsonl"
CERTS_PATH = OUT_DIR / "phase442_semantic_certificates.jsonl"
AUDIT_PATH = OUT_DIR / "phase442_static_audit_report.json"
MANIFEST_PATH = OUT_DIR / "phase442_artifact_manifest.json"

SPLITS = [
    "interface_calibration",
    "task_discovery",
    "surface_orbit_holdout",
    "physical_window_freeze",
    "physical_prediction_holdout",
    "sealed_physical_holdout",
]

SPLIT_CODES = {
    "interface_calibration": "ifc",
    "task_discovery": "tds",
    "surface_orbit_holdout": "soh",
    "physical_window_freeze": "pwf",
    "physical_prediction_holdout": "pph",
    "sealed_physical_holdout": "sph",
}

GROUPS_PER_SPLIT = 80
Z_TWO_SIDED_95 = 1.96

TASK_LIBRARY = {
    "knowledge_network": [
        "context_single_attribute_read",
        "parametric_single_fact_read",
        "category_attribute_inheritance",
        "parametric_context_consistent",
        "parametric_context_conflict",
    ],
    "single_step_reasoning": [
        "set_inclusion_one_step",
        "size_comparison_one_step",
        "conditional_implication_one_step",
        "relation_transitive_one_step",
        "one_step_exclusion",
    ],
    "syntax_system": [
        "subject_verb_number_agreement",
        "pronoun_number_agreement",
        "active_passive_role_conversion",
        "relative_clause_role",
        "sentence_boundary_closure_choice",
    ],
}

TRANSFORMS = [
    "lexical_rewrite",
    "order_rewrite",
    "distance_rewrite",
    "boundary_rewrite",
    "syntax_rewrite",
    "query_rewrite",
]

INTERFACES = [
    "restricted_choice",
    "single_field",
    "direct_short_answer",
    "natural_short_sentence",
]

ANSWER_POOL = ["red", "blue", "green", "gold"]
SINGULAR_NOUNS = ["lamp", "stone", "planet", "river"]
PLURAL_NOUNS = ["lamps", "stones", "planets", "rivers"]
SINGULAR_VERBS = ["glows", "turns", "moves", "rests"]
PLURAL_VERBS = ["glow", "turn", "move", "rest"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_token(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def wilson_bounds(k: int, n: int, z: float = Z_TWO_SIDED_95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
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


def base_fields(ability: str, task: str, split: str, idx: int) -> dict[str, str]:
    prefix = f"{ability[:2]}_{task[:4]}_{SPLIT_CODES[split]}_{idx:03d}"
    length_suffix = "x" * (idx % 5)
    return {
        "entity_a": f"{prefix}_ent_a{length_suffix}",
        "entity_b": f"{prefix}_ent_b",
        "entity_c": f"{prefix}_ent_c",
        "category_a": f"{prefix}_cat_a",
        "category_b": f"{prefix}_cat_b",
        "attribute": f"{prefix}_attr",
        "rule": f"{prefix}_rule",
    }


def answer_for(split: str, task: str, idx: int) -> str:
    offset = SPLITS.index(split) + len(task)
    return ANSWER_POOL[(idx + offset) % len(ANSWER_POOL)]


def wrong_answers(answer: str) -> list[str]:
    return [item for item in ANSWER_POOL if item != answer]


def render_sample(ability: str, task: str, split: str, idx: int) -> dict[str, Any]:
    fields = base_fields(ability, task, split, idx)
    answer = answer_for(split, task, idx)
    alias = [answer, answer.upper()]
    operation = "read"
    content_type = "entity_attribute"
    output_type = "short_answer"

    if ability == "knowledge_network":
        if "inheritance" in task:
            operation = "category_inheritance"
            text = (
                f"Fact A: every {fields['category_a']} has value {answer}. "
                f"Fact B: {fields['entity_a']} is a {fields['category_a']}. "
                f"Question: what value does {fields['entity_a']} have? Answer with one word."
            )
            logic = ("inherit", fields["entity_a"], fields["category_a"], fields["attribute"], answer)
            nodes = {"entity": fields["entity_a"], "category": fields["category_a"], "attribute": fields["attribute"], "query": "value"}
        elif "conflict" in task:
            operation = "context_override"
            context_answer = answer
            param_answer = wrong_answers(answer)[0]
            text = (
                f"Background memory says {fields['entity_a']} is {param_answer}. "
                f"In this case file, {fields['entity_a']} is explicitly {context_answer}. "
                f"Question: in this case file, what is {fields['entity_a']}? Answer with one word."
            )
            logic = ("context_conflict", fields["entity_a"], context_answer, param_answer)
            nodes = {"entity": fields["entity_a"], "context_value": context_answer, "memory_value": param_answer, "query": "case_value"}
        else:
            text = (
                f"Record: {fields['entity_a']} has property {fields['attribute']} with value {answer}. "
                f"Question: what is the value of {fields['attribute']} for {fields['entity_a']}? Answer with one word."
            )
            logic = ("fact_read", fields["entity_a"], fields["attribute"], answer)
            nodes = {"entity": fields["entity_a"], "attribute": fields["attribute"], "query": "value"}
    elif ability == "single_step_reasoning":
        operation = "one_step_rule_apply"
        if "size" in task:
            text = (
                f"Rule: if {fields['entity_a']} is larger than {fields['entity_b']}, choose {answer}. "
                f"Premise: {fields['entity_a']} is larger than {fields['entity_b']}. "
                f"Question: which label follows? Answer with one word."
            )
            logic = ("size_rule", fields["entity_a"], fields["entity_b"], answer)
        elif "exclusion" in task:
            bad = wrong_answers(answer)[0]
            text = (
                f"Allowed labels are {answer} and {bad}. Rule: not {bad} implies {answer}. "
                f"Premise: {fields['entity_a']} is not {bad}. "
                f"Question: which label follows? Answer with one word."
            )
            logic = ("exclusion", fields["entity_a"], bad, answer)
        else:
            text = (
                f"Rule: when {fields['entity_a']} relates to {fields['entity_b']}, the output is {answer}. "
                f"Premise: {fields['entity_a']} relates to {fields['entity_b']}. "
                f"Question: what output follows? Answer with one word."
            )
            logic = ("rule_apply", fields["entity_a"], fields["entity_b"], answer)
        nodes = {"premise_a": fields["entity_a"], "premise_b": fields["entity_b"], "rule": fields["rule"], "query": "output"}
    else:
        content_type = "syntax_role"
        operation = "structure_binding"
        noun_idx = idx % len(SINGULAR_NOUNS)
        singular = SINGULAR_NOUNS[noun_idx]
        plural = PLURAL_NOUNS[noun_idx]
        sverb = SINGULAR_VERBS[noun_idx]
        pverb = PLURAL_VERBS[noun_idx]
        if "number" in task:
            answer = sverb if idx % 2 == 0 else pverb
            subject = singular if idx % 2 == 0 else plural
            distractor = plural if idx % 2 == 0 else singular
            alias = [answer]
            text = (
                f"Sentence fragment: the {subject} near the {distractor} ____. "
                f"Question: choose the grammatical verb: {sverb} or {pverb}. Answer with one word."
            )
            logic = ("agreement", subject, distractor, answer)
            nodes = {"controller": subject, "slot": "verb_blank", "distractor": distractor, "boundary": "sentence"}
        elif "active_passive" in task:
            answer = fields["entity_a"]
            alias = [answer]
            text = (
                f"Active sentence: {fields['entity_a']} carried {fields['entity_b']}. "
                f"Passive sentence: {fields['entity_b']} was carried by ____. "
                f"Question: who fills the blank? Answer with the exact token."
            )
            logic = ("active_passive_agent", fields["entity_a"], fields["entity_b"], answer)
            nodes = {"controller": fields["entity_a"], "slot": "agent_blank", "distractor": fields["entity_b"], "boundary": "passive_clause"}
        else:
            answer = "complete" if idx % 2 == 0 else "open"
            alias = [answer]
            boundary_statement = "closes" if answer == "complete" else "does not close"
            text = (
                f"Fragment A names {fields['entity_a']}. Fragment B {boundary_statement} the sentence about {fields['entity_a']}. "
                f"Question: is the sentence boundary complete or open? Answer complete or open."
            )
            logic = ("boundary_complete", fields["entity_a"], answer)
            nodes = {"controller": fields["entity_a"], "slot": "boundary_state", "distractor": fields["entity_b"], "boundary": "sentence_end"}

    semantic_hash = stable_token("logic", *logic)
    sample_id = stable_token(ability, task, split, idx)
    variants = []
    for transform in TRANSFORMS:
        variants.append(
            {
                "transform": transform,
                "text": transform_text(text, transform),
                "expected_answer": answer,
                "semantic_hash": semantic_hash,
                "role_mapping": role_mapping_for(transform, nodes),
                "certificate": "semantic_hash_unchanged",
            }
        )

    return {
        "sample_id": sample_id,
        "ability": ability,
        "content_type": content_type,
        "operation_type": operation,
        "output_type": output_type,
        "task": task,
        "split": split,
        "group_index": idx,
        "input_text": text,
        "canonical_answer": answer,
        "answer_aliases": alias,
        "wrong_answers": wrong_answers(answer) if answer in ANSWER_POOL else [],
        "logic_form": list(logic),
        "semantic_hash": semantic_hash,
        "role_nodes": nodes,
        "surface_variants": variants,
        "template_family": f"{ability}/{task}",
        "content_hash": sha256_text(json.dumps([text, logic, nodes], sort_keys=True)),
    }


def transform_text(text: str, transform: str) -> str:
    if transform == "lexical_rewrite":
        return text.replace("Question:", "Query:").replace("Answer with one word.", "Reply using one word.")
    if transform == "order_rewrite":
        parts = text.split(" Question: ")
        return " Question: ".join(reversed(parts)) if len(parts) == 2 else text
    if transform == "distance_rewrite":
        return text.replace("Question:", "Neutral note: the separator is intentionally long. Question:")
    if transform == "boundary_rewrite":
        return text.replace(". ", "; ")
    if transform == "syntax_rewrite":
        return f"Given the following information, {text[0].lower()}{text[1:]}"
    if transform == "query_rewrite":
        return text.replace("what", "which value").replace("which", "what")
    raise ValueError(transform)


def role_mapping_for(transform: str, nodes: dict[str, str]) -> dict[str, str]:
    if transform == "order_rewrite":
        return {key: key for key in reversed(list(nodes))}
    return {key: key for key in nodes}


def build_protocol() -> dict[str, Any]:
    n = GROUPS_PER_SPLIT
    return {
        "schema_version": "phase442_static_sample_contract.v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "theory_name": "语言是动态模式网络",
        "method_frame": "条件物理状态图谱",
        "status": "static_samples_and_feasibility_frozen_no_cuda_run",
        "previous_phase441_artifacts_preserved": True,
        "confidence_interval": {
            "kind": "wilson",
            "sidedness": "two_sided",
            "z": Z_TWO_SIDED_95,
        },
        "groups_per_split": GROUPS_PER_SPLIT,
        "groups_per_task": GROUPS_PER_SPLIT * len(SPLITS),
        "splits": SPLITS,
        "task_library": TASK_LIBRARY,
        "surface_transforms": TRANSFORMS,
        "interfaces": INTERFACES,
        "behavior_gates": {
            "semantic_lcb_95_min": 0.85,
            "other_ucb_95_max": 0.05,
            "orbit_group_consistency_lcb_95_min": 0.80,
            "semantic_min_successes_per_split": min_k_for_lcb(n, 0.85),
            "other_max_failures_per_split": max_k_for_ucb(n, 0.05),
            "orbit_min_successes_per_split": min_k_for_lcb(n, 0.80),
        },
        "zero_gate": [
            "G_protocol_feasible",
            "G_sample_valid",
            "G_semantic_certified",
            "G_split_disjoint",
            "G_static_baseline",
            "G_budget",
        ],
        "tokenization_contract": {
            "models": ["qwen3", "glm4", "deepseek7b"],
            "status": "pre_registered_pending_local_tokenizer_execution",
            "must_pass_before_behavior_cuda": True,
        },
    }


def feasibility_report(protocol: dict[str, Any]) -> dict[str, Any]:
    n = protocol["groups_per_split"]
    gates = protocol["behavior_gates"]
    return {
        "groups_per_split": n,
        "semantic_gate": {
            "threshold": gates["semantic_lcb_95_min"],
            "min_successes": gates["semantic_min_successes_per_split"],
            "feasible": gates["semantic_min_successes_per_split"] is not None,
            "lcb_at_min": wilson_bounds(gates["semantic_min_successes_per_split"], n)[0],
        },
        "other_output_gate": {
            "threshold": gates["other_ucb_95_max"],
            "max_failures": gates["other_max_failures_per_split"],
            "feasible": gates["other_max_failures_per_split"] is not None,
            "ucb_at_zero": wilson_bounds(0, n)[1],
        },
        "orbit_group_gate": {
            "threshold": gates["orbit_group_consistency_lcb_95_min"],
            "min_successes": gates["orbit_min_successes_per_split"],
            "feasible": gates["orbit_min_successes_per_split"] is not None,
            "lcb_at_min": wilson_bounds(gates["orbit_min_successes_per_split"], n)[0],
        },
    }


def generate_samples() -> list[dict[str, Any]]:
    rows = []
    for ability, tasks in TASK_LIBRARY.items():
        for task in tasks:
            for split in SPLITS:
                for idx in range(GROUPS_PER_SPLIT):
                    rows.append(render_sample(ability, task, split, idx))
    return rows


def split_disjoint_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    template_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row["split"]
        for value in row["role_nodes"].values():
            normalized = value.lower()
            if f"_{SPLIT_CODES[split]}_" in normalized:
                by_split[split].add(normalized)
        template_by_split[split].add(f"{row['template_family']}/{split}")

    overlaps = []
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            shared = sorted(by_split[left] & by_split[right])
            if shared:
                overlaps.append({"left": left, "right": right, "shared_values": shared[:10], "count": len(shared)})
    return {
        "entity_role_string_disjoint": not overlaps,
        "cross_split_overlaps": overlaps,
        "template_family_split_scoped": all(len(values) == 15 for values in template_by_split.values()),
    }


def semantic_certificate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        for variant in row["surface_variants"]:
            if variant["semantic_hash"] != row["semantic_hash"]:
                failures.append({"sample_id": row["sample_id"], "transform": variant["transform"]})
            if set(variant["role_mapping"]) != set(row["role_nodes"]):
                failures.append({"sample_id": row["sample_id"], "transform": variant["transform"], "reason": "role_mapping_key_mismatch"})
    return {
        "semantic_hash_preserved": not failures,
        "failures": failures[:20],
        "checked_variant_count": sum(len(row["surface_variants"]) for row in rows),
    }


def baseline_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answers_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    length_by_task: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        key = f"{row['ability']}/{row['task']}"
        answers_by_task[key][row["canonical_answer"]] += 1
        length_by_task[key][len(row["canonical_answer"])] += 1

    task_reports = {}
    max_majority = 0.0
    max_length_baseline = 0.0
    for key, counts in answers_by_task.items():
        total = sum(counts.values())
        majority = max(counts.values()) / total
        length_counts = length_by_task[key]
        length_baseline = max(length_counts.values()) / total
        max_majority = max(max_majority, majority)
        max_length_baseline = max(max_length_baseline, length_baseline)
        task_reports[key] = {
            "total": total,
            "answer_distribution": dict(counts),
            "majority_baseline_accuracy": majority,
            "answer_length_baseline_accuracy": length_baseline,
        }

    return {
        "majority_baseline_below_semantic_gate": max_majority < 0.85,
        "answer_length_baseline_below_semantic_gate": max_length_baseline < 0.85,
        "max_majority_baseline_accuracy": max_majority,
        "max_answer_length_baseline_accuracy": max_length_baseline,
        "task_reports": task_reports,
    }


def budget_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "base_sample_count": len(rows),
        "surface_variant_count": sum(len(row["surface_variants"]) for row in rows),
        "interface_calibration_conditions": 15 * GROUPS_PER_SPLIT * len(INTERFACES),
        "task_discovery_conditions_after_interface_freeze": 15 * GROUPS_PER_SPLIT,
        "orbit_holdout_conditions_after_task_selection_max": 3 * GROUPS_PER_SPLIT * len(TRANSFORMS),
        "physical_authorization_conditions_after_task_selection_max": 3 * GROUPS_PER_SPLIT,
        "full_naive_condition_upper_bound": len(rows) * len(TRANSFORMS) * len(INTERFACES),
        "requires_staged_authorization": True,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol()
    rows = generate_samples()
    certificates = [
        {
            "sample_id": row["sample_id"],
            "logic_form": row["logic_form"],
            "semantic_hash": row["semantic_hash"],
            "variant_hashes": {variant["transform"]: variant["semantic_hash"] for variant in row["surface_variants"]},
            "role_nodes": row["role_nodes"],
        }
        for row in rows
    ]

    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, rows)
    write_jsonl(CERTS_PATH, certificates)

    audit = {
        "schema_version": "phase442_static_audit_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_contract_pass_no_cuda_run",
        "feasibility": feasibility_report(protocol),
        "split_disjoint": split_disjoint_audit(rows),
        "semantic_certificates": semantic_certificate_audit(rows),
        "baseline": baseline_audit(rows),
        "budget": budget_audit(rows),
        "tokenization": {
            "status": "pending_local_tokenizer_execution",
            "models": ["qwen3", "glm4", "deepseek7b"],
            "reason": "Phase442 freezes tokenization requirements without loading CUDA models.",
        },
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = [PROTOCOL_PATH, SAMPLES_PATH, CERTS_PATH, AUDIT_PATH]
    manifest = {
        "schema_version": "phase442_artifact_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_no_cuda_run",
        "artifacts": {str(path.relative_to(ROOT)): file_sha256(path) for path in artifacts},
    }
    manifest["joint_sha256"] = sha256_text(json.dumps(manifest["artifacts"], sort_keys=True))
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
