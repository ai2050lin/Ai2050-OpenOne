#!/usr/bin/env python3
"""Phase472 label-mapping reversal protocol.

Static-only protocol. Reuses the Phase465 logic task with two evidence-query
orders and two label mappings:
  mu_ab: true -> A, false -> B
  mu_ba: true -> B, false -> A

No model run, no CUDA, no physical trace.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase472_label_mapping_reversal_protocol"
PROTOCOL_PATH = OUT_DIR / "phase472_label_mapping_reversal_protocol.json"
SAMPLES_PATH = OUT_DIR / "phase472_label_mapping_reversal_samples.jsonl"
AUDIT_PATH = OUT_DIR / "phase472_label_mapping_reversal_static_audit.json"
MANIFEST_PATH = OUT_DIR / "phase472_label_mapping_reversal_manifest.json"

PAIRS = 96
ITEMS = ["mira", "naro", "lina", "pavo", "suri", "toma", "vexa", "rilo", "dina", "kavo", "zuri", "fena"]
KINDS = ["dax", "nup", "vorn", "kel", "siv", "marn", "tul", "prax"]
PROPS = ["luminous", "brittle", "hollow", "woven", "silent", "curved", "dense", "nimble"]
TRANSFORMS = ["order_evidence_first", "order_claim_first"]
LABEL_MAPPINGS = {
    "mu_ab": {True: "A", False: "B", "instruction": "Reply with A if the claim is true, and B if it is false."},
    "mu_ba": {True: "B", False: "A", "instruction": "Reply with B if the claim is true, and A if it is false."},
}


def stable_hash(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pick(pool: list[str], idx: int, offset: int = 0) -> str:
    return pool[(idx + offset) % len(pool)]


def variants(facts: list[str], claim: str, answer: str, label_mapping: str, semantic_hash: str) -> list[dict[str, str]]:
    fact_text = " ".join(facts)
    instruction = str(LABEL_MAPPINGS[label_mapping]["instruction"])
    raw = [
        ("order_evidence_first", f"Records: {fact_text} Claim: {claim} {instruction}"),
        ("order_claim_first", f"Claim: {claim} Records: {fact_text} {instruction}"),
    ]
    return [
        {
            "track": "label_mapping_order",
            "transform": transform,
            "text": text,
            "expected_label": answer,
            "semantic_hash": semantic_hash,
            "label_mapping": label_mapping,
        }
        for transform, text in raw
    ]


def make_pair(pair_index: int) -> list[dict[str, Any]]:
    code = f"p472_{pair_index:03d}"
    target_position = pair_index % 2
    base_true = pair_index % 4 in (0, 1)
    names = [
        f"{pick(ITEMS, pair_index)}_{code}_{stable_hash('item', pair_index, 0, n=4)}",
        f"{pick(ITEMS, pair_index, 5)}_{code}_{stable_hash('item', pair_index, 1, n=4)}",
    ]
    kinds = [
        f"{pick(KINDS, pair_index)}_{code}_{stable_hash('kind', pair_index, 0, n=4)}",
        f"{pick(KINDS, pair_index, 3)}_{code}_{stable_hash('kind', pair_index, 1, n=4)}",
    ]
    props = [
        f"{pick(PROPS, pair_index)}_{code}_{stable_hash('prop', pair_index, 0, n=4)}",
        f"{pick(PROPS, pair_index, 4)}_{code}_{stable_hash('prop', pair_index, 1, n=4)}",
    ]
    facts = [
        f"All members of {kinds[0]} have trait {props[0]}.",
        f"{names[0]} is a member of {kinds[0]}.",
        f"All members of {kinds[1]} have trait {props[1]}.",
        f"{names[1]} is a member of {kinds[1]}.",
    ]
    target_entity = names[target_position]
    true_prop = props[target_position]
    false_prop = props[1 - target_position]
    rows = []
    for role, is_true in (("base", base_true), ("counterfactual", not base_true)):
        query_prop = true_prop if is_true else false_prop
        query_position = target_position if is_true else 1 - target_position
        claim = f"{target_entity} has trait {query_prop}."
        for label_mapping in ("mu_ab", "mu_ba"):
            answer = str(LABEL_MAPPINGS[label_mapping][is_true])
            logic = {
                "phase": "phase472",
                "pair_index": pair_index,
                "role": role,
                "target_position": target_position,
                "query_position": query_position,
                "truth_value": is_true,
                "label_mapping": label_mapping,
            }
            semantic_hash = stable_hash(json.dumps(logic, sort_keys=True), n=20)
            rows.append({
                "sample_id": stable_hash("phase472", pair_index, role, label_mapping),
                "source_sample_id": stable_hash("phase472", pair_index, role),
                "source_pair_id": stable_hash("phase472", pair_index, "pair"),
                "pair_index": pair_index,
                "pair_role": role,
                "label_mapping": label_mapping,
                "model_target": "glm4",
                "ability": "knowledge_network",
                "task": "label_mapping_reversal_truth",
                "split": "phase472_label_mapping_reversal",
                "canonical_answer": answer,
                "truth_value": is_true,
                "logic_form": logic,
                "facts": facts,
                "claim": claim,
                "role_nodes": {
                    "target_position": target_position,
                    "query_position": query_position,
                    "target_entity": target_entity,
                    "query_property": query_prop,
                },
                "semantic_hash": semantic_hash,
                "surface_variants": variants(facts, claim, answer, label_mapping, semantic_hash),
            })
    return rows


def build_samples() -> list[dict[str, Any]]:
    rows = []
    for pair_index in range(PAIRS):
        rows.extend(make_pair(pair_index))
    return rows


def baseline_predictions(sample: dict[str, Any], variant: dict[str, str]) -> dict[str, str]:
    nodes = sample["role_nodes"]
    text = variant["text"]
    return {
        "constant_A": "A",
        "constant_B": "B",
        "truth_value": "A" if sample["truth_value"] else "B",
        "inverse_truth_value": "B" if sample["truth_value"] else "A",
        "mu_ab": "A" if sample["label_mapping"] == "mu_ab" else "B",
        "role_base": "A" if sample["pair_role"] == "base" else "B",
        "target_position_0": "A" if nodes["target_position"] == 0 else "B",
        "query_position_0": "A" if nodes["query_position"] == 0 else "B",
        "query_matches_target_position": "A" if nodes["query_position"] == nodes["target_position"] else "B",
        "claim_first_template": "A" if text.startswith("Claim:") else "B",
        "text_length_even": "A" if len(text) % 2 == 0 else "B",
    }


def static_baseline_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [(sample, variant) for sample in samples for variant in sample["surface_variants"]]
    reports = {}
    failing = []
    for baseline in baseline_predictions(rows[0][0], rows[0][1]):
        correct = sum(1 for sample, variant in rows if baseline_predictions(sample, variant)[baseline] == sample["canonical_answer"])
        n = len(rows)
        acc = correct / n
        reports[baseline] = {"n": n, "correct": correct, "accuracy": acc}
        if baseline not in {"truth_value", "inverse_truth_value", "query_matches_target_position"} and acc > 0.55:
            failing.append({"baseline": baseline, "accuracy": acc})
    return {
        "pass": not failing,
        "reports": reports,
        "failing_baselines": failing,
        "oracle_or_semantic_baselines_not_counted": ["truth_value", "inverse_truth_value", "query_matches_target_position"],
    }


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    variant_rows = [(sample, variant) for sample in samples for variant in sample["surface_variants"]]
    counters = {
        "labels": Counter(s["canonical_answer"] for s in samples),
        "truth_values": Counter(s["truth_value"] for s in samples),
        "label_mapping": Counter(s["label_mapping"] for s in samples),
        "label_mapping_answer": Counter((s["label_mapping"], s["canonical_answer"]) for s in samples),
        "label_mapping_truth_answer": Counter((s["label_mapping"], s["truth_value"], s["canonical_answer"]) for s in samples),
        "roles": Counter(s["pair_role"] for s in samples),
        "label_role": Counter((s["canonical_answer"], s["pair_role"]) for s in samples),
        "variant_transform_label_mapping_answer": Counter((v["transform"], s["label_mapping"], s["canonical_answer"]) for s, v in variant_rows),
    }
    pass_flag = (
        set(counters["labels"].values()) == {192}
        and set(counters["truth_values"].values()) == {192}
        and set(counters["label_mapping"].values()) == {192}
        and set(counters["label_mapping_answer"].values()) == {96}
        and set(counters["label_mapping_truth_answer"].values()) == {96}
        and set(counters["roles"].values()) == {192}
        and set(counters["label_role"].values()) == {96}
        and set(counters["variant_transform_label_mapping_answer"].values()) == {96}
    )
    return {
        "pass": pass_flag,
        **{key: {str(k): v for k, v in value.items()} for key, value in counters.items()},
    }


def certificate_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    transform_counts = Counter()
    for sample in samples:
        for variant in sample["surface_variants"]:
            transform_counts[variant["transform"]] += 1
            if variant["expected_label"] != sample["canonical_answer"] or variant["semantic_hash"] != sample["semantic_hash"]:
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"]})
    return {"pass": not failures, "transform_counts": dict(transform_counts), "failure_count": len(failures), "failures": failures[:20]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    protocol = {
        "schema_version": "phase472_label_mapping_reversal_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "label_mapping_reversal_static_only_no_model_run",
        "target_model": "glm4",
        "pairs": PAIRS,
        "samples": len(samples),
        "variants": len(samples) * len(TRANSFORMS),
        "transforms": TRANSFORMS,
        "label_mappings": list(LABEL_MAPPINGS),
        "physical_trace_authorized": False,
    }
    balance = balance_audit(samples)
    certs = certificate_audit(samples)
    baseline = static_baseline_audit(samples)
    all_pass = balance["pass"] and certs["pass"] and baseline["pass"]
    audit = {
        "schema_version": "phase472_label_mapping_reversal_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if all_pass else "static_fail_no_model_run",
        "balance": balance,
        "certificates": certs,
        "static_baselines": baseline,
        "authorization": {
            "glm4_behavior_label_mapping_authorized": all_pass,
            "physical_trace_authorized": False,
        },
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, samples)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {str(path.relative_to(ROOT)): sha256_file(path) for path in [PROTOCOL_PATH, SAMPLES_PATH, AUDIT_PATH]}
    manifest = {
        "schema_version": "phase472_label_mapping_reversal_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_no_model_run",
        "artifacts": artifacts,
        "joint_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(AUDIT_PATH)


if __name__ == "__main__":
    main()
