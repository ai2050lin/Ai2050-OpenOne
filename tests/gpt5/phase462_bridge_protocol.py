#!/usr/bin/env python3
"""Phase462 bridge protocol.

No model run, no CUDA, no physical trace. Keeps the two-entity/four-fact
difficulty of Phase452 while breaking base=A and counterfactual=B binding.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase462_bridge_protocol"
PROTOCOL_PATH = OUT_DIR / "phase462_bridge_protocol.json"
SAMPLES_PATH = OUT_DIR / "phase462_bridge_samples.jsonl"
AUDIT_PATH = OUT_DIR / "phase462_bridge_static_audit.json"
MANIFEST_PATH = OUT_DIR / "phase462_bridge_manifest.json"

PAIRS = 96
ITEMS = ["mira", "naro", "lina", "pavo", "suri", "toma", "vexa", "rilo", "dina", "kavo", "zuri", "fena"]
KINDS = ["dax", "nup", "vorn", "kel", "siv", "marn", "tul", "prax"]
PROPS = ["luminous", "brittle", "hollow", "woven", "silent", "curved", "dense", "nimble"]

CORE_TRANSFORMS = [
    "bridge_lexical_frame",
    "bridge_numbered_facts",
    "bridge_evidence_claim",
    "bridge_claim_sync",
]
STRESS_TRANSFORMS = [
    "bridge_claim_first",
    "bridge_semicolon_compact",
]


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


def label(is_true: bool) -> str:
    return "A" if is_true else "B"


def instruction(noun: str = "claim") -> str:
    return f"Reply with A if the {noun} is true, and B if it is false."


def variants(facts: list[str], claim: str, answer: str, semantic_hash: str) -> list[dict[str, str]]:
    fact_text = " ".join(facts)
    numbered = " ".join(f"Fact {idx + 1}: {fact}" for idx, fact in enumerate(facts))
    dense = " ; ".join(fact.rstrip(".") for fact in facts)
    raw = [
        ("core", "bridge_lexical_frame", f"Records: {fact_text} Claim: {claim} {instruction()}"),
        ("core", "bridge_numbered_facts", f"{numbered} Claim: {claim} {instruction()}"),
        ("core", "bridge_evidence_claim", f"Evidence: {fact_text} Claim to evaluate: {claim} {instruction()}"),
        ("core", "bridge_claim_sync", f"{fact_text} The claim is: {claim} {instruction()}"),
        ("stress", "bridge_claim_first", f"Claim: {claim} Evidence follows: {fact_text} {instruction()}"),
        ("stress", "bridge_semicolon_compact", f"{dense}. Claim: {claim} {instruction()}"),
    ]
    return [
        {"track": track, "transform": transform, "text": text, "expected_label": answer, "semantic_hash": semantic_hash}
        for track, transform, text in raw
    ]


def make_pair(pair_index: int) -> list[dict[str, Any]]:
    code = f"p462_{pair_index:03d}"
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
        claim = f"{target_entity} has trait {query_prop}."
        logic = {
            "phase": "phase462",
            "pair_index": pair_index,
            "role": role,
            "target_position": target_position,
            "query_position": target_position if is_true else 1 - target_position,
            "truth_value": is_true,
        }
        semantic_hash = stable_hash(json.dumps(logic, sort_keys=True), n=20)
        rows.append({
            "sample_id": stable_hash("phase462", pair_index, role),
            "source_pair_id": stable_hash("phase462", pair_index, "pair"),
            "pair_index": pair_index,
            "pair_role": role,
            "model_target": "glm4",
            "ability": "knowledge_network",
            "task": "bridge_marker_truth",
            "split": "phase462_bridge_replicate",
            "canonical_answer": label(is_true),
            "truth_value": is_true,
            "logic_form": logic,
            "facts": facts,
            "claim": claim,
            "role_nodes": {
                "target_position": target_position,
                "query_position": target_position if is_true else 1 - target_position,
                "target_entity": target_entity,
                "query_property": query_prop,
            },
            "semantic_hash": semantic_hash,
            "core_transforms": CORE_TRANSFORMS,
            "stress_transforms": STRESS_TRANSFORMS,
            "surface_variants": variants(facts, claim, label(is_true), semantic_hash),
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
        "role_base": "A" if sample["pair_role"] == "base" else "B",
        "role_counterfactual": "A" if sample["pair_role"] == "counterfactual" else "B",
        "target_position_0": "A" if nodes["target_position"] == 0 else "B",
        "query_position_0": "A" if nodes["query_position"] == 0 else "B",
        "query_matches_target_position": "A" if nodes["query_position"] == nodes["target_position"] else "B",
        "numbered_template": "A" if "Fact 1:" in text else "B",
        "claim_first_template": "A" if text.startswith("Claim:") else "B",
        "semicolon_template": "A" if " ; " in text else "B",
        "pair_index_even": "A" if sample["pair_index"] % 2 == 0 else "B",
        "pair_index_mod4_low": "A" if sample["pair_index"] % 4 in (0, 1) else "B",
        "text_length_even": "A" if len(text) % 2 == 0 else "B",
    }


def static_baseline_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    reports = {}
    failing = []
    for track_name, transforms in (("core", CORE_TRANSFORMS), ("stress", STRESS_TRANSFORMS), ("all", CORE_TRANSFORMS + STRESS_TRANSFORMS)):
        rows = [(sample, variant) for sample in samples for variant in sample["surface_variants"] if variant["transform"] in transforms]
        reports[track_name] = {}
        for baseline in baseline_predictions(rows[0][0], rows[0][1]):
            correct = sum(1 for sample, variant in rows if baseline_predictions(sample, variant)[baseline] == sample["canonical_answer"])
            n = len(rows)
            acc = correct / n
            reports[track_name][baseline] = {"n": n, "correct": correct, "accuracy": acc}
            if baseline != "query_matches_target_position" and acc > 0.55:
                failing.append({"track": track_name, "baseline": baseline, "accuracy": acc})
    return {
        "pass": not failing,
        "reports": reports,
        "failing_baselines": failing,
        "oracle_baseline_not_counted": "query_matches_target_position",
    }


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "labels": Counter(s["canonical_answer"] for s in samples),
        "roles": Counter(s["pair_role"] for s in samples),
        "target_positions": Counter(s["role_nodes"]["target_position"] for s in samples),
        "query_positions": Counter(s["role_nodes"]["query_position"] for s in samples),
        "label_role": Counter((s["canonical_answer"], s["pair_role"]) for s in samples),
        "label_role_target_query": Counter((
            s["canonical_answer"],
            s["pair_role"],
            s["role_nodes"]["target_position"],
            s["role_nodes"]["query_position"],
        ) for s in samples),
    }
    pass_flag = (
        set(counters["labels"].values()) == {96}
        and set(counters["roles"].values()) == {96}
        and set(counters["target_positions"].values()) == {96}
        and set(counters["query_positions"].values()) == {96}
        and set(counters["label_role"].values()) == {48}
        and set(counters["label_role_target_query"].values()) == {24}
    )
    return {
        "pass": pass_flag,
        **{key: {str(k): v for k, v in value.items()} for key, value in counters.items()},
    }


def certificate_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    track_counts = Counter()
    for sample in samples:
        for variant in sample["surface_variants"]:
            track_counts[variant["track"]] += 1
            if variant["expected_label"] != sample["canonical_answer"] or variant["semantic_hash"] != sample["semantic_hash"]:
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"]})
    return {"pass": not failures, "track_counts": dict(track_counts), "failure_count": len(failures), "failures": failures[:20]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    protocol = {
        "schema_version": "phase462_bridge_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "bridge_protocol_static_only_no_model_run",
        "target_model": "glm4",
        "target_task": "knowledge_network/bridge_marker_truth",
        "pairs": PAIRS,
        "samples": len(samples),
        "core_transforms": CORE_TRANSFORMS,
        "stress_transforms": STRESS_TRANSFORMS,
        "base_label_balanced": True,
        "counterfactual_label_balanced": True,
        "keeps_phase452_difficulty": "two_entities_four_facts",
        "physical_trace_authorized": False,
    }
    balance = balance_audit(samples)
    certs = certificate_audit(samples)
    baseline = static_baseline_audit(samples)
    all_pass = balance["pass"] and certs["pass"] and baseline["pass"]
    audit = {
        "schema_version": "phase462_bridge_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if all_pass else "static_fail_no_model_run",
        "balance": balance,
        "certificates": certs,
        "static_baselines": baseline,
        "authorization": {
            "glm4_behavior_bridge_authorized": all_pass,
            "physical_trace_authorized": False,
        },
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, samples)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {str(path.relative_to(ROOT)): sha256_file(path) for path in [PROTOCOL_PATH, SAMPLES_PATH, AUDIT_PATH]}
    manifest = {
        "schema_version": "phase462_bridge_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_no_model_run",
        "artifacts": artifacts,
        "joint_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(AUDIT_PATH)


if __name__ == "__main__":
    main()
