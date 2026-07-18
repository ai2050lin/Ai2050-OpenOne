#!/usr/bin/env python3
"""Phase482 relation-geometry static protocol.

Freezes two non-interchangeable tracks before any model run:
  A. label_post_relation_geometry: label mapping is not visible at relation roles.
  B. label_pre_mapping_visible_control: label mapping is visible before relation roles.

Also freezes projection, layer families, roles, distances, success gates and
open/holdout/sealed split manifests. Static only: no model, no CUDA.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase482_relation_geometry_protocol"
PROTOCOL_PATH = OUT_DIR / "phase482_relation_geometry_protocol.json"
SAMPLES_PATH = OUT_DIR / "phase482_relation_geometry_samples.jsonl"
AUDIT_PATH = OUT_DIR / "phase482_relation_geometry_static_audit.json"
MANIFEST_PATH = OUT_DIR / "phase482_relation_geometry_manifest.json"

OPEN_SPLITS = ("geometry_window_freeze", "physical_prediction_holdout")
SEALED_SPLIT = "sealed_physical_holdout"
ALL_SPLITS = (*OPEN_SPLITS, SEALED_SPLIT)
PAIRS_PER_SPLIT = 96
SUBPROTOCOLS = ("label_post_relation_geometry", "label_pre_mapping_visible_control")
TEMPLATES = ("records_claim", "claim_records")
LABEL_MAPPINGS = {
    "mu_ab": {True: "A", False: "B", "instruction": "Map: true=A; false=B."},
    "mu_ba": {True: "B", False: "A", "instruction": "Map: true=B; false=A."},
}
PROJECTION = {
    "type": "rademacher",
    "seed": 48220260716,
    "dimension_k": 256,
    "entry_values": ["-1/sqrt(k)", "+1/sqrt(k)"],
    "frozen_before_model_run": True,
}
ROLES = (
    "evidence_block_end",
    "claim_end",
    "label_instruction_end",
    "terminal_token",
)
LAYER_FAMILIES = {
    "early": [0, 8],
    "mid_front": [9, 20],
    "mid_back": [21, 32],
    "late": [33, 39],
    "final": [40, 40],
}


def stable_hash(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def code(prefix: str, split_index: int, pair_index: int, slot: int) -> str:
    return f"{prefix}{split_index:01d}{pair_index:03d}{slot:01d}"


def fact_bundle(split_index: int, pair_index: int) -> dict[str, Any]:
    entities = [code("e", split_index, pair_index, 0), code("e", split_index, pair_index, 1)]
    kinds = [code("k", split_index, pair_index, 0), code("k", split_index, pair_index, 1)]
    props = [code("p", split_index, pair_index, 0), code("p", split_index, pair_index, 1)]
    facts = [
        f"All {kinds[0]} have marker {props[0]}.",
        f"{entities[0]} is one {kinds[0]}.",
        f"All {kinds[1]} have marker {props[1]}.",
        f"{entities[1]} is one {kinds[1]}.",
    ]
    return {"entities": entities, "kinds": kinds, "props": props, "facts": facts}


def render_text(subprotocol: str, template: str, facts: list[str], claim: str, instruction: str) -> str:
    fact_text = " ".join(facts)
    if subprotocol == "label_post_relation_geometry":
        if template == "records_claim":
            return f"Records: {fact_text} Claim: {claim} {instruction}"
        return f"Claim: {claim} Records: {fact_text} {instruction}"
    if subprotocol == "label_pre_mapping_visible_control":
        if template == "records_claim":
            return f"{instruction} Records: {fact_text} Claim: {claim}"
        return f"{instruction} Claim: {claim} Records: {fact_text}"
    raise ValueError(subprotocol)


def make_rows(split: str, split_index: int, pair_index: int) -> list[dict[str, Any]]:
    bundle = fact_bundle(split_index, pair_index)
    target_position = pair_index % 2
    target_entity = bundle["entities"][target_position]
    true_prop = bundle["props"][target_position]
    false_prop = bundle["props"][1 - target_position]
    source_pair_id = stable_hash("phase482", split, pair_index, "pair")
    rows = []
    for pair_role, truth_value in (("base_true", True), ("counterfactual_false", False)):
        query_prop = true_prop if truth_value else false_prop
        claim = f"{target_entity} has marker {query_prop}."
        for label_mapping, mapping in LABEL_MAPPINGS.items():
            expected = str(mapping[truth_value])
            for subprotocol in SUBPROTOCOLS:
                variants = []
                for template in TEMPLATES:
                    text = render_text(subprotocol, template, bundle["facts"], claim, str(mapping["instruction"]))
                    variants.append({
                        "track": "phase482_relation_geometry",
                        "subprotocol": subprotocol,
                        "template": template,
                        "text": text,
                        "expected_label": expected,
                        "label_mapping": label_mapping,
                    })
                logic = {
                    "phase": "phase482",
                    "split": split,
                    "pair_index": pair_index,
                    "pair_role": pair_role,
                    "truth_value": truth_value,
                    "label_mapping": label_mapping,
                    "subprotocol": subprotocol,
                    "target_position": target_position,
                    "query_property": query_prop,
                }
                rows.append({
                    "sample_id": stable_hash("phase482", split, pair_index, pair_role, label_mapping, subprotocol),
                    "source_sample_id": stable_hash("phase482", split, pair_index, pair_role),
                    "source_pair_id": source_pair_id,
                    "pair_index": pair_index,
                    "pair_role": pair_role,
                    "split": split,
                    "sealed": split == SEALED_SPLIT,
                    "subprotocol": subprotocol,
                    "label_mapping": label_mapping,
                    "canonical_answer": expected,
                    "truth_value": truth_value,
                    "facts": bundle["facts"],
                    "claim": claim,
                    "role_nodes": {
                        "target_position": target_position,
                        "target_entity": target_entity,
                        "query_property": query_prop,
                    },
                    "logic_form": logic,
                    "semantic_hash": stable_hash(json.dumps(logic, sort_keys=True), n=20),
                    "surface_variants": variants,
                })
    return rows


def build_samples() -> list[dict[str, Any]]:
    rows = []
    for split_index, split in enumerate(ALL_SPLITS):
        for pair_index in range(PAIRS_PER_SPLIT):
            rows.extend(make_rows(split, split_index, pair_index))
    return rows


def variant_rows(samples: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(sample, variant) for sample in samples for variant in sample["surface_variants"]]


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    open_samples = [row for row in samples if not row["sealed"]]
    counters = {
        "split": Counter(row["split"] for row in samples),
        "open_split": Counter(row["split"] for row in open_samples),
        "truth_by_split": Counter((row["split"], row["truth_value"]) for row in samples),
        "mapping_by_split": Counter((row["split"], row["label_mapping"]) for row in samples),
        "subprotocol_by_split": Counter((row["split"], row["subprotocol"]) for row in samples),
        "answer_by_mapping_truth": Counter((row["label_mapping"], row["truth_value"], row["canonical_answer"]) for row in samples),
        "variant_template_by_subprotocol": Counter(
            (row["split"], row["subprotocol"], variant["template"]) for row, variant in variant_rows(samples)
        ),
    }
    pass_flag = (
        set(counters["split"].values()) == {PAIRS_PER_SPLIT * 2 * 2 * 2}
        and set(counters["truth_by_split"].values()) == {PAIRS_PER_SPLIT * 2 * 2}
        and set(counters["mapping_by_split"].values()) == {PAIRS_PER_SPLIT * 2 * 2}
        and set(counters["subprotocol_by_split"].values()) == {PAIRS_PER_SPLIT * 2 * 2}
        and set(counters["answer_by_mapping_truth"].values()) == {PAIRS_PER_SPLIT * len(ALL_SPLITS) * len(SUBPROTOCOLS)}
        and set(counters["variant_template_by_subprotocol"].values()) == {PAIRS_PER_SPLIT * 2 * 2}
    )
    return {"pass": pass_flag, **{key: {str(k): v for k, v in counter.items()} for key, counter in counters.items()}}


def equality_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    keyed = {
        (row["split"], row["source_pair_id"], row["pair_role"], row["label_mapping"], row["subprotocol"]): row
        for row in samples
    }
    for row in samples:
        if row["pair_role"] != "base_true":
            continue
        false_row = keyed.get((row["split"], row["source_pair_id"], "counterfactual_false", row["label_mapping"], row["subprotocol"]))
        if false_row is None:
            failures.append({"kind": "missing_counterfactual", "sample_id": row["sample_id"]})
            continue
        if len(row["claim"]) != len(false_row["claim"]):
            failures.append({"kind": "claim_length_mismatch", "sample_id": row["sample_id"]})
        if len(row["facts"]) != len(false_row["facts"]):
            failures.append({"kind": "fact_count_mismatch", "sample_id": row["sample_id"]})
        for true_variant, false_variant in zip(row["surface_variants"], false_row["surface_variants"], strict=True):
            if len(true_variant["text"]) != len(false_variant["text"]):
                failures.append({
                    "kind": "variant_length_mismatch",
                    "sample_id": row["sample_id"],
                    "template": true_variant["template"],
                })
    for split in ALL_SPLITS:
        for pair_index in range(PAIRS_PER_SPLIT):
            for subprotocol in SUBPROTOCOLS:
                ab = [row for row in samples if row["split"] == split and row["pair_index"] == pair_index and row["subprotocol"] == subprotocol and row["label_mapping"] == "mu_ab"]
                ba = [row for row in samples if row["split"] == split and row["pair_index"] == pair_index and row["subprotocol"] == subprotocol and row["label_mapping"] == "mu_ba"]
                for left, right in zip(sorted(ab, key=lambda x: x["pair_role"]), sorted(ba, key=lambda x: x["pair_role"]), strict=True):
                    for lv, rv in zip(left["surface_variants"], right["surface_variants"], strict=True):
                        if len(lv["text"]) != len(rv["text"]):
                            failures.append({"kind": "mapping_text_length_mismatch", "sample_id": left["sample_id"]})
    return {"pass": not failures, "failure_count": len(failures), "failures": failures[:30]}


def baseline_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = variant_rows(samples)
    baselines = {
        "constant_A": lambda s, v: "A",
        "constant_B": lambda s, v: "B",
        "mu_ab": lambda s, v: "A" if s["label_mapping"] == "mu_ab" else "B",
        "truth_only_without_mapping": lambda s, v: "A" if s["truth_value"] else "B",
        "truth_plus_mapping_oracle": lambda s, v: s["canonical_answer"],
        "role_base": lambda s, v: "A" if s["pair_role"] == "base_true" else "B",
        "template_records": lambda s, v: "A" if v["template"] == "records_claim" else "B",
        "subprotocol_label_post": lambda s, v: "A" if s["subprotocol"] == "label_post_relation_geometry" else "B",
        "text_length_even": lambda s, v: "A" if len(v["text"]) % 2 == 0 else "B",
    }
    reports = {}
    failing = []
    for name, fn in baselines.items():
        correct = sum(1 for sample, variant in rows if fn(sample, variant) == sample["canonical_answer"])
        acc = correct / len(rows)
        reports[name] = {"n": len(rows), "correct": correct, "accuracy": acc}
        if name != "truth_plus_mapping_oracle" and acc > 0.55:
            failing.append({"baseline": name, "accuracy": acc})
    oracle_pass = reports["truth_plus_mapping_oracle"]["accuracy"] == 1.0
    if not oracle_pass:
        failing.append({"baseline": "truth_plus_mapping_oracle", "accuracy": reports["truth_plus_mapping_oracle"]["accuracy"]})
    return {
        "pass": not failing and oracle_pass,
        "reports": reports,
        "failing_baselines": failing,
        "oracle_baselines_not_counted": ["truth_plus_mapping_oracle"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    protocol = {
        "schema_version": "phase482_relation_geometry_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_protocol_frozen_no_model_run_no_cuda",
        "splits": {
            "open": list(OPEN_SPLITS),
            "sealed": SEALED_SPLIT,
            "pairs_per_split": PAIRS_PER_SPLIT,
            "phase482_may_read": list(OPEN_SPLITS),
            "phase482_must_not_read": [SEALED_SPLIT],
        },
        "count_contract": {
            "split_count": len(ALL_SPLITS),
            "pairs_per_split": PAIRS_PER_SPLIT,
            "truth_roles_per_pair": 2,
            "label_mappings": len(LABEL_MAPPINGS),
            "subprotocols": len(SUBPROTOCOLS),
            "templates": len(TEMPLATES),
            "expected_sample_records": len(ALL_SPLITS) * PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS),
            "expected_variant_records": len(ALL_SPLITS) * PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS) * len(TEMPLATES),
            "open_sample_records": len(OPEN_SPLITS) * PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS),
            "open_variant_records": len(OPEN_SPLITS) * PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS) * len(TEMPLATES),
            "sealed_sample_records": PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS),
            "sealed_variant_records": PAIRS_PER_SPLIT * 2 * len(LABEL_MAPPINGS) * len(SUBPROTOCOLS) * len(TEMPLATES),
            "record_naming": {
                "sample_record": "one logical sample with embedded surface_variants",
                "variant_record": "one concrete prompt/template row after expanding surface_variants",
            },
        },
        "subprotocols": list(SUBPROTOCOLS),
        "templates": list(TEMPLATES),
        "label_mappings": list(LABEL_MAPPINGS),
        "projection": PROJECTION,
        "roles": list(ROLES),
        "layer_families": LAYER_FAMILIES,
        "distances": ["cosine_distance", "euclidean_distance"],
        "primary_quality": "Q_R = (D_cf - D_surface) / (D_cf + D_surface + eps)",
        "mapping_visible_gate": "D_mu < D_cf",
        "component_window_gate": "E[delta_M] > 0, P(delta_M > 0) >= 0.70, clustered CI excludes zero, true/false and both mappings pass, holdout replicates",
        "forbidden": ["head_scan", "channel_scan", "neuron_scan", "sealed_split_read", "post_result_projection_change"],
    }
    balance = balance_audit(samples)
    equality = equality_audit(samples)
    baseline = baseline_audit(samples)
    actual_sample_records = len(samples)
    actual_variant_records = sum(len(row["surface_variants"]) for row in samples)
    count_contract = {
        "pass": (
            actual_sample_records == protocol["count_contract"]["expected_sample_records"]
            and actual_variant_records == protocol["count_contract"]["expected_variant_records"]
        ),
        "actual_sample_records": actual_sample_records,
        "actual_variant_records": actual_variant_records,
        "expected_sample_records": protocol["count_contract"]["expected_sample_records"],
        "expected_variant_records": protocol["count_contract"]["expected_variant_records"],
    }
    all_pass = balance["pass"] and equality["pass"] and baseline["pass"] and count_contract["pass"]
    audit = {
        "schema_version": "phase482_relation_geometry_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if all_pass else "static_fail_no_model_run",
        "balance": balance,
        "count_contract": count_contract,
        "equality": equality,
        "static_baselines": baseline,
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, samples)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "phase482_relation_geometry_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(PROTOCOL_PATH.relative_to(ROOT)): sha256_file(PROTOCOL_PATH),
            str(SAMPLES_PATH.relative_to(ROOT)): sha256_file(SAMPLES_PATH),
            str(AUDIT_PATH.relative_to(ROOT)): sha256_file(AUDIT_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(PROTOCOL_PATH)
    print(SAMPLES_PATH)
    print(AUDIT_PATH)
    print("status", audit["status"])


if __name__ == "__main__":
    main()
