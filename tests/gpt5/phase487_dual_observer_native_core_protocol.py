#!/usr/bin/env python3
"""Phase487 observer correction and native-core protocol freeze.

Static only. This script:

1. Re-audits the Phase486 ledger with mutually exclusive outcomes.
2. Freezes a target-blind output-event parser and conformance suite.
3. Creates genuinely distinct identity/plain tracks in split-isolated files.

It does not load a model, use CUDA, or read a sealed split during evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE486_ROWS = (
    ROOT
    / "tests"
    / "gpt5"
    / "result"
    / "phase486_readonly_other_event_audit"
    / "phase486_readonly_other_event_rows.jsonl"
)
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase487_dual_observer_native_core_protocol"
PROTOCOL_PATH = OUT_DIR / "phase487_protocol.json"
MANIFEST_PATH = OUT_DIR / "phase487_manifest.json"
AUDIT_PATH = OUT_DIR / "phase487_static_audit.json"
CONFORMANCE_PATH = OUT_DIR / "phase487_event_parser_conformance.json"
CORRECTION_PATH = OUT_DIR / "phase487_phase486_corrected_audit.json"

MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = (
    "behavior_qualification",
    "geometry_window",
    "physical_prediction",
    "sealed_physical",
)
SPLIT_FILES = {
    split: OUT_DIR / f"phase487_{split}_samples.jsonl"
    for split in SPLITS
}
SEALED_SPLIT = "sealed_physical"
FAMILIES = ("marker_inheritance", "signal_assignment")
TRACKS = (
    "identity",
    "native_plain_candidate",
    "formal_light_stress",
    "order_stress_claim_first",
)
PAIRS_PER_FAMILY_SPLIT = 64
LABEL_MAPPINGS = {
    "mu_ab": {True: "A", False: "B"},
    "mu_ba": {True: "B", False: "A"},
}


def stable_hash(*parts: object, n: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def truth_from_label(label: str, mapping: str) -> bool:
    if mapping == "mu_ab":
        return label == "A"
    if mapping == "mu_ba":
        return label == "B"
    raise ValueError(mapping)


def parse_output_event(text: str, mapping: str) -> dict[str, Any]:
    """Parse an output without access to its normative truth or expected label."""

    raw = text.strip()
    lowered = raw.lower()
    result: dict[str, Any] = {
        "event_type": "unrelated_or_unresolved",
        "label_value": None,
        "semantic_value": None,
        "recoverable": False,
        "ambiguous": False,
        "reason": "no frozen verdict pattern matched",
    }
    if not raw:
        result.update(event_type="empty", reason="empty generation")
        return result

    # Mapping echoes are instructions copied from the prompt, not verdicts.
    mapping_echo = bool(
        re.search(r"\b(?:map|mapping|coding)\b", lowered)
        or (
            re.search(r"\btrue\s*=\s*[ab]\b", lowered)
            and re.search(r"\bfalse\s*=\s*[ab]\b", lowered)
        )
    )
    if mapping_echo:
        result.update(event_type="mapping_echo", reason="output repeats label mapping")
        return result

    if re.fullmatch(r"[AB]", raw):
        label = raw
        result.update(
            event_type="strict_single_label",
            label_value=label,
            semantic_value=truth_from_label(label, mapping),
            recoverable=True,
            reason="output is exactly one A/B label",
        )
        return result

    if re.search(r"\btrue\b", lowered) and re.search(r"\bfalse\b", lowered):
        result.update(event_type="ambiguous_truth_words", ambiguous=True, reason="both truth words occur without verdict syntax")
        return result

    label_match = re.match(
        r"^(?:answer|label|verdict)?\s*(?:is|:|=)?\s*\(?([AB])\)?(?:\s|[.,;:!?)\-]|$)",
        raw,
        flags=re.IGNORECASE,
    )
    parenthetical_label = re.search(r"\(([AB])\)(?:\s|[.,;:!?]|$)", raw, flags=re.IGNORECASE)
    truth_match = re.match(
        r"^(?:the\s+(?:claim|statement)\s+is\s+|verdict\s*(?:is|:|=)\s*)?(true|false)\b",
        lowered,
    )
    yes_no_match = re.match(r"^(yes|no)\b", lowered)

    label_value = None
    if label_match:
        label_value = label_match.group(1).upper()
    elif parenthetical_label:
        label_value = parenthetical_label.group(1).upper()
    semantic_value: bool | None = None
    semantic_source = None
    if truth_match:
        semantic_value = truth_match.group(1) == "true"
        semantic_source = "truth_word"
    elif yes_no_match:
        # Every Phase487 question has fixed positive polarity: "is ... true?"
        semantic_value = yes_no_match.group(1) == "yes"
        semantic_source = "yes_no_word"
    else:
        conclusion = re.search(
            r"\b(?:therefore|thus|so|conclusion|answer|verdict)[,:]?\s*"
            r"(?:the\s+(?:claim|statement)\s+is\s+)?(true|false)\b",
            lowered,
        )
        if conclusion:
            semantic_value = conclusion.group(1) == "true"
            semantic_source = "explicit_conclusion"

    if label_value is not None:
        label_truth = truth_from_label(label_value, mapping)
        if semantic_value is not None and semantic_value != label_truth:
            result.update(
                event_type="conflicting_label_and_truth",
                label_value=label_value,
                ambiguous=True,
                reason="explicit label and truth verdict conflict",
            )
            return result
        result.update(
            event_type="wrapped_single_label",
            label_value=label_value,
            semantic_value=label_truth,
            recoverable=True,
            reason="output begins with an explicit A/B verdict",
        )
        return result

    if semantic_value is not None:
        event_type = {
            "truth_word": "truth_word_verdict",
            "yes_no_word": "yes_no_verdict",
            "explicit_conclusion": "explicit_explanation_conclusion",
        }[semantic_source]
        result.update(
            event_type=event_type,
            semantic_value=semantic_value,
            recoverable=True,
            reason=f"frozen {semantic_source} verdict pattern",
        )
        return result

    if re.match(r"^[a-z][a-z0-9_]*\d", lowered):
        result.update(event_type="content_continuation", reason="output continues synthetic record content")
    elif lowered.startswith(("to determine", "step", "the claim", "the statement", "we need", "first")):
        result.update(event_type="truncated_explanation", reason="explanation has no explicit verdict")
    return result


def conformance_cases() -> list[dict[str, Any]]:
    return [
        {"text": "A", "mapping": "mu_ab", "event_type": "strict_single_label", "label_value": "A", "semantic_value": True},
        {"text": "B", "mapping": "mu_ab", "event_type": "strict_single_label", "label_value": "B", "semantic_value": False},
        {"text": "A", "mapping": "mu_ba", "event_type": "strict_single_label", "label_value": "A", "semantic_value": False},
        {"text": "B", "mapping": "mu_ba", "event_type": "strict_single_label", "label_value": "B", "semantic_value": True},
        {"text": "A.", "mapping": "mu_ab", "event_type": "wrapped_single_label", "label_value": "A", "semantic_value": True},
        {"text": "Answer: B", "mapping": "mu_ab", "event_type": "wrapped_single_label", "label_value": "B", "semantic_value": False},
        {"text": "Label is A because it follows.", "mapping": "mu_ba", "event_type": "wrapped_single_label", "label_value": "A", "semantic_value": False},
        {"text": "true", "mapping": "mu_ab", "event_type": "truth_word_verdict", "label_value": None, "semantic_value": True},
        {"text": "false. Explanation follows.", "mapping": "mu_ba", "event_type": "truth_word_verdict", "label_value": None, "semantic_value": False},
        {"text": "The claim is true.", "mapping": "mu_ab", "event_type": "truth_word_verdict", "label_value": None, "semantic_value": True},
        {"text": "Yes, the claim follows.", "mapping": "mu_ab", "event_type": "yes_no_verdict", "label_value": None, "semantic_value": True},
        {"text": "No, it does not.", "mapping": "mu_ab", "event_type": "yes_no_verdict", "label_value": None, "semantic_value": False},
        {"text": "Therefore, the statement is false.", "mapping": "mu_ab", "event_type": "explicit_explanation_conclusion", "label_value": None, "semantic_value": False},
        {"text": "Conclusion: true", "mapping": "mu_ab", "event_type": "explicit_explanation_conclusion", "label_value": None, "semantic_value": True},
        {"text": "Map: true=A; false=B.", "mapping": "mu_ab", "event_type": "mapping_echo", "label_value": None, "semantic_value": None},
        {"text": "true=A; false=B", "mapping": "mu_ab", "event_type": "mapping_echo", "label_value": None, "semantic_value": None},
        {"text": "Coding: true corresponds to B; false corresponds to A.", "mapping": "mu_ba", "event_type": "mapping_echo", "label_value": None, "semantic_value": None},
        {"text": "true (B)", "mapping": "mu_ab", "event_type": "conflicting_label_and_truth", "label_value": "B", "semantic_value": None},
        {"text": "false (A)", "mapping": "mu_ab", "event_type": "conflicting_label_and_truth", "label_value": "A", "semantic_value": None},
        {"text": "true (B)", "mapping": "mu_ba", "event_type": "wrapped_single_label", "label_value": "B", "semantic_value": True},
        {"text": "false (A)", "mapping": "mu_ba", "event_type": "wrapped_single_label", "label_value": "A", "semantic_value": False},
        {"text": "To determine the validity", "mapping": "mu_ab", "event_type": "truncated_explanation", "label_value": None, "semantic_value": None},
        {"text": "Step 1:", "mapping": "mu_ab", "event_type": "truncated_explanation", "label_value": None, "semantic_value": None},
        {"text": "The claim that e10000", "mapping": "mu_ab", "event_type": "truncated_explanation", "label_value": None, "semantic_value": None},
        {"text": "e10000 has marker", "mapping": "mu_ab", "event_type": "content_continuation", "label_value": None, "semantic_value": None},
        {"text": "true or false", "mapping": "mu_ab", "event_type": "ambiguous_truth_words", "label_value": None, "semantic_value": None},
        {"text": "", "mapping": "mu_ab", "event_type": "empty", "label_value": None, "semantic_value": None},
        {"text": "I cannot answer", "mapping": "mu_ab", "event_type": "unrelated_or_unresolved", "label_value": None, "semantic_value": None},
    ]


def run_conformance() -> dict[str, Any]:
    rows = []
    passed = 0
    for index, expected in enumerate(conformance_cases()):
        actual = parse_output_event(expected["text"], expected["mapping"])
        keys = ("event_type", "label_value", "semantic_value")
        ok = all(actual[key] == expected[key] for key in keys)
        passed += int(ok)
        rows.append({"case_index": index, "pass": ok, "expected": expected, "actual": actual})
    return {
        "schema_version": "phase487_event_parser_conformance.v1",
        "status": "pass" if passed == len(rows) else "fail",
        "target_blind_by_signature": True,
        "normative_truth_argument_present": False,
        "n": len(rows),
        "passed": passed,
        "rate": passed / len(rows),
        "scope": "engineering_conformance_only_not_independent_human_precision",
        "cases": rows,
    }


def corrected_phase486_audit() -> dict[str, Any]:
    source = load_jsonl(PHASE486_ROWS)
    rows = []
    for row in source:
        parsed = parse_output_event(row["generated_text"], row["label_mapping"])
        semantic_outcome = "unrecoverable"
        if parsed["semantic_value"] is not None:
            semantic_outcome = "correct" if parsed["semantic_value"] == row["truth_value"] else "wrong"
        label_outcome = "unrecoverable"
        if parsed["label_value"] is not None:
            label_outcome = "correct" if parsed["label_value"] == row["expected_label"] else "wrong"
        strict_outcome = "not_strict"
        if parsed["event_type"] == "strict_single_label":
            strict_outcome = "correct" if parsed["label_value"] == row["expected_label"] else "wrong"
        rows.append({**row, **parsed, "semantic_outcome_v2": semantic_outcome, "label_outcome_v2": label_outcome, "strict_outcome_v2": strict_outcome})

    def report(items: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            "n": len(items),
            "strict_exact": dict(Counter(row["strict_outcome_v2"] for row in items)),
            "semantic": dict(Counter(row["semantic_outcome_v2"] for row in items)),
            "label": dict(Counter(row["label_outcome_v2"] for row in items)),
            "events": dict(Counter(row["event_type"] for row in items)),
            **({field: items[0][field] for field in fields} if items else {}),
        }

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["variant_track"],)].append(row)
    reports = [report(items, ("variant_track",)) for _key, items in sorted(groups.items())]
    totals = report(rows, ())
    core = [row for row in rows if row["variant_track"] != "order_stress_claim_first"]
    core_report = report(core, ())
    for payload in (totals, core_report, *reports):
        semantic = payload["semantic"]
        payload["semantic_ledger_closes"] = sum(semantic.values()) == payload["n"]
        strict = payload["strict_exact"]
        payload["strict_ledger_closes"] = sum(strict.values()) == payload["n"]
    legacy_core = {
        "n": 4608,
        "semantic_after_recovery": 4196,
        "strict_wrong": 91,
        "legacy_unrecoverable_or_ambiguous": 321,
    }
    return {
        "schema_version": "phase487_phase486_corrected_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "readonly_correction_complete",
        "input": str(PHASE486_ROWS.relative_to(ROOT)),
        "parser_does_not_receive_normative_truth": True,
        "legacy_core_summary": legacy_core,
        "corrected_all_open": totals,
        "corrected_core_tracks": core_report,
        "by_track": reports,
        "correction_notes": [
            "Phase485 prefix classification was not an exact single-label event test.",
            "Phase486 legacy remainder mixed recovered semantic errors with unrecoverable outputs.",
            "Phase484 identity and core_surface_plain prompts were byte-identical.",
        ],
    }


def code(prefix: str, split_index: int, family_index: int, pair_index: int, slot: int) -> str:
    return f"{prefix}{split_index}{family_index}{pair_index:03d}{slot}"


def build_facts(split_index: int, family_index: int, pair_index: int) -> dict[str, Any]:
    entities = [code("u", split_index, family_index, pair_index, slot) for slot in range(2)]
    groups = [code("g", split_index, family_index, pair_index, slot) for slot in range(2)]
    attrs = [code("v", split_index, family_index, pair_index, slot) for slot in range(2)]
    family = FAMILIES[family_index]
    if family == "marker_inheritance":
        blocks = [
            [f"Every {groups[i]} carries marker {attrs[i]}.", f"{entities[i]} belongs to {groups[i]}."]
            for i in range(2)
        ]
        claim = lambda entity, attr: f"{entity} carries marker {attr}."
    elif family == "signal_assignment":
        blocks = [
            [f"All members of {groups[i]} use signal {attrs[i]}.", f"{entities[i]} is assigned to {groups[i]}."]
            for i in range(2)
        ]
        claim = lambda entity, attr: f"{entity} uses signal {attr}."
    else:
        raise ValueError(family)
    if pair_index % 2:
        blocks.reverse()
    return {
        "family": family,
        "entities": entities,
        "groups": groups,
        "attrs": attrs,
        "facts": [fact for block in blocks for fact in block],
        "claim_fn": claim,
    }


def render_body(track: str, facts: list[str], claim: str) -> str:
    fact_text = " ".join(facts)
    if track == "identity":
        return f"Records: {fact_text}\nClaim: {claim}"
    if track == "native_plain_candidate":
        return f"Evidence: {fact_text}\nStatement: {claim}"
    if track == "formal_light_stress":
        return f"Given items - {fact_text}\nAssess - {claim}"
    if track == "order_stress_claim_first":
        return f"Claim: {claim}\nRecords: {fact_text}"
    raise ValueError(track)


def mapping_text(track: str, mapping: str) -> str:
    true_label = LABEL_MAPPINGS[mapping][True]
    false_label = LABEL_MAPPINGS[mapping][False]
    if track in {"identity", "order_stress_claim_first"}:
        return f"Map: true={true_label}; false={false_label}."
    if track == "native_plain_candidate":
        return f"Coding: true corresponds to {true_label}; false corresponds to {false_label}."
    if track == "formal_light_stress":
        return f"Use {true_label} for true and {false_label} for false."
    raise ValueError(track)


def render_prompts(track: str, body: str, mapping: str, mapping_position: str) -> dict[str, str]:
    semantic_prompt = f"{body}\nQuestion: Is the claim true or false?\nVerdict:"
    mapping_line = mapping_text(track, mapping)
    if mapping_position == "before":
        mapped_body = f"{mapping_line}\n{body}"
    else:
        mapped_body = f"{body}\n{mapping_line}"
    event_prompt = f"{mapped_body}\nReturn only one mapped label, A or B, then stop.\nLabel:"
    return {"semantic_prompt": semantic_prompt, "event_prompt": event_prompt}


def build_split(split: str, split_index: int) -> list[dict[str, Any]]:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for pair_index in range(PAIRS_PER_FAMILY_SPLIT):
            bundle = build_facts(split_index, family_index, pair_index)
            target_slot = (pair_index + family_index) % 2
            true_attr = bundle["attrs"][target_slot]
            false_attr = bundle["attrs"][1 - target_slot]
            target_entity = bundle["entities"][target_slot]
            mapping_position = "before" if pair_index % 2 == 0 else "after"
            source_pair_id = stable_hash("phase487", split, family, pair_index, "pair")
            for truth_value, pair_role in ((True, "entailed"), (False, "counterfactual")):
                query_attr = true_attr if truth_value else false_attr
                claim = bundle["claim_fn"](target_entity, query_attr)
                for mapping in LABEL_MAPPINGS:
                    sample_id = stable_hash("phase487", split, family, pair_index, truth_value, mapping)
                    variants = []
                    for track in TRACKS:
                        body = render_body(track, bundle["facts"], claim)
                        prompts = render_prompts(track, body, mapping, mapping_position)
                        variants.append({
                            "track": track,
                            "track_class": (
                                "native_core_candidate"
                                if track in {"identity", "native_plain_candidate"}
                                else "stress_control"
                            ),
                            "body": body,
                            **prompts,
                        })
                    rows.append({
                        "schema_version": "phase487_sample.v1",
                        "sample_id": sample_id,
                        "source_pair_id": source_pair_id,
                        "split": split,
                        "sealed": split == SEALED_SPLIT,
                        "family": family,
                        "pair_index": pair_index,
                        "pair_role": pair_role,
                        "truth_value": truth_value,
                        "label_mapping": mapping,
                        "expected_label": LABEL_MAPPINGS[mapping][truth_value],
                        "target_slot": target_slot,
                        "mapping_position": mapping_position,
                        "fact_order": "swapped" if pair_index % 2 else "canonical",
                        "facts": bundle["facts"],
                        "claim": claim,
                        "surface_variants": variants,
                    })
    return rows


def build_protocol() -> dict[str, Any]:
    return {
        "schema_version": "phase487_dual_observer_native_core_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_model_run",
        "models_in_required_order": list(MODELS),
        "tracks": {
            "native_core_candidates": ["identity", "native_plain_candidate"],
            "formal_stress": ["formal_light_stress"],
            "topology_stress": ["order_stress_claim_first"],
        },
        "channels": {
            "semantic_candidate": "compare next-token logits for ' true' and ' false' at a fixed positive-polarity verdict boundary",
            "label_candidate": "compare first-step generation logits for ' A' and ' B' at the mapped-label boundary",
            "output_event": "greedy free generation parsed by the frozen target-blind event parser",
        },
        "sample_design": {
            "families": list(FAMILIES),
            "pairs_per_family_split": PAIRS_PER_FAMILY_SPLIT,
            "truth_values": [True, False],
            "label_mappings": list(LABEL_MAPPINGS),
            "target_slots": [0, 1],
            "mapping_positions": ["before", "after"],
            "fact_orders": ["canonical", "swapped"],
        },
        "split_isolation": {
            "separate_files": True,
            "behavior_script_may_read": ["behavior_qualification"],
            "geometry_script_may_read_after_gate": ["geometry_window"],
            "prediction_script_may_read_after_window_freeze": ["physical_prediction"],
            "sealed_read_authorized": False,
        },
        "gates": {
            "semantic_identity_lcb95_min": 0.95,
            "semantic_plain_lcb95_min": 0.95,
            "semantic_identity_plain_intersection_lcb95_min": 0.90,
            "label_identity_lcb95_min": 0.95,
            "label_plain_lcb95_min": 0.95,
            "event_unrecoverable_ucb95_max": 0.05,
            "strict_event_reported_separately": True,
            "relation_geometry_requires_semantic_gates": True,
            "event_map_requires_parser_conformance_and_more_than_one_event_class": True,
        },
        "forbidden": [
            "read sealed_physical",
            "modify parser after seeing new generations",
            "pool stress tracks into native-core denominator",
            "treat free-generation format as relation-semantic failure",
            "head channel or neuron scan",
            "causal intervention",
        ],
    }


def static_audit(split_rows: dict[str, list[dict[str, Any]]], conformance: dict[str, Any]) -> dict[str, Any]:
    expected_samples = len(FAMILIES) * PAIRS_PER_FAMILY_SPLIT * 2 * len(LABEL_MAPPINGS)
    failures = []
    split_reports = {}
    for split, rows in split_rows.items():
        variants = [variant for row in rows for variant in row["surface_variants"]]
        counts = {
            "samples": len(rows),
            "variants": len(variants),
            "truth": dict(Counter(str(row["truth_value"]) for row in rows)),
            "mapping": dict(Counter(row["label_mapping"] for row in rows)),
            "family": dict(Counter(row["family"] for row in rows)),
            "track": dict(Counter(variant["track"] for variant in variants)),
        }
        split_reports[split] = counts
        if len(rows) != expected_samples:
            failures.append(f"{split}: sample count {len(rows)} != {expected_samples}")
        if len(variants) != expected_samples * len(TRACKS):
            failures.append(f"{split}: variant count mismatch")
        for row in rows:
            by_track = {variant["track"]: variant for variant in row["surface_variants"]}
            if by_track["identity"]["body"] == by_track["native_plain_candidate"]["body"]:
                failures.append(f"{split}/{row['sample_id']}: identity/plain body collision")
            if by_track["identity"]["semantic_prompt"] == by_track["native_plain_candidate"]["semantic_prompt"]:
                failures.append(f"{split}/{row['sample_id']}: identity/plain semantic prompt collision")
            if by_track["identity"]["event_prompt"] == by_track["native_plain_candidate"]["event_prompt"]:
                failures.append(f"{split}/{row['sample_id']}: identity/plain event prompt collision")
    if conformance["status"] != "pass":
        failures.append("event parser conformance failed")
    return {
        "schema_version": "phase487_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if not failures else "static_fail_no_model_run",
        "failures": failures[:100],
        "checks": {
            "parser_target_blind_signature": conformance["target_blind_by_signature"],
            "parser_conformance_pass": conformance["status"] == "pass",
            "identity_plain_byte_distinct_for_every_sample": not any("collision" in failure for failure in failures),
            "split_files_physically_separate": True,
            "sealed_split_read_during_static_generation": False,
        },
        "split_reports": split_reports,
        "authorization": {
            "new_behavior_qualification_authorized": not failures,
            "geometry_collection_authorized": False,
            "sealed_read_authorized": False,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conformance = run_conformance()
    CONFORMANCE_PATH.write_text(json.dumps(conformance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    correction = corrected_phase486_audit()
    CORRECTION_PATH.write_text(json.dumps(correction, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = build_protocol()
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_rows = {split: build_split(split, index) for index, split in enumerate(SPLITS)}
    for split, rows in split_rows.items():
        write_jsonl(SPLIT_FILES[split], rows)
    audit = static_audit(split_rows, conformance)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "phase487_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(source_path),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "event_parser_conformance_sha256": sha256_file(CONFORMANCE_PATH),
        "phase486_correction_sha256": sha256_file(CORRECTION_PATH),
        "split_files": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "sample_count": len(split_rows[split]),
                "sealed": split == SEALED_SPLIT,
            }
            for split, path in SPLIT_FILES.items()
        },
        "static_audit_sha256": sha256_file(AUDIT_PATH),
        "frozen_before_new_model_run": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(PROTOCOL_PATH)
    print(CORRECTION_PATH)
    print(AUDIT_PATH)
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
