#!/usr/bin/env python3
"""Phase1281: freeze C024 expectation-reversal natural-use materials.

The target is not a metalinguistic relation label.  It is the change in full
continuation log probability produced by a discourse construction that says a
stated expectation was satisfied or violated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 1281
CAMPAIGN = "C024"
CONTRACT_ID = "EXP-C024-WP00-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1281_c024_expectation_reversal_contract_audit.py"
OUT = ROOT / "tests/glm5/result/phase1281_c024_expectation_reversal_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_expectation_worlds.jsonl"
FINAL = OUT / "analysis/final.json"

MODEL_PATH = Path(MODEL_CONFIGS["qwen3"]["path"])
PARTITION_COUNTS = {"discovery": 64, "selection": 64, "confirmation": 128}
SURFACES = ("coordination", "adverbial", "expectation", "evaluation", "report")
PANELS = (
    "consistency", "contrast", "carrier_consistency", "lexical_consistency",
    "carrier_contrast", "lexical_contrast",
)
CONSISTENCY_PANELS = ("consistency", "carrier_consistency", "lexical_consistency")
CONTRAST_PANELS = ("contrast", "carrier_contrast", "lexical_contrast")
AXES = (
    ("weight", "light", "heavy"), ("size", "small", "large"),
    ("temperature", "cold", "hot"), ("texture", "smooth", "rough"),
    ("sound", "quiet", "loud"), ("brightness", "bright", "dim"),
    ("strength", "weak", "strong"), ("flexibility", "rigid", "flexible"),
    ("speed", "slow", "fast"), ("cleanliness", "clean", "dirty"),
    ("moisture", "dry", "wet"), ("hardness", "soft", "hard"),
    ("length", "short", "long"), ("openness", "open", "closed"),
    ("capacity", "empty", "full"), ("width", "narrow", "wide"),
)
NAMES = {
    "discovery": ("Adele", "Boris", "Celia", "Damon", "Elsa", "Faris", "Gina", "Hector"),
    "selection": ("Iris", "Jonas", "Kira", "Lance", "Mabel", "Nico", "Orla", "Perry"),
    "confirmation": ("Quinn", "Rhea", "Silas", "Tessa", "Ulric", "Vera", "Wes", "Xena"),
}
ITEMS = {
    "discovery": ("crate", "parcel", "drink", "fabric", "engine", "lamp", "signal", "panel", "vehicle", "uniform", "towel", "cushion", "cable", "gate", "container", "corridor"),
    "selection": ("package", "box", "soup", "surface", "machine", "screen", "signal", "sheet", "cart", "jacket", "cloth", "pillow", "rope", "door", "tank", "path"),
    "confirmation": ("suitcase", "bundle", "tea", "material", "motor", "display", "signal", "board", "train", "coat", "blanket", "mattress", "wire", "hatch", "bottle", "lane"),
}
CONTRAST_CUES = {
    "coordination": "but",
    "adverbial": "However",
    "expectation": "Contrary to that expectation",
    "evaluation": "mistaken",
    "report": "contradicted the prediction",
}
CARRIER_CUES = {
    "coordination": "blue",
    "adverbial": "Silver",
    "expectation": "Beside the old window",
    "evaluation": "wooden",
    "report": "measured the old table",
}
THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "core_sign_accuracy_min": 0.85,
    "null_sign_accuracy_min": 0.80,
    "effect_positive_fraction_min": 0.85,
    "median_functional_effect_min": 4.0,
    "lexical_specific_ratio_max": 0.30,
    "generation_parse_coverage_min": 0.75,
    "generation_sign_accuracy_min": 0.70,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def note(cue: str) -> str:
    return f'A vocabulary card displayed the expression "{cue}". '


def bare_context(surface: str, mode: str, name: str, item: str, expected: str) -> tuple[str, list[int], list[int]]:
    if surface == "coordination":
        prefix = f"{name} expected the {item} to be {expected}, "
        cue = "and" if mode == "consistency" else "but"
        suffix = " it was"
    elif surface == "adverbial":
        prefix = f"{name} expected the {item} to be {expected}. "
        cue = "Indeed" if mode == "consistency" else "However"
        suffix = ", it was"
    elif surface == "expectation":
        prefix = f"{name} expected the {item} to be {expected}. "
        cue = "As expected" if mode == "consistency" else "Contrary to that expectation"
        suffix = ", it was"
    elif surface == "evaluation":
        prefix = f"{name}'s expectation that the {item} would be {expected} proved "
        cue = "correct" if mode == "consistency" else "mistaken"
        suffix = "; it was"
    else:
        prefix = f"{name} predicted that the {item} would be {expected}. The final result "
        cue = "matched the prediction" if mode == "consistency" else "contradicted the prediction"
        suffix = ": it was"
    cue_span = [len(prefix), len(prefix) + len(cue)]
    context = prefix + cue + suffix
    expectation_end = context.find(expected) + len(expected)
    return context, cue_span, [expectation_end - 1, expectation_end]


def add_note(context: str, cue_span: list[int], expectation_span: list[int], cue: str) -> tuple[str, list[int], list[int], list[int]]:
    prefix = note(cue)
    note_start = prefix.find(cue)
    return (
        prefix + context,
        [cue_span[0] + len(prefix), cue_span[1] + len(prefix)],
        [expectation_span[0] + len(prefix), expectation_span[1] + len(prefix)],
        [note_start, note_start + len(cue)],
    )


def make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, count in PARTITION_COUNTS.items():
        for index in range(count):
            axis_index = index % len(AXES)
            occurrence = index // len(AXES)
            axis, left, right = AXES[axis_index]
            orientation = occurrence % 2
            expected, opposite = (left, right) if orientation == 0 else (right, left)
            name = NAMES[partition][(axis_index + occurrence) % len(NAMES[partition])]
            item = ITEMS[partition][axis_index]
            contexts: dict[str, dict[str, str]] = {}
            events: dict[str, dict[str, dict[str, list[int]]]] = {}
            for surface in SURFACES:
                consistency, c_cue, c_expect = bare_context(surface, "consistency", name, item, expected)
                contrast, r_cue, r_expect = bare_context(surface, "contrast", name, item, expected)
                contrast_cue = CONTRAST_CUES[surface]
                carrier_cue = CARRIER_CUES[surface]
                cc, cc_cue, cc_expect, cc_note = add_note(consistency, c_cue, c_expect, carrier_cue)
                lc, lc_cue, lc_expect, lc_note = add_note(consistency, c_cue, c_expect, contrast_cue)
                cr, cr_cue, cr_expect, cr_note = add_note(contrast, r_cue, r_expect, carrier_cue)
                lr, lr_cue, lr_expect, lr_note = add_note(contrast, r_cue, r_expect, contrast_cue)
                contexts[surface] = {
                    "consistency": consistency,
                    "contrast": contrast,
                    "carrier_consistency": cc,
                    "lexical_consistency": lc,
                    "carrier_contrast": cr,
                    "lexical_contrast": lr,
                }
                events[surface] = {
                    "consistency": {"expectation_end": c_expect, "relation_cue": c_cue, "context_end": [len(consistency) - 1, len(consistency)]},
                    "contrast": {"expectation_end": r_expect, "relation_cue": r_cue, "context_end": [len(contrast) - 1, len(contrast)]},
                    "carrier_consistency": {"note_cue": cc_note, "expectation_end": cc_expect, "relation_cue": cc_cue, "context_end": [len(cc) - 1, len(cc)]},
                    "lexical_consistency": {"note_cue": lc_note, "expectation_end": lc_expect, "relation_cue": lc_cue, "context_end": [len(lc) - 1, len(lc)]},
                    "carrier_contrast": {"note_cue": cr_note, "expectation_end": cr_expect, "relation_cue": cr_cue, "context_end": [len(cr) - 1, len(cr)]},
                    "lexical_contrast": {"note_cue": lr_note, "expectation_end": lr_expect, "relation_cue": lr_cue, "context_end": [len(lr) - 1, len(lr)]},
                }
            row = {
                "row_id": f"{partition}-{index:04d}",
                "partition": partition,
                "axis": axis,
                "axis_index": axis_index,
                "orientation": orientation,
                "name": name,
                "item": item,
                "expected_adjective": expected,
                "opposite_adjective": opposite,
                "continuations": {
                    "expected": f" {expected} in the final report.",
                    "opposite": f" {opposite} in the final report.",
                },
                "contexts": contexts,
                "event_char_spans": events,
            }
            row["row_digest"] = digest(row)
            rows.append(row)
    return rows


def span_tokens(offsets: list[tuple[int, int]], span: list[int]) -> list[int]:
    return [index for index, (left, right) in enumerate(offsets) if right > span[0] and left < span[1]]


def tokenizer_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    event_indices: dict[str, Any] = {}
    material_ids: dict[str, Any] = {}
    fast_slow_equal = True
    prefix_stable = True
    equal_candidate_lengths = True
    candidates_multitoken = True
    note_lengths_equal = True
    cue_lengths_matched = True
    contexts_end_correctly = True
    for surface in SURFACES:
        cue_lengths_matched &= len(slow.encode(" " + CONTRAST_CUES[surface], add_special_tokens=False)) == len(slow.encode(" " + CARRIER_CUES[surface], add_special_tokens=False))
    for row in rows:
        row_events: dict[str, Any] = {}
        row_material: dict[str, Any] = {}
        for surface in SURFACES:
            row_events[surface] = {}
            row_material[surface] = {}
            for panel in PANELS:
                context = row["contexts"][surface][panel]
                contexts_end_correctly &= context.endswith("it was")
                fast_value = fast(context, add_special_tokens=False, return_offsets_mapping=True)
                fast_ids = list(fast_value["input_ids"])
                slow_ids = slow.encode(context, add_special_tokens=False)
                offsets = [tuple(value) for value in fast_value["offset_mapping"]]
                fast_slow_equal &= fast_ids == slow_ids
                row_material[surface][panel] = {"context_ids": slow_ids, "continuations": {}}
                row_events[surface][panel] = {}
                for event, span in row["event_char_spans"][surface][panel].items():
                    indices = span_tokens(offsets, span)
                    if not indices:
                        raise RuntimeError(f"empty event span {row['row_id']} {surface} {panel} {event}")
                    row_events[surface][panel][event] = {"start": min(indices), "end": max(indices)}
                lengths = []
                for identity, continuation in row["continuations"].items():
                    full = slow.encode(context + continuation, add_special_tokens=False)
                    prefix_stable &= full[:len(slow_ids)] == slow_ids
                    suffix = full[len(slow_ids):]
                    lengths.append(len(suffix))
                    candidates_multitoken &= len(suffix) >= 4
                    row_material[surface][panel]["continuations"][identity] = suffix
                equal_candidate_lengths &= lengths[0] == lengths[1]
            note_lengths_equal &= len(row_material[surface]["carrier_consistency"]["context_ids"]) == len(row_material[surface]["lexical_consistency"]["context_ids"])
            note_lengths_equal &= len(row_material[surface]["carrier_contrast"]["context_ids"]) == len(row_material[surface]["lexical_contrast"]["context_ids"])
        event_indices[row["row_id"]] = row_events
        material_ids[row["row_id"]] = row_material
    return {
        "fast_slow_tokenization_equal": fast_slow_equal,
        "context_prefix_stable_under_continuation": prefix_stable,
        "candidate_lengths_equal_within_world": equal_candidate_lengths,
        "all_candidates_multitoken": candidates_multitoken,
        "carrier_lexical_context_lengths_equal": note_lengths_equal,
        "contrast_carrier_cue_token_lengths_matched": cue_lengths_matched,
        "all_contexts_end_it_was": contexts_end_correctly,
        "event_token_indices": event_indices,
        "token_material_digest": digest(material_ids),
    }


def semantic_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partition_axis_orientation = {}
    for partition in PARTITION_COUNTS:
        subset = [row for row in rows if row["partition"] == partition]
        partition_axis_orientation[partition] = {
            f"{axis}.{orientation}": sum(row["axis"] == axis and row["orientation"] == orientation for row in subset)
            for axis, _, _ in AXES for orientation in (0, 1)
        }
    return {
        "row_count": len(rows),
        "partition_axis_orientation_counts": partition_axis_orientation,
        "row_ids_unique": len({row["row_id"] for row in rows}) == len(rows),
        "world_descriptions_unique": len({(row["partition"], row["name"], row["item"], row["expected_adjective"], row["opposite_adjective"]) for row in rows}) == len(rows),
        "expected_opposite_distinct": all(row["expected_adjective"] != row["opposite_adjective"] for row in rows),
        "orientation_balanced": all(abs(value[f"{axis}.0"] - value[f"{axis}.1"]) <= 1 for value in partition_axis_orientation.values() for axis, _, _ in AXES),
        "expected_present_opposite_absent": all(
            row["expected_adjective"] in context and row["opposite_adjective"] not in context
            for row in rows for surface in SURFACES for context in row["contexts"][surface].values()
        ),
        "all_panels_registered": all(set(row["contexts"][surface]) == set(PANELS) for row in rows for surface in SURFACES),
        "semantic_scope": "Explicit expectation satisfaction versus violation over curated antonym pairs; no claim that this exhausts linguistic contrast.",
        "independent_human_labels": False,
    }


def protocol_payload(rows: list[dict[str, Any]], tokens: dict[str, Any], semantics: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1281.c024.expectation.reversal.contract.v1",
        "object": "full-continuation probability effect of expectation satisfaction versus expectation violation",
        "partitions": PARTITION_COUNTS,
        "surfaces": list(SURFACES),
        "panels": list(PANELS),
        "consistency_panels": list(CONSISTENCY_PANELS),
        "contrast_panels": list(CONTRAST_PANELS),
        "axes": [list(value) for value in AXES],
        "contrast_cues": CONTRAST_CUES,
        "carrier_cues": CARRIER_CUES,
        "thresholds": THRESHOLDS,
        "token_audit": tokens,
        "semantic_audit": semantics,
        "world_digest": digest([{key: row[key] for key in ("row_id", "partition", "axis", "orientation", "row_digest")} for row in rows]),
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "hard_stops": [
            "Phase1281 is zero-model; no formal model output is inspected.",
            "The object is expectation satisfaction/violation, not all uses of contrast.",
            "Full multi-token continuation log probability is primary; a one-token adjective diagnostic cannot replace it.",
            "Expected and opposite orientations are balanced for every semantic axis.",
            "Quoted contrast cue is compared with an equal-token carrier cue under both consistency and contrast contexts.",
            "No independent human annotation is present; claims are limited to the explicit logical-pragmatic contract.",
            "Behavior failure denies hidden-state study; behavior success authorizes Qwen3 only.",
            "No threshold, wording, adjective pair, surface, panel or denominator may change after formal scoring starts.",
        ],
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def run(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    rows = make_rows()
    tokens = tokenizer_audit(rows)
    semantics = semantic_audit(rows)
    required_tokens = [value for key, value in tokens.items() if isinstance(value, bool)]
    required_semantics = [semantics[key] for key in (
        "row_ids_unique", "world_descriptions_unique", "expected_opposite_distinct",
        "orientation_balanced", "expected_present_opposite_absent", "all_panels_registered",
    )]
    if not all(required_tokens + required_semantics):
        raise RuntimeError(f"contract qualification failed: {tokens} {semantics}")
    write_jsonl(MATERIAL, rows)
    protocol = protocol_payload(rows, tokens, semantics)
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "model_path_for_token_audit_only": str(MODEL_PATH),
    })
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "expectation_reversal_natural_use_contract_frozen",
        "row_count": len(rows),
        "context_count": len(rows) * len(SURFACES) * len(PANELS),
        "scored_sequence_count": len(rows) * len(SURFACES) * len(PANELS) * 2,
        "semantic_scope": semantics["semantic_scope"],
        "human_annotation_status": "not_present_claim_scope_limited",
        "protocol_digest": protocol["protocol_digest"],
        "authorization": "phase1282_qwen3_multitoken_behavior_and_generation",
    }
    atomic_json(FINAL, final)
    print(canonical_json(final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.force)
