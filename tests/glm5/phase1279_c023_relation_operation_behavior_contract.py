#!/usr/bin/env python3
"""Phase1279: freeze the C023 relation-operation behavior contract.

This is a zero-model phase.  It creates disjoint discovery, selection and
confirmation worlds for four English discourse-connective operations.  The
factorial panels separate a connector in the operative clause boundary from
the same word appearing only in a quoted vocabulary note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 1279
CAMPAIGN = "C023"
CONTRACT_ID = "EXP-C023-WP00-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1279_c023_relation_operation_behavior_contract_audit.py"
OUT = ROOT / "tests/glm5/result/phase1279_c023_relation_operation_behavior_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_relation_worlds.jsonl"
SUMMARY = OUT / "analysis/final.json"

MODEL_PATH = Path(MODEL_CONFIGS["qwen3"]["path"])
OPERATIONS = ("contrast", "addition", "cause", "sequence")
CANONICAL = {
    "contrast": "but",
    "addition": "and",
    "cause": "so",
    "sequence": "then",
}
SYNONYM = {
    "contrast": "however",
    "addition": "moreover",
    "cause": "therefore",
    "sequence": "afterward",
}
PANELS = ("base", "target", "wrong", "null", "joint", "surface", "implicit")
CAUSAL_PANELS = ("base", "target", "wrong", "null", "joint")
PARTITION_COUNTS = {"discovery": 64, "selection": 64, "confirmation": 128}
PARTITION_SEEDS = {"discovery": 127_910_1, "selection": 127_920_1, "confirmation": 127_930_1}
LABEL_ORDERS = list(permutations(OPERATIONS))
THRESHOLDS = {
    "candidate_finite_fraction_min": 1.0,
    "factorial_cell_accuracy_min": 0.90,
    "surface_cell_accuracy_min": 0.85,
    "implicit_cell_accuracy_min": 0.80,
    "operation_macro_accuracy_min": 0.85,
    "gold_margin_median_min": 0.50,
}


VOCAB = {
    "discovery": {
        "names": ("Adele", "Boris", "Celia", "Damon", "Elsa", "Faris", "Gina", "Hector"),
        "contrast": (
            ("parcel", "light", "heavy"), ("hall", "empty", "crowded"),
            ("road", "clear", "blocked"), ("fabric", "smooth", "rough"),
            ("drink", "cold", "warm"), ("room", "quiet", "noisy"),
            ("rope", "short", "long"), ("screen", "bright", "dim"),
        ),
        "addition": (
            ("notebook", "pencil"), ("map", "compass"), ("hammer", "wrench"),
            ("scarf", "gloves"), ("plate", "spoon"), ("ticket", "receipt"),
            ("camera", "tripod"), ("bottle", "cup"),
        ),
        "cause": (
            ("switch", "lamp lit"), ("valve", "water flowed"), ("alarm", "staff departed"),
            ("lever", "gate opened"), ("heater", "room warmed"), ("brake", "cart stopped"),
            ("pump", "tank filled"), ("bell", "class assembled"),
        ),
        "sequence": (
            ("signed the form", "mailed the form"), ("washed the cup", "shelved the cup"),
            ("opened the case", "inspected the case"), ("folded the letter", "sealed the letter"),
            ("checked the lock", "closed the door"), ("read the note", "filed the note"),
            ("measured the board", "cut the board"), ("charged the phone", "packed the phone"),
        ),
    },
    "selection": {
        "names": ("Iris", "Jonas", "Kira", "Lance", "Mabel", "Nico", "Orla", "Perry"),
        "contrast": (
            ("crate", "small", "large"), ("corridor", "vacant", "busy"),
            ("trail", "dry", "muddy"), ("surface", "soft", "hard"),
            ("soup", "mild", "spicy"), ("engine", "silent", "loud"),
            ("cable", "thin", "thick"), ("window", "clean", "cloudy"),
        ),
        "addition": (
            ("folder", "marker"), ("chart", "ruler"), ("pliers", "saw"),
            ("jacket", "boots"), ("bowl", "fork"), ("pass", "invoice"),
            ("radio", "antenna"), ("kettle", "mug"),
        ),
        "cause": (
            ("button", "fan started"), ("tap", "basin filled"), ("siren", "guards arrived"),
            ("handle", "drawer opened"), ("stove", "water boiled"), ("pedal", "machine stopped"),
            ("motor", "belt moved"), ("chime", "guests entered"),
        ),
        "sequence": (
            ("stamped the card", "posted the card"), ("rinsed the jar", "stored the jar"),
            ("unlocked the chest", "examined the chest"), ("wrapped the parcel", "labeled the parcel"),
            ("tested the latch", "shut the cabinet"), ("copied the memo", "archived the memo"),
            ("marked the pipe", "drilled the pipe"), ("updated the tablet", "boxed the tablet"),
        ),
    },
    "confirmation": {
        "names": ("Quinn", "Rhea", "Silas", "Tessa", "Ulric", "Vera", "Wes", "Xena"),
        "contrast": (
            ("package", "fragile", "sturdy"), ("theater", "deserted", "full"),
            ("path", "level", "steep"), ("panel", "flexible", "rigid"),
            ("tea", "sweet", "bitter"), ("workshop", "calm", "chaotic"),
            ("chain", "loose", "tight"), ("mirror", "shiny", "dull"),
        ),
        "addition": (
            ("ledger", "eraser"), ("diagram", "protractor"), ("chisel", "mallet"),
            ("coat", "mittens"), ("tray", "knife"), ("voucher", "statement"),
            ("projector", "screen"), ("flask", "glass"),
        ),
        "cause": (
            ("sensor", "beacon flashed"), ("faucet", "channel flooded"), ("horn", "workers paused"),
            ("crank", "platform rose"), ("burner", "metal softened"), ("clutch", "wheel halted"),
            ("turbine", "generator spun"), ("buzzer", "visitors gathered"),
        ),
        "sequence": (
            ("approved the permit", "scanned the permit"), ("polished the vase", "displayed the vase"),
            ("lifted the lid", "photographed the contents"), ("addressed the envelope", "sealed the envelope"),
            ("verified the hinge", "latched the box"), ("summarized the article", "indexed the article"),
            ("aligned the tile", "cemented the tile"), ("backed up the laptop", "shipped the laptop"),
        ),
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


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


def content_pair(partition: str, operation: str, index: int) -> tuple[str, str, str]:
    bank = VOCAB[partition]
    name = bank["names"][index % len(bank["names"])]
    item = bank[operation][(index // len(bank["names"])) % len(bank[operation])]
    if operation == "contrast":
        noun, expected, actual = item
        return f"{name} expected the {noun} to be {expected}", f"the {noun} was {actual}", f"{partition}.{operation}.{name}.{noun}"
    if operation == "addition":
        first, second = item
        return f"{name} packed the {first}", f"{name} packed the {second}", f"{partition}.{operation}.{name}.{first}.{second}"
    if operation == "cause":
        trigger, effect = item
        return f"{name} activated the {trigger}", f"the {effect}", f"{partition}.{operation}.{name}.{trigger}"
    first, second = item
    return f"{name} {first}", f"{name} {second}", f"{partition}.{operation}.{name}.{index}"


def label_instruction(order: tuple[str, ...]) -> str:
    return "Use exactly one label from this list: " + ", ".join(order) + ". Answer:"


def explicit_prompt(c1: str, c2: str, marker: str, order: tuple[str, ...], margin: str | None = None) -> tuple[str, dict[str, list[int]]]:
    prefix = ""
    spans: dict[str, list[int]] = {}
    if margin is not None:
        note_a = 'Vocabulary note: the isolated word "'
        note_b = '" is printed in the margin. '
        note_start = len(note_a)
        prefix = note_a + margin + note_b
        spans["margin_word"] = [note_start, note_start + len(margin)]
    start = prefix + 'Read the sentence: "' + c1 + ", "
    marker_start = len(start)
    sentence = start + marker + " " + c2 + '." '
    c2_end = len(start + marker + " " + c2)
    question = "What relation does the connector joining the two clauses express? "
    query_start = len(sentence)
    prompt = sentence + question + label_instruction(order)
    spans.update({
        "relation_marker": [marker_start, marker_start + len(marker)],
        "clause2_end": [c2_end - 1, c2_end],
        "query_end": [query_start + len(question) - 2, query_start + len(question) - 1],
        "answer_boundary": [len(prompt) - 1, len(prompt)],
    })
    return prompt, spans


def implicit_prompt(c1: str, c2: str, operation: str, order: tuple[str, ...]) -> str:
    if operation == "contrast":
        sentence = f'Although {c1}, {c2}.'
    elif operation == "addition":
        sentence = f'Two facts are combined here: {c1}; {c2}.'
    elif operation == "cause":
        sentence = f'{c2[0].upper() + c2[1:]} because {c1}.'
    else:
        sentence = f'After {c1}, {c2}.'
    return (
        f'Read the sentence: "{sentence}" What relation connects the two clauses? '
        + label_instruction(order)
    )


def make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, count in PARTITION_COUNTS.items():
        for index in range(count):
            target = OPERATIONS[index % len(OPERATIONS)]
            alternatives = [value for value in OPERATIONS if value != target]
            base = alternatives[(index // 4) % 3]
            wrong = [value for value in alternatives if value != base][(index // 12) % 2]
            order = LABEL_ORDERS[(index * 7 + PARTITION_SEEDS[partition]) % len(LABEL_ORDERS)]
            # Index within the operation, not the global interleaved row index,
            # so every confirmation world receives a distinct name/item pair.
            c1, c2, content_id = content_pair(partition, target, index // len(OPERATIONS))
            base_prompt, base_spans = explicit_prompt(c1, c2, CANONICAL[base], order)
            target_prompt, target_spans = explicit_prompt(c1, c2, CANONICAL[target], order)
            wrong_prompt, wrong_spans = explicit_prompt(c1, c2, CANONICAL[wrong], order)
            null_prompt, null_spans = explicit_prompt(c1, c2, CANONICAL[base], order, margin=CANONICAL[target])
            joint_prompt, joint_spans = explicit_prompt(c1, c2, CANONICAL[target], order, margin=CANONICAL[target])
            surface_prompt, surface_spans = explicit_prompt(c1, c2, SYNONYM[target], order)
            panels = {
                "base": base_prompt,
                "target": target_prompt,
                "wrong": wrong_prompt,
                "null": null_prompt,
                "joint": joint_prompt,
                "surface": surface_prompt,
                "implicit": implicit_prompt(c1, c2, target, order),
            }
            spans = {
                "base": base_spans,
                "target": target_spans,
                "wrong": wrong_spans,
                "null": null_spans,
                "joint": joint_spans,
                "surface": surface_spans,
            }
            row = {
                "row_id": f"{partition}-{index:04d}",
                "partition": partition,
                "index": index,
                "content_id": content_id,
                "content_operation": target,
                "operations": {"base": base, "target": target, "wrong": wrong},
                "markers": {
                    "base": CANONICAL[base], "target": CANONICAL[target], "wrong": CANONICAL[wrong],
                    "surface": SYNONYM[target],
                },
                "label_order": list(order),
                "expected": {
                    "base": base, "target": target, "wrong": wrong, "null": base,
                    "joint": target, "surface": target, "implicit": target,
                },
                "panels": panels,
                "event_char_spans": spans,
            }
            row["row_digest"] = digest(row)
            rows.append(row)
    return rows


def overlapping_token_indices(offsets: list[tuple[int, int]], span: list[int]) -> list[int]:
    start, end = span
    return [index for index, (left, right) in enumerate(offsets) if right > start and left < end]


def token_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    candidate_ids: dict[str, int] = {}
    for operation in OPERATIONS:
        encoded = slow.encode(" " + operation, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"candidate is not one token: {operation} -> {encoded}")
        candidate_ids[operation] = int(encoded[0])
    event_indices: dict[str, dict[str, dict[str, int]]] = {}
    panel_ids: dict[str, dict[str, list[int]]] = {}
    fast_slow_equal = True
    marker_single = True
    suffix_single = True
    active_one_difference = True
    null_joint_one_difference = True
    lengths: list[int] = []
    for row in rows:
        row_events: dict[str, dict[str, int]] = {}
        row_ids: dict[str, list[int]] = {}
        for panel, prompt in row["panels"].items():
            slow_ids = slow.encode(prompt, add_special_tokens=False)
            fast_value = fast(prompt, add_special_tokens=False, return_offsets_mapping=True)
            fast_ids = list(fast_value["input_ids"])
            offsets = [tuple(value) for value in fast_value["offset_mapping"]]
            fast_slow_equal &= slow_ids == fast_ids
            row_ids[panel] = slow_ids
            lengths.append(len(slow_ids))
            for operation in OPERATIONS:
                extended = slow.encode(prompt + " " + operation, add_special_tokens=False)
                suffix_single &= extended[:-1] == slow_ids and extended[-1] == candidate_ids[operation]
            if panel in row["event_char_spans"]:
                positions: dict[str, int] = {}
                for event, span in row["event_char_spans"][panel].items():
                    tokens = overlapping_token_indices(offsets, span)
                    if event == "relation_marker":
                        marker_single &= len(tokens) == 1
                    if len(tokens) != 1:
                        raise RuntimeError(f"event span not unique: {row['row_id']} {panel} {event} {tokens}")
                    positions[event] = int(tokens[0])
                row_events[panel] = positions
        active = [row_ids[name] for name in ("base", "target", "wrong")]
        active_one_difference &= len({len(value) for value in active}) == 1
        if len({len(value) for value in active}) == 1:
            active_one_difference &= all(sum(a != b for a, b in zip(active[0], other)) == 1 for other in active[1:])
        null_joint_one_difference &= len(row_ids["null"]) == len(row_ids["joint"])
        if len(row_ids["null"]) == len(row_ids["joint"]):
            null_joint_one_difference &= sum(a != b for a, b in zip(row_ids["null"], row_ids["joint"])) == 1
        event_indices[row["row_id"]] = row_events
        panel_ids[row["row_id"]] = row_ids
    return {
        "candidate_token_ids": candidate_ids,
        "candidate_tokens_unique": len(set(candidate_ids.values())) == len(OPERATIONS),
        "fast_slow_tokenization_equal": fast_slow_equal,
        "candidate_suffix_single_token": suffix_single,
        "operative_markers_single_token": marker_single,
        "active_panels_equal_length_one_token_difference": active_one_difference,
        "null_joint_equal_length_one_token_difference": null_joint_one_difference,
        "min_prompt_tokens": min(lengths),
        "max_prompt_tokens": max(lengths),
        "event_token_indices": event_indices,
        "token_material_digest": digest(panel_ids),
    }


def balance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for partition in PARTITION_COUNTS:
        subset = [row for row in rows if row["partition"] == partition]
        result[partition] = {
            "row_count": len(subset),
            "target_counts": {operation: sum(row["operations"]["target"] == operation for row in subset) for operation in OPERATIONS},
            "base_target_pairs": sorted({(row["operations"]["base"], row["operations"]["target"]) for row in subset}),
            "wrong_target_pairs": sorted({(row["operations"]["wrong"], row["operations"]["target"]) for row in subset}),
            "label_orders_used": len({tuple(row["label_order"]) for row in subset}),
        }
    return result


def protocol_payload(rows: list[dict[str, Any]], tokens: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_id": CONTRACT_ID,
        "schema_version": "phase1279.c023.relation.behavior.contract.v1",
        "claim_target": "Qwen3 connective-conditioned relation-address behavior, followed only if qualified by a typed causal closure test",
        "operations": list(OPERATIONS),
        "canonical_markers": CANONICAL,
        "heldout_surface_markers": SYNONYM,
        "panels": list(PANELS),
        "causal_panels": list(CAUSAL_PANELS),
        "partition_counts": PARTITION_COUNTS,
        "partition_seeds": PARTITION_SEEDS,
        "thresholds": THRESHOLDS,
        "balance": balance_summary(rows),
        "token_audit": tokens,
        "source_hashes": {"main": file_sha256(SCRIPT), "auditor": file_sha256(AUDITOR)},
        "world_digest": digest([{key: row[key] for key in ("row_id", "partition", "content_id", "row_digest")} for row in rows]),
        "hard_stops": [
            "No model output is inspected in Phase1279.",
            "The three partitions use disjoint names and content identifiers.",
            "Factorial behavior follows the connector in the operative clause boundary; a quoted occurrence is a matched lexical null.",
            "Surface and implicit panels test behavioral breadth but cannot select causal components.",
            "A behavior pass authorizes Qwen3-only typed causal study; it does not prove abstract relation semantics.",
            "A causal pass can establish at most a connective-conditioned relation-address subroutine under this prompt family.",
            "No threshold, prompt, partition, candidate set or denominator may change after formal scoring begins.",
            "GLM4 and DS7B are only eligible after a Qwen3 confirmation pass under a separately frozen transfer contract.",
        ],
    }
    return {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}


def run(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    rows = make_rows()
    tokens = token_audit(rows)
    required = (
        tokens["candidate_tokens_unique"], tokens["fast_slow_tokenization_equal"],
        tokens["candidate_suffix_single_token"], tokens["operative_markers_single_token"],
        tokens["active_panels_equal_length_one_token_difference"],
        tokens["null_joint_equal_length_one_token_difference"],
    )
    if not all(required):
        raise RuntimeError(f"token contract failed: {tokens}")
    balances = balance_summary(rows)
    if not all(value["row_count"] == PARTITION_COUNTS[key] for key, value in balances.items()):
        raise RuntimeError("partition count drift")
    if not all(len(value["base_target_pairs"]) == 12 and len(value["wrong_target_pairs"]) == 12 for value in balances.values()):
        raise RuntimeError("ordered-pair coverage failed")
    write_jsonl(MATERIAL, rows)
    protocol = protocol_payload(rows, tokens)
    atomic_json(PROTOCOL, protocol)
    atomic_json(ENVIRONMENT, {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "model_path": str(MODEL_PATH), "row_count": len(rows),
    })
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "relation_operation_behavior_contract_frozen",
        "row_count": len(rows),
        "prompt_count": len(rows) * len(PANELS),
        "partition_counts": PARTITION_COUNTS,
        "balance": balances,
        "token_checks": {key: value for key, value in tokens.items() if key not in ("event_token_indices",)},
        "protocol_digest": protocol["protocol_digest"],
        "authorization": "phase1280_qwen3_behavior_only",
    }
    atomic_json(SUMMARY, final)
    print(canonical_json(final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.force)
