#!/usr/bin/env python3
"""Freeze Phase1027 local binding-state transport and controls."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import render_chat, tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1024_lexical_semantic_protocol as phase1024


PHASE = 1027
PROTOCOL_REVISION = 1
MODELS = phase1024.MODELS
SPLITS = phase1024.SPLITS
ROLES = ("target_end", "focus_end", "pre_output")
INTERVENTIONS = (
    "matched_focus",
    "scrambled_focus",
    "matched_bos_delta",
)
SURFACE_COUNT = 8
TARGET_COUNT = 8
PATCH_DEPTH = {
    "qwen3": 13,
    "glm4": 13,
    "deepseek7b": 10,
}
READOUT_DEPTH = {
    "qwen3": 31,
    "glm4": 19,
    "deepseek7b": 27,
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1027_binding_transport"
)


CONCEPTS = {
    "discovery": (
        ("plum", "fruit"),
        ("rabbit", "animal"),
        ("truck", "vehicle"),
        ("nurse", "profession"),
        ("theater", "place"),
        ("spoon", "object"),
        ("purple", "color"),
        ("ankle", "body_part"),
    ),
    "confirmation": (
        ("cherry", "fruit"),
        ("horse", "animal"),
        ("ship", "vehicle"),
        ("chef", "profession"),
        ("airport", "place"),
        ("clock", "object"),
        ("white", "color"),
        ("wrist", "body_part"),
    ),
}

NONCES = {
    "discovery": (
        "brin",
        "calo",
        "drux",
        "falin",
        "hevo",
        "kest",
        "mora",
        "sulen",
    ),
    "confirmation": (
        "bex",
        "curna",
        "daro",
        "gim",
        "lavek",
        "nuro",
        "pesh",
        "votin",
    ),
}

TEMPLATES = {
    "discovery": (
        'Temporary dictionary: "{nonce}" is another name for {target}. '
        'Later query: what broad meaning is carried by "{nonce}"?'
    ),
    "confirmation": (
        'For this sentence alone, use code "{nonce}" to represent {target}. '
        'Determine the semantic class of "{nonce}":'
    ),
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def span(prompt: str, value: str, *, last: bool) -> tuple[int, int, str]:
    start = prompt.rfind(value) if last else prompt.find(value)
    if start < 0:
        raise RuntimeError(f"missing fragment {value!r}")
    return start, start + len(value), value


def build_common_cases() -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        for surface_index, nonce in enumerate(NONCES[split]):
            for target_index, (target, family) in enumerate(CONCEPTS[split]):
                prompt = TEMPLATES[split].format(
                    nonce=nonce,
                    target=target,
                )
                rows.append({
                    "schema_version": "phase1027_common_case.v1",
                    "phase": PHASE,
                    "case_index": len(rows),
                    "case_key": (
                        f"{split}.s{surface_index}.t{target_index}"
                    ),
                    "split": split,
                    "surface_index": surface_index,
                    "surface": nonce,
                    "target_index": target_index,
                    "target": target,
                    "target_family": family,
                    "prompt": prompt,
                    "role_fragments": {
                        "target_end": span(prompt, target, last=False),
                        "focus_end": span(prompt, nonce, last=True),
                    },
                })
    return rows


def build_pairs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["split"],
            int(row["surface_index"]),
            int(row["target_index"]),
        ): int(row["case_index"])
        for row in cases
    }
    rows = []
    for split in SPLITS:
        for surface_index in range(SURFACE_COUNT):
            for target_index in range(TARGET_COUNT):
                for donor_index in range(TARGET_COUNT):
                    if donor_index == target_index:
                        continue
                    scrambled = (donor_index + 1) % TARGET_COUNT
                    while scrambled in {target_index, donor_index}:
                        scrambled = (scrambled + 1) % TARGET_COUNT
                    rows.append({
                        "schema_version": "phase1027_pair.v1",
                        "pair_index": len(rows),
                        "pair_key": (
                            f"{split}.s{surface_index}.t{target_index}."
                            f"d{donor_index}"
                        ),
                        "split": split,
                        "surface_index": surface_index,
                        "target_index": target_index,
                        "donor_index": donor_index,
                        "scrambled_donor_index": scrambled,
                        "target_case_index": by_key[
                            (split, surface_index, target_index)
                        ],
                        "donor_case_index": by_key[
                            (split, surface_index, donor_index)
                        ],
                        "scrambled_case_index": by_key[
                            (split, surface_index, scrambled)
                        ],
                    })
    return rows


def model_case(
    tokenizer,
    model_name: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    rendered = render_chat(tokenizer, model_name, row["prompt"])
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    positions = offset_token_spans(
        tokenizer,
        rendered,
        row["prompt"],
        row["role_fragments"],
    )
    result = dict(row)
    result.pop("role_fragments", None)
    result.update({
        "schema_version": "phase1027_model_case.v1",
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            "target_end": int(positions["target_end"][1]),
            "focus_end": int(positions["focus_end"][1]),
            "pre_output": len(input_ids) - 1,
        },
        "prompt_token_count": len(input_ids),
    })
    return result


def main() -> None:
    cases = build_common_cases()
    pairs = build_pairs(cases)
    concept_words = {
        word
        for split in phase1024.SPLITS
        for word, _ in phase1024.CONCEPTS[split]
    }
    nonce_words = {
        value
        for split in phase1024.SPLITS
        for value in phase1024.NONCES[split]
    }
    common_checks = {
        "clean_case_count": len(cases) == 128,
        "pair_count": len(pairs) == 896,
        "case_keys_unique": (
            len({row["case_key"] for row in cases}) == len(cases)
        ),
        "pair_keys_unique": (
            len({row["pair_key"] for row in pairs}) == len(pairs)
        ),
        "all_ordered_nonself_pairs": all(
            row["target_index"] != row["donor_index"]
            for row in pairs
        ),
        "scrambled_is_distinct": all(
            row["scrambled_donor_index"]
            not in {row["target_index"], row["donor_index"]}
            for row in pairs
        ),
        "new_concepts": not any(
            word in concept_words
            for split in SPLITS
            for word, _ in CONCEPTS[split]
        ),
        "new_nonces": not any(
            nonce in nonce_words
            for split in SPLITS
            for nonce in NONCES[split]
        ),
        "balanced_pairs": set(Counter(
            row["split"] for row in pairs
        ).values()) == {448},
    }
    common_audit = {
        "schema_version": "phase1027_common_audit.v1",
        "checks": common_checks,
        "all_checks_passed": all(common_checks.values()),
    }
    if not common_audit["all_checks_passed"]:
        raise RuntimeError(json.dumps(common_audit, ensure_ascii=False))

    prereg = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "precision": "fp16",
        "quantization": "none",
        "models": MODELS,
        "roles": ROLES,
        "interventions": INTERVENTIONS,
        "surface_count": SURFACE_COUNT,
        "target_count": TARGET_COUNT,
        "patch_depth_frozen_from_phase1026": PATCH_DEPTH,
        "readout_depth_frozen_from_prior_finite_atlas": READOUT_DEPTH,
        "primary_measure": (
            "cross-surface downstream donor-vs-target cosine margin shift "
            "at pre_output after a focus-position state transplant"
        ),
        "controls": {
            "scrambled_focus": (
                "transplant a third concept at the same position"
            ),
            "matched_bos_delta": (
                "apply the intended donor-minus-target vector at position 0"
            ),
        },
        "replication_gate": {
            "both_splits_required": True,
            "clean_target_top1_minimum": 0.50,
            "matched_donor_top1_minimum": 0.25,
            "matched_margin_shift_minimum": 0.02,
            "matched_minus_scrambled_donor_top1_minimum": 0.10,
            "matched_minus_bos_donor_top1_minimum": 0.10,
            "captured_tensors_must_be_finite": True,
        },
        "claim_limit": (
            "local internal-state transport only; no correct-output, "
            "sufficiency, complete token mechanism, brain homology, or "
            "optimality claim"
        ),
        "clean_case_digest": digest(cases),
        "pair_digest": digest(pairs),
    }
    prereg["protocol_digest"] = digest(prereg)
    protocol_dir = OUT_ROOT / "protocol"
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.common.json", common_audit)
    write_jsonl(protocol_dir / "common_cases.jsonl", cases)
    write_jsonl(protocol_dir / "pairs.jsonl", pairs)

    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = [model_case(tokenizer, model_name, row) for row in cases]
        checks = {
            "case_count": len(rows) == len(cases),
            "case_indices_dense": (
                [row["case_index"] for row in rows]
                == list(range(len(rows)))
            ),
            "roles_present": all(
                set(row["role_positions"]) == set(ROLES) for row in rows
            ),
            "positions_in_range": all(
                all(
                    0 <= value < len(row["input_ids"])
                    for value in row["role_positions"].values()
                )
                for row in rows
            ),
            "focus_after_target": all(
                row["role_positions"]["focus_end"]
                > row["role_positions"]["target_end"]
                for row in rows
            ),
        }
        audit = {
            "model": model_name,
            "prompt_tokens": {
                "minimum": min(row["prompt_token_count"] for row in rows),
                "maximum": max(row["prompt_token_count"] for row in rows),
            },
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
        if not audit["all_checks_passed"]:
            raise RuntimeError(json.dumps(audit))
        write_jsonl(protocol_dir / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_dir / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        del tokenizer
    write_json(
        protocol_dir / "audit.models.json",
        {
            "models": model_audits,
            "all_checks_passed": all(
                row["all_checks_passed"] for row in model_audits.values()
            ),
        },
    )
    print(json.dumps({
        "protocol_digest": prereg["protocol_digest"],
        "clean_case_count": len(cases),
        "pair_count": len(pairs),
        "patch_depth": PATCH_DEPTH,
        "readout_depth": READOUT_DEPTH,
        "audit": common_audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
