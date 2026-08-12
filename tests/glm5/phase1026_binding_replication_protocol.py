#!/usr/bin/env python3
"""Freeze the enlarged independent Phase1026 binding replication."""

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


PHASE = 1026
PROTOCOL_REVISION = 1
MODELS = phase1024.MODELS
SPLITS = phase1024.SPLITS
ROLES = ("target_end", "focus_end", "pre_output")
CONDITIONS = (
    "target_bound",
    "distractor_bound",
    "cooccur_unbound",
    "reversed_target",
)
SURFACE_COUNT = 8
TARGET_COUNT = 8
PRIMARY_DEPTH = {
    "qwen3": 13,
    "glm4": 13,
    "deepseek7b": 10,
}
AUXILIARY_DEPTHS = {
    "qwen3": (5,),
    "glm4": (),
    "deepseek7b": (4,),
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1026_binding_replication"
)


CONCEPTS = {
    "discovery": (
        ("mango", "fruit"),
        ("eagle", "animal"),
        ("train", "vehicle"),
        ("engineer", "profession"),
        ("museum", "place"),
        ("lamp", "object"),
        ("green", "color"),
        ("knee", "body_part"),
    ),
    "confirmation": (
        ("peach", "fruit"),
        ("dolphin", "animal"),
        ("bus", "vehicle"),
        ("lawyer", "profession"),
        ("school", "place"),
        ("bottle", "object"),
        ("yellow", "color"),
        ("elbow", "body_part"),
    ),
}

NONCES = {
    "discovery": (
        "lum",
        "prax",
        "siv",
        "gorn",
        "mek",
        "vash",
        "tul",
        "roven",
    ),
    "confirmation": (
        "vep",
        "murn",
        "tivo",
        "nax",
        "selk",
        "jora",
        "pim",
        "dovel",
    ),
}

TEMPLATES = {
    "discovery": {
        "target_bound": (
            'Glossary rule: label "{nonce}" denotes {target}. The word '
            "{distractor} is only a decoy. Classify the repeated label "
            '"{nonce}" by meaning:'
        ),
        "distractor_bound": (
            'Glossary rule: label "{nonce}" denotes {distractor}. The word '
            "{target} is only a decoy. Classify the repeated label "
            '"{nonce}" by meaning:'
        ),
        "cooccur_unbound": (
            'Glossary rule: label "{other_nonce}" denotes {distractor}. '
            'The items {target} and "{nonce}" are merely listed nearby. '
            'Classify the repeated label "{nonce}" by meaning:'
        ),
        "reversed_target": (
            'Glossary rule: {target} denotes label "{nonce}". The word '
            "{distractor} is only a decoy. Classify the repeated label "
            '"{nonce}" by meaning:'
        ),
    },
    "confirmation": {
        "target_bound": (
            'In this one-use codebook, "{nonce}" stands for {target}; '
            "{distractor} has no role. What semantic class should the code "
            '"{nonce}" inherit?'
        ),
        "distractor_bound": (
            'In this one-use codebook, "{nonce}" stands for {distractor}; '
            "{target} has no role. What semantic class should the code "
            '"{nonce}" inherit?'
        ),
        "cooccur_unbound": (
            'In this one-use codebook, "{other_nonce}" stands for '
            '{distractor}; {target} and "{nonce}" only co-occur. What '
            'semantic class should the code "{nonce}" inherit?'
        ),
        "reversed_target": (
            'In this one-use codebook, {target} stands for "{nonce}"; '
            "{distractor} has no role. What semantic class should the code "
            '"{nonce}" inherit?'
        ),
    },
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
        concepts = CONCEPTS[split]
        nonces = NONCES[split]
        for surface_index, nonce in enumerate(nonces):
            other_nonce = nonces[(surface_index + 3) % len(nonces)]
            for target_index, (target, family) in enumerate(concepts):
                distractor_index = (
                    target_index + surface_index + 1
                ) % len(concepts)
                distractor, distractor_family = concepts[distractor_index]
                for condition in CONDITIONS:
                    prompt = TEMPLATES[split][condition].format(
                        nonce=nonce,
                        other_nonce=other_nonce,
                        target=target,
                        distractor=distractor,
                    )
                    rows.append({
                        "schema_version": "phase1026_common_case.v1",
                        "phase": PHASE,
                        "case_key": (
                            f"{split}.{condition}.s{surface_index}."
                            f"t{target_index}"
                        ),
                        "split": split,
                        "condition": condition,
                        "surface_index": surface_index,
                        "surface": nonce,
                        "other_surface": other_nonce,
                        "target_index": target_index,
                        "target": target,
                        "target_family": family,
                        "distractor_index": distractor_index,
                        "distractor": distractor,
                        "distractor_family": distractor_family,
                        "prompt": prompt,
                        "role_fragments": {
                            "target_end": span(prompt, target, last=False),
                            "focus_end": span(prompt, nonce, last=True),
                        },
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
        "schema_version": "phase1026_model_case.v1",
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


def common_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        (row["split"], row["condition"]) for row in rows
    )
    expected_total = (
        len(SPLITS) * len(CONDITIONS) * SURFACE_COUNT * TARGET_COUNT
    )
    checks = {
        "case_count": len(rows) == expected_total,
        "case_keys_unique": len({row["case_key"] for row in rows}) == len(rows),
        "cells_balanced": set(counts.values()) == {
            SURFACE_COUNT * TARGET_COUNT
        },
        "new_concepts_vs_phase1025": not any(
            word in {
                value
                for split in phase1024.SPLITS
                for value, _ in phase1024.CONCEPTS[split]
            }
            for split in SPLITS
            for word, _ in CONCEPTS[split]
        ),
        "new_nonces_vs_phase1025": not any(
            nonce in {
                value
                for split in phase1024.SPLITS
                for value in phase1024.NONCES[split]
            }
            for split in SPLITS
            for nonce in NONCES[split]
        ),
        "distractor_varies_by_surface": all(
            len({
                row["distractor_index"]
                for row in rows
                if row["split"] == split
                and row["condition"] == condition
                and row["target_index"] == target_index
            }) == SURFACE_COUNT
            for split in SPLITS
            for condition in CONDITIONS
            for target_index in range(TARGET_COUNT)
        ),
    }
    return {
        "schema_version": "phase1026_common_audit.v1",
        "counts": {"|".join(key): value for key, value in counts.items()},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    common = build_common_cases()
    audit = common_audit(common)
    if not audit["all_checks_passed"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False))
    prereg = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "precision": "fp16",
        "quantization": "none",
        "models": MODELS,
        "conditions": CONDITIONS,
        "roles": ROLES,
        "surface_count": SURFACE_COUNT,
        "target_count": TARGET_COUNT,
        "primary_depth_frozen_from_phase1025": PRIMARY_DEPTH,
        "auxiliary_depths_frozen_from_phase1025": AUXILIARY_DEPTHS,
        "selection_policy": (
            "one primary joint depth per model is frozen from Phase1025 "
            "before any Phase1026 state is observed; no Phase1026 layer "
            "selection is permitted"
        ),
        "replication_gate": {
            "both_splits_required": True,
            "alignment_bound_minimum": 0.50,
            "alignment_control_margin_minimum": 0.20,
            "retrieval_bound_minimum": 0.30,
            "retrieval_control_margin_minimum": 0.15,
            "captured_tensors_must_be_finite": True,
        },
        "claim_limit": (
            "independent enlarged observational replication only; no "
            "behavioral ability, causal transport, brain homology, or "
            "optimality claim"
        ),
        "common_case_digest": digest(common),
    }
    prereg["protocol_digest"] = digest(prereg)
    protocol_dir = OUT_ROOT / "protocol"
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.common.json", audit)
    write_jsonl(protocol_dir / "common_cases.jsonl", common)

    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = [model_case(tokenizer, model_name, row) for row in common]
        checks = {
            "case_count": len(rows) == len(common),
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
            "record_ids_unique": (
                len({row["record_id"] for row in rows}) == len(rows)
            ),
        }
        model_audit = {
            "model": model_name,
            "prompt_tokens": {
                "minimum": min(row["prompt_token_count"] for row in rows),
                "maximum": max(row["prompt_token_count"] for row in rows),
            },
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
        if not model_audit["all_checks_passed"]:
            raise RuntimeError(json.dumps(model_audit))
        write_jsonl(protocol_dir / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_dir / f"audit.{model_name}.json", model_audit)
        model_audits[model_name] = model_audit
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
        "case_count_per_model": len(common),
        "audit": audit,
        "primary_depth": PRIMARY_DEPTH,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
