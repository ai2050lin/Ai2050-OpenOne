#!/usr/bin/env python3
"""Freeze the Phase1028 role-by-depth causal leverage map."""

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
import phase1026_binding_replication_protocol as phase1026
import phase1027_binding_transport_protocol as phase1027


PHASE = 1028
PROTOCOL_REVISION = 1
MODELS = phase1024.MODELS
SPLITS = phase1024.SPLITS
ROLES = (
    "definition_nonce_end",
    "relation_end",
    "concept_end",
    "query_nonce_end",
    "pre_output",
)
CONFIRMATION_MODES = (
    "matched",
    "scrambled_concept",
    "matched_wrong_position",
)
SURFACE_COUNT = 8
TARGET_COUNT = 8
PATCH_DEPTHS = {
    "qwen3": (1, 5, 9, 13, 17, 21, 25, 29, 31),
    "glm4": (1, 4, 7, 10, 13, 16, 18),
    "deepseek7b": (1, 4, 7, 10, 13, 16, 19, 22, 25),
}
READOUT_DEPTH = {
    "qwen3": 35,
    "glm4": 19,
    "deepseek7b": 27,
}
WRONG_ROLE = {
    "definition_nonce_end": "relation_end",
    "relation_end": "definition_nonce_end",
    "concept_end": "relation_end",
    "query_nonce_end": "definition_nonce_end",
    "pre_output": "definition_nonce_end",
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1028_role_depth_causal_map"
)


CONCEPTS = {
    "discovery": (
        ("grape", "fruit"),
        ("bear", "animal"),
        ("motorcycle", "vehicle"),
        ("architect", "profession"),
        ("restaurant", "place"),
        ("table", "object"),
        ("black", "color"),
        ("foot", "body_part"),
    ),
    "confirmation": (
        ("lemon", "fruit"),
        ("zebra", "animal"),
        ("boat", "vehicle"),
        ("farmer", "profession"),
        ("park", "place"),
        ("cup", "object"),
        ("pink", "color"),
        ("ear", "body_part"),
    ),
}

NONCES = {
    "discovery": (
        "avel",
        "brom",
        "cestin",
        "dulp",
        "fenor",
        "gavil",
        "hask",
        "jumen",
    ),
    "confirmation": (
        "kelp",
        "lorin",
        "mav",
        "nusel",
        "orvik",
        "raben",
        "tef",
        "zulin",
    ),
}

TEMPLATES = {
    "discovery": {
        "template": (
            'Binding statement: code "{nonce}" represents the concept '
            '{target}. Retrieval statement: code "{nonce}" now requires '
            "a broad category:"
        ),
        "relation": "represents",
    },
    "confirmation": {
        "template": (
            'For this example, symbol "{nonce}" stands for {target}. '
            'When symbol "{nonce}" is queried, its general class is:'
        ),
        "relation": "stands for",
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


def span(
    prompt: str,
    value: str,
    *,
    occurrence: str,
) -> tuple[int, int, str]:
    if occurrence == "first":
        start = prompt.find(value)
    elif occurrence == "last":
        start = prompt.rfind(value)
    else:
        raise ValueError(occurrence)
    if start < 0:
        raise RuntimeError(f"missing fragment {value!r}")
    return start, start + len(value), value


def build_common_cases() -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        template = TEMPLATES[split]["template"]
        relation = TEMPLATES[split]["relation"]
        for surface_index, nonce in enumerate(NONCES[split]):
            for target_index, (target, family) in enumerate(CONCEPTS[split]):
                prompt = template.format(nonce=nonce, target=target)
                rows.append({
                    "schema_version": "phase1028_common_case.v1",
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
                        "definition_nonce_end": span(
                            prompt, nonce, occurrence="first"
                        ),
                        "relation_end": span(
                            prompt, relation, occurrence="first"
                        ),
                        "concept_end": span(
                            prompt, target, occurrence="first"
                        ),
                        "query_nonce_end": span(
                            prompt, nonce, occurrence="last"
                        ),
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
        surface_indices = (
            range(4) if split == "discovery" else range(SURFACE_COUNT)
        )
        for surface_index in surface_indices:
            for target_index in range(TARGET_COUNT):
                donor_indices = (
                    (
                        (target_index + 1) % TARGET_COUNT,
                        (target_index + 3) % TARGET_COUNT,
                    )
                    if split == "discovery"
                    else tuple(
                        value
                        for value in range(TARGET_COUNT)
                        if value != target_index
                    )
                )
                for donor_index in donor_indices:
                    if donor_index == target_index:
                        raise RuntimeError("self donor")
                    scrambled = (donor_index + 1) % TARGET_COUNT
                    while scrambled in {target_index, donor_index}:
                        scrambled = (scrambled + 1) % TARGET_COUNT
                    rows.append({
                        "schema_version": "phase1028_pair.v1",
                        "pair_index": len(rows),
                        "split_pair_index": sum(
                            prior["split"] == split for prior in rows
                        ),
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
        "schema_version": "phase1028_model_case.v1",
        "model": model_name,
        "record_id": f"{model_name}.{row['case_key']}",
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(positions[role][1])
            for role in ROLES
            if role != "pre_output"
        } | {
            "pre_output": len(input_ids) - 1,
        },
        "prompt_token_count": len(input_ids),
    })
    return result


def previous_words() -> tuple[set[str], set[str]]:
    concept_words = set()
    nonce_words = set()
    for module in (phase1024, phase1026, phase1027):
        for split in SPLITS:
            concept_words.update(
                value for value, _ in module.CONCEPTS[split]
            )
            nonce_words.update(module.NONCES[split])
    return concept_words, nonce_words


def main() -> None:
    cases = build_common_cases()
    pairs = build_pairs(cases)
    concepts_prior, nonces_prior = previous_words()
    pair_counts = Counter(row["split"] for row in pairs)
    common_checks = {
        "clean_case_count": len(cases) == 128,
        "case_keys_unique": (
            len({row["case_key"] for row in cases}) == len(cases)
        ),
        "pair_count": len(pairs) == 512,
        "pair_split_counts": pair_counts == {
            "discovery": 64,
            "confirmation": 448,
        },
        "pair_keys_unique": (
            len({row["pair_key"] for row in pairs}) == len(pairs)
        ),
        "all_nonself_pairs": all(
            row["target_index"] != row["donor_index"]
            for row in pairs
        ),
        "scrambled_distinct": all(
            row["scrambled_donor_index"]
            not in {row["target_index"], row["donor_index"]}
            for row in pairs
        ),
        "new_concepts": not any(
            word in concepts_prior
            for split in SPLITS
            for word, _ in CONCEPTS[split]
        ),
        "new_nonces": not any(
            word in nonces_prior
            for split in SPLITS
            for word in NONCES[split]
        ),
    }
    common_audit = {
        "schema_version": "phase1028_common_audit.v1",
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
        "roles_in_causal_order": ROLES,
        "confirmation_modes": CONFIRMATION_MODES,
        "surface_count": SURFACE_COUNT,
        "target_count": TARGET_COUNT,
        "patch_depths": PATCH_DEPTHS,
        "readout_depth": READOUT_DEPTH,
        "wrong_role_control": WRONG_ROLE,
        "discovery_selection": (
            "freeze the best donor_top1/margin candidate per role plus "
            "the global top three; confirmation cannot add candidates"
        ),
        "confirmation_gate": {
            "clean_target_top1_minimum": 0.50,
            "matched_donor_top1_minimum": 0.25,
            "matched_margin_shift_minimum": 0.02,
            "matched_minus_scrambled_donor_top1_minimum": 0.10,
            "matched_minus_wrong_position_donor_top1_minimum": 0.10,
            "all_arrays_must_be_finite": True,
        },
        "interpretation_rule": (
            "Phase1027 tested only query_nonce_end; multi-position "
            "necessity is not inferred until all upstream singleton roles "
            "fail independent confirmation"
        ),
        "claim_limit": (
            "role-depth causal leverage map only; no full token mechanism, "
            "minimal alliance, correct-output, brain homology, efficiency, "
            "or optimality claim"
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
                    0 <= position < len(row["input_ids"])
                    for position in row["role_positions"].values()
                )
                for row in rows
            ),
            "strict_causal_role_order": all(
                [
                    row["role_positions"][role] for role in ROLES
                ] == sorted(
                    row["role_positions"][role] for role in ROLES
                )
                and len({
                    row["role_positions"][role] for role in ROLES
                }) == len(ROLES)
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
            raise RuntimeError(json.dumps(audit, ensure_ascii=False))
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
        "pair_counts": dict(pair_counts),
        "patch_depths": PATCH_DEPTHS,
        "readout_depth": READOUT_DEPTH,
        "audit": common_audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
