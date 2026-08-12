#!/usr/bin/env python3
"""Freeze Phase1003 output-surface and EOS stress cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import (
    render_chat,
    tokenizer_for,
)
from phase1003_crossparadigm_protocol import (
    ANCHOR_ROLES,
    ANSWER_PREFIX,
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    digest,
    selected_directional_rows,
    write_json,
    write_jsonl,
)


ROLLOUT_ROOT = OUT_ROOT / "rollout_surface_stress"
SURFACES = (
    "bare",
    "short_sentence",
    "explanation",
    "two_sentence",
    "correction",
)


def instruction(surface: str) -> str:
    if surface == "bare":
        return (
            "Reply with only the lowercase color word. Stop immediately "
            "after that word."
        )
    if surface == "short_sentence":
        return (
            "Reply exactly in this form: The answer is [value]. Replace "
            "[value] with the lowercase color. Stop after the period."
        )
    if surface == "explanation":
        return (
            "Reply exactly in this form: [name] has color [value] because "
            "the registry assigns it. Replace [name] with the queried name "
            "and [value] with the lowercase color. Stop after the period."
        )
    if surface == "two_sentence":
        return (
            "Reply exactly in this form: The answer is [value]. The "
            "registry confirms this. Replace [value] with the lowercase "
            "color. Stop after the second period."
        )
    if surface == "correction":
        return (
            "Reply exactly in this form: The answer is not [wrong]; it is "
            "[value]. Replace [wrong] with the other recorded color and "
            "[value] with the queried lowercase color. Stop after the "
            "period."
        )
    raise ValueError(surface)


def answer_body(
    surface: str,
    query_name: str,
    value: str,
    foil: str,
) -> str:
    if surface == "bare":
        return value
    if surface == "short_sentence":
        return f"The answer is {value}."
    if surface == "explanation":
        return (
            f"{query_name} has color {value} because the registry "
            "assigns it."
        )
    if surface == "two_sentence":
        return (
            f"The answer is {value}. The registry confirms this."
        )
    if surface == "correction":
        return f"The answer is not {foil}; it is {value}."
    raise ValueError(surface)


def answer_text(
    model_name: str,
    surface: str,
    query_name: str,
    value: str,
    foil: str,
) -> str:
    return (
        ANSWER_PREFIX[model_name]
        + answer_body(surface, query_name, value, foil)
    )


def prompt_body(raw_prompt: str) -> str:
    marker = "\nAnswer exactly in this form:"
    if marker not in raw_prompt:
        raise RuntimeError("base prompt instruction marker missing")
    return raw_prompt.split(marker, 1)[0]


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    common_prefix_checks = []
    semantic_steps: dict[str, set[int]] = {
        surface: set() for surface in SURFACES
    }
    answer_widths: dict[str, set[int]] = {
        surface: set() for surface in SURFACES
    }
    for split in ("discovery", "confirmation"):
        rows = selected_directional_rows(
            model_name, "color", split
        )
        base_cases = [row["target"] for row in rows]
        if len({case["record_id"] for case in base_cases}) != 64:
            raise RuntimeError(
                f"{model_name}/{split}: target cases are not unique"
            )
        for base in base_cases:
            body = prompt_body(base["raw_prompt"])
            for surface in SURFACES:
                raw_prompt = f"{body}\n{instruction(surface)}"
                rendered = render_chat(
                    tokenizer, model_name, raw_prompt
                )
                input_ids = [
                    int(value)
                    for value in tokenizer.encode(
                        rendered, add_special_tokens=False
                    )
                ]
                expected_text = answer_text(
                    model_name,
                    surface,
                    base["query_entity"],
                    base["gold"],
                    base["foil"],
                )
                expected_ids = [
                    int(value)
                    for value in tokenizer.encode(
                        expected_text, add_special_tokens=False
                    )
                ]
                candidate_answers = {
                    label: [
                        int(value)
                        for value in tokenizer.encode(
                            answer_text(
                                model_name,
                                surface,
                                base["query_entity"],
                                label,
                                base["foil"],
                            ),
                            add_special_tokens=False,
                        )
                    ]
                    for label in DOMAINS["color"]
                }
                widths = {
                    len(values)
                    for values in candidate_answers.values()
                }
                if len(widths) != 1:
                    raise RuntimeError(
                        f"{model_name}/{surface}: candidate width drift"
                    )
                varying_steps = [
                    step
                    for step in range(next(iter(widths)))
                    if len({
                        values[step]
                        for values in candidate_answers.values()
                    }) > 1
                ]
                if len(varying_steps) != 1:
                    raise RuntimeError(
                        f"{model_name}/{surface}: expected one value "
                        f"step, got {varying_steps}"
                    )
                semantic_step = varying_steps[0]
                semantic_steps[surface].add(semantic_step)
                answer_widths[surface].add(len(expected_ids))
                combined = [
                    int(value)
                    for value in tokenizer.encode(
                        rendered + expected_text,
                        add_special_tokens=False,
                    )
                ]
                if combined != input_ids + expected_ids:
                    raise RuntimeError(
                        f"{model_name}/{surface}: answer boundary drift"
                    )
                max_role_position = max(
                    int(base["role_positions"][role])
                    for role in ANCHOR_ROLES
                )
                common_prefix = 0
                for left, right in zip(
                    base["input_ids"], input_ids
                ):
                    if left != right:
                        break
                    common_prefix += 1
                common_prefix_checks.append(
                    common_prefix > max_role_position
                )
                role_positions = {
                    role: int(base["role_positions"][role])
                    for role in ANCHOR_ROLES
                }
                role_positions["answer_boundary"] = (
                    len(input_ids) - 1
                )
                cases.append({
                    "schema_version": (
                        "phase1003_rollout_surface_case.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "record_id": (
                        f"{base['record_id']}.surface.{surface}"
                    ),
                    "base_record_id": base["record_id"],
                    "domain": "color",
                    "split": split,
                    "surface": surface,
                    "world_id": base["world_id"],
                    "template": base["template"],
                    "raw_prompt": raw_prompt,
                    "rendered_prompt": rendered,
                    "input_ids": input_ids,
                    "input_token_count": len(input_ids),
                    "query_entity": base["query_entity"],
                    "gold": base["gold"],
                    "foil": base["foil"],
                    "candidate_labels": list(DOMAINS["color"]),
                    "candidate_token_ids": {
                        label: values[semantic_step]
                        for label, values in candidate_answers.items()
                    },
                    "semantic_step": semantic_step,
                    "answer_text": expected_text,
                    "answer_token_ids": expected_ids,
                    "answer_token_count": len(expected_ids),
                    "anchor_roles": list(ANCHOR_ROLES),
                    "role_positions": role_positions,
                })
    root = ROLLOUT_ROOT / "protocol" / model_name
    write_jsonl(root / "cases.jsonl", cases)
    audit = {
        "schema_version": (
            "phase1003_rollout_surface_protocol_audit.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "surface_count": len(SURFACES),
        "case_count_per_surface_split": {
            f"{surface}:{split}": sum(
                case["surface"] == surface
                and case["split"] == split
                for case in cases
            )
            for surface in SURFACES
            for split in ("discovery", "confirmation")
        },
        "surfaces": list(SURFACES),
        "semantic_steps": {
            surface: sorted(steps)
            for surface, steps in semantic_steps.items()
        },
        "answer_widths": {
            surface: sorted(widths)
            for surface, widths in answer_widths.items()
        },
        "all_anchor_positions_in_unchanged_prompt_prefix": all(
            common_prefix_checks
        ),
        "case_digest": digest(cases),
    }
    write_json(root / "protocol_audit.json", audit)
    return audit


def main() -> None:
    audits = {
        model_name: build_model(model_name)
        for model_name in MODELS
    }
    prereg = {
        "schema_version": (
            "phase1003_rollout_surface_preregistration.v1"
        ),
        "phase": PHASE,
        "surfaces_in_fixed_order": list(SURFACES),
        "protocol_audits": audits,
        "behavior_exact_gate": 0.90,
        "behavior_semantic_gate": 0.95,
        "full_anchor_semantic_gate": 0.75,
        "full_anchor_exact_gate": 0.75,
        "noop_sequence_gate": 0.99,
        "eos_observed_gate": 0.95,
        "cross_model_minimum": 2,
        "internal_results_used_to_select_surfaces": False,
        "surface_selection_uses_behavior_results": False,
        "causal_conditions": [
            "target clean",
            "target-state no-op",
            "full five-anchor source",
        ],
        "claim_boundary": (
            "These are fixed controlled output surfaces, not unrestricted "
            "open-language generation. EOS is measured by whether greedy "
            "generation terminates immediately after the expected surface."
        ),
    }
    prereg["preregistration_digest"] = digest(prereg)
    write_json(ROLLOUT_ROOT / "preregistered_protocol.json", prereg)
    print(json.dumps(prereg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
