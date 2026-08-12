#!/usr/bin/env python3
"""Freeze the Phase1017 same-token contextual semantic-branch protocol."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1009_crossfamily_response_protocol import (
    PromptBuilder,
    digest,
    role_token_positions,
    write_json,
    write_jsonl,
)


PHASE = 1017
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
PROMPT_MODES = ("raw", "native_chat")
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
OUTPUT_MODES = ("semantic",)
WORLDS = tuple(range(4))
NEUTRAL_TARGETS = ("item", "term")
FACTORIAL_STATES = (
    "a0_l0",
    "a1_l0",
    "a0_l1",
    "a1_l1",
    "n0_l0",
    "n1_l0",
    "n0_l1",
    "n1_l1",
)
STATES = FACTORIAL_STATES + ("identity",)
CAPTURE_ROLES = (
    "cue",
    "target_word",
    "option0",
    "option1",
    "query_operator",
    "answer_boundary",
)
WORDS = {
    "bank": {
        "labels": ("river", "finance"),
        "discovery": (("water", "shore"), ("money", "loan")),
        "confirmation": (("stream", "coast"), ("credit", "lender")),
    },
    "bat": {
        "labels": ("animal", "sports"),
        "discovery": (("cave", "wing"), ("baseball", "pitcher")),
        "confirmation": (("night", "furry"), ("stadium", "inning")),
    },
    "crane": {
        "labels": ("bird", "machine"),
        "discovery": (("feather", "marsh"), ("building", "lifting")),
        "confirmation": (("nest", "wing"), ("steel", "lift")),
    },
    "seal": {
        "labels": ("animal", "stamp"),
        "discovery": (("ocean", "swim"), ("wax", "document")),
        "confirmation": (("marine", "fish"), ("letter", "signature")),
    },
    "spring": {
        "labels": ("season", "coil"),
        "discovery": (("flower", "April"), ("metal", "tension")),
        "confirmation": (("warm", "bloom"), ("compress", "spiral")),
    },
    "club": {
        "labels": ("group", "stick"),
        "discovery": (("member", "meeting"), ("swing", "hit")),
        "confirmation": (("society", "join"), ("golf", "wooden")),
    },
    "match": {
        "labels": ("contest", "flame"),
        "discovery": (("game", "score"), ("fire", "candle")),
        "confirmation": (("tournament", "opponent"), ("ignite", "spark")),
    },
    "port": {
        "labels": ("harbor", "computer"),
        "discovery": (("ship", "dock"), ("network", "socket")),
        "confirmation": (("vessel", "coast"), ("server", "cable")),
    },
    "file": {
        "labels": ("document", "tool"),
        "discovery": (("folder", "paper"), ("metal", "teeth")),
        "confirmation": (("record", "office"), ("workshop", "smooth")),
    },
    "mouse": {
        "labels": ("animal", "device"),
        "discovery": (("cheese", "tail"), ("cursor", "click")),
        "confirmation": (("small", "pet"), ("computer", "button")),
    },
    "jam": {
        "labels": ("food", "traffic"),
        "discovery": (("bread", "berry"), ("cars", "road")),
        "confirmation": (("toast", "sweet"), ("highway", "congestion")),
    },
    "pitch": {
        "labels": ("throw", "sound"),
        "discovery": (("baseball", "ball"), ("music", "tone")),
        "confirmation": (("pitcher", "field"), ("frequency", "voice")),
    },
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1017_contextual_semantic_niche_atlas"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def state_factors(state: str) -> tuple[str, int, int]:
    if state == "identity":
        return "ambiguous", 0, 0
    target_kind = "ambiguous" if state[0] == "a" else "neutral"
    return target_kind, int(state[1]), int(state[-1])


def boundary_token_id(tokenizer, rendered: str, label: str) -> int:
    base = tokenizer.encode(rendered, add_special_tokens=False)
    extended = tokenizer.encode(
        rendered + " " + label,
        add_special_tokens=False,
    )
    if extended[:len(base)] != base or len(extended) != len(base) + 1:
        raise RuntimeError(
            f"answer {label!r} is not one token at answer boundary"
        )
    return int(extended[-1])


def render_case(
    *,
    word: str,
    split: str,
    template: int,
    output_mode: str,
    world: int,
    state: str,
) -> tuple[
    str,
    dict[str, tuple[int, int, str]],
    str,
    str,
    dict[str, Any],
]:
    target_kind, branch, lexical = state_factors(state)
    spec = WORDS[word]
    cue = spec[split][branch][lexical]
    labels = list(spec["labels"])
    if world % 2:
        labels.reverse()
    neutral = NEUTRAL_TARGETS[(world // 2) % len(NEUTRAL_TARGETS)]
    target = word if target_kind == "ambiguous" else neutral
    branch_label = spec["labels"][branch]

    if output_mode != "semantic":
        raise RuntimeError(f"unsupported output mode {output_mode}")
    candidates = labels
    gold = branch_label
    foil = candidates[1 - candidates.index(gold)]

    builder = PromptBuilder()
    if template == 0:
        builder.add("Clue:")
        builder.mark("cue", cue)
        builder.add(". Ambiguous word:")
        builder.mark("target_word", target)
        builder.add(". Possible meanings:")
        builder.mark("option0", labels[0])
        builder.add(" or")
        builder.mark("option1", labels[1])
        builder.add(".")
        builder.mark("query_operator", "Reply")
        builder.add(" with exactly one listed meaning, not the word:")
    elif template == 1:
        builder.add("Hint:")
        builder.mark("cue", cue)
        builder.add(". Term:")
        builder.mark("target_word", target)
        builder.add(". Available senses:")
        builder.mark("option0", labels[0])
        builder.add(" versus")
        builder.mark("option1", labels[1])
        builder.add(".")
        builder.mark("query_operator", "Select")
        builder.add(" exactly one sense and output only that sense:")
    elif template == 2:
        builder.add("Signal:")
        builder.mark("cue", cue)
        builder.add(". Expression:")
        builder.mark("target_word", target)
        builder.add(". Candidate meanings:")
        builder.mark("option0", labels[0])
        builder.add(" or")
        builder.mark("option1", labels[1])
        builder.add(".")
        builder.mark("query_operator", "Pick")
        builder.add(" one candidate and write only it:")
    else:
        builder.add("Context clue:")
        builder.mark("cue", cue)
        builder.add(". Polysemous item:")
        builder.mark("target_word", target)
        builder.add(". Meaning choices:")
        builder.mark("option0", labels[0])
        builder.add(" and")
        builder.mark("option1", labels[1])
        builder.add(".")
        builder.mark("query_operator", "Decide")
        builder.add(" and return only the chosen meaning:")

    raw_prompt, spans = builder.finish()
    metadata = {
        "target_kind": target_kind,
        "branch": int(branch),
        "lexical": int(lexical),
        "cue": cue,
        "target": target,
        "ambiguous_word": word,
        "sense_labels": list(spec["labels"]),
        "ordered_sense_labels": labels,
        "neutral_target": neutral,
        "candidate_order_reversed": bool(world % 2),
        "output_mode": output_mode,
    }
    return raw_prompt, spans, gold, foil, metadata


def build_case(
    *,
    tokenizer,
    model_name: str,
    prompt_mode: str,
    word: str,
    split: str,
    template: int,
    output_mode: str,
    world: int,
    unit_id: str,
    state: str,
) -> dict[str, Any]:
    raw_prompt, spans, gold, foil, metadata = render_case(
        word=word,
        split=split,
        template=template,
        output_mode=output_mode,
        world=world,
        state=state,
    )
    rendered = (
        raw_prompt
        if prompt_mode == "raw"
        else render_chat(tokenizer, model_name, raw_prompt)
    )
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    positions = role_token_positions(
        tokenizer,
        rendered,
        raw_prompt,
        spans,
    )
    positions["answer_boundary"] = len(input_ids) - 1
    if set(positions) != set(CAPTURE_ROLES):
        raise RuntimeError(f"role drift: {sorted(positions)}")
    candidate_ids = {
        label: boundary_token_id(tokenizer, rendered, label)
        for label in (gold, foil)
    }
    if candidate_ids[gold] == candidate_ids[foil]:
        raise RuntimeError("candidate boundary token collision")
    return {
        "schema_version": "phase1017_semantic_niche_case.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "word": word,
        "split": split,
        "template": int(template),
        "output_mode": output_mode,
        "world": int(world),
        "unit_id": unit_id,
        "record_id": f"{unit_id}.{state}",
        "state": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_positions": {
            role: int(positions[role]) for role in CAPTURE_ROLES
        },
        "gold": gold,
        "foil": foil,
        "candidate_labels": [gold, foil],
        "candidate_token_ids": candidate_ids,
        **metadata,
    }


def edit_positions(left: list[int], right: list[int]) -> list[int]:
    if len(left) != len(right):
        return []
    return [
        index
        for index, (a, b) in enumerate(zip(left, right))
        if int(a) != int(b)
    ]


def audit_unit(
    unit: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = {
        state: by_id[unit["record_ids"][state]]
        for state in STATES
    }
    lengths = {len(case["input_ids"]) for case in cases.values()}
    positions = {
        tuple(case["role_positions"][role] for role in CAPTURE_ROLES)
        for case in cases.values()
    }
    identity_exact = (
        cases["identity"]["input_ids"] == cases["a0_l0"]["input_ids"]
    )
    cue_position = cases["a0_l0"]["role_positions"]["cue"]
    target_position = cases["a0_l0"]["role_positions"]["target_word"]

    branch_edits = []
    lexical_edits = []
    target_edits = []
    for target in ("a", "n"):
        for lexical in (0, 1):
            branch_edits.append(edit_positions(
                cases[f"{target}0_l{lexical}"]["input_ids"],
                cases[f"{target}1_l{lexical}"]["input_ids"],
            ))
        for branch in (0, 1):
            lexical_edits.append(edit_positions(
                cases[f"{target}{branch}_l0"]["input_ids"],
                cases[f"{target}{branch}_l1"]["input_ids"],
            ))
    for branch in (0, 1):
        for lexical in (0, 1):
            target_edits.append(edit_positions(
                cases[f"a{branch}_l{lexical}"]["input_ids"],
                cases[f"n{branch}_l{lexical}"]["input_ids"],
            ))
    valid = bool(
        len(lengths) == 1
        and len(positions) == 1
        and identity_exact
        and all(row == [cue_position] for row in branch_edits)
        and all(row == [cue_position] for row in lexical_edits)
        and all(row == [target_position] for row in target_edits)
        and cue_position < target_position
    )
    return {
        "unit_id": unit["unit_id"],
        "valid": valid,
        "token_count": next(iter(lengths)) if len(lengths) == 1 else None,
        "role_positions_equal": len(positions) == 1,
        "identity_exact": identity_exact,
        "branch_edits": branch_edits,
        "lexical_edits": lexical_edits,
        "target_edits": target_edits,
        "cue_position": int(cue_position),
        "target_position": int(target_position),
    }


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_audits = []

    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        for prompt_mode in PROMPT_MODES:
            cases = []
            units = []
            for word in WORDS:
                for split in SPLITS:
                    for template in TEMPLATES_BY_SPLIT[split]:
                        for output_mode in OUTPUT_MODES:
                            for world in WORLDS:
                                unit_id = (
                                    f"p1017.{model_name}.{prompt_mode}."
                                    f"{word}.{split}.t{template}."
                                    f"{output_mode}.w{world}"
                                )
                                record_ids = {}
                                for state in STATES:
                                    case = build_case(
                                        tokenizer=tokenizer,
                                        model_name=model_name,
                                        prompt_mode=prompt_mode,
                                        word=word,
                                        split=split,
                                        template=template,
                                        output_mode=output_mode,
                                        world=world,
                                        unit_id=unit_id,
                                        state=state,
                                    )
                                    cases.append(case)
                                    record_ids[state] = case["record_id"]
                                units.append({
                                    "schema_version": (
                                        "phase1017_semantic_niche_unit.v1"
                                    ),
                                    "phase": PHASE,
                                    "protocol_revision": PROTOCOL_REVISION,
                                    "model": model_name,
                                    "prompt_mode": prompt_mode,
                                    "word": word,
                                    "split": split,
                                    "template": int(template),
                                    "output_mode": output_mode,
                                    "world": int(world),
                                    "unit_id": unit_id,
                                    "record_ids": record_ids,
                                })
            by_id = {case["record_id"]: case for case in cases}
            audits = [audit_unit(unit, by_id) for unit in units]
            if not all(row["valid"] for row in audits):
                bad = [row for row in audits if not row["valid"]]
                raise RuntimeError(
                    f"{model_name}/{prompt_mode}: invalid units {bad[:2]}"
                )
            case_path = (
                protocol_root
                / f"cases.{model_name}.{prompt_mode}.jsonl"
            )
            unit_path = (
                protocol_root
                / f"units.{model_name}.{prompt_mode}.jsonl"
            )
            write_jsonl(case_path, cases)
            write_jsonl(unit_path, units)
            summary_rows.append({
                "model": model_name,
                "prompt_mode": prompt_mode,
                "unit_count": len(units),
                "case_count": len(cases),
                "word_count": len(WORDS),
                "split_counts": dict(Counter(
                    row["split"] for row in units
                )),
                "output_mode_counts": dict(Counter(
                    row["output_mode"] for row in units
                )),
                "all_units_valid": True,
            })
            all_audits.extend({
                "model": model_name,
                "prompt_mode": prompt_mode,
                **row,
            } for row in audits)

    preregistration = {
        "schema_version": "phase1017_semantic_niche_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "prompt_modes": list(PROMPT_MODES),
        "words": list(WORDS),
        "splits": list(SPLITS),
        "templates_by_split": {
            key: list(value)
            for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "output_modes": list(OUTPUT_MODES),
        "worlds": list(WORLDS),
        "states": list(STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "primary_questions": [
            (
                "Does the same target word produce a repeatable "
                "target-conditioned branch interaction across heldout cues "
                "and templates?"
            ),
            (
                "Are physical components reused across words while branch "
                "interaction directions remain word-conditioned?"
            ),
            (
                "Does the interaction distinguish correct from failed "
                "behavior after generic cue routing is subtracted?"
            ),
        ],
        "claim_limits": [
            (
                "Fixed-model forward passes test contextual state "
                "reconstruction, not persistent weight plasticity."
            ),
            (
                "Semantic niche is an operational state-family description, "
                "not a mechanism equation."
            ),
            (
                "Repeated response is not a causal edge or sufficient "
                "decision mechanism."
            ),
        ],
        "descriptive_threshold_grid": {
            "direction_consistency": [0.30, 0.45, 0.60],
            "lexical_alignment": [0.20, 0.40, 0.60],
            "interaction_fraction": [0.10, 0.20, 0.40],
            "identity_maximum": 1e-6,
        },
    }
    preregistration["protocol_digest"] = digest(preregistration)
    write_json(protocol_root / "preregistration.json", preregistration)
    write_jsonl(protocol_root / "unit_audit.jsonl", all_audits)
    write_json(protocol_root / "summary.json", {
        "schema_version": "phase1017_semantic_niche_protocol_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": preregistration["protocol_digest"],
        "rows": summary_rows,
        "model_mode_count": len(summary_rows),
        "unit_count": sum(row["unit_count"] for row in summary_rows),
        "case_count": sum(row["case_count"] for row in summary_rows),
        "all_units_valid": all(
            row["all_units_valid"] for row in summary_rows
        ),
        "discovery_confirmation_cue_overlap": {
            word: sorted(
                set(sum(WORDS[word]["discovery"], ()))
                & set(sum(WORDS[word]["confirmation"], ()))
            )
            for word in WORDS
        },
    })
    print(json.dumps({
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": preregistration["protocol_digest"],
        "model_mode_count": len(summary_rows),
        "unit_count": sum(row["unit_count"] for row in summary_rows),
        "case_count": sum(row["case_count"] for row in summary_rows),
        "all_units_valid": True,
    }, indent=2))


if __name__ == "__main__":
    main()
