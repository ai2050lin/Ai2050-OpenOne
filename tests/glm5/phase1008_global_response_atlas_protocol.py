#!/usr/bin/env python3
"""Freeze Phase1008 multi-factor global response-atlas protocol.

The protocol deliberately separates observation from mechanism claims. It
creates role-aligned counterfactual worlds, but it does not use response data
to choose tokens, templates, operations, layers, or components.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1006_autoregressive_temporal_aggregation_protocol import ANSWER_PREFIX


PHASE = 1008
PROTOCOL_REVISION = 1
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation")
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
PAIR_OPERATIONS = ("B", "Q", "BQ", "E", "O", "N", "I")
ANALYSIS_OPERATIONS = PAIR_OPERATIONS + ("X",)
WORLDS_PER_POOL_TEMPLATE = 4

# All names and code words were already tokenizer-audited independently for
# each of the three local models in Phase1007. Discovery and confirmation
# vocabularies remain disjoint.
NAME_POOLS = {
    "discovery": (
        ("Alan", "Alec", "Andy", "Ben", "Brad", "Carl"),
        ("Chris", "Christopher", "Damian", "Eli", "Evan", "Finn"),
        ("Fred", "Gary", "Gordon", "Grant", "Greg", "Howard"),
    ),
    "confirmation": (
        ("Joshua", "Justin", "Keith", "Ken", "Kenneth", "Kyle"),
        ("Lee", "Leo", "Logan", "Louis", "Marcus", "Matt"),
        ("Matthew", "Max", "Nicholas", "Nick", "Oliver", "Oscar"),
    ),
}
CODE_POOLS = {
    "discovery": (
        ("clear", "quartz"),
        ("dense", "velvet"),
        ("sharp", "bronze"),
        ("mild", "coral"),
    ),
    "confirmation": (
        ("quiet", "pearl"),
        ("rapid", "moss"),
        ("fresh", "copper"),
        ("plain", "ivory"),
    ),
}
PROMPT_ROLES = (
    "fact_entity_0",
    "fact_value_0_word0",
    "fact_value_0_word1",
    "fact_entity_1",
    "fact_value_1_word0",
    "fact_value_1_word1",
    "nuisance_entity",
    "nuisance_value_word0",
    "nuisance_value_word1",
    "query_entity",
    "answer_boundary",
)
TIME_STAGES = ("prompt", "semantic0", "semantic1", "termination")
EVIDENCE_AXES = (
    "O_observation",
    "R_repetition",
    "S_specificity",
    "C_candidate_competition",
    "N_natural_rollout",
    "M_cross_model",
    "H_local_causality",
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1008_global_response_atlas"
)


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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def phrase(code: tuple[str, str] | list[str]) -> str:
    return f"{code[0]} {code[1]}"


def answer_text(model_name: str, code: tuple[str, str] | list[str]) -> str:
    return f"{ANSWER_PREFIX[model_name]}{phrase(code)}"


def one_token_id(tokenizer, text: str) -> int:
    values = tokenizer.encode(text, add_special_tokens=False)
    if len(values) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {values}")
    return int(values[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def render_user_prompt(
    template: int,
    facts: list[tuple[str, list[str], str]],
    query: str,
) -> str:
    """Render three facts; the third logical role is always a nuisance fact."""
    instruction = (
        "Reply exactly as Answer: [word1] [word2]. Replace the brackets "
        "with the requested lowercase two-word code. Add nothing else."
    )
    rendered_facts = []
    for index, (entity, code, _) in enumerate(facts):
        value = phrase(code)
        if template == 0:
            rendered_facts.append(
                f"Code record {index + 1}: {entity} has {value}."
            )
        elif template == 1:
            rendered_facts.append(
                f"Assignment {index + 1} gives {value} to {entity}."
            )
        elif template == 2:
            rendered_facts.append(
                f"Registry slot {index + 1} lists {entity} beside {value}."
            )
        elif template == 3:
            rendered_facts.append(
                f"Ledger mapping {index + 1} is {entity} -> {value}."
            )
        else:
            raise KeyError(template)
    if template == 0:
        question = f"Requested person: {query}."
    elif template == 1:
        question = f"Return the complete assignment for {query}."
    elif template == 2:
        question = f"Look up the full registry value for {query}."
    else:
        question = f"Give the two-part ledger value mapped from {query}."
    return f"{' '.join(rendered_facts)}\n{question}\n{instruction}"


def answer_protocol(tokenizer, model_name: str, split: str) -> dict[str, Any]:
    answers = {
        phrase(code): [
            int(value)
            for value in tokenizer.encode(
                answer_text(model_name, code),
                add_special_tokens=False,
            )
        ]
        for code in CODE_POOLS[split]
    }
    widths = {len(ids) for ids in answers.values()}
    if len(widths) != 1:
        raise RuntimeError(f"{model_name}/{split}: answer width drift {widths}")
    width = next(iter(widths))
    varying = [
        index
        for index in range(width)
        if len({ids[index] for ids in answers.values()}) > 1
    ]
    if (
        len(varying) != 2
        or varying[1] != varying[0] + 1
        or varying[1] != width - 1
    ):
        raise RuntimeError(
            f"{model_name}/{split}: semantic answer steps {varying}"
        )
    prefixes = {tuple(ids[:varying[0]]) for ids in answers.values()}
    if len(prefixes) != 1:
        raise RuntimeError(f"{model_name}/{split}: answer prefix drift")
    candidates = []
    for logical_step, absolute_step in enumerate(varying):
        candidates.append({
            code[logical_step]: int(answers[phrase(code)][absolute_step])
            for code in CODE_POOLS[split]
        })
    return {
        "answers": answers,
        "semantic_steps": varying,
        "protocol_prefix_ids": list(next(iter(prefixes))),
        "candidate_ids_by_step": candidates,
    }


def rotated(values: tuple[Any, ...], amount: int) -> list[Any]:
    offset = amount % len(values)
    return list(values[offset:] + values[:offset])


def state_specs(
    names: list[str],
    codes: list[tuple[str, str]],
    query_role: int,
    base_order: list[int],
) -> dict[str, dict[str, Any]]:
    """Build minimally controlled semantic states before text rendering."""
    base_entities = names[:3]
    renamed_entities = names[3:6]
    base_codes = [list(value) for value in codes[:3]]
    replacement_code = list(codes[3])

    def make(
        operation: str,
        entities: list[str],
        assignments: list[list[str]],
        query_index: int,
        order: list[int],
    ) -> dict[str, Any]:
        logical = [
            (entities[0], assignments[0], "focal0"),
            (entities[1], assignments[1], "focal1"),
            (entities[2], assignments[2], "nuisance"),
        ]
        return {
            "operation": operation,
            "entities": list(entities),
            "assigned_codes": [list(value) for value in assignments],
            "query_role": int(query_index),
            "query_entity": entities[query_index],
            "display_order": list(order),
            "facts": [logical[index] for index in order],
            "gold_code": list(assignments[query_index]),
            "foil_code": list(assignments[1 - query_index]),
        }

    swapped = [base_codes[1], base_codes[0], base_codes[2]]
    opposite_query = 1 - query_role
    states = {
        "base": make(
            "base",
            base_entities,
            base_codes,
            query_role,
            base_order,
        ),
        "B": make(
            "B",
            base_entities,
            swapped,
            query_role,
            base_order,
        ),
        "Q": make(
            "Q",
            base_entities,
            base_codes,
            opposite_query,
            base_order,
        ),
        "BQ": make(
            "BQ",
            base_entities,
            swapped,
            opposite_query,
            base_order,
        ),
        "E": make(
            "E",
            renamed_entities,
            base_codes,
            query_role,
            base_order,
        ),
        "O": make(
            "O",
            base_entities,
            base_codes,
            query_role,
            [2, base_order[1], base_order[0]],
        ),
        "N": make(
            "N",
            [base_entities[0], base_entities[1], renamed_entities[2]],
            [base_codes[0], base_codes[1], replacement_code],
            query_role,
            base_order,
        ),
    }
    if states["BQ"]["gold_code"] != states["base"]["gold_code"]:
        raise RuntimeError("BQ failed the frozen same-answer construction")
    return states


def build_case(
    *,
    tokenizer,
    model_name: str,
    split: str,
    template: int,
    unit_id: str,
    state_name: str,
    spec: dict[str, Any],
    name_ids: dict[str, int],
    word_ids: dict[str, int],
    answer: dict[str, Any],
) -> dict[str, Any]:
    raw_prompt = render_user_prompt(template, spec["facts"], spec["query_entity"])
    rendered = render_chat(tokenizer, model_name, raw_prompt)
    ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    raw_start = rendered.index(raw_prompt)
    raw_end = raw_start + len(raw_prompt)
    prefix_ids = [
        int(value)
        for value in tokenizer.encode(
            rendered[:raw_start], add_special_tokens=False
        )
    ]
    through_user_ids = [
        int(value)
        for value in tokenizer.encode(
            rendered[:raw_end], add_special_tokens=False
        )
    ]
    if (
        ids[:len(prefix_ids)] != prefix_ids
        or ids[:len(through_user_ids)] != through_user_ids
    ):
        raise RuntimeError("user-content token span drift")

    entities = spec["entities"]
    query = spec["query_entity"]
    entity_fact_positions: dict[int, int] = {}
    for logical_index, entity in enumerate(entities[:2]):
        found = positions_of(ids, name_ids[entity])
        expected = 2 if entity == query else 1
        if len(found) != expected:
            raise RuntimeError(
                f"{model_name}/{unit_id}/{state_name}: "
                f"{entity} positions {found}"
            )
        entity_fact_positions[logical_index] = found[0]
    nuisance_found = positions_of(ids, name_ids[entities[2]])
    if len(nuisance_found) != 1:
        raise RuntimeError(f"nuisance position drift: {nuisance_found}")
    query_found = positions_of(ids, name_ids[query])
    if len(query_found) != 2:
        raise RuntimeError(f"query position drift: {query_found}")

    value_positions = []
    for code in spec["assigned_codes"]:
        positions = []
        for word in code:
            found = positions_of(ids, word_ids[word])
            if len(found) != 1:
                raise RuntimeError(f"{word}: code positions {found}")
            positions.append(found[0])
        value_positions.append(positions)

    role_positions = {
        "fact_entity_0": entity_fact_positions[0],
        "fact_value_0_word0": value_positions[0][0],
        "fact_value_0_word1": value_positions[0][1],
        "fact_entity_1": entity_fact_positions[1],
        "fact_value_1_word0": value_positions[1][0],
        "fact_value_1_word1": value_positions[1][1],
        "nuisance_entity": nuisance_found[0],
        "nuisance_value_word0": value_positions[2][0],
        "nuisance_value_word1": value_positions[2][1],
        "query_entity": query_found[-1],
        "answer_boundary": len(ids) - 1,
    }
    if tuple(role_positions) != PROMPT_ROLES:
        raise RuntimeError("role order drift")

    gold = phrase(spec["gold_code"])
    foil = phrase(spec["foil_code"])
    answer_ids = answer["answers"][gold]
    extended = [
        int(value)
        for value in tokenizer.encode(
            rendered + answer_text(model_name, spec["gold_code"]),
            add_special_tokens=False,
        )
    ]
    if extended != ids + answer_ids:
        raise RuntimeError("answer token boundary drift")
    return {
        "schema_version": "phase1008_case.v1",
        "phase": PHASE,
        "model": model_name,
        "split": split,
        "template": int(template),
        "unit_id": unit_id,
        "record_id": f"{unit_id}.{state_name}",
        "state": state_name,
        "operation": spec["operation"],
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": ids,
        "role_positions": role_positions,
        "entities": list(spec["entities"]),
        "assigned_codes": [phrase(value) for value in spec["assigned_codes"]],
        "query_role": int(spec["query_role"]),
        "query_entity": spec["query_entity"],
        "display_order": list(spec["display_order"]),
        "gold": gold,
        "gold_parts": list(spec["gold_code"]),
        "foil": foil,
        "foil_parts": list(spec["foil_code"]),
        "answer_text": answer_text(model_name, spec["gold_code"]),
        "answer_token_ids": list(answer_ids),
        "semantic_steps": list(answer["semantic_steps"]),
        "protocol_prefix_ids": list(answer["protocol_prefix_ids"]),
        "candidate_ids_by_step": answer["candidate_ids_by_step"],
    }


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    all_names = tuple(
        name
        for split in SPLITS
        for pool in NAME_POOLS[split]
        for name in pool
    )
    all_words = tuple(
        word
        for split in SPLITS
        for code in CODE_POOLS[split]
        for word in code
    )
    name_ids = {name: one_token_id(tokenizer, " " + name) for name in all_names}
    word_ids = {word: one_token_id(tokenizer, " " + word) for word in all_words}
    if len(set(name_ids.values())) != len(name_ids):
        raise RuntimeError(f"{model_name}: name token collision")
    if len(set(word_ids.values())) != len(word_ids):
        raise RuntimeError(f"{model_name}: code token collision")
    answer_by_split = {
        split: answer_protocol(tokenizer, model_name, split)
        for split in SPLITS
    }

    cases: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    widths: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for split in SPLITS:
        for template in TEMPLATES_BY_SPLIT[split]:
            for pool_index, pool in enumerate(NAME_POOLS[split]):
                for world_index in range(WORLDS_PER_POOL_TEMPLATE):
                    names = rotated(pool, world_index)
                    codes = rotated(CODE_POOLS[split], world_index)
                    query_role = world_index % 2
                    base_order = (
                        [0, 1, 2] if world_index < 2 else [1, 0, 2]
                    )
                    unit_id = (
                        f"{model_name}.{split[0]}t{template}."
                        f"p{pool_index}.w{world_index}"
                    )
                    specs = state_specs(
                        names,
                        codes,
                        query_role,
                        base_order,
                    )
                    case_ids = {}
                    for state_name, spec in specs.items():
                        case = build_case(
                            tokenizer=tokenizer,
                            model_name=model_name,
                            split=split,
                            template=template,
                            unit_id=unit_id,
                            state_name=state_name,
                            spec=spec,
                            name_ids=name_ids,
                            word_ids=word_ids,
                            answer=answer_by_split[split],
                        )
                        cases.append(case)
                        case_ids[state_name] = case["record_id"]
                        widths[(split, template, state_name)].add(
                            len(case["input_ids"])
                        )
                    units.append({
                        "schema_version": "phase1008_unit.v1",
                        "phase": PHASE,
                        "model": model_name,
                        "split": split,
                        "template": int(template),
                        "name_pool": int(pool_index),
                        "world_index": int(world_index),
                        "unit_id": unit_id,
                        "query_role": int(query_role),
                        "case_ids": case_ids,
                        "operation_pairs": {
                            operation: {
                                "base": case_ids["base"],
                                "variant": (
                                    case_ids["base"]
                                    if operation == "I"
                                    else case_ids[operation]
                                ),
                            }
                            for operation in PAIR_OPERATIONS
                        },
                    })

    width_audit = {
        f"{split}.t{template}.{state}": sorted(values)
        for (split, template, state), values in sorted(widths.items())
    }
    if any(len(values) != 1 for values in width_audit.values()):
        raise RuntimeError(f"{model_name}: prompt width drift {width_audit}")
    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "units.jsonl", units)
    summary = {
        "schema_version": "phase1008_model_protocol.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(units),
        "pair_count": len(units) * len(PAIR_OPERATIONS),
        "split_unit_counts": {
            split: sum(unit["split"] == split for unit in units)
            for split in SPLITS
        },
        "prompt_widths": width_audit,
        "tokenizer_audit": {
            "single_token_name_count": len(name_ids),
            "single_token_code_word_count": len(word_ids),
            "name_collisions": 0,
            "code_collisions": 0,
        },
        "answer_protocol": answer_by_split,
    }
    write_json(model_root / "summary.json", summary)
    return summary


def build_protocol() -> dict[str, Any]:
    summaries = [build_model(model_name) for model_name in MODELS]
    payload = {
        "schema_version": "phase1008_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Multi-factor global response tracing and repeated functional "
            "topology discovery"
        ),
        "models_in_required_execution_order": list(MODELS),
        "splits": list(SPLITS),
        "templates_by_split": {
            split: list(values)
            for split, values in TEMPLATES_BY_SPLIT.items()
        },
        "name_pools": {
            split: [list(pool) for pool in NAME_POOLS[split]]
            for split in SPLITS
        },
        "code_pools": {
            split: [list(code) for code in CODE_POOLS[split]]
            for split in SPLITS
        },
        "pair_operations": list(PAIR_OPERATIONS),
        "analysis_operations": list(ANALYSIS_OPERATIONS),
        "operation_definitions": {
            "B": {
                "name": "binding_flip",
                "output": "changes",
                "edit": "swap the two focal value assignments",
            },
            "Q": {
                "name": "query_flip",
                "output": "changes",
                "edit": "query the other focal entity",
            },
            "BQ": {
                "name": "binding_query_flip",
                "output": "same_as_base",
                "edit": "swap bindings and query the other entity",
            },
            "E": {
                "name": "entity_rename",
                "output": "same_as_base",
                "edit": "rename all entities with one-token held-out names",
            },
            "O": {
                "name": "fact_order",
                "output": "same_as_base",
                "edit": "move nuisance first and reverse focal fact order",
            },
            "N": {
                "name": "nuisance_replace",
                "output": "same_as_base",
                "edit": "replace only the irrelevant entity and value",
            },
            "I": {
                "name": "identity_repeat",
                "output": "same_as_base",
                "edit": "repeat the identical input as a numerical floor",
            },
            "X": {
                "name": "factorial_interaction_measurement",
                "output": "not_an_input_operation",
                "edit": "h(BQ)-h(B)-h(Q)+h(base)",
            },
            "T": {
                "name": "template_replication_stratum",
                "output": "same_semantic_task",
                "edit": (
                    "independent templates are compared by semantic role "
                    "and normalized depth, never raw token position"
                ),
            },
        },
        "control_semantics": {
            "I": "the only universal numerical no-change control",
            "E": (
                "same-output lexical invariance reference; not a negative "
                "control at renamed entity roles"
            ),
            "O": (
                "same-output order invariance reference; internal position "
                "responses are expected"
            ),
            "N": (
                "nuisance reference outside edited nuisance roles; not a "
                "negative control at the edited distractor"
            ),
            "BQ": (
                "same-output computation-change contrast that separates "
                "answer identity from internal response"
            ),
        },
        "stages": list(TIME_STAGES),
        "prompt_roles": list(PROMPT_ROLES),
        "components": (
            "residual_depth_0_to_L",
            "attention_output_layer_1_to_L",
            "mlp_output_layer_1_to_L",
        ),
        "behavior_qualification": {
            "semantic_pair": (
                "base and variant both predict semantic token 0, autonomous "
                "semantic token 1, and teacher-forced semantic token 1"
            ),
            "rollout_pair": (
                "base and variant both naturally generate the exact answer "
                "and terminate immediately"
            ),
            "policy": (
                "failed cells remain in the atlas with validity flags; they "
                "do not stop other cells or erase observations"
            ),
        },
        "measurement_contract": {
            "raw_magnitude": "L2 norm of paired response difference",
            "state_normalized_magnitude": (
                "raw magnitude divided by mean paired state norm"
            ),
            "direction_consistency": (
                "exact mean pairwise cosine from the sum of unit response "
                "directions; reported as a measurement, not a theory"
            ),
            "first_peak_persistence": (
                "computed only along role-aligned depth trajectories"
            ),
            "factorial_interaction": (
                "descriptive BQ-B-Q+base residual, not an assumed mechanism"
            ),
            "cross_model": (
                "compare normalized depth, semantic role, component, stage, "
                "and event order; never raw coordinates"
            ),
            "edge_vocabulary": (
                "global scan may create co_response edges only; transport, "
                "mediation, and causality require separate interventions"
            ),
            "dimension_reduction": (
                "PCA/UMAP/t-SNE may visualize retained motifs but cannot "
                "select or validate them"
            ),
        },
        "evidence_axes": list(EVIDENCE_AXES),
        "storage_contract": {
            "global": (
                "save behavior, per-unit scalar response arrays, aggregate "
                "direction consistency, event metadata, and summaries"
            ),
            "forbidden": "do not save all raw hidden/component tensors",
            "refinement": (
                "head/neuron/KV raw slices only after cross-pool and "
                "cross-template repeated regions are identified"
            ),
        },
        "descriptive_selection_rule": {
            "purpose": "select refinement candidates, not prove mechanisms",
            "within_trajectory_peak_fraction": 0.90,
            "required_name_pools": 2,
            "required_templates": 2,
            "required_splits": 2,
            "minimum_semantic_qualified_pairs_per_split": 8,
        },
        "model_summaries": summaries,
    }
    payload["preregistration_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "protocol.json", payload)
    return payload


def main() -> None:
    protocol = build_protocol()
    print(json.dumps({
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "preregistration_digest": protocol["preregistration_digest"],
        "model_units": {
            row["model"]: row["unit_count"]
            for row in protocol["model_summaries"]
        },
        "pair_operations": list(PAIR_OPERATIONS),
        "analysis_operations": list(ANALYSIS_OPERATIONS),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
