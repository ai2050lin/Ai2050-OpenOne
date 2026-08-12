#!/usr/bin/env python3
"""Freeze Phase1094 semantic-alias versus graph-topology orthogonalization.

The experiment asks a narrower question than Phase1093.  Each relation edge
uses two occurrence-specific words, so no exact value string is shared between
edges.  In the coherent condition the two occurrences assigned to one graph
node are genuine synonyms.  In the scrambled condition the exact same word
multiset is retained, but the second occurrence is deranged across concepts.
Two degree-matched non-isomorphic graphs separate generic degree/incidence
effects from synonym-node reuse.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_protocol as phase1092
import phase1093_independent_relation_protocol as prior


PHASE = 1094
PROTOCOL_REVISION = 1
MODELS = prior.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
ATTRIBUTES = ("size", "color")
PRIMARY_ATTRIBUTE = "size"
SECONDARY_ATTRIBUTE = "color"
TOPOLOGIES = ("cycle8", "cycle3_5")
COHERENCES = ("coherent", "scrambled")
SURFACES = ("en", "zh")
BASE_WORLDS = ("celestial", "mineral", "maritime", "architectural")
SPLITS = ("discovery", "confirmation")
PANELS = ("active", "field_null")
TEMPLATE_IDS = (0, 1)
OUTPUT_SET_IDS = (0,)
ITEMS_PER_CELL_SPLIT = 3
GENERATION_STEPS = 6
GENERATION_UNITS_PER_CELL_SPLIT = 1
TARGET_RELATIVE_DEPTH_MIN = 0.25
TARGET_RELATIVE_DEPTH_MAX = 0.70
CAPTURE_ROLES = ("dossier_end", "query_end", "answer_boundary")
SIGNED_FIELDS = ("active_binding", "field_null", "content")
SIGNED_PROJECTION_DIM = 96
SIGNED_PROJECTION_REPLICATES = 2
SIGNED_PROJECTION_SEED = 1094001
STATES = tuple(
    f"t{template}_c{panel}_m{target}_q{binding}_w0"
    for template in TEMPLATE_IDS
    for panel in PANELS
    for target in (0, 1)
    for binding in (0, 1)
)
FACT_ORDERS = tuple(itertools.permutations(("entity0", "entity1", "anchor")))


CONCEPT_IDS = tuple(range(8))
CONCEPT_META = {
    "size": (
        "tiny", "huge", "short", "tall",
        "narrow", "wide", "thin", "thick",
    ),
    "color": (
        "red", "blue", "green", "yellow",
        "black", "white", "orange", "purple",
    ),
}

# [surface][split][attribute][concept][occurrence_alias]
# Discovery and confirmation are disjoint.  The two aliases within a split are
# also occurrence-specific and each appears in exactly one relation edge.
ALIASES = {
    "en": {
        "discovery": {
            "size": (
                ("microscopic", "miniature"),
                ("mammoth", "immense"),
                ("stubby", "stunted"),
                ("statuesque", "soaring"),
                ("needle-thin", "slimline"),
                ("roomy", "outspread"),
                ("gauzy", "papery"),
                ("bulky", "hefty"),
            ),
            "color": (
                ("ruby", "carmine"),
                ("cerulean", "sapphire"),
                ("jade", "malachite"),
                ("lemon", "saffron"),
                ("onyx", "charcoal"),
                ("alabaster", "pearl"),
                ("copper", "pumpkin"),
                ("plum", "amethyst"),
            ),
        },
        "confirmation": {
            "size": (
                ("wee", "petite"),
                ("gigantic", "monumental"),
                ("dwarfish", "low-rise"),
                ("elevated", "high-reaching"),
                ("close-set", "straitened"),
                ("spacious", "sweeping"),
                ("wispy", "waferlike"),
                ("substantial", "heavyset"),
            ),
            "color": (
                ("vermilion", "garnet"),
                ("indigo", "navy"),
                ("olive", "mint"),
                ("canary", "ochre"),
                ("pitch-dark", "raven"),
                ("chalky", "milky"),
                ("coral", "rust"),
                ("mauve", "orchid-purple"),
            ),
        },
    },
    "zh": {
        "discovery": {
            "size": (
                ("微缩", "袖珍"),
                ("宏伟", "浩大"),
                ("矮墩", "短矮"),
                ("修长", "峻拔"),
                ("窄细", "纤窄"),
                ("辽阔", "广展"),
                ("蝉翼般薄", "薄绢般"),
                ("厚墩", "粗壮"),
            ),
            "color": (
                ("朱砂", "胭脂"),
                ("靛青", "天青"),
                ("翡翠", "孔雀石色"),
                ("柠檬色", "藏红花色"),
                ("曜石色", "炭墨"),
                ("雪玉", "珍珠色"),
                ("铜橙", "南瓜色"),
                ("梅子色", "紫晶色"),
            ),
        },
        "confirmation": {
            "size": (
                ("细微", "迷你"),
                ("硕巨", "雄伟"),
                ("低身", "矮壮"),
                ("高拔", "颀长"),
                ("狭长", "收窄"),
                ("舒展", "宽展"),
                ("轻纱般", "薄片状"),
                ("肥厚", "敦实"),
            ),
            "color": (
                ("丹霞色", "石榴色"),
                ("宝蓝", "藏蓝"),
                ("橄榄色", "薄荷色"),
                ("鹅黄", "赭黄"),
                ("漆墨", "玄色"),
                ("粉笔色", "乳色"),
                ("珊瑚色", "锈色"),
                ("藕荷色", "兰花紫"),
            ),
        },
    },
}


TOPOLOGY_EDGES = {
    "cycle8": (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 0),
    ),
    "cycle3_5": (
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 6), (6, 7), (7, 3),
    ),
}
SCRAMBLE_SHIFT = 3


def occurrence_alias_indices(topology: str) -> tuple[tuple[int, int], ...]:
    """Assign alias 0/1 to the two occurrences of every degree-two slot."""
    counts: Counter[int] = Counter()
    rows = []
    for left, right in TOPOLOGY_EDGES[topology]:
        rows.append((counts[left], counts[right]))
        counts[left] += 1
        counts[right] += 1
    if set(counts.values()) != {2}:
        raise ValueError(f"topology is not degree two: {topology}")
    return tuple(rows)


OCCURRENCE_ALIAS_INDICES = {
    topology: occurrence_alias_indices(topology) for topology in TOPOLOGIES
}


def semantic_concept(slot: int, alias_index: int, coherence: str) -> int:
    if coherence == "coherent" or alias_index == 0:
        return slot
    return (slot + SCRAMBLE_SHIFT) % len(CONCEPT_IDS)


def operation_name(attribute: str, topology: str, coherence: str, edge: int) -> str:
    return f"{attribute}_{topology}_{coherence}_edge{edge:02d}"


OPERATIONS = tuple(
    operation_name(attribute, topology, coherence, edge)
    for attribute in ATTRIBUTES
    for topology in TOPOLOGIES
    for coherence in COHERENCES
    for edge in range(8)
)
OPERATION_META: dict[str, dict[str, Any]] = {}
for attribute in ATTRIBUTES:
    for topology in TOPOLOGIES:
        for coherence in COHERENCES:
            for edge_index, slot_pair in enumerate(TOPOLOGY_EDGES[topology]):
                alias_pair = OCCURRENCE_ALIAS_INDICES[topology][edge_index]
                semantic_pair = tuple(
                    semantic_concept(slot, alias_index, coherence)
                    for slot, alias_index in zip(slot_pair, alias_pair)
                )
                name = operation_name(attribute, topology, coherence, edge_index)
                OPERATION_META[name] = {
                    "attribute": attribute,
                    "topology": topology,
                    "coherence": coherence,
                    "edge_index": edge_index,
                    "slot_pair": list(slot_pair),
                    "alias_index_pair": list(alias_pair),
                    "semantic_pair": list(semantic_pair),
                    "semantic_names": [
                        CONCEPT_META[attribute][index] for index in semantic_pair
                    ],
                }


WORLDS = tuple(
    f"{world}@{surface}" for world in BASE_WORLDS for surface in SURFACES
)
CELLS = tuple(f"{operation}__{world}" for operation in OPERATIONS for world in WORLDS)
FAMILIES = CELLS


def entries(*rows: tuple[str, str, str]) -> tuple[dict[str, str], ...]:
    return tuple({"id": key, "en": en, "zh": zh} for key, en, zh in rows)


ENTITY_POOLS = {
    "celestial": {
        "discovery": entries(
            ("orion94", "Orion", "猎户"), ("lyra94", "Lyra", "天琴"),
            ("draco94", "Draco", "天龙"), ("cygnus94", "Cygnus", "天鹅"),
            ("aquila94", "Aquila", "天鹰"), ("perseus94", "Perseus", "英仙"),
            ("cetus94", "Cetus", "鲸鱼"), ("ara94", "Ara", "天坛"),
            ("lupus94", "Lupus", "豺狼"),
        ),
        "confirmation": entries(
            ("vela94", "Vela", "船帆"), ("pictor94", "Pictor", "绘架"),
            ("dorado94", "Dorado", "剑鱼"), ("volans94", "Volans", "飞鱼"),
            ("columba94", "Columba", "天鸽"), ("caelum94", "Caelum", "雕具"),
            ("fornax94", "Fornax", "天炉"), ("mensa94", "Mensa", "山案"),
            ("norma94", "Norma", "矩尺"),
        ),
    },
    "mineral": {
        "discovery": entries(
            ("zircon94", "zircon", "锆石"), ("calcite94", "calcite", "方解石"),
            ("feldspar94", "feldspar", "长石"), ("mica94", "mica", "云母"),
            ("gypsum94", "gypsum", "石膏"), ("talc94", "talc", "滑石"),
            ("olivine94", "olivine", "橄榄石"), ("pyrite94", "pyrite", "黄铁矿"),
            ("fluorite94", "fluorite", "萤石"),
        ),
        "confirmation": entries(
            ("halite94", "halite", "岩盐"), ("barite94", "barite", "重晶石"),
            ("apatite94", "apatite", "磷灰石"), ("dolomite94", "dolomite", "白云石"),
            ("kaolinite94", "kaolinite", "高岭石"), ("beryl94", "beryl", "绿柱石"),
            ("spinel94", "spinel", "尖晶石"), ("corundum94", "corundum", "刚玉"),
            ("rutile94", "rutile", "金红石"),
        ),
    },
    "maritime": {
        "discovery": entries(
            ("cutter94", "cutter", "快艇"), ("dhow94", "dhow", "三角帆船"),
            ("corvette94", "corvette", "轻护舰"), ("frigate94", "frigate", "巡防舰"),
            ("schooner94", "schooner", "纵帆船"), ("catamaran94", "catamaran", "双体船"),
            ("sloop94", "sloop", "单桅帆船"), ("yawl94", "yawl", "双桅小帆船"),
            ("trawler94", "trawler", "拖网船"),
        ),
        "confirmation": entries(
            ("brigantine94", "brigantine", "双桅帆船"), ("caravel94", "caravel", "卡拉维尔帆船"),
            ("galleon94", "galleon", "盖伦船"), ("longship94", "longship", "长船"),
            ("junk94", "junk", "中式帆船"), ("outrigger94", "outrigger", "支架独木舟"),
            ("hydrofoil94", "hydrofoil", "水翼船"), ("trimaran94", "trimaran", "三体船"),
            ("tugboat94", "tugboat", "拖船"),
        ),
    },
    "architectural": {
        "discovery": entries(
            ("pagoda94", "pagoda", "宝塔"), ("archway94", "archway", "拱门"),
            ("rotunda94", "rotunda", "圆厅"), ("cenotaph94", "cenotaph", "衣冠冢"),
            ("minaret94", "minaret", "宣礼塔"), ("pavilion94", "pavilion", "亭阁"),
            ("basilica94", "basilica", "长殿"), ("colonnade94", "colonnade", "柱廊"),
            ("belfry94", "belfry", "钟楼"),
        ),
        "confirmation": entries(
            ("ziggurat94", "ziggurat", "塔庙"), ("stupa94", "stupa", "窣堵坡"),
            ("triumphal94", "triumphal arch", "凯旋门"), ("campanile94", "campanile", "钟塔"),
            ("cupola94", "cupola", "穹顶"), ("arcade94", "arcade", "券廊"),
            ("propylaea94", "propylaea", "山门"), ("barbican94", "barbican", "瓮城"),
            ("clerestory94", "clerestory", "高侧窗"),
        ),
    },
}


ANSWER_LABELS = {"en": ("Yes", "No"), "zh": ("是", "否")}
ASSISTANT_PREFILLS = {"en": "\nAnswer:", "zh": "\n答案："}
SHELLS = prior.SHELLS
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1094_semantic_topology_orthogonal"
SOURCE_PHASE1093 = prior.OUT_ROOT / "analysis" / "final_summary.json"

EVIDENCE_THRESHOLDS = {
    "minimum_candidate_accuracy": 0.80,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_generation_accuracy": 0.75,
    "minimum_behavior_worlds_per_edge": 3,
    "minimum_behavior_edges_per_condition": 6,
    "minimum_behavior_models": 2,
    "maximum_projection_median_abs_norm_error": 0.08,
    "maximum_projection_p95_abs_norm_error": 0.20,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_query_tolerance": 1e-8,
    "minimum_edge_top1": 6,
    "permutation_p_max": 0.01,
    "minimum_content_identity_advantage": 0.10,
    "minimum_incidence_fit": 0.45,
    "minimum_content_over_null_fit_advantage": 0.10,
    "minimum_coherent_over_scrambled_advantage": 0.15,
    "minimum_true_over_slot_scrambled_advantage": 0.10,
    "minimum_required_cells": 6,
    "minimum_required_models": 2,
    "minimum_residual_family_advantage": 0.10,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": "All static audits pass, including exact alias-multiset matching, degree matching, non-isomorphic connectivity, lexical holdout, truth matching, and tokenizer checks.",
    "P2": "At least two healthy FP16 models pass size in both topologies, both coherence conditions, and both languages; color coherent controls also pass.",
    "P3": "At least two behavior-authorized models pass hidden finite-state and independent dual-projection audits.",
    "P4": "With no exact value string shared across edges, coherent synonym nodes preserve discovery-to-confirmation edge identity above matched null in size.",
    "P5": "The coherent hidden Gram fits the preregistered slot-incidence graph more strongly than the exact-word-matched scrambled Gram in at least two models.",
    "P6": "In scrambled panels, hidden geometry follows actual semantic-concept incidence more strongly than the nominal slot graph in at least two models.",
    "P7": "The primary size result repeats in both degree-matched topologies and both natural-language surfaces.",
    "P8": "Color independently repeats the semantic-alias advantage in at least two models; failure blocks a two-family claim.",
    "P9": "The semantic-incidence result is visible across at least two model implementations after comparing function geometry rather than raw coordinates.",
    "P10": "After removing incidence geometry, any remaining size-versus-color family residual repeats prospectively above matched null.",
    "P11": "Only after P1-P10 may a descriptive physical band be promoted for later causal localization; Phase1094 itself never closes causality.",
}


write_json = prior.write_json
write_jsonl = prior.write_jsonl
read_json = prior.read_json
read_jsonl = prior.read_jsonl
digest = prior.digest
tokenizer_for = prior.tokenizer_for
offset_token_spans = prior.offset_token_spans
behavior = prior.behavior
mark_source = prior.mark_source


def split_world(value: str) -> tuple[str, str]:
    return tuple(value.split("@", 1))  # type: ignore[return-value]


def split_cell(cell: str) -> tuple[str, str]:
    return tuple(cell.split("__", 1))  # type: ignore[return-value]


def state_factors(state: str) -> tuple[int, str, int, int, int]:
    return prior.state_factors(state)


def surface_entity(entity: dict[str, str], surface: str) -> str:
    return str(entity[surface])


def operation_alias_ids(operation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    meta = OPERATION_META[operation]
    return tuple(
        (int(concept), int(alias_index))
        for concept, alias_index in zip(
            meta["semantic_pair"], meta["alias_index_pair"]
        )
    )  # type: ignore[return-value]


def operation_surface_pair(operation: str, surface: str, split: str) -> tuple[str, str]:
    attribute = str(OPERATION_META[operation]["attribute"])
    return tuple(
        str(ALIASES[surface][split][attribute][concept][alias_index])
        for concept, alias_index in operation_alias_ids(operation)
    )  # type: ignore[return-value]


def fact_text(surface: str, entity: str, value: str) -> str:
    if surface == "en":
        return f"The {entity} was described as {value}."
    return f"记录将{entity}描述为{value}。"


def question_text(surface: str, entity: str, value: str) -> str:
    if surface == "en":
        return f"Was the {entity} described as {value}?"
    return f"记录是否将{entity}描述为{value}？"


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    operation, world_surface = split_cell(cell)
    world, surface = split_world(world_surface)
    pair = operation_surface_pair(operation, surface, split)
    pool = ENTITY_POOLS[world][split]
    rows = []
    for local_index in range(ITEMS_PER_CELL_SPLIT):
        anchor_variant = local_index % 2
        rows.append({
            "item_id": f"{cell}.{split}.{local_index:02d}",
            "entity0": pool[local_index],
            "entity1": pool[(local_index + 3) % len(pool)],
            "anchor": pool[(local_index + 6) % len(pool)],
            "anchor_variant": anchor_variant,
            "anchor_value": pair[anchor_variant],
            "fact_order_index": local_index,
            "operation": operation,
            "attribute": OPERATION_META[operation]["attribute"],
            "topology": OPERATION_META[operation]["topology"],
            "coherence": OPERATION_META[operation]["coherence"],
            "world": world,
            "surface": surface,
        })
    return tuple(rows)


def build_case(
    tokenizer,
    model_name: str,
    cell: str,
    split: str,
    item: dict[str, Any],
    state: str,
    case_index: int,
) -> dict[str, Any]:
    operation, world_surface = split_cell(cell)
    world, surface = split_world(world_surface)
    template, panel, target_variant, binding, output_set = state_factors(state)
    meta = OPERATION_META[operation]
    pair = operation_surface_pair(operation, surface, split)
    bound_values = pair if binding == 0 else tuple(reversed(pair))
    selected = item["entity0"] if panel == "active" else item["anchor"]
    selected_entity = surface_entity(selected, surface)
    semantic_answer = (
        int(binding != target_variant)
        if panel == "active"
        else int(int(item["anchor_variant"]) != target_variant)
    )
    answer_labels = ANSWER_LABELS[surface]
    target_answer = answer_labels[semantic_answer]

    entities = {
        role: surface_entity(item[role], surface)
        for role in ("entity0", "entity1", "anchor")
    }
    fact_values = {
        "entity0": bound_values[0],
        "entity1": bound_values[1],
        "anchor": str(item["anchor_value"]),
    }
    facts = {
        role: fact_text(surface, entities[role], fact_values[role])
        for role in ("entity0", "entity1", "anchor")
    }
    order = FACT_ORDERS[int(item["fact_order_index"]) % len(FACT_ORDERS)]
    separator = " " if surface == "en" else ""
    dossier = separator.join(facts[role] for role in order)
    query_value = pair[target_variant]
    question = question_text(surface, selected_entity, query_value)
    raw_prompt = SHELLS[surface][template].format(dossier=dossier, question=question)
    raw_spans = {
        "dossier_end": mark_source.mark(raw_prompt, facts[order[-1]], occurrence="first"),
        "query_end": mark_source.mark(raw_prompt, question, occurrence="last"),
    }
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    ) + ASSISTANT_PREFILLS[surface]
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(tokenizer, rendered, raw_prompt, raw_spans)
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    prefix = " " if surface == "en" else ""
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(tokenizer, rendered, prefix, answer)
        for index, answer in enumerate(answer_labels)
    }
    return {
        "schema_version": "phase1094_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{model_name}.{cell}.{split}.{item['item_id']}.{state}",
        "unit_id": f"{cell}.{split}.{item['item_id']}",
        "family": cell,
        "cell": cell,
        "operation": operation,
        "attribute": meta["attribute"],
        "topology": meta["topology"],
        "coherence": meta["coherence"],
        "edge_index": int(meta["edge_index"]),
        "slot_pair": list(meta["slot_pair"]),
        "semantic_pair": list(meta["semantic_pair"]),
        "semantic_names": list(meta["semantic_names"]),
        "alias_index_pair": list(meta["alias_index_pair"]),
        "surface_pair": list(pair),
        "world": world,
        "world_surface": world_surface,
        "surface": surface,
        "split": split,
        "item_id": item["item_id"],
        "state": state,
        "template": template,
        "panel": panel,
        "mapping": target_variant,
        "target_variant": target_variant,
        "query": binding,
        "binding": binding,
        "output_set": output_set,
        "entity_ids": {
            role: item[role]["id"] for role in ("entity0", "entity1", "anchor")
        },
        "entities": entities,
        "selected_entity": selected_entity,
        "anchor_variant": int(item["anchor_variant"]),
        "anchor_value": item["anchor_value"],
        "bound_values": list(bound_values),
        "query_value": query_value,
        "facts": facts,
        "dossier": dossier,
        "question": question,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {
            role: [int(span[0]), int(span[1])] for role, span in role_spans.items()
        },
        "role_positions": {
            role: int(span[1]) for role, span in role_spans.items()
        },
        "semantic_answer_index": semantic_answer,
        "answer_index": semantic_answer,
        "target_answer": target_answer,
        "answer_labels": list(answer_labels),
        "candidate_token_ids": candidate_token_ids,
        "candidate_first_token_ids": {
            key: [int(values[0])] for key, values in candidate_token_ids.items()
        },
        "expected_class": f"a{semantic_answer}",
        "continuation_prefix": prefix,
    }


def signed_pair_records(state_tensor, values, template: int, output_set: int):
    return phase1092.signed_pair_records(state_tensor, values, template, output_set)


def operation_names(
    attribute: str,
    topology: str | None = None,
    coherence: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        operation for operation in OPERATIONS
        if OPERATION_META[operation]["attribute"] == attribute
        and (topology is None or OPERATION_META[operation]["topology"] == topology)
        and (coherence is None or OPERATION_META[operation]["coherence"] == coherence)
    )


def incidence_pairs(topology: str, coherence: str, *, semantic: bool) -> tuple[tuple[int, int], ...]:
    operations = operation_names(PRIMARY_ATTRIBUTE, topology, coherence)
    key = "semantic_pair" if semantic else "slot_pair"
    return tuple(tuple(int(v) for v in OPERATION_META[op][key]) for op in operations)  # type: ignore[return-value]


def _all_aliases(surface: str, split: str, attribute: str) -> tuple[str, ...]:
    return tuple(
        str(value)
        for pair in ALIASES[surface][split][attribute]
        for value in pair
    )


def _all_current_entity_ids() -> list[str]:
    return [
        str(row["id"])
        for world in ENTITY_POOLS.values()
        for split in world.values()
        for row in split
    ]


def _prior_entity_ids() -> set[str]:
    modules = (phase1092, prior)
    return {
        str(row["id"])
        for module in modules
        for world in module.ENTITY_POOLS.values()
        for split in world.values()
        for row in split
    }


def _prior_value_words() -> set[str]:
    words = {
        str(value)
        for surface in phase1092.SURFACES
        for attribute in phase1092.ATTRIBUTES
        for value in phase1092.VALUE_SURFACES[surface][attribute].values()
    }
    words.update(
        str(value)
        for surface in prior.SURFACES
        for split in prior.SPLITS
        for attribute in prior.ATTRIBUTES
        for value in prior.LEXICALIZATIONS[surface][split][attribute]
    )
    return words


def _carrier_normalized(row: dict[str, Any]) -> str:
    text = str(row["raw_prompt"])
    for value in sorted(
        _all_aliases(row["surface"], row["split"], row["attribute"]),
        key=len,
        reverse=True,
    ):
        text = text.replace(value, "<VALUE>")
    return text


def _components(edges: tuple[tuple[int, int], ...]) -> list[int]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    unseen = set(CONCEPT_IDS)
    sizes = []
    while unseen:
        stack = [unseen.pop()]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            for neighbor in neighbors[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(count)
    return sorted(sizes)


def audit_model(model_name: str, tokenizer, cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    checks: dict[str, bool] = {}
    checks["complete_factorial_units"] = all(
        {row["state"] for row in rows} == set(STATES) for rows in by_unit.values()
    )
    checks["all_design_factors_present"] = (
        {row["attribute"] for row in cases} == set(ATTRIBUTES)
        and {row["topology"] for row in cases} == set(TOPOLOGIES)
        and {row["coherence"] for row in cases} == set(COHERENCES)
        and {row["surface"] for row in cases} == set(SURFACES)
        and {row["world"] for row in cases} == set(BASE_WORLDS)
    )
    checks["active_truth_formula"] = all(
        row["semantic_answer_index"]
        == int(int(row["binding"]) != int(row["target_variant"]))
        for row in cases if row["panel"] == "active"
    )
    checks["null_truth_formula"] = all(
        row["semantic_answer_index"]
        == int(int(row["anchor_variant"]) != int(row["target_variant"]))
        for row in cases if row["panel"] == "field_null"
    )
    checks["one_true_one_false_every_side"] = all(
        sorted(
            row["semantic_answer_index"]
            for row in rows
            if row["template"] == template
            and row["panel"] == panel
            and row["binding"] == binding
        ) == [0, 1]
        for rows in by_unit.values()
        for template in TEMPLATE_IDS
        for panel in PANELS
        for binding in (0, 1)
    )
    checks["binding_swap_exact_token_multiset"] = all(
        Counter(next(row for row in rows if row["state"] == f"t{template}_c{panel}_m{target}_q0_w0")["input_ids"])
        == Counter(next(row for row in rows if row["state"] == f"t{template}_c{panel}_m{target}_q1_w0")["input_ids"])
        for rows in by_unit.values()
        for template in TEMPLATE_IDS
        for panel in PANELS
        for target in (0, 1)
    )
    checks["question_fixed_across_binding"] = all(
        next(row for row in rows if row["state"] == f"t{template}_c{panel}_m{target}_q0_w0")["question"]
        == next(row for row in rows if row["state"] == f"t{template}_c{panel}_m{target}_q1_w0")["question"]
        for rows in by_unit.values()
        for template in TEMPLATE_IDS
        for panel in PANELS
        for target in (0, 1)
    )
    checks["role_positions_valid"] = all(
        all(0 <= int(row["role_positions"][role]) < len(row["input_ids"]) for role in CAPTURE_ROLES)
        for row in cases
    )
    checks["query_after_dossier"] = all(
        row["role_positions"]["dossier_end"]
        < row["role_positions"]["query_end"]
        <= row["role_positions"]["answer_boundary"]
        for row in cases
    )
    checks["single_token_outputs"] = all(
        all(len(value) == 1 for value in row["candidate_token_ids"].values())
        for row in cases
    )
    checks["natural_language_surfaces"] = all(
        (
            row["surface"] == "en"
            and ("Answer only Yes or No" in row["raw_prompt"] or "Reply only Yes or No" in row["raw_prompt"])
        ) or (
            row["surface"] == "zh"
            and ("只回答是或否" in row["raw_prompt"] or "只用是或否作答" in row["raw_prompt"])
        )
        for row in cases
    )
    checks["no_translation_instruction"] = all(
        "translate" not in row["raw_prompt"].lower() and "翻译" not in row["raw_prompt"]
        for row in cases
    )
    current_ids = _all_current_entity_ids()
    checks["entity_splits_and_phases_disjoint"] = (
        len(current_ids) == len(set(current_ids))
        and set(current_ids).isdisjoint(_prior_entity_ids())
        and all(
            {row["id"] for row in ENTITY_POOLS[world]["discovery"]}.isdisjoint(
                {row["id"] for row in ENTITY_POOLS[world]["confirmation"]}
            )
            for world in BASE_WORLDS
        )
    )
    all_current_words = {
        value
        for surface in SURFACES
        for split in SPLITS
        for attribute in ATTRIBUTES
        for value in _all_aliases(surface, split, attribute)
    }
    checks["alias_strings_unique_and_phase_heldout"] = (
        len(all_current_words) == 128
        and all_current_words.isdisjoint(_prior_value_words())
        and all(
            set(_all_aliases(surface, "discovery", attribute)).isdisjoint(
                set(_all_aliases(surface, "confirmation", attribute))
            )
            for surface in SURFACES for attribute in ATTRIBUTES
        )
    )
    checks["degree_matched_nonisomorphic_topologies"] = (
        all(
            set(Counter(node for edge in TOPOLOGY_EDGES[topology] for node in edge).values()) == {2}
            for topology in TOPOLOGIES
        )
        and _components(TOPOLOGY_EDGES["cycle8"]) == [8]
        and _components(TOPOLOGY_EDGES["cycle3_5"]) == [3, 5]
    )
    checks["semantic_scramble_is_deranged"] = all(
        OPERATION_META[operation]["slot_pair"] != OPERATION_META[operation]["semantic_pair"]
        for operation in OPERATIONS
        if OPERATION_META[operation]["coherence"] == "scrambled"
        and any(index == 1 for index in OPERATION_META[operation]["alias_index_pair"])
    )
    checks["coherent_semantics_equal_slots"] = all(
        OPERATION_META[operation]["slot_pair"] == OPERATION_META[operation]["semantic_pair"]
        for operation in OPERATIONS
        if OPERATION_META[operation]["coherence"] == "coherent"
    )
    checks["same_alias_multiset_across_coherence"] = all(
        Counter(
            value
            for operation in operation_names(attribute, topology, coherence)
            for value in operation_surface_pair(operation, surface, split)
        ) == Counter(_all_aliases(surface, split, attribute))
        for attribute in ATTRIBUTES
        for topology in TOPOLOGIES
        for coherence in COHERENCES
        for surface in SURFACES
        for split in SPLITS
    )
    checks["same_alias_multiset_across_topology"] = all(
        Counter(
            value
            for operation in operation_names(attribute, topology, coherence)
            for value in operation_surface_pair(operation, surface, split)
        ) == Counter(_all_aliases(surface, split, attribute))
        for attribute in ATTRIBUTES
        for coherence in COHERENCES
        for topology in TOPOLOGIES
        for surface in SURFACES
        for split in SPLITS
    )
    checks["each_alias_string_used_once_per_condition"] = all(
        all(value == 1 for value in Counter(
            value
            for operation in operation_names(attribute, topology, coherence)
            for value in operation_surface_pair(operation, surface, split)
        ).values())
        for attribute in ATTRIBUTES
        for topology in TOPOLOGIES
        for coherence in COHERENCES
        for surface in SURFACES
        for split in SPLITS
    )
    unique_token_diagnostics: dict[str, Any] = {}
    unique_token_ok = True
    for surface in SURFACES:
        for split in SPLITS:
            for attribute in ATTRIBUTES:
                aliases = _all_aliases(surface, split, attribute)
                token_sets = [
                    set(tokenizer.encode(value, add_special_tokens=False))
                    for value in aliases
                ]
                aliases_with_unique_token = sum(
                    bool(tokens - set().union(*[
                        other for j, other in enumerate(token_sets) if j != index
                    ]))
                    for index, tokens in enumerate(token_sets)
                )
                key = f"{surface}__{split}__{attribute}"
                unique_token_diagnostics[key] = {
                    "aliases_with_at_least_one_unique_token": aliases_with_unique_token,
                    "alias_count": len(aliases),
                }
                unique_token_ok &= aliases_with_unique_token >= 12
    checks["at_least_12_of_16_aliases_have_unique_token"] = bool(unique_token_ok)
    carrier_rows: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in cases:
        local_index = str(row["item_id"]).rsplit(".", 1)[-1]
        key = (
            int(row["edge_index"]), row["world"], row["surface"], row["split"],
            local_index, row["state"],
        )
        carrier_rows[key].append(_carrier_normalized(row))
    checks["shared_carrier_across_attribute_topology_coherence"] = all(
        len(rows) == len(ATTRIBUTES) * len(TOPOLOGIES) * len(COHERENCES)
        and len(set(rows)) == 1
        for rows in carrier_rows.values()
    )
    checks["all_checks_boolean"] = all(isinstance(value, bool) for value in checks.values())
    return {
        "schema_version": "phase1094_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "unique_token_diagnostics": unique_token_diagnostics,
        "all_checks_passed": all(checks.values()),
        "answer_token_widths": {
            surface: {
                answer: len(tokenizer.encode(answer, add_special_tokens=False))
                for answer in labels
            }
            for surface, labels in ANSWER_LABELS.items()
        },
        "case_digest": digest(cases),
    }


def build_model_cases(model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    case_index = 0
    for cell in CELLS:
        for split in SPLITS:
            for item in split_items(cell, split):
                for state in STATES:
                    cases.append(build_case(
                        tokenizer, model_name, cell, split, item, state, case_index
                    ))
                    case_index += 1
    return cases, audit_model(model_name, tokenizer, cases)


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    model_case_digests = {}
    model_audits = {}
    for model_name in MODELS:
        cases, audit = build_model_cases(model_name)
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit
        print({
            "phase": PHASE,
            "model": model_name,
            "case_count": len(cases),
            "audit_passed": audit["all_checks_passed"],
        })

    prereg = {
        "schema_version": "phase1094_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "attributes": list(ATTRIBUTES),
        "primary_attribute": PRIMARY_ATTRIBUTE,
        "secondary_attribute": SECONDARY_ATTRIBUTE,
        "topologies": list(TOPOLOGIES),
        "topology_edges": {key: [list(edge) for edge in value] for key, value in TOPOLOGY_EDGES.items()},
        "topology_components": {key: _components(value) for key, value in TOPOLOGY_EDGES.items()},
        "coherences": list(COHERENCES),
        "scramble_shift": SCRAMBLE_SHIFT,
        "operations": list(OPERATIONS),
        "operation_meta": OPERATION_META,
        "surfaces": list(SURFACES),
        "base_worlds": list(BASE_WORLDS),
        "worlds": list(WORLDS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "template_ids": list(TEMPLATE_IDS),
        "output_set_ids": list(OUTPUT_SET_IDS),
        "capture_roles": list(CAPTURE_ROLES),
        "relative_depth_range": [TARGET_RELATIVE_DEPTH_MIN, TARGET_RELATIVE_DEPTH_MAX],
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": "Compare within-model relation Gram and preregistered graph fits, never raw projected vectors.",
        },
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "generation_units_per_cell_split": GENERATION_UNITS_PER_CELL_SPLIT,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1093_summary_digest": (
            read_json(SOURCE_PHASE1093).get("summary_digest")
            if SOURCE_PHASE1093.exists() else None
        ),
        "model_case_digests": model_case_digests,
        "model_audits": model_audits,
        "interpretation_limits": [
            "Synonym assignments are researcher-defined and shades or size adjectives are not exact perceptual equivalences.",
            "Exact alias strings are disjoint, but tokenizer subpieces can overlap; each model audit reports unique-token coverage.",
            "A semantic-incidence fit is evidence of relative synonym-node reuse in this task, not a complete concept code.",
            "A Gram match is rotation tolerant and does not identify shared neurons or a shared raw vector.",
            "Field-null retains the carrier and remains a matched task-route control, not a no-computation state.",
            "All physical maps are descriptive until an independently confirmed candidate later passes causal intervention.",
        ],
        "automatic_next": {
            "hidden_if": "At least two numerically healthy models pass the preregistered behavior conditions.",
            "semantic_map_if": "Run only if P5 or P6 passes prospectively in at least two models.",
            "causal_if": "Never in Phase1094; require P4-P10 and a separate independent replication first.",
            "otherwise": "Keep the generic directed binding skeleton and redesign the semantic orthogonalization without deleting stable descriptive structure.",
        },
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_checks = {
        "all_model_audits_passed": all(row["all_checks_passed"] for row in model_audits.values()),
        "model_order_frozen": tuple(prereg["sequential_model_order"]) == MODELS,
        "fp16_no_quantization": PRECISION == "fp16" and QUANTIZATION == "none",
        "large_case_count": int(prereg["case_count_per_model"]) >= 49152,
        "two_families_two_topologies_two_coherences": (
            len(ATTRIBUTES) == len(TOPOLOGIES) == len(COHERENCES) == 2
        ),
        "four_new_worlds": len(BASE_WORLDS) == 4,
        "complete_language_surfaces": set(SURFACES) == {"en", "zh"},
    }
    global_checks["all_checks_boolean"] = all(isinstance(value, bool) for value in global_checks.values())
    global_audit = {
        "schema_version": "phase1094_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": global_checks,
        "all_checks_passed": all(global_checks.values()),
    }
    global_audit["audit_digest"] = digest(global_audit)
    write_json(protocol_root / "audit.json", global_audit)
    print({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "protocol_digest": prereg["protocol_digest"],
        "audit_passed": global_audit["all_checks_passed"],
    })


if __name__ == "__main__":
    main()
