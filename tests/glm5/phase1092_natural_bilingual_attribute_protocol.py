#!/usr/bin/env python3
"""Freeze Phase1092 natural English/Chinese attribute-binding protocol.

The phase compares complete monolingual narratives rather than inserting
translated value words into one fixed English shell.  Color is the target
family; size and material are matched non-color attribute families.  Every
binding side is truth-balanced in both active and field-null panels.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1089_truth_matched_color_binding_protocol as base


PHASE = 1092
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
ATTRIBUTES = ("color", "size", "material")
SURFACES = ("en", "zh")
BASE_WORLDS = ("household", "workshop", "ornaments", "rare_artifacts")
SPLITS = ("discovery", "confirmation")
PANELS = ("active", "field_null")
TEMPLATE_IDS = (0, 1)
OUTPUT_SET_IDS = (0,)
ITEMS_PER_CELL_SPLIT = 4
GENERATION_STEPS = 6
GENERATION_UNITS_PER_CELL_SPLIT = 1
TARGET_RELATIVE_DEPTH_MIN = 0.20
TARGET_RELATIVE_DEPTH_MAX = 0.60
CAPTURE_ROLES = (
    "entity0_fact_end",
    "entity1_fact_end",
    "dossier_end",
    "query_end",
    "answer_boundary",
)
SIGNED_FIELDS = ("active_binding", "field_null", "content")
SIGNED_PROJECTION_DIM = 96
SIGNED_PROJECTION_REPLICATES = 2
SIGNED_PROJECTION_SEED = 1092001
STATES = tuple(
    f"t{template}_c{panel}_m{target}_q{binding}_w0"
    for template in TEMPLATE_IDS
    for panel in PANELS
    for target in (0, 1)
    for binding in (0, 1)
)
FACT_ORDERS = tuple(itertools.permutations(("entity0", "entity1", "anchor")))


ATTRIBUTE_VALUES = {
    "color": (
        "red", "blue", "green", "yellow",
        "black", "white", "orange", "purple",
    ),
    "size": (
        "tiny", "huge", "short", "tall",
        "narrow", "wide", "thin", "thick",
    ),
    "material": (
        "wood", "metal", "glass", "stone",
        "paper", "plastic", "ceramic", "leather",
    ),
}
VALUE_SURFACES = {
    "en": {
        "color": {
            "red": "red", "blue": "blue", "green": "green",
            "yellow": "yellow", "black": "black", "white": "white",
            "orange": "orange", "purple": "purple",
        },
        "size": {
            "tiny": "tiny", "huge": "huge", "short": "short",
            "tall": "tall", "narrow": "narrow", "wide": "wide",
            "thin": "thin", "thick": "thick",
        },
        "material": {
            "wood": "wood", "metal": "metal", "glass": "glass",
            "stone": "stone", "paper": "paper", "plastic": "plastic",
            "ceramic": "ceramic", "leather": "leather",
        },
    },
    "zh": {
        "color": {
            "red": "红色", "blue": "蓝色", "green": "绿色",
            "yellow": "黄色", "black": "黑色", "white": "白色",
            "orange": "橙色", "purple": "紫色",
        },
        "size": {
            "tiny": "微型", "huge": "巨型", "short": "低矮",
            "tall": "高挑", "narrow": "狭窄", "wide": "宽阔",
            "thin": "纤薄", "thick": "厚实",
        },
        "material": {
            "wood": "木材", "metal": "金属", "glass": "玻璃",
            "stone": "石材", "paper": "纸张", "plastic": "塑料",
            "ceramic": "陶瓷", "leather": "皮革",
        },
    },
}


def ring_pairs(attribute: str) -> tuple[tuple[str, str], ...]:
    values = ATTRIBUTE_VALUES[attribute]
    return tuple(
        (values[index], values[(index + 1) % len(values)])
        for index in range(len(values))
    )


ATTRIBUTE_PAIRS = {
    attribute: ring_pairs(attribute) for attribute in ATTRIBUTES
}
OPERATIONS = tuple(
    f"{attribute}_{left}_{right}"
    for attribute in ATTRIBUTES
    for left, right in ATTRIBUTE_PAIRS[attribute]
)
OPERATION_META = {
    f"{attribute}_{left}_{right}": {
        "attribute": attribute,
        "pair_index": pair_index,
        "values": (left, right),
    }
    for attribute in ATTRIBUTES
    for pair_index, (left, right) in enumerate(ATTRIBUTE_PAIRS[attribute])
}
WORLDS = tuple(
    f"{world}@{surface}" for world in BASE_WORLDS for surface in SURFACES
)
CELLS = tuple(
    f"{operation}__{world_surface}"
    for operation in OPERATIONS
    for world_surface in WORLDS
)
FAMILIES = CELLS


def entries(*rows: tuple[str, str, str]) -> tuple[dict[str, str], ...]:
    return tuple({"id": key, "en": en, "zh": zh} for key, en, zh in rows)


ENTITY_POOLS = {
    "household": {
        "discovery": entries(
            ("lantern", "lantern", "灯笼"),
            ("kettle", "kettle", "水壶"),
            ("mirror", "mirror", "镜子"),
            ("basket", "basket", "篮子"),
            ("helmet", "helmet", "头盔"),
            ("pillow", "pillow", "枕头"),
            ("bottle", "bottle", "瓶子"),
            ("carpet", "carpet", "地毯"),
            ("violin", "violin", "小提琴"),
        ),
        "confirmation": entries(
            ("hammer", "hammer", "锤子"),
            ("teapot", "teapot", "茶壶"),
            ("compass", "compass", "指南针"),
            ("jacket", "jacket", "夹克"),
            ("ladder", "ladder", "梯子"),
            ("camera", "camera", "相机"),
            ("wallet", "wallet", "钱包"),
            ("anchor", "anchor", "锚"),
            ("trumpet", "trumpet", "小号"),
        ),
    },
    "workshop": {
        "discovery": entries(
            ("chisel", "chisel", "凿子"),
            ("mallet", "mallet", "木槌"),
            ("clamp", "clamp", "夹具"),
            ("anvil", "anvil", "铁砧"),
            ("handsaw", "handsaw", "手锯"),
            ("drill", "drill", "电钻"),
            ("plane", "plane", "刨子"),
            ("wrench", "wrench", "扳手"),
            ("file", "file", "锉刀"),
        ),
        "confirmation": entries(
            ("lathe", "lathe", "车床"),
            ("spindle", "spindle", "主轴"),
            ("caliper", "caliper", "卡尺"),
            ("vise", "vise", "台钳"),
            ("router", "router", "铣刀"),
            ("grinder", "grinder", "砂轮机"),
            ("gauge", "gauge", "量规"),
            ("trowel", "trowel", "抹子"),
            ("auger", "auger", "螺旋钻"),
        ),
    },
    "ornaments": {
        "discovery": entries(
            ("pendant", "pendant", "吊坠"),
            ("brooch", "brooch", "胸针"),
            ("figurine", "figurine", "小雕像"),
            ("medallion", "medallion", "纪念章"),
            ("mask", "mask", "面具"),
            ("goblet", "goblet", "高脚杯"),
            ("casket", "casket", "首饰盒"),
            ("plaque", "plaque", "饰板"),
            ("bead", "bead", "珠子"),
        ),
        "confirmation": entries(
            ("reliquary", "reliquary", "圣物盒"),
            ("cameo", "cameo", "浮雕饰品"),
            ("amulet", "amulet", "护符"),
            ("censer", "censer", "香炉"),
            ("diadem", "diadem", "冠饰"),
            ("statuette", "statuette", "雕像"),
            ("chalice", "chalice", "圣杯"),
            ("tripod", "tripod vessel", "三足器"),
            ("urn", "urn", "瓮"),
        ),
    },
    "rare_artifacts": {
        "discovery": entries(
            ("taotie_vessel", "taotie vessel", "饕餮纹器"),
            ("qilin_figurine", "qilin figurine", "麒麟雕像"),
            ("pixiu_pendant", "pixiu pendant", "貔貅吊坠"),
            ("tengu_mask", "tengu mask", "天狗面具"),
            ("djinn_lamp", "djinn lamp", "精灵灯"),
            ("sphinx_statuette", "sphinx statuette", "狮身人面像"),
            ("kraken_carving", "kraken carving", "海怪雕刻"),
            ("gryphon_brooch", "gryphon brooch", "狮鹫胸针"),
            ("basilisk_plaque", "basilisk plaque", "蛇怪饰板"),
        ),
        "confirmation": entries(
            ("chimera_urn", "chimera urn", "奇美拉瓮"),
            ("golem_tablet", "golem tablet", "魔像石板"),
            ("kelpie_charm", "kelpie charm", "水马护符"),
            ("banshee_bell", "banshee bell", "女妖铃"),
            ("manticore_medallion", "manticore medallion", "蝎尾狮徽章"),
            ("phoenix_censer", "phoenix censer", "凤凰香炉"),
            ("dragon_reliquary", "dragon reliquary", "龙纹圣物盒"),
            ("unicorn_goblet", "unicorn goblet", "独角兽杯"),
            ("selkie_cameo", "selkie cameo", "海豹女浮雕"),
        ),
    },
}


ANSWER_LABELS = {
    "en": ("Yes", "No"),
    "zh": ("是", "否"),
}
ASSISTANT_PREFILLS = {"en": "\nAnswer:", "zh": "\n答案："}
SHELLS = {
    "en": {
        0: (
            "A conservator recorded three objects before an exhibition. "
            "{dossier} Later, a colleague asked: {question} "
            "Answer only Yes or No."
        ),
        1: (
            "An archive note described several objects. {dossier} "
            "To verify one detail, answer this question: {question} "
            "Reply with only Yes or No."
        ),
    },
    "zh": {
        0: (
            "一位保管员在展览前记录了三件物品。{dossier}"
            "随后，同事问：{question}只回答是或否。"
        ),
        1: (
            "一份档案描述了几件物品。{dossier}"
            "为核对其中一个细节，请回答：{question}只用是或否作答。"
        ),
    },
}


OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1092_natural_bilingual_attribute"
)
SOURCE_PHASE1091 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1091_cross_surface_color_signed" / "analysis" / "final_summary.json"
)


EVIDENCE_THRESHOLDS = {
    "minimum_candidate_accuracy": 0.80,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_generation_accuracy": 0.75,
    "minimum_behavior_worlds_per_pair": 3,
    "minimum_behavior_pairs_per_attribute": 6,
    "minimum_behavior_attributes": 2,
    "minimum_behavior_models": 2,
    "maximum_projection_median_abs_norm_error": 0.08,
    "maximum_projection_p95_abs_norm_error": 0.20,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_query_tolerance": 1e-8,
    "minimum_pair_top1": 6,
    "permutation_p_max": 0.01,
    "minimum_content_identity_advantage": 0.10,
    "minimum_cross_language_gram_cosine": 0.50,
    "minimum_cross_language_gram_advantage": 0.10,
    "minimum_cross_language_attributes": 2,
    "minimum_cross_language_models": 2,
    "minimum_heldout_worlds": 3,
    "minimum_cross_model_gram_cosine": 0.50,
    "minimum_cross_model_gram_advantage": 0.10,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "All static audits pass, including complete monolingual English and "
        "Chinese shells, exact binding token multisets, truth matching, and "
        "independent entity splits."
    ),
    "P2": (
        "At least two FP16 models pass both languages for at least six of "
        "eight pairs in three worlds for at least two attribute families."
    ),
    "P3": "Both signed sketches and finite hidden-state audits pass in two models.",
    "P4": (
        "Within each language, centered pair identity repeats across independent "
        "splits for at least two attributes in two models."
    ),
    "P5": (
        "English-to-Chinese and Chinese-to-English pair identity each retrieve "
        "six of eight pairs and beat matched field-null by 0.10 for at least "
        "two attributes in two models."
    ),
    "P6": (
        "Cross-language pair Gram reaches 0.50 and beats matched field-null by "
        "0.10 in both directions and sketches for two attributes in two models."
    ),
    "P7": (
        "Cross-language geometry transfers into three held-out entity worlds "
        "for at least two attributes in two models."
    ),
    "P8": (
        "At least two directed healthy model pairs preserve cross-language "
        "attribute-pair geometry above matched null."
    ),
    "P9": (
        "A repeatable descriptive bilingual attribute map appears in the "
        "preregistered 0.20-0.60 range, without authorizing causal localization."
    ),
}


write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest
tokenizer_for = base.tokenizer_for
offset_token_spans = base.offset_token_spans
behavior = base.behavior
mark_source = base.mark_source


def split_world(value: str) -> tuple[str, str]:
    return tuple(value.split("@", 1))  # type: ignore[return-value]


def split_cell(cell: str) -> tuple[str, str]:
    return tuple(cell.split("__", 1))  # type: ignore[return-value]


def state_factors(state: str) -> tuple[int, str, int, int, int]:
    for template in TEMPLATE_IDS:
        for panel in PANELS:
            prefix = f"t{template}_c{panel}_m"
            if state.startswith(prefix):
                target, remainder = state[len(prefix):].split("_q", 1)
                binding, output_set = remainder.split("_w", 1)
                return template, panel, int(target), int(binding), int(output_set)
    raise ValueError(f"invalid state: {state}")


def operation_values(operation: str) -> tuple[str, str]:
    return tuple(OPERATION_META[operation]["values"])  # type: ignore[return-value]


def surface_entity(entity: dict[str, str], surface: str) -> str:
    return str(entity[surface])


def fact_text(attribute: str, surface: str, entity: str, value: str) -> str:
    if surface == "en":
        if attribute == "color":
            return f"The {entity} appeared {value}."
        if attribute == "size":
            return f"The {entity} was {value} in size."
        return f"The {entity} was made of {value}."
    if attribute == "color":
        return f"{entity}呈现为{value}。"
    if attribute == "size":
        return f"{entity}的尺寸为{value}。"
    return f"{entity}由{value}制成。"


def question_text(attribute: str, surface: str, entity: str, value: str) -> str:
    if surface == "en":
        if attribute == "color":
            return f"Did the {entity} appear {value}?"
        if attribute == "size":
            return f"Was the {entity} {value} in size?"
        return f"Was the {entity} made of {value}?"
    if attribute == "color":
        return f"{entity}是否呈现为{value}？"
    if attribute == "size":
        return f"{entity}的尺寸是否为{value}？"
    return f"{entity}是否由{value}制成？"


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    operation, world_surface = split_cell(cell)
    world, surface = split_world(world_surface)
    value0, value1 = operation_values(operation)
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
            "anchor_value": (value0, value1)[anchor_variant],
            "fact_order_index": local_index,
            "operation": operation,
            "attribute": OPERATION_META[operation]["attribute"],
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
    attribute = str(OPERATION_META[operation]["attribute"])
    canonical_values = operation_values(operation)
    bound_values = canonical_values if binding == 0 else tuple(reversed(canonical_values))
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
        "entity0": surface_entity(item["entity0"], surface),
        "entity1": surface_entity(item["entity1"], surface),
        "anchor": surface_entity(item["anchor"], surface),
    }
    canonical_fact_values = {
        "entity0": bound_values[0],
        "entity1": bound_values[1],
        "anchor": str(item["anchor_value"]),
    }
    facts = {
        role: fact_text(
            attribute,
            surface,
            entities[role],
            VALUE_SURFACES[surface][attribute][canonical_fact_values[role]],
        )
        for role in ("entity0", "entity1", "anchor")
    }
    order = FACT_ORDERS[int(item["fact_order_index"]) % len(FACT_ORDERS)]
    separator = " " if surface == "en" else ""
    dossier = separator.join(facts[role] for role in order)
    query_value = VALUE_SURFACES[surface][attribute][
        canonical_values[target_variant]
    ]
    question = question_text(
        attribute, surface, selected_entity, query_value
    )
    raw_prompt = SHELLS[surface][template].format(
        dossier=dossier, question=question
    )
    raw_spans = {
        "entity0_fact_end": mark_source.mark(
            raw_prompt, facts["entity0"], occurrence="first"
        ),
        "entity1_fact_end": mark_source.mark(
            raw_prompt, facts["entity1"], occurrence="first"
        ),
        "dossier_end": mark_source.mark(
            raw_prompt, facts[order[-1]], occurrence="first"
        ),
        "query_end": mark_source.mark(
            raw_prompt, question, occurrence="last"
        ),
    }
    rendered = behavior.render_native(
        tokenizer, model_name, raw_prompt, with_system=False
    ) + ASSISTANT_PREFILLS[surface]
    input_ids = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    role_spans = offset_token_spans(
        tokenizer, rendered, raw_prompt, raw_spans
    )
    role_spans["answer_boundary"] = (len(input_ids) - 1, len(input_ids) - 1)
    prefix = " " if surface == "en" else ""
    candidate_token_ids = {
        f"a{index}": behavior.continuation_ids(
            tokenizer, rendered, prefix, answer
        )
        for index, answer in enumerate(answer_labels)
    }
    return {
        "schema_version": "phase1092_case.v1",
        "phase": PHASE,
        "model": model_name,
        "case_index": case_index,
        "record_id": f"{model_name}.{cell}.{split}.{item['item_id']}.{state}",
        "unit_id": f"{cell}.{split}.{item['item_id']}",
        "family": cell,
        "cell": cell,
        "operation": operation,
        "attribute": attribute,
        "pair_index": int(OPERATION_META[operation]["pair_index"]),
        "canonical_pair": list(canonical_values),
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
        "query_value": canonical_values[target_variant],
        "surface_query_value": query_value,
        "facts": facts,
        "dossier": dossier,
        "question": question,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": {
            role: [int(span[0]), int(span[1])]
            for role, span in role_spans.items()
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
    return base.signed_pair_records(
        state_tensor, values, template, output_set
    )


def audit_model(
    model_name: str, tokenizer, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    checks: dict[str, bool] = {}
    checks["complete_factorial_units"] = all(
        {row["state"] for row in rows} == set(STATES)
        for rows in by_unit.values()
    )
    checks["all_attributes_present"] = {
        row["attribute"] for row in cases
    } == set(ATTRIBUTES)
    checks["all_surfaces_present"] = {
        row["surface"] for row in cases
    } == set(SURFACES)
    checks["all_worlds_present"] = {
        row["world"] for row in cases
    } == set(BASE_WORLDS)
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
        Counter(
            next(row for row in rows if row["state"] == (
                f"t{template}_c{panel}_m{target}_q0_w0"
            ))["input_ids"]
        ) == Counter(
            next(row for row in rows if row["state"] == (
                f"t{template}_c{panel}_m{target}_q1_w0"
            ))["input_ids"]
        )
        for rows in by_unit.values()
        for template in TEMPLATE_IDS
        for panel in PANELS
        for target in (0, 1)
    )
    checks["question_fixed_across_binding"] = all(
        next(row for row in rows if row["state"] == (
            f"t{template}_c{panel}_m{target}_q0_w0"
        ))["question"]
        == next(row for row in rows if row["state"] == (
            f"t{template}_c{panel}_m{target}_q1_w0"
        ))["question"]
        for rows in by_unit.values()
        for template in TEMPLATE_IDS
        for panel in PANELS
        for target in (0, 1)
    )
    checks["role_positions_valid"] = all(
        all(
            0 <= int(row["role_positions"][role]) < len(row["input_ids"])
            for role in CAPTURE_ROLES
        ) for row in cases
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
    checks["full_chinese_surface"] = all(
        "只回答是或否" in row["raw_prompt"]
        or "只用是或否作答" in row["raw_prompt"]
        for row in cases if row["surface"] == "zh"
    )
    checks["full_english_surface"] = all(
        ("Answer only Yes or No" in row["raw_prompt"]
         or "Reply with only Yes or No" in row["raw_prompt"])
        for row in cases if row["surface"] == "en"
    )
    checks["no_explicit_translation_instruction"] = all(
        "translate" not in row["raw_prompt"].lower()
        and "翻译" not in row["raw_prompt"]
        for row in cases
    )
    checks["independent_entity_splits"] = all(
        {row["id"] for row in ENTITY_POOLS[world]["discovery"]}.isdisjoint(
            {row["id"] for row in ENTITY_POOLS[world]["confirmation"]}
        ) for world in BASE_WORLDS
    )
    checks["balanced_anchor_orientation"] = all(
        Counter(
            row["anchor_variant"] for row in cases
            if row["cell"] == cell and row["split"] == split
            and row["state"] == "t0_cactive_m0_q0_w0"
        ) == Counter({0: 2, 1: 2})
        for cell in CELLS for split in SPLITS
    )
    checks["balanced_pair_rings"] = all(
        all(value == 2 for value in Counter(
            item for pair in ATTRIBUTE_PAIRS[attribute] for item in pair
        ).values())
        for attribute in ATTRIBUTES
    )
    checks["all_checks_boolean"] = all(
        isinstance(value, bool) for value in checks.values()
    )
    return {
        "schema_version": "phase1092_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
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


def build_model_cases(
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    case_index = 0
    for cell in CELLS:
        for split in SPLITS:
            for item in split_items(cell, split):
                for state in STATES:
                    cases.append(build_case(
                        tokenizer, model_name, cell, split,
                        item, state, case_index,
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
        "schema_version": "phase1092_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "attributes": list(ATTRIBUTES),
        "operations": list(OPERATIONS),
        "attribute_pairs": {
            key: [list(pair) for pair in value]
            for key, value in ATTRIBUTE_PAIRS.items()
        },
        "surfaces": list(SURFACES),
        "base_worlds": list(BASE_WORLDS),
        "worlds": list(WORLDS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "template_ids": list(TEMPLATE_IDS),
        "output_set_ids": list(OUTPUT_SET_IDS),
        "capture_roles": list(CAPTURE_ROLES),
        "relative_depth_range": [
            TARGET_RELATIVE_DEPTH_MIN, TARGET_RELATIVE_DEPTH_MAX
        ],
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": (
                "Compare only within-model pair Gram geometry."
            ),
        },
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": len(CELLS) * len(SPLITS)
        * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS)
        * ITEMS_PER_CELL_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "generation_units_per_cell_split": GENERATION_UNITS_PER_CELL_SPLIT,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1091_summary_digest": (
            read_json(SOURCE_PHASE1091)["summary_digest"]
            if SOURCE_PHASE1091.exists() else None
        ),
        "model_case_digests": model_case_digests,
        "model_audits": model_audits,
        "interpretation_limits": [
            "Natural narratives remain controlled synthetic prompts, not corpus samples.",
            "Cross-language similarity can still include generic bilingual task routing.",
            "Size and material are comparison families, not pure no-computation controls.",
            "A descriptive map never authorizes components or neuron causality.",
        ],
        "automatic_next": {
            "hidden_if": (
                "At least two numerically healthy models pass both languages "
                "for two attributes."
            ),
            "causal_if": "Never from Phase1092 alone.",
            "otherwise": "Stop after behavior and retain the protocol failure.",
        },
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_checks = {
        "all_model_audits_passed": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
        "model_order_frozen": tuple(prereg["sequential_model_order"]) == MODELS,
        "fp16_no_quantization": PRECISION == "fp16" and QUANTIZATION == "none",
        "case_count_large": int(prereg["case_count_per_model"]) >= 24000,
        "three_attribute_families": len(ATTRIBUTES) == 3,
        "complete_language_surfaces": set(SURFACES) == {"en", "zh"},
        "all_checks_boolean": True,
    }
    global_checks["all_checks_boolean"] = all(
        isinstance(value, bool) for value in global_checks.values()
    )
    global_audit = {
        "schema_version": "phase1092_protocol_audit.v1",
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
