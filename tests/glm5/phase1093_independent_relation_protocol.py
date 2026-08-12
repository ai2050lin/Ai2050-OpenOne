#!/usr/bin/env python3
"""Freeze Phase1093 independent bilingual relation-geometry replication.

The protocol follows the strongest Phase1092 candidate instead of widening the
atlas.  Size is the primary preregistered family and color is an independent
secondary replication.  Both families share one carrier.  Discovery and
confirmation use disjoint natural synonyms, all entity worlds are new, and
active and field-null panels have exactly matched truth marginals.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_protocol as prior


PHASE = 1093
PROTOCOL_REVISION = 1
MODELS = prior.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
ATTRIBUTES = ("size", "color")
SURFACES = ("en", "zh")
BASE_WORLDS = ("fauna", "flora", "instruments", "transport", "antiquities")
SPLITS = ("discovery", "confirmation")
PANELS = ("active", "field_null")
TEMPLATE_IDS = (0, 1)
OUTPUT_SET_IDS = (0,)
ITEMS_PER_CELL_SPLIT = 6
GENERATION_STEPS = 6
GENERATION_UNITS_PER_CELL_SPLIT = 1
TARGET_RELATIVE_DEPTH_MIN = 0.15
TARGET_RELATIVE_DEPTH_MAX = 0.80
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
SIGNED_PROJECTION_SEED = 1093001
STATES = tuple(
    f"t{template}_c{panel}_m{target}_q{binding}_w0"
    for template in TEMPLATE_IDS
    for panel in PANELS
    for target in (0, 1)
    for binding in (0, 1)
)
FACT_ORDERS = tuple(itertools.permutations(("entity0", "entity1", "anchor")))


# The eight positions preserve only the Phase1092 relation ordering.  The
# physical words are entirely new and differ again between the two splits.
CONCEPT_IDS = tuple(f"v{index}" for index in range(8))
CONCEPT_META = {
    "size": (
        "very_small", "very_large", "low_stature", "high_stature",
        "small_width", "large_width", "small_thickness", "large_thickness",
    ),
    "color": (
        "red_family", "blue_family", "green_family", "yellow_family",
        "black_family", "white_family", "orange_family", "purple_family",
    ),
}
LEXICALIZATIONS = {
    "en": {
        "discovery": {
            "size": (
                "minuscule", "colossal", "squat", "towering",
                "slender", "expansive", "filmy", "chunky",
            ),
            "color": (
                "crimson", "azure", "emerald", "amber",
                "ebony", "ivory", "tangerine", "violet",
            ),
        },
        "confirmation": {
            "size": (
                "diminutive", "enormous", "low-set", "lofty",
                "constricted", "broad", "delicate", "stout",
            ),
            "color": (
                "scarlet", "cobalt", "verdant", "golden",
                "jet-black", "snow-white", "apricot", "lilac",
            ),
        },
    },
    "zh": {
        "discovery": {
            "size": (
                "微小", "庞大", "矮小", "高耸",
                "细窄", "宽大", "轻薄", "粗厚",
            ),
            "color": (
                "绯红", "天蓝", "翠绿", "琥珀黄",
                "墨黑", "象牙白", "橘红", "紫罗兰色",
            ),
        },
        "confirmation": {
            "size": (
                "小巧", "硕大", "低平", "挺拔",
                "窄小", "开阔", "单薄", "厚重",
            ),
            "color": (
                "猩红", "钴蓝", "碧绿", "金黄",
                "乌黑", "雪白", "杏橙色", "淡紫色",
            ),
        },
    },
}


def ring_pairs(attribute: str) -> tuple[tuple[str, str], ...]:
    del attribute
    return tuple(
        (CONCEPT_IDS[index], CONCEPT_IDS[(index + 1) % len(CONCEPT_IDS)])
        for index in range(len(CONCEPT_IDS))
    )


ATTRIBUTE_PAIRS = {attribute: ring_pairs(attribute) for attribute in ATTRIBUTES}
OPERATIONS = tuple(
    f"{attribute}_pair{pair_index:02d}"
    for attribute in ATTRIBUTES
    for pair_index in range(len(CONCEPT_IDS))
)
OPERATION_META = {
    f"{attribute}_pair{pair_index:02d}": {
        "attribute": attribute,
        "pair_index": pair_index,
        "values": ATTRIBUTE_PAIRS[attribute][pair_index],
        "concepts": (
            CONCEPT_META[attribute][pair_index],
            CONCEPT_META[attribute][(pair_index + 1) % len(CONCEPT_IDS)],
        ),
        "phase1092_pair_index": pair_index,
    }
    for attribute in ATTRIBUTES
    for pair_index in range(len(CONCEPT_IDS))
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
    "fauna": {
        "discovery": entries(
            ("otter", "otter", "水獭"), ("badger", "badger", "獾"),
            ("heron", "heron", "苍鹭"), ("yak", "yak", "牦牛"),
            ("gecko", "gecko", "壁虎"), ("falcon", "falcon", "隼"),
            ("beaver", "beaver", "河狸"), ("llama", "llama", "羊驼"),
            ("walrus", "walrus", "海象"), ("cobra", "cobra", "眼镜蛇"),
            ("robin", "robin", "知更鸟"), ("marmot", "marmot", "旱獭"),
        ),
        "confirmation": entries(
            ("dolphin", "dolphin", "海豚"), ("leopard", "leopard", "豹子"),
            ("iguana", "iguana", "鬣蜥"), ("moose", "moose", "驼鹿"),
            ("crane", "crane", "鹤"), ("alpaca", "alpaca", "羊驼兽"),
            ("bison", "bison", "野牛"), ("pelican", "pelican", "鹈鹕"),
            ("jaguar", "jaguar", "美洲豹"), ("ferret", "ferret", "雪貂"),
            ("stork", "stork", "鹳"), ("lynx", "lynx", "猞猁"),
        ),
    },
    "flora": {
        "discovery": entries(
            ("maple", "maple", "枫树"), ("orchid", "orchid", "兰花"),
            ("cactus", "cactus", "仙人掌"), ("cedar", "cedar", "雪松"),
            ("tulip", "tulip", "郁金香"), ("bamboo", "bamboo", "竹子"),
            ("moss", "moss", "苔藓"), ("fern", "fern", "蕨类"),
            ("lotus", "lotus", "荷花"), ("willow", "willow", "柳树"),
            ("peony", "peony", "牡丹"), ("spruce", "spruce", "云杉"),
        ),
        "confirmation": entries(
            ("birch", "birch", "桦树"), ("lily", "lily", "百合"),
            ("agave", "agave", "龙舌兰"), ("cypress", "cypress", "柏树"),
            ("dahlia", "dahlia", "大丽花"), ("reed", "reed", "芦苇"),
            ("lichen", "lichen", "地衣"), ("palm", "palm", "棕榈"),
            ("iris", "iris", "鸢尾"), ("poplar", "poplar", "杨树"),
            ("azalea", "azalea", "杜鹃花"), ("fir", "fir", "冷杉"),
        ),
    },
    "instruments": {
        "discovery": entries(
            ("cello", "cello", "大提琴"), ("oboe", "oboe", "双簧管"),
            ("harp", "harp", "竖琴"), ("drum", "drum", "鼓"),
            ("flute", "flute", "长笛"), ("banjo", "banjo", "班卓琴"),
            ("cymbal", "cymbal", "钹"), ("clarinet", "clarinet", "单簧管"),
            ("trombone", "trombone", "长号"), ("piano", "piano", "钢琴"),
            ("sitar", "sitar", "西塔琴"), ("gong", "gong", "锣"),
        ),
        "confirmation": entries(
            ("bassoon", "bassoon", "巴松管"), ("mandolin", "mandolin", "曼陀林"),
            ("lyre", "lyre", "里拉琴"), ("bongo", "bongo", "邦戈鼓"),
            ("piccolo", "piccolo", "短笛"), ("lute", "lute", "鲁特琴"),
            ("tambourine", "tambourine", "铃鼓"), ("saxophone", "saxophone", "萨克斯管"),
            ("tuba", "tuba", "大号"), ("organ", "organ", "管风琴"),
            ("koto", "koto", "筝"), ("marimba", "marimba", "马林巴琴"),
        ),
    },
    "transport": {
        "discovery": entries(
            ("scooter", "scooter", "踏板车"), ("canoe", "canoe", "独木舟"),
            ("tram", "tram", "有轨电车"), ("tractor", "tractor", "拖拉机"),
            ("ferry", "ferry", "渡轮"), ("bicycle", "bicycle", "自行车"),
            ("glider", "glider", "滑翔机"), ("kayak", "kayak", "皮划艇"),
            ("wagon", "wagon", "四轮车"), ("subway", "subway", "地铁列车"),
            ("van", "van", "厢式车"), ("sled", "sled", "雪橇"),
        ),
        "confirmation": entries(
            ("moped", "moped", "机动脚踏车"), ("raft", "raft", "木筏"),
            ("trolley", "trolley", "无轨电车"), ("bulldozer", "bulldozer", "推土机"),
            ("barge", "barge", "驳船"), ("tricycle", "tricycle", "三轮车"),
            ("balloon", "balloon", "热气球"), ("dinghy", "dinghy", "小艇"),
            ("carriage", "carriage", "马车"), ("monorail", "monorail", "单轨列车"),
            ("minibus", "minibus", "小巴"), ("sleigh", "sleigh", "雪橇车"),
        ),
    },
    "antiquities": {
        "discovery": entries(
            ("astrolabe", "astrolabe", "星盘"), ("amphora", "amphora", "双耳陶罐"),
            ("torc", "torc", "凯尔特颈环"), ("scarab", "scarab", "圣甲虫饰"),
            ("obelisk", "obelisk", "方尖碑"), ("stela", "stela", "石刻碑"),
            ("cartouche", "cartouche", "王名圈饰"), ("krater", "krater", "双耳调酒缸"),
            ("oinochoe", "oinochoe", "单柄酒壶"), ("aryballos", "aryballos", "球形油瓶"),
            ("pyxis", "pyxis", "圆形妆奁"), ("lekythos", "lekythos", "细颈油瓶"),
        ),
        "confirmation": entries(
            ("rhyton", "rhyton", "角形饮器"), ("kantharos", "kantharos", "高柄双耳杯"),
            ("hydria", "hydria", "三柄水罐"), ("ostracon", "ostracon", "陶片文书"),
            ("ushabti", "ushabti", "陪葬俑"), ("palmette", "palmette", "棕叶纹饰"),
            ("labrys", "labrys", "双刃斧"), ("cippus", "cippus", "界碑"),
            ("naos", "naos", "神龛"), ("sistrum", "sistrum", "叉铃"),
            ("menhir", "menhir", "独石碑"), ("dolmen", "dolmen", "石棚墓"),
        ),
    },
}


ANSWER_LABELS = {"en": ("Yes", "No"), "zh": ("是", "否")}
ASSISTANT_PREFILLS = {"en": "\nAnswer:", "zh": "\n答案："}
SHELLS = {
    "en": {
        0: (
            "During an inventory check, a curator wrote three observations. "
            "{dossier} A reviewer then asked: {question} Answer only Yes or No."
        ),
        1: (
            "Three observations were entered in a field log. {dossier} "
            "The final verification question was: {question} Reply only Yes or No."
        ),
    },
    "zh": {
        0: (
            "在一次清点中，记录员写下了三条观察。{dossier}"
            "随后审核员问：{question}只回答是或否。"
        ),
        1: (
            "现场日志中记载了三条观察。{dossier}"
            "最后需要核对的问题是：{question}只用是或否作答。"
        ),
    },
}


OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1093_independent_relation"
SOURCE_PHASE1092 = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1092_natural_bilingual_attribute" / "analysis" / "final_summary.json"
)
EVIDENCE_THRESHOLDS = {
    "minimum_candidate_accuracy": 0.80,
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_generation_accuracy": 0.75,
    "minimum_behavior_worlds_per_pair": 4,
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
    "minimum_heldout_worlds": 4,
    "minimum_cross_model_gram_cosine": 0.50,
    "minimum_cross_model_gram_advantage": 0.10,
    "minimum_cross_phase_gram_cosine": 0.50,
    "minimum_cross_phase_gram_advantage": 0.10,
    "minimum_cross_phase_cell_fraction": 0.75,
    "alignment_ridge": 0.10,
    "minimum_alignment_top1_fraction": 0.75,
    "minimum_alignment_cosine_advantage": 0.10,
    "minimum_alignment_gain": 0.05,
}
PROSPECTIVE_PREDICTIONS = {
    "P1": "All static, lexical holdout, truth, tokenizer, and FP16 protocol audits pass.",
    "P2": "At least two models pass both languages and both attribute families.",
    "P3": "At least two authorized models pass hidden-state and dual-projection audits.",
    "P4": "Disjoint synonym splits preserve within-language pair identity above matched null in both families and two models.",
    "P5": "The Phase1092 size Gram candidate repeats across language and across Qwen3/GLM4 with matched-null advantage.",
    "P6": "Size relation geometry repeats from Phase1092 to Phase1093 in Qwen3 and GLM4 despite new words, carriers, and entities.",
    "P7": "The GLM4 color candidate independently repeats; a second model must also pass color Gram before a two-family invariant is claimed.",
    "P8": "Cross-language size geometry transfers to at least four held-out entity worlds in two models.",
    "P9": "A preregistered low-rank language alignment improves held-out identity without fitting confirmation pairs.",
    "P10": "A descriptive relation band repeats across models or phases; no causal claim follows from location alone.",
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


def surface_value(
    surface: str, split: str, attribute: str, canonical_value: str
) -> str:
    index = CONCEPT_IDS.index(canonical_value)
    return str(LEXICALIZATIONS[surface][split][attribute][index])


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
        role: surface_entity(item[role], surface)
        for role in ("entity0", "entity1", "anchor")
    }
    canonical_fact_values = {
        "entity0": bound_values[0],
        "entity1": bound_values[1],
        "anchor": str(item["anchor_value"]),
    }
    facts = {
        role: fact_text(
            surface,
            entities[role],
            surface_value(surface, split, attribute, canonical_fact_values[role]),
        )
        for role in ("entity0", "entity1", "anchor")
    }
    order = FACT_ORDERS[int(item["fact_order_index"]) % len(FACT_ORDERS)]
    separator = " " if surface == "en" else ""
    dossier = separator.join(facts[role] for role in order)
    query_value = surface_value(
        surface, split, attribute, canonical_values[target_variant]
    )
    question = question_text(surface, selected_entity, query_value)
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
        "query_end": mark_source.mark(raw_prompt, question, occurrence="last"),
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
        "schema_version": "phase1093_case.v1",
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
        "concept_pair": list(OPERATION_META[operation]["concepts"]),
        "phase1092_pair_index": int(OPERATION_META[operation]["phase1092_pair_index"]),
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
    return prior.signed_pair_records(state_tensor, values, template, output_set)


def _prior_entity_ids() -> set[str]:
    return {
        str(row["id"])
        for world in prior.ENTITY_POOLS.values()
        for split in world.values()
        for row in split
    }


def _all_current_entity_ids() -> list[str]:
    return [
        str(row["id"])
        for world in ENTITY_POOLS.values()
        for split in world.values()
        for row in split
    ]


def _lexical_sets(surface: str, split: str) -> set[str]:
    return {
        value
        for attribute in ATTRIBUTES
        for value in LEXICALIZATIONS[surface][split][attribute]
    }


def _carrier_normalized(row: dict[str, Any]) -> str:
    text = str(row["raw_prompt"])
    for attribute in ATTRIBUTES:
        for value in LEXICALIZATIONS[row["surface"]][row["split"]][attribute]:
            text = text.replace(value, "<VALUE>")
    return text


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
        Counter(next(row for row in rows if row["state"] == (
            f"t{template}_c{panel}_m{target}_q0_w0"
        ))["input_ids"])
        == Counter(next(row for row in rows if row["state"] == (
            f"t{template}_c{panel}_m{target}_q1_w0"
        ))["input_ids"])
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
        "Answer only Yes or No" in row["raw_prompt"]
        or "Reply only Yes or No" in row["raw_prompt"]
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
    current_ids = _all_current_entity_ids()
    checks["all_entity_ids_unique"] = len(current_ids) == len(set(current_ids))
    checks["entities_held_out_from_phase1092"] = set(current_ids).isdisjoint(
        _prior_entity_ids()
    )
    checks["balanced_anchor_orientation"] = all(
        Counter(
            row["anchor_variant"] for row in cases
            if row["cell"] == cell and row["split"] == split
            and row["state"] == "t0_cactive_m0_q0_w0"
        ) == Counter({0: 3, 1: 3})
        for cell in CELLS for split in SPLITS
    )
    checks["balanced_pair_rings"] = all(
        all(value == 2 for value in Counter(
            item for pair in ATTRIBUTE_PAIRS[attribute] for item in pair
        ).values())
        for attribute in ATTRIBUTES
    )
    checks["discovery_confirmation_words_disjoint"] = all(
        _lexical_sets(surface, "discovery").isdisjoint(
            _lexical_sets(surface, "confirmation")
        ) for surface in SURFACES
    )
    prior_words = {
        str(word)
        for surface in SURFACES
        for attribute in ATTRIBUTES
        for word in prior.VALUE_SURFACES[surface][attribute].values()
    }
    current_words = {
        word for surface in SURFACES for split in SPLITS
        for word in _lexical_sets(surface, split)
    }
    checks["attribute_words_held_out_from_phase1092"] = current_words.isdisjoint(
        prior_words
    )
    carrier_rows: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for row in cases:
        local_index = str(row["item_id"]).rsplit(".", 1)[-1]
        key = (
            int(row["pair_index"]), row["world"], row["surface"],
            row["split"], local_index, row["state"],
        )
        carrier_rows[key][str(row["attribute"])] = _carrier_normalized(row)
    checks["shared_carrier_across_attributes"] = all(
        set(rows) == set(ATTRIBUTES)
        and len(set(rows.values())) == 1
        for rows in carrier_rows.values()
    )
    checks["all_checks_boolean"] = all(
        isinstance(value, bool) for value in checks.values()
    )
    return {
        "schema_version": "phase1093_protocol_model_audit.v1",
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
        "schema_version": "phase1093_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "attributes": list(ATTRIBUTES),
        "primary_attribute": "size",
        "secondary_attribute": "color",
        "operations": list(OPERATIONS),
        "operation_meta": OPERATION_META,
        "attribute_pairs": {
            key: [list(pair) for pair in value]
            for key, value in ATTRIBUTE_PAIRS.items()
        },
        "concept_meta": {key: list(value) for key, value in CONCEPT_META.items()},
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
        "cross_phase_normalized_band": [0.45, 0.62],
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": "Compare only within-model pair Gram geometry.",
        },
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": len(CELLS) * len(SPLITS)
        * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "generation_steps": GENERATION_STEPS,
        "generation_units_per_cell_split": GENERATION_UNITS_PER_CELL_SPLIT,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source_phase1092_summary_digest": (
            read_json(SOURCE_PHASE1092)["summary_digest"]
            if SOURCE_PHASE1092.exists() else None
        ),
        "model_case_digests": model_case_digests,
        "model_audits": model_audits,
        "interpretation_limits": [
            "The synonym correspondence is researcher-defined and may be imperfect.",
            "Shade words test color-family relations, not exact perceptual equivalence.",
            "A Gram match is rotation tolerant and does not identify a shared vector.",
            "Field-null retains the carrier and is not a no-computation baseline.",
            "The signed atlas is descriptive and never establishes causality alone.",
        ],
        "automatic_next": {
            "hidden_if": "At least two healthy FP16 models pass both attributes in both languages.",
            "alignment_if": "Run offline only after the primary size Gram gate passes.",
            "causal_if": "Only after P1-P9 pass; never from a descriptive hotspot alone.",
            "otherwise": "Stop at the failed preregistered gate and retain the boundary result.",
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
        "case_count_larger_than_phase1092": int(prereg["case_count_per_model"]) > 24576,
        "two_preregistered_families": ATTRIBUTES == ("size", "color"),
        "five_new_worlds": len(BASE_WORLDS) == 5,
        "complete_language_surfaces": set(SURFACES) == {"en", "zh"},
        "all_checks_boolean": True,
    }
    global_checks["all_checks_boolean"] = all(
        isinstance(value, bool) for value in global_checks.values()
    )
    global_audit = {
        "schema_version": "phase1093_protocol_audit.v1",
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
