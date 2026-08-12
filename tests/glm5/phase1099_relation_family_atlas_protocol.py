#!/usr/bin/env python3
"""Freeze the Phase1099 relation-family held-out atlas protocol."""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1098_relative_relation_geometry_protocol as base


PHASE = 1099
PROTOCOL_REVISION = 4
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
FAMILIES = (
    "physical_magnitude",
    "temporal_order",
    "spatial_order",
    "social_status",
    "epistemic_causal",
)
SURFACES = ("en", "zh")
SPLITS = ("discovery", "confirmation")
RELATION_SPLITS = ("discovery", "confirmation")
TEMPLATES = (0, 1, 2, 3)
TEMPLATES_BY_SPLIT = {"discovery": (0, 1), "confirmation": (2, 3)}
PANELS = base.PANELS
TASKS = base.TASKS
ORIENTATIONS = base.ORIENTATIONS
CARRIER_ORDERS = base.CARRIER_ORDERS
ITEMS_PER_TEMPLATE = 4
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
CONTINUATION_PREFIX = base.CONTINUATION_PREFIX
GENERATION_STEPS = 6
GENERATION_ITEMS_PER_CELL = 2
CAPTURE_ROLES = base.CAPTURE_ROLES
PRE_TASK_ROLES = base.PRE_TASK_ROLES
FIELDS = base.FIELDS
PRIMARY_FIELD = base.PRIMARY_FIELD
CONTROL_FIELDS = base.CONTROL_FIELDS
STATES = base.STATES
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1099_relation_family_atlas"
PHASE1098_ROOT = base.OUT_ROOT
SOURCE_PHASE1098 = PHASE1098_ROOT / "analysis" / "final_summary.json"
SOURCE_PHASE1098_AUDIT = PHASE1098_ROOT / "audit" / "result_audit.json"


write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest


EVIDENCE_THRESHOLDS = {
    "minimum_candidate_finite_fraction": 0.95,
    "minimum_candidate_accuracy": 0.80,
    "minimum_generation_accuracy": 0.65,
    "minimum_relations_per_family": 4,
    "minimum_relation_split_relations_per_family": 2,
    "minimum_families_per_model": 5,
    "minimum_behavior_models": 2,
    "minimum_hidden_finite_fraction": 0.97,
    "pre_task_tolerance": 1e-8,
    "minimum_family_geometry_cosine": 0.70,
    "minimum_family_permutation_margin": 0.02,
    "minimum_field_specificity_advantage": 0.05,
    "minimum_split_records": 6,
    "minimum_cross_language_fraction": 0.75,
    "minimum_cross_model_fraction": 0.75,
    "minimum_cross_model_pairs": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": "All source, token, factorial, family-balance, relation-holdout, and name-holdout audits pass.",
    "P2": "At least two models pass all five families, with at least four of six relations and two of three relations per holdout half in every family.",
    "P3": "At least two behavior-authorized models pass hidden finiteness, duplicate identity, and exact pre-task-zero audits.",
    "P4": "The five-family relational-execution graph transfers from three discovery relations to three unseen confirmation relations and across independent templates in at least six of eight directional cells in two models.",
    "P5": "The same family graph transfers across English and Chinese in at least 75 percent of preregistered cells in two models.",
    "P6": "The family graph transfers in both directions across at least two model pairs in at least 75 percent of cells, while selecting relational execution over every matched field control.",
    "P7": "All primary family signatures exclude candidate logits, output margins, generated scores, PCA, and learned probes.",
    "P8": "A failed family gate stops component selection and causal intervention; descriptive hotspots cannot upgrade evidence.",
}


def _relation(
    family: str,
    relation_split: str,
    noun_en: str,
    direct_en: str,
    high_en: str,
    low_en: str,
    noun_zh: str,
    direct_zh: str,
    high_zh: str,
    low_zh: str,
    q_high_en: str | None = None,
    q_low_en: str | None = None,
) -> dict[str, str]:
    return {
        "family": family,
        "relation_split": relation_split,
        "noun_en": noun_en,
        "direct_en": direct_en,
        "high_en": high_en,
        "low_en": low_en,
        "noun_zh": noun_zh,
        "direct_zh": direct_zh,
        "high_zh": high_zh,
        "low_zh": low_zh,
        "q_high_en": q_high_en or f"is {high_en}",
        "q_low_en": q_low_en or f"is {low_en}",
    }


RELATION_DEFINITIONS = {
    "height": _relation("physical_magnitude", "discovery", "height", "is taller than", "taller", "shorter", "身高", "比 {right} 更高", "更高", "更矮"),
    "weight": _relation("physical_magnitude", "discovery", "weight", "is heavier than", "heavier", "lighter", "重量", "比 {right} 更重", "更重", "更轻"),
    "speed": _relation("physical_magnitude", "discovery", "speed", "is faster than", "faster", "slower", "速度", "比 {right} 更快", "更快", "更慢"),
    "brightness": _relation("physical_magnitude", "confirmation", "brightness", "is brighter than", "brighter", "dimmer", "亮度", "比 {right} 更亮", "更亮", "更暗"),
    "temperature": _relation("physical_magnitude", "confirmation", "temperature", "is warmer than", "warmer", "cooler", "温度", "比 {right} 更热", "更热", "更冷"),
    "price": _relation("physical_magnitude", "confirmation", "price", "is more expensive than", "more expensive", "less expensive", "价格", "比 {right} 更贵", "更贵", "更便宜"),

    "arrival_time": _relation("temporal_order", "discovery", "arrival time", "arrived earlier than", "earlier to arrive", "later to arrive", "到达时间", "比 {right} 更早到达", "更早到达", "更晚到达", "arrived earlier", "arrived later"),
    "departure_time": _relation("temporal_order", "discovery", "departure time", "departed earlier than", "earlier to depart", "later to depart", "出发时间", "比 {right} 更早出发", "更早出发", "更晚出发", "departed earlier", "departed later"),
    "start_time": _relation("temporal_order", "discovery", "start time", "started earlier than", "earlier to start", "later to start", "开始时间", "比 {right} 更早开始", "更早开始", "更晚开始", "started earlier", "started later"),
    "finish_time": _relation("temporal_order", "confirmation", "finish time", "finished earlier than", "earlier to finish", "later to finish", "完成时间", "比 {right} 更早完成", "更早完成", "更晚完成", "finished earlier", "finished later"),
    "registration_time": _relation("temporal_order", "confirmation", "registration time", "registered earlier than", "earlier to register", "later to register", "登记时间", "比 {right} 更早登记", "更早登记", "更晚登记", "registered earlier", "registered later"),
    "publication_time": _relation("temporal_order", "confirmation", "publication time", "was published earlier than", "earlier in publication", "later in publication", "发布时间", "比 {right} 更早发布", "更早发布", "更晚发布", "was published earlier", "was published later"),

    "north_position": _relation("spatial_order", "discovery", "northward position", "is farther north than", "farther north", "farther south", "南北位置", "比 {right} 更靠北", "更靠北", "更靠南"),
    "east_position": _relation("spatial_order", "discovery", "eastward position", "is farther east than", "farther east", "farther west", "东西位置", "比 {right} 更靠东", "更靠东", "更靠西"),
    "elevation": _relation("spatial_order", "discovery", "elevation", "is higher than", "higher", "lower", "海拔", "比 {right} 海拔更高", "海拔更高", "海拔更低"),
    "forward_position": _relation("spatial_order", "confirmation", "forward position", "is farther forward than", "farther forward", "farther back", "前后位置", "比 {right} 更靠前", "更靠前", "更靠后"),
    "distance": _relation("spatial_order", "confirmation", "distance", "is farther away than", "farther away", "closer", "距离", "比 {right} 距离更远", "距离更远", "距离更近"),
    "depth": _relation("spatial_order", "confirmation", "depth", "is deeper than", "deeper", "shallower", "深度", "比 {right} 更深", "更深", "更浅"),

    "authority": _relation("social_status", "discovery", "authority", "is more authoritative than", "more authoritative", "less authoritative", "权威程度", "比 {right} 更有权威", "更有权威", "权威更低"),
    "seniority": _relation("social_status", "discovery", "seniority", "is more senior than", "more senior", "more junior", "资历", "比 {right} 资历更深", "资历更深", "资历更浅"),
    "influence": _relation("social_status", "discovery", "influence", "is more influential than", "more influential", "less influential", "影响力", "比 {right} 影响力更大", "影响力更大", "影响力更小"),
    "popularity": _relation("social_status", "confirmation", "popularity", "is more popular than", "more popular", "less popular", "受欢迎程度", "比 {right} 更受欢迎", "更受欢迎", "较不受欢迎"),
    "responsibility": _relation("social_status", "confirmation", "responsibility", "has more responsibility than", "more responsible", "less responsible", "责任程度", "比 {right} 承担更多责任", "责任更多", "责任更少"),
    "leadership_rank": _relation("social_status", "confirmation", "leadership rank", "has a higher leadership rank than", "higher-ranked", "lower-ranked", "领导级别", "比 {right} 领导级别更高", "级别更高", "级别更低"),

    "causal_influence": _relation("epistemic_causal", "discovery", "causal influence", "has more causal influence than", "more causally influential", "less causally influential", "因果影响", "比 {right} 因果影响更强", "因果影响更强", "因果影响更弱"),
    "evidence_strength": _relation("epistemic_causal", "discovery", "evidence strength", "has stronger evidence than", "better supported", "less supported", "证据强度", "比 {right} 证据更强", "证据更强", "证据更弱"),
    "likelihood": _relation("epistemic_causal", "discovery", "likelihood", "is more likely than", "more likely", "less likely", "可能性", "比 {right} 更可能", "可能性更高", "可能性更低"),
    "certainty": _relation("epistemic_causal", "confirmation", "certainty", "is more certain than", "more certain", "less certain", "确定性", "比 {right} 更确定", "确定性更高", "确定性更低"),
    "explanatory_power": _relation("epistemic_causal", "confirmation", "explanatory power", "has greater explanatory power than", "more explanatory", "less explanatory", "解释力", "比 {right} 解释力更强", "解释力更强", "解释力更弱"),
    "dependency_strength": _relation("epistemic_causal", "confirmation", "dependency strength", "has stronger dependency than", "more dependent", "less dependent", "依赖强度", "比 {right} 依赖更强", "依赖更强", "依赖更弱"),
}


RELATIONS = tuple(RELATION_DEFINITIONS)
RELATION_FAMILY = {name: row["family"] for name, row in RELATION_DEFINITIONS.items()}
RELATION_SPLIT = {name: row["relation_split"] for name, row in RELATION_DEFINITIONS.items()}


def _spec(row: dict[str, str]) -> dict[str, dict[str, tuple[str, ...]]]:
    noun_en = row["noun_en"]
    noun_zh = row["noun_zh"]
    direct_en = row["direct_en"]
    direct_zh = row["direct_zh"]
    high_en, low_en = row["high_en"], row["low_en"]
    high_zh, low_zh = row["high_zh"], row["low_zh"]
    q_high_en, q_low_en = row["q_high_en"], row["q_low_en"]
    return {
        "en": {
            "positive": (
                f"{{left}} {direct_en} {{right}}",
                f"{{left}} {direct_en} {{right}}",
                f"{{left}} {direct_en} {{right}}",
                f"{{left}} {direct_en} {{right}}",
            ),
            "max": (
                f"Which label {q_high_en}",
                f"Which label {q_high_en}",
                f"Which label {q_high_en}",
                f"Which label {q_high_en}",
            ),
            "min": (
                f"Which label {q_low_en}",
                f"Which label {q_low_en}",
                f"Which label {q_low_en}",
                f"Which label {q_low_en}",
            ),
            "roles": (high_en, low_en),
        },
        "zh": {
            "positive": (
                f"{{left}} {direct_zh.format(right='{right}')}",
                f"{{left}} {direct_zh.format(right='{right}')}",
                f"{{left}} {direct_zh.format(right='{right}')}",
                f"{{left}} {direct_zh.format(right='{right}')}",
            ),
            "max": (
                f"哪个标签{high_zh}",
                f"哪个标签{high_zh}",
                f"哪个标签{high_zh}",
                f"哪个标签{high_zh}",
            ),
            "min": (
                f"哪个标签{low_zh}",
                f"哪个标签{low_zh}",
                f"哪个标签{low_zh}",
                f"哪个标签{low_zh}",
            ),
            "roles": (high_zh, low_zh),
        },
    }


RELATION_SPECS = {name: _spec(row) for name, row in RELATION_DEFINITIONS.items()}


SHELLS = {
    "en": {
        0: "Observed relation: {fact}. Unrelated display: {carrier}. Query: {question}? Return one label.",
        1: "Use this evidence: {fact}. Neutral order: {carrier}. Please decide: {question}? Answer with one label.",
        2: "A record states that {fact}. Incidental sequence: {carrier}. Determine: {question}? Write one label only.",
        3: "Consider the relation that {fact}. Displayed list: {carrier}. Final query: {question}? Give exactly one label.",
    },
    "zh": {
        0: "观察关系： {fact}。无关列表： {carrier}。问题： {question}？只回答一个标签。",
        1: "使用这项证据： {fact}。中性顺序： {carrier}。请判断： {question}？用一个标签回答。",
        2: "记录表明： {fact}。附带顺序： {carrier}。请确定： {question}？只写一个标签。",
        3: "考虑这项关系： {fact}。显示列表： {carrier}。最终问题： {question}？仅给出一个标签。",
    },
}


BRANCH_MARKERS = base.BRANCH_MARKERS
_BASE_BUILD_CASE = base.build_case
_BASE_AUDIT_MODEL = base.audit_model
_BASE_OLD_NAMES = base.old_names


def old_names() -> set[str]:
    values = set(_BASE_OLD_NAMES())
    prior = PHASE1098_ROOT / "protocol" / "preregistration.json"
    if prior.exists():
        values.update(read_json(prior).get("selected_names", []))
    return values


def select_names(tokenizers: dict[str, Any]) -> tuple[str, ...]:
    excluded = old_names()
    candidates = tuple(dict.fromkeys(
        base.phase1097.EXTRA_NAME_CANDIDATES
        + base.phase1096.ADDITIONAL_NAME_CANDIDATES
        + base.name_source.HELDOUT_NAME_CANDIDATES
    ))
    ranked = sorted(candidates, key=lambda value: hashlib.sha256(f"phase1099|{value}".encode("utf-8")).hexdigest())
    used_ids = {model: set() for model in MODELS}
    selected: list[str] = []
    required = len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2
    for name in ranked:
        if name in excluded:
            continue
        token_ids: dict[str, int] = {}
        for model_name, tokenizer in tokenizers.items():
            values = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(values) != 1 or int(values[0]) in used_ids[model_name]:
                break
            token_ids[model_name] = int(values[0])
        if len(token_ids) != len(MODELS):
            continue
        selected.append(name)
        for model_name, token_id in token_ids.items():
            used_ids[model_name].add(token_id)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(f"need {required} new one-token names, found {len(selected)}")
    return tuple(selected)


def build_case(*args, **kwargs) -> dict[str, Any]:
    row = _BASE_BUILD_CASE(*args, **kwargs)
    relation = str(row["relation"])
    old_unit = str(row["unit_id"])
    old_superunit = str(row["superunit_id"])
    row["schema_version"] = "phase1099_relation_family_case.v1"
    row["unit_id"] = old_unit.replace("phase1098.", "phase1099.", 1)
    row["superunit_id"] = old_superunit.replace("phase1098.", "phase1099.", 1)
    row["record_id"] = f'{row["unit_id"]}.{row["state"]}'
    row["family"] = RELATION_FAMILY[relation]
    row["relation_split"] = RELATION_SPLIT[relation]
    return row


def audit_model(model_name: str, rows: list[dict[str, Any]], selected_names: tuple[str, ...]) -> dict[str, Any]:
    result = _BASE_AUDIT_MODEL(model_name, rows, selected_names)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["superunit_id"])].append(row)
    checks = result["checks"]
    checks["five_families_complete"] = set(RELATION_FAMILY.values()) == set(FAMILIES)
    checks["six_relations_per_family"] = all(sum(value == family for value in RELATION_FAMILY.values()) == 6 for family in FAMILIES)
    checks["three_relations_per_family_holdout_half"] = all(
        sum(RELATION_FAMILY[relation] == family and RELATION_SPLIT[relation] == split for relation in RELATIONS) == 3
        for family in FAMILIES for split in RELATION_SPLITS
    )
    checks["complete_relation_family_atlas_per_superunit"] = all(
        {row["relation"] for row in values} == set(RELATIONS)
        and {row["family"] for row in values} == set(FAMILIES)
        for values in grouped.values()
    )
    checks["family_and_relation_split_metadata_exact"] = all(
        row["family"] == RELATION_FAMILY[row["relation"]]
        and row["relation_split"] == RELATION_SPLIT[row["relation"]]
        for row in rows
    )
    result["schema_version"] = "phase1099_protocol_model_audit.v1"
    result["phase"] = PHASE
    result["all_checks_passed"] = all(checks.values())
    result.pop("audit_digest", None)
    result["audit_digest"] = digest(result)
    return result


def _configure_base() -> None:
    values = {
        "PHASE": PHASE,
        "PROTOCOL_REVISION": PROTOCOL_REVISION,
        "MODELS": MODELS,
        "PRECISION": PRECISION,
        "QUANTIZATION": QUANTIZATION,
        "RELATIONS": RELATIONS,
        "SURFACES": SURFACES,
        "SPLITS": SPLITS,
        "TEMPLATES": TEMPLATES,
        "TEMPLATES_BY_SPLIT": TEMPLATES_BY_SPLIT,
        "PANELS": PANELS,
        "TASKS": TASKS,
        "ORIENTATIONS": ORIENTATIONS,
        "CARRIER_ORDERS": CARRIER_ORDERS,
        "ITEMS_PER_TEMPLATE": ITEMS_PER_TEMPLATE,
        "ASSISTANT_PREFILL": ASSISTANT_PREFILL,
        "CONTINUATION_PREFIX": CONTINUATION_PREFIX,
        "GENERATION_STEPS": GENERATION_STEPS,
        "GENERATION_ITEMS_PER_CELL": GENERATION_ITEMS_PER_CELL,
        "CAPTURE_ROLES": CAPTURE_ROLES,
        "PRE_TASK_ROLES": PRE_TASK_ROLES,
        "FIELDS": FIELDS,
        "PRIMARY_FIELD": PRIMARY_FIELD,
        "CONTROL_FIELDS": CONTROL_FIELDS,
        "STATES": STATES,
        "OUT_ROOT": OUT_ROOT,
        "EVIDENCE_THRESHOLDS": EVIDENCE_THRESHOLDS,
        "PROSPECTIVE_PREDICTIONS": PROSPECTIVE_PREDICTIONS,
        "RELATION_SPECS": RELATION_SPECS,
        "SHELLS": SHELLS,
        "BRANCH_MARKERS": BRANCH_MARKERS,
        "old_names": old_names,
        "select_names": select_names,
        "build_case": build_case,
        "audit_model": audit_model,
    }
    for key, value in values.items():
        setattr(base, key, value)


_configure_base()


state_factors = base.state_factors
split_for_template = base.split_for_template
build_model_cases = base.build_model_cases


def main() -> None:
    tokenizers = {model_name: tokenizer_for(model_name) for model_name in MODELS}
    selected_names = select_names(tokenizers)
    protocol_root = OUT_ROOT / "protocol"
    model_audits: dict[str, Any] = {}
    case_digests: dict[str, str] = {}
    row_count = 0
    for model_name in MODELS:
        rows = build_model_cases(tokenizers[model_name], model_name, selected_names)
        row_count = len(rows)
        audit = audit_model(model_name, rows, selected_names)
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        case_digests[model_name] = audit["case_digest"]
        print({"phase": PHASE, "model": model_name, "cases": len(rows), "audit_passed": audit["all_checks_passed"]})
    source_summary = read_json(SOURCE_PHASE1098) if SOURCE_PHASE1098.exists() else {}
    source_audit = read_json(SOURCE_PHASE1098_AUDIT) if SOURCE_PHASE1098_AUDIT.exists() else {}
    prereg = {
        "schema_version": "phase1099_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "relations": list(RELATIONS),
        "relation_family": RELATION_FAMILY,
        "relation_split": RELATION_SPLIT,
        "surfaces": list(SURFACES),
        "templates": list(TEMPLATES),
        "templates_by_split": {key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()},
        "panels": list(PANELS),
        "states": list(STATES),
        "items_per_template": ITEMS_PER_TEMPLATE,
        "generation_steps": GENERATION_STEPS,
        "generation_items_per_cell": GENERATION_ITEMS_PER_CELL,
        "case_count_per_model": row_count,
        "unit_count_per_model": len(RELATIONS) * len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE,
        "superunit_count_per_model": len(SURFACES) * len(TEMPLATES) * ITEMS_PER_TEMPLATE,
        "selected_names": list(selected_names),
        "capture_roles": list(CAPTURE_ROLES),
        "fields": list(FIELDS),
        "primary_field": PRIMARY_FIELD,
        "control_fields": list(CONTROL_FIELDS),
        "sampled_event_grid": "residual/attention/MLP at relative depth 0.0,0.1,...,1.0, deduplicated per architecture",
        "primary_object": "eventwise signed relation-centered 5x5 family block Gram graph built from disjoint 3-relation halves",
        "forbidden_primary_inputs": ["candidate logits", "output margin", "generation score", "PCA", "learned probe"],
        "family_permutation_test": "all 5! family-label permutations",
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "automatic_next_rule": "Only P1-P7 jointly authorize an independent family-selective causal stage; otherwise stop automatic continuation.",
        "revision_history": {
            "revision_1": "Rejected before behavior: a generic higher/lower confirmation phrase reversed the natural meaning of earlier temporal relations.",
            "revision_2": "Rejected before hidden-state access: Qwen3 temporal failures and broad DS7B failures concentrated in the relational panel and generic meta-comparison templates; all artifacts are archived.",
            "revision_3": "Rejected before hidden-state access: direct facts removed the meta-comparison burden, but generated English temporal questions remained unnatural and the eight-sample generation cell made the 0.65 threshold resolve effectively to 0.75.",
            "revision_4": "Final behavior revision: uses natural finite-verb questions for the six temporal relations and restores two generation items per cell. Relations, relation holdouts, names, candidate cases, thresholds, fields, and family gates are unchanged. No further behavior revision is authorized.",
        },
        "source_phase1098_summary_digest": source_summary.get("summary_digest"),
        "source_phase1098_result_audit_digest": source_audit.get("audit_digest"),
        "model_case_digests": case_digests,
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    aggregate = {
        "schema_version": "phase1099_protocol_audit.v1",
        "phase": PHASE,
        "model_audits": model_audits,
        "source_phase1098_present": SOURCE_PHASE1098.exists(),
        "source_phase1098_audit_present": SOURCE_PHASE1098_AUDIT.exists(),
    }
    aggregate["all_checks_passed"] = (
        all(row["all_checks_passed"] for row in model_audits.values())
        and aggregate["source_phase1098_present"]
        and aggregate["source_phase1098_audit_present"]
    )
    aggregate["audit_digest"] = digest(aggregate)
    write_json(protocol_root / "audit.json", aggregate)
    print({"phase": PHASE, "protocol_digest": prereg["protocol_digest"], "all_checks_passed": aggregate["all_checks_passed"], "selected_names": len(selected_names)})


if __name__ == "__main__":
    main()
