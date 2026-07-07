#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE = 234
SOURCE_PHASE = 233
RESULT_ROOT = Path("tests/result/phase234_pattern_family_atlas_matrix")


FAMILIES: list[dict[str, Any]] = [
    {
        "family_id": "content_knowledge",
        "family_name": "内容知识模式族",
        "core_question": "对象、属性、类别、关系、事实如何写入当前状态并进入读出竞争。",
        "mechanism_hypothesis": "PromptPattern -> ObjectRelationState -> MLPProduct -> ResidualWrite -> TargetPressure -> CompetitorPressure -> ReadoutWinner",
        "modes": [
            "object_identity",
            "attribute_answer",
            "category_membership",
            "object_relation_value",
            "fact_recall",
            "relation_completion",
        ],
        "priority": 1,
    },
    {
        "family_id": "output_protocol",
        "family_name": "输出协议模式族",
        "core_question": "模型如何决定短答、解释、列表、复读、停止和续写。",
        "mechanism_hypothesis": "InstructionFrame -> AnswerAnchor -> StepCondition -> GatedActivation -> ReadoutRegimeSelection",
        "modes": [
            "short_answer",
            "explain_answer",
            "list_answer",
            "repeat_answer",
            "target_seeded",
            "answer_boundary",
            "continuation",
            "period_stop",
        ],
        "priority": 1,
    },
    {
        "family_id": "reasoning_constraint",
        "family_name": "推理约束模式族",
        "core_question": "逻辑约束如何写入状态、维持传播，并与 because 等读出机制竞争。",
        "mechanism_hypothesis": "ReasonTrigger -> ConstraintState -> StateMaintainPath -> ReadoutCompetition",
        "modes": [
            "because_reason",
            "causal_reasoning",
            "comparison",
            "condition_if_then",
            "counterfactual",
            "negation",
            "double_negation",
            "scope_binding",
            "multi_step_reasoning",
        ],
        "priority": 2,
    },
    {
        "family_id": "syntax_structure",
        "family_name": "语法结构模式族",
        "core_question": "语法边界、连接、嵌套是否直接参与读出机制选择。",
        "mechanism_hypothesis": "SyntaxTrigger -> BoundaryState -> ReadoutRegimeSelection -> ContinuationStopFormat",
        "modes": [
            "subject_verb_object",
            "modifier_binding",
            "clause_embedding",
            "coordination",
            "contrast_connector",
            "causal_connector",
            "question_form",
            "quotation",
            "punctuation_boundary",
        ],
        "priority": 2,
    },
    {
        "family_id": "language_action",
        "family_name": "语言动作模式族",
        "core_question": "模型如何选择回答、解释、总结、翻译、分类、续写等语言动作。",
        "mechanism_hypothesis": "TaskFrame -> ActionMode -> OutputProtocol -> RolloutPattern",
        "modes": [
            "answer",
            "explain",
            "summarize",
            "translate",
            "classify",
            "continue",
            "format_convert",
            "ask_clarification",
        ],
        "priority": 2,
    },
    {
        "family_id": "cross_lingual",
        "family_name": "跨语言模式族",
        "core_question": "语言无关逻辑层是否共享，输出语言化如何发生。",
        "mechanism_hypothesis": "LanguageSpecificInput -> SharedConstraintState -> OutputLanguageReadout",
        "modes": [
            "en_to_en",
            "zh_to_en",
            "en_to_zh",
            "cross_lingual_attribute",
            "cross_lingual_negation",
            "cross_lingual_explain",
            "translation",
        ],
        "priority": 3,
    },
    {
        "family_id": "readout_competition",
        "family_name": "竞争读出模式族",
        "core_question": "目标答案为什么被 because、period、For、be、echo 等机制压住。",
        "mechanism_hypothesis": "TargetPressure -> CompetitorSource -> CompetitorPressureField -> WinnerRegime -> SecondCompetitorTakeover",
        "modes": [
            "target_answer",
            "because_reason",
            "period_stop",
            "echo_repeat",
            "for_continuation",
            "the_continuation",
            "then_continuation",
            "be_continuation",
            "answer_boundary",
            "newline_boundary",
            "second_competitor_takeover",
        ],
        "priority": 1,
    },
    {
        "family_id": "state_drift",
        "family_name": "状态维持与漂移模式族",
        "core_question": "模型为什么早期正确、后续漂移，或者格式/解释/停止失稳。",
        "mechanism_hypothesis": "InitialState -> StateMaintainPath -> StepCondition -> RegimeSwitch -> DriftType",
        "modes": [
            "stable_answer",
            "early_correct_late_drift",
            "explain_drift",
            "echo_drift",
            "format_drift",
            "next_task_drift",
            "stop_failure",
        ],
        "priority": 2,
    },
    {
        "family_id": "closure",
        "family_name": "闭合模式族",
        "core_question": "模型真正完成语言任务需要哪些内部条件。",
        "mechanism_hypothesis": "AnswerCorrect & PatternMatched & BoundaryStable & DoneStateStable & ModelStopExecuted & NoDrift",
        "modes": [
            "answer_correct",
            "pattern_matched",
            "boundary_stable",
            "done_state_stable",
            "model_stop_executed",
            "no_drift",
            "model_close",
        ],
        "priority": 1,
    },
]

LEVELS = [
    {"level": 1, "level_id": "behavior", "metrics": ["answer_correct", "pattern_match", "drift_type", "output_length", "stop_type"]},
    {"level": 2, "level_id": "prompt_trigger", "metrics": ["prompt_variant", "anchor_removed", "instruction_replaced", "trigger_token"]},
    {"level": 3, "level_id": "gate_up_product", "metrics": ["gate_delta", "up_delta", "product_delta", "down_out_delta", "product_recompute_error"]},
    {"level": 4, "level_id": "residual_state", "metrics": ["residual_direction", "projection_shift", "state_propagation"]},
    {"level": 5, "level_id": "readout_competition", "metrics": ["target_logit", "target_rank", "winner_regime", "second_competitor", "remaining_gap"]},
    {"level": 6, "level_id": "competitor_source", "metrics": ["source_delta", "source_suppression", "winner_changed", "margin_help"]},
    {"level": 7, "level_id": "rollout", "metrics": ["token_sequence", "regime_sequence", "drift_step", "stop_step"]},
    {"level": 8, "level_id": "closure", "metrics": ["model_close", "done_state_stable", "model_stop_executed", "no_drift"]},
]

SEED_OBJECTS = [
    {"object": "apple", "relation": "color", "target": "red", "category": "fruit"},
    {"object": "banana", "relation": "color", "target": "yellow", "category": "fruit"},
    {"object": "grass", "relation": "color", "target": "green", "category": "plant"},
    {"object": "snow", "relation": "color", "target": "white", "category": "weather"},
    {"object": "coal", "relation": "color", "target": "black", "category": "material"},
    {"object": "lemon", "relation": "taste", "target": "sour", "category": "fruit"},
    {"object": "hammer", "relation": "function", "target": "hit", "category": "tool"},
    {"object": "wheel", "relation": "part_of", "target": "car", "category": "part"},
]

PROTOCOLS = [
    {"protocol_id": "short", "instruction": "Answer with one word."},
    {"protocol_id": "explain", "instruction": "Answer with the answer first, then one short reason using because."},
    {"protocol_id": "repeat", "instruction": "Answer with exactly the same answer word twice, separated by a comma."},
    {"protocol_id": "list", "instruction": "Answer as a short list."},
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def family_rows() -> list[dict[str, Any]]:
    return [
        {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "row_kind": "phase234_pattern_family_row",
            **family,
            "mode_count": len(family["modes"]),
        }
        for family in FAMILIES
    ]


def mode_rows() -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for idx, mode in enumerate(family["modes"], start=1):
            rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase234_pattern_mode_row",
                    "family_id": family["family_id"],
                    "family_name": family["family_name"],
                    "family_priority": family["priority"],
                    "mode_id": mode,
                    "mode_order": idx,
                    "mechanism_hypothesis": family["mechanism_hypothesis"],
                    "required_levels": [x["level_id"] for x in LEVELS],
                }
            )
    return rows


def test_case_rows() -> list[dict[str, Any]]:
    rows = []
    case_id = 0
    for item in SEED_OBJECTS:
        for protocol in PROTOCOLS:
            case_id += 1
            prompt = f"What is the {item['relation']} of {item['object']}?\n{protocol['instruction']}\nAnswer:"
            rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase234_seed_test_case_row",
                    "case_id": f"pf_case_{case_id:04d}",
                    "family_id": "content_knowledge",
                    "linked_family_ids": ["output_protocol", "readout_competition", "state_drift", "closure"],
                    "mode_id": "object_relation_value",
                    "protocol_id": protocol["protocol_id"],
                    "object": item["object"],
                    "relation": item["relation"],
                    "target": item["target"],
                    "category": item["category"],
                    "prompt": prompt,
                    "expected_pattern": protocol["protocol_id"],
                    "test_levels": [x["level_id"] for x in LEVELS],
                }
            )
    extra_cases = [
        ("reason_negation_0001", "reasoning_constraint", "negation", "If the apple is not green, is it green?\nAnswer with yes or no.\nAnswer:", "no"),
        ("syntax_boundary_0001", "syntax_structure", "punctuation_boundary", "Answer: red. Continue?\nAnswer with one word.\nAnswer:", "red"),
        ("readout_takeover_0001", "readout_competition", "second_competitor_takeover", "What is the color of apple?\nAnswer with the answer first, then one short reason using because.\nAnswer:", "red"),
        ("closure_stop_0001", "closure", "model_close", "What is the color of snow?\nAnswer with one word and stop.\nAnswer:", "white"),
    ]
    for case_id_text, family_id, mode_id, prompt, target in extra_cases:
        rows.append(
            {
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "row_kind": "phase234_seed_test_case_row",
                "case_id": case_id_text,
                "family_id": family_id,
                "linked_family_ids": ["readout_competition", "state_drift", "closure"],
                "mode_id": mode_id,
                "protocol_id": "special",
                "object": "",
                "relation": "",
                "target": target,
                "category": "",
                "prompt": prompt,
                "expected_pattern": mode_id,
                "test_levels": [x["level_id"] for x in LEVELS],
            }
        )
    return rows


def phase_plan_rows() -> list[dict[str, Any]]:
    steps = [
        ("phase234", "pattern_family_matrix", "建立模式族、模式、测试层级、样例任务的机器可读矩阵。"),
        ("phase235", "behavior_family_benchmark", "先跑行为层模式分类，覆盖 qwen3、GLM4、DS7B。"),
        ("phase236", "prompt_trigger_family_atlas", "对高差异模式做 prompt trigger 和 anchor 消融。"),
        ("phase237", "gate_product_family_atlas", "采集 gate/up/product/down_out 跨模式差分。"),
        ("phase238", "readout_competition_family_atlas", "统一记录 winner、second competitor、remaining gap。"),
        ("phase239", "source_suppression_family_validation", "选择最稳定模式做 source-level suppression。"),
        ("phase240", "closure_candidate_family_validation", "选择少数模式尝试完整闭合。"),
    ]
    return [
        {
            "phase": PHASE,
            "source_phase": SOURCE_PHASE,
            "row_kind": "phase234_program_phase_row",
            "program_phase_id": phase_id,
            "task_id": task_id,
            "objective": objective,
            "order": idx,
        }
        for idx, (phase_id, task_id, objective) in enumerate(steps, start=1)
    ]


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 234 Pattern Family Atlas Matrix", ""]
    lines.append(f"families: {payload['family_count']}")
    lines.append(f"modes: {payload['mode_count']}")
    lines.append(f"seed_test_cases: {payload['seed_test_case_count']}")
    lines.extend(["", "## Families", "", "| priority | family | modes | mechanism |", "| ---: | --- | ---: | --- |"])
    for row in payload["families"]:
        lines.append(f"| {row['priority']} | {row['family_id']} / {row['family_name']} | {row['mode_count']} | {row['mechanism_hypothesis']} |")
    lines.extend(["", "## Test Levels", "", "| level | id | metrics |", "| ---: | --- | --- |"])
    for row in LEVELS:
        lines.append(f"| {row['level']} | {row['level_id']} | {', '.join(row['metrics'])} |")
    lines.extend(["", "## Seed Cases", "", "| case | family | mode | protocol | target | prompt |", "| --- | --- | --- | --- | --- | --- |"])
    for row in payload["seed_test_cases"][:40]:
        prompt = row["prompt"].replace("\n", "\\n")
        lines.append(f"| {row['case_id']} | {row['family_id']} | {row['mode_id']} | {row['protocol_id']} | {row['target']} | {prompt} |")
    lines.extend(["", "## Program Phases", "", "| order | phase | task | objective |", "| ---: | --- | --- | --- |"])
    for row in payload["phase_plan"]:
        lines.append(f"| {row['order']} | {row['program_phase_id']} | {row['task_id']} | {row['objective']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    families = family_rows()
    modes = mode_rows()
    cases = test_case_rows()
    phases = phase_plan_rows()
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Pattern family atlas matrix",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "family_count": len(families),
        "mode_count": len(modes),
        "seed_test_case_count": len(cases),
        "test_level_count": len(LEVELS),
        "families": families,
        "test_levels": LEVELS,
        "seed_test_cases": cases,
        "phase_plan": phases,
    }
    write_json(out_dir / "phase234_pattern_family_atlas_matrix.json", payload)
    write_jsonl(out_dir / "phase234_pattern_family_rows.jsonl", families)
    write_jsonl(out_dir / "phase234_pattern_mode_rows.jsonl", modes)
    write_jsonl(out_dir / "phase234_seed_test_case_rows.jsonl", cases)
    write_jsonl(out_dir / "phase234_program_phase_rows.jsonl", phases)
    write_markdown(out_dir / "phase234_pattern_family_atlas_matrix.md", payload)
    print(json.dumps({"phase": PHASE, "status": "complete", "families": len(families), "modes": len(modes), "seed_test_cases": len(cases)}, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase234 pattern family atlas matrix")
    parser.add_argument("--round-name", default="pattern_family_atlas_matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.round_name)


if __name__ == "__main__":
    main()
