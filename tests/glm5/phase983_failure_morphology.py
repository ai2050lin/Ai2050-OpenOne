#!/usr/bin/env python3
"""CPU-only post-hoc failure morphology for the sealed Phase 983 audit.

This report is deliberately descriptive.  It cannot change the preregistered
Phase 983 decision, authorize a holdout, or support an internal-mechanism claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""

GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402


OUTPUT_PATH = core.OUT / "failure_morphology.json"
STATE_ORDER = ("V", "C", "I_protocol", "I_sem")
CHECKPOINT_ORDER = ("256", "512", "1024", "1536", "2048")


def verify_source_audit(audit: dict[str, Any]) -> None:
    core.require(audit.get("schema_version") == core.SCHEMA_VERSION,
                 "audit schema changed")
    core.require(audit.get("phase") == core.PHASE, "audit phase changed")
    core.require(audit.get("experiment") == core.EXPERIMENT,
                 "audit experiment changed")
    core.require(audit.get("integrity_decision") == "GO",
                 "source integrity is not GO")
    core.require(audit.get("scientific_decision") == "NO-GO",
                 "failure morphology is only authorized after NO-GO")
    expected_hash = core.sha256_json(core.without_fields(
        audit, "audit_sha256", "created_at_utc"))
    core.require(audit.get("audit_sha256") == expected_hash,
                 "source audit self-hash invalid")
    gate = audit.get("gate_decision")
    checkpoints = audit.get("checkpoint_descriptive_summaries")
    core.require(isinstance(gate, dict) and isinstance(checkpoints, dict),
                 "source audit summaries missing")
    core.require(gate.get("pooling_used") is False,
                 "source audit unexpectedly pooled models")


def ordered_counts(value: dict[str, Any]) -> dict[str, int]:
    core.require(isinstance(value, dict) and set(value) == set(STATE_ORDER),
                 "terminal-state count schema changed")
    result: dict[str, int] = {}
    for state in STATE_ORDER:
        count = value[state]
        core.require(isinstance(count, int) and not isinstance(count, bool)
                     and count >= 0, "terminal-state count invalid")
        result[state] = count
    core.require(sum(result.values()) == core.ITEM_COUNT,
                 "terminal-state counts do not sum to 256")
    return result


def transition_flux(matrix: dict[str, Any]) -> dict[str, Any]:
    core.require(isinstance(matrix, dict) and set(matrix) == set(STATE_ORDER),
                 "transition matrix rows changed")
    cells: dict[str, int] = {}
    for source in STATE_ORDER:
        row = matrix[source]
        core.require(isinstance(row, dict) and set(row) == set(STATE_ORDER),
                     "transition matrix columns changed")
        for target in STATE_ORDER:
            value = row[target]
            core.require(isinstance(value, int) and not isinstance(value, bool)
                         and value >= 0, "transition count invalid")
            cells[f"{source}->{target}"] = value
    core.require(sum(cells.values()) == core.ITEM_COUNT,
                 "transition matrix does not sum to 256")

    into_v = {f"{state}->V": cells[f"{state}->V"]
              for state in STATE_ORDER if state != "V"}
    out_of_v = {f"V->{state}": cells[f"V->{state}"]
                for state in STATE_ORDER if state != "V"}
    gross_into_v = sum(into_v.values())
    gross_out_of_v = sum(out_of_v.values())
    off_diagonal = [
        {"transition": key, "count": value}
        for key, value in cells.items()
        if key.split("->")[0] != key.split("->")[1] and value > 0
    ]
    off_diagonal.sort(key=lambda item: (-item["count"], item["transition"]))
    return {
        "matrix": {
            source: {target: cells[f"{source}->{target}"]
                     for target in STATE_ORDER}
            for source in STATE_ORDER
        },
        "gross_into_V": gross_into_v,
        "gross_out_of_V": gross_out_of_v,
        "net_into_V": gross_into_v - gross_out_of_v,
        "into_V_components": into_v,
        "out_of_V_components": out_of_v,
        "dominant_off_diagonal_transitions": off_diagonal[:6],
    }


def checkpoint_profile(value: dict[str, Any]) -> dict[str, Any]:
    core.require(isinstance(value, dict)
                 and set(value) == {"A", "B", "decision_transition_matrix"},
                 "checkpoint stream schema changed")
    profiles: dict[str, Any] = {}
    delta_v: dict[str, int] = {}
    delta_c_plus_protocol: dict[str, int] = {}
    for checkpoint in CHECKPOINT_ORDER:
        a = ordered_counts(value["A"][checkpoint])
        b = ordered_counts(value["B"][checkpoint])
        delta_v[checkpoint] = b["V"] - a["V"]
        delta_c_plus_protocol[checkpoint] = (
            b["C"] + b["I_protocol"] - a["C"] - a["I_protocol"])
        profiles[checkpoint] = {
            "A": a,
            "B": b,
            "delta_V_B_minus_A": delta_v[checkpoint],
            "delta_C_plus_I_protocol_B_minus_A":
                delta_c_plus_protocol[checkpoint],
        }
    first_nonnegative = next(
        (checkpoint for checkpoint in CHECKPOINT_ORDER
         if delta_v[checkpoint] >= 0), None)
    return {
        "by_checkpoint": profiles,
        "delta_V_path": delta_v,
        "delta_C_plus_I_protocol_path": delta_c_plus_protocol,
        "first_nonnegative_delta_V_checkpoint": first_nonnegative,
        "checkpoint_role": "descriptive_only; 2048 alone sets the decision",
    }


def stratum_delta_v(strata: dict[str, Any]) -> dict[str, int]:
    core.require(isinstance(strata, dict), "stratum summary missing")
    output: dict[str, int] = {}
    for key, value in strata.items():
        accounting = None
        if isinstance(value, dict):
            accounting = value.get("accounting", value)
        core.require(isinstance(accounting, dict), "stratum accounting missing")
        delta = accounting.get("delta_V")
        core.require(isinstance(delta, int) and not isinstance(delta, bool),
                     "stratum delta_V invalid")
        output[str(key)] = delta
    return output


def stream_morphology(
    result: dict[str, Any], checkpoint: dict[str, Any],
) -> dict[str, Any]:
    overall = result["overall_accounting"]
    a = ordered_counts(overall["external_bundle_A_counts"])
    b = ordered_counts(overall["external_bundle_B_counts"])
    flux = transition_flux(checkpoint["decision_transition_matrix"])
    core.require(flux["net_into_V"] == overall["delta_V"],
                 "transition flux does not reproduce delta_V")

    task_delta: dict[str, int] = {}
    for task, accounting in sorted(result["task_accounting"].items()):
        delta = accounting.get("delta_V")
        core.require(isinstance(delta, int) and not isinstance(delta, bool),
                     "task delta_V invalid")
        task_delta[task] = delta

    primary_failed = sorted(
        key for key, passed in result["primary_checks"].items() if not passed)
    coverage_failed = sorted(
        key for key, passed in result["coverage_checks"].items() if not passed)
    label_swap_failed = sorted(
        key for key, passed in result["label_and_swap_checks"].items()
        if not passed)
    paired = result["paired_metrics"]
    profile = checkpoint_profile(checkpoint)
    core.require(profile["delta_V_path"]["2048"] == overall["delta_V"],
                 "checkpoint decision delta does not match gate delta")
    return {
        "decision_counts": {"A": a, "B": b},
        "decision_contrasts": {
            "delta_V": overall["delta_V"],
            "delta_C": overall["delta_C"],
            "delta_I_protocol": overall["delta_I_protocol"],
            "delta_C_plus_I_protocol": overall["delta_C_plus_I_protocol"],
            "R_I_sem": overall["R_I_sem"],
        },
        "paired": {
            "I_sem_A_to_V_B": paired["I_sem_A_to_V_B"],
            "V_A_to_I_sem_B": paired["V_A_to_I_sem_B"],
            "net_I_sem_to_V_minus_V_to_I_sem":
                paired["net_I_sem_to_V_minus_V_to_I_sem"],
            "bad_paired_leakage": paired["bad_paired_leakage"],
            "bad_leakage_terms": paired["bad_leakage_terms"],
        },
        "transition_flux": flux,
        "checkpoint_profile": profile,
        "task_delta_V": task_delta,
        "difficulty_delta_V": stratum_delta_v(result["difficulty_accounting"]),
        "gold_label_delta_V": stratum_delta_v(result["gold_strata"]),
        "swap_surface_delta_V": stratum_delta_v(result["swap_strata"]),
        "qualifying_tasks": sorted(result["qualifying_tasks"]),
        "stream_passed": result["external_bundle_stream_passed"],
        "failed_primary_checks": primary_failed,
        "failed_coverage_checks": coverage_failed,
        "failed_label_swap_checks": label_swap_failed,
    }


def classify_model(streams: dict[str, Any]) -> str:
    delta_v = [value["decision_contrasts"]["delta_V"]
               for value in streams.values()]
    r_i_sem = [value["decision_contrasts"]["R_I_sem"]
               for value in streams.values()]
    c_plus_p = [value["decision_contrasts"]["delta_C_plus_I_protocol"]
                for value in streams.values()]
    leakage = [value["paired"]["bad_paired_leakage"]
               for value in streams.values()]
    if max(delta_v) < 0 and max(r_i_sem) == 0:
        return "B延迟或破坏终止，但没有可观测语义错误救援信号"
    if max(delta_v) < 0 and min(r_i_sem) >= 26 and min(c_plus_p) > 12:
        return "B确有语义错误救援，但被协议错误与未终止增量反向淹没"
    if min(delta_v) > 0 and max(r_i_sem) < 26 and min(leakage) > 12:
        return "B主要改善协议终止，语义救援很弱且成对坏泄漏仍超阈值"
    return "跨流形态混合，不能归为单一稳定失败机制"


def model_morphology(
    model_result: dict[str, Any], checkpoints: dict[str, Any],
) -> dict[str, Any]:
    stream_names = tuple(f"stream_{stream}" for stream in core.STREAMS)
    core.require(set(model_result["stream_results"]) == set(stream_names),
                 "model stream set changed")
    core.require(set(checkpoints) == set(stream_names),
                 "checkpoint stream set changed")
    streams = {
        name: stream_morphology(model_result["stream_results"][name],
                                checkpoints[name])
        for name in stream_names
    }
    tasks = sorted(next(iter(streams.values()))["task_delta_V"])
    core.require(all(sorted(value["task_delta_V"]) == tasks
                     for value in streams.values()), "task sets changed")
    task_vectors = {
        task: [streams[name]["task_delta_V"][task] for name in stream_names]
        for task in tasks
    }
    consistently_helped = [
        task for task, values in task_vectors.items() if min(values) >= 3]
    consistently_harmed = [
        task for task, values in task_vectors.items() if max(values) <= -3]
    mixed_or_small = [
        task for task in tasks
        if task not in consistently_helped and task not in consistently_harmed]
    delta_v_vector = [
        streams[name]["decision_contrasts"]["delta_V"]
        for name in stream_names]
    first_nonnegative = [
        streams[name]["checkpoint_profile"][
            "first_nonnegative_delta_V_checkpoint"]
        for name in stream_names]
    return {
        "streams": streams,
        "model_pass": model_result["model_pass"],
        "all_three_streams_passed":
            model_result["all_three_streams_passed"],
        "common_qualifying_tasks":
            sorted(model_result["common_qualifying_tasks"]),
        "common_qualifying_task_count":
            model_result["common_qualifying_task_count"],
        "decision_delta_V_vector_stream_0_1_2": delta_v_vector,
        "delta_V_signs": ["positive" if value > 0 else
                          "negative" if value < 0 else "zero"
                          for value in delta_v_vector],
        "first_nonnegative_checkpoint_by_stream_0_1_2": first_nonnegative,
        "task_delta_V_vectors_stream_0_1_2": task_vectors,
        "consistently_helped_tasks_delta_V_at_least_3": consistently_helped,
        "consistently_harmed_tasks_delta_V_at_most_minus_3": consistently_harmed,
        "mixed_or_small_task_directions": mixed_or_small,
        "descriptive_failure_phenotype": classify_model(streams),
    }


def build_payload(audit: dict[str, Any]) -> dict[str, Any]:
    verify_source_audit(audit)
    gate = audit["gate_decision"]
    results = gate["model_results"]
    checkpoints = audit["checkpoint_descriptive_summaries"]
    core.require(set(results) == set(core.MODEL_ORDER), "model results changed")
    core.require(set(checkpoints) == set(core.MODEL_ORDER),
                 "checkpoint models changed")
    models = {
        model: model_morphology(results[model], checkpoints[model])
        for model in core.MODEL_ORDER
    }
    common = set(models[core.MODEL_ORDER[0]]["common_qualifying_tasks"])
    for model in core.MODEL_ORDER[1:]:
        common &= set(models[model]["common_qualifying_tasks"])
    signs_by_stream = {
        f"stream_{stream}": {
            model: models[model]["delta_V_signs"][stream]
            for model in core.MODEL_ORDER
        }
        for stream in core.STREAMS
    }
    return {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "analysis_name": "sealed_NO_GO_failure_morphology",
        "analysis_scope": "CPU-only post-hoc descriptive counts",
        "source_audit_sha256": audit["audit_sha256"],
        "source_audit_file_sha256": core.sha256_file(core.COMBINED_AUDIT_PATH),
        "source_protocol_sha256": audit["protocol_sha256"],
        "source_scientific_decision": audit["scientific_decision"],
        "primary_decision_unchanged": True,
        "can_set_primary_decision": False,
        "can_authorize_holdout": False,
        "can_authorize_mechanism": False,
        "models_pooled": False,
        "statistical_significance_claimed": False,
        "causal_mechanism_claimed": False,
        "models": models,
        "cross_model": {
            "cross_model_pass": gate["cross_model_pass"],
            "all_three_models_passed": gate["all_three_models_passed"],
            "common_qualifying_tasks_recomputed": sorted(common),
            "common_qualifying_task_count_recomputed": len(common),
            "delta_V_signs_by_stream": signs_by_stream,
            "universal_positive_delta_V": all(
                sign == "positive"
                for values in signs_by_stream.values()
                for sign in values.values()),
            "basic_reading": (
                "同一自然语言配置束在三个原生模板下产生不同外部形态；"
                "不存在跨模型稳定正向结果。"
            ),
        },
        "interpretation_boundaries": [
            "终态四分箱是外部测量口径，不是神经网络内部通道。",
            "V+C+I_protocol+I_sem=N只是记账恒等式。",
            "冻结共同随机数下的成对转移是关联计数，不是个体反事实因果。",
            "检查点只描述预算轨迹；只有2048-token终态参与主判定。",
            "模型使用各自原生聊天模板，因此外部指令对比不等于共同thinking开关。",
            "本报告在看到NO-GO后生成，不能反向选择门槛或候选配置。",
        ],
        "next_stage_constraints": {
            "run_holdout_now": False,
            "run_internal_mechanism_now": False,
            "exclude_failed_model": False,
            "relax_thresholds": False,
            "reuse_formal_rows_for_candidate_selection": False,
            "allowed_now": "保存失败形态并设计新的、独立预注册的开发阶段",
        },
    }


def verify_existing(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    core.require(set(existing) == set(expected) | {
        "morphology_sha256", "created_at_utc"},
        "failure morphology schema changed")
    expected_hash = core.sha256_json(expected)
    core.require(existing.get("morphology_sha256") == expected_hash,
                 "failure morphology self-hash invalid")
    core.require(core.without_fields(
        existing, "morphology_sha256", "created_at_utc") == expected,
        "failure morphology payload changed")
    timestamp = existing.get("created_at_utc")
    core.require(isinstance(timestamp, str) and timestamp,
                 "failure morphology timestamp missing")


def static_self_test() -> dict[str, Any]:
    matrix = {
        state: {target: 0 for target in STATE_ORDER}
        for state in STATE_ORDER
    }
    matrix["V"]["V"] = 200
    matrix["V"]["C"] = 10
    matrix["C"]["V"] = 20
    matrix["C"]["C"] = 26
    flux = transition_flux(matrix)
    tests = {
        "matrix_sums_to_256": sum(
            value for row in matrix.values() for value in row.values()) == 256,
        "gross_into_V": flux["gross_into_V"] == 20,
        "gross_out_of_V": flux["gross_out_of_V"] == 10,
        "net_into_V": flux["net_into_V"] == 10,
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    }
    core.require(all(tests.values()), "failure morphology self-test failed")
    return {"tests": tests, "files_written": False}


def run(write: bool, verify: bool) -> dict[str, Any]:
    if not write and not verify:
        return static_self_test()
    audit = core.load_json(core.COMBINED_AUDIT_PATH, "Phase983 combined audit")
    expected = build_payload(audit)
    if OUTPUT_PATH.exists():
        existing = core.load_json(OUTPUT_PATH, "Phase983 failure morphology")
        verify_existing(existing, expected)
        return {
            "existing": True,
            "files_written": False,
            "morphology_sha256": existing["morphology_sha256"],
            "morphology_file_sha256": core.sha256_file(OUTPUT_PATH),
            "source_scientific_decision":
                existing["source_scientific_decision"],
        }
    core.require(write, "failure morphology absent; --verify cannot create it")
    document = {
        **expected,
        "morphology_sha256": core.sha256_json(expected),
        "created_at_utc": core.utc_now(),
    }
    core.atomic_write_json(OUTPUT_PATH, document)
    installed = core.load_json(OUTPUT_PATH, "installed Phase983 failure morphology")
    verify_existing(installed, expected)
    core.require(installed == document, "installed morphology serialization changed")
    return {
        "existing": False,
        "files_written": True,
        "morphology_sha256": installed["morphology_sha256"],
        "morphology_file_sha256": core.sha256_file(OUTPUT_PATH),
        "source_scientific_decision": installed["source_scientific_decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.write, args.verify), ensure_ascii=False,
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
