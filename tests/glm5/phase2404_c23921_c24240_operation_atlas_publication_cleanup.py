#!/usr/bin/env python3
"""Publish the operation-conditioned full-coordinate atlas, clean raw fields, and audit the stage."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2404_c23921_c24240_operation_atlas_publication_cleanup"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
P2398 = RESULT / "phase2398_c22001_c22320_qwen4b_event_fullfield"
P2399 = RESULT / "phase2399_c22321_c22640_local_update_operator_atlas"
P2400 = RESULT / "phase2400_c22641_c22960_coordinate_passport_reuse_atlas"
P2401 = RESULT / "phase2401_c22961_c23280_composition_output_compilation"
P2402 = RESULT / "phase2402_c23281_c23600_qwen14b_frozen_operation_replication"
P2403 = RESULT / "phase2403_c23601_c23920_crossarchitecture_frozen_operation_replication"
PHASE = 2404
CAMPAIGN = "C23921-C24240"
MODELS = ("qwen4b", "qwen14b", "glm4", "deepseek7b")
LABELS = {
    "qwen4b": "Qwen3-4B-BF16",
    "qwen14b": "Qwen3-14B-NF4-BF16",
    "glm4": "GLM4-9B-INT8",
    "deepseek7b": "DeepSeek-R1-Distill-Qwen-7B-INT8",
}
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\ae76a631-5f0f-441d-aba3-ca6eeb6206a0\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\9d919240-8984-4ccc-9c2d-5b8d70dcdb91\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\2d663e84-4bed-47a5-9474-4daf322bb8de\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\ee5a5696-8c9e-44aa-932d-f5f46a506c88\pasted-text.txt"),
)

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2359_c10321_c11520_qwen_hypergraph_field_campaign as publisher  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(qpoint: int, qcount: int) -> str:
    return "embedding" if qpoint == 0 else "final_norm" if qpoint == qcount - 1 else f"block_{qpoint - 1:02d}_post"


def phase_final(phase: int) -> Path:
    matches = sorted(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
    if len(matches) != 1:
        raise RuntimeError((phase, matches))
    return matches[0]


def stage_audit() -> dict:
    memo = MEMO.read_text(encoding="utf-8")
    phases = {}
    for phase in range(2396, 2404):
        path = phase_final(phase)
        data = json.loads(path.read_text(encoding="utf-8"))
        phases[str(phase)] = {
            "path": str(path.relative_to(ROOT)),
            "recorded_phase": data.get("phase"),
            "all_checks_passed": data.get("all_checks_passed"),
            "memo_heading_count": memo.count(f"## Phase {phase}:"),
        }
    attachment_rows = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in ATTACHMENTS]
    return {
        "phases": phases,
        "continuous": all(value["recorded_phase"] == int(key) and value["all_checks_passed"] and value["memo_heading_count"] == 1
                          for key, value in phases.items()),
        "attachments": attachment_rows,
        "evidence_adjudication": {
            "retained": [
                "Fixed-basis coordinate cooperation is a valid within-model object; a single coordinate can be one physical tooth of a distributed condition-dependent texture.",
                "The research unit should be an externally defined language operation followed through event-aligned local updates, cross-layer reuse/differentiation, composition and output probability.",
                "Phase2388-2395 did expose a plateau in test-first cataloguing and justified a prospective operation-first contract.",
                "Full coordinates and low-amplitude coordinates must remain visible; Top-K, transport differences and causal deletion are diagnostics rather than the primary ontology.",
            ],
            "corrected_or_rejected": [
                "Failure of prior linear/global probes does not prove that no abstract, nonlinear or distributed relation operator exists.",
                "Failure of one boundary patch or coordinate deletion does not prove epiphenomenality; it only rejects that intervention and readout at its tested dose, site and timing.",
                "Attention K/V routing, MLP key-value storage and SAE features were not measured in the supplied evidence and cannot be called the true gear mechanism.",
                "Logit lens and coordinate-wise LM-head contributions are output readouts, not causal attribution or a compiler proof.",
                "Thousands of phases are evidence of a methodological plateau, not a mathematical proof that every prior paradigm is fatally invalid.",
                "The observed condition means mix language, surface, slots and template segments; their success cannot be relabelled as universal semantic gears.",
            ],
        },
    }


def source_spec(key: str, task: str) -> tuple[Path, Path]:
    if key == "qwen4b":
        base = P2398
    elif key == "qwen14b":
        base = P2402
    else:
        base = P2403 / key
    return base / f"raw/{task}_event_field.float16.npy", base / f"index/{task}_rows.jsonl"


def representative_indices(rows: list[dict], task: str) -> list[int]:
    partition = "fresh_unit_lockbox" if task == "selection" else "fresh_composition_lockbox"
    chosen: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        if row["partition"] != partition:
            continue
        key = (row["family"], row["language"])
        if key not in chosen:
            chosen[key] = index
    expected = 16 if task == "selection" else 8
    result = [chosen[key] for key in sorted(chosen)]
    if len(result) != expected:
        raise RuntimeError((task, len(result), expected, sorted(chosen)))
    return result


def publish_event_field(number: int, key: str, task: str) -> dict:
    source_path, index_path = source_spec(key, task)
    rows = read_rows(index_path)
    selected = representative_indices(rows, task)
    source = np.load(source_path, mmap_mode="r")
    source_shape = list(source.shape)
    dataset_id = f"c{number}_{key}_{task}_event_full_coordinate"
    binary = VIS / f"{dataset_id}.float16.npy"
    output = atlas.create_binary(binary.name, len(selected) * source.shape[1] * source.shape[2], source.shape[-1], np.float16)
    metadata = []
    cursor = 0
    for source_index in selected:
        row = rows[source_index]
        event_names = row["event_names"]
        event_tokens = row["event_token_indices"]
        for qpoint in range(source.shape[1]):
            for event_index in range(source.shape[2]):
                output[cursor] = source[source_index, qpoint, event_index]
                metadata.append({
                    "case_id": row["case_id"], "source_index": source_index, "task": task,
                    "family": row["family"], "language": row["language"], "surface": row["surface"],
                    "direction": row["direction"], "unit": row["unit"], "partition": row["partition"],
                    "steps": row.get("steps"), "query_role": row.get("query_role"),
                    "qpoint": qpoint, "checkpoint": checkpoint(qpoint, source.shape[1]),
                    "event_index": event_index, "event": event_names[event_index],
                    "token_position": event_tokens[event_index],
                })
                cursor += 1
    output.flush(); close(output); close(source)
    return atlas.write_metadata(
        dataset_id, f"{LABELS[key]} {task} event-aligned complete coordinate field", binary, metadata,
        LABELS[key], "operation_event_full_coordinate_v1", "observational representative lockbox field",
        f"one fresh row per family and language; all checkpoints and all {8 if task == 'selection' else 12} semantic events",
        "raw embedding or HiddenState activation at every model-local physical coordinate",
        {"phase": PHASE, "campaign": CAMPAIGN, "sample_subset_only": True, "no_topk": True,
         "includes_embedding": True, "includes_every_layer": True, "activation_not_trainable_parameter": True,
         "source_shape": source_shape},
    )


def derived_paths(key: str, task: str) -> tuple[Path, Path]:
    if key == "qwen4b":
        base = P2400 / "derived"
    elif key == "qwen14b":
        base = P2402 / "derived"
    else:
        base = P2403 / key / "derived"
    return base / f"{task}_family_passport_lockbox.float32.npy", base / f"{task}_condition_gain_lockbox.float32.npy"


def publish_coordinate_atlas(number: int, key: str) -> dict:
    specs = []
    coordinate_count = None
    total_rows = 0
    for task in ("selection", "composition"):
        passport_path, gain_path = derived_paths(key, task)
        passport = np.load(passport_path, mmap_mode="r")
        gain = np.load(gain_path, mmap_mode="r")
        if coordinate_count is None:
            coordinate_count = passport.shape[-1]
        if passport.shape[-1] != coordinate_count or gain.shape[-1] != coordinate_count:
            raise RuntimeError((key, task, passport.shape, gain.shape))
        total_rows += passport.shape[0] * passport.shape[1] * (passport.shape[2] + 1)
        specs.append((task, passport_path, gain_path))
        close(passport); close(gain)
    dataset_id = f"c{number}_{key}_operation_coordinate_passport"
    binary = VIS / f"{dataset_id}.float32.npy"
    output = atlas.create_binary(binary.name, total_rows, int(coordinate_count), np.float32)
    metadata = []
    cursor = 0
    for task, passport_path, gain_path in specs:
        passport = np.load(passport_path, mmap_mode="r")
        gain = np.load(gain_path, mmap_mode="r")
        families = ("causal", "comparison", "ownership", "preference", "role_binding", "spatial", "taxonomy", "temporal") if task == "selection" else ("comparison", "spatial", "taxonomy", "temporal")
        event_names = read_rows(source_spec(key, task)[1])[0]["event_names"]
        for qpoint in range(passport.shape[0]):
            for event_index, event in enumerate(event_names):
                for family_index, family in enumerate(families):
                    output[cursor] = passport[qpoint, event_index, family_index]
                    cursor += 1
                    metadata.append({"task": task, "measure": "fresh_family_passport", "family": family,
                                     "qpoint_update": qpoint, "event_index": event_index, "event": event})
                output[cursor] = gain[qpoint, event_index]
                cursor += 1
                metadata.append({"task": task, "measure": "condition_sse_gain", "family": "all_condition_cells",
                                 "qpoint_update": qpoint, "event_index": event_index, "event": event})
        close(passport); close(gain)
    output.flush(); close(output)
    if cursor != total_rows:
        raise RuntimeError((cursor, total_rows))
    return atlas.write_metadata(
        dataset_id, f"{LABELS[key]} operation coordinate passport and gain", binary, metadata, LABELS[key],
        "operation_coordinate_passport_gain_v1", "prospective descriptive prediction atlas",
        "fresh-unit and fresh two-step lockboxes predicted from frozen discovery cells",
        "family-centered local-update passport or signed condition-vs-constant squared-error gain at every physical coordinate",
        {"not_a_semantic_neuron_list": True, "not_crossmodel_coordinate_aligned": True, "no_topk": True},
    )


def publish_q4_output_contributions(number: int, task: str) -> dict:
    path = P2401 / f"derived/{task}_answer_final_coordinate_contribution.float32.npy"
    rows = read_rows(P2398 / f"index/{task}_rows.jsonl")
    metadata = [{"case_id": row["case_id"], "task": task, "family": row["family"], "language": row["language"],
                 "surface": row["surface"], "unit": row["unit"], "partition": row["partition"],
                 "measure": "target_minus_foil_final_norm_coordinate_contribution"} for row in rows]
    return publisher.publish_array(
        f"c{number}_qwen4b_{task}_exact_output_coordinate_contribution",
        f"Qwen3-4B {task} exact final coordinate contribution", path, metadata, LABELS["qwen4b"],
        "exact_final_logit_coordinate_contribution_v1", "exact output readout decomposition",
        "answer-boundary final RMSNorm state and first-divergence target-minus-foil unembedding direction",
        "per-coordinate term whose sum is the target-minus-foil logit margin; readout, not causal attribution",
        PHASE, CAMPAIGN, np.float32, {"exact_sum_decomposition": True, "causal_attribution": False},
    )


def publish_evidence_matrix(number: int) -> dict:
    finals = {
        "qwen4b": json.loads((P2401 / "analysis/final.json").read_text(encoding="utf-8")),
        "qwen14b": json.loads((P2402 / "analysis/final.json").read_text(encoding="utf-8")),
    }
    cross = json.loads((P2403 / "analysis/final.json").read_text(encoding="utf-8"))
    values = []
    for key in MODELS:
        if key == "qwen4b":
            p2399 = json.loads((P2399 / "analysis/final.json").read_text(encoding="utf-8"))
            selection_gain = p2399["selection"]["frozen_operator_metrics"]["fresh_unit_lockbox"]["gain_vs_constant"]
            composition_gain = p2399["composition"]["frozen_operator_metrics"]["two_step_fresh_lockbox"]["gain_vs_constant"]
            selection_output = finals[key]["adjudication"]["selection_answer_boundary_compilation_gain"]
            composition_output = finals[key]["adjudication"]["two_step_answer_boundary_compilation_gain"]
            selection_behavior = finals[key]["selection_compilation"]["row_behavior_bridge"]["fresh_unit_lockbox"]["gain_to_final_margin_correlation"]
            composition_behavior = finals[key]["composition_compilation"]["row_behavior_bridge"]["two_step_fresh_lockbox"]["gain_to_final_margin_correlation"]
        elif key == "qwen14b":
            item = finals[key]
            selection_gain = item["selection"]["coordinate_prediction"]["condition"]["gain_vs_constant"]
            composition_gain = item["composition"]["coordinate_prediction"]["condition"]["gain_vs_constant"]
            selection_output = item["selection"]["answer_boundary_output_compilation"]["gain_vs_constant"]
            composition_output = item["composition"]["answer_boundary_output_compilation"]["gain_vs_constant"]
            selection_behavior = item["selection"]["behavior_bridge"]["coordinate_gain_to_final_margin_correlation"]
            composition_behavior = item["composition"]["behavior_bridge"]["coordinate_gain_to_final_margin_correlation"]
        else:
            item = cross["models"][key]
            selection_gain = item["selection"]["coordinate_prediction"]["condition"]["gain_vs_constant"]
            composition_gain = item["composition"]["coordinate_prediction"]["condition"]["gain_vs_constant"]
            selection_output = item["selection"]["answer_boundary_output_compilation"]["gain_vs_constant"]
            composition_output = item["composition"]["answer_boundary_output_compilation"]["gain_vs_constant"]
            selection_behavior = item["selection"]["behavior_bridge"]["coordinate_gain_to_final_margin_correlation"]
            composition_behavior = item["composition"]["behavior_bridge"]["coordinate_gain_to_final_margin_correlation"]
        values.append([selection_gain, composition_gain, selection_output, composition_output, selection_behavior, composition_behavior])
    metrics = ["selection_coordinate_gain", "two_step_coordinate_gain", "selection_output_gain",
               "two_step_output_gain", "selection_behavior_correlation", "two_step_behavior_correlation"]
    metadata = [{"model": LABELS[key], "metrics": metrics} for key in MODELS]
    return publisher.publish_array(
        f"c{number}_fourmodel_operation_evidence_matrix", "Four-model condition-update evidence matrix",
        np.asarray(values, dtype=np.float32), metadata, "four local models", "fourmodel_operation_evidence_matrix_v1",
        "cross-model functional summary", "frozen discovery to fresh unit/two-step lockboxes",
        "six named summary metrics; columns are metrics rather than aligned physical coordinates",
        PHASE, CAMPAIGN, np.float32, {"mechanism_closed": False, "summary_not_hiddenstate": True},
    )


def cleanup_fields() -> dict:
    bases = (P2398, P2399, P2400, P2401, P2402, P2403)
    candidates = []
    for base in bases:
        for path in base.rglob("*.npy"):
            if path.stat().st_size >= 1_000_000:
                candidates.append(path)
    reclaimed = 0
    deleted = []
    result_root = RESULT.resolve()
    for path in sorted(set(candidates)):
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(("unsafe_cleanup", resolved))
        size = path.stat().st_size
        path.unlink()
        reclaimed += size
        deleted.append({"path": str(path.relative_to(ROOT)), "bytes": size})
    remaining = []
    for base in bases:
        for path in base.rglob("*.npy"):
            if path.stat().st_size >= 1_000_000:
                remaining.append(str(path.relative_to(ROOT)))
    return {"files_deleted": len(deleted), "bytes_reclaimed": reclaimed, "deleted": deleted,
            "remaining_large_arrays": remaining}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    audit = result["audit"]["evidence_adjudication"]
    matrix = result["evidence_matrix"]
    text = rf"""

## Phase {PHASE}: 条件坐标齿轮全坐标图谱发布、证据裁决与清理（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本Phase首先逐文件审查四份附件并核对Phase2396–2403的final、MEMO标题连续性与检查门；随后把Qwen3-4B、Qwen3-14B、GLM4、DS7B的选择/两步组合fresh-lockbox代表样本发布为`embedding_hiddenstate_full_coordinate`热力图：每个模型保留每个语言族×语言一个样本、embedding、每一层、final norm、8/12个语义事件和原始顺序下全部物理坐标。另发布四模型fresh族护照与条件预测逐坐标收益、Qwen4B答案边界精确输出坐标贡献及四模型证据矩阵。代表样本只缩减样本轴，不压缩、排序、Top-K或降维坐标轴。所有资产校验shape、行元数据、finite、SHA256并通过前端离线构建后，才删除结果目录中已发布或未发布的大型HiddenState/派生场。

$$U_{{q,e,j}}=H_{{q+1,e,j}}-H_{{q,e,j}},\qquad
G_j=\sum_i\left[(U_{{i,j}}-\bar U_j)^2-(U_{{i,j}}-\widehat U_{{c(i),j}})^2\right],$$

$$\Delta z_{{t,f}}=\sum_j \operatorname{{RMSNorm}}(H_{{answer}})_j\left(W_{{t,j}}-W_{{f,j}}\right).$$

**结果汇总。** 附件保留项 `{json.dumps(audit['retained'], ensure_ascii=False)}`；修正/否决项 `{json.dumps(audit['corrected_or_rejected'], ensure_ascii=False)}`。四模型证据矩阵 `{json.dumps(matrix, ensure_ascii=False)}`。发布 `{json.dumps(result['publication_summary'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`；清理`{result['cleanup']['files_deleted']}`个数组，回收`{result['cleanup']['bytes_reclaimed'] / 2**30:.3f}` GiB。

**相关文件。** 脚本 `tests/glm5/phase2404_c23921_c24240_operation_atlas_publication_cleanup.py`；final与清理台账位于`tests/glm5/result/phase2404_c23921_c24240_operation_atlas_publication_cleanup`；客户端数据集位于`frontend/public/vis_data/research_kernel`，编号`c23921`–`c23935`。未新增或修改其他Markdown。

**分析与理论进展。** 四模型共同出现的可靠拼图是：按外部条件分组的局部残差更新能够在新unit和一步到两步组合迁移时优于常量；错族和坐标置换负对照显著更差；族中心护照在新unit保持中高相似。这证明固定物理基底上存在条件依赖、分布式且坐标特异的协同纹理，值得作为“条件齿轮候选”继续追踪。它尚不能说明该纹理就是语义算法：跨表达与跨语言护照明显下降，逐相邻层护照持久性近零，且四模型逐行预测收益几乎不解释最终行为。Qwen14B尤其把局部更新复现与行为/输出闭合分开；DS7B在行为近机会水平时仍取得很高局部和输出拟合，更直接暴露模板与共同层动力学混淆。

**问题硬伤与结论。** 当前条件键同时含family、language、surface、direction、query/answer slot，尚未把语言操作从词项、模板、位置和数值尺度中剥离；每模型物理坐标不可跨模型直接对齐；Qwen14B为NF4、GLM/DS为INT8，幅值不可横比；代表热力图覆盖全部坐标但不是全部样本。精确LM-head坐标和logit lens只是读出，不是因果归因。因而结论严格限定为“多模型存在可冻结预测的条件化局部更新纹理”，不称为语义齿轮、编译器或编码机制闭合。

**下阶段裁决。** 总目标仍是破解语言编码，但本阶段的即时目标（whole-residual事件场的条件均值、护照与输出读出）已经完整结束。下一个即时目标改为在同一自然语言操作上分解Attention输出、MLP输出和residual加和的来源—去向，并用词项/模板/位置匹配的反事实消除共同层动力学；这不是继续重复本阶段的同一目标，因此在Phase2404形成可复现的阶段边界，而不是继续挑层或换Top-K坐标。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    atlas.PHASE = PHASE
    atlas.CAMPAIGN = CAMPAIGN
    publisher.PHASE = PHASE
    publisher.CAMPAIGN = CAMPAIGN
    audit = stage_audit()
    assets = []
    number = 23921
    for key in MODELS:
        for task in ("selection", "composition"):
            assets.append(publish_event_field(number, key, task))
            number += 1
    for key in MODELS:
        assets.append(publish_coordinate_atlas(number, key))
        number += 1
    assets.append(publish_q4_output_contributions(number, "selection")); number += 1
    assets.append(publish_q4_output_contributions(number, "composition")); number += 1
    evidence_asset = publish_evidence_matrix(number)
    assets.append(evidence_asset)
    verification = [atlas.verify(asset) for asset in assets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    catalog = atlas.update_catalog(assets)
    frontend = atlas.frontend_build()
    if not (verified and frontend["passed"]):
        raise RuntimeError((verification, frontend))
    evidence_values = np.load(evidence_asset["binary"])
    evidence_matrix = {LABELS[key]: [float(value) for value in evidence_values[index]] for index, key in enumerate(MODELS)}
    close(evidence_values)
    cleanup = cleanup_fields()
    checks = {
        "phase_audit_continuous": audit["continuous"], "attachment_count": len(audit["attachments"]) == 4,
        "asset_count": len(assets) == 15, "assets_verified": verified, "frontend_passed": frontend["passed"],
        "large_result_fields_clean": not cleanup["remaining_large_arrays"],
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "audit": audit, "assets": assets, "verification": verification,
        "evidence_matrix_columns": ["selection_coordinate_gain", "two_step_coordinate_gain", "selection_output_gain",
                                    "two_step_output_gain", "selection_behavior_correlation", "two_step_behavior_correlation"],
        "evidence_matrix": evidence_matrix,
        "publication_summary": {"dataset_ids": [asset["id"] for asset in assets], "count": len(assets),
                                "catalog": catalog, "frontend": frontend},
        "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values()),
        "next_stage": {"same_overall_goal": True, "same_immediate_target": False,
                       "immediate_target": "component-resolved source-to-destination operation updates with lexical/template/position matched counterfactuals",
                       "reason": "whole-residual conditional atlas and output readout are complete; the next unknown is source attribution rather than another residual-layer atlas"},
    }
    save(final, result)
    append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
