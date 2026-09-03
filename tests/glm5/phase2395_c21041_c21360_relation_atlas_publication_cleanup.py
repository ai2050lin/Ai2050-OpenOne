#!/usr/bin/env python3
"""Publish the semantic/context relation atlas, verify the client, clean raw fields, and audit the stage."""
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
OUT = RESULT / "phase2395_c21041_c21360_relation_atlas_publication_cleanup"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
P2390 = RESULT / "phase2390_c19441_c19760_qwen_semantic_lexical_fullfield"
P2391 = RESULT / "phase2391_c19761_c20080_semantic_lexical_adjudication"
P2392 = RESULT / "phase2392_c20081_c20400_contextual_coordinate_gear_atlas"
P2393 = RESULT / "phase2393_c20401_c20720_frozen_context_causal_adjudication"
P2394 = RESULT / "phase2394_c20721_c21040_crossmodel_contextual_relation_replication"
PHASE = 2395
CAMPAIGN = "C21041-C21360"
MODELS = ("qwen4b", "qwen14b", "glm4", "deepseek7b")
LABELS = {
    "qwen4b": "Qwen3-4B-BF16",
    "qwen14b": "Qwen3-14B-NF4-BF16",
    "glm4": "GLM4-9B-INT8",
    "deepseek7b": "DeepSeek-R1-Distill-Qwen-7B-INT8",
}
PARTITIONS = ("discovery", "confirmation", "fresh_unit_lockbox")
FAMILIES = ("preference", "taxonomy", "temporal", "causal", "comparison", "spatial", "role_binding", "ownership_transfer")
LANGUAGES = ("en", "zh")

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2359_c10321_c11520_qwen_hypergraph_field_campaign as campaign  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def checkpoint(qpoint: int, qcount: int) -> str:
    return "embedding" if qpoint == 0 else "final_norm" if qpoint == qcount - 1 else f"block_{qpoint - 1:02d}_post"


def phase_final(phase: int) -> Path:
    matches = sorted(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
    if len(matches) != 1: raise RuntimeError((phase, matches))
    return matches[0]


def stage_audit() -> dict:
    memo = MEMO.read_text(encoding="utf-8")
    phases = {}
    for phase in range(2388, 2395):
        path = phase_final(phase); data = json.loads(path.read_text(encoding="utf-8"))
        phases[str(phase)] = {
            "path": str(path), "recorded_phase": data.get("phase"),
            "all_checks_passed": data.get("all_checks_passed"),
            "memo_heading_count": memo.count(f"## Phase {phase}:"),
        }
    return {
        "phases": phases,
        "continuous": all(value["recorded_phase"] == int(key) and value["all_checks_passed"] and value["memo_heading_count"] == 1
                          for key, value in phases.items()),
        "attachment_audit": {
            "retained": [
                "Prior independent-sentence fields can match output-preparation states within a model, but the effect must be separated from lexical identity.",
                "Stable physical coordinate organization is a valid within-model descriptive object; coordinate labels are not aligned across models.",
                "Qwen4B layer25/head10 was a frozen observational attention candidate worth testing.",
                "Native chat generation repaired the prior truncation confound but did not close exact semantic behavior.",
            ],
            "corrected_or_rejected": [
                "Phase2380 0.9951 was sentence-end, not pre-sentence; the pre-sentence value was 0.9043.",
                "Deep independent-sentence matching did not establish semantic or compositional texture; the new cross-surface lockboxes failed.",
                "Donor controls did not provide full causal separation.",
                "Attention mass did not establish a router or copy mechanism; the frozen head failed necessity controls.",
                "Diagonal affine coefficients are fitted readout parameters, not demonstrated runtime gears.",
                "No curvature basin, geodesic, diffeomorphism, manifold intervention, or new mathematical closure was measured.",
            ],
        },
    }


def model_base(key: str) -> Path:
    return P2390 / key if key.startswith("qwen") else P2394 / key


def publish_boundary(number: int, key: str) -> dict:
    base = model_base(key); path = base / "raw/semantic_selection_prompt_boundary.float16.npy"
    values = np.load(path, mmap_mode="r"); rows = read_rows(base / "index/selection_rows.jsonl"); metadata = []
    for row in rows:
        for qpoint in range(values.shape[1]):
            metadata.append({
                "case_id": row["case_id"], "group_id": row["group_id"], "family": row["family"],
                "language": row["language"], "unit": row["unit"], "partition": row["partition"],
                "relation_bit": row["relation_bit"], "qpoint": qpoint,
                "checkpoint": checkpoint(qpoint, values.shape[1]), "anchor": "last_prompt_token",
            })
    shape = list(values.shape); close(values)
    return campaign.publish_array(
        f"c{number}_{key}_semantic_context_boundary_full",
        f"{LABELS[key]} semantic-context boundary complete field", path, metadata, LABELS[key],
        "semantic_context_boundary_full_coordinate_v1", "observational context-conditioned field",
        "384 same-lexicon opposite-relation prompts; embedding, every block and final norm",
        "raw embedding/HiddenState activation at every model-local physical coordinate",
        PHASE, CAMPAIGN, np.float16, {"shape3d": shape, "important_result": True},
    )


def publish_reference(number: int, key: str) -> dict:
    base = P2390 / key; source_path = base / "raw/reference_prompt_target_all_token.float16.npy"
    mask_path = base / "raw/reference_prompt_target_mask.uint8.npy"
    source = np.load(source_path, mmap_mode="r"); mask = np.load(mask_path, mmap_mode="r")
    rows = [row for row in read_rows(base / "index/selection_rows.jsonl") if row["partition"] == "fresh_unit_lockbox"][:16]
    valid_total = int(mask.sum()) * source.shape[1]
    dataset_id = f"c{number}_{key}_semantic_reference_all_token_full"
    binary_path = VIS / f"{dataset_id}.float16.npy"
    binary = atlas.create_binary(binary_path.name, valid_total, source.shape[-1], np.float16)
    metadata = []; cursor = 0
    for local, row in enumerate(rows):
        sequence = row["prompt_ids"] + row["target_ids"]
        count = int(mask[local].sum())
        if count != len(sequence): raise RuntimeError((key, local, count, len(sequence)))
        for qpoint in range(source.shape[1]):
            binary[cursor:cursor + count] = source[local, qpoint, :count]
            metadata.extend({
                "case_id": row["case_id"], "family": row["family"], "language": row["language"],
                "unit": row["unit"], "relation_bit": row["relation_bit"], "partition": row["partition"],
                "qpoint": qpoint, "checkpoint": checkpoint(qpoint, source.shape[1]),
                "token_position": position, "token_id": sequence[position],
                "region": "prompt" if position < len(row["prompt_ids"]) else "teacher_forced_target",
            } for position in range(count))
            cursor += count
    if cursor != valid_total: raise RuntimeError((cursor, valid_total))
    binary.flush(); close(binary); close(source); close(mask)
    return atlas.write_metadata(
        dataset_id, f"{LABELS[key]} semantic reference all-token complete field", binary_path, metadata,
        LABELS[key], "semantic_reference_all_token_full_coordinate_v1", "observational complete token field",
        "16 fresh-unit references; padded positions removed; prompt plus teacher-forced target",
        "raw embedding/HiddenState activation for each token, checkpoint and physical coordinate",
        {"phase": PHASE, "campaign": CAMPAIGN, "no_topk": True, "activation_not_parameter": True, "important_result": True},
    )


def relation_paths(key: str) -> tuple[Path, Path]:
    if key.startswith("qwen"):
        base = P2392 / key / "derived"
        return base / "all_layer_partition_relation_response.float32.npy", base / "selected_coordinate_fingerprint.float32.npy"
    base = P2394 / key / "derived"
    return base / "all_layer_partition_relation_response.float32.npy", base / "frozen_context_coordinate_fingerprint.float32.npy"


def publish_relation(number: int, key: str) -> dict:
    response_path, _ = relation_paths(key); values = np.load(response_path, mmap_mode="r"); metadata = []
    for qpoint in range(values.shape[0]):
        for partition in PARTITIONS:
            for family in FAMILIES:
                for language in LANGUAGES:
                    metadata.append({
                        "qpoint": qpoint, "checkpoint": checkpoint(qpoint, values.shape[0]), "partition": partition,
                        "family": family, "language": language, "contrast": "relation_bit_0_minus_1",
                    })
    shape = list(values.shape); close(values)
    return campaign.publish_array(
        f"c{number}_{key}_context_relation_response_all_layer",
        f"{LABELS[key]} contextual relation response by layer", response_path, metadata, LABELS[key],
        "context_relation_response_all_layer_full_coordinate_v1", "observational family-conditioned contrast field",
        "discovery, confirmation and fresh-unit lockbox; eight relation families and two languages",
        "bit0-minus-bit1 response at every model-local physical activation coordinate",
        PHASE, CAMPAIGN, np.float32, {"shape5d": shape, "not_a_causal_gear": True},
    )


def publish_fingerprint(number: int, key: str) -> dict:
    _, fingerprint_path = relation_paths(key); source = np.load(fingerprint_path, mmap_mode="r")
    values = np.asarray(source.T, dtype=np.float32); close(source)
    metadata = []
    for measure in ("signed_normalized", "absolute_normalized"):
        for family in FAMILIES:
            metadata.append({"measure": measure, "family": family, "language_pool": "mean_en_zh",
                             "coordinate_order": "original_model_local_physical"})
    return campaign.publish_array(
        f"c{number}_{key}_context_coordinate_fingerprint",
        f"{LABELS[key]} contextual relation coordinate fingerprint", values, metadata, LABELS[key],
        "context_coordinate_fingerprint_full_coordinate_v1", "descriptive selected-checkpoint coordinate atlas",
        "eight family responses averaged across languages, signed and absolute views",
        "normalized family response for every physical coordinate; no Top-K or coordinate sorting",
        PHASE, CAMPAIGN, np.float32, {"not_a_semantic_neuron_list": True},
    )


def publish_interventions() -> list[dict]:
    assets = []
    conditions = ("clean", "self_patch_selected", "opposite_patch_selected", "opposite_plus_ranked10_rescue",
                  "flip_all_dose_0.5", "flip_all_dose_1.0", "flip_ranked10", "flip_random10", "opposite_patch_wrong_layer")
    for number, key in ((21055, "qwen4b"), (21056, "qwen14b")):
        path = P2393 / key / "raw/semantic_intervention_scores.float32.npy"; values = np.load(path, mmap_mode="r")
        metadata = [{"condition": condition, "lockbox_row": row, "columns": ["target_mean_logprob", "foil_mean_logprob", "margin"]}
                    for condition in conditions for row in range(values.shape[1])]
        close(values)
        assets.append(campaign.publish_array(
            f"c{number}_{key}_frozen_context_intervention_scores", f"{LABELS[key]} frozen context intervention scores",
            path, metadata, LABELS[key], "frozen_context_intervention_score_heatmap_v1", "teacher-forced causal diagnostic",
            "96 fresh-unit rows; frozen layer/coordinate/opposite-state/wrong-layer controls",
            "three score columns: target logprob, foil logprob and target-minus-foil margin",
            PHASE, CAMPAIGN, np.float32, {"causal_gate_passed": False},
        ))
    path = P2393 / "qwen4b/raw/attention_head_intervention_scores.float32.npy"; values = np.load(path, mmap_mode="r")
    attention_conditions = ("clean", "selected_l25h10_dose_0.5", "selected_l25h10_dose_1.0",
                            "wrong_head_l25h11_dose_1.0", "wrong_layer_l24h10_dose_1.0")
    metadata = [{"condition": condition, "lockbox_row": row, "columns": ["target_mean_logprob", "foil_mean_logprob", "margin"]}
                for condition in attention_conditions for row in range(values.shape[1])]
    close(values)
    assets.append(campaign.publish_array(
        "c21057_qwen4b_attention_candidate_intervention_scores", "Qwen3-4B frozen attention-head intervention scores",
        path, metadata, LABELS["qwen4b"], "frozen_attention_intervention_score_heatmap_v1", "teacher-forced causal diagnostic",
        "256 long-sentence lockbox rows; selected head, dose, wrong-head and wrong-layer controls",
        "three score columns; this is an effect matrix, not a HiddenState coordinate field",
        PHASE, CAMPAIGN, np.float32, {"candidate_layer": 25, "candidate_head": 10, "causal_gate_passed": False},
    ))
    return assets


def publish_summary() -> dict:
    p2389 = json.loads((RESULT / "phase2389_c19121_c19440_crossmodel_autonomous_capability/analysis/final.json").read_text(encoding="utf-8"))
    p2391 = json.loads((P2391 / "analysis/final.json").read_text(encoding="utf-8"))
    p2392 = json.loads((P2392 / "analysis/final.json").read_text(encoding="utf-8"))
    p2394 = json.loads((P2394 / "analysis/final.json").read_text(encoding="utf-8"))
    values = []
    for key in MODELS:
        if key.startswith("qwen"):
            static = p2391["comparison"][key]["cross_surface"]
            context = p2392["summary"][key]["full_coordinate_lockbox"]
            cross = p2392["summary"][key]["generalization"]["cross_language_mean"]
            held = p2392["summary"][key]["generalization"]["heldout_family_mean"]
        else:
            static = p2394["summary"][key]["isolated_cross_surface"]
            context = p2394["summary"][key]["context_lockbox"]
            cross = p2394["summary"][key]["cross_language"]
            held = p2394["summary"][key]["heldout_family"]
        autonomous = p2389["comparison"][key]["semantic_exact"]
        values.append([static, context, cross, held, autonomous])
    metadata = [{"model": LABELS[key], "metrics": ["isolated_cross_surface", "context_lockbox", "cross_language", "heldout_family", "autonomous_exact"]}
                for key in MODELS]
    return campaign.publish_array(
        "c21058_fourmodel_relation_evidence_matrix", "Four-model relation evidence matrix", np.asarray(values, dtype=np.float32),
        metadata, "four local models", "fourmodel_relation_evidence_heatmap_v1", "cross-model evidence summary",
        "all rows use frozen discovery/confirmation/fresh-unit rules; physical coordinates are not aligned across models",
        "five accuracy columns; summary metrics, not model parameters or activation coordinates",
        PHASE, CAMPAIGN, np.float32, {"mechanism_closed": False},
    )


def cleanup_raw() -> dict:
    candidates: list[Path] = []
    for key in ("qwen4b", "qwen14b"):
        raw = P2390 / key / "raw"
        candidates.extend(raw / name for name in (
            "independent_end.float16.npy", "independent_mean.float16.npy",
            "reference_prompt_target_all_token.float16.npy", "reference_prompt_target_mask.uint8.npy",
            "semantic_selection_prompt_boundary.float16.npy",
        ))
        candidates.extend((P2391 / key / "derived/selected_embedding_residual.float16.npy",
                           P2391 / key / "derived/selected_relation_response.float32.npy"))
        derived = P2392 / key / "derived"
        candidates.extend(derived / name for name in (
            "all_layer_partition_relation_response.float32.npy", "selected_coordinate_fingerprint.float32.npy",
            "selected_coordinate_group_ids.int32.npy", "confirmation_frozen_coordinate_rank.int32.npy",
            "confirmation_frozen_coordinate_score.float32.npy", "family_response_cosine.float32.npy",
        ))
        candidates.append(P2393 / key / "raw/semantic_intervention_scores.float32.npy")
    candidates.append(P2393 / "qwen4b/raw/attention_head_intervention_scores.float32.npy")
    for key in ("glm4", "deepseek7b"):
        raw = P2394 / key / "raw"; derived = P2394 / key / "derived"
        candidates.extend(raw / name for name in (
            "independent_end.float16.npy", "independent_mean.float16.npy", "semantic_selection_prompt_boundary.float16.npy",
        ))
        candidates.extend(derived / name for name in (
            "all_layer_partition_relation_response.float32.npy", "frozen_context_coordinate_fingerprint.float32.npy",
            "frozen_layer_isolated_surface_response.float32.npy",
        ))
    deleted = []; reclaimed = 0
    for path in candidates:
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents: raise RuntimeError(("unsafe_cleanup", resolved))
        if path.exists():
            size = path.stat().st_size; path.unlink(); reclaimed += size
            deleted.append({"path": str(path), "bytes": size})
    remaining = []
    for base in (P2390, P2391, P2392, P2393, P2394):
        for path in base.rglob("*.npy"):
            if path.stat().st_size > 1_000_000 and any(term in path.name.lower() for term in (
                "hidden", "field", "boundary", "independent", "reference", "response", "fingerprint", "residual"
            )):
                remaining.append(str(path))
    return {"deleted": deleted, "files_deleted": len(deleted), "bytes_reclaimed": reclaimed,
            "remaining_unpublished_hiddenstate_fields": remaining}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    reclaimed = result["cleanup"]["bytes_reclaimed"] / 2**30
    text = rf"""

## Phase {PHASE}: 语义—上下文关系全坐标图谱发布、清理与总审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 审计Phase2388–2394的final、MEMO连续性、附件裁决、四模型顺序、冻结深度和因果对照。把四模型384条关系选择prompt的embedding/每层/final-norm最后token完整物理坐标、Qwen双模型16条全部token参考场、四模型全层族×语言×partition关系响应、四模型坐标指纹、Qwen状态干预、Qwen4 Attention候选干预和四模型证据矩阵发布到客户端`embedding_hiddenstate_full_coordinate`热力图类型。所有资产先校验shape、行元数据、finite、SHA256，再做前端离线构建；只有全部通过后删除结果目录的重复或未发布HiddenState场。

$$\mathrm{{publish}}\Rightarrow\mathrm{{verify}}(\mathrm{{shape}},\mathrm{{rows}},\mathrm{{finite}},\mathrm{{SHA256}}),\qquad
\mathrm{{cleanup}}\Leftarrow\mathrm{{verify}}\land\mathrm{{frontend\ build}}.$$

**结果汇总。** 阶段/附件审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；发布 `{json.dumps(result['publication_summary'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`；清理 `{result['cleanup']['files_deleted']}`个数组、回收 `{reclaimed:.3f}` GiB。

**相关文件。** 脚本 `tests/glm5/phase2395_c21041_c21360_relation_atlas_publication_cleanup.py`；final与清理台账位于 `tests/glm5/result/phase2395_c21041_c21360_relation_atlas_publication_cleanup`；客户端数据集`c21041`–`c21058`位于`frontend/public/vis_data/research_kernel`。未新增或修改其他Markdown。

**理论进展。** 本阶段最可靠拼图是：孤立句不存在跨表达稳定的通用关系向量；Qwen双模型在候选句和查询共同出现时形成可读的、全坐标分布式上下文关系场，但其方向跨语言、跨未见关系族和生成成功均不稳定；GLM4与DS7B在冻结相对深度下更弱，否决四模型普遍性。上下文形成而非静态句向量是下一轮应测的计算对象。

**问题硬伤与结论。** 当前场是按族×语言拟合的外部线性读出，关系响应在confirmation/lockbox的余弦稳定性低；物理坐标只在模型内有意义。相反关系状态替换没有按预期损害行为，冻结10%坐标不能救援，L25/H10消融不优于对照，因此没有必要性、充分性或机制闭合。附件二关于“教科书级因果剥离”、语义纹理、复制路由、仿射齿轮和几何流形的升级表述均被修正。下一阶段总体目标仍是破解语言编码，但即时目标已变化：从本批关系场测绘转为新材料上的“样本条件化、逐token局部更新律”，不能在同一批候选句边界上继续挑层或换坐标；本轮在完成全部计划与同目标自动续研后形成阶段边界。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    audit_result = stage_audit()
    assets = [
        *(publish_boundary(number, key) for number, key in zip(range(21041, 21045), MODELS)),
        publish_reference(21045, "qwen4b"), publish_reference(21046, "qwen14b"),
        *(publish_relation(number, key) for number, key in zip(range(21047, 21051), MODELS)),
        *(publish_fingerprint(number, key) for number, key in zip(range(21051, 21055), MODELS)),
        *publish_interventions(), publish_summary(),
    ]
    verification = [atlas.verify(asset) for asset in assets]
    verified = all(all(value for key, value in item.items() if key != "id") for item in verification)
    catalog = atlas.update_catalog(assets); frontend = atlas.frontend_build()
    if not (verified and frontend["passed"]): raise RuntimeError((verification, frontend))
    cleanup = cleanup_raw()
    checks = {
        "phase_audit_continuous": audit_result["continuous"], "asset_count": len(assets) == 18,
        "assets_verified": verified, "frontend_passed": frontend["passed"],
        "raw_hiddenstate_fields_clean": not cleanup["remaining_unpublished_hiddenstate_fields"],
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "audit": audit_result, "assets": assets,
        "verification": verification,
        "publication_summary": {"dataset_ids": [asset["id"] for asset in assets], "count": len(assets),
                                "catalog": catalog, "frontend": frontend},
        "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values()),
        "next_stage": {
            "same_overall_goal": True, "same_immediate_target": False,
            "immediate_target": "sample-conditioned token-by-token local update laws on new natural language families",
            "reason": "the frozen relation-field atlas, cross-model replication and bounded causal adjudication are complete; more layer selection on this material would reuse the same evidence",
        },
    }
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
