#!/usr/bin/env python3
"""Publish exact-coordinate label-free atlases, clean raw fields, and audit the campaign."""
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
OUT = RESULT / "phase2387_c18481_c18800_publication_cleanup_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
P2378 = RESULT / "phase2378_c15601_c15920_label_free_binding_contract"
P2379 = RESULT / "phase2379_c15921_c16240_qwen_label_free_full_field"
P2380 = RESULT / "phase2380_c16241_c16560_object_slot_progress_adjudication"
P2381 = RESULT / "phase2381_c16561_c16880_residual_component_routing"
P2382 = RESULT / "phase2382_c16881_c17200_crossmodel_label_free_binding"
P2383 = RESULT / "phase2383_c17201_c17520_content_position_confound"
P2384 = RESULT / "phase2384_c17521_c17840_isolated_sentence_content_field"
P2385 = RESULT / "phase2385_c17841_c18160_crossmodel_isolated_content"
P2386 = RESULT / "phase2386_c18161_c18480_chat_generation_closure"
PHASE = 2387
CAMPAIGN = "C18481-C18800"

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
    memo = MEMO.read_text(encoding="utf-8"); phases = {}
    for phase in range(2378, 2387):
        path = phase_final(phase); data = json.loads(path.read_text(encoding="utf-8"))
        phases[str(phase)] = {"path": str(path), "recorded_phase": data.get("phase"), "all_checks_passed": data.get("all_checks_passed"),
                              "memo_heading_count": memo.count(f"## Phase {phase}:")}
    return {"phases": phases, "continuous": all(value["recorded_phase"] == int(key) and value["all_checks_passed"] and
                                                  value["memo_heading_count"] == 1 for key, value in phases.items()),
            "retained": [
                "Label-free complete-sequence preference is behaviorally positive in all four tested models, with model-specific strength.",
                "A sentence encoded independently of source position can be conditionally matched to the teacher-forced output-preparation state in all four models.",
                "Cross-content donors and row-specific coordinate permutations sharply reduce matching, so content and consistent within-model coordinate texture both matter.",
                "Qwen4B and Qwen14B show confirmation-selected deep-layer gains over embedding means; GLM4 and DS7B are best explained by embedding means on this material.",
                "Qwen4B attention routing is a live candidate: one frozen head reaches 0.6191 source-span mass accuracy, while MLP output/intermediate centroid gates do not pass.",
                "Native chat generation fixes truncation and raises exact four-line output to 0.40625, but reverse and coreference behavior remain weak.",
            ],
            "rejected_or_corrected": [
                "Source-slot decodability does not establish packaged objects or an autonomous pointer machine.",
                "Fixed global coordinate permutation can be re-fit successfully; absolute coordinate labels are not an intrinsic semantic codebook.",
                "Row-specific coordinate permutation failing supports stable coordinate-indexed texture, not one-coordinate/one-concept semantics.",
                "Attention mass and attention-output decodability are observational route candidates, not proof of copying or causal responsibility.",
                "MLP output and all 9728 intermediate activations fail the frozen centroid gate; the claim that diagonal affine equals an MLP beta gate is rejected.",
                "The original generation failure was protocol-confounded; the repaired result improves greatly but still does not close content behavior.",
            ]}


def publish_all_token() -> dict:
    source = np.load(P2379 / "raw/qwen4b_reference_prompt_target_all_token.float16.npy", mmap_mode="r")
    rows = read_rows(P2379 / "index/reference_all_token_rows.jsonl"); total = sum(row["token_count"] * source.shape[1] for row in rows)
    binary_path = VIS / "c18481_qwen4b_label_free_prompt_output_all_token.float16.npy"
    binary = atlas.create_binary(binary_path.name, total, source.shape[-1], np.float16); metadata = []; cursor = 0
    for local, row in enumerate(rows):
        for qpoint in range(source.shape[1]):
            count = row["token_count"]; binary[cursor:cursor + count] = source[local, qpoint, :count]
            metadata.extend({"case_id": row["case_id"], "family": row["family"], "language": row["language"], "unit": row["unit"],
                             "surface": row["surface"], "reverse": row["reverse"], "source_perm": row["source_perm"],
                             "qpoint": qpoint, "checkpoint": checkpoint(qpoint, source.shape[1]), "token_position": position,
                             "token_id": row["token_ids"][position], "token": row["tokens"][position],
                             "region": "prompt" if position < row["prompt_token_count"] else "teacher_forced_output"}
                            for position in range(count))
            cursor += count
    binary.flush(); close(binary); close(source)
    return atlas.write_metadata("c18481_qwen4b_label_free_prompt_output_all_token", "Qwen3-4B label-free prompt/output all-token field",
        binary_path, metadata, "Qwen3-4B-BF16", "label_free_prompt_output_all_token_v1", "observational complete token field",
        "16 fresh joint-lockbox references; padded positions removed", "raw word embedding or HiddenState activation at every physical coordinate",
        {"phase": PHASE, "campaign": CAMPAIGN, "no_topk": True, "activation_not_parameter": True, "important_result": True})


def publish_output_lockbox() -> dict:
    all_rows = [row for row in read_rows(P2378 / "material/label_free_natural_binding.jsonl") if row["task"] == "exact_copy"]
    indices = [i for i, row in enumerate(all_rows) if row["partition"] == "fresh_joint_lockbox"]
    source = np.load(P2379 / "raw/qwen4b_output_progress_anchors.float16.npy", mmap_mode="r")
    row_count = len(indices) * source.shape[1] * source.shape[2] * source.shape[3]
    binary_path = VIS / "c18482_qwen4b_label_free_output_progress_lockbox.float16.npy"
    binary = atlas.create_binary(binary_path.name, row_count, source.shape[-1], np.float16); metadata = []; cursor = 0
    offset_names = ("pre_sentence", "early_token_2", "sentence_end")
    for index in indices:
        row = all_rows[index]; block = np.asarray(source[index], dtype=np.float16).reshape(-1, source.shape[-1])
        binary[cursor:cursor + len(block)] = block; cursor += len(block)
        for target_slot, sentence_id in enumerate(row["target_order"]):
            source_slot = row["source_perm"].index(sentence_id)
            for offset, offset_name in enumerate(offset_names):
                for qpoint in range(source.shape[3]):
                    metadata.append({"case_id": row["case_id"], "family": row["family"], "unit": row["unit"],
                        "language": row["language"], "surface": row["surface"], "reverse": row["reverse"],
                        "source_perm": row["source_perm"], "target_slot": target_slot, "sentence_id": sentence_id,
                        "source_slot": source_slot, "offset": offset, "offset_name": offset_name, "qpoint": qpoint,
                        "checkpoint": checkpoint(qpoint, source.shape[3])})
    binary.flush(); close(binary); close(source)
    return atlas.write_metadata("c18482_qwen4b_label_free_output_progress_lockbox", "Qwen3-4B label-free output-progress lockbox",
        binary_path, metadata, "Qwen3-4B-BF16", "label_free_output_progress_full_coordinate_v1",
        "teacher-forced dynamic observational field", "256 fresh unit plus unseen source-permutation rows",
        "raw HiddenState activation at pre/early/end output anchors in every physical coordinate",
        {"phase": PHASE, "campaign": CAMPAIGN, "no_topk": True, "activation_not_parameter": True, "shape5d_subset": [256, 4, 3, 38, 2560]})


def panel_rows() -> list[dict]:
    all_rows = [row for row in read_rows(P2378 / "material/label_free_natural_binding.jsonl") if row["task"] == "exact_copy"]
    index = read_rows(P2381 / "index/component_panel_rows.jsonl")
    return [all_rows[int(row["source_index"])] for row in index]


def publish_isolated(dataset_id: str, title: str, path: Path, summary: str) -> dict:
    rows = panel_rows(); values = np.load(path, mmap_mode="r"); metadata = []
    for row in rows:
        for sentence_id in range(4):
            for qpoint in range(values.shape[2]):
                metadata.append({"case_id": row["case_id"], "family": row["family"], "unit": row["unit"],
                    "language": row["language"], "surface": row["surface"], "partition": row["partition"],
                    "sentence_id": sentence_id, "sentence": row["sentences"][sentence_id], "qpoint": qpoint,
                    "checkpoint": checkpoint(qpoint, values.shape[2])})
    close(values)
    return campaign.publish_array(dataset_id, title, path, metadata, "Qwen3-4B-BF16", "isolated_sentence_full_coordinate_v1",
        "position-isolated observational content field", "768 rows x four isolated natural sentences x every checkpoint",
        summary, PHASE, CAMPAIGN, np.float16, {"important_result": True})


def publish_components() -> list[dict]:
    rows = panel_rows(); assets = []
    component_path = P2381 / "raw/qwen4b_pre_sentence_attention_mlp.float16.npy"; values = np.load(component_path, mmap_mode="r"); metadata = []
    for row in rows:
        for target_slot, sentence_id in enumerate(row["target_order"]):
            for layer in range(values.shape[2]):
                for component in ("attention_output", "mlp_output"):
                    metadata.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                        "surface": row["surface"], "partition": row["partition"], "reverse": row["reverse"],
                        "target_slot": target_slot, "sentence_id": sentence_id, "source_slot": row["source_perm"].index(sentence_id),
                        "layer": layer, "component": component, "anchor": "pre_sentence"})
    close(values)
    assets.append(campaign.publish_array("c18485_qwen4b_pre_sentence_attention_mlp", "Qwen3-4B pre-sentence Attention/MLP components",
        component_path, metadata, "Qwen3-4B-BF16", "pre_sentence_residual_components_v1", "observational component field",
        "768-row panel; every target slot and all 36 layers", "raw Attention or MLP residual update at every physical hidden coordinate",
        PHASE, CAMPAIGN, np.float16, {"attention_slot_lockbox": 0.3935546875, "mlp_slot_lockbox": 0.2451171875}))
    gate_path = P2381 / "raw/qwen4b_pre_sentence_mlp_intermediate.float16.npy"; gates = np.load(gate_path, mmap_mode="r"); metadata = []
    for row in rows:
        for target_slot, sentence_id in enumerate(row["target_order"]):
            for layer in range(gates.shape[2]): metadata.append({"case_id": row["case_id"], "family": row["family"],
                "language": row["language"], "surface": row["surface"], "partition": row["partition"],
                "reverse": row["reverse"], "target_slot": target_slot, "sentence_id": sentence_id,
                "source_slot": row["source_perm"].index(sentence_id), "layer": layer, "component": "silu_gate_times_up",
                "anchor": "pre_sentence"})
    close(gates)
    assets.append(campaign.publish_array("c18486_qwen4b_pre_sentence_mlp_intermediate", "Qwen3-4B pre-sentence complete MLP intermediate",
        gate_path, metadata, "Qwen3-4B-BF16", "pre_sentence_mlp_intermediate_v1", "observational complete intermediate field",
        "768-row panel; all 9728 intermediate neurons in all 36 layers", "raw SwiGLU product before down projection; activation, not a beta coefficient",
        PHASE, CAMPAIGN, np.float16, {"mlp_intermediate_slot_lockbox": 0.2294921875}))
    return assets


def publish_diagnostics() -> list[dict]:
    assets = []
    r2_path = P2380 / "derived/diagonal_object_match_coordinate_r2.float32.npy"; r2 = np.load(r2_path, mmap_mode="r"); metadata = []
    for offset, name in enumerate(("pre_sentence", "early_token_2", "sentence_end")):
        for qpoint in range(r2.shape[1]):
            for method in ("global_diagonal", "output_direction_conditioned_diagonal"):
                metadata.append({"offset": offset, "offset_name": name, "qpoint": qpoint, "checkpoint": checkpoint(qpoint, r2.shape[1]), "method": method})
    close(r2)
    assets.append(campaign.publish_array("c18487_qwen4b_object_match_coordinate_r2", "Qwen3-4B object-match coordinate R2",
        r2_path, metadata, "Qwen3-4B-BF16", "object_match_coordinate_r2_v1", "held-out coordinate diagnostic",
        "three output offsets x all checkpoints x two diagonal models", "per-physical-coordinate lockbox R2; not a neuron causal score",
        PHASE, CAMPAIGN, np.float32))
    for number, name, filename, width in ((18488, "attention output", "attention_output_coordinate_slot_eta2.float32.npy", 2560),
                                          (18489, "MLP output", "mlp_output_coordinate_slot_eta2.float32.npy", 2560),
                                          (18490, "MLP intermediate", "mlp_intermediate_coordinate_slot_eta2.float32.npy", 9728)):
        path = P2381 / "derived" / filename; values = np.load(path, mmap_mode="r")
        metadata = [{"layer": layer, "component": name, "meaning": "coordinate eta-squared for source-slot label on discovery panel"}
                    for layer in range(values.shape[0])]; close(values)
        assets.append(campaign.publish_array(f"c{number}_qwen4b_{filename.split('.')[0]}", f"Qwen3-4B {name} coordinate slot eta2",
            path, metadata, "Qwen3-4B-BF16", "component_coordinate_slot_eta2_v1", "descriptive coordinate association",
            f"all 36 layers x all {width} component coordinates", "eta-squared at every component coordinate; finite-sample descriptive",
            PHASE, CAMPAIGN, np.float32))
    return assets


def publish_routing_subset() -> dict:
    routing_rows = read_rows(P2381 / "index/routing_rows.jsonl"); source = np.load(P2381 / "raw/qwen4b_pre_sentence_source_attention_mass.float16.npy", mmap_mode="r")
    indices = [i for i, row in enumerate(routing_rows) if row["partition"] == "fresh_joint_lockbox"][:32]
    values = np.asarray(source[indices], dtype=np.float16); close(source); metadata = []
    for index in indices:
        row = routing_rows[index]
        for layer in range(values.shape[1]):
            for target_slot in range(4):
                for head in range(values.shape[3]): metadata.append({"case_id": row["case_id"], "family": row["family"],
                    "language": row["language"], "surface": row["surface"], "reverse": row["reverse"],
                    "source_perm": row["source_perm"], "layer": layer, "target_slot": target_slot, "head": head,
                    "selected_candidate": layer == 25 and head == 10})
    return campaign.publish_array("c18491_qwen4b_attention_source_span_mass", "Qwen3-4B all-head source-sentence attention mass",
        values, metadata, "Qwen3-4B-BF16", "attention_source_span_mass_v1", "observational attention routing subset",
        "32 fresh lockbox rows x all layers x all target slots x all heads", "four coordinates are exact attention mass to four complete source-sentence spans",
        PHASE, CAMPAIGN, np.float16, {"selected_head_lockbox_accuracy": 0.619140625})


def publish_crossmodel_bridges() -> list[dict]:
    final = json.loads((P2385 / "analysis/final.json").read_text(encoding="utf-8")); assets = []
    labels = {"qwen14b": "Qwen3-14B-NF4", "glm4": "GLM4-9B-INT8", "deepseek7b": "DS-R1-Distill-Qwen-7B-INT8"}
    for number, key in enumerate(("qwen14b", "glm4", "deepseek7b"), start=18492):
        rows = read_rows(P2382 / key / "material/rows.jsonl"); selected = final["models"][key]["selected"]
        isolated_path = P2385 / key / "raw/isolated_mean.float16.npy"; isolated = np.load(isolated_path, mmap_mode="r")
        output_path = P2382 / key / "raw/output_pre_sentence.float16.npy"; output = np.load(output_path, mmap_mode="r")
        qsource, qoutput = int(selected["qpoint"]), int(final["models"][key]["output_qpoint"])
        values = np.stack((np.asarray(isolated[:, :, 0], dtype=np.float16), np.asarray(isolated[:, :, qsource], dtype=np.float16),
                           np.asarray(output[:, :, qoutput], dtype=np.float16)), axis=2)
        close(isolated); close(output); metadata = []
        for row in rows:
            for slot in range(4):
                metadata.extend((
                    {"case_id": row["case_id"], "family": row["family"], "language": row["language"], "surface": row["surface"],
                     "partition": row["partition"], "slot": slot, "field": "isolated_sentence_embedding_mean", "qpoint": 0},
                    {"case_id": row["case_id"], "family": row["family"], "language": row["language"], "surface": row["surface"],
                     "partition": row["partition"], "slot": slot, "field": "isolated_sentence_selected_mean", "qpoint": qsource},
                    {"case_id": row["case_id"], "family": row["family"], "language": row["language"], "surface": row["surface"],
                     "partition": row["partition"], "slot": slot, "field": "output_pre_sentence", "qpoint": qoutput},
                ))
        assets.append(campaign.publish_array(f"c{number}_{key}_isolated_output_bridge", f"{labels[key]} isolated-content/output bridge",
            values, metadata, labels[key], "crossmodel_isolated_output_bridge_v1", "model-local functional checkpoint bridge",
            "768-row panel; embedding mean, selected isolated mean and selected output-pre field", "raw model-local activation in every physical coordinate",
            PHASE, CAMPAIGN, np.float16, {"lockbox_accuracy": selected["lockbox_accuracy"], "embedding_mean": selected["embedding_mean_lockbox"]}))
    return assets


def cleanup_raw() -> dict:
    candidates = [
        P2379 / "raw/qwen4b_prompt_boundary.float16.npy", P2379 / "raw/qwen4b_source_sentence_end.float16.npy",
        P2379 / "raw/qwen4b_output_progress_anchors.float16.npy", P2379 / "raw/qwen4b_reference_prompt_target_all_token.float16.npy",
        P2381 / "raw/qwen4b_pre_sentence_attention_mlp.float16.npy", P2381 / "raw/qwen4b_pre_sentence_mlp_intermediate.float16.npy",
        P2381 / "raw/qwen4b_pre_sentence_source_attention_mass.float16.npy",
        P2384 / "raw/qwen4b_isolated_sentence_end.float16.npy", P2384 / "raw/qwen4b_isolated_sentence_token_mean.float16.npy",
    ]
    for key in ("qwen14b", "glm4", "deepseek7b"):
        candidates.extend((P2382 / key / "raw/source_sentence_end.float16.npy", P2382 / key / "raw/output_pre_sentence.float16.npy",
                           P2385 / key / "raw/isolated_end.float16.npy", P2385 / key / "raw/isolated_mean.float16.npy"))
    deleted, reclaimed = [], 0
    for path in candidates:
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents: raise RuntimeError(("unsafe_cleanup", resolved))
        if path.exists():
            size = path.stat().st_size; path.unlink(); reclaimed += size; deleted.append({"path": str(path), "bytes": size})
    remaining = []
    for base in (P2379, P2381, P2382, P2384, P2385):
        for path in base.rglob("*.npy"):
            if "raw" in path.parts and any(term in path.name.lower() for term in ("hidden", "field", "source_sentence", "output_pre", "isolated", "attention_mlp", "mlp_intermediate", "attention_mass")):
                remaining.append(str(path))
    return {"deleted": deleted, "files_deleted": len(deleted), "bytes_reclaimed": reclaimed,
            "remaining_unpublished_hiddenstate_fields": remaining}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 无标签句内容机制图谱发布、清理与总审计（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2378–2386逐一核对final、MEMO连续性、材料锁箱、跨模型顺序和所有纠错边界。重要结果发布到可视化客户端`embedding_hiddenstate_full_coordinate`热力图族：16条prompt+输出全部token场；256条锁箱的输出前/早期/句末全层全坐标；Qwen4独立句end/mean全层场；Attention/MLP全部2560坐标；MLP全部9728中间神经元；逐坐标$R^2$/$\eta^2$；全head来源句注意力质量；三模型embedding/选定独立句/输出前桥接。逐资产校验shape、metadata、finite、SHA256并通过前端构建后，删除结果目录的重复/未完整发布HiddenState原场。

$$\mathrm{{publish}}\Rightarrow\mathrm{{verify}}(\mathrm{{shape}},\mathrm{{rows}},\mathrm{{finite}},\mathrm{{SHA256}}),\qquad
\mathrm{{cleanup}}\Leftarrow\mathrm{{verify}}\land\mathrm{{frontend\ build}}.$$

**结果汇总。** 总审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；发布 `{json.dumps(result['publication_summary'], ensure_ascii=False)}`；验证/前端 `{json.dumps(result['checks'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2387_c18481_c18800_publication_cleanup_audit.py`；最终审计位于 `tests/glm5/result/phase2387_c18481_c18800_publication_cleanup_audit`；客户端数据集`c18481`–`c18494`。未新增或修改其他Markdown。

**理论进展。** 当前最强新拼图不是显式指针/群齿轮，而是“条件句内容匹配”：独立句的词嵌入均值已携带强身份纹理，Qwen4/14中层进一步加入可复用组合纹理；输出句前状态按目标位置和方向与相应内容纹理匹配。Attention存在来源路由候选，MLP门控假说未通过。生成接口修复证明内容能力远高于旧估计，但逆序和共指仍阻止行为闭合。

**问题硬伤与结论。** 逐坐标仿射仍是观察预测器；没有选择性干预、错层/错head、剂量和救援，因此没有机制闭合。材料仍有高词汇重叠，GLM/DS最佳即embedding，DS也非独立架构。下一阶段总体目标相同，但即时目标已经从“发现无标签内容图谱”转为“对冻结Attention路线做因果必要性/充分性与同词汇改写不变性测试”，属于新的证据层级；本轮已自动连续完成位置混淆、独立句、跨模型和生成协议四个同目标续研Phase，形成可审计阶段边界。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return
    audit = stage_audit()
    assets = [publish_all_token(), publish_output_lockbox(),
              publish_isolated("c18483_qwen4b_isolated_sentence_end_full", "Qwen3-4B isolated-sentence end full field",
                               P2384 / "raw/qwen4b_isolated_sentence_end.float16.npy", "raw last-token activation for every isolated sentence and checkpoint"),
              publish_isolated("c18484_qwen4b_isolated_sentence_mean_full", "Qwen3-4B isolated-sentence token-mean full field",
                               P2384 / "raw/qwen4b_isolated_sentence_token_mean.float16.npy", "full-coordinate token mean; all physical coordinates retained"),
              *publish_components(), *publish_diagnostics(), publish_routing_subset(), *publish_crossmodel_bridges()]
    verification = [atlas.verify(asset) for asset in assets]
    verified = all(all(value for key, value in item.items() if key != "id") for item in verification)
    catalog = atlas.update_catalog(assets); frontend = atlas.frontend_build()
    if not (verified and frontend["passed"]): raise RuntimeError((verification, frontend))
    cleanup = cleanup_raw()
    checks = {"phase_audit_continuous": audit["continuous"], "asset_count": len(assets) == 14, "assets_verified": verified,
              "frontend_passed": frontend["passed"], "raw_hiddenstate_fields_clean": not cleanup["remaining_unpublished_hiddenstate_fields"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "audit": audit, "assets": assets, "verification": verification,
              "publication_summary": {"dataset_ids": [asset["id"] for asset in assets], "count": len(assets), "catalog": catalog,
                                      "frontend": frontend}, "cleanup": cleanup, "checks": checks,
              "all_checks_passed": all(checks.values()),
              "next_stage": {"same_overall_goal": True, "same_immediate_target": False,
                             "immediate_target": "causal necessity/sufficiency of frozen attention routing plus lexical-matched paraphrase invariance",
                             "reason": "observational natural-content atlas, cross-model replication and generation-protocol correction are complete"}}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__": main()
