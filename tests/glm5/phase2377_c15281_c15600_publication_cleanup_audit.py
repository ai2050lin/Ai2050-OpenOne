#!/usr/bin/env python3
"""Final evidence audit, exact-coordinate visualization publication and raw-field cleanup."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2377_c15281_c15600_publication_cleanup_audit"
PHASE = 2377
CAMPAIGN = "C15281-C15600"
P2368 = RESULT / "phase2368_c12481_c12720_longrange_operator_contract"
P2369 = RESULT / "phase2369_c12721_c13040_qwen_longrange_full_field"
P2370 = RESULT / "phase2370_c13041_c13360_pointer_group_operator"
P2371 = RESULT / "phase2371_c13361_c13680_advanced_math_tournament"
P2373 = RESULT / "phase2373_c14001_c14320_fresh_flagship_generation"
P2374 = RESULT / "phase2374_c14321_c14640_crossmodel_longrange_operator"
P2375 = RESULT / "phase2375_c14641_c14960_clean_taxonomy_lockbox"
P2376 = RESULT / "phase2376_c14961_c15280_s5_operator_autocontinuation"

sys.path.insert(0, str(TESTS))
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2359_c10321_c11520_qwen_hypergraph_field_campaign as campaign  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(16 << 20): h.update(block)
    return h.hexdigest()


def checkpoint(q: int, count: int) -> str:
    return "embedding" if q == 0 else "final_norm" if q == count - 1 else f"block_{q:02d}_post"


def phase_final(phase: int) -> Path:
    matches = sorted(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
    if len(matches) != 1: raise RuntimeError((phase, matches))
    return matches[0]


def stage_audit() -> dict:
    memo = MEMO.read_text(encoding="utf-8"); phases = {}
    for phase in range(2368, 2377):
        path = phase_final(phase); data = json.loads(path.read_text(encoding="utf-8"))
        phases[str(phase)] = {"final": str(path), "exists": path.exists(), "recorded_phase": data.get("phase"),
                              "memo_heading_count": memo.count(f"## Phase {phase}:")}
    return {"phases": phases, "continuous": all(v["exists"] and v["recorded_phase"] == int(k) and v["memo_heading_count"] == 1 for k, v in phases.items()),
            "correction_ledger": [
                "Phase2370 initial memo q0 R2=1 was a zero-variance embedding degeneracy; final and Phase2371 correct it to q31 R2=0.2130.",
                "Phase2371 initial CMI fixed sigma and was unidentifiable; final uses both source orders: median 0.12846 bit versus shuffled 0.01386.",
                "Phase2372 flagship R2=1 used repeated prompts and is retracted.",
                "Phase2373 fresh taxonomy retained 240 cross-unit duplicate prompts and is superseded by Phase2375; attitude fresh result remains usable.",
                "Phase2375 has 2048/2048 unique prompts and distinct first answer tokens; clean taxonomy lockbox R2=0.7497.",
            ],
            "retained_evidence": [
                "Qwen4 autonomous four-sentence index order exact=0.8841; long-copy marker order=1.0 but code preservation=0.1484 and verbatim exact=0.",
                "Source-slot decodability is robust in Qwen4, Qwen14B and GLM4; it is the strongest behavior-qualified cross-model structural result.",
                "Qwen4 diagonal adjacent-swap response predicts S4 lockbox R2=0.2130 and S5 lockbox R2=0.2521, beating direct templates and coordinate permutation.",
                "Behavior-qualified Qwen14B and GLM4 do not reproduce positive operator R2; a universal or model-independent operator claim is false.",
                "Clean taxonomy relation-chain response is language-conditioned, shallow (q1), order<=2 and lockbox-positive R2=0.7497.",
            ],
            "rejected_overclaims": [
                "No explicit internal permutation group, tensor block movement, pointer automaton, topology isomorphism or new closed mathematics has been demonstrated.",
                "Exact S4 Fourier decomposition and stable irrep spectra are properties of the experimental permutation index; Fourier templates fail predictive lockboxes.",
                "OT marker alignment is lexically confounded; H0 persistence is label invariant; HOSVD is reconstruction rather than target prediction.",
                "Activation coordinates are not trained parameters and cross-model coordinate numbers are not aligned.",
            ]}


def publish_s4_full() -> dict:
    rows = read_rows(P2368 / "material/long_sentence_permutation.jsonl"); source = P2369 / "raw/qwen4b_long_boundary_all_layers.float16.npy"
    values = np.load(source, mmap_mode="r"); meta = []
    for row in rows:
        for q in range(values.shape[1]):
            meta.append({"case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "language": row["language"],
                         "task": row["task"], "source_perm": row["source_perm"], "target_perm": row["target_perm"],
                         "qpoint": q, "checkpoint": checkpoint(q, values.shape[1])})
    return campaign.publish_array("c15281_qwen4b_s4_long_all_layer_full_field", "Qwen3-4B S4 long-sentence complete field", source, meta,
        "Qwen3-4B-BF16", "s4_long_all_layer_full_coordinate_v1", "observational complete field",
        "7104 frozen long prompts; embedding, 36 blocks and final norm", "raw activation at every model-local physical coordinate; activation, not trained parameter",
        PHASE, CAMPAIGN, np.float16, {"shape3d": list(values.shape), "important_result": True})


def publish_all_token() -> dict:
    source = np.load(P2369 / "raw/qwen4b_long_reference_all_token_all_layers.float16.npy", mmap_mode="r")
    index = read_rows(P2369 / "index/long_reference_all_token_rows.jsonl"); total = sum(r["token_count"] * source.shape[1] for r in index)
    binary = atlas.create_binary("c15282_qwen4b_long_all_token_full_field.float16.npy", total, source.shape[-1], np.float16)
    meta, cursor = [], 0
    for ri, row in enumerate(index):
        for q in range(source.shape[1]):
            n = row["token_count"]; binary[cursor:cursor + n] = source[ri, q, :n]
            for pos in range(n): meta.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                "source_perm": row["source_perm"], "target_perm": row["target_perm"], "qpoint": q,
                "checkpoint": checkpoint(q, source.shape[1]), "token_position": pos, "token_id": row["token_ids"][pos], "token": row["tokens"][pos]})
            cursor += n
    binary.flush(); del binary, source
    return atlas.write_metadata("c15282_qwen4b_long_all_token_full_field", "Qwen3-4B long-sentence all-token exact field",
        VIS / "c15282_qwen4b_long_all_token_full_field.float16.npy", meta, "Qwen3-4B-BF16", "long_all_token_full_coordinate_v1",
        "observational complete token field", "16 fresh-unit references; padded tokens removed",
        "raw word embedding or HiddenState activation at a physical token position",
        {"phase": PHASE, "campaign": CAMPAIGN, "no_topk": True, "activation_not_parameter": True, "important_result": True})


def publish_trajectory_and_diagnostics() -> list[dict]:
    trajectory = P2373 / "raw/qwen4b_index_generation_trajectory.float16.npy"; rows = read_rows(P2373 / "material/trajectory_rows.jsonl")
    values = np.load(trajectory, mmap_mode="r"); meta = []
    for row in rows:
        for step in range(values.shape[1]):
            for q in range(values.shape[2]): meta.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                "source_perm": row["source_perm"], "target_perm": row["target_perm"], "generation_step": step, "qpoint": q,
                "checkpoint": checkpoint(q, values.shape[2])})
    asset1 = campaign.publish_array("c15283_qwen4b_s4_generation_trajectory", "Qwen3-4B S4 autonomous generation trajectory", trajectory, meta,
        "Qwen3-4B-BF16", "s4_generation_trajectory_full_coordinate_v1", "observational dynamic field",
        "64 fresh-unit rows x 8 greedy steps x every checkpoint", "decision-boundary activation in every physical coordinate",
        PHASE, CAMPAIGN, np.float16, {"shape4d": list(values.shape)})
    operator = np.load(P2370 / "derived/operator_coordinate_lockbox_r2.float32.npy")
    operator = np.nan_to_num(operator, nan=0.0, posinf=0.0, neginf=0.0)
    methods = ["identity", "translation", "diagonal_affine", "direct_template", "coordinate_permuted_affine"]
    meta2 = [{"qpoint": q, "checkpoint": checkpoint(q, 38), "method": method,
              "note": "q0 zero-variance coordinates are encoded as 0 after explicit degeneracy audit"} for q in range(38) for method in methods]
    asset2 = campaign.publish_array("c15284_qwen4b_s4_operator_coordinate_r2", "Qwen3-4B S4 operator lockbox coordinate R2",
        operator, meta2, "Qwen3-4B-BF16", "operator_coordinate_lockbox_r2_v1", "held-out coordinate diagnostic",
        "fresh units4-5 and unseen source order", "per-physical-coordinate response R2; zero is not a neuron score",
        PHASE, CAMPAIGN, np.float32, {"selected_qpoint": 31, "selected_method": "diagonal_affine"})
    cmi = np.load(P2371 / "derived/q26_coordinate_cmi_slot_given_identity.float32.npy")
    meta3 = [{"condition": name, "qpoint": 26, "meaning": "I(coordinate bin; source slot | first sentence identity), bits"}
             for name in ("actual", "within_identity_label_shuffle")]
    asset3 = campaign.publish_array("c15285_qwen4b_source_slot_coordinate_cmi", "Qwen3-4B source-slot conditional information",
        cmi, meta3, "Qwen3-4B-BF16", "coordinate_cmi_v1", "descriptive association",
        "1536 fresh-unit rows across both source orders", "binned CMI for every physical q26 coordinate; finite-sample biased",
        PHASE, CAMPAIGN, np.float32)
    return [asset1, asset2, asset3]


def publish_s5_full() -> dict:
    rows = read_rows(P2376 / "material/s5_long_sentence_index.jsonl"); source = P2376 / "raw/qwen4b_s5_boundary_all_layers.float16.npy"
    values = np.load(source, mmap_mode="r"); meta = []
    for row in rows:
        for q in range(values.shape[1]): meta.append({"case_id": row["case_id"], "family": row["family"], "unit": row["unit"],
            "partition": row["partition"], "language": row["language"], "source_perm": row["source_perm"], "target_perm": row["target_perm"],
            "qpoint": q, "checkpoint": checkpoint(q, values.shape[1])})
    return campaign.publish_array("c15286_qwen4b_s5_long_all_layer_full_field", "Qwen3-4B S5 long-sentence complete field", source, meta,
        "Qwen3-4B-BF16", "s5_long_all_layer_full_coordinate_v1", "observational complete field",
        "5760 unique prompts; all 120 S5 permutations", "raw activation at every model-local physical coordinate; activation, not trained parameter",
        PHASE, CAMPAIGN, np.float16, {"shape3d": list(values.shape), "operator_lockbox_r2": 0.25208105925973945})


def publish_checkpoint_subset(dataset_id: str, title: str, source_path: Path, rows: list[dict], indices: list[int],
                              qpoints: list[int], model: str, schema: str, boundary: str, extra: dict | None = None) -> dict:
    source = np.load(source_path, mmap_mode="r"); values = np.stack([np.asarray(source[indices, q], dtype=np.float16) for q in qpoints], axis=1)
    meta = [{"case_id": rows[i]["case_id"], "qpoint": q, "checkpoint": checkpoint(q, source.shape[1]),
             **{k: rows[i].get(k) for k in ("system", "family", "unit", "language", "surface", "cell", "source_perm", "target_perm") if k in rows[i]}}
            for i in indices for q in qpoints]
    return campaign.publish_array(dataset_id, title, values, meta, model, schema, "observational selected checkpoints", boundary,
        "raw embedding or HiddenState activation in every model-local physical coordinate", PHASE, CAMPAIGN, np.float16, extra)


def publish_flagships_and_crossmodel() -> list[dict]:
    fresh_rows = read_rows(P2373 / "material/fresh_lexical_flagship.jsonl"); attitude = [i for i, r in enumerate(fresh_rows) if r["system"] == "attitude_role"]
    assets = [publish_checkpoint_subset("c15287_qwen4b_fresh_attitude_checkpoints", "Qwen3-4B fresh attitude-role checkpoints",
        P2373 / "raw/qwen4b_fresh_flagship_boundary.float16.npy", fresh_rows, attitude, [0, 1, 37], "Qwen3-4B-BF16",
        "fresh_attitude_selected_checkpoint_v1", "1920 attitude rows only; taxonomy rows excluded after duplicate audit", {"lockbox_r2": 0.3921977973154266})]
    clean_rows = read_rows(P2375 / "material/clean_taxonomy_factorial.jsonl")
    assets.append(publish_checkpoint_subset("c15288_qwen4b_clean_taxonomy_checkpoints", "Qwen3-4B leak-free taxonomy checkpoints",
        P2375 / "raw/qwen4b_clean_taxonomy_boundary.float16.npy", clean_rows, list(range(len(clean_rows))), [0, 1, 37],
        "Qwen3-4B-BF16", "clean_taxonomy_selected_checkpoint_v1", "2048 unique prompts; q1 selected on confirmation", {"lockbox_r2": 0.7496671784824338}))
    cross = json.loads((P2374 / "analysis/final.json").read_text(encoding="utf-8"))
    labels = {"qwen14b": "Qwen3-14B-NF4", "glm4": "GLM4-9B-INT8", "deepseek7b": "DeepSeek-R1-Distill-Qwen-7B-INT8"}
    for number, key in enumerate(("qwen14b", "glm4", "deepseek7b"), start=15289):
        base = P2374 / key; rows = read_rows(base / "material/rows.jsonl"); model = cross["models"][key]
        source = np.load(base / "raw/boundary.float16.npy", mmap_mode="r")
        qpoints = sorted({0, model["pointer"]["selected_qpoint"], model["operator"]["selected_qpoint"], source.shape[1] - 1})
        assets.append(publish_checkpoint_subset(f"c{number}_{key}_longrange_checkpoints", f"{labels[key]} long-range checkpoints",
            base / "raw/boundary.float16.npy", rows, list(range(len(rows))), qpoints, labels[key],
            "crossmodel_longrange_selected_checkpoint_v1", "same 768 S4 prompts; model-local coordinates only",
            {"pointer_lockbox_accuracy": model["pointer"]["lockbox_accuracy"], "operator_lockbox_r2": model["operator"]["lockbox_response_r2"],
             "quantization": model["collection"]["quantization"], "qpoints": qpoints}))
    return assets


def cleanup_raw() -> dict:
    candidates = [
        P2369 / "raw/qwen4b_long_boundary_all_layers.float16.npy", P2369 / "raw/qwen4b_long_reference_all_token_all_layers.float16.npy",
        P2369 / "raw/qwen4b_flagship_boundary_all_layers.float16.npy", P2373 / "raw/qwen4b_fresh_flagship_boundary.float16.npy",
        P2373 / "raw/qwen4b_index_generation_trajectory.float16.npy", P2375 / "raw/qwen4b_clean_taxonomy_boundary.float16.npy",
        P2376 / "raw/qwen4b_s5_boundary_all_layers.float16.npy",
        P2374 / "qwen14b/raw/boundary.float16.npy", P2374 / "glm4/raw/boundary.float16.npy", P2374 / "deepseek7b/raw/boundary.float16.npy",
    ]
    deleted, total = [], 0
    for path in candidates:
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents: raise RuntimeError(("unsafe_cleanup", resolved))
        if path.exists():
            size = path.stat().st_size; path.unlink(); total += size; deleted.append({"path": str(path), "bytes": size})
    remaining = []
    for base in (P2369, P2373, P2374, P2375, P2376):
        for path in base.rglob("*.npy"):
            if any(term in path.name.lower() for term in ("boundary", "trajectory", "all_token", "hiddenstate")) and "decisions" not in path.name.lower(): remaining.append(str(path))
    return {"deleted": deleted, "bytes_reclaimed": total, "remaining_unpublished_hiddenstate_fields": remaining}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 全阶段严格审计、参数级热力图发布、场清理与下一边界（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2368–2376逐一核对final、连续MEMO、三次材料纠错和最终取代关系；只把通过行为/预测/锁箱/坐标对照的结论保留。重要结果发布为客户端`embedding_hiddenstate_full_coordinate`：S4与S5全部样本×全部层×全部物理坐标、16条长句全部token场、64×8步生成轨迹、逐坐标算子R²/CMI、干净双旗舰检查点，以及三种跨模型各自embedding/关键HiddenState/final norm。逐文件验证shape、row metadata、finite和SHA256并通过前端构建后，才删除结果目录中的重复或未发布HiddenState原始场。

$$\mathrm{{SHA256}}(B_{{client}})=h_{{metadata}},\qquad \mathrm{{cleanup}}\Leftarrow \mathrm{{verify}}\land\mathrm{{frontend\ build}}.$$

**结果汇总。** 总审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；发布数据 `{json.dumps(result['publication']['datasets'], ensure_ascii=False)}`；验证/前端 `{json.dumps(result['checks'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2377_c15281_c15600_publication_cleanup_audit.py`；最终结果 `tests/glm5/result/phase2377_c15281_c15600_publication_cleanup_audit`；客户端数据`c15281`–`c15291`。

**理论进展、问题硬伤与结论。** 当前最可靠的普遍拼图不是显式群齿轮，而是行为合格模型中可读的来源槽位绑定；Qwen4B额外存在跨$S_4/S_5$的晚层逐坐标仿射响应规律，但Qwen14B/GLM4反证其跨模型普遍性。知识链存在强浅层语言条件复用，仍可能包含词汇/查询捷径。长句生成明确区分“顺序计划”与“内容无损”：前者强，后者弱。附件二的群论、OT、张量、拓扑只留下有边界的诊断结果，没有任何一个构成编码机制闭合。

**下一阶段第一性原理。** 总目标仍是破解有限参数如何实现语言，但即时问题已从“显式标记下的排列响应律”变化为“无显式标记、无重复词、长度不等时，模型如何维持对象身份—来源槽位—输出内容绑定”。下一大阶段应以自然段落的隐式语义排序、同首词干扰、跨句共指和完整内容保持为主，先预测teacher-forced输出锚点的句对象/槽位/句内偏移，再做选择性干预。由于即时目标已经改变，本轮在完成同目标$S_5$自动续研后形成合理阶段边界，而不是无限重复同一排列材料。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    audit = stage_audit()
    assets = [publish_s4_full(), publish_all_token(), *publish_trajectory_and_diagnostics(), publish_s5_full(), *publish_flagships_and_crossmodel()]
    verification = [atlas.verify(asset) for asset in assets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    catalog = atlas.update_catalog(assets); frontend = atlas.frontend_build()
    if not (verified and frontend["passed"]): raise RuntimeError((verification, frontend))
    cleanup = cleanup_raw()
    checks = {"phase_audit_continuous": audit["continuous"], "assets_verified": verified, "frontend_passed": frontend["passed"],
              "raw_hiddenstate_fields_clean": not cleanup["remaining_unpublished_hiddenstate_fields"], "asset_count": len(assets) == 11}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "audit": audit,
              "publication": {"datasets": json.loads(json.dumps(assets, default=str)), "verification": verification, "catalog": catalog, "frontend": frontend},
              "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values()),
              "next_stage": {"same_overall_goal": True, "same_immediate_target": False,
                "immediate_target": "label-free object identity, source-slot and content-preserving output binding under unequal natural sentences",
                "reason": "S4-to-S5 same-target continuation is complete; cross-model evidence rejects a universal explicit permutation operator."}}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
