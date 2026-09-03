#!/usr/bin/env python3
"""Publish the NTP predictive-field campaign and clean undisplayed raw fields."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2296 = RESULT / "phase2296_c3101_c3160_ntp_predictive_contract"
P2297 = RESULT / "phase2297_c3161_c3260_qwen4b_ntp_predictive_field"
P2298 = RESULT / "phase2298_c3261_c3340_full_vocabulary_accounting"
P2299 = RESULT / "phase2299_c3341_c3440_predictive_timing_coordinate_structure"
P2300 = RESULT / "phase2300_c3441_c3500_fisher_audit_q14_contract"
P2301 = RESULT / "phase2301_c3501_c3600_qwen14_ntp_timing_replication"
OUT = RESULT / "phase2302_c3601_c3700_ntp_atlas_publication"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
Q4_MODEL = ROOT / "models/hf/Qwen3-4B"

PHASE = 2302
CAMPAIGN = "C3601-C3700"
REPRESENTATIVE_UNIT = 26
FISHER_MASS_GATE = 1e-8

Q4_ROWS = P2296 / "material/ntp_natural_bilingual.jsonl"
Q4_BOUNDARY = P2297 / "raw/qwen4b_ntp_boundary_all_checkpoints.float16.npy"
Q4_ALL_TOKEN = P2297 / "raw/qwen4b_ntp_representative_all_token.float16.npy"
Q4_ALL_TOKEN_ROWS = P2297 / "index/representative_all_token_rows.jsonl"
Q4_CONTRIBUTIONS = P2297 / "atlas/qwen4b_target_wrong_coordinate_contributions.float16.npy"
Q4_FISHER = P2297 / "atlas/qwen4b_output_fisher_diagonal.float32.npy"
Q4_FISHER_ROWS = P2297 / "index/fisher_rows.jsonl"
Q4_LOGITS = P2297 / "raw/qwen4b_ntp_full_vocabulary_logits.float16.npy"

Q14_ROWS = P2301 / "material/qwen14_ntp_fresh_rows.jsonl"
Q14_FIELD = P2301 / "raw/qwen14_ntp_selected_checkpoints.float16.npy"
Q14_LOGITS = P2301 / "raw/qwen14_ntp_full_vocabulary_logits.float16.npy"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def checkpoint_label(q: int, layers: int) -> str:
    if q == 0:
        return "embedding"
    if q == layers + 1:
        return "final_norm"
    return f"block_{q:02d}_post"


def case_metadata(row: dict) -> dict:
    return {
        "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"],
        "partition": row["partition"], "state": int(row["state"]),
        "unit": int(row["unit"]), "target_text": row["ntp_target_text"],
        "wrong_text": row["ntp_wrong_text"],
    }


def metadata(
    dataset_id: str,
    title: str,
    binary: Path,
    rows: list[dict],
    schema: str,
    model: str,
    claim_level: str,
    boundary: str,
    coordinate_semantics: str,
    heatmap_type: str,
    extra: dict | None = None,
) -> dict:
    values = np.load(binary, mmap_mode="r")
    try:
        shape = list(values.shape)
        finite = bool(np.isfinite(values).all())
    finally:
        close_memmap(values)
    if len(shape) != 2 or len(rows) != shape[0] or not finite:
        raise RuntimeError(("invalid_publication_asset", dataset_id, shape, len(rows), finite))
    info = {
        "schema": schema,
        "generated_at": datetime.now().astimezone().isoformat(),
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "id": dataset_id,
        "title": title,
        "model": model,
        "binary_url": "/vis_data/research_kernel/" + binary.name,
        "binary_shape": shape,
        "binary_sha256": file_hash(binary),
        "coordinate_count": shape[-1],
        "coordinate_semantics": coordinate_semantics,
        "coordinate_order": "original model-local order; no Top-K, PCA, averaging, or coordinate reordering",
        "heatmap_type": heatmap_type,
        "claim_level": claim_level,
        "boundary": boundary,
        "rows": rows,
    }
    if extra:
        info.update(extra)
    meta_path = VIS / f"{dataset_id}.json"
    save_json(meta_path, info)
    return {
        "id": dataset_id, "title": title, "metadata": meta_path, "binary": binary,
        "shape": shape, "sha256": info["binary_sha256"], "model": model,
        "schema": schema, "claim_level": claim_level, "boundary": boundary,
        "heatmap_type": heatmap_type,
    }


def create_binary(name: str, dtype: np.dtype, shape: tuple[int, int]) -> np.memmap:
    VIS.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(VIS / name, mode="w+", dtype=dtype, shape=shape)


def publish_q4_boundary(rows: list[dict]) -> dict:
    selected = [(i, row) for i, row in enumerate(rows)
                if row["partition"] == "fresh_lockbox" and int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(Q4_BOUNDARY, mmap_mode="r")
    binary = VIS / "c3601_qwen4b_ntp_boundary_trajectory.float16.npy"
    output = create_binary(binary.name, np.float16, (len(selected) * source.shape[1], source.shape[-1]))
    meta_rows: list[dict] = []
    cursor = 0
    try:
        for source_i, row in selected:
            for q in range(source.shape[1]):
                output[cursor] = source[source_i, q]
                meta_rows.append({"row": cursor, **case_metadata(row), "checkpoint": q,
                                  "checkpoint_label": checkpoint_label(q, 36),
                                  "metric": "boundary_activation"})
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c3601_qwen4b_ntp_boundary_trajectory", "Qwen3-4B NTP Boundary Trajectory",
        binary, meta_rows, "ai2050.ntp-boundary-trajectory.v1", "Qwen3-4B",
        "full_coordinate_observation",
        "Fresh-lockbox unit26 for all six families, two languages, two surfaces and two states; embedding, all 36 post-block checkpoints, final norm, and all 2560 coordinates.",
        "Qwen3-4B runtime activation coordinates at the natural lexical answer boundary",
        "embedding_hiddenstate_full_coordinate",
    )


def publish_q4_all_token() -> dict:
    index = read_jsonl(Q4_ALL_TOKEN_ROWS)
    selected = [row for row in index if row["family"] == "attitude_event"]
    source = np.load(Q4_ALL_TOKEN, mmap_mode="r")
    binary = VIS / "c3602_qwen4b_ntp_attitude_all_token.float16.npy"
    output = create_binary(binary.name, np.float16, (len(selected), source.shape[-1]))
    tokenizer = AutoTokenizer.from_pretrained(str(Q4_MODEL), local_files_only=True,
                                               trust_remote_code=True, use_fast=False)
    decoded: dict[int, str] = {}
    meta_rows = []
    try:
        for cursor, row in enumerate(selected):
            output[cursor] = source[int(row["row"])]
            token_id = int(row["token_id"])
            if token_id not in decoded:
                decoded[token_id] = tokenizer.decode([token_id])
            meta_rows.append({
                "row": cursor, "case_id": row["case_id"], "family": row["family"],
                "language": row["language"], "surface": row["surface"],
                "state": int(row["state"]), "checkpoint": int(row["checkpoint"]),
                "checkpoint_label": checkpoint_label(int(row["checkpoint"]), 36),
                "token_position": int(row["token"]), "token_id": token_id,
                "token_text": decoded[token_id], "metric": "token_activation",
            })
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c3602_qwen4b_ntp_attitude_all_token", "Qwen3-4B Attitude NTP All-Token Field",
        binary, meta_rows, "ai2050.ntp-all-token-field.v1", "Qwen3-4B",
        "full_coordinate_observation",
        "All eight fresh-lockbox unit26 attitude-event cases; every real prompt token, every checkpoint, and every physical activation coordinate.",
        "Qwen3-4B runtime activation coordinates for each actual prompt token",
        "embedding_hiddenstate_full_coordinate",
    )


def publish_q4_contributions(rows: list[dict]) -> dict:
    binary = VIS / "c3603_qwen4b_ntp_output_coordinate_contributions.float16.npy"
    shutil.copy2(Q4_CONTRIBUTIONS, binary)
    meta_rows = [{"row": i, **case_metadata(row), "checkpoint": 37,
                  "checkpoint_label": "final_norm", "metric": "target_minus_wrong_logit_contribution"}
                 for i, row in enumerate(rows)]
    parent = load_json(P2297 / "analysis/final.json")
    structure = load_json(P2299 / "analysis/final.json")
    return metadata(
        "c3603_qwen4b_ntp_output_coordinate_contributions",
        "Qwen3-4B Exact Output Coordinate Contributions", binary, meta_rows,
        "ai2050.ntp-output-coordinate-contribution.v1", "Qwen3-4B",
        "exact_final_linear_decomposition",
        "Every material row and all 2560 final-norm coordinates. Values sum to the first-token target-minus-wrong logit margin up to recorded float16 error; they do not decompose earlier causal computation.",
        "signed per-coordinate contribution h_j*(W_target,j-W_wrong,j)",
        "embedding_hiddenstate_full_coordinate",
        {"decomposition_max_abs_error_float16": parent["contributions"]["decomposition_max_abs_error_float16"],
         "coordinate_structure_summary": structure["coordinate_structure"]},
    )


def publish_q4_fisher() -> dict:
    binary = VIS / "c3604_qwen4b_ntp_output_fisher_diagonal.float32.npy"
    shutil.copy2(Q4_FISHER, binary)
    source = np.load(Q4_FISHER, mmap_mode="r")
    index = read_jsonl(Q4_FISHER_ROWS)
    masses = np.sum(source, axis=1, dtype=np.float64)
    meta_rows = [{**row, "fisher_mass": float(masses[i]),
                  "shape_eligible": bool(masses[i] > FISHER_MASS_GATE),
                  "metric": "categorical_output_fisher_diagonal"}
                 for i, row in enumerate(index)]
    close_memmap(source)
    audit = load_json(P2300 / "analysis/final.json")["fisher_audit"]
    return metadata(
        "c3604_qwen4b_ntp_output_fisher_diagonal", "Qwen3-4B Output Fisher Diagonal",
        binary, meta_rows, "ai2050.ntp-output-fisher-diagonal.v1", "Qwen3-4B",
        "exact_local_output_sensitivity_with_degeneracy_flags",
        "Exact categorical Fisher diagonal for 48 representative final states. Saturated rows remain visible and are explicitly marked ineligible for shape summaries.",
        "model-local final-state coordinate sensitivity to the complete vocabulary distribution",
        "embedding_hiddenstate_full_coordinate",
        {"mass_gate": FISHER_MASS_GATE, "audit": audit},
    )


def publish_logits(source_path: Path, rows: list[dict], dataset_id: str, title: str,
                   model: str, row_selector, extra: dict | None = None) -> dict:
    selected = [(i, row) for i, row in enumerate(rows) if row_selector(row)]
    source = np.load(source_path, mmap_mode="r")
    binary = VIS / f"{dataset_id}.float16.npy"
    output = create_binary(binary.name, np.float16, (len(selected), source.shape[-1]))
    meta_rows = []
    try:
        for cursor, (source_i, row) in enumerate(selected):
            output[cursor] = source[source_i]
            meta_rows.append({"row": cursor, **case_metadata(row),
                              "metric": "next_token_logit"})
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        dataset_id, title, binary, meta_rows, "ai2050.ntp-full-vocabulary-logits.v1", model,
        "complete_next_token_output_observation",
        "Representative fresh-lockbox unit26 rows retain every tokenizer-vocabulary logit. Vocabulary IDs are model-output coordinates, not HiddenState coordinates or probabilities.",
        "tokenizer vocabulary logit indexed by vocabulary token ID",
        "full_vocabulary_predictive_distribution", extra,
    )


def publish_q14_field(rows: list[dict]) -> dict:
    parent = load_json(P2301 / "analysis/final.json")
    qpoints = [int(value) for value in parent["field"]["qpoints"]]
    source = np.load(Q14_FIELD, mmap_mode="r")
    binary = VIS / "c3606_qwen14_ntp_frozen_checkpoints.float16.npy"
    output = create_binary(binary.name, np.float16, (len(rows) * len(qpoints), source.shape[-1]))
    meta_rows = []
    cursor = 0
    try:
        for source_i, row in enumerate(rows):
            for q_i, q in enumerate(qpoints):
                output[cursor] = source[source_i, q_i]
                meta_rows.append({"row": cursor, **case_metadata(row), "checkpoint": q,
                                  "checkpoint_label": checkpoint_label(q, 40),
                                  "metric": "boundary_activation",
                                  "family_frozen_checkpoint": int(parent["replication"]["families"][row["family"]]["checkpoint"]),
                                  "family_checkpoint_passed": bool(parent["replication"]["families"][row["family"]]["passed"])})
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c3606_qwen14_ntp_frozen_checkpoints", "Qwen3-14B Frozen NTP Checkpoints",
        binary, meta_rows, "ai2050.qwen14-ntp-frozen-checkpoints.v1", "Qwen3-14B",
        "prospective_cross_scale_timing_test",
        "All 288 fresh rows at q28, q33, and final norm q41, retaining all 5120 model-local coordinates and both passed and failed family outcomes. Qwen3-4B coordinate IDs are not aligned.",
        "Qwen3-14B runtime activation coordinates at frozen model-relative checkpoints",
        "embedding_hiddenstate_full_coordinate",
        {"frozen_replication": parent["replication"]},
    )


def verify(dataset: dict) -> dict:
    info = load_json(dataset["metadata"])
    values = np.load(dataset["binary"], mmap_mode="r")
    try:
        checks = {
            "shape": list(values.shape) == info["binary_shape"] == dataset["shape"],
            "rows": len(info["rows"]) == values.shape[0],
            "coordinate_count": values.shape[-1] == info["coordinate_count"],
            "finite": bool(np.isfinite(values).all()),
        }
    finally:
        close_memmap(values)
    checks["sha256"] = file_hash(dataset["binary"]) == info["binary_sha256"]
    return {"id": dataset["id"], **checks}


def serializable(dataset: dict) -> dict:
    return {**dataset,
            "metadata": str(dataset["metadata"].relative_to(ROOT)),
            "binary": str(dataset["binary"].relative_to(ROOT))}


def update_catalog(datasets: list[dict]) -> dict:
    catalog = load_json(CATALOG)
    ids = {row["id"] for row in datasets}
    entries = [{
        "id": row["id"], "title": row["title"], "phase": PHASE,
        "campaign": CAMPAIGN, "model": row["model"],
        "source_path": "/vis_data/research_kernel/" + row["metadata"].name,
        "binary_path": "/vis_data/research_kernel/" + row["binary"].name,
        "source_schema": row["schema"], "coordinate_count": row["shape"][-1],
        "row_count": row["shape"][0], "claim_level": row["claim_level"],
        "boundary": row["boundary"], "kinds": [row["heatmap_type"]],
    } for row in datasets]
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", [])
                                      if row.get("id") not in ids]
    fields = [{
        "id": row["id"], "title": row["title"],
        "url": "/vis_data/research_kernel/" + row["metadata"].name,
        "phase": PHASE, "full_coordinate": True,
        "heatmap_type": row["heatmap_type"],
    } for row in datasets]
    catalog["field_datasets"] = fields + [row for row in catalog.get("field_datasets", [])
                                           if row.get("id") not in ids]
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {"added": sorted(ids), "dataset_count": len(catalog["datasets"]),
            "field_dataset_count": len(catalog["field_datasets"])}


def frontend_build() -> dict:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        command = [npm, "run", "build"]
    else:
        candidates = sorted(Path.home().glob(
            "AppData/Local/OpenAI/Codex/runtimes/cua_node/*/bin/node.exe"), reverse=True)
        if not candidates:
            raise FileNotFoundError("No Node runtime is available")
        command = [str(candidates[0]), str(ROOT / "frontend/node_modules/vite/bin/vite.js"), "build"]
    completed = subprocess.run(command, cwd=ROOT / "frontend", capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=900)
    return {"command": command, "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:],
            "passed": completed.returncode == 0}


def cleanup(paths: list[Path]) -> dict:
    result_root = RESULT.resolve()
    rows = []
    total = 0
    for path in paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(("cleanup_outside_result", str(path)))
        if not path.exists():
            rows.append({"path": str(path.relative_to(ROOT)), "status": "already_absent",
                         "bytes_deleted": 0})
            continue
        size = path.stat().st_size
        sha = file_hash(path)
        path.unlink()
        total += size
        rows.append({"path": str(path.relative_to(ROOT)),
                     "status": "deleted_after_verified_visual_derivative",
                     "sha256_before": sha, "bytes_deleted": size})
    ledger = {"files": rows, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    p2298 = load_json(P2298 / "analysis/final.json")
    p2299 = load_json(P2299 / "analysis/final.json")
    p2300 = load_json(P2300 / "analysis/final.json")
    p2301 = load_json(P2301 / "analysis/final.json")
    assets = [{"id": row["id"], "shape": row["shape"], "sha256": row["sha256"],
               "claim_level": row["claim_level"]} for row in result["datasets"]]
    text = rf"""

## Phase {PHASE}: 下一词预测场全坐标图谱发布与总审计（{CAMPAIGN}） [{stamp}]

**证据审查与修正。** 附件关于训练目标的核心提醒成立：自回归模型用每个前缀预测下一token，因此本项目不能把单个HiddenState先验命名为静态“词义”，而应同时登记前缀、当前状态、后续词表竞争和完整答案序列。但“由下一词训练”不推出任一检查点HiddenState是未来的充分统计量、贝叶斯信念态、世界模型、Koopman不变量、拓扑洞、测地流形或因果齿轮。Phase2288-2295的窄阳性仍只在Qwen模型、受控提示和冻结角色边界内有效；本期据此把行为、完整词表输出、层间形成时序、精确输出贡献和局部敏感度分开记账。

**测试原理与用例。** 大阶段使用六个自然词汇答案构式：施事—受事、态度—事件、比较次序、位置绑定、领属查询和关系从句绑定；每族含中英、narrative/dialogue、两种状态和四个预冻结分区，共1536行。Qwen3-4B保存回答边界的embedding、36个block后状态、final norm、全部2560坐标及完整151936词表logits；fresh-lockbox unit26另保存全部真实token轨迹。形成层只从discovery+confirmation选择，两个fresh分区只检验。Qwen3-14B只运行预选三族、两个fresh分区和相对深度冻结点，不搬运4B坐标编号。发布面板同时保留阳性和阴性；Fisher饱和行显式标为退化。

**公式。** 完整词汇答案使用逐token条件概率：

$$
S(y\mid x)=\sum_{{r=1}}^{{|y|}}\log p(y_r\mid x,y_{{<r}}),
\qquad
\bar S(y\mid x)=S(y\mid x)/|y|.
$$

完整下一token分布的状态/表面对照为：

$$
A_f=\mathbb E D_{{JS}}(P_{{s=0}},P_{{s=1}})
-\mathbb E D_{{JS}}(P_{{narr}},P_{{dialogue}}).
$$

final norm到第一token目标竞争的逐坐标精确分账为：

$$
z_{{y^+}}-z_{{y^-}}
=\sum_{{j=1}}^d h_j\left(W_{{y^+,j}}-W_{{y^-,j}}\right).
$$

Fisher对角仅在总质量合格时报告形状：

$$
G_{{jj}}=\operatorname{{Var}}_{{v\sim p}}[W_{{v,j}}],
\qquad \sum_jG_{{jj}}>10^{{-8}}.
$$

4B到14B只映射相对层深，并要求两个fresh分区分别过门：

$$
q_{{14}}=\operatorname{{round}}(40q_4/36),
\qquad \operatorname{{Acc}}_{{sign}}(q_{{14}},p)\ge0.75.
$$

**结果汇总。** Qwen3-4B完整词汇序列行为为6/6族合格；第一token总体准确率为 `{p2298['overall']['first_token_accuracy']}`，状态翻转JS大于匹配表面改写JS为 `{p2298['overall']['families_state_above_surface']}/6` 族。六族4B冻结形成点均在fresh数据达到门槛，形成点与逐坐标结构为 `{json.dumps(p2299['timing'], ensure_ascii=False)}`、`{json.dumps(p2299['coordinate_structure'], ensure_ascii=False)}`。Fisher原始对角全部有限，但 `{p2300['fisher_audit']['eligible_rows']}/48` 行总质量合格、`{p2300['fisher_audit']['degenerate_rows']}/48` 行饱和退化。Qwen3-14B完整词汇序列行为3/3族合格；冻结形成时点只通过 `{len(p2301['replication']['passed_families'])}/3` 族，即 `{p2301['replication']['passed_families']}`，另两族失败并保留。发布资产 `{json.dumps(assets, ensure_ascii=False)}`；逐项验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端目录 `{json.dumps(result['catalog'], ensure_ascii=False)}`；前端构建通过 `{result['frontend_build']['passed']}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**理论进展、问题硬伤与结论。** 当前最可靠拼图是：自然词汇回答接口上，事实状态对完整词表竞争的影响明显大于匹配表面改写；目标竞争在4B中具有族特异的晚层形成时序；最终竞争由约八百至近一千个有效参与坐标的宽有符号分账组成；14B只复现施事—受事的冻结相对时点，否定了“同一家族三个构式共享一个简单相对时钟”的强结论。这些是模型本地的预测输出规律，不是静态语义向量、共同坐标字典或因果电路。硬伤包括：自然度没有独立人类盲评；身份答案词只保证身份分区而非全部对象词完全隔离；任务仍是问答接口；4B与14B训练相关；logit lens是外加统一尺；Fisher在高置信输出处大量退化；全token面板是代表样本而非全1536行。基础代数和概率账已足够表达本期结果，没有证据要求或确认新数学。

**下一阶段判定。** 本Campaign的具体目标“把前缀条件、完整下一token竞争、形成时序和全坐标输出分账连成可复核图谱，并做14B冻结复验”已经完成。继续研究自由自然续写、更多语言族、未提示的长期未来或因果干预会改变材料与研究对象，不属于同一冻结目标，必须另立前瞻合同，不能沿本锁箱自动续跑。脚本 `tests/glm5/phase2302_c3601_c3700_ntp_atlas_publication.py`；结果 `tests/glm5/result/phase2302_c3601_c3700_ntp_atlas_publication`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def published_dataset(dataset_id: str) -> dict:
    meta_path = VIS / f"{dataset_id}.json"
    info = load_json(meta_path)
    binary = VIS / Path(info["binary_url"]).name
    return {
        "id": dataset_id, "title": info["title"], "metadata": meta_path,
        "binary": binary, "shape": info["binary_shape"],
        "sha256": info["binary_sha256"], "model": info["model"],
        "schema": info["schema"], "claim_level": info["claim_level"],
        "boundary": info["boundary"], "heatmap_type": info["heatmap_type"],
    }


def finish_result(datasets: list[dict], verification: list[dict], catalog: dict,
                  build: dict, cleanup_result: dict) -> dict:
    parents = [load_json(path / "analysis/final.json") for path in
               (P2296, P2297, P2298, P2299, P2300, P2301)]
    checks = {
        "all_parent_phases_passed": all(row["all_checks_passed"] for row in parents),
        "all_assets_verified": all(all(value for key, value in row.items() if key != "id")
                                     for row in verification),
        "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
        "frontend_built": bool(build["passed"]),
        "raw_hiddenstate_fields_cleaned": all(not path.exists()
                                               for path in (Q4_BOUNDARY, Q4_ALL_TOKEN, Q14_FIELD)),
        "q4_coordinates_exact": all(row["shape"][-1] in (2560, 151936)
                                    for row in datasets if row["model"] == "Qwen3-4B"),
        "q14_coordinates_exact": all(row["shape"][-1] in (5120, 151936)
                                     for row in datasets if row["model"] == "Qwen3-14B"),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "evidence_audit": {
            "retained": [
                "next-token training makes prefix-conditioned future competition a necessary analysis axis",
                "Phase2288-2295 model-local sample-state and cross-language observations remain valid within scope",
                "behavior, hidden-state timing, complete output distribution, and local sensitivity require separate ledgers",
            ],
            "corrected": [
                "HiddenState is not established as a sufficient statistic for all futures",
                "no Bayesian belief-state, world-model, Koopman, topology, geodesic, wave, or causal-gear conclusion is licensed",
                "Qwen3-14B replicated only one of three frozen model-relative formation checkpoints",
            ],
        },
        "datasets": [serializable(row) for row in datasets],
        "verification": verification, "catalog": catalog,
        "frontend_build": build, "cleanup": cleanup_result, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "The NTP campaign links natural lexical behavior, complete next-token competition, exact-coordinate "
            "boundary trajectories, final output contributions, and a preregistered Qwen3-14B timing test. "
            "Only agent-patient timing replicated across scale; this is predictive-field evidence, not a sufficient-state, "
            "shared-neuron, advanced-mathematical, or causal closure claim."
        ),
        "next_stage_same_specific_target": False,
        "next_stage_reason": "Free continuation, new language families, or causal tests require new materials and preregistration.",
    }
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    return result


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = load_json(final_path)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    dataset_ids = [
        "c3601_qwen4b_ntp_boundary_trajectory",
        "c3602_qwen4b_ntp_attitude_all_token",
        "c3603_qwen4b_ntp_output_coordinate_contributions",
        "c3604_qwen4b_ntp_output_fisher_diagonal",
        "c3605_qwen4b_ntp_representative_vocabulary_logits",
        "c3606_qwen14_ntp_frozen_checkpoints",
        "c3607_qwen14_ntp_representative_vocabulary_logits",
    ]
    cleanup_path = OUT / "cleanup/ledger.json"
    if cleanup_path.exists() and all((VIS / f"{dataset_id}.json").exists() for dataset_id in dataset_ids):
        datasets = [published_dataset(dataset_id) for dataset_id in dataset_ids]
        verification = [verify(row) for row in datasets]
        catalog = update_catalog(datasets)
        build = frontend_build()
        if not build["passed"]:
            raise RuntimeError(("frontend_build_failed_during_recovery", build))
        result = finish_result(datasets, verification, catalog, build, load_json(cleanup_path))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parents = [load_json(path / "analysis/final.json") for path in
               (P2296, P2297, P2298, P2299, P2300, P2301)]
    if not all(row["all_checks_passed"] for row in parents):
        raise RuntimeError("A parent phase is not authorized")
    q4_rows = read_jsonl(Q4_ROWS)
    q14_rows = read_jsonl(Q14_ROWS)
    datasets = [
        publish_q4_boundary(q4_rows),
        publish_q4_all_token(),
        publish_q4_contributions(q4_rows),
        publish_q4_fisher(),
        publish_logits(
            Q4_LOGITS, q4_rows, "c3605_qwen4b_ntp_representative_vocabulary_logits",
            "Qwen3-4B Representative Complete Vocabulary Logits", "Qwen3-4B",
            lambda row: row["partition"] == "fresh_lockbox" and int(row["unit"]) == REPRESENTATIVE_UNIT,
            {"vocabulary_size": 151936},
        ),
        publish_q14_field(q14_rows),
        publish_logits(
            Q14_LOGITS, q14_rows, "c3607_qwen14_ntp_representative_vocabulary_logits",
            "Qwen3-14B Representative Complete Vocabulary Logits", "Qwen3-14B",
            lambda row: row["partition"] == "fresh_lockbox" and int(row["unit"]) == REPRESENTATIVE_UNIT,
            {"vocabulary_size": 151936},
        ),
    ]
    verification = [verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id") for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))
    catalog = update_catalog(datasets)
    build = frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))
    cleanup_result = cleanup([Q4_BOUNDARY, Q4_ALL_TOKEN, Q14_FIELD])
    result = finish_result(datasets, verification, catalog, build, cleanup_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
