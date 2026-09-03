#!/usr/bin/env python3
"""Publish the declarative-continuation campaign and clean superseded raw fields."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2303 = RESULT / "phase2303_c3701_c3780_declarative_continuation_contract"
P2304 = RESULT / "phase2304_c3781_c3900_qwen4b_declarative_field"
P2305 = RESULT / "phase2305_c3901_c4020_interface_state_accounting"
P2306 = RESULT / "phase2306_c4021_c4160_corrected_surface_replication"
P2307 = RESULT / "phase2307_c4161_c4240_qwen14_corrected_replication"
OUT = RESULT / "phase2308_c4241_c4320_declarative_atlas_publication"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
Q4_MODEL = ROOT / "models/hf/Qwen3-4B"

PHASE = 2308
CAMPAIGN = "C4241-C4320"
REPRESENTATIVE_UNIT = 26

Q4_ROWS = P2306 / "material/corrected_declarative_continuation_bilingual.jsonl"
Q4_BOUNDARY = P2306 / "raw/qwen4b_corrected_boundary_all_checkpoints.float16.npy"
Q4_LOGITS = P2306 / "raw/qwen4b_corrected_full_vocabulary_logits.float16.npy"
Q4_CONTRIBUTIONS = P2306 / "atlas/qwen4b_corrected_target_wrong_contributions.float16.npy"

Q4_ALL_TOKEN = P2304 / "raw/qwen4b_declarative_six_family_all_token.float16.npy"
Q4_ALL_TOKEN_ROWS = P2304 / "index/six_family_all_token_rows.jsonl"
Q4_UNCORRECTED_BOUNDARY = P2304 / "raw/qwen4b_declarative_boundary_all_checkpoints.float16.npy"
Q4_UNCORRECTED_LOGITS = P2304 / "raw/qwen4b_declarative_full_vocabulary_logits.float16.npy"

INTERFACE_DELTA = P2305 / "atlas/qwen4b_qa_to_declarative_unit26_signed_delta.float16.npy"
INTERFACE_DELTA_ROWS = P2305 / "index/interface_delta_rows.jsonl"

Q14_ROWS = P2307 / "material/qwen14_corrected_fresh_rows.jsonl"
Q14_FIELD = P2307 / "raw/qwen14_corrected_selected_checkpoints.float16.npy"
Q14_LOGITS = P2307 / "raw/qwen14_corrected_full_vocabulary_logits.float16.npy"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


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
        "case_id": row["case_id"],
        "family": row["family"],
        "language": row["language"],
        "surface": row["surface"],
        "partition": row["partition"],
        "state": int(row["state"]),
        "unit": int(row["unit"]),
        "target_text": row["ntp_target_text"],
        "wrong_text": row["ntp_wrong_text"],
        "target_mention_order": row["target_mention_order"],
    }


def create_binary(name: str, dtype: np.dtype, shape: tuple[int, int]) -> np.memmap:
    VIS.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(VIS / name, mode="w+", dtype=dtype, shape=shape)


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
        "coordinate_order": (
            "original model-local order; no Top-K, PCA, averaging, compression, "
            "or coordinate reordering"
        ),
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
        "id": dataset_id,
        "title": title,
        "metadata": meta_path,
        "binary": binary,
        "shape": shape,
        "sha256": info["binary_sha256"],
        "model": model,
        "schema": schema,
        "claim_level": claim_level,
        "boundary": boundary,
        "heatmap_type": heatmap_type,
    }


def publish_q4_boundary(rows: list[dict]) -> dict:
    selected = [(i, row) for i, row in enumerate(rows)
                if row["partition"] == "fresh_lockbox"
                and int(row["unit"]) == REPRESENTATIVE_UNIT]
    source = np.load(Q4_BOUNDARY, mmap_mode="r")
    binary = VIS / "c4241_qwen4b_corrected_declarative_boundary.float16.npy"
    output = create_binary(binary.name, np.float16,
                           (len(selected) * source.shape[1], source.shape[-1]))
    meta_rows: list[dict] = []
    cursor = 0
    try:
        for source_i, row in selected:
            for q in range(source.shape[1]):
                output[cursor] = source[source_i, q]
                meta_rows.append({
                    "row": cursor,
                    **case_metadata(row),
                    "checkpoint": q,
                    "checkpoint_label": checkpoint_label(q, 36),
                    "metric": "declarative_boundary_activation",
                })
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c4241_qwen4b_corrected_declarative_boundary",
        "Qwen3-4B Corrected Declarative Boundary Trajectory",
        binary,
        meta_rows,
        "ai2050.declarative-boundary-trajectory.v1",
        "Qwen3-4B",
        "full_coordinate_observation",
        "Corrected same-order fresh-lockbox unit26; all six families, two languages, "
        "two surfaces, two states, embedding, all 36 post-block checkpoints and final norm.",
        "Qwen3-4B activation coordinate at the raw declarative continuation boundary",
        "embedding_hiddenstate_full_coordinate",
    )


def publish_q4_all_token() -> dict:
    index = read_jsonl(Q4_ALL_TOKEN_ROWS)
    source = np.load(Q4_ALL_TOKEN, mmap_mode="r")
    binary = VIS / "c4242_qwen4b_declarative_six_family_all_token.float16.npy"
    shutil.copy2(Q4_ALL_TOKEN, binary)
    tokenizer = AutoTokenizer.from_pretrained(
        str(Q4_MODEL), local_files_only=True, trust_remote_code=True, use_fast=False
    )
    decoded: dict[int, str] = {}
    meta_rows: list[dict] = []
    try:
        if source.shape[0] != len(index):
            raise RuntimeError(("all_token_index_mismatch", source.shape, len(index)))
        for row in index:
            token_id = int(row["token_id"])
            if token_id not in decoded:
                decoded[token_id] = tokenizer.decode([token_id])
            q = int(row["checkpoint"])
            meta_rows.append({
                **row,
                "checkpoint_label": checkpoint_label(q, 36),
                "token_position": int(row["token"]),
                "token_text": decoded[token_id],
                "metric": "token_activation",
            })
    finally:
        close_memmap(source)
    return metadata(
        "c4242_qwen4b_declarative_six_family_all_token",
        "Qwen3-4B Declarative Six-Family All-Token Field",
        binary,
        meta_rows,
        "ai2050.declarative-all-token-field.v1",
        "Qwen3-4B",
        "full_coordinate_observation",
        "One fresh-lockbox unit26 narrative case for each family and state; every real "
        "prompt token, every checkpoint and all 2560 activation coordinates.",
        "Qwen3-4B activation coordinate for each actual raw-prefix token",
        "embedding_hiddenstate_full_coordinate",
        {"source_order_note": "This inherited field is narrative-only; its rows are not used for pure surface contrasts."},
    )


def publish_q4_contributions(rows: list[dict]) -> dict:
    binary = VIS / "c4243_qwen4b_corrected_output_contributions.float16.npy"
    shutil.copy2(Q4_CONTRIBUTIONS, binary)
    meta_rows = [{
        "row": i,
        **case_metadata(row),
        "checkpoint": 37,
        "checkpoint_label": "final_norm",
        "metric": "target_minus_wrong_first_token_logit_contribution",
    } for i, row in enumerate(rows)]
    parent = load_json(P2306 / "analysis/final.json")
    return metadata(
        "c4243_qwen4b_corrected_output_contributions",
        "Qwen3-4B Corrected Exact Output Contributions",
        binary,
        meta_rows,
        "ai2050.declarative-output-contribution.v1",
        "Qwen3-4B",
        "exact_final_linear_decomposition",
        "All corrected material rows and all 2560 final-norm coordinates. The values "
        "decompose first-token target-minus-wrong logits, not earlier causal computation.",
        "signed h_j*(W_target,j-W_wrong,j) contribution in original coordinate order",
        "embedding_hiddenstate_full_coordinate",
        {"parent_contribution_audit": parent.get("contributions", {})},
    )


def publish_interface_delta() -> dict:
    binary = VIS / "c4244_qwen4b_qa_to_declarative_signed_delta.float16.npy"
    shutil.copy2(INTERFACE_DELTA, binary)
    meta_rows = read_jsonl(INTERFACE_DELTA_ROWS)
    parent = load_json(P2305 / "analysis/final.json")
    return metadata(
        "c4244_qwen4b_qa_to_declarative_signed_delta",
        "Qwen3-4B QA-to-Declarative Signed Activation Delta",
        binary,
        meta_rows,
        "ai2050.interface-signed-activation-delta.v1",
        "Qwen3-4B",
        "observational_interface_contrast",
        "Fresh-lockbox unit26 source-order-matched QA versus raw declarative boundaries; "
        "embedding, all post-block checkpoints, final norm and all 2560 coordinates.",
        "signed declarative-minus-QA activation coordinate; the two interfaces also differ in prompt package",
        "embedding_hiddenstate_full_coordinate",
        {
            "confound_audit": parent.get("evidence_audit", {}),
            "warning": (
                "This asset visualizes a broad interface-package contrast. It is not a pure "
                "semantic-operation, pure surface, or causal effect."
            ),
        },
    )


def publish_logits(
    source_path: Path,
    rows: list[dict],
    dataset_id: str,
    title: str,
    model: str,
    selector: Callable[[dict], bool],
) -> dict:
    selected = [(i, row) for i, row in enumerate(rows) if selector(row)]
    source = np.load(source_path, mmap_mode="r")
    binary = VIS / f"{dataset_id}.float16.npy"
    output = create_binary(binary.name, np.float16, (len(selected), source.shape[-1]))
    meta_rows: list[dict] = []
    try:
        for cursor, (source_i, row) in enumerate(selected):
            output[cursor] = source[source_i]
            meta_rows.append({"row": cursor, **case_metadata(row), "metric": "next_token_logit"})
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        dataset_id,
        title,
        binary,
        meta_rows,
        "ai2050.declarative-full-vocabulary-logits.v1",
        model,
        "complete_next_token_output_observation",
        "Representative fresh-lockbox unit26 rows retain all 151936 vocabulary logits. "
        "Vocabulary IDs are output coordinates, not HiddenState coordinates.",
        "raw next-token logit indexed by tokenizer vocabulary ID",
        "full_vocabulary_predictive_distribution",
        {"vocabulary_size": 151936},
    )


def publish_q14_field(rows: list[dict]) -> dict:
    parent = load_json(P2307 / "analysis/final.json")
    qpoints = [int(q) for q in parent["field"]["qpoints"]]
    source = np.load(Q14_FIELD, mmap_mode="r")
    binary = VIS / "c4246_qwen14_corrected_frozen_checkpoints.float16.npy"
    output = create_binary(binary.name, np.float16,
                           (len(rows) * len(qpoints), source.shape[-1]))
    meta_rows: list[dict] = []
    cursor = 0
    try:
        for source_i, row in enumerate(rows):
            for q_i, q in enumerate(qpoints):
                output[cursor] = source[source_i, q_i]
                family_result = parent["replication"]["families"][row["family"]]
                meta_rows.append({
                    "row": cursor,
                    **case_metadata(row),
                    "checkpoint": q,
                    "checkpoint_label": checkpoint_label(q, 40),
                    "metric": "declarative_boundary_activation",
                    "frozen_checkpoint": int(family_result["checkpoint"]),
                    "frozen_timing_passed": bool(family_result["passed"]),
                })
                cursor += 1
        output.flush()
    finally:
        close_memmap(output)
        close_memmap(source)
    return metadata(
        "c4246_qwen14_corrected_frozen_checkpoints",
        "Qwen3-14B Corrected Frozen Declarative Checkpoints",
        binary,
        meta_rows,
        "ai2050.qwen14-declarative-frozen-checkpoints.v1",
        "Qwen3-14B",
        "prospective_cross_scale_timing_test",
        "All 288 preselected fresh rows at frozen q33 and final norm q41, retaining all "
        "5120 model-local coordinates and both positive and negative outcomes.",
        "Qwen3-14B activation coordinates at preregistered model-relative checkpoints",
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
    return {
        **dataset,
        "metadata": str(dataset["metadata"].relative_to(ROOT)),
        "binary": str(dataset["binary"].relative_to(ROOT)),
    }


def update_catalog(datasets: list[dict]) -> dict:
    catalog = load_json(CATALOG)
    ids = {row["id"] for row in datasets}
    entries = [{
        "id": row["id"],
        "title": row["title"],
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": row["model"],
        "source_path": "/vis_data/research_kernel/" + row["metadata"].name,
        "binary_path": "/vis_data/research_kernel/" + row["binary"].name,
        "source_schema": row["schema"],
        "coordinate_count": row["shape"][-1],
        "row_count": row["shape"][0],
        "claim_level": row["claim_level"],
        "boundary": row["boundary"],
        "kinds": [row["heatmap_type"]],
    } for row in datasets]
    catalog["datasets"] = entries + [row for row in catalog.get("datasets", [])
                                      if row.get("id") not in ids]
    fields = [{
        "id": row["id"],
        "title": row["title"],
        "url": "/vis_data/research_kernel/" + row["metadata"].name,
        "phase": PHASE,
        "full_coordinate": True,
        "heatmap_type": row["heatmap_type"],
    } for row in datasets]
    catalog["field_datasets"] = fields + [row for row in catalog.get("field_datasets", [])
                                           if row.get("id") not in ids]
    catalog["generated_at"] = datetime.now().astimezone().isoformat()
    save_json(CATALOG, catalog)
    return {
        "added": sorted(ids),
        "dataset_count": len(catalog["datasets"]),
        "field_dataset_count": len(catalog["field_datasets"]),
    }


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
    completed = subprocess.run(
        command,
        cwd=ROOT / "frontend",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
        "passed": completed.returncode == 0,
    }


def cleanup(paths: list[Path]) -> dict:
    result_root = RESULT.resolve()
    rows: list[dict] = []
    total = 0
    for path in paths:
        resolved = path.resolve()
        if result_root not in resolved.parents:
            raise RuntimeError(("cleanup_outside_result", str(path)))
        if not path.exists():
            rows.append({
                "path": str(path.relative_to(ROOT)),
                "status": "already_absent",
                "bytes_deleted": 0,
            })
            continue
        size = path.stat().st_size
        sha = file_hash(path)
        path.unlink()
        total += size
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "status": "deleted_after_verified_visual_derivative",
            "sha256_before": sha,
            "bytes_deleted": size,
        })
    ledger = {"files": rows, "bytes_deleted": total}
    save_json(OUT / "cleanup/ledger.json", ledger)
    return ledger


def evidence_summary() -> dict:
    q4 = load_json(P2306 / "analysis/final.json")
    q14 = load_json(P2307 / "analysis/final.json")
    full_vocab = load_json(P2306 / "probability/full_vocabulary_summary.json")
    return {
        "q4_qualified_families": q4["sequence_ledger"]["qualified_families"],
        "q14_qualified_families": q14["sequence_ledger"]["qualified_families"],
        "q14_timing_passed_families": q14["replication"]["passed_families"],
        "q4_probability_overall": full_vocab["overall"],
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    q4 = load_json(P2306 / "analysis/final.json")
    q14 = load_json(P2307 / "analysis/final.json")
    probability = load_json(P2306 / "probability/full_vocabulary_summary.json")["overall"]
    assets = [{"id": row["id"], "shape": row["shape"], "sha256": row["sha256"]}
              for row in result["datasets"]]
    text = rf"""

## Phase {PHASE}: 原始续写接口全坐标图谱发布与大阶段总审计（{CAMPAIGN}） [{stamp}]

**证据审查与结论修正。** 本阶段审查并整合 Phase2303--2307。附件关于“下一个词训练使前缀条件下的未来竞争成为必要分析轴”的提醒成立，但不能据此把任一 HiddenState 命名为完整语义、充分状态、世界模型或因果齿轮。未修正材料曾把 narrative/dialogue 与相反的事实出现顺序绑定；Phase2305 在揭盲后如实标记该混杂，Phase2306 重新冻结按 unit 配额的同顺序材料并前瞻复验。修复前后 Qwen3-4B 都只有施事--受事、态度--事件、位置绑定 3/6 族通过严格完整序列行为门，说明失败不由该顺序错误单独造成。Qwen3-14B 在预选三族中只有施事--受事与位置绑定通过行为门，且 0/3 族通过冻结 q30 到 q33 的形成时钟复验。因此可保留“接口、状态、表面与输出贡献可以分账”的窄结论；必须删除“跨规模共享简单形成时钟”或“已发现语言齿轮”的过度结论。

**测试原理、材料与用例。** 六个双语构式为施事--受事、态度--事件、比较次序、位置绑定、领属查询和关系从句绑定；每族含英文/中文、narrative/dialogue、两种事实状态、四个冻结分区，共 1536 条原始 declarative prefix。典型英文用例是 `Leo Bell handed the atlas to Amina Arden. The person who handed over the atlas was`，目标续写 `Leo Bell`；中文使用对应自然续写。Phase2306 保证同一 unit 的两个表面具有相同目标事实出现顺序。行为门同时计算目标与错误答案的总对数概率和长度归一化概率；完整词表比较保留 151936 个 logits；HiddenState 场保留 embedding、每个 block 后检查点、final norm 与全部物理坐标。14B 只运行预注册 fresh 分区和冻结族，模型依次加载，未搬运 4B 坐标编号。

**公式。** 完整答案序列分数为：
$$
S(y\mid x)=\sum_{{r=1}}^{{|y|}}\log p(y_r\mid x,y_{{<r}}),
\qquad \bar S(y\mid x)=S(y\mid x)/|y|.
$$

完整词表分布的接口、状态和纯表面变化分别记为：
$$
D_I=D_{{JS}}(P_{{QA}},P_{{decl}}),\qquad
D_S=D_{{JS}}(P_{{s=0}},P_{{s=1}}),\qquad
D_U=D_{{JS}}(P_{{narr}},P_{{dialogue}})\big|_{{\text{{same fact order}}}}.
$$

final norm 对首 token 目标竞争的逐坐标精确分账为：
$$
z_{{y^+}}-z_{{y^-}}
=\sum_{{j=1}}^d h_j\left(W_{{y^+,j}}-W_{{y^-,j}}\right).
$$

冻结的跨规模相对时钟为：
$$
q_{{14}}=\operatorname{{round}}(40q_4/36)=33,
\qquad \operatorname{{Acc}}_{{sign}}(q_{{14}},p)\ge0.75
\quad \text{{for both fresh partitions}}.
$$

**结果汇总。** Qwen3-4B 修正材料严格行为门通过 `{len(q4['sequence_ledger']['qualified_families'])}/6` 族：`{q4['sequence_ledger']['qualified_families']}`；整体 mean-score 准确率 `{q4['sequence_ledger']['overall']['mean_accuracy']}`，sum-score 准确率 `{q4['sequence_ledger']['overall']['sum_accuracy']}`。完整词表总体 JS 为：接口且事实顺序匹配 `{probability['interface_source_order_matched']['js']['mean']}`，接口加事实顺序变化 `{probability['interface_plus_fact_order_change']['js']['mean']}`，状态翻转 `{probability['state_flip']['js']['mean']}`，同顺序纯表面变化 `{probability['surface_same_mention_order']['js']['mean']}`。这说明广义 QA 到原始续写的输入包变化最大；但状态 JS 并不普遍大于表面 JS，不能把全部内部变化归因于语义状态。逐坐标目标--错误贡献的距离与完整词表 JS 不是同一测量对象，不能相互替代。Qwen3-14B 行为通过 `{len(q14['sequence_ledger']['qualified_families'])}/3` 个预选族：`{q14['sequence_ledger']['qualified_families']}`；冻结形成时钟通过 `{len(q14['replication']['passed_families'])}/3`：`{q14['replication']['passed_families']}`。发布资产 `{json.dumps(assets, ensure_ascii=False)}`；逐项验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；前端构建通过 `{result['frontend_build']['passed']}`；清理账本 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**理论进展。** 本大阶段把“语言状态”进一步拆成四个不能混称的对象：原始续写行为、完整下一 token 竞争、边界 HiddenState 形成轨迹、final norm 的输出线性分账。结果支持一种条件化、接口敏感、族特异的预测场图景；不支持固定语义向量、共享坐标字典、跨规模统一形成时钟或因果闭合。基础概率与线性代数已经足够表达本阶段结果，没有证据要求新数学。未来若出现不能被接口、词序、词汇身份和输出码解释的稳定组合规律，再讨论新结构才有实证基础。

**问题硬伤。** 材料自然度仍缺独立人类盲评；两种“接口”改变了整套提示包而非只改变一个语言因素；自由续写 exact-prefix 只是描述性指标，不是冻结严格门；4B 与 14B 属同一模型家族，不能充当独立架构复现；14B 使用统一 logit lens，它是外加读尺而非模型内部声明；只有代表 unit 的全 token 场进入客户端；输出贡献只精确分解最终线性读出，不分解早期计算；观察性 JS、L1 距离和形成时钟均不等于因果机制。

**结论与下一步授权。** 本 Campaign 的具体目标已经完成：把无问句原始续写与 QA 接口分开，修复表面--词序混杂，保存全坐标与完整词表证据，执行 14B 冻结复验，并将重要阳性和阴性结果同时发布。下一阶段若继续，应新建“自由自然文本多步未来轨迹”合同，加入独立自然度审计与不同架构；它会改变材料和研究对象，不应在本锁箱下自动追加模型运行。脚本 `tests/glm5/phase2308_c4241_c4320_declarative_atlas_publication.py`；结果 `tests/glm5/result/phase2308_c4241_c4320_declarative_atlas_publication`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def published_dataset(dataset_id: str) -> dict:
    meta_path = VIS / f"{dataset_id}.json"
    info = load_json(meta_path)
    binary = VIS / Path(info["binary_url"]).name
    return {
        "id": dataset_id,
        "title": info["title"],
        "metadata": meta_path,
        "binary": binary,
        "shape": info["binary_shape"],
        "sha256": info["binary_sha256"],
        "model": info["model"],
        "schema": info["schema"],
        "claim_level": info["claim_level"],
        "boundary": info["boundary"],
        "heatmap_type": info["heatmap_type"],
    }


def finish_result(
    datasets: list[dict],
    verification: list[dict],
    catalog: dict,
    build: dict,
    cleanup_result: dict,
) -> dict:
    parents = [load_json(path / "analysis/final.json")
               for path in (P2303, P2304, P2305, P2306, P2307)]
    checks = {
        "all_parent_phases_passed": all(row["all_checks_passed"] for row in parents),
        "all_assets_verified": all(
            all(value for key, value in row.items() if key != "id") for row in verification
        ),
        "catalog_updated": set(catalog["added"]) == {row["id"] for row in datasets},
        "frontend_built": bool(build["passed"]),
        "raw_hiddenstate_fields_cleaned": all(not path.exists() for path in (
            Q4_BOUNDARY, Q4_ALL_TOKEN, Q4_UNCORRECTED_BOUNDARY, Q14_FIELD
        )),
        "raw_full_vocabulary_fields_cleaned": all(not path.exists() for path in (
            Q4_LOGITS, Q4_UNCORRECTED_LOGITS, Q14_LOGITS
        )),
        "q4_assets_uncompressed": all(row["shape"][-1] in (2560, 151936)
                                        for row in datasets if row["model"] == "Qwen3-4B"),
        "q14_assets_uncompressed": all(row["shape"][-1] in (5120, 151936)
                                         for row in datasets if row["model"] == "Qwen3-14B"),
        "no_browser_or_client_connection": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed",
        "evidence_audit": {
            "retained": [
                "raw declarative continuation separates a useful output interface from QA behavior",
                "corrected same-order contrasts permit separate interface, state, and pure-surface ledgers",
                "complete vocabulary, full checkpoint trajectories, and exact final-output contributions are distinct observables",
            ],
            "corrected": [
                "the first surface contrast was confounded by opposite fact order",
                "state change does not universally dominate pure surface change",
                "none of three frozen Qwen3-4B formation clocks replicated in Qwen3-14B",
                "no fixed semantic vector, shared coordinate dictionary, or causal language gear is established",
            ],
        },
        "summary": evidence_summary(),
        "datasets": [serializable(row) for row in datasets],
        "verification": verification,
        "catalog": catalog,
        "frontend_build": build,
        "cleanup": cleanup_result,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "Corrected raw declarative continuation qualifies three of six Qwen3-4B families, while "
            "Qwen3-14B qualifies two of three frozen families and replicates zero frozen formation clocks. "
            "Interface, state, surface, checkpoint trajectory and exact final-output contributions are now "
            "separately visible, but this remains observational predictive-field evidence rather than causal closure."
        ),
        "next_stage_same_specific_target": False,
        "next_stage_reason": (
            "Free natural continuation, independent human naturality review, and independent architectures "
            "require new materials and a new preregistration."
        ),
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
        "c4241_qwen4b_corrected_declarative_boundary",
        "c4242_qwen4b_declarative_six_family_all_token",
        "c4243_qwen4b_corrected_output_contributions",
        "c4244_qwen4b_qa_to_declarative_signed_delta",
        "c4245_qwen4b_corrected_declarative_vocabulary_logits",
        "c4246_qwen14_corrected_frozen_checkpoints",
        "c4247_qwen14_corrected_declarative_vocabulary_logits",
    ]
    cleanup_path = OUT / "cleanup/ledger.json"
    if cleanup_path.exists() and all((VIS / f"{dataset_id}.json").exists()
                                     for dataset_id in dataset_ids):
        datasets = [published_dataset(dataset_id) for dataset_id in dataset_ids]
        verification = [verify(row) for row in datasets]
        catalog = update_catalog(datasets)
        build = frontend_build()
        if not build["passed"]:
            raise RuntimeError(("frontend_build_failed_during_recovery", build))
        result = finish_result(datasets, verification, catalog, build, load_json(cleanup_path))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parents = [load_json(path / "analysis/final.json")
               for path in (P2303, P2304, P2305, P2306, P2307)]
    if not all(row["all_checks_passed"] for row in parents):
        raise RuntimeError("A parent phase is not authorized")

    q4_rows = read_jsonl(Q4_ROWS)
    q14_rows = read_jsonl(Q14_ROWS)
    datasets = [
        publish_q4_boundary(q4_rows),
        publish_q4_all_token(),
        publish_q4_contributions(q4_rows),
        publish_interface_delta(),
        publish_logits(
            Q4_LOGITS,
            q4_rows,
            "c4245_qwen4b_corrected_declarative_vocabulary_logits",
            "Qwen3-4B Corrected Declarative Complete Vocabulary Logits",
            "Qwen3-4B",
            lambda row: row["partition"] == "fresh_lockbox"
            and int(row["unit"]) == REPRESENTATIVE_UNIT,
        ),
        publish_q14_field(q14_rows),
        publish_logits(
            Q14_LOGITS,
            q14_rows,
            "c4247_qwen14_corrected_declarative_vocabulary_logits",
            "Qwen3-14B Corrected Declarative Complete Vocabulary Logits",
            "Qwen3-14B",
            lambda row: row["partition"] == "fresh_lockbox"
            and int(row["unit"]) == REPRESENTATIVE_UNIT,
        ),
    ]
    verification = [verify(row) for row in datasets]
    if not all(all(value for key, value in row.items() if key != "id")
               for row in verification):
        raise RuntimeError(("asset_verification_failed", verification))

    catalog = update_catalog(datasets)
    build = frontend_build()
    if not build["passed"]:
        raise RuntimeError(("frontend_build_failed", build))

    cleanup_result = cleanup([
        Q4_BOUNDARY,
        Q4_LOGITS,
        Q4_ALL_TOKEN,
        Q4_UNCORRECTED_BOUNDARY,
        Q4_UNCORRECTED_LOGITS,
        Q14_FIELD,
        Q14_LOGITS,
    ])
    result = finish_result(datasets, verification, catalog, build, cleanup_result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
