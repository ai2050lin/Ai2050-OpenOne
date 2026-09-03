#!/usr/bin/env python3
"""Rebuild Qwen3-4B multilingual and complex-family coordinate fields."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
sys.path.insert(0, str(TESTS))

import phase2265_c1433_c1468_independent_bilingual_contract as bi_contract  # noqa: E402
import phase2266_c1469_c1504_qwen4b_independent_fullfield as capture  # noqa: E402
import phase2274_c1721_c1770_broad_construction_contract as broad_contract  # noqa: E402
import phase2275_c1771_c1820_qwen4b_broad_fullfield as broad_capture  # noqa: E402
import phase2281_c2101_c2160_multilingual_operator_contract as campaign  # noqa: E402


PHASE = 2282
CAMPAIGN = "C2161-C2220"
BI_SOURCE = RESULT / "phase2266_c1469_c1504_qwen4b_independent_fullfield"
BROAD_SOURCE = RESULT / "phase2275_c1771_c1820_qwen4b_broad_fullfield"
BI_ROWS = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract/material/independent_bilingual_qwen_compiled.jsonl"
BROAD_ROWS = RESULT / "phase2274_c1721_c1770_broad_construction_contract/material/broad_construction_qwen_compiled.jsonl"

BI_FIELD = OUT / "raw/qwen4b_bilingual_role_field.float16.npy"
BI_INDEX = OUT / "raw/qwen4b_bilingual_role_index.jsonl"
BI_PROGRESS = OUT / "raw/qwen4b_bilingual_role_progress.json"
COMPLEX_FIELD = OUT / "raw/qwen4b_complex_role_field.float16.npy"
COMPLEX_INDEX = OUT / "raw/qwen4b_complex_role_index.jsonl"
COMPLEX_PROGRESS = OUT / "raw/qwen4b_complex_role_progress.json"
TOKEN_FIELD = OUT / "raw/qwen4b_representative_token_field.float16.npy"
TOKEN_INDEX = OUT / "raw/qwen4b_representative_token_index.jsonl"
TOKEN_PROGRESS = OUT / "raw/qwen4b_representative_token_progress.json"
TOKEN_FAMILIES = (
    "patient_binding", "location_state", "temporal_order", "quantifier_sharing",
    "conditional_consequence", "classification_chain",
)
TOKEN_UNITS = (16, 24)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_role(field: Path, index: Path, progress: Path) -> None:
    capture.ROLE_FIELD = field
    capture.ROLE_INDEX = index
    capture.ROLE_PROGRESS = progress


def configure_token() -> None:
    capture.TOKEN_FIELD = TOKEN_FIELD
    capture.TOKEN_INDEX = TOKEN_INDEX
    capture.TOKEN_PROGRESS = TOKEN_PROGRESS


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: Qwen3-4B中英与复杂构式全坐标原场重建（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 本期不重新出题或重跑行为门，而严格读取 Phase2265/2266 的 3072 行中英材料及双行为账、Phase2274/2275 的 3072 行英文广构式材料及双行为账。中英主场只包含既有十个合格构式：受事者、受事、关系从句、属性、位置、持有、状态、时间、量词和比较；复杂英文场加入条件后件、合取真值和两跳分类链。错误回答样本仍保留，不能后筛。一次 Qwen3-4B BF16 CUDA 会话依次采集中英主场、复杂场和代表全 token 场，完成后释放模型。

**数学公式与测量对象。** 六角色场和真实 token 场分别为：

$$
\mathcal F=\left\{{H_{{i,q,r,j}}\right\}},\qquad
\mathcal T=\left\{{H_{{i,q,t,j}}\right\}},
$$

其中 $q$ 覆盖 embedding、36 个 block 后状态和 final norm，$r$ 覆盖 primary、secondary、relation、context、query、boundary，$j=1,\ldots,2560$ 是模型本地运行时激活坐标。代表 token 场冻结 unit16、unit24，并覆盖中英受事、位置、时间、量词以及英文条件和分类链。本期不读取 Attention、MLP、权重或梯度，不进行 PCA、Top-K、余弦筛选或差分搬运。

**结果汇总与门槛。** 中英场 `{json.dumps(result['bilingual_field'], ensure_ascii=False)}`；复杂场 `{json.dumps(result['complex_field'], ensure_ascii=False)}`；全 token 场 `{json.dumps(result['token_field'], ensure_ascii=False)}`；冻结行为来源 `{json.dumps(result['behavior_sources'], ensure_ascii=False)}`；量化与放置 `{json.dumps(result['model'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`，总通过 `{result['all_checks_passed']}`。

**相关文件。** 脚本 `tests/glm5/phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild.py`；结果与可重建原场 `tests/glm5/result/phase2282_c2161_c2220_qwen4b_multilingual_field_rebuild`。材料、模型版本、逐样本索引、角色位置、精度、形状、进度和 SHA-256 均落盘；最终可视化派生验证前不清理原场。

**分析、理论进展、问题硬伤与结论。** 这是观测资产重建，不是新机制阳性。它恢复了可在同一逐样本原场上比较语义状态、表面、语言和复杂组合的条件，但中英是人工平行模板，独立人类盲评仍为 NA；末 token 角色代表、多 token 语义跨度和 float16 写盘仍是近似；旧行为账依赖模型文件与材料哈希未变化。严格结论：`{result['strict_conclusion']}` 下一阶段只能按冻结分区在 discovery 拟合，在 confirmation 选择，再揭示 fresh confirmation 与 lockbox。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    contract_result = load(campaign.OUT / "analysis/final.json")
    if not contract_result["all_checks_passed"]:
        raise RuntimeError("Phase2281 contract failed")
    bi_final = load(BI_SOURCE / "analysis/final.json")
    broad_final = load(BROAD_SOURCE / "analysis/final.json")
    bi_all = read_rows(BI_ROWS)
    broad_all = read_rows(BROAD_ROWS)
    bi_candidates = read_rows(BI_SOURCE / "behavior/candidate.jsonl")
    bi_generated = read_rows(BI_SOURCE / "behavior/generation.jsonl")
    broad_candidates = read_rows(BROAD_SOURCE / "behavior/candidate.jsonl")
    broad_generated = read_rows(BROAD_SOURCE / "behavior/generation.jsonl")
    bi = [row for row in bi_all if row["family"] in campaign.BILINGUAL_FAMILIES]
    complex_rows = [row for row in broad_all if row["family"] in campaign.COMPLEX_FAMILIES]
    bi_index = capture.index_rows(bi, bi_candidates, bi_generated)
    complex_index = broad_capture.index_rows(complex_rows, broad_candidates, broad_generated)
    token_rows = [row for row in bi if row["family"] in TOKEN_FAMILIES and row["unit"] in TOKEN_UNITS]
    token_rows += [row for row in complex_rows if row["family"] in TOKEN_FAMILIES and row["unit"] in TOKEN_UNITS]
    model = None
    try:
        model, tokenizer, device, placement = bi_contract.legacy.parent.model_base.qwen_model()
        configure_role(BI_FIELD, BI_INDEX, BI_PROGRESS)
        bilingual_field = capture.capture_role_field(model, device, bi, bi_index)
        configure_role(COMPLEX_FIELD, COMPLEX_INDEX, COMPLEX_PROGRESS)
        complex_field = capture.capture_role_field(model, device, complex_rows, complex_index)
        configure_token()
        token_field = capture.capture_all_token_field(model, device, token_rows)
        bf16 = bi_contract.legacy.parent.model_base.scope.parent.previous.model_base()
        quantization = bf16.quantization_audit(model)
    finally:
        if model is not None:
            bi_contract.legacy.parent.model_base.scope.parent.previous.model_base().release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    hashes = {name: file_hash(path) for name, path in {
        "bilingual_field": BI_FIELD, "bilingual_index": BI_INDEX,
        "complex_field": COMPLEX_FIELD, "complex_index": COMPLEX_INDEX,
        "token_field": TOKEN_FIELD, "token_index": TOKEN_INDEX,
    }.items()}
    checks = {
        "bilingual_rows": len(bi) == 10 * 256,
        "complex_rows": len(complex_rows) == 3 * 192,
        "token_panel_multiple_families": len({row["family"] for row in token_rows}) == len(TOKEN_FAMILIES),
        "bilingual_shape": bilingual_field.get("shape") == [len(bi), 38, 6, 2560],
        "complex_shape": complex_field.get("shape") == [len(complex_rows), 38, 6, 2560],
        "token_full_coordinate": token_field.get("shape", [0])[-1] == 2560,
        "frozen_bilingual_behavior": set(bi_final["behavior"]["qualified_families"]) == set(campaign.BILINGUAL_FAMILIES),
        "frozen_complex_behavior": set(campaign.COMPLEX_FAMILIES).issubset(set(broad_final["behavior"]["qualified_families"])),
        "all_hashes": all(len(value) == 64 for value in hashes.values()),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": {"name": "Qwen3-4B", "placement": placement, "quantization": quantization},
        "behavior_sources": {
            "bilingual_final_sha256": file_hash(BI_SOURCE / "analysis/final.json"),
            "broad_final_sha256": file_hash(BROAD_SOURCE / "analysis/final.json"),
            "behavior_rerun": False,
        },
        "bilingual_field": bilingual_field, "complex_field": complex_field,
        "token_field": token_field, "hashes": hashes, "checks": checks,
        "all_checks_passed": all(checks.values()),
        "strict_conclusion": "Frozen behavior-qualified Qwen3-4B bilingual and complex construction fields were reconstructed at every checkpoint, role, and activation coordinate; this is a reusable observational asset, not a mechanism result.",
        "next_authorization": "Run the preregistered cross-surface and cross-language coordinate-operator tournament.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
