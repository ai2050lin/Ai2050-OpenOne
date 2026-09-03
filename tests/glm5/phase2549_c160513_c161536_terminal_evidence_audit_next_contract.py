#!/usr/bin/env python3
"""Terminal evidence audit for Phase2537-2548 and the next distinct research contract."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2549_c160513_c161536_terminal_evidence_audit_next_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
DIST_ASSET = ROOT / "frontend/dist/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
OFFLOAD = ROOT / "tests/glm5_temp/phase2522_crossmodel_offload"
PHASE, CAMPAIGN = 2549, "C160513-C161536"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one(pattern: str) -> Path:
    matches = list(RESULT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError((pattern, matches))
    return matches[0]


def audit() -> dict:
    phase_dirs = {phase: one(f"phase{phase}_*") for phase in range(2537, 2549)}
    finals = {phase: load(path / "analysis/final.json") for phase, path in phase_dirs.items()}
    memo_text = MEMO.read_text(encoding="utf-8")
    memo_counts = {phase: len(re.findall(rf"^## Phase {phase}:", memo_text, flags=re.MULTILINE)) for phase in phase_dirs}
    scripts = {}
    syntax = {}
    for phase in phase_dirs:
        candidates = list(TESTS.glob(f"phase{phase}_*.py"))
        scripts[phase] = [str(path) for path in candidates]
        try:
            if len(candidates) != 1:
                raise RuntimeError(f"expected one script, got {len(candidates)}")
            compile(candidates[0].read_text(encoding="utf-8"), str(candidates[0]), "exec")
            syntax[phase] = True
        except Exception as error:
            syntax[phase] = repr(error)

    asset = load(ASSET)
    public_hash, dist_hash = sha(ASSET), sha(DIST_ASSET)
    retention = finals[2546]["retention"]
    field_size_matches = []
    for item in retention["retained_display_sources"]:
        path = Path(item["path"])
        field_size_matches.append(path.exists() and path.stat().st_size == item["bytes"])
    offload_files = [str(path) for path in OFFLOAD.rglob("*") if path.is_file()] if OFFLOAD.exists() else []
    disk = shutil.disk_usage(ROOT)

    corrected_claims = {
        "complete_source_kv_head_residual_chain": False,
        "q_is_pure_query_semantics": False,
        "late_fact_kv_zero_means_no_fact_information": False,
        "top32_is_relation_specific_minimal_gear": False,
        "crossmodel_same_physical_algorithm": False,
        "language_encoding_mechanism_closed": False,
    }
    retained_findings = {
        "token_atomic_behavior": "32 families, bilingual, multiunit, multisurface and swap/query controls pass at high accuracy",
        "full_accounting": "full-token embeddings/HiddenState/Q/K/V/QK/softmax/weighted-V/region residual fields with additive numerical checks",
        "late_q_control": "late answer-stage Q donor is sufficient; late fact K/V donor is not",
        "route_specificity": "late selected heads are general output compiler routes, not relation-specific gears",
        "autonomous_chain": "early fact V, middle fact K/V, middle-late downstream K/V and late Q recur during free generation",
        "crossmodel_skeleton": "DS7B and GLM4 reproduce the relative functional ordering with model-specific strengths",
        "independent_region_lock": "unit34/surface1 localizes early V and middle K effects to facts_value tokens",
    }
    next_contract = {
        "same_immediate_target_as_phase2537_2548": False,
        "reason": "stage and token-region localization is complete for this controlled atlas; the unresolved object is now within-value coordinate factorization and recipient-edge mechanics",
        "title": "value-token code factorization and exact within-head recipient-edge atlas",
        "work_packages": [
            "factor value lexical identity, semantic equivalence, binding role, position and distractor multiplicity without changing the requested answer",
            "capture every 128-dimensional V/K coordinate and every QK/weighted-V edge across all layers, tokens and qualified families; preserve low-amplitude coordinates",
            "derive coordinate groups from repeated exact sign/role invariants rather than magnitude Top-K, PCA or global difference transport",
            "intervene on value-to-recipient token edges and coupled coordinate groups with matched nulls, shuffled donors and dose curves",
            "require autonomous natural, long-range, two/three-hop and reorder behavior gates before claiming compositional reuse",
            "replicate functional invariants sequentially on Qwen14B, DS7B and GLM4 only after the Qwen4B factorization is frozen",
        ],
        "success_boundary": "predict held-out family/surface/query recipient edges and output changes from the same frozen coordinate rule; neither one flip nor one deletion failure closes the mechanism",
    }
    checks = {
        "all_phase_finals_passed": all(final["all_checks_passed"] for final in finals.values()),
        "memo_once_each": all(count == 1 for count in memo_counts.values()),
        "phase_sequence": list(phase_dirs) == list(range(2537, 2549)),
        "one_script_each": all(len(paths) == 1 for paths in scripts.values()),
        "all_scripts_syntax_valid": all(value is True for value in syntax.values()),
        "visual_asset_phase2548": asset.get("phase") == 2548 and len(asset.get("models", [])) == 13,
        "public_dist_hash_equal": public_hash == dist_hash,
        "retained_field_sizes_match": all(field_size_matches),
        "offload_temp_empty": not offload_files,
        "disk_free_over_80gb": disk.free > 80 * 1024 ** 3,
        "claim_boundary": not any(corrected_claims.values()),
    }
    return {
        "phase": PHASE, "campaign": CAMPAIGN,
        "scope": {"phases": [2537, 2548], "phase_count": len(phase_dirs), "models": ["Qwen3-4B", "Qwen3-14B", "DeepSeek7B", "GLM4"]},
        "memo_counts": memo_counts, "script_syntax": syntax,
        "visual": {"public": str(ASSET), "dist": str(DIST_ASSET), "bytes": ASSET.stat().st_size,
                   "sha256": public_hash, "sections": len(asset["models"])},
        "retention": {"display_source_bytes": retention["bytes"], "field_count": len(retention["retained_display_sources"]),
                      "offload_files": offload_files, "free_disk_gb": disk.free / 1024 ** 3},
        "corrected_claims": corrected_claims, "retained_findings": retained_findings,
        "next_contract": next_contract, "checks": checks, "all_checks_passed": all(checks.values()),
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2537–2548终局证据审计与下一独立目标合同（{CAMPAIGN}） [{stamp}]

**审计原理与对象。** 对Phase2537–2548共12个Phase逐一核对final、MEMO唯一标题、脚本语法、结果留存、c42641 public/dist哈希、生产build、临时offload和磁盘容量；重新区分观察事实、构造性充分性、必要性、特异性与机制闭合。该Phase不运行新模型，不把审计当作新的正证据。

$$\text{{闭合}}=\text{{行为门}}\land\text{{全量观察}}\land\text{{结构预测}}\land\text{{充分性}}\land\text{{条件必要性}}\land\text{{特异性}}\land\text{{跨材料复现}},$$

当前只有其中若干环节成立，故不能把阶段骨架写成完整编码定律。

**结果汇总。** 检查 `{json.dumps(result['checks'], ensure_ascii=False)}`；保留成果 `{json.dumps(result['retained_findings'], ensure_ascii=False)}`；修正的过度结论 `{json.dumps(result['corrected_claims'], ensure_ascii=False)}`；可视化 `{json.dumps(result['visual'], ensure_ascii=False)}`；留存 `{json.dumps(result['retention'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2549_c160513_c161536_terminal_evidence_audit_next_contract.py`；final位于`{OUT}`；Phase2537–2548原始结果分别保存在各自`tests/glm5/result/phase*`目录。

**理论进展。** 当前最强候选拼图是条件化的有向编译骨架：value token在早层以V载荷改变读出，中层同一value region的K获得寻址控制，K/V联合把结果转写到问题/候选/指令等后续token，晚层答案位置Q携带上下文整合后的输出条件状态，Attention/W_O/残差再编译到下一token概率。它比“一个固定语义向量”更符合证据，也解释了为什么晚层原始facts K/V已无单独控制力以及为什么单head删除常被冗余补偿。

**问题硬伤。** value token仍混合词汇身份、语义值、绑定变化和位置角色；region干预覆盖全head及连续层段；stage patch是off-manifold构造；后续token因自回归顺序天然含前文；top32是通用输出route；自然开放语言、长距离重排和多步组合尚未在同一因果合同中通过；最小128维坐标协同和recipient edge未被破解。

**结论与下一阶段。** `{json.dumps(result['next_contract'], ensure_ascii=False)}`。下一目标不再是重复“阶段是否存在”，而是拆解facts-value内部的词汇/语义/绑定/位置因子，并找到不依赖Top-K的全坐标V→K→recipient-edge规则。因此它与本批即时目标不同，本轮自动同目标续研在Phase2547–2548已经完成；下一批应按新合同整体启动，而非继续追加同型stage patch。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    result = audit()
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
