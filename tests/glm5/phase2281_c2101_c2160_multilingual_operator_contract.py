#!/usr/bin/env python3
"""Freeze the multilingual full-coordinate operator campaign for Phase 2281."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2281_c2101_c2160_multilingual_operator_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2281
CAMPAIGN = "C2101-C2160"
P2265 = RESULT / "phase2265_c1433_c1468_independent_bilingual_contract"
P2266 = RESULT / "phase2266_c1469_c1504_qwen4b_independent_fullfield"
P2274 = RESULT / "phase2274_c1721_c1770_broad_construction_contract"
P2275 = RESULT / "phase2275_c1771_c1820_qwen4b_broad_fullfield"
P2276 = RESULT / "phase2276_c1821_c1890_full_coordinate_structure_tournament"
P2277 = RESULT / "phase2277_c1891_c1960_coordinate_causal_identification"
P2278 = RESULT / "phase2278_c1961_c2030_qwen14_relative_depth_replication"
P2280 = RESULT / "phase2280_c2061_c2100_coordinate_reuse_observation"

BILINGUAL_FAMILIES = (
    "recipient_binding", "patient_binding", "relative_clause_binding", "property_state",
    "location_state", "possession_state", "status_state", "temporal_order",
    "quantifier_sharing", "comparison_order",
)
COMPLEX_FAMILIES = ("conditional_consequence", "conjunction_truth", "classification_chain")
CAUSAL_ANCHORS = ("patient_binding", "location_state")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 多语言条件算子与局部交互大合同（{CAMPAIGN}） [{stamp}]

**附件审查、测试原理与测试用例。** Phase2274-2280 的原始账本支持三项窄结论：Qwen3-4B 的 13 个行为合格构式中有 7 个模型本地逐坐标预测结构；三个冻结坐标联盟均未优于等规模控制；三个中层结构没有按相对层深直接迁移到 Qwen3-14B。附件一对这些边界的判断基本正确。附件二把半密度联合干预误差直接命名为高阶语义协同、把稠密掩码重合命名为多路复用、把分段预测命名为神经开关，并断言必须使用流形、规范场和新数学，均超过证据。本合同不重造小材料，而联合使用已经冻结的 3072 行中英双语十二构式材料与 3072 行英文十六构式材料。内部主对象是中英合同中十个既有双行为合格构式；英文复杂面板增加条件后件、合取真值和两跳分类链。例子包括中英受事绑定、位置状态、时间顺序、量词共享，以及英文“若事件A则事件B”和“A属于B、B属于C”的受控组合。

**冻结对象、分区、算法与公式。** Qwen3-4B 先顺序重建 embedding、36 个 block 后状态、final norm、六个角色和全部 2560 个运行时激活坐标；不读取 Attention、MLP、权重或梯度。对状态变化、表面变化和语言变化分别记账：

$$
R^{{state}}_{{i,q,r,j}}=H_{{i,q,r,j}}(s=1)-H_{{i,q,r,j}}(s=0),
$$

$$
I^{{state\times surface}}_{{i,q,r,j}}
=H_{{1,p}}-H_{{0,p}}-H_{{1,d}}+H_{{0,d}}.
$$

算子竞赛保留全坐标误差，比较族均值、同坐标仿射、四分位分段、符号-幅值、语言迁移、表面迁移、共享族、错族、错配、上一检查点、纯表面和纯输出码控制；不使用 PCA、Top-K 或余弦筛选。因果观察不从高幅坐标选点，而使用覆盖全部坐标的确定性等规模掩码，密度冻结为 $1/64,1/32,1/16,1/8,1/4,1/2$，剂量冻结为 $0.25,0.5,1.0$，分别在 discovery、confirmation、fresh confirmation、fresh lockbox 中拟合、选结构、授权和裁决：

$$
\Delta M_i(z,\alpha)=M_i(H_i^1-\alpha z\odot R_i)-M_i(H_i^1).
$$

**预注册路线与停止条件。** Phase2282 重建完整场；Phase2283 比较模型本地、跨表面和跨语言逐坐标算子；Phase2284 计算不压缩的二阶析因/Walsh 交互图；Phase2285 对最多两个中层锚点执行多密度、多剂量和单坐标加和基线；Phase2286 只有在 Qwen3-4B 出现前瞻跨语言或跨表面阳性时才加载 Qwen3-14B，GLM4 与 DS7B 沿用同一材料上已经冻结的行为资格，不把未资格化路线解释为机制阴性；Phase2287 发布精确坐标图谱并清理未展示原场。一条路线失败只关闭该路线，其余观察路线继续。所有选择在 discovery/confirmation 完成，fresh lockbox 不用于改门槛。

**结果汇总、相关文件与工程审计。** 冻结结果 `{json.dumps(result, ensure_ascii=False)}`。脚本 `tests/glm5/phase2281_c2101_c2160_multilingual_operator_contract.py`；结果 `tests/glm5/result/phase2281_c2101_c2160_multilingual_operator_contract`。模型执行顺序固定为 Qwen3-4B、条件授权的 Qwen3-14B；GLM4 与 DS7B 的现有行为账只用于决定 NA，不并行加载模型。完整原场在最终图谱验证前不得清理；最终仅清理未显示且可由脚本、材料、索引和哈希重建的张量。

**理论进展、问题硬伤与结论。** 理论主体“条件化输出场闭合理论”和 RDC 不改名。本期只把候选对象收紧为“模型本地、角色-深度-基态条件化响应算子及其交互”，没有发现超图、流形、李群、纤维丛或新数学。中英材料仍是研究者编写的受控句，独立人类自然度盲评为 NA；旧行为账与本期重建场之间依赖相同模型版本和材料哈希；同坐标预测可能是一般残差动力；小剂量非加性也可能来自普通网络非线性。严格结论：合同和证据边界冻结完成，没有新增模型机制结果。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = load(final)
        append_memo(result)
        return result
    prior_paths = {
        "bilingual_contract": P2265 / "analysis/final.json",
        "bilingual_behavior": P2266 / "analysis/final.json",
        "broad_contract": P2274 / "analysis/final.json",
        "broad_behavior": P2275 / "analysis/final.json",
        "predictive": P2276 / "analysis/final.json",
        "causal": P2277 / "analysis/final.json",
        "qwen14": P2278 / "analysis/final.json",
        "reuse": P2280 / "analysis/final.json",
    }
    prior = {key: load(path) for key, path in prior_paths.items()}
    preregistration = {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model": True,
        "research_object": "multilingual role-depth-state conditioned full-coordinate response operators",
        "bilingual_families": list(BILINGUAL_FAMILIES),
        "complex_english_families": list(COMPLEX_FAMILIES),
        "languages": ["en", "zh"], "surfaces": ["direct", "paraphrase", "context_control"],
        "partitions": ["discovery", "confirmation", "fresh_confirmation", "fresh_lockbox"],
        "causal_anchors_maximum": list(CAUSAL_ANCHORS),
        "mask_densities": [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2],
        "doses": [0.25, 0.5, 1.0],
        "models_in_order": ["qwen3-4b", "qwen3-14b_if_authorized"],
        "existing_cross_architecture_qualification": {"glm4": "location_only", "deepseek7b": "none"},
        "forbidden_discovery": ["PCA", "Top-K", "cosine screening", "attention", "MLP", "weight inspection"],
        "advanced_math_status": "exploratory_only_after_basic_factorial_and_dose_response_controls",
        "human_blind_review": "NA_not_available",
    }
    save(OUT / "protocol/preregistration.json", preregistration)
    checks = {
        "all_prior_files_exist": all(path.exists() for path in prior_paths.values()),
        "all_prior_checks_passed": all(bool(value.get("all_checks_passed")) for value in prior.values()),
        "bilingual_family_identity": set(BILINGUAL_FAMILIES) == set(prior["bilingual_behavior"]["behavior"]["qualified_families"]),
        "broad_complex_behavior_qualified": set(COMPLEX_FAMILIES).issubset(set(prior["broad_behavior"]["behavior"]["qualified_families"])),
        "q4_predictive_seven": len(prior["predictive"]["lockbox_passed_families"]) == 7,
        "strict_causal_zero": len(prior["causal"]["strict_bidirectional_families"]) == 0,
        "q14_current_replication_zero": len(prior["qwen14"]["structure"]["lockbox_passed_families"]) == 0,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed",
        "timestamp": datetime.now().astimezone().isoformat(),
        "audit": {
            "retained": [
                "seven Qwen3-4B model-local coordinate predictors",
                "three failed coordinate-coalition validations",
                "zero of three current Qwen3-14B relative-depth replications",
                "no reuse enrichment beyond density-preserving controls",
            ],
            "rejected_overclaims": [
                "piecewise prediction proves neuron switches",
                "half-density nonadditivity proves semantic hyperedges",
                "dense overlap proves multiplexing",
                "current evidence requires manifold or gauge mathematics",
            ],
        },
        "preregistration": preregistration,
        "source_hashes": {key: file_hash(path) for key, path in prior_paths.items()},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "The next campaign is frozen around full-coordinate multilingual operator observation, factorial interaction, and scale-controlled intervention; no new mathematics or causal gear is assumed.",
        "next_authorization": "Reconstruct Qwen3-4B bilingual and complex-family role fields without changing prior behavior qualification.",
    }
    save(final, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
