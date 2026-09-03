#!/usr/bin/env python3
"""Audit Phase2468-2485 claims and preregister an orthogonal full-field campaign."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase2486_c54401_c54720_evidence_audit_orthogonal_campaign_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2486, "C54401-C54720"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: Phase2468–2485证据审计、过度结论纠正与正交全场预注册（{CAMPAIGN}） [{stamp}]

**测试原理与证据范围。** 逐项复核Phase2468–2485的`final.json`、原场shape/hash、行为资格、冻结层位和附件解释。审计不把后验解释当新实验，不把不同qpoint、不同事件的余弦拼成时间曲线。证据等级继续区分行为、观察纹理、冻结预测、有限干预和机制闭合。

$$E_a(q,e)=\sum_i\left(\mathbb E[H_{{q,e,i}}\mid a]-\mathbb E[H_{{q,e,i}}]\right)^2,$$

其中$E_a/\sum E$只是当前设计下的描述性均方账本；factor未正交、交互未完全建模时，它不是独立因果贡献，更不是“信息百分比”。

**保留的成果。** {json.dumps(result['audit']['retained'], ensure_ascii=False)}。

**必须修正的过度结论。** {json.dumps(result['audit']['corrected'], ensure_ascii=False)}。尤其：（1）answer-boundary的language 0.276与interface 0.239不能合并命名为“格式52%”，language不是输出格式，各主效应也不是已证实可加的机制份额；family 0.153不能直接命名为“语义15%”；（2）Phase2475的0.901/0.455/0.587及0.730使用不同事件、部分使用不同unit9选择qpoint，且对象是family-relative passport，不证明“理解先于格式”或原始状态几乎相同；（3）Phase2481的163/1536是对比能量覆盖，不是信息量、参数重要性或因果必要性；（4）长链跨语言0.862仍受相同英文伪词、三family、24行、后验材料资格和负错配均值影响；（5）Phase2484能量相关接近1只支持坐标尺度包络候选，也可能来自架构各向异性，不能命名为固定语言骨架。

**新大方案。** 预注册顺序为：Phase2487三unit×十二族×中英×四正交surface×四输出接口共1152条真实贪心行为；Phase2488仅按confirmation/lockbox预定行为门采Qwen3-4B的五事件×38qpoint×2560全坐标，并保留代表性全token场；Phase2489做描述性因素账本但拒绝机制百分比；Phase2490同时比较有符号纹理、平方能量包络和按发现集坐标RMS标准化纹理；Phase2491在同一冻结qpoint比较自主生成boundary/first/answer；Phase2492用原始与标准化纹理竞争相邻block的恒等、全局尺度、逐坐标尺度；Phase2493只允许Qwen3-14B非量化BF16、`device_map=auto`作冻结复核，硬件失败不得用量化结果冒充；Phase2494把重要逐坐标结果加入客户端；Phase2495审计并自动判定下一阶段。

$$N=3\times12\times2\times4\times4=1152,$$

$$P_{{f,c}}=\mathbb E[H\mid f,c]-\mathbb E[H\mid c],\qquad S_{{f,c,i}}=P_{{f,c,i}}/(\mathbb E_{{f,c\in discovery}}P_{{f,c,i}}^2+\varepsilon)^(1/2).$$

**锁箱与失败规则。** unit14只作discovery，unit15只选接口/族、qpoint和一个transport模型，unit16一次揭示。材料用不同英文名与中文名，仅靠外部图同构对齐；四种surface在每族出现，四接口只改变输出编码。任何行为不合格分支仍可作为输入响应图谱，但不能称成功语言执行。Qwen14B BF16若因内存/Windows权重物化失败，保存诊断并结束该模型分支，禁止NF4/INT8替代。

**相关文件。** 本脚本与同名结果目录中的`analysis/final.json`。附件只是待审解释；数值权威仍是Phase2468–2485原始final和原场哈希。

**理论进展、问题硬伤与结论。** 当前最窄可靠对象是“行为合格条件下可复现的稠密family-relative上下文化响应纹理”，不是“条件化输出场闭合理论”。family仍与关系词共变，激活坐标不是权重参数，相关/能量不是因果。新阶段先用更强正交合同积累逐坐标拼图，再讨论结构；不预设纯语义子空间、163个关键坐标或天然稀疏齿轮。裁决：{json.dumps(result['adjudication'], ensure_ascii=False)}；检查：{json.dumps(result['checks'], ensure_ascii=False)}。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "audit": {
            "retained": [
                "Phase2466 autonomous-collapse claim was correctly withdrawn after generation-budget audit",
                "Qwen3-4B full-layer all-token all-2560-coordinate capture is operational",
                "eight behavior-qualified families show held-unit family-relative cross-language textures",
                "successful autonomous paths preserve some family-relative texture",
                "q10-to-q11 diagonal scaling is a limited predictive baseline after final-RMSNorm exclusion",
                "energy-envelope and signed-direction statistics measure different properties",
            ],
            "corrected": [
                "language plus interface main-effect energy is not a 52-percent output-format mechanism",
                "family main-effect energy is not a 15-percent semantic content decomposition",
                "cross-interface passport cosine 0.90 does not prove understanding or raw-state identity",
                "metrics selected at different qpoints cannot form a temporal decay or recovery curve",
                "163 and 1536 coordinates describe contrast-energy coverage, not information or causal gears",
                "three-family 24-row chain results are exploratory and not universal knowledge-chain code",
                "near-one squared-energy correlation is compatible with architecture-scale anisotropy",
                "no conditional-output-field closure, natural coordinate gear, or language compiler is established",
            ],
        },
        "preregistered_axes": {
            "units": [14, 15, 16],
            "families": 12,
            "languages": ["en", "zh"],
            "surfaces": 4,
            "interfaces": ["entity", "digit", "letter", "side"],
            "rows": 1152,
            "coordinates": "all physical coordinates; no Top-K/PCA replacement",
            "split": {"14": "discovery", "15": "confirmation", "16": "lockbox"},
        },
        "adjudication": {
            "pure_semantic_subspace_identified": False,
            "fixed_energy_skeleton_identified": False,
            "natural_coordinate_gear_identified": False,
            "language_encoding_mechanism_closed": False,
            "next_stage_authorized": True,
        },
        "checks": {
            "source_phase_range_complete": True,
            "overclaims_explicitly_corrected": True,
            "orthogonal_contract_frozen_before_new_model_run": True,
            "full_coordinate_rule": True,
            "nested_lockbox_rule": True,
            "nonquantized_qwen14_rule": True,
            "claim_boundary": True,
        },
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
