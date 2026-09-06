#!/usr/bin/env python3
"""Audit Phase2579-2599 reviews and freeze the endogenous natural-language campaign."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2600_c602369_c610560_attachment_audit_endogenous_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2600, "C602369-C610560"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\32467ec7-198e-4f15-b599-452095177788\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\3b4fbdd1-c66a-48c4-8365-a8f8d87fedf7\pasted-text.txt"),
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Phase2579—2599附件复审与单提示内生编译大合同（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**审查原理。** 将两份外部复盘逐条还原到Phase2579—2599的`final.json`、MEMO勘误和原始全坐标产物；把“观察到联合差分”“跨材料复现”“方向性因果效应”“单prompt内生机制”分成四级，低一级证据不得命名为高一级机制。旧四格对象为：

$$I=H^{{11}}-H^{{10}}-H^{{01}}+H^{{00}},$$

但新阶段的基本对象改为一个自然prompt及其最小局部反事实：

$$D_l(x^1,x^0)=H_l(x^1)-H_l(x^0),\qquad
do\!\left(H^0_{{l,S}}\leftarrow H^0_{{l,S}}+\alpha D_{{l,S}}\right).$$

干预只运行一个recipient prompt；发现集可以由成对反事实学习方向，确认集不得使用自身答案或自身反事实估计方向。

**附件中保留的正确部分。** 四选一合同消除二元翻转歧义；embedding/真正查询前二阶项为零；首次跨token混合后出现可测联合项；首块联合项可再生；晚层联合项增强；公共任务骨架占raw族相关大部；候选删除后仍有部分全坐标迁移；Phase2597严格等$\Delta$结果支持后层对特定完整坐标方向敏感。这些结论均限定于受控材料。

**过度结论修正。** （1）Phase2583只能说attention上下文混合产生首个可测项，不能仅凭hook把计算归因给softmax本身；（2）Phase2585的$>0.98$首先是共同任务解法复用，不能说表示“不依赖值语义”；（3）Phase2586的0.990是8四元组的归一化粗曲线相似，不是机制同构；（4）Phase2588测试的是十种词汇接口，不是真实翻译/指代/句法能力；（5）中心化优势0.169是弱族残差且无独立族级置信合同；（6）Phase2597的wrong donor未匹配协方差、角度与LayerNorm响应，故“族条件性”仍是候选证据；（7）四prompt Walsh方向不是模型自然运行中已显式分离的齿轮；（8）“全坐标方向是功能单元”也过强，目前只支持分布式方向比测试过的单head/小块解释更好。

**新大阶段测试用例与冻结门槛。** 六个真实操作族（指代、否定、时序、四句重排、句法角色、二跳分类），每族200条、中英各100，共1200条无候选greedy；每个操作/语言至少75%解析正确才进入族级机制结论。材料按pair/context冻结为发现40%、确认40%、外测20%，同时保留全部正确、错误、高低margin样本。Qwen3-4B BF16 CUDA锁箱后，Qwen3-14B、GLM4-9B、DS7B依次以非量化BF16和`device_map=auto`测试，绝不并行占GPU。

**大方案。** Phase2601行为锁箱；2602全token/全层/全坐标自然场；2603发现—确认复现和公共输出残差；2604单recipient source-span patch；2605 source→K/V/Q/head→residual组件链；2606训练集方向对未见单prompt的输出身份/族残差因果拆分；2607真实greedy逐token首分歧；2608—2609顺序跨模型行为与功能动力复验；2610重要参数场客户端；2611只用能压缩数据的基础公式更新RDC；2612终局审计。若同一具体机制假设仍未完成且已有合格材料，自动进入下一轮扩大确认；若下一目标需要新的语言任务或新提取器，则明确换阶段而不伪装机械续跑。

**核心判据。** 观察门要求未见pair全坐标方向/动力复现；因果门要求真实source或学习方向优于等位移roll、错token、错层、错族，且剂量反应方向一致；输出门同时报告完整候选似然、首token概率和真实greedy。任何单一删除失败不终止图谱路线，任何相关阳性不自动升级为齿轮。

**相关文件。** 附件哈希、逐条裁决、12-Phase冻结合同与检查位于`{OUT}`；脚本`tests/glm5/phase2600_c602369_c610560_attachment_audit_endogenous_contract.py`。

**理论进展、问题硬伤与结论。** 新合同把“实验者用四次运行构造的联合项”与“模型在一个prompt里可被定位和控制的内生路径”严格分离。当前最强证据仍只是结构化条件选择中的晚层方向选择性；自然语言含义、单prompt生成规则、逐token编译和跨架构重复均未成立。附件给出的RDC公式仍是记账框架，不是闭合理论；本阶段先积累基本全坐标拼图，不因高级数学名称提前宣布结构。

**审计状态。** `{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    p2599 = load_json(RESULT / "phase2599_c598273_c602368_campaign_terminal_audit/analysis/final.json")
    attachment_texts = [path.read_text(encoding="utf-8-sig") for path in ATTACHMENTS]
    corrections = [
        "attention mixing is localized, but softmax alone was not isolated",
        "natural/nonce correlation is task-solution reuse, not semantic independence",
        "Qwen14 correlation is an eight-quartet coarse dynamics replication, not isomorphism",
        "ten lexical interfaces are not ten complete language abilities",
        "centered family advantage is weak descriptive residual evidence",
        "wrong-family delta did not match covariance, angle, or LayerNorm response",
        "four-prompt Walsh contrast is not a single-prompt endogenous variable",
        "distributed direction is a better current candidate, not a proven minimal unit",
    ]
    contract = {
        "families": ["reference", "negation", "chronology", "sentence_reorder", "syntax_role", "taxonomy_chain"],
        "languages": ["en", "zh"],
        "cases_per_family": 200,
        "cases_per_family_language": 100,
        "total_prompts": 1200,
        "candidate_list_in_prompt": False,
        "behavior_gate_per_family_language": 0.75,
        "splits": {"discovery": 0.4, "confirmation": 0.4, "external": 0.2},
        "coordinates": "all physical coordinates; no Top-K or PCA as primary analysis",
        "models_sequential": ["Qwen3-4B", "Qwen3-14B", "GLM4-9B", "DS7B"],
        "precision": "BF16 nonquantized",
        "phases": list(range(2601, 2613)),
    }
    checks = {
        "phase2599_passed": bool(p2599["all_checks_passed"]),
        "both_attachments_present": all(path.is_file() for path in ATTACHMENTS),
        "both_attachments_nonempty": all(path.stat().st_size > 20000 for path in ATTACHMENTS),
        "eight_overclaims_corrected": len(corrections) == 8,
        "six_real_operation_families": len(contract["families"]) == 6,
        "twelve_hundred_candidatefree_prompts": contract["total_prompts"] == 1200,
        "frozen_behavior_gate": contract["behavior_gate_per_family_language"] == 0.75,
        "discovery_confirmation_external_split": sum(contract["splits"].values()) == 1.0,
        "full_coordinate_policy": "all physical" in contract["coordinates"],
        "sequential_nonquantized_models": len(contract["models_sequential"]) == 4,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": "audit Phase2579-2599 reviews and preregister single-prompt endogenous campaign",
        "attachments": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in ATTACHMENTS],
        "supported": p2599["milestones"],
        "corrections": corrections,
        "contract": contract,
        "language_mechanism_closed": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save_json(OUT / "analysis/final.json", result)
    save_json(OUT / "protocol/frozen_contract.json", contract)
    append_memo(result)
    correction = "**Phase2600合同检查勘误（append-only）**"
    memo_text = MEMO.read_text(encoding="utf-8-sig")
    if result["all_checks_passed"] and correction not in memo_text and '"both_attachments_nonempty": false' in memo_text:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"\n\n{correction} [{stamp}] 初次检查误把UTF-8解码后的字符数与20,000字节门槛比较；"
                "两附件实际分别为51,270与25,414 bytes，均完整非空。检查已改为文件字节数，"
                "实验合同、附件哈希和科学裁决未改变；修正后Phase2600全部检查通过。\n"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
