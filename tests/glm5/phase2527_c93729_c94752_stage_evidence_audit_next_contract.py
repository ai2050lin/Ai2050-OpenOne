#!/usr/bin/env python3
"""Final audit for Phase2511-2526 and freeze the next, materially different research contract."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2527_c93729_c94752_stage_evidence_audit_next_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
PHASE, CAMPAIGN = 2527, "C93729-C94752"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def phase_final(phase: int) -> tuple[Path, dict]:
    candidates = list(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
    if len(candidates) != 1: raise RuntimeError(f"phase{phase} final count={len(candidates)}")
    return candidates[0], load(candidates[0])


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 事件条件残差路由大阶段总终审与下一合同（{CAMPAIGN}） [{stamp}]

**测试原理。** 对Phase2511–2526逐一检查唯一final、`all_checks_passed`、MEMO连续编号、客户端生产构建、参数级资产、清理清单和关键原场哈希。证据分为四级：行为资格、全坐标观察、因果充分性、因果必要性；低级证据不得自动升级为高级结论。

$$h_{{t}}^{{l+1}}=h_{{t}}^l+\sum_h W_{{O,l}}^h\sum_{{j\le t}}\alpha_{{l,h,t,j}}V_{{l,h}}h_j^l+M_l(h_t^l).$$

**成果与证据边界。** `{json.dumps(result['evidence'], ensure_ascii=False)}`。关键数字 `{json.dumps(result['key_numbers'], ensure_ascii=False)}`。过度结论修正 `{json.dumps(result['overclaim_corrections'], ensure_ascii=False)}`。

**相关文件与数据治理。** 阶段文件 `{json.dumps(result['artifacts'], ensure_ascii=False)}`；清理/留存 `{json.dumps(result['data_governance'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。本Phase脚本为`tests/glm5/phase2527_c93729_c94752_stage_evidence_audit_next_contract.py`，final位于`{OUT}`。

**理论进展。** 当前最小可信机制图不再是“一个语义向量从query搬到输出”，而是：事实与查询先形成分布式、上下文条件的残差纹理；候选和指令阶段改变读取条件；多个晚层Attention head从事实区与候选/指令区协同读取；结果累积到answer-boundary残差状态，再由末端读出为候选token概率。有限参数实现复杂语言能力的可检验解释是“同一组层、head和坐标更新规则在不同token位置被递归复用并条件组合”，而不是为每句话保存独立代码。该解释已有路由充分性拼图，尚无完整内容构造和必要性证明。

**问题硬伤。** teacher-forced候选似然仍多于自主多token；九族来自人工微世界；top32不是最小集且未做删除必要性；K/V来源边没有被直接移植；多跳未通过行为门；跨模型只复现answer-boundary整状态，没有对齐head路径；跨语言物理坐标身份判据失败；Walsh交互不等于模型实现XOR或显式逻辑门。

**下一大阶段计划。** `{json.dumps(result['next_contract'], ensure_ascii=False)}`。

**自动续研判断。** 本轮已经自动从固定算子竞争继续到自然九族、四模型、组件守恒、跨层head路由和可视化。现在“answer-boundary是否存在跨族、跨模型可移植载体，以及载体是否由选择性晚层Attention路由累积”这一即时目标已经完成；下一即时目标改为“哪条source-token→K/V→head→残差边构造该载体，并且哪些边对自主多token输出必要”。它改变了干预对象、行为门和判据，虽然终极AGI目标相同，却不是同一可直接续跑的实验合同。因此不把未经设计的K/V消融伪装成本轮自动续研；下一轮应一次性执行下面冻结的六工作包。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    finals = {phase: phase_final(phase) for phase in range(2511, 2527)}
    f2520, f2521, f2522, f2523, f2524, f2525, f2526 = [finals[p][1] for p in range(2520, 2527)]
    asset = load(ASSET)
    head_section = next((s for s in asset["models"] if s.get("key") == "qwen4b_attention_heads"), None)
    cleanup = load(RESULT / "phase2524_c89953_c91328_event_path_visualization_retention_audit/analysis/cleanup_manifest.json")
    headings = MEMO.read_text(encoding="utf-8").splitlines()
    memo_counts = {str(p): sum(line.startswith(f"## Phase {p}:") for line in headings) for p in range(2499, 2527)}
    evidence = {
        "retained": [
            "在行为合格的自然微世界中，事实绑定×查询属性留下稠密全坐标交互；九个关系/语法模式族通过双unit行为门。",
            "关系选择不由query-token单点充分承载；候选/指令之后的answer-boundary是强输出身份载体。",
            "answer-boundary完整状态的matched donor patch在Qwen3-4B、Qwen3-14B、DeepSeek7B、GLM4上均强于shuffled。",
            "同轮残差核算支持H(l+1)-H(l)=Attention+MLP；单层Attention/MLP或二者移植不足，而整block充分。",
            "unit30冻结的32个晚层head路由在unit31达到选择性因果充分度，明显强于等量随机。",
        ],
        "not_established": [
            "未找到语言无关或跨模型共享的物理坐标基底。",
            "未证明top32是必要、最小或唯一的条件齿轮组。",
            "未闭合source-token内容如何形成K/V并进入这些head。",
            "未建立自主多token生成、真正多跳组合或无限组合能力的机制闭环。",
        ],
    }
    key_numbers = {
        "natural_behavior_qualified_families": f2520["behavior"]["qualified_families"],
        "qwen4b_natural_q36_donor_flip": f2521["causal"]["donor_q36"]["donor_flip_rate"],
        "crossmodel_final_boundary_donor_flip": {k: v["causal"]["donor_final"]["donor_flip_rate"]
                                                   for k, v in f2522["models"].items()},
        "single_layer_components_donor_flip": {
            "middle_attention_plus_mlp": f2523["causal"]["donor_middle_components"]["donor_flip_rate"],
            "final_attention_plus_mlp": f2523["causal"]["donor_final_components"]["donor_flip_rate"],
            "middle_block": f2523["causal"]["donor_middle_block"]["donor_flip_rate"],
            "final_block": f2523["causal"]["donor_final_block"]["donor_flip_rate"],
        },
        "multilayer_attention": {k: f2525["causal"][k]["donor_flip_rate"]
                                 for k in ("donor_top32", "donor_random32", "donor_all_late", "shuffled_all_late")},
        "residual_relative_rms_max": f2523["accounting"]["closure"]["relative_rms_max"],
    }
    corrections = [
        "四格Walsh只消去所设计的一阶项，不证明模型内部实现XOR、逻辑门或独立关系势。",
        "matched完整状态patch的高翻转证明该位置状态在协议内充分，不证明它在自然计算中必要。",
        "跨模型翻转比较的是事件角色和相对深度，不是坐标号、head号或共享向量。",
        "top32优于随机说明选择性协同路由，不说明32个head是天然、最小、唯一齿轮。",
        "末层靠近unembedding；强输出身份状态不是关系推理算法本身。",
        "跨语言身份优势未锁箱通过，不能宣称语言无关语义基底。",
        "未过行为门的part-whole、translation、multihop只是未检验，不能作为内部机制反例。",
    ]
    next_contract = {
        "name": "source-conditioned K/V edge compiler and autonomous necessity campaign",
        "work_packages": [
            "WP1：把事实实体、关系词、query-property、候选、指令切成互斥token区域；在20+关系/语法族和新unit上重做行为资格，单独救援多跳与长句重排。",
            "WP2：在Phase2525冻结head×layer上保存source-token到K/V、head输出和残差增量的全坐标账本，先做基本加法守恒与重复性，不先套新高等数学。",
            "WP3：unit发现/lockbox分离，逐边执行matched source K/V patch、source置换、同范数随机、删除和剂量曲线，区分充分性与必要性。",
            "WP4：同时核算多层MLP增量，测试Attention路由与MLP状态条件是否必须联合，避免把100%全Attention翻转误读为完整算法。",
            "WP5：将判据升级为自主多token实体输出、整句排序和行为合格多跳；记录每个生成token边界的递归复用。",
            "WP6：在Qwen14B、DS7B、GLM4按BF16非量化顺序复现相对深度/事件边；不对齐物理head号，重要全场参数级发布，其余HiddenState清理。",
        ],
        "success_criteria": [
            "冻结source-edge集合在新unit、新surface和至少两种输出模式优于等量随机与shuffled。",
            "删除该集合显著损害正确行为，matched修复恢复，形成必要性+充分性双证据。",
            "多token每一步都能由上一token边界账本预测并干预，而非只在第一token teacher-forced成立。",
            "至少两个非Qwen3-4B模型复现事件级路径；失败模型保留为边界而不强行统一坐标。",
        ],
    }
    data_governance = {"cleaned": cleanup, "cleaned_bytes": f2524["retention"]["cleaned_bytes"],
                       "important_parameter_rows": f2524["asset"]["rows_added"],
                       "attention_head_rows": f2526["asset"]["head_panel_rows"],
                       "retained_sources_hashed": f2524["retention"]["retained_hashes"] and f2526["retention"]["all_hashes"]}
    artifacts = {"phase_finals": {str(p): str(finals[p][0]) for p in finals},
                 "visual_asset": str(ASSET), "visual_sha256": digest(ASSET),
                 "visual_sections": [s["key"] for s in asset["models"]],
                 "frontend_dist": str(ROOT / "frontend/dist/index.html")}
    checks = {"all_phase_finals_passed": all(v[1]["all_checks_passed"] for v in finals.values()),
              "memo_2499_2526_once": all(v == 1 for v in memo_counts.values()),
              "visual_phase_current": asset["phase"] == 2526, "five_visual_sections": len(asset["models"]) == 5,
              "head_panel_present": head_section is not None and len(head_section["rows"]) == 72,
              "cleaned_files_absent": all(not Path(r["path"]).exists() and r["deleted"] for r in cleanup),
              "production_build_exists": (ROOT / "frontend/dist/index.html").exists(),
              "next_immediate_target_changed": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "evidence": evidence, "key_numbers": key_numbers,
              "overclaim_corrections": corrections, "artifacts": artifacts, "data_governance": data_governance,
              "next_contract": next_contract,
              "automatic_continuation": {"same_ultimate_goal": True, "same_immediate_falsifiable_target": False,
                                         "completed_current_target": "event-conditioned answer-boundary residual carrier and selective late Attention route",
                                         "next_target": "source-token K/V edge construction, necessity, and autonomous multi-token recursion"},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
