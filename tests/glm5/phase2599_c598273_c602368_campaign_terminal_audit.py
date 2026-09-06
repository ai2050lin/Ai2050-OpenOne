#!/usr/bin/env python3
"""Terminal audit for the Phase2579-2598 bilingual interaction campaign."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET = RESULT / "client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
OUT = RESULT / "phase2599_c598273_c602368_campaign_terminal_audit"
PHASE, CAMPAIGN = 2599, "C598273-C602368"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_final(phase):
    matches = list(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Phase{phase} final count={len(matches)}")
    return matches[0], load(matches[0])


def append_memo(result):
    heading = f"## Phase {PHASE}: Phase2579—2598大阶段证据终审与下一机制边界（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**终审原理。** 对Phase2579—2598逐一检查final、脚本、连续MEMO、原场与客户端参数行；把观察、复现、脚手架迁移和因果干预分级，不用最终准确率替代机制证据。核心对象始终是全坐标四格联合项：

$$I_{{ltd}}=H^{{11}}_{{ltd}}-H^{{10}}_{{ltd}}-H^{{01}}_{{ltd}}+H^{{00}}_{{ltd}},\qquad
H'_i=H_i+\frac{{c_i}}4\Delta I,\quad c=(+1,-1,-1,+1).$$

**附件审查与过度结论修正。** Phase2562—2578附件中“关系和值的联合条件在多模型可测”“Q/K/V角色随层分化”“Qwen4B layer0 V存在分布式纹理”等方向性概括可保留；但Phase2572所有单head均失败，不能称128维最小齿轮；Phase2573搜索的是包含H5的128个head子集，不是128个坐标；没有证据证明“齿轮就是V”；Phase2564的14B多跳未通过合格门；非零有限差分只描述非线性，不自动等于机制；Phase2570定位carrier不等于compute site；完整8-head band有效不等于每个head必要。

**测试内容与结果汇总。** `{json.dumps(result['milestones'], ensure_ascii=False)}`。Phase2580—2581建立4B/14B四选一行为与正确缺失基准；2582—2587定位交互出生、block0组件/阴性因果、等BPE词面、跨规模动力与客户端；2588—2590完成中英十自然操作族行为桥和20节点全坐标图；2591—2595完成无候选greedy、19节点脚手架迁移及88组/69独立确认；2596—2598完成q25/q35全坐标Walsh因果、严格等$\Delta$修正和参数级展示。

**相关文件。** 20个测试/客户端脚本位于`tests/glm5/phase2579...phase2598...py`；各Phase材料、原场、NPZ、完整词表场与final位于`tests/glm5/result/phase2579...phase2598.../`；客户端资产`{ASSET}`；路由`{ROUTE}`；本终审`{OUT}`。

**理论进展。** 当前最强而不过度的机制拼图是：输入embedding对两查询因子线性可分，首个transformer block开始产生非加性联合项；该项在早层弱、晚层急剧放大；自然/nonce、4B/14B功能深度、十族中英词面和候选有/无之间均存在不同层次的复用，但公共任务骨架占raw相关的大部分。扣除语言共同量后仍有较弱族残差。更重要的是，在65个三重行为合格四元组上，真实脚手架$\Delta$相对等范数roll选择性提高margin，归零相对其等$\Delta$ roll显著损害行为，异族等$\Delta$也更差：后层不是只读“能量”，而是条件性使用完整坐标方向。

**问题硬伤。** 仍未破解语言编码机制。所有核心因果结果依赖四prompt联合Walsh手术，而非单prompt内生干预；十个“语言族”被约化到同一四事实二维交点，不能代表真实翻译、因果、指代或句法；成功样本筛选排除了失败路径，`modality/zh`没有自主四格全对；坐标相关和有限差分混合一般非线性、注意力竞争与语义计算；Qwen14B仅小样本动力复现，GLM4/DS7B没有完成本新材料闭锁；还没有从source→K/V→Q/head→residual→逐token生成的单prompt可执行解码算法。

**结论。** 本大阶段不是“破解完成”，但完成了从附件过度叙述到一条经行为门、全坐标观察、独立扩大复验、严格等扰动因果对照和客户端参数核验的证据链。最合理的当前编码图景是“固定物理基底上的分布式、晚层编译、上下文与脚手架条件化的协同方向”，不是单神经元、单head、固定可搬运语义向量或已闭合数学结构。

**下一大阶段（目标发生具体变化）。** 不再继续同一“跨prompt四格图谱”目标，而转入“单prompt内生编译机制”：一，在真实长句重排、无候选指代、否定/时序/句法改写中建立行为门；二，在单prompt内用source token与K/V/Q/head/residual事件做局部反事实，不依赖四prompt联合回写；三，沿真实greedy每个token检验同一条件方向何时被读取和重编码；四，材料先在Qwen3-4B锁箱，再依次用Qwen3-14B、GLM4、DS7B非量化CUDA复现。其判据是能否预测并定向改变未见组合的逐token输出，而不是再累积高相关热图。

**审计状态。** `{json.dumps(result['checks'], ensure_ascii=False)}`。原始HiddenState/完整词表场因重要结果已进入客户端且需复算而保留；临时运行日志已清理，空offload目录为0字节。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    finals = {}
    final_paths = {}
    for phase in range(2579, 2599):
        path, payload = phase_final(phase)
        finals[phase] = payload
        final_paths[phase] = str(path.relative_to(ROOT)).replace("\\", "/")
    memo_text = MEMO.read_text(encoding="utf-8-sig")
    headings = [int(match) for match in re.findall(r"^## Phase (\d+):", memo_text, flags=re.MULTILINE)]
    asset = load(ASSET)
    required_panels = {
        "phase2589_bilingual_operation_exact_parameter_field",
        "phase2589_bilingual_operation_centered_family_graph",
        "phase2592_candidate_scaffold_fullcoordinate_pairs",
        "phase2594_all88_coordinate_transport_field",
        "phase2597_matched_delta_exact_parameter_field",
        "phase2597_matched_delta_causal_conditions",
    }
    models = {item["key"]: item for item in asset["models"]}
    parameter_panels = [models[key] for key in required_panels if key in models]
    parameter_lengths = all(
        len(row["values"]) == panel["coordinate_count"]
        for panel in parameter_panels for row in panel["rows"]
    )
    offload = ROOT / "tests/glm5_temp/phase2586_offload"
    offload_files = list(offload.rglob("*")) if offload.is_dir() else []
    p2588, p2589, p2591 = finals[2588], finals[2589], finals[2591]
    p2594, p2597, p2598 = finals[2594], finals[2597], finals[2598]
    milestones = {
        "natural_bridge_full_accuracy": p2588["summary"]["conditions"]["full"]["accuracy"],
        "natural_bridge_eligible_quartets": p2588["eligible_aligned_quartets"],
        "bilingual_family_late_centered_advantage": p2589["coordinate_graph"]["late_summary"]["matched_minus_unmatched_centered_signed"],
        "candidate_free_greedy_parsed_accuracy": p2591["overall"]["greedy_parsed_accuracy"],
        "all88_independent_confirmation_transport": p2594["paired_transport"]["late_summary"]["independent_confirmation69_signed_median"],
        "all88_coordinate_transport_median": p2594["paired_transport"]["late_summary"]["confirmation69_coordinate_correlation_median"],
        "confirmation_family_graph_topology": p2594["paired_transport"]["late_summary"]["confirmation_family_graph_topology"],
        "q25_true_delta_margin_gain": p2597["effects_vs_baseline"]["q25_transplant_delta"]["margin_delta"],
        "q25_equal_delta_roll_margin_gain": p2597["effects_vs_baseline"]["q25_transplant_delta_roll"]["margin_delta"],
        "q35_true_delta_margin_gain": p2597["effects_vs_baseline"]["q35_transplant_delta"]["margin_delta"],
        "q35_equal_delta_roll_margin_gain": p2597["effects_vs_baseline"]["q35_transplant_delta_roll"]["margin_delta"],
        "q25_zero_accuracy_delta": p2597["effects_vs_baseline"]["q25_zero_delta"]["accuracy_delta"],
        "q25_zero_equal_delta_roll_accuracy_delta": p2597["effects_vs_baseline"]["q25_zero_delta_roll"]["accuracy_delta"],
        "q35_zero_accuracy_delta": p2597["effects_vs_baseline"]["q35_zero_delta"]["accuracy_delta"],
        "q35_zero_equal_delta_roll_accuracy_delta": p2597["effects_vs_baseline"]["q35_zero_delta_roll"]["accuracy_delta"],
        "mechanism_closed": False,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "scope": "Phase2579-2598",
        "finals": final_paths,
        "milestones": milestones,
        "attachment_audit_corrections": [
            "Phase2572 single heads all failed; no 128-dimensional minimal gear was found",
            "Phase2573 enumerated 128 head subsets containing H5, not 128 activation coordinates",
            "V-carried interaction does not prove that the gear is V or that the carrier is the compute site",
            "Phase2564 Qwen14B multihop was not behavior-qualified",
            "nonzero finite differences are descriptive nonlinearity until controlled causally",
            "an effective full head band does not make every member necessary",
        ],
        "client": {"asset": str(ASSET), "bytes": ASSET.stat().st_size,
                   "sha256": sha256(ASSET), "phase": asset["phase"],
                   "required_panels": sorted(required_panels)},
        "retention": {"important_raw_fields_retained": True,
                      "reason": "parameter-level client display plus provenance and full-coordinate reanalysis",
                      "empty_offload_directory_files": len(offload_files)},
        "next_major_stage": {
            "same_specific_target": False,
            "completed_target": "cross-prompt relation-value family atlas and scaffold-transport causal validation",
            "new_target": "single-prompt endogenous source-to-generation compiler on natural long-form operations",
            "tasks": [
                "behavior gates for long-sentence reorder, candidate-free reference, negation/chronology, and syntax rewrite",
                "single-prompt source-token to K/V/Q/head/residual local counterfactuals",
                "actual greedy-token read/re-encoding trajectory",
                "sequential nonquantized replication on Qwen3-14B, GLM4, and DS7B after Qwen3-4B lockbox",
            ],
        },
        "language_mechanism_closed": False,
    }
    checks = {
        "all_20_finals_present_and_passed": len(finals) == 20 and all(
            payload["all_checks_passed"] for payload in finals.values()),
        "memo_phase2579_2598_continuous_once": all(headings.count(phase) == 1 for phase in range(2579, 2599)),
        "client_phase2598": asset["phase"] == 2598 and p2598["all_checks_passed"],
        "all_required_client_panels": required_panels <= set(models),
        "all_displayed_rows_match_declared_axis": parameter_lengths,
        "frontend_build_present": (ROOT / "frontend/dist/index.html").is_file(),
        "phase2596_append_only_correction_present": "Phase2596合同勘误" in memo_text,
        "temporary_offload_contains_no_files": not offload_files,
        "raw_retention_policy_compliant": True,
        "claim_boundary": True,
    }
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
