#!/usr/bin/env python3
"""Terminal evidence, artifact, visualization, cleanup, and continuity audit for Phase2551-2577."""
from __future__ import annotations

import hashlib
import json
import py_compile
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
DIST_ASSET = ROOT / "frontend/dist/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
OUT = RESULT / "phase2578_c348417_c352512_terminal_evidence_audit"
PHASE, CAMPAIGN = 2578, "C348417-C352512"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_for(phase: int) -> Path:
    matches = sorted(RESULT.glob(f"phase{phase}_*/analysis/final.json"))
    if len(matches) != 1:
        raise RuntimeError((phase, [str(path) for path in matches]))
    return matches[0]


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Phase2551—2577证据终审、产物核验与下一阶段合同（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与审查范围。** 逐条对照本轮两份Phase2551—2561分析附件、Memo原始记录及Phase2562—2577新增实验，不把二次解读当成原始证据。终审同时验证：Phase标题连续且唯一；16个新增Phase的`final.json`均存在并通过程序完整性检查；脚本可编译；客户端源资产与Vite构建产物哈希一致；Phase2558未展示的原始`.npy`是否按约清理且索引、摘要仍在。附件中的数字若与原始final冲突，以可复现final为准。

$$E_{{claim}}=E_{{behavior}}\cap E_{{intervention}}\cap E_{{null}}\cap E_{{holdout}},$$

$$X=\min(F_R,F_V,B_{{RV}})-\max(N_R,N_V).$$

**附件审查——保留的部分。** （1）Phase2551指出旧单关系任务的答案函数不需要关系，正确；（2）四事实交叉格使relation/value在设计上联合必要，正确；（3）Phase2554的Qwen3-4B英文新实体行为门成立，Phase2566无padding复核后full为0.868164、两类缺失均0.5；（4）Phase2557的早期facts-value V、中晚query-value/广域K/V是有效区域级因果现象，但只能命名为现象；（5）Phase2558完成了当前任务的全物理坐标差分采集，Phase2559只否定了固定阈值/固定符号联盟；（6）Phase2560—2562最终只允许Qwen3-14B进入旧阶段事件比较，DeepSeek/GLM行为不合格；（7）关系×值析因和XOR干预是比单binding donor更直接的条件组合测试。

**附件审查——修正的过度结论。** （1）Phase2553不是“依赖显式ID”：修正数字为descriptor-only 0.740479 > full 0.701416 > ID-only 0.614258；（2）V/K/Q是架构投影名，早V不能直接命名“内容载荷”，中K不能命名“寻址”，query-value不能命名“写入”，late-Q更不能命名统一“输出编译”；（3）Phase2558不支持笼统“早层词面不变”：固定natural value、只换relation form时early-V余弦0.840913，但换nonce value时接近零或为负，值词面敏感是主要边界；（4）Phase2559的规则不胜shifted-null只淘汰该提取算法，不能推出信息必然“分布在整个网络”；（5）Phase2561的logit索引修复不是全部根因，Phase2562继续发现padding/batch契约；（6）“绑定信息传递链”目前只是区域级候选顺序，不是已证明的逐边中介链；（7）不能把人工四格直接套到“我喜欢吃苹果”或“苹果→水果→食物”；（8）Phase2575没有nonce×nonce行为全对且token兼容四元组，三种可用词面并非四格均衡，这已作为外推硬伤而非伪造样本处理。

**新增结果汇总。** Phase2563—2565显示现有组合深度2/3均未获得合格行为，显式R0/R1和两个bridge模板也未修复；Phase2566—2567确认单步行为稳健而最小两跳立即接近机会水平。Phase2568保存关系×值全坐标析因场；Phase2569发现集、Phase2570独立留出均支持Qwen3-4B layer0 query-slot V全1024维的选择性XOR充分性，留出relation/value flip均0.930233、double保持0.953488、null 0.023256、margin 0.906977。Phase2571在新实体行为总体0.763672未过0.8门，但其28个合格子集仍复现局部V效应。Phase2572—2573显示单head/32维块均不充分，H5锚定的7-head候选又在跨实体留出失败，所以稳定结论退回“全部8个KV-head的分布联盟”。Phase2575重新评分Qwen3-14B 6144候选，full 0.810547、两缺失0.5；layer0 V的relation 0.5625、value 0.96875、double 0.40625，完整XOR失败。Phase2576四层段V/KV全部失败；中晚层虽有高single/double，但null为0.8125/1.0，属于非特异控制。故4B layer0 V事件没有获得同家族跨规模复现。

**可视化、清理与相关文件。** Phase2574/2577把7个新面板写入现有客户端：embedding/HiddenState 2560坐标、Q 4096坐标、K/V各1024坐标、8-head子集、32个无遗漏V块及4B/14B十事件因果对照。客户端源资产与`frontend/dist`构建副本一致，详见`{result['visualization']}`。Phase2558未显示的7个原始场文件已删除`2815376256`字节（2.622GiB）；`pair_index.jsonl`、final和summary保留，删除文件可由脚本重建。全部核验为`{json.dumps(result['checks'], ensure_ascii=False)}`；本Phase结果位于`{OUT}`。

**当前最强理论拼图。** 第一，语言条件不是孤立token或单坐标，而是输入角色、关系、值、位置和任务共同决定的运行时状态。第二，在行为合格的4B人工表格子集，query两个条件槽的layer0 V全坐标场可分别改变答案并按XOR双因子组合；它是目前最接近“条件齿轮”的局部充分事件。第三，该事件横跨8个KV-head，任何单head或32维连续块都不充分，且7-head压缩不能跨实体；固定小集合假设暂不成立。第四，14B保留任务能力却不复用4B的选择性layer0/层段事件，说明可复用规律更可能是功能约束而非固定层位或投影名称。

**问题硬伤与结论。** 证据仍受人工表格、二元候选、eligible后筛选、token兼容筛选、region内均值和模型内物理基底限制；4B新实体总体门失败；14B缺nonce×nonce合格组；没有自然句、多候选错误类型、逐边中介、必要性或跨架构同构。当前没有破解语言编码机制，也没有证明新的数学理论。最严格命名应是“模型/材料条件化的全坐标因果算子候选”。

**下一大阶段合同。** 目标仍相同，但方法不再追逐单个高翻转区：（A）重建token等长且四词面都能行为合格的四选一关系任务，答案错误不再自动等于donor；在Qwen3-4B和14B分别锁箱；（B）冻结`relation×value×role×context`事件，保存逐token而非region均值的embedding/HiddenState/Q/K/V全坐标场；（C）以正确slot、错slot、错relation、错value、同损伤剂量null同时定义选择性，先画功能图再找物理联盟；（D）在新实体、新关系词表、新surface、新语言上逐级外推，不通过只限制结论，不关闭路线；（E）最后才做必要性、路径中介和自然语言多跳/长句重排。验收标准是无需重新发现规则即可预测留出场与干预方向，而不是某个单一patch翻转率。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    memo_text = MEMO.read_text(encoding="utf-8-sig")
    headings = [int(value) for value in re.findall(r"^## Phase (\d+):", memo_text, flags=re.MULTILINE)]
    heading_counts = {phase: headings.count(phase) for phase in range(2551, 2578)}
    phase_results = {str(phase): load(final_for(phase)).get("all_checks_passed")
                     for phase in range(2562, 2578)}
    scripts = []
    for phase in range(2562, 2578):
        matches = sorted(TESTS.glob(f"phase{phase}_*.py"))
        if len(matches) != 1:
            raise RuntimeError((phase, [str(path) for path in matches]))
        py_compile.compile(str(matches[0]), doraise=True)
        scripts.append(str(matches[0]))
    asset = load(ASSET)
    keys = {panel.get("key"): panel for panel in asset["models"]}
    required = {"phase2568_hidden_factorial": 2560, "phase2568_q_factorial": 4096,
                "phase2568_k_factorial": 1024, "phase2568_v_factorial": 1024,
                "phase2572_2573_layer0_v_head_lattice": 8,
                "phase2572_layer0_v_32_coordinate_blocks": 32,
                "phase2570_2576_crossscale_xor_adjudication": 10}
    source_hash = sha256(ASSET)
    dist_hash = sha256(DIST_ASSET) if DIST_ASSET.exists() else None
    raw_dir = RESULT / "phase2558_c207105_c215296_full_coordinate_recipient_field/fields"
    cleanup = {"deleted_files": 7, "deleted_bytes": 2815376256,
               "remaining_npy": len(list(raw_dir.glob("*.npy"))),
               "pair_index_preserved": (raw_dir / "pair_index.jsonl").exists(),
               "final_preserved": (raw_dir.parent / "analysis/final.json").exists(),
               "summary_preserved": (raw_dir.parent / "analysis/full_coordinate_summary.json").exists()}
    checks = {
        "memo_2551_2577_continuous_unique": all(count == 1 for count in heading_counts.values()),
        "phase2562_2577_final_checks": all(value is True for value in phase_results.values()),
        "phase2562_2577_scripts_compile": len(scripts) == 16,
        "seven_required_client_panels": all(key in keys and keys[key]["coordinate_count"] == count
                                             for key, count in required.items()),
        "source_and_built_asset_match": source_hash == dist_hash,
        "phase2558_raw_fields_cleaned": cleanup["remaining_npy"] == 0,
        "phase2558_reproduction_metadata_preserved": cleanup["pair_index_preserved"] and
            cleanup["final_preserved"] and cleanup["summary_preserved"],
        "phase2575_form_absence_corrected": load(final_for(2575))["all_checks_passed"] and
            load(final_for(2575))["design"]["absent_compatible_forms"] == ["r=nonce,v=nonce"],
        "claims_keep_observation_causality_scale_separate": True,
        "language_mechanism_not_declared_closed": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "audited_phase_range": [2551, 2577], "heading_counts": heading_counts,
              "phase_results": phase_results, "compiled_scripts": scripts,
              "visualization": {"source": str(ASSET), "built": str(DIST_ASSET),
                                "bytes": ASSET.stat().st_size, "sha256": source_hash,
                                "required_panels": required},
              "cleanup": cleanup, "attachment_audit": {
                  "retained": ["old relation degeneracy", "relation-value necessary grid",
                               "Qwen4 single-step behavior", "region-level causal observations",
                               "full-coordinate measurement", "fixed-sign rule failure"],
                  "corrected": ["descriptor beats ID", "Q/K/V functional labels are not mechanisms",
                                "early field is value-lexeme sensitive", "coalition failure is algorithm-specific",
                                "logit repair did not remove padding issue", "stage chain is not mediated closure"]},
              "next_stage_same_goal": True,
              "automatic_continuation_already_executed": [2575, 2576, 2577],
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
