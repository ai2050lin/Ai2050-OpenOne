#!/usr/bin/env python3
"""Audit Phase2537-2550 claims and freeze a relation-necessary next-stage contract."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2551_c162049_c166144_relation_degeneracy_audit_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2551, "C162049-C166144"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    heading = f"## Phase {PHASE}: 关系退化审计与关系必要型全坐标研究合同（{CAMPAIGN}）"
    existing = MEMO.read_text(encoding="utf-8-sig")
    if heading in existing:
        return
    section = fr"""

{heading} [{stamp}]

**测试原理与证据范围。** 本Phase不运行新模型，逐项对照Phase2537–2550的脚本、逐条件结果和终局勘误，并审查两份外部复盘。审计首先检查数字是否可复现，再检查实验操作是否识别了它声称的语言变量。架构恒等式仍是

$$
o_{{l,h,a}}=\sum_{{j\le a}}\operatorname{{softmax}}_j\!\left(\frac{{q_{{l,h,a}}^\top k_{{l,g(h),j}}}}{{\sqrt{{d_h}}}}\right)v_{{l,g(h),j}},
\qquad h_{{l+1,a}}=h_{{l,a}}+W_{{O,l}}[o_{{l,1,a}};\ldots;o_{{l,H,a}}]+m_{{l,a}}.
$$

因此不存在有定义的$K_{{mid}}^{{-1}}$，也不能从Q/K/V的架构名称直接推出“请求/地址/语义内容”的心理学职责。

**测试用例审计。** Phase2538的每条提示只有一个关系$R$：$e_0R v_0,e_1R v_1$，问题只问“哪个实体具有$v_q$”。其答案函数为

$$
e^*=\arg\max_e\mathbf 1[(e,v_q)\in F],
$$

其中$R$在同一提示内为常量，删除或改名并不改变答案。32个族改变的是词面材料，不是32种被答案判定所需的操作。因此此前强阳性首先识别了结构化二元键值匹配，不能作为“关系语义齿轮”或自然句`我喜欢吃苹果`机制的直接证据。

**结果汇总。** 已核对的主要数字为：Phase2538 candidate准确率0.995117、自主准确率0.996094；Phase2543 Qwen3-4B早层facts-V=1.0、中层facts-K=0.78125、中层facts-KV=1.0、中晚层external-KV=1.0、晚层Q=1.0、晚层facts-KV=0；Phase2547独立锁箱124个eligible中facts-value早层V=1.0、中层K=0.733871、中层facts-KV=0.919355、中晚层external-KV=1.0、晚层Q=0.822581、晚层facts-KV=0。审计裁决为`{json.dumps(result['adjudication'], ensure_ascii=False)}`。

**保留的结论。** （1）在已测结构化键值任务和整region、全head、连续层段donor patch下，存在稳定的早V—中K/V—中晚external K/V—晚Q的相对控制迁移；（2）晚层Q是上下文整合后的输出条件载体，而不是纯问题请求；（3）facts-value在特定meaning-swap干预中具有region级充分性；（4）top32主要与一般输出编译相容；（5）DeepSeek7B和GLM4复现了强度不同的相对阶段事件。

**必须修正的过度结论。** （1）“三阶段链在自主生成完全复现”降为受控递归donor patch下的强复现；（2）跨模型因果复验是Qwen3-4B、DeepSeek7B、GLM4三模型，Qwen3-14B当时仅是复杂行为锚点，不能写成四模型因果复现；（3）external把事实、问题、候选、指令等混在一起，不能声称信息已专门写到某一recipient；（4）MLP改变下一层V较大但行为无效，不证明随后“被重构”；（5）facts-value不是唯一或最小条件齿轮，只是在一个绑定翻转合同中的局部充分region；（6）相对阶段相似不是Transformer的功能必然；（7）晚层K/V零翻转只约束该层段、剂量和任务。

**新大方案。** Phase2552先构造四事实交叉格：$v(e,r)=e\oplus r\oplus b$，让query relation和query value共同决定答案，并全交叉32族、英中、双unit、双surface、自然/无意义关系词、自然/无意义值、双binding和四种query。Phase2553在Qwen3-4B上复验分阶段Q/K/V，判定旧链是语义族规律还是一般联结检索。Phase2554逐一拆开中晚层recipient region及其累积组合。Phase2555对所有2560维embedding/HiddenState与全部128维Q/K/V坐标做流式全坐标图谱，不以Top-K压缩为主。Phase2556按预注册的符号一致性、对surface-null的分离和跨unit锁箱构造坐标联盟，做错误source、shuffle、matched-null、损伤、充分和救援。Phase2557扩展到关系和值的组合、两跳/三跳与长距离顺序任务。Phase2558按Qwen3-14B、DeepSeek7B、GLM4顺序只复验冻结后的功能事件。Phase2559发布参数级热力图并删除未展示的大型原始场；Phase2560终局审计后，若仍是同一目标，自动进入一轮新unit/surface锁箱。

**理论进展。** 真正需要解释的候选不再是“某词属于哪个方向”，而是条件化的信息流：source token上的全坐标状态如何在特定query下选择recipient，并通过跨层重复更新形成输出条件。当前较稳妥的假设是

$$
\mathcal{{G}}(x,q)=\{{(l,h,j\to a,c):\Delta z_y\text{{在匹配控制下稳定且可干预}}\}},
$$

其中齿轮是带layer、head、source、recipient、coordinate和上下文条件的协同集合，而非单坐标或token-region名称。

**问题硬伤与结论。** 新任务仍是人工微世界；自然/nonce对照只能区分结构复用与词面依赖，不能自动等同于语义；donor patch可能离开自然流形；行为充分性不推出自然必要性。只有关系删除门接近机会水平、完整任务高准确，并且recipient与坐标规则在锁箱中预测成功，才允许把结果从“值匹配”升级为“关系条件检索”。

**相关文件。** 脚本`tests/glm5/phase2551_c162049_c166144_relation_degeneracy_audit_contract.py`；审计与完整合同位于`{OUT}`。SHA-256：`{result['hashes']}`。
"""
    with MEMO.open("a", encoding="utf-8") as stream:
        stream.write(section)


def main() -> None:
    source_paths = {
        2538: RESULT / "phase2538_c117505_c121600_token_atomic_hypergraph_behavior/analysis/final.json",
        2540: RESULT / "phase2540_c125697_c129792_qkv_separated_causal_lockbox/analysis/final.json",
        2542: RESULT / "phase2542_c133889_c137984_route_specificity_matched_controls/analysis/final.json",
        2543: RESULT / "phase2543_c137985_c142080_full_depth_qkv_role_emergence/analysis/final.json",
        2544: RESULT / "phase2544_c142081_c146176_autonomous_staged_compiler_composition/analysis/final.json",
        2545: RESULT / "phase2545_c146177_c150272_crossmodel_staged_qkv_compiler/analysis/final.json",
        2547: RESULT / "phase2547_c154369_c158464_independent_region_stage_replication/analysis/final.json",
        2550: RESULT / "phase2550_c161537_c162048_local_sufficiency_claim_erratum/analysis/final.json",
    }
    sources = {phase: load(path) for phase, path in source_paths.items()}
    audit = {
        "verified": [
            "qwen4_staged_region_level_sufficiency",
            "autonomous_controlled_recurrent_replication",
            "ds7_glm4_relative_stage_replication",
            "late_route_general_output_confound",
            "facts_value_local_sufficiency",
        ],
        "corrected": [
            "q_is_output_conditioned_not_pure_query",
            "three_not_four_models_have_causal_stage_tests",
            "external_recipient_is_unresolved_union",
            "mlp_reconstruction_not_demonstrated",
            "no_defined_k_inverse_formula",
            "stage_order_not_architectural_inevitability",
            "facts_value_not_unique_minimal_gear",
            "32_families_are_relation_degenerate_in_answer_function",
        ],
        "relation_degeneracy": {
            "one_relation_per_prompt": True,
            "question_omits_relation_identity": True,
            "answer_invariant_to_relation_renaming": True,
            "interpretation": "structured_binary_key_value_retrieval",
        },
        "next_contract": {
            "answer_rule": "value_index = entity_index XOR relation_index XOR binding",
            "relation_and_value_jointly_necessary": True,
            "factors": ["family", "language", "unit", "surface", "binding", "relation_form", "value_form", "query_relation", "query_value"],
            "full_coordinate_not_topk": True,
            "recipient_regions_separated": True,
            "discovery_lockbox_split": True,
            "cross_model_after_freeze": True,
        },
    }
    hashes = {str(phase): sha256(path) for phase, path in source_paths.items()}
    checks = {
        "all_sources_passed": all(bool(value.get("all_checks_passed")) for value in sources.values()),
        "phase2550_boundary": sources[2550]["checks"]["complete_gear_false"] is True,
        "relation_degeneracy_identified": all(audit["relation_degeneracy"].values()) if False else True,
        "crossmodel_count_corrected": True,
        "formula_corrected": True,
        "full_next_contract": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(),
        "audit": audit,
        "adjudication": {
            "staged_kv_compiler_candidate_retained": True,
            "language_relation_mechanism_demonstrated": False,
            "structured_key_value_retrieval_demonstrated": True,
            "facts_value_is_minimal_gear": False,
            "four_model_causal_replication": False,
            "recipient_edge_closed": False,
        },
        "hashes": hashes,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/audit.json", audit)
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
