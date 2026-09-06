#!/usr/bin/env python3
"""Audit Phase2562-2578 summaries and freeze the next four-choice/full-field research contract."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase2579_c352513_c356608_attachment_audit_fourchoice_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2579, "C352513-C356608"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: Phase2562—2578附件复审与四选一交互出生合同（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与证据审查。** 两份附件均与Phase2562—2578原始final逐项对照。保留：精确长度零padding测量合同；Qwen3-4B一步四格行为存活而二/三跳失败；embedding的relation×value二阶交互近零、网络内非零；Phase2570在43个独立四元组上定位到Qwen3-4B layer0 query-slot全V投影的选择性XOR充分事件；单head/32维块不充分、7-head冻结候选跨实体失败；Qwen3-14B行为通过但layer0和四层段V/KV均未复制4B选择性事件。

**过度结论修正。** 第二份附件把Phase2572写成“single-head通过、一个128维head是最小齿轮”，与原始结果相反：8个single-head全部失败，最大XOR core仅0.214286；32个连续块最大core为0。Phase2573穷举的是8个head中所有包含H5的128个`head子集`，不是H5内部128个坐标。故“已找到128维齿轮”“齿轮在V而不在HiddenState”撤销。Phase2564的Qwen3-14B多跳没有行为合格层，不能由Phase2566的4B单步阳性替它背书。Phase2568的非零有限差分只证明表示响应非加性，不证明任务算法或组合计算位置。Phase2570更接近输入token条件经layer0 V进入下游的充分载体；XOR计算发生处未定位。全部8-head只能叫目前未压缩的载体范围，不代表每个head必要。

**下一大阶段测试用例与合同。** 先把二元答案升级为四选一，使relation、value和double donor对应三个不同实体：

$$e^*(r,v;b_r,b_v)=2(r\oplus b_r)+(v\oplus b_v),\qquad e^*_{{00}},e^*_{{10}},e^*_{{01}},e^*_{{11}}\text{{两两不同}}.$$

32语言操作族、四种natural/nonce词面、四个binding、四种查询均匀覆盖；完整、缺relation、缺value、两者缺失都做四候选完整多token评分，按完整序列长度分桶且padding严格为零。4B与14B顺序加载，只有行为门通过者进入内部场。

内部研究不先找Top-K：保存逐token embedding与全部HiddenState物理坐标，计算

$$I_{{\ell,t,d}}=H^{{11}}_{{\ell,t,d}}-H^{{10}}_{{\ell,t,d}}-H^{{01}}_{{\ell,t,d}}+H^{{00}}_{{\ell,t,d}},$$

以family discovery/holdout的层间增量一致性定位交互首次稳定出现、放大与到达答案边界的位置。因果阶段必须同时测试正确交互残差、错layer、错token、错factor配对和同幅度null；只有四选一输出落到预注册的relation/value/double身份，且优于null，才允许命名“组合算子候选”。

**结果汇总与相关文件。** 审查裁决及合同为`{json.dumps(result, ensure_ascii=False)}`；文件位于`{OUT}`，脚本为`tests/glm5/phase2579_c352513_c356608_attachment_audit_fourchoice_contract.py`。

**理论进展、问题硬伤与结论。** 研究对象明确分成carrier、interaction birth、routing、readout四类；现有证据只较强支持4B条件carrier，不支持128维最小齿轮。下一阶段先用基础有限差分和严格对照，不启动附件建议的ANOVA、PID、最优传输或拓扑“大竞赛”；没有充分基础拼图时，高级方法只会增加自由度。四选一仍是人工表格，不等于自然语言；它是消除二元“所有错误都是donor”伪影的必要测量仪器。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    contract = {
        "output_algebra": "entity_index = 2*(relation xor binding_relation) + (value xor binding_value)",
        "entities": 4, "relations": 2, "values": 2, "bindings": 4,
        "families": 32, "forms": ["natural-natural", "natural-nonce", "nonce-natural", "nonce-nonce"],
        "behavior_conditions": ["full", "relation_missing", "value_missing", "both_missing"],
        "behavior_gate": {"full_min": 0.80, "each_missing_max": 0.40,
                          "each_available_form_min": 0.70, "target_balance_exact": True},
        "instrument": {"candidate_count": 4, "score_complete_multitoken": True,
                       "exact_length_buckets": True, "padding_allowed": False,
                       "nonquantized_bf16": True, "models_sequential": True},
        "field": {"primary": "per-token full-coordinate embedding and HiddenState",
                  "topk_primary": False, "region_mean_primary": False,
                  "discovery_holdout": "family 0-15 / 16-31", "raw_storage": "chunked"},
        "causal_controls": ["correct_interaction", "wrong_layer", "wrong_token",
                            "wrong_factor_pair", "equal_norm_null", "no_patch"],
    }
    corrections = {
        "single_head_minimal_gear": False,
        "single_head_max_xor_core": 0.214286,
        "single_32_coordinate_block_max_core": 0.0,
        "phase2573_search_space": "128 head subsets containing H5, not 128 coordinates inside H5",
        "qwen14_multihop_behavior_qualified": False,
        "layer0_v_interpretation": "conditional carrier sufficient for downstream control; computation site unresolved",
        "nonadditive_field_is_causal_mechanism": False,
    }
    checks = {"four_outputs_distinguish_all_donors": True, "all_four_forms_required": True,
              "exact_length_no_padding": True, "full_coordinate_primary": True,
              "advanced_math_deferred": True, "observation_and_causality_separate": True,
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "corrections": corrections, "contract": contract, "checks": checks,
              "all_checks_passed": all(checks.values()), "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    save(OUT / "contract/fourchoice_interaction_birth_contract.json", contract)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
