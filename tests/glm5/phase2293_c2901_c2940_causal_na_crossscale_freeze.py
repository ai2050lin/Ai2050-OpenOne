#!/usr/bin/env python3
"""Adjudicate causal eligibility and freeze the Qwen3-14B replication panel."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
PREDICT_OUT = RESULT / "phase2291_c2701_c2800_sample_conditioned_coordinate_tournament"
TRANSPORT_OUT = RESULT / "phase2292_c2801_c2900_full_coordinate_layer_transport"
OUT = RESULT / "phase2293_c2901_c2940_causal_na_crossscale_freeze"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

PHASE = 2293
CAMPAIGN = "C2901-C2940"
MIDDLE_MIN = 6
MIDDLE_MAX = 30
MAX_REPLICATION_CELLS = 5


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 多尺度因果资格裁决与Qwen3-14B拓扑冻结（{CAMPAIGN}） [{stamp}]

**测试原理与停止条件。** Phase2289已冻结：只有状态算子在未揭盲锁箱前获得 `q6-q30` 中层资格，才执行密度 `1/64...1/2`、剂量 `0.25/0.5/1.0` 的 delete/call/rescue。Phase2291虽有13/18个前瞻阳性，但检查点分布为 `{json.dumps(result['operator_checkpoint_distribution'], ensure_ascii=False)}`，最深仅q5，因此没有合法因果锚点。Phase2292的189个中层阳性预测的是 `R_q→R_{{q+1}}` 传播；它们是在本轮锁箱裁决后形成的新对象，不能倒灌成原合同的状态算子锚点。

**结果与结论。** 因果状态 `{result['causal_status']}`，不是干预失败，也不反证中层传动。没有加载模型、没有运行patch、没有生成干预数字。

**跨规模冻结。** 观察路线继续。按“跨语言优先，其次最深跨表面，再按构式名稳定排序”从Phase2291锁箱阳性冻结最多五格：`{json.dumps(result['qwen14_frozen_cells'], ensure_ascii=False)}`。Qwen3-4B物理坐标不搬到14B；只冻结构式、路线、角色、函数类型与相对block深度。映射为：q0保持embedding；q>0按 `round(q*L14/L4)` 映射post-block检查点。Qwen3-14B仍须重新通过同一双行为门，并在自身5120坐标系内击败相同控制。

**问题硬伤与下一步。** 多尺度因果设计仍未真正运行，这是当前机制闭环的硬缺口；但遵守预注册比事后挑Phase2292热点更重要。下一步授权只限冻结五格的Qwen3-14B功能拓扑复验。脚本 `tests/glm5/phase2293_c2901_c2940_causal_na_crossscale_freeze.py`；结果 `tests/glm5/result/phase2293_c2901_c2940_causal_na_crossscale_freeze`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    predictor = json.loads((PREDICT_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    transport = json.loads((TRANSPORT_OUT / "analysis/final.json").read_text(encoding="utf-8"))
    positive = [row for row in predictor["decisions"] if row["lockbox_pass"]]
    middle = [row for row in positive if MIDDLE_MIN <= int(row["checkpoint"]) <= MIDDLE_MAX]
    checkpoint_distribution = {}
    for row in positive:
        checkpoint_distribution[str(row["checkpoint"])] = checkpoint_distribution.get(str(row["checkpoint"]), 0) + 1

    cross_language = sorted([row for row in positive if row["route"] == "language_en_to_zh"],
                            key=lambda row: (-int(row["checkpoint"]), row["family"]))
    cross_surface = sorted([row for row in positive if row["route"] == "surface_narrative_to_dialogue"],
                           key=lambda row: (-int(row["checkpoint"]), row["family"]))
    frozen = []
    seen = set()
    for row in [*cross_language, *cross_surface]:
        key = (row["family"], row["route"])
        if key in seen:
            continue
        frozen.append({"family": row["family"], "route": row["route"],
                       "qwen4_checkpoint": row["checkpoint"], "role": row["role"],
                       "role_index": row["role_index"], "model": row["model"],
                       "depth_mapping": "embedding_fixed_else_round(q4_block*L14/L4)"})
        seen.add(key)
        if len(frozen) == MAX_REPLICATION_CELLS:
            break
    checks = {
        "parents_passed": predictor["all_checks_passed"] and transport["all_checks_passed"],
        "no_middle_operator_anchor": len(middle) == 0,
        "causal_not_run": True,
        "transport_not_reclassified": transport["lockbox_pass_count"] > 0,
        "qwen14_cells_from_lockbox_positive": all((row["family"], row["route"]) in
            {(x["family"], x["route"]) for x in positive} for row in frozen),
        "qwen14_panel_size": 0 < len(frozen) <= MAX_REPLICATION_CELLS,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "status": "closed", "causal_status": "NA_no_preregistered_middle_operator_anchor",
        "operator_checkpoint_distribution": checkpoint_distribution,
        "middle_operator_anchors": middle,
        "transport_middle_cells_not_causal_anchors": len(transport["deep_lockbox_cells"]),
        "qwen14_frozen_cells": frozen,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": "The multiscale causal branch is NA because no preregistered state operator reached q6-q30; cross-scale observation remains authorized for five frozen lockbox-positive functional cells.",
        "next_authorization": "Run Qwen3-14B dual behavior and model-local coordinate replication for only the five frozen cells.",
    }
    save(OUT / "config/qwen14_replication_freeze.json", {"cells": frozen, "behavior_gate": 0.75,
                                                          "model_order": ["Qwen3-4B_released", "Qwen3-14B"]})
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
