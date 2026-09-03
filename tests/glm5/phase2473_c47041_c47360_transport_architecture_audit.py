#!/usr/bin/env python3
"""Audit and correct the final-RMSNorm transport confound in Phase2472."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2472 = next(RESULT.glob("phase2472_*"))
OUT = RESULT / "phase2473_c47041_c47360_transport_architecture_audit"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2473, "C47041-C47360"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: q36→q37 final RMSNorm伪传动剔除与block→block锁箱更正（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 审查Phase2472初次自动选择，发现最大逐坐标增益落在q36→q37；按Phase2470 qpoint语义，q37是final norm，RMSNorm本来就是由样本RMS与已知逐坐标权重实现的架构变换，不能作为新语言传动。保持unit9英文拟合、unit9中文选择、unit10锁箱不变，预先排除q0→q1 embedding边界和q36→q37 final norm，只允许q1→q2至q35→q36的block间转移竞争。

$$q37=\operatorname{{RMSNorm}}(q36),\qquad \text{{known architecture}}\neq\text{{discovered language transport}}.$$

**结果汇总。** 更正后选择 `{json.dumps(result['corrected']['selected_transition'], ensure_ascii=False)}`；确认 `{json.dumps(result['corrected']['selected_confirmation'], ensure_ascii=False)}`；unit10锁箱 `{json.dumps(result['corrected']['lockbox_unit10'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2473_c47041_c47360_transport_architecture_audit.py`；更正后的Phase2472脚本/final与本Phase final。历史MEMO保持append-only，本条明确取代Phase2472中把q36→q37当候选的任何解读。

**分析与理论进展。** 合格block转移改为q10→q11。unit10中pooled逐坐标尺度$R^2$、接口条件尺度及family身份均可直接与identity/global比较；这支持“可冻结预测的坐标尺度变化”候选，但仍可能是通用层数值动力学。只有接口条件模型相对pooled的小幅增益，不能单独命名为语义条件守卫。

**问题硬伤与结论。** family条件尺度训练样本过少，虽锁箱分数更高也不能当作自然齿轮。对角模型忽略坐标混合，且family专属措辞仍可能贡献预测。当前只保留L2级基本预测候选；final norm高分被完全降级为架构正对照。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    corrected = json.loads((P2472 / "analysis/final.json").read_text(encoding="utf-8"))["transport"]
    selection = corrected["selected_transition"]
    lockbox = corrected["lockbox_unit10"]
    checks = {
        "final_norm_excluded": selection != [36, 37],
        "embedding_boundary_excluded": selection != [0, 1],
        "block_transition": 1 <= selection[0] < selection[1] <= 36,
        "five_baselines": len(lockbox) == 5,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "corrected": {"selected_transition": selection, "selected_confirmation": corrected["selected_confirmation"], "lockbox_unit10": lockbox},
        "adjudication": {
            "q36_q37_language_transport": False,
            "q36_q37_role": "known final RMSNorm architecture control",
            "block_diagonal_transport_candidate": lockbox["pooled_diagonal"]["r2_vs_zero"] > lockbox["global"]["r2_vs_zero"],
            "semantic_transport_mechanism_closed": False,
            "language_encoding_mechanism_closed": False,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
