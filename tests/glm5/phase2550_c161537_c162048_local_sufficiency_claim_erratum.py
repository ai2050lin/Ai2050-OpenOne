#!/usr/bin/env python3
"""Correct a machine-readable local-sufficiency label without rewriting MEMO history."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
P2547 = RESULT / "phase2547_c154369_c158464_independent_region_stage_replication"
P2549 = RESULT / "phase2549_c160513_c161536_terminal_evidence_audit_next_contract"
OUT = RESULT / "phase2550_c161537_c162048_local_sufficiency_claim_erratum"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c42641_output_conditioned_crossmodel_field.json"
PHASE, CAMPAIGN = 2550, "C161537-C162048"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: facts-value局部充分性字段的证据等级勘误（{CAMPAIGN}） [{stamp}]

**审查原理与对象。** 终审发现Phase2547机器可读裁决曾使用`single_region_is_complete_gear=true`。原始数字——124个行为合格case中早期facts-value V donor flip为1.0——没有错误；错误在于把“指定连续层段、全head、整region替换下的局部充分性”命名成“完整齿轮”。本Phase修正脚本和final字段，并以追加方式明确覆盖旧标签，不改写历史MEMO。

$$\operatorname{{Suff}}(R\mid I,T,M)=1\;\not\Rightarrow\;\operatorname{{MinimalGear}}(R)=1,$$

其中$I$为具体干预、$T$为结构化任务、$M$为Qwen3-4B；充分性不推出最小性、必要性、特异性或普遍性。

**结果汇总。** `{json.dumps(result['correction'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 修正`tests/glm5/phase2547_c154369_c158464_independent_region_stage_replication.py`及其`analysis/final.json`；本勘误脚本和final位于`{OUT}`。可视化原有claim boundary已经明确“不构成普遍语义原子”，无需重建190MB资产。

**分析、理论进展与硬伤。** 正确结论是：facts-value是当前干预合同中唯一单独达到完整donor翻转的token region，因而是下一阶段优先拆解对象；它仍包含词汇、语义、绑定与位置混杂，并覆盖多个token、全head和九层，绝不是已识别的最小条件坐标齿轮。

**结论。** Phase2547的所有实验数值保留；仅将证据等级从“完整齿轮”降为“该干预中的region级充分性”。Phase2549给出的下一独立目标合同保持不变。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    f47 = load(P2547 / "analysis/final.json")
    f49 = load(P2549 / "analysis/final.json")
    asset = load(ASSET)
    adjudication = f47["adjudication"]
    correction = {
        "unchanged_measurement": {"early_v_facts_value_donor_flip": f47["summary"]["conditions"]["early_v_facts_value"]["donor_flip"], "eligible_cases": f47["summary"]["eligible_cases"]},
        "correct_field": "facts_value_region_sufficient_in_this_intervention",
        "correct_value": adjudication["facts_value_region_sufficient_in_this_intervention"],
        "forbidden_inference": "single_region_is_complete_gear",
        "forbidden_inference_value": adjudication["single_region_is_complete_gear"],
        "supersedes": "the overbroad adjudication label embedded in the Phase2547 MEMO result dump; numerical measurements are unchanged",
    }
    script = ROOT / "tests/glm5/phase2547_c154369_c158464_independent_region_stage_replication.py"
    script_text = script.read_text(encoding="utf-8")
    checks = {
        "phase2547_passed": f47["all_checks_passed"], "phase2549_passed": f49["all_checks_passed"],
        "local_sufficiency_true": correction["correct_value"] is True,
        "complete_gear_false": correction["forbidden_inference_value"] is False,
        "script_reproducible_label": "facts_value_region_sufficient_in_this_intervention" in script_text,
        "visual_boundary_already_safe": "does not establish a universal semantic atom" in asset["claim_boundary"],
        "measurements_unchanged": correction["unchanged_measurement"] == {"early_v_facts_value_donor_flip": 1.0, "eligible_cases": 124},
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "correction": correction,
        "files": {"phase2547_final": str(P2547 / "analysis/final.json"), "sha256": sha(P2547 / "analysis/final.json")},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
