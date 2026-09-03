#!/usr/bin/env python3
"""C869 append-only reconciliation of the Phase 2230 cross-model panel."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = RESULT / "phase2233_c869_cross_model_panel_reconciliation"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def find(pattern: str) -> Path:
    values = list(RESULT.glob(pattern))
    if len(values) != 1:
        raise RuntimeError((pattern, values))
    return values[0]


def append_memo(result: dict) -> None:
    marker = "## Phase 2233:"
    existing = MEMO.read_text(encoding="utf-8-sig")
    if marker in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase 2233: 跨模型面板分母与可比性追加重裁 [${{STAMP}}]

**审计原理。** 本期对应 `C869`，只读取 Phase 2228 的 Qwen3-4B 候选行为逐题结果、Phase 2230 的冻结 48 题面板及各 worker 正式结果。它不重跑模型，不改写旧文件，不改变 0.75 行为门。

$$
\operatorname{{Acc}}_m=\frac{{1}}{{48}}\sum_{{i=1}}^{{48}}\mathbf{{1}}[\hat y_{{m,i}}=y_i].
$$

**结果汇总。**
```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**严格结论。** Qwen3-4B 与 Qwen3-14B 在同一 48 题候选面板上都为 `40/48=0.8333`，行为上均获 HiddenState 资格；DeepSeek-7B 为 `24/48=0.5000`，未获资格；GLM4 在中文角色跨度编译时失败，属于接口 `NA`。但是 Qwen3-4B 曲线来自其行为合格语言族切片中实际保留的 32 行，而 Qwen3-14B worker 曲线来自全部 48 行，所以现有曲线只能做部分描述性比较，不能称为严格跨模型机制复现、坐标同构或功能拓扑确认。

**问题、硬伤与下一步。** 跨模型比较必须预先冻结共同的逐行行为资格和完全相同的 HiddenState 分母；模型 tokenizer 的角色跨度失败必须保持 NA。当前精确重裁已完成，不授权依据这些曲线升级理论或开展坐标对齐。相关正式文件为 `{(OUT / 'analysis/final.json').relative_to(ROOT)}`。
""".replace("${STAMP}", stamp)
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run():
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    phase2228 = find("phase2228_*")
    phase2230 = find("phase2230_*")
    panel = rows(phase2230 / "material/cross_model_48_case_panel.jsonl")
    qwen4_behavior = rows(phase2228 / "behavior/parent_candidate.jsonl")
    qwen4_panel = [row for row in qwen4_behavior if re.search(r"-u1[23]-c[01]$", row["case_id"])]
    qwen4_correct = sum(bool(row["correct"]) for row in qwen4_panel)
    p2230 = load(phase2230 / "analysis/final.json")
    q14_behavior = rows(phase2230 / "raw/qwen3_14b/behavior.jsonl")
    ds_behavior = rows(phase2230 / "raw/deepseek7b/behavior.jsonl")
    qwen4_curve_rows = sum(
        row["unit"] in (12, 13) and row["cell_i"] in (0, 1)
        for row in rows(phase2228 / "raw/parent/hidden_index.jsonl")
    )
    ledger = {
        "qwen3-4b": {"correct": qwen4_correct, "rows": len(qwen4_panel), "accuracy": qwen4_correct / len(qwen4_panel), "qualified": qwen4_correct / len(qwen4_panel) >= 0.75, "curve_rows": qwen4_curve_rows},
        "qwen3-14b": {"correct": sum(bool(r["correct"]) for r in q14_behavior), "rows": len(q14_behavior), "accuracy": sum(bool(r["correct"]) for r in q14_behavior) / len(q14_behavior), "qualified": p2230["workers"]["qwen3_14b"].get("qualified"), "curve_rows": len(q14_behavior)},
        "deepseek7b": {"correct": sum(bool(r["correct"]) for r in ds_behavior), "rows": len(ds_behavior), "accuracy": sum(bool(r["correct"]) for r in ds_behavior) / len(ds_behavior), "qualified": p2230["workers"]["deepseek7b"].get("qualified"), "curve_rows": 0},
        "glm4": {"rows": 0, "accuracy": None, "qualified": None, "curve_rows": 0, "status": "NA_role_span_compile_failure"},
    }
    checks = {
        "panel_48": len(panel) == 48,
        "qwen4_denominator_48": len(qwen4_panel) == 48,
        "qwen14_denominator_48": len(q14_behavior) == 48,
        "deepseek_denominator_48": len(ds_behavior) == 48,
        "qwen4_exact_40": qwen4_correct == 40,
        "qwen14_exact_40": ledger["qwen3-14b"]["correct"] == 40,
        "deepseek_exact_24": ledger["deepseek7b"]["correct"] == 24,
        "curve_denominator_mismatch_recorded": qwen4_curve_rows == 32 and len(q14_behavior) == 48,
    }
    result = {
        "phase": 2233, "campaign": "C869", "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "all_checks_passed": all(checks.values()),
        "behavior_ledger": ledger,
        "strict_conclusion": "Both Qwen models pass the same behavior panel, but their retained curve denominators differ (32 versus 48), so only partial descriptive comparison is legal.",
        "scientific_results_changed": False,
        "interpretation_narrowed": True,
        "next_authorization": "A future cross-model curve must freeze an identical per-row HiddenState denominator after tokenizer-specific role compilation; this exact reconciliation is complete.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
