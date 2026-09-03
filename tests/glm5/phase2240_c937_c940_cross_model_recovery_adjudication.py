from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase2239_c925_c936_cross_model_exact_semantic_panel as cross  # noqa: E402


PHASE = 2240
CAMPAIGNS = tuple(f"C{i}" for i in range(937, 941))
OUT = ROOT / "tests/glm5/result/phase2240_c937_c940_cross_model_recovery_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_memo(result: dict) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    formula = r"""
$$
D(m,n;f)=\frac{1}{21|R|}\sum_{t,r}
\left|\widetilde\rho_{m,f,r}(t)-\widetilde\rho_{n,f,r}(t)\right|.
$$
"""
    text = f"""

## Phase {PHASE}: Qwen4B/Qwen14B 完整场恢复与跨模型拓扑正式重裁 [{stamp}]

**重裁原因。** Phase 2239 的两个 Qwen worker 已完成 96 条行为和全场，却因 `numpy.bool_` 无法写 JSON 而被总账误记为 worker-error；因此 Phase 2239 的 `qualified_models=[]` 和空拓扑比较作废。本期 C937-C940 只从原始行为 JSONL、全场 NPY 和索引恢复账本，没有重新加载模型、改材料或改门槛。

**恢复证据。** Qwen3-4B 候选/生成准确率为 `{result['model_results']['qwen3']['candidate_accuracy']}` / `{result['model_results']['qwen3']['generation_accuracy']}`，全场形状 `{result['model_results']['qwen3']['field']['shape']}`；Qwen3-14B 为 `{result['model_results']['qwen3_14b']['candidate_accuracy']}` / `{result['model_results']['qwen3_14b']['generation_accuracy']}`，形状 `{result['model_results']['qwen3_14b']['field']['shape']}`。GLM4 和 DS7B 的行为未合格结论保持不变，内部场仍为 NA。

**测试原理。** 两个 Qwen 均在相同 96 条语义行内计算真假全坐标响应；每层六角色能量归一化后按相对层深插值到 21 点。物理坐标不跨模型对齐。
{formula}
**结果。** 12 个语言族均获得 Qwen4B/Qwen14B 拓扑比较。总体平均绝对角色份额距离为 `{result['comparison_summary']['qwen3|qwen3_14b']['mean_absolute_role_share_distance']}`，最大逐点距离为 `{result['comparison_summary']['qwen3|qwen3_14b']['max_absolute_role_share_distance']}`。逐族结果保存于 `topology_comparisons.jsonl`。

**严格解释、问题与硬伤。** 距离较小只表示这套受控任务上的相对层深角色能量组织接近，不证明同一坐标、同一算法、同一因果电路或新的数学同构。每族只有一个词汇单元；角色 RMS 是跨模型摘要，会抹去模型内坐标符号与低值细节；14B 采用磁盘卸载；总体行为过门不代表每个族逐项过门。

**结论和授权。** 合法跨模型结论限定为 Qwen4B/Qwen14B 的功能拓扑比较；GLM4/DS7B 仅有行为边界。下一步将完整物理场和逐坐标语言族响应导出至可视化客户端，哈希核验后只清理已被客户端二进制覆盖的本轮原始大场。理论名称和 RDC 原则不变，新数学闭合仍未获授权。

**相关文件。** 脚本 `tests/glm5/phase2240_c937_c940_cross_model_recovery_adjudication.py`；结果 `{OUT.relative_to(ROOT)}`；恢复源 `tests/glm5/result/phase2239_c925_c936_cross_model_exact_semantic_panel`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    model_results = {
        model: json.loads((cross.OUT / model / "analysis/final.json").read_text(encoding="utf-8"))
        for model in cross.MODELS
    }
    qualified = [model for model, result in model_results.items()
                 if result.get("all_checks_passed") and result.get("behavior_qualified")]
    topologies = {model: cross.response_topology(model, model_results[model]) for model in qualified}
    comparisons = cross.compare_topologies(topologies)
    cross.write_rows(OUT / "analysis/topology_comparisons.jsonl", comparisons)
    summary = {}
    for pair in sorted({(row["left"], row["right"]) for row in comparisons}):
        subset = [row for row in comparisons if (row["left"], row["right"]) == pair]
        summary[f"{pair[0]}|{pair[1]}"] = {
            "families": len(subset),
            "mean_absolute_role_share_distance": float(np.mean([row["mean_absolute_role_share_distance"] for row in subset])),
            "max_absolute_role_share_distance": float(np.max([row["max_absolute_role_share_distance"] for row in subset])),
        }
    checks = {
        "qwen4_artifact_recovered": model_results["qwen3"].get("loader") == "artifact_resume",
        "qwen14_artifact_recovered": model_results["qwen3_14b"].get("loader") == "artifact_resume",
        "qualified_models_exact": qualified == ["qwen3", "qwen3_14b"],
        "twelve_family_comparisons": len(comparisons) == 12,
        "finite": all(np.isfinite(row["mean_absolute_role_share_distance"]) for row in comparisons),
    }
    result = {
        "phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed_corrected",
        "timestamp": datetime.now().astimezone().isoformat(), "checks": checks,
        "all_checks_passed": all(checks.values()),
        "supersedes": "Phase2239 qualified_models/topology only; GLM4 and DeepSeek behavior results remain valid",
        "model_results": model_results, "qualified_models": qualified,
        "comparison_summary": summary, "topology_comparisons": comparisons,
        "strict_conclusion": "Qwen4B and Qwen14B admit exact-semantic relative-depth role-topology comparison; no coordinate identity or causal isomorphism follows.",
        "next_authorization": "Export visual full-coordinate atlases and clean only verified duplicated raw fields.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
