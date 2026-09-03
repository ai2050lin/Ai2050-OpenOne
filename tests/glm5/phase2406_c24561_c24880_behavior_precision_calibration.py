#!/usr/bin/env python3
"""Four-model behavior/token calibration and Qwen14B BF16 auto-placement feasibility ledger."""
from __future__ import annotations

import gc
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import torch

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2405 = RESULT / "phase2405_c24241_c24560_deconfounded_operation_contract"
OUT = RESULT / "phase2406_c24561_c24880_behavior_precision_calibration"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2406
CAMPAIGN = "C24561-C24880"
MODEL_ORDER = ("qwen4b", "qwen14b", "glm4", "deepseek7b")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2397_c21681_c22000_operation_behavior_token_calibration as old  # noqa: E402

old.OUT = OUT


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def bf16_ledger() -> dict:
    return {
        "requested": "Qwen3-14B BF16 with device_map=auto and partial CPU/disk placement",
        "probe_script": "tests/glm5_temp/qwen14b_bf16_auto_probe.py",
        "attempts": [
            {"max_memory": {"cuda:0": "14GiB", "cpu": "26GiB"}, "async_workers": 4, "outcome": "Windows access violation during safetensors materialization near weight 194/443"},
            {"max_memory": {"cuda:0": "14GiB", "cpu": "26GiB"}, "async_workers": 1, "outcome": "same access violation near weight 197/443"},
            {"max_memory": {"cuda:0": "14GiB", "cpu": "26GiB"}, "async_workers": 0, "outcome": "same access violation near weight 197/443"},
            {"max_memory": {"cuda:0": "13GiB", "cpu": "14GiB", "disk": "remainder"}, "async_workers": 0, "outcome": "same access violation near weight 175/443"},
        ],
        "adjudication": "BF16 auto placement is not viable on this Windows/Python3.14/torch host; failure is host-runtime weight materialization, not a model result or CUDA OOM.",
        "fallback": "Qwen3-14B NF4 storage with BF16 computation, model-local results labelled separately",
        "precision_comparison_allowed": False,
    }


def run_model(key: str, source: list[dict]) -> dict:
    final = OUT / key / "analysis/final.json"
    if final.exists():
        return json.loads(final.read_text(encoding="utf-8"))
    model, tokenizer, label = capability.load_model(key)
    try:
        compiled, calibration = old.compile_rows(tokenizer, source)
        index = OUT / key / "index/operation_rows.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compiled), encoding="utf-8")
        teacher_batch = {"qwen4b": 16, "qwen14b": 4, "glm4": 2, "deepseek7b": 4}[key]
        generation_batch = {"qwen4b": 8, "qwen14b": 3, "glm4": 2, "deepseek7b": 3}[key]
        teacher_rows, teacher = old.score_rows(key, model, compiled, teacher_batch)
        lock_names = {"fresh_unit_lockbox", "deep_fresh_unit_lockbox", "joint_template_unit_lockbox"}
        lockbox = [row for row in compiled if row["partition"] in lock_names]
        generation_rows, autonomous = old.generate_lockbox(key, model, tokenizer, lockbox, generation_batch)
    finally:
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    checks = {"compiled_rows": calibration["rows"] == 2176, "exact_anchor": calibration["raw_prompt_exact_rate"] >= 0.95,
              "monotonic_events": calibration["event_monotonic_rate"] == 1.0,
              "teacher_rows": len(teacher_rows) == 2176, "autonomous_rows": len(generation_rows) == 448,
              "finite_teacher": math.isfinite(teacher["mean"])}
    result = {"model": key, "model_label": label, "calibration": calibration, "teacher": teacher,
              "autonomous": autonomous, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result)
    return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 四模型整模板行为校准与Qwen14B BF16可行性裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Qwen3-4B BF16、Qwen3-14B、GLM4、DS7B按固定顺序单模型驻留CUDA；全部2176条任务计算完整答案相对foil的平均logprob和首分歧token margin，并对448条新unit/深新unit/模板+unit联合锁箱做关闭thinking、贪心原生chat生成。每模型重新将字符事件锚定到真实chat token。Qwen14B先独立尝试BF16 `device_map=auto`：逐步限制并行加载、GPU/CPU内存和磁盘offload；只有前向finite才允许标为BF16。

$$M_{{div}}=z_{{y_k}}-z_{{\tilde y_k}},\qquad k=\min\{{i:y_i\neq\tilde y_i\}},$$

$$\mathrm{{BF16\ viable}}\iff \mathrm{{load\ complete}}\land\mathrm{{forward\ finite}}.$$

**结果汇总。** 行为 `{json.dumps(result['summary'], ensure_ascii=False)}`；BF16台账 `{json.dumps(result['qwen14b_bf16'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2406_c24561_c24880_behavior_precision_calibration.py`；BF16探针`tests/glm5_temp/qwen14b_bf16_auto_probe.py`；四模型逐样本teacher、自主生成、token索引和final位于`tests/glm5/result/phase2406_c24561_c24880_behavior_precision_calibration`。

**分析与理论进展。** 行为门按整个新表面材料重新定义，而不是沿用旧模板能力。首分歧margin是下一token接口，自主exact/target-present是协议敏感的补充。BF16自动分层四次均在同一权重物化路径发生Windows访问冲突，且不是显存OOM；因此后续Qwen14B使用NF4存储/BF16计算，不能和Qwen4B BF16幅值横比。

**问题硬伤与结论。** teacher偏好不等于自主执行；自然化模板仍可能触发格式性失败。BF16失败是本地主机运行时限制，不能推断模型精度性质。后续主发现仍用Qwen4B BF16；四模型只复验无量纲、模型内冻结关系，所有绝对坐标和幅值均禁止跨模型直接比较。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    source = read_rows(P2405 / "material/all_rows.jsonl")
    models = {}
    for key in MODEL_ORDER:
        models[key] = run_model(key, source)
    summary = {key: {"teacher_target_over_foil": value["teacher"]["target_over_foil"],
                     "teacher_mean_margin": value["teacher"]["mean"],
                     "autonomous_exact": value["autonomous"]["exact"],
                     "autonomous_target_present": value["autonomous"]["target_present"],
                     "template_lockbox_teacher": value["teacher"]["by_partition"]["template_lockbox"]["target_over_foil"],
                     "anchor_exact_rate": value["calibration"]["raw_prompt_exact_rate"]} for key, value in models.items()}
    checks = {"sequential": list(models) == list(MODEL_ORDER), "models": all(value["all_checks_passed"] for value in models.values()),
              "bf16_adjudicated": not bf16_ledger()["precision_comparison_allowed"], "finite": all(math.isfinite(v["teacher"]["mean"]) for v in models.values())}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "summary": summary,
              "qwen14b_bf16": bf16_ledger(), "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
