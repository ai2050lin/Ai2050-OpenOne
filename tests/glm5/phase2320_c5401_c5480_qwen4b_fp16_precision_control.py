#!/usr/bin/env python3
"""Control the apparent Qwen3-14B local linearity for FP16 versus BF16 precision."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2315 = RESULT / "phase2315_c5041_c5100_active_response_contract"
P2318 = RESULT / "phase2318_c5241_c5320_crossmodel_directional_topology"
P2319 = RESULT / "phase2319_c5321_c5400_active_response_atlas_cleanup"
OUT = RESULT / "phase2320_c5401_c5480_qwen4b_fp16_precision_control"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
ROWS_PATH = P2315 / "material/natural_active_response_bilingual.jsonl"
PHASE = 2320
CAMPAIGN = "C5401-C5480"

sys.path.insert(0, str(TESTS))
import phase1332_bf16_utils as model_base  # noqa: E402
import phase2318_c5241_c5320_crossmodel_directional_topology as cross  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_metrics(value: dict) -> dict:
    return {
        "prediction": value["prediction"],
        "median_pair_superposition_relative_mse": value["median_pair_superposition_relative_mse"],
        "median_even_to_odd_l2": value["median_even_to_odd_l2"],
        "relative_response_topology": value["relative_response_topology"],
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: Qwen3-4B FP16 数值精度对照与 14B 局部线性重裁（{CAMPAIGN}） [{stamp}]

**测试原理、对象与用例。** Phase2318 中 Qwen3-14B 使用 FP16，而 Qwen3-4B、GLM 与 DeepSeek 使用 BF16；因此 14B 的小成对叠加误差可能受数值分辨率混淆。本期冻结并复用同一 64 行 fresh 行为面板、32 行 fresh_lockbox 主动面板、4 个 Rademacher 基方向、2 个成对方向、相对源深度约 `0.28/0.56/0.83`、`q+1/q+4/final_norm` 目标和 1% 源状态范数剂量，仅把 Qwen3-4B 以非量化 FP16 重载。候选与自由续写重新执行；内部响应保存全部 2560 个坐标。
$$
E_{{pair}}=\frac{{\lVert D(r_a+r_b)-D(r_a)-D(r_b)\rVert_2^2}}{{\lVert D(r_a+r_b)\rVert_2^2+\varepsilon}},\qquad
Q_{{even/odd}}=\frac{{\lVert (H^++H^-)/2-H^0\rVert_2}}{{\lVert (H^+-H^-)/2\rVert_2+\varepsilon}}.
$$

**结果汇总与门槛。** FP16 行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；同一 4B 的 BF16 参考 `{json.dumps(result['bf16_reference'], ensure_ascii=False)}`；4B FP16 `{json.dumps(result['fp16_metrics'], ensure_ascii=False)}`；14B FP16 `{json.dumps(result['qwen14_fp16_reference'], ensure_ascii=False)}`；比较裁决 `{json.dumps(result['comparison'], ensure_ascii=False)}`。本期不设“线性即语义”通过门；门只裁决精度是否足以解释此前差异。结果文件 `tests/glm5/result/phase2320_c5401_c5480_qwen4b_fp16_precision_control`，脚本 `tests/glm5/phase2320_c5401_c5480_qwen4b_fp16_precision_control.py`。

**分析、理论进展、硬伤与结论。** 理论主体仍为“条件化输出场闭合理论”。若 4B FP16 的 `E_pair` 和 `Q_even/odd` 相对 BF16 大幅下降，则 Phase2318 的规模比较受精度严重混淆；若仍接近 BF16，才保留规模/模型差异候选。无论哪种结果，随机 Rademacher 方向的局部线性都不是语言族齿轮。FP16 与 BF16 会同时改变舍入、微小响应可见性和钩子输出；单剂量仍不能界定真正微分区；面板只有 32 行且没有独立人类盲评。下一步发布本期 FP16 全坐标导数和偶响应，并以精度重裁后的结论约束图谱观察。"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    for parent in (P2318, P2319):
        value = json.loads((parent / "analysis/final.json").read_text(encoding="utf-8"))
        if not value["all_checks_passed"]:
            raise RuntimeError(("parent_not_authorized", parent))
    raw_rows = cross.behavior_rows(cross.read_rows(ROWS_PATH))
    model = tokenizer = None
    try:
        model, tokenizer, device = model_base.load_model(
            "qwen3", dtype=torch.float16, use_8bit=False,
        )
        dtypes = model_base.parameter_dtype_counts(model)
        compiled = cross.compile_rows(tokenizer, raw_rows)
        cross.write_rows(OUT / "material/compiled_fresh_panel.jsonl", compiled)
        scores = cross.sequence_scores(model, device, compiled, 32)
        cross.write_rows(OUT / "behavior/sequence_scores.jsonl", scores)
        free = cross.free_generation(model, tokenizer, device, compiled, 32)
        cross.write_rows(OUT / "behavior/free_generation.jsonl", free)
        behavior = cross.behavior_summary(scores, free)
        save(OUT / "behavior/summary.json", behavior)
        active = cross.active_rows(compiled)
        field = cross.active_capture(model, device, active, OUT)
        index_rows = cross.read_rows(OUT / "index/active_rows.jsonl")
        metrics = cross.functional_metrics(
            ROOT / field["derivative"], ROOT / field["even"], ROOT / field["norms"],
            index_rows, "Qwen3-4B-FP16",
        )
        save(OUT / "analysis/functional_metrics.json", metrics)
        phase2318 = json.loads((P2318 / "analysis/final.json").read_text(encoding="utf-8"))
        bf16 = phase2318["qwen4_reference"]
        q14 = phase2318["models"]["qwen3_14b"]["functional_metrics"]
        bf16_pair = float(bf16["median_pair_superposition_relative_mse"])
        fp16_pair = float(metrics["median_pair_superposition_relative_mse"])
        bf16_even = float(bf16["median_even_to_odd_l2"])
        fp16_even = float(metrics["median_even_to_odd_l2"])
        comparison = {
            "pair_mse_ratio_fp16_over_bf16": fp16_pair / bf16_pair,
            "even_odd_ratio_fp16_over_bf16": fp16_even / bf16_even,
            "pair_mse_drop_at_least_50_percent": fp16_pair <= 0.5 * bf16_pair,
            "even_odd_drop_at_least_50_percent": fp16_even <= 0.5 * bf16_even,
            "precision_materially_confounds_scale_comparison": (
                fp16_pair <= 0.5 * bf16_pair or fp16_even <= 0.5 * bf16_even
            ),
            "fp16_4b_matches_q14_within_factor_two_pair": (
                0.5 <= fp16_pair / float(q14["median_pair_superposition_relative_mse"]) <= 2.0
            ),
            "claim_boundary": "numerical precision control for local random-direction response only",
        }
        checks = {
            "parents_authorized": True,
            "fp16_nonquantized": dtypes.get("float16", 0) > 0,
            "all_behavior_rows": behavior["rows"] == 64,
            "all_active_rows": field["shape"][0] == 32,
            "all_coordinates": field["shape"][-1] == 2560,
            "frozen_probe_count": field["shape"][2] == 6,
            "same_relative_sources": len(field["sources"]) == 3,
            "no_semantic_linearity_claim": True,
        }
        result = {
            "phase": PHASE, "campaign": CAMPAIGN,
            "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
            "model": "Qwen3-4B", "precision": "float16", "parameter_dtypes": dtypes,
            "behavior": behavior, "field": field,
            "bf16_reference": compact_metrics(bf16),
            "fp16_metrics": compact_metrics(metrics),
            "qwen14_fp16_reference": compact_metrics(q14),
            "comparison": comparison, "checks": checks,
            "all_checks_passed": all(checks.values()),
            "strict_conclusion": (
                "This phase isolates the FP16/BF16 measurement confound for local random-direction "
                "propagation. It does not identify a language-family gear or a semantic circuit."
            ),
            "next_authorization": "Publish exact-coordinate FP16 derivative/even fields and update the atlas.",
        }
        save(final_path, result)
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            model_base.release_bf16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
