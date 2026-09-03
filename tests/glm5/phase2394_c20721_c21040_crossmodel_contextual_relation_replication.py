#!/usr/bin/env python3
"""Frozen GLM4/DS7B replication of isolated and contextual relation fields."""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2388 = RESULT / "phase2388_c18801_c19120_semantic_lexical_contract"
P2389 = RESULT / "phase2389_c19121_c19440_crossmodel_autonomous_capability"
OUT = RESULT / "phase2394_c20721_c21040_crossmodel_contextual_relation_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2394
CAMPAIGN = "C20721-C21040"
MODELS = ("glm4", "deepseek7b")
FAMILIES = ("preference", "taxonomy", "temporal", "causal", "comparison", "spatial", "role_binding", "ownership_transfer")
LANGUAGES = ("en", "zh")
FROZEN_RELATIVE_DEPTH = 0.5 * (26 / 37 + 27 / 41)

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as capture  # noqa: E402
import phase2391_c19761_c20080_semantic_lexical_adjudication as adjudicate  # noqa: E402
import phase2392_c20081_c20400_contextual_coordinate_gear_atlas as atlas  # noqa: E402
import phase2393_c20401_c20720_frozen_context_causal_adjudication as causal  # noqa: E402

# Reuse the already audited all-coordinate collectors while changing only their output root.
capture.OUT = OUT


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {
        "base": base,
        "independent_rows": base / "index/independent_rows.jsonl",
        "selection_rows": base / "index/selection_rows.jsonl",
        "independent_mean": base / "raw/independent_mean.float16.npy",
        "boundary": base / "raw/semantic_selection_prompt_boundary.float16.npy",
        "scores": base / "raw/semantic_selection_sequence_scores.float32.npy",
        "score_progress": base / "raw/progress_scores_batched.json",
        "context_response": base / "derived/all_layer_partition_relation_response.float32.npy",
        "isolated_response": base / "derived/frozen_layer_isolated_surface_response.float32.npy",
        "fingerprint": base / "derived/frozen_context_coordinate_fingerprint.float32.npy",
        "final": base / "analysis/final.json",
    }


def collect_batched_scores(key: str, model, rows: list[dict], batch_size: int) -> dict:
    p = paths(key)
    if p["scores"].exists():
        matrix = np.load(p["scores"], mmap_mode="r")
    else:
        target = causal.sequence_scores(model, rows, "target_ids", batch_size)
        foil = causal.sequence_scores(model, rows, "foil_ids", batch_size)
        matrix = np.stack((target, foil, target - foil), axis=1).astype(np.float32)
        np.save(p["scores"], matrix, allow_pickle=False)
        save(p["score_progress"], {"completed": len(rows), "shape": list(matrix.shape), "batched": True})
    result = {
        "rows": len(rows),
        "target_over_foil": float(np.mean(matrix[:, 2] > 0)),
        "mean_margin": float(np.mean(matrix[:, 2])),
        "by_partition": {part: float(np.mean(matrix[[i for i, row in enumerate(rows) if row["partition"] == part], 2] > 0))
                         for part in ("discovery", "confirmation", "fresh_unit_lockbox")},
    }
    close(matrix)
    return result


def frozen_isolated(values: np.ndarray, rows: list[dict]) -> dict:
    evaluated = adjudicate.evaluate(values, rows)
    return {
        "confirmation_canonical": evaluated["confirmation_canonical"],
        "confirmation_cross_surface": evaluated["confirmation_cross_surface"],
        "lockbox_same_bag_direction": evaluated["lockbox_same_bag_direction"],
        "lockbox_cross_surface": evaluated["lockbox_cross_surface"],
        "lockbox_cross_surface_by_family": evaluated["lockbox_cross_surface_by_family"],
    }


def analyze_model(key: str, label: str, collection: dict, behavior: dict) -> dict:
    p = paths(key)
    independent_rows = read_rows(p["independent_rows"])
    selection_rows = read_rows(p["selection_rows"])
    mean = np.load(p["independent_mean"], mmap_mode="r")
    boundary = np.load(p["boundary"], mmap_mode="r")
    qcount = int(boundary.shape[1])
    qpoint = int(round(FROZEN_RELATIVE_DEPTH * (qcount - 1)))

    isolated_q0 = frozen_isolated(np.asarray(mean[:, 0], dtype=np.float32), independent_rows)
    isolated_frozen = frozen_isolated(np.asarray(mean[:, qpoint], dtype=np.float32), independent_rows)
    isolated_response, isolated_stability = adjudicate.contrast_response(
        np.asarray(mean[:, qpoint], dtype=np.float32), independent_rows
    )

    context_response = atlas.relation_response(boundary, selection_rows)
    values = np.asarray(boundary[:, qpoint], dtype=np.float32)
    contextual = adjudicate.evaluate_boundary(values, selection_rows)
    generalization = atlas.cross_condition_generalization(values, selection_rows)
    disc, conf, lock = context_response[qpoint, 0], context_response[qpoint, 1], context_response[qpoint, 2]
    stability = {
        "confirmation_cosine": float(np.mean([atlas.cosine(disc[fi, li], conf[fi, li]) for fi in range(8) for li in range(2)])),
        "lockbox_cosine": float(np.mean([atlas.cosine(disc[fi, li], lock[fi, li]) for fi in range(8) for li in range(2)])),
        "cross_language_cosine": float(np.mean([atlas.cosine(disc[fi, 0], disc[fi, 1]) for fi in range(8)])),
        "coordinate_sign_confirmation": float(np.mean(np.sign(disc) == np.sign(conf))),
        "coordinate_sign_lockbox": float(np.mean(np.sign(disc) == np.sign(lock))),
    }
    fingerprints, group_summary, _ = atlas.group_signature(disc)
    p["context_response"].parent.mkdir(parents=True, exist_ok=True)
    np.save(p["context_response"], context_response.astype(np.float32), allow_pickle=False)
    np.save(p["isolated_response"], isolated_response.astype(np.float32), allow_pickle=False)
    np.save(p["fingerprint"], fingerprints.astype(np.float32), allow_pickle=False)
    autonomous = json.loads((P2389 / key / "analysis/final.json").read_text(encoding="utf-8"))
    result = {
        "model": key,
        "model_label": label,
        "collection": collection,
        "frozen_relative_depth": FROZEN_RELATIVE_DEPTH,
        "frozen_qpoint": qpoint,
        "isolated_embedding": isolated_q0,
        "isolated_frozen": isolated_frozen,
        "isolated_cross_surface_response_stability": isolated_stability,
        "contextual": contextual,
        "contextual_response_stability": stability,
        "generalization": generalization,
        "coordinate_groups": group_summary,
        "teacher_forced_behavior": behavior,
        "autonomous_behavior": autonomous.get("semantic_selection", autonomous),
        "claim_boundary": "relative depth frozen from Qwen before GLM/DS inspection; readout is not a causal gear",
        "all_checks_passed": (
            collection["independent"]["shape"][0] == 768
            and collection["boundary"]["shape"][0] == 384
            and math.isfinite(contextual["lockbox_accuracy"])
        ),
    }
    close(mean); close(boundary)
    save(p["final"], result)
    return result


def run_model(key: str, source_independent: list[dict], source_selection: list[dict]) -> dict:
    p = paths(key)
    if p["final"].exists(): return json.loads(p["final"].read_text(encoding="utf-8"))
    model, tokenizer, label = capability.load_model(key)
    try:
        independent = capture.compile_independent(tokenizer, source_independent)
        selection = capture.compile_selection(tokenizer, source_selection)
        write_rows(p["independent_rows"], independent)
        write_rows(p["selection_rows"], selection)
        batch = 5 if key == "glm4" else 6
        collection = {
            "independent": capture.collect_independent(key, model, independent, batch),
            "boundary": capture.collect_boundary(key, model, selection, batch),
        }
        behavior = collect_batched_scores(key, model, selection, batch)
        return analyze_model(key, label, collection, behavior)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: GLM4与DS7B冻结深度的语义—上下文关系全坐标复现（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不查看GLM4/DS7B结果选层，而先由Qwen4B q26/37与Qwen14B q27/41冻结相对深度均值 `{FROZEN_RELATIVE_DEPTH:.6f}`，映射到各模型物理checkpoint。两模型按GLM4→DS7B顺序独占CUDA。对Phase2388的768个独立句采集embedding、每层和final norm的句末/全token均值全部坐标；对384个同词反关系二选一上下文采集最后prompt token的全部checkpoint/坐标。以discovery拟合族×语言全坐标方向，在confirmation及fresh-unit锁箱复测；另计算跨语言、留一关系族和正确整句对反关系foil的teacher-forced logprob。

$$q_M=\operatorname{{round}}\left[\frac12\left(\frac{{26}}{{37}}+\frac{{27}}{{41}}\right)(Q_M-1)\right],\qquad
R_{{q,f,\ell,j}}=\mathbb E[H_{{q,j}}|d=0]-\mathbb E[H_{{q,j}}|d=1].$$

**结果汇总。** 跨模型冻结复现 `{json.dumps(result['summary'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2394_c20721_c21040_crossmodel_contextual_relation_replication.py`；逐模型全坐标场、全层关系响应、坐标指纹、行为分数和final位于 `tests/glm5/result/phase2394_c20721_c21040_crossmodel_contextual_relation_replication`。

**理论进展、问题硬伤与结论。** 冻结相对深度避免在新模型上挑最好层，但不同架构的同相对深度不保证同计算阶段；INT8与BF16/NF4的激活幅度不可直接比较。独立句跨表面失败与上下文场成功可共同支持“关系依上下文形成”，仍不证明模型运行时使用本研究的线性判别。DS7B若受thinking模板/截断影响，teacher-forced阳性与自主失败必须并列报告，不能把协议失败解释成没有内部关系信息。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    independent = read_rows(P2388 / "material/independent_relation_sentences.jsonl")
    selection = read_rows(P2388 / "material/semantic_selection_rows.jsonl")
    models = {key: run_model(key, independent, selection) for key in MODELS}
    summary = {
        key: {
            "qpoint": value["frozen_qpoint"],
            "isolated_cross_surface": value["isolated_frozen"]["lockbox_cross_surface"],
            "embedding_cross_surface": value["isolated_embedding"]["lockbox_cross_surface"],
            "context_lockbox": value["contextual"]["lockbox_accuracy"],
            "context_by_family": value["contextual"]["lockbox_by_family"],
            "cross_language": value["generalization"]["cross_language_mean"],
            "heldout_family": value["generalization"]["heldout_family_mean"],
            "teacher_target_over_foil": value["teacher_forced_behavior"]["target_over_foil"],
        }
        for key, value in models.items()
    }
    adjudication_result = {
        "four_model_contextual_readout": all(v >= .65 for v in [
            summary["glm4"]["context_lockbox"], summary["deepseek7b"]["context_lockbox"],
            .78125, .7395833333333334,
        ]),
        "four_model_static_cross_surface": all(v >= .60 for v in [
            summary["glm4"]["isolated_cross_surface"], summary["deepseek7b"]["isolated_cross_surface"],
            .46875, .5208333333333334,
        ]),
        "universal_cross_language_operator": all(v >= .65 for v in [
            summary["glm4"]["cross_language"], summary["deepseek7b"]["cross_language"], .5729166666666666, .5729166666666666,
        ]),
        "unseen_family_operator": all(v >= .65 for v in [
            summary["glm4"]["heldout_family"], summary["deepseek7b"]["heldout_family"], .53125, .48958333333333337,
        ]),
        "mechanism_closed": False,
    }
    checks = {
        "sequential_models": list(models) == list(MODELS),
        "all_model_checks": all(value["all_checks_passed"] for value in models.values()),
        "full_coordinate_arrays": all(paths(key)["context_response"].exists() and paths(key)["fingerprint"].exists() for key in MODELS),
        "finite": all(math.isfinite(item) for value in summary.values() for item in (
            value["isolated_cross_surface"], value["context_lockbox"], value["teacher_target_over_foil"]
        )),
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "models": models, "summary": summary,
              "adjudication": adjudication_result, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
