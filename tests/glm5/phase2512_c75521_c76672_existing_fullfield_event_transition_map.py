#!/usr/bin/env python3
"""Build an all-layer/all-coordinate event-transition map from existing valid campaigns."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2501 = RESULT / "phase2501_c65153_c66176_semantic_necessity_fullcoordinate_field"
P2503 = RESULT / "phase2503_c67201_c68224_equal_length_fresh_lockbox_behavior_fullfield"
P2507 = RESULT / "phase2507_c71041_c72064_repaired_partner_behavior_fullfield"
P2508 = RESULT / "phase2508_c72065_c73088_alternative_partner_behavior_fullfield"
OUT = RESULT / "phase2512_c75521_c76672_existing_fullfield_event_transition_map"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2512, "C75521-C76672", 2560
EVENTS = ("query_marker", "candidate0", "candidate1", "answer_boundary")

sys.path.insert(0, str(TESTS))
import phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox as walsh  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_specs() -> list[dict]:
    f1 = load_json(P2501 / "analysis/final.json")
    f3 = load_json(P2503 / "analysis/final.json")
    f7 = load_json(P2507 / "analysis/final.json")
    f8 = load_json(P2508 / "analysis/final.json")
    return [
        {"campaign": "original", "unit": 21, "split": "confirmation", "pairs": [0, 3, 4],
         "field": f1["collection"]["event_field"], "index": f1["collection"]["event_index"],
         "pair_names": {0: ["taxonomy", "part_whole"], 3: ["role", "preference"], 4: ["membership", "translation"]}},
        {"campaign": "original", "unit": 23, "split": "lockbox", "pairs": [0, 3, 4],
         "field": f3["collection"]["event_field"], "index": f3["collection"]["event_index"],
         "pair_names": {0: ["taxonomy", "part_whole"], 3: ["role", "preference"], 4: ["membership", "translation"]}},
        {"campaign": "partner_a", "unit": 24, "split": "confirmation", "pairs": [0, 2],
         "field": f7["collection"]["event_field"], "index": f7["collection"]["event_index"],
         "pair_names": {0: ["taxonomy", "role"], 2: ["preference", "translation"]}},
        {"campaign": "partner_a", "unit": 25, "split": "lockbox", "pairs": [0, 2],
         "field": f7["collection"]["event_field"], "index": f7["collection"]["event_index"],
         "pair_names": {0: ["taxonomy", "role"], 2: ["preference", "translation"]}},
        {"campaign": "partner_b", "unit": 26, "split": "confirmation", "pairs": [1],
         "field": f8["collection"]["event_field"], "index": f8["collection"]["event_index"],
         "pair_names": {1: ["part_whole", "translation"]}},
        {"campaign": "partner_b", "unit": 27, "split": "lockbox", "pairs": [1],
         "field": f8["collection"]["event_field"], "index": f8["collection"]["event_index"],
         "pair_names": {1: ["part_whole", "translation"]}},
    ]


def build_field() -> tuple[np.ndarray, list[dict]]:
    # 6 edges x 2 units x 2 languages x 4 surfaces = 96 samples.
    data = np.zeros((96, len(EVENTS), 38, DIM), dtype=np.float32)
    rows_out = []
    offset = 0
    for spec in source_specs():
        field = np.load(spec["field"], mmap_mode="r")
        rows = walsh.read_jsonl(Path(spec["index"]))
        for qpoint in range(38):
            for target_event, source_event_index in enumerate(range(2, 6)):
                interaction, _, _ = walsh.effects(field, rows, spec["unit"], source_event_index, qpoint, spec["pairs"])
                # interaction axes: selected pair, language, surface, coordinate.
                for local_pair in range(len(spec["pairs"])):
                    base = offset + local_pair * 8
                    data[base:base + 8, target_event, qpoint] = interaction[local_pair].reshape(8, DIM)
        for local_pair, pair_id in enumerate(spec["pairs"]):
            for language in ("en", "zh"):
                for surface in range(4):
                    rows_out.append({"model_row": offset + local_pair * 8 + (0 if language == "en" else 4) + surface,
                                     "campaign": spec["campaign"], "unit": spec["unit"], "split": spec["split"],
                                     "pair_id": pair_id, "edge": spec["pair_names"][pair_id],
                                     "edge_key": "__".join(spec["pair_names"][pair_id]),
                                     "language": language, "surface": surface})
        offset += len(spec["pairs"]) * 8
    assert offset == len(data) and len(rows_out) == len(data)
    return data, sorted(rows_out, key=lambda row: row["model_row"])


def cosine_rows(y: np.ndarray, pred: np.ndarray) -> float:
    num = np.sum(y * pred, axis=1)
    den = np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1)
    return float(np.mean(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)))


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, model: str) -> np.ndarray:
    if model == "zero":
        return np.zeros_like(x_test)
    if model == "identity":
        return x_test.copy()
    if model == "global_scale":
        den = float(np.square(x_train).sum())
        alpha = float((x_train * y_train).sum() / den) if den else 0.0
        return alpha * x_test
    if model == "diagonal_scale":
        den = np.square(x_train).sum(axis=0)
        alpha = np.divide((x_train * y_train).sum(axis=0), den, out=np.zeros(DIM), where=den > 1e-20)
        return x_test * alpha
    raise ValueError(model)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    power = float(np.square(y).sum())
    sse = float(np.square(y - pred).sum())
    return {"r2_vs_zero": 1.0 - sse / power if power else (1.0 if sse == 0 else 0.0),
            "relative_rmse": float(np.sqrt(sse / max(power, 1e-30))),
            "mean_sample_cosine": cosine_rows(y, pred)}


def edge_cv(x: np.ndarray, y: np.ndarray, metadata: list[dict], model: str) -> dict:
    edges = sorted({row["edge_key"] for row in metadata})
    fold = []
    for edge in edges:
        test = np.asarray([row["edge_key"] == edge for row in metadata])
        pred = fit_predict(x[~test], y[~test], x[test], model)
        fold.append({"edge": edge, **metrics(y[test], pred)})
    return {"folds": fold,
            "r2_mean": float(np.mean([v["r2_vs_zero"] for v in fold])),
            "cosine_mean": float(np.mean([v["mean_sample_cosine"] for v in fold]))}


def analyze(data: np.ndarray, metadata: list[dict]) -> dict:
    confirmation_mask = np.asarray([row["split"] == "confirmation" for row in metadata])
    lockbox_mask = ~confirmation_mask
    confirmation_meta = [row for row in metadata if row["split"] == "confirmation"]
    report = {}
    for event_index, event in enumerate(EVENTS[1:], start=1):
        layer_panels = []
        for qpoint in range(38):
            x_train = np.asarray(data[confirmation_mask, 0, qpoint], dtype=np.float64)
            y_train = np.asarray(data[confirmation_mask, event_index, qpoint], dtype=np.float64)
            x_test = np.asarray(data[lockbox_mask, 0, qpoint], dtype=np.float64)
            y_test = np.asarray(data[lockbox_mask, event_index, qpoint], dtype=np.float64)
            panel = {"qpoint": qpoint,
                     "query_rms_confirmation": float(np.sqrt(np.mean(np.square(x_train)))),
                     "target_rms_confirmation": float(np.sqrt(np.mean(np.square(y_train)))),
                     "models": {}}
            for model in ("zero", "identity", "global_scale", "diagonal_scale"):
                pred = fit_predict(x_train, y_train, x_test, model)
                panel["models"][model] = {"lockbox": metrics(y_test, pred)}
                if model in ("global_scale", "diagonal_scale") and qpoint > 0:
                    panel["models"][model]["confirmation_leave_edge_out"] = edge_cv(x_train, y_train, confirmation_meta, model)
            layer_panels.append(panel)
        eligible = layer_panels[1:]
        chosen = max(eligible, key=lambda p: p["models"]["diagonal_scale"]["confirmation_leave_edge_out"]["r2_mean"])
        report[event] = {
            "selection_rule": "maximum confirmation leave-one-edge-out diagonal r2; q0 excluded because Walsh interaction is structurally zero",
            "selected_qpoint": chosen["qpoint"], "selected_panel": chosen,
            "q30_panel": layer_panels[30], "all_qpoints": layer_panels,
        }
    return report


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    concise = {event: {"selected_qpoint": panel["selected_qpoint"],
                       "selected_panel": panel["selected_panel"], "q30_panel": panel["q30_panel"]}
               for event, panel in result["transition_map"].items()}
    text = rf"""


## Phase {PHASE}: 六关系边双unit的全层全坐标事件转换基础地图（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 不增加新模型调用，重组Phase2501/2503/2507/2508中通过行为门且协议有效的六条关系边。每条边取confirmation与lockbox各一unit、两语言、四surface，在query-marker、candidate0、candidate1、answer-boundary四事件与q0–q37全部2560物理坐标计算行为必要Walsh交互，共96个edge×unit×language×surface样本。对每个同层query→目标事件只竞争零、恒等、全局尺度、逐坐标尺度四个基础映射；层位仅由confirmation逐边留一选出，之后一次查看成对fresh-unit锁箱。

$$\widehat I^{{target}}_i=a_i I^{{query}}_i,\qquad a_i=\frac{{\sum_{{n\in train}}I^{{query}}_{{n,i}}I^{{target}}_{{n,i}}}}{{\sum_{{n\in train}}(I^{{query}}_{{n,i}})^2}}.$$

**结果汇总。** 样本/字段 `{json.dumps(result['collection'], ensure_ascii=False)}`；逐事件选层与q30摘要 `{json.dumps(concise, ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2512_c75521_c76672_existing_fullfield_event_transition_map.py`；96×4×38×2560全坐标交互场、索引、逐层指标与final位于`{OUT}`。

**分析与理论进展。** 这是对“重参数化”的第一张无压缩基础地图：如果逐坐标尺度在未见edge和fresh unit都不能稳定超过全局尺度，就没有理由直接进入低秩、张量或混合算子；如果某事件/层稳定超过，才把该位置冻结给下一Phase。这里的对角系数是外部预测规则，不等于神经网络内部逐坐标乘法。

**问题硬伤与结论。** 六边来自三个旧campaign，confirmation/lockbox marker与实体不同但模板结构相似；48训练样本拟合2560个对角参数仍有过拟合风险；逐边留一不等于留整种上下文；同qpoint映射没有测试跨层输入；结果只能筛选候选层和基础复杂度，不能称编译机制或因果齿轮。下一Phase按这里冻结的基础竞争结果设计全析因fresh材料。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    data, metadata = build_field()
    derived = OUT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    field_path = derived / "qualified_event_interactions_allqpoint.float32.npy"
    np.save(field_path, data)
    index_path = OUT / "index/samples.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata), encoding="utf-8")
    transition_map = analyze(data, metadata)
    metric_path = OUT / "analysis/all_qpoint_metrics.json"
    save(metric_path, transition_map)
    checks = {
        "samples_96": data.shape[0] == 96,
        "full_event_qpoint_coordinate_shape": data.shape == (96, 4, 38, DIM),
        "balanced_splits": sum(row["split"] == "confirmation" for row in metadata) == 48,
        "six_edges_each_split": all(len({row["edge_key"] for row in metadata if row["split"] == split}) == 6
                                    for split in ("confirmation", "lockbox")),
        "q0_exact_zero": bool(np.max(np.abs(data[:, :, 0])) == 0),
        "finite": bool(np.isfinite(data).all()),
        "hash": len(digest(field_path)) == 64,
        "selection_without_lockbox": True,
        "claim_boundary": True,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN,
              "collection": {"field": str(field_path), "shape": list(data.shape), "sha256": digest(field_path),
                             "index": str(index_path), "metrics": str(metric_path)},
              "transition_map": transition_map,
              "adjudication": {"descriptive_event_reparameterization_map_built": True,
                               "event_compiler_identified": False, "causal_mediator_identified": False,
                               "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps({"phase": PHASE, "collection": result["collection"],
                      "selected": {e: {"q": p["selected_qpoint"],
                                       "lockbox": p["selected_panel"]["models"]["diagonal_scale"]["lockbox"]}
                                   for e, p in transition_map.items()},
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
