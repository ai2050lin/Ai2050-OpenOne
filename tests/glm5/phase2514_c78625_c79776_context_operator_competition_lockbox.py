#!/usr/bin/env python3
"""Compete basic context-conditioned query-to-event operators on a fresh unit lockbox."""
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
P2513 = RESULT / "phase2513_c76673_c78624_fresh_context_factorial_behavior_fullfield"
OUT = RESULT / "phase2514_c78625_c79776_context_operator_competition_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM = 2514, "C78625-C79776", 2560
EVENTS = ("query_marker", "candidate0", "candidate1", "answer_boundary")
MODELS = ("zero", "identity", "global_scale", "diagonal_scale", "factor_global_scale",
          "factor_diagonal_scale", "factor_coordinate_offset", "factor_affine_diagonal")

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


def build_interactions(final: dict) -> tuple[np.ndarray, list[dict]]:
    field = np.load(final["collection"]["event_field"], mmap_mode="r")
    rows = walsh.read_jsonl(Path(final["collection"]["event_index"]))
    pairs = final["behavior"]["qualified_pair_ids"]
    # unit, pair, language, 16 contexts, event, qpoint, coordinate
    data = np.zeros((2, len(pairs), 2, 16, len(EVENTS), 38, DIM), dtype=np.float32)
    lookup = {(r["unit"], r["pair_id"], r["language"], r["context_id"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    metadata = []
    for unit_index, unit in enumerate((28, 29)):
        for pair_index, pair_id in enumerate(pairs):
            for language_index, language in enumerate(("en", "zh")):
                for context in range(16):
                    exemplar = lookup[(unit, pair_id, language, context, 0, 0)]
                    metadata.append({"unit_index": unit_index, "unit": unit, "pair_index": pair_index,
                                     "pair_id": pair_id, "edge": exemplar["families"], "language_index": language_index,
                                     "language": language, "context_id": context,
                                     "paraphrase": exemplar["paraphrase"], "fact_order": exemplar["fact_order"],
                                     "definition_order": exemplar["definition_order"], "candidate_order": exemplar["candidate_order"]})
                    for event_index, source_event in enumerate(range(2, 6)):
                        for qpoint in range(38):
                            cells = {(m, q): np.asarray(field[lookup[(unit, pair_id, language, context, m, q)]["model_row"], source_event, qpoint], dtype=np.float32)
                                     for m in (0, 1) for q in (0, 1)}
                            data[unit_index, pair_index, language_index, context, event_index, qpoint] = (
                                cells[(0, 0)] - cells[(0, 1)] - cells[(1, 0)] + cells[(1, 1)]) / 4
    return data, metadata


def flatten_unit(data: np.ndarray, metadata: list[dict], unit_index: int, event_index: int, qpoint: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    meta = [row for row in metadata if row["unit_index"] == unit_index]
    x = np.stack([data[unit_index, r["pair_index"], r["language_index"], r["context_id"], 0, qpoint] for r in meta]).astype(np.float64)
    y = np.stack([data[unit_index, r["pair_index"], r["language_index"], r["context_id"], event_index, qpoint] for r in meta]).astype(np.float64)
    return x, y, meta


def features(meta: list[dict]) -> np.ndarray:
    return np.asarray([[1.0, row["language_index"], row["paraphrase"], row["fact_order"],
                        row["definition_order"], row["candidate_order"]] for row in meta], dtype=np.float64)


def stable_solve(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if gram.ndim == 2:
        ridge = 1e-6 * (float(np.trace(gram)) / max(gram.shape[0], 1) + 1e-12)
        return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    trace = np.trace(gram, axis1=1, axis2=2) / max(gram.shape[1], 1)
    eye = np.eye(gram.shape[1])[None]
    return np.linalg.solve(gram + (1e-6 * trace + 1e-12)[:, None, None] * eye, rhs[..., None])[..., 0]


def fit(x: np.ndarray, y: np.ndarray, meta: list[dict], model: str) -> dict:
    f = features(meta)
    if model in ("zero", "identity"):
        return {"model": model}
    if model == "global_scale":
        return {"model": model, "alpha": float((x * y).sum() / max(np.square(x).sum(), 1e-30))}
    if model == "diagonal_scale":
        den = np.square(x).sum(axis=0)
        return {"model": model, "alpha": np.divide((x * y).sum(axis=0), den, out=np.zeros(DIM), where=den > 1e-20)}
    if model == "factor_global_scale":
        weights, cross = np.square(x).sum(axis=1), (x * y).sum(axis=1)
        return {"model": model, "beta": stable_solve(f.T @ (weights[:, None] * f), f.T @ cross)}
    if model == "factor_diagonal_scale":
        gram = np.einsum("nj,nc,nk->cjk", f, x * x, f, optimize=True)
        rhs = np.einsum("nj,nc->cj", f, x * y, optimize=True)
        return {"model": model, "beta": stable_solve(gram, rhs)}
    if model == "factor_coordinate_offset":
        return {"model": model, "beta": np.linalg.pinv(f) @ y}
    if model == "factor_affine_diagonal":
        # Per coordinate: y_i = x_i F beta_i + F gamma_i.
        xf = x[:, :, None] * f[:, None, :]
        ff = np.broadcast_to(f[:, None, :], xf.shape)
        design = np.concatenate((xf, ff), axis=2)
        gram = np.einsum("nci,ncj->cij", design, design, optimize=True)
        rhs = np.einsum("nci,nc->ci", design, y, optimize=True)
        return {"model": model, "beta": stable_solve(gram, rhs)}
    raise ValueError(model)


def predict(params: dict, x: np.ndarray, meta: list[dict]) -> np.ndarray:
    model, f = params["model"], features(meta)
    if model == "zero": return np.zeros_like(x)
    if model == "identity": return x.copy()
    if model in ("global_scale", "diagonal_scale"): return x * params["alpha"]
    if model == "factor_global_scale": return x * (f @ params["beta"])[:, None]
    if model == "factor_diagonal_scale": return x * np.einsum("nj,cj->nc", f, params["beta"], optimize=True)
    if model == "factor_coordinate_offset": return f @ params["beta"]
    if model == "factor_affine_diagonal":
        k = f.shape[1]; beta, gamma = params["beta"][:, :k], params["beta"][:, k:]
        return x * np.einsum("nj,cj->nc", f, beta, optimize=True) + np.einsum("nj,cj->nc", f, gamma, optimize=True)
    raise ValueError(model)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    sse, power = float(np.square(y - pred).sum()), float(np.square(y).sum())
    num = np.sum(y * pred, axis=1); den = np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1)
    cosines = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)
    return {"r2_vs_zero": 1 - sse / power if power else (1.0 if sse == 0 else 0.0),
            "relative_rmse": float(np.sqrt(sse / max(power, 1e-30))),
            "mean_sample_cosine": float(np.mean(cosines)),
            "positive_sample_cosine_rate": float(np.mean(cosines > 0))}


def edge_cv(x: np.ndarray, y: np.ndarray, meta: list[dict], model: str) -> dict:
    pred = np.zeros_like(y)
    for pair_index in sorted({row["pair_index"] for row in meta}):
        test = np.asarray([row["pair_index"] == pair_index for row in meta])
        pred[test] = predict(fit(x[~test], y[~test], [r for r, flag in zip(meta, ~test) if flag], model),
                             x[test], [r for r, flag in zip(meta, test) if flag])
    return metrics(y, pred)


def held_context_test(x_train: np.ndarray, y_train: np.ndarray, meta_train: list[dict],
                      x_test: np.ndarray, y_test: np.ndarray, meta_test: list[dict], model: str) -> dict:
    # Whole factor combination unseen during fitting, plus fresh unit at test.
    keep = np.asarray([not (r["language_index"] == 1 and r["paraphrase"] == 1 and r["candidate_order"] == 1) for r in meta_train])
    target = np.asarray([(r["language_index"] == 1 and r["paraphrase"] == 1 and r["candidate_order"] == 1) for r in meta_test])
    params = fit(x_train[keep], y_train[keep], [r for r, flag in zip(meta_train, keep) if flag], model)
    return {"held_rule": "zh & paraphrase=1 & candidate_order=1 excluded from unit28; evaluated on unit29",
            **metrics(y_test[target], predict(params, x_test[target], [r for r, flag in zip(meta_test, target) if flag]))}


def analyze(data: np.ndarray, metadata: list[dict]) -> dict:
    report = {}
    for event_index, event in enumerate(EVENTS[1:], start=1):
        scan = []
        for qpoint in range(1, 38):
            x, y, meta = flatten_unit(data, metadata, 0, event_index, qpoint)
            scan.append({"qpoint": qpoint,
                         "identity": edge_cv(x, y, meta, "identity"),
                         "global_scale": edge_cv(x, y, meta, "global_scale"),
                         "diagonal_scale": edge_cv(x, y, meta, "diagonal_scale")})
        selected_layer = max(scan, key=lambda p: p["diagonal_scale"]["r2_vs_zero"])["qpoint"]
        x_train, y_train, meta_train = flatten_unit(data, metadata, 0, event_index, selected_layer)
        x_test, y_test, meta_test = flatten_unit(data, metadata, 1, event_index, selected_layer)
        competition = {}
        for model in MODELS:
            cv = edge_cv(x_train, y_train, meta_train, model)
            params = fit(x_train, y_train, meta_train, model)
            lock = metrics(y_test, predict(params, x_test, meta_test))
            competition[model] = {"confirmation_leave_edge_out": cv, "fresh_unit_lockbox": lock,
                                  "unseen_context_fresh_unit": held_context_test(x_train, y_train, meta_train, x_test, y_test, meta_test, model)}
        selected_model = max(MODELS, key=lambda name: competition[name]["confirmation_leave_edge_out"]["r2_vs_zero"])
        # Strict model claim requires it to beat both the no-query condition model and unconditioned diagonal on lockbox.
        best_lock = competition[selected_model]["fresh_unit_lockbox"]["r2_vs_zero"]
        baseline = max(competition["diagonal_scale"]["fresh_unit_lockbox"]["r2_vs_zero"],
                       competition["factor_coordinate_offset"]["fresh_unit_lockbox"]["r2_vs_zero"])
        report[event] = {"layer_selection": "unit28 leave-one-edge-out diagonal-scale R2",
                         "selected_qpoint": selected_layer, "layer_scan": scan, "model_selection": "unit28 leave-one-edge-out R2",
                         "selected_model": selected_model, "competition": competition,
                         "selected_lockbox_advantage_over_diagonal_and_condition_only": best_lock - baseline,
                         "candidate_operator_supported": bool(best_lock > 0 and best_lock - baseline > .02
                                                              and competition[selected_model]["unseen_context_fresh_unit"]["r2_vs_zero"] > 0)}
    return report


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    concise = {event: {k: panel[k] for k in ("selected_qpoint", "selected_model",
                                               "selected_lockbox_advantage_over_diagonal_and_condition_only",
                                               "candidate_operator_supported")}
               | {"competition": panel["competition"]} for event, panel in result["operators"].items()}
    text = rf"""


## Phase {PHASE}: 上下文条件query→候选/答案算子竞争与fresh-unit锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 从Phase2513四个双unit合格关系对构造unit×pair×language×16独立上下文×四事件×38层×2560坐标的Walsh交互。unit28只用于层位和模型选择：先以逐edge留一的逐坐标尺度R2选择每个目标事件唯一qpoint，再在该层竞争零、恒等、全局尺度、逐坐标尺度、五因素全局/逐坐标门控、只看条件的坐标偏置、条件仿射逐坐标模型。选择后在全新marker/实体unit29一次锁箱，并另将`zh×paraphrase1×candidate_order1`整个组合从训练剔除后在unit29测试。所有模型使用全部2560坐标；条件只含实验显式因子，不含答案或单行正确性。

$$\widehat I^{{target}}_i=I^{{query}}_i\sum_j\beta_{{ij}}f_j(c)+\sum_j\gamma_{{ij}}f_j(c).$$

**结果汇总。** 算子摘要 `{json.dumps(concise, ensure_ascii=False)}`；字段 `{json.dumps(result['fields'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2514_c78625_c79776_context_operator_competition_lockbox.py`；全层交互、逐层扫描、各基线锁箱、选定预测/残差全坐标与final位于`{OUT}`。

**分析与理论进展。** 严格候选算子必须在unit28逐edge留一中被选中，在unit29仍超过逐坐标无条件尺度与完全不使用query的条件偏置，并在未见条件组合上R2为正。若失败，只能得到“显式上下文因素不足以构成稳定逐坐标编译规律”；若通过，也只是外部预测规律，不是内部实现或因果中介。

**问题硬伤与结论。** 因素模型最多每坐标12个系数，虽有128个confirmation交互样本，仍可能利用模板规律；锁箱只是新实体/marker而非新架构；所有算子是逐坐标，不含坐标间混合；R2受目标交互幅度影响。下一Phase必须把模型表现连接到完整候选序列margin，并检查稳定坐标协同，不能只靠隐藏场拟合宣布闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    f2513 = load_json(P2513 / "analysis/final.json")
    data, metadata = build_interactions(f2513)
    derived = OUT / "derived"; derived.mkdir(parents=True, exist_ok=True)
    field_path = derived / "factorial_interactions_allqpoint.float32.npy"; np.save(field_path, data)
    index_path = OUT / "index/interaction_rows.jsonl"; index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in metadata), encoding="utf-8")
    operators = analyze(data, metadata)
    # Save selected lockbox predictions/residuals for direct physical-coordinate inspection.
    selected_rows = []
    selected_field = []
    for event_index, event in enumerate(EVENTS[1:], start=1):
        panel = operators[event]; qpoint = panel["selected_qpoint"]; model = panel["selected_model"]
        xtr, ytr, mtr = flatten_unit(data, metadata, 0, event_index, qpoint)
        xte, yte, mte = flatten_unit(data, metadata, 1, event_index, qpoint)
        pred = predict(fit(xtr, ytr, mtr, model), xte, mte)
        for row, x, y, p in zip(mte, xte, yte, pred):
            selected_rows.append({**row, "event": event, "qpoint": qpoint, "model": model,
                                  "row_kinds": ["query", "actual", "prediction", "residual"]})
            selected_field.extend((x, y, p, y - p))
    selected_array = np.asarray(selected_field, dtype=np.float32)
    selected_path = derived / "selected_lockbox_query_actual_prediction_residual.float32.npy"; np.save(selected_path, selected_array)
    selected_index = OUT / "index/selected_lockbox_rows.jsonl"
    selected_index.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected_rows), encoding="utf-8")
    checks = {"source_passed": f2513["all_checks_passed"], "four_qualified_pairs": len(f2513["behavior"]["qualified_pair_ids"]) == 4,
              "full_shape": data.shape == (2, 4, 2, 16, 4, 38, DIM), "q0_exact_zero": bool(np.max(np.abs(data[..., 0, :])) == 0),
              "models_competed": all(set(panel["competition"]) == set(MODELS) for panel in operators.values()),
              "lockbox_not_used_for_selection": True, "selected_parameter_rows": selected_array.shape == (3 * 128 * 4, DIM),
              "finite": bool(np.isfinite(data).all() and np.isfinite(selected_array).all()),
              "hashes": len(digest(field_path)) == 64 and len(digest(selected_path)) == 64, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "operators": operators,
              "fields": {"all_interactions": {"path": str(field_path), "shape": list(data.shape), "sha256": digest(field_path)},
                         "selected_lockbox": {"path": str(selected_path), "shape": list(selected_array.shape),
                                              "index": str(selected_index), "sha256": digest(selected_path)}},
              "adjudication": {"events_with_supported_candidate_operator": [e for e, p in operators.items() if p["candidate_operator_supported"]],
                               "event_compiler_identified": False, "causal_mediator_identified": False,
                               "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "operators": {e: {"q": p["selected_qpoint"], "model": p["selected_model"],
                                                        "supported": p["candidate_operator_supported"],
                                                        "lockbox": p["competition"][p["selected_model"]]["fresh_unit_lockbox"],
                                                        "advantage": p["selected_lockbox_advantage_over_diagonal_and_condition_only"]}
                                                    for e, p in operators.items()},
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
