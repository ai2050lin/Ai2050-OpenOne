#!/usr/bin/env python3
"""Adjudicate semantic relation structure against lexical and embedding explanations."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2390 = RESULT / "phase2390_c19441_c19760_qwen_semantic_lexical_fullfield"
OUT = RESULT / "phase2391_c19761_c20080_semantic_lexical_adjudication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2391
CAMPAIGN = "C19761-C20080"
MODELS = ("qwen4b", "qwen14b")
FAMILIES = ("preference", "taxonomy", "temporal", "causal", "comparison", "spatial", "role_binding", "ownership_transfer")
LANGUAGES = ("en", "zh")
EPS = 1e-6


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


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def field_paths(key: str) -> dict[str, Path]:
    base = P2390 / key
    return {"mean": base / "raw/independent_mean.float16.npy", "end": base / "raw/independent_end.float16.npy",
            "boundary": base / "raw/semantic_selection_prompt_boundary.float16.npy",
            "independent_rows": base / "index/independent_rows.jsonl", "selection_rows": base / "index/selection_rows.jsonl"}


def index_rows(rows: list[dict]) -> dict[tuple, int]:
    return {(row["group_id"], int(row["relation_bit"]), row["form"]): index for index, row in enumerate(rows)}


def fit_discriminants(values: np.ndarray, rows: list[dict], train_form: str = "canonical", shuffled: dict[int, int] | None = None) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    params = {}
    for family in FAMILIES:
        for language in LANGUAGES:
            chosen = [i for i, row in enumerate(rows) if row["partition"] == "discovery" and row["form"] == train_form
                      and row["family"] == family and row["language"] == language]
            labels = np.array([shuffled.get(i, int(rows[i]["relation_bit"])) if shuffled else int(rows[i]["relation_bit"]) for i in chosen])
            x = values[chosen]
            means = [x[labels == bit].mean(0) for bit in (0, 1)]
            pooled = np.sqrt(0.5 * (x[labels == 0].var(0) + x[labels == 1].var(0)))
            floor = max(float(np.median(pooled)) * 0.10, EPS); scale = np.maximum(pooled, floor)
            midpoint = 0.5 * (means[0] + means[1]); direction = (means[0] - means[1]) / scale
            params[(family, language)] = (midpoint.astype(np.float32), (direction / scale).astype(np.float32))
    return params


def predict(values: np.ndarray, rows: list[dict], params: dict, partition: str, form: str) -> tuple[float, list[dict]]:
    results = []
    for index, row in enumerate(rows):
        if row["partition"] != partition or row["form"] != form: continue
        midpoint, direction = params[(row["family"], row["language"])]
        score = float(np.dot(values[index] - midpoint, direction)); prediction = int(score < 0)
        results.append({"index": index, "group_id": row["group_id"], "family": row["family"], "language": row["language"],
                        "relation_bit": int(row["relation_bit"]), "prediction": prediction, "score": score, "correct": prediction == int(row["relation_bit"])})
    return float(np.mean([row["correct"] for row in results])), results


def evaluate(values: np.ndarray, rows: list[dict]) -> dict:
    params = fit_discriminants(values, rows)
    confirm_can, _ = predict(values, rows, params, "confirmation", "canonical")
    confirm_para, _ = predict(values, rows, params, "confirmation", "paraphrase")
    lock_can, lock_can_rows = predict(values, rows, params, "fresh_unit_lockbox", "canonical")
    lock_para, lock_para_rows = predict(values, rows, params, "fresh_unit_lockbox", "paraphrase")
    by_family = {}
    for family in FAMILIES:
        selected = [row for row in lock_para_rows if row["family"] == family]
        by_family[family] = float(np.mean([row["correct"] for row in selected]))
    return {"confirmation_canonical": confirm_can, "confirmation_cross_surface": confirm_para,
            "lockbox_same_bag_direction": lock_can, "lockbox_cross_surface": lock_para,
            "lockbox_cross_surface_by_family": by_family, "lockbox_rows": lock_para_rows, "samebag_rows": lock_can_rows}


def evaluate_boundary(values: np.ndarray, selection_rows: list[dict]) -> dict:
    rows = [{**row, "form": "canonical"} for row in selection_rows]
    params = fit_discriminants(values, rows)
    confirmation, _ = predict(values, rows, params, "confirmation", "canonical")
    lockbox, lock_rows = predict(values, rows, params, "fresh_unit_lockbox", "canonical")
    return {"confirmation_accuracy": confirmation, "lockbox_accuracy": lockbox,
            "lockbox_by_family": {family: float(np.mean([row["correct"] for row in lock_rows if row["family"] == family])) for family in FAMILIES}}


def diagonal_residual(deep: np.ndarray, embedding: np.ndarray, rows: list[dict]) -> np.ndarray:
    train = np.array([i for i, row in enumerate(rows) if row["partition"] == "discovery"], dtype=int)
    x, y = embedding[train], deep[train]; xm, ym = x.mean(0), y.mean(0); centered = x - xm
    a = ((centered * (y - ym)).sum(0) / np.maximum((centered * centered).sum(0), EPS)).astype(np.float32)
    b = (ym - a * xm).astype(np.float32)
    return (deep - (embedding * a + b)).astype(np.float32)


def bootstrap(correct_rows: list[dict], seed: int, repeats: int = 400) -> dict:
    groups = sorted({row["group_id"] for row in correct_rows}); by_group = {group: [row for row in correct_rows if row["group_id"] == group] for group in groups}
    rng = np.random.default_rng(seed); values = []
    for _ in range(repeats):
        sample = rng.choice(groups, size=len(groups), replace=True); selected = [row for group in sample for row in by_group[group]]
        values.append(float(np.mean([row["correct"] for row in selected])))
    return {"cluster": "family-language-unit group", "groups": len(groups), "repeats": repeats,
            "mean": float(np.mean(values)), "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}


def permutation_null(values: np.ndarray, rows: list[dict], observed: float, seed: int, repeats: int = 128) -> dict:
    rng = np.random.default_rng(seed); accuracies = []
    for _ in range(repeats):
        labels = {}
        for family in FAMILIES:
            for language in LANGUAGES:
                chosen = [i for i, row in enumerate(rows) if row["partition"] == "discovery" and row["form"] == "canonical"
                          and row["family"] == family and row["language"] == language]
                shuffled = rng.permutation([int(rows[i]["relation_bit"]) for i in chosen])
                labels.update({i: int(label) for i, label in zip(chosen, shuffled)})
        params = fit_discriminants(values, rows, shuffled=labels)
        accuracy, _ = predict(values, rows, params, "fresh_unit_lockbox", "paraphrase"); accuracies.append(accuracy)
    return {"repeats": repeats, "mean": float(np.mean(accuracies)), "q95": float(np.quantile(accuracies, .95)),
            "observed_exceedance_p": float((1 + sum(value >= observed for value in accuracies)) / (repeats + 1))}


def contrast_response(values: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, dict]:
    lookup = index_rows(rows); response = np.empty((len(FAMILIES), len(LANGUAGES), 2, values.shape[1]), dtype=np.float32)
    for fi, family in enumerate(FAMILIES):
        for li, language in enumerate(LANGUAGES):
            groups = sorted({row["group_id"] for row in rows if row["family"] == family and row["language"] == language and row["partition"] == "discovery"})
            for form_index, form in enumerate(("canonical", "paraphrase")):
                response[fi, li, form_index] = np.stack([values[lookup[(group, 0, form)]] - values[lookup[(group, 1, form)]] for group in groups]).mean(0)
    cosine, sign = [], []
    for fi in range(len(FAMILIES)):
        for li in range(len(LANGUAGES)):
            a, b = response[fi, li, 0], response[fi, li, 1]
            cosine.append(float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS)))
            sign.append(float(np.mean(np.sign(a) == np.sign(b))))
    return response, {"mean_cross_surface_cosine": float(np.mean(cosine)), "family_language_cosines": cosine,
                      "mean_coordinate_sign_agreement": float(np.mean(sign)), "family_language_sign_agreement": sign}


def fit_diagonal(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm, ym = source.mean(0), target.mean(0); centered = source - xm
    a = (centered * (target - ym)).sum(0) / np.maximum((centered * centered).sum(0), EPS); return a.astype(np.float32), (ym - a * xm).astype(np.float32)


def bridge_accuracy(source: np.ndarray, boundary: np.ndarray, independent_rows: list[dict], selection_rows: list[dict], partition: str, form: str) -> float:
    lookup = index_rows(independent_rows); params = {}
    for family in FAMILIES:
        for language in LANGUAGES:
            indices = [i for i,row in enumerate(selection_rows) if row["partition"] == "discovery" and row["family"] == family and row["language"] == language]
            x = np.stack([source[lookup[(selection_rows[i]["group_id"], int(selection_rows[i]["relation_bit"]), "canonical")]] for i in indices])
            params[(family, language)] = fit_diagonal(x, boundary[indices])
    correct = total = 0
    for index, row in enumerate(selection_rows):
        if row["partition"] != partition: continue
        a, b = params[(row["family"], row["language"])]
        candidates = np.stack([source[lookup[(row["group_id"], bit, form)]] for bit in (0, 1)])
        errors = np.square(candidates * a + b - boundary[index]).mean(1); choice = int(errors.argmin())
        correct += int(choice == int(row["relation_bit"])); total += 1
    return correct / total


def analyze_model(key: str) -> dict:
    p = field_paths(key); rows = read_rows(p["independent_rows"]); selection_rows = read_rows(p["selection_rows"])
    fields = {"mean": np.load(p["mean"], mmap_mode="r"), "end": np.load(p["end"], mmap_mode="r")}; boundary_map = np.load(p["boundary"], mmap_mode="r")
    layer_scan = []
    for field_name, field in fields.items():
        for qpoint in range(field.shape[1]):
            result = evaluate(np.asarray(field[:, qpoint], dtype=np.float32), rows)
            layer_scan.append({"field": field_name, "qpoint": qpoint, **{k:v for k,v in result.items() if not k.endswith("_rows")}})
    selected = max(layer_scan, key=lambda item: (item["confirmation_cross_surface"], item["confirmation_canonical"], -item["qpoint"], item["field"] == "mean"))
    field = fields[selected["field"]]; values = np.asarray(field[:, selected["qpoint"]], dtype=np.float32); evaluated = evaluate(values, rows)
    embedding = np.asarray(fields["mean"][:, 0], dtype=np.float32); embedding_eval = evaluate(embedding, rows)
    residual = diagonal_residual(values, embedding, rows); residual_eval = evaluate(residual, rows)
    null = permutation_null(values, rows, evaluated["lockbox_cross_surface"], PHASE + (0 if key == "qwen4b" else 100))
    response, response_stats = contrast_response(values, rows)
    derived = OUT / key / "derived"; derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "selected_relation_response.float32.npy", response.astype(np.float32), allow_pickle=False)
    np.save(derived / "selected_embedding_residual.float16.npy", residual.astype(np.float16), allow_pickle=False)
    # Select output checkpoint only from confirmation target-bit decoding.
    output_layers = []
    for qpoint in range(boundary_map.shape[1]):
        output_layers.append({"qpoint": qpoint, **evaluate_boundary(np.asarray(boundary_map[:, qpoint], dtype=np.float32), selection_rows)})
    output_selected = max(output_layers, key=lambda item: (item["confirmation_accuracy"], -item["qpoint"]))
    boundary = np.asarray(boundary_map[:, output_selected["qpoint"]], dtype=np.float32)
    bridge = {"selected_source_to_output": bridge_accuracy(values, boundary, rows, selection_rows, "fresh_unit_lockbox", selected["field"] == "mean" and "paraphrase" or "paraphrase"),
              "selected_source_canonical": bridge_accuracy(values, boundary, rows, selection_rows, "fresh_unit_lockbox", "canonical"),
              "embedding_paraphrase": bridge_accuracy(embedding, boundary, rows, selection_rows, "fresh_unit_lockbox", "paraphrase"),
              "source_qpoint": selected["qpoint"], "source_field": selected["field"], "output_qpoint": output_selected["qpoint"]}
    family_pass = sum(value >= .55 for value in evaluated["lockbox_cross_surface_by_family"].values())
    result = {"analysis_version": 3, "model": key, "layer_scan": layer_scan, "selected": {**selected,
        "lockbox_same_bag_direction": evaluated["lockbox_same_bag_direction"], "lockbox_cross_surface": evaluated["lockbox_cross_surface"],
        "lockbox_cross_surface_by_family": evaluated["lockbox_cross_surface_by_family"],
        "embedding_cross_surface": embedding_eval["lockbox_cross_surface"], "deep_gain_over_embedding": evaluated["lockbox_cross_surface"] - embedding_eval["lockbox_cross_surface"],
        "embedding_residual_cross_surface": residual_eval["lockbox_cross_surface"], "family_pass_count": family_pass,
        "bootstrap": bootstrap(evaluated["lockbox_rows"], PHASE), "permutation_null": null, "response_stability": response_stats},
        "output_layer_scan": output_layers, "output_selected": output_selected, "bridge": bridge,
        "semantic_relation_supported": evaluated["lockbox_cross_surface"] >= .65 and evaluated["lockbox_cross_surface"] - embedding_eval["lockbox_cross_surface"] >= .05 and family_pass >= 6,
        "limits": ["entity words remain shared across surfaces", "selection over checkpoints uses confirmation", "four relation templates per language are not a complete language theory"]}
    for item in fields.values(): close(item)
    close(boundary_map); save(OUT / key / "analysis/final.json", result); return result


def append_memo(result: dict) -> None:
    memo_text = MEMO.read_text(encoding="utf-8")
    if f"## Phase {PHASE}:" in memo_text:
        marker = "**执行修正（Phase2391输出层空集合）**"
        if result.get("analysis_version", 0) >= 2 and marker not in memo_text:
            with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("\n\n**执行修正（Phase2391输出层空集合）**：首次输出层扫描复用了独立句的双form汇总函数，"
                             "而句前场只有一种form，因而产生不参与选择的空paraphrase指标NaN。已保留失败痕迹并改为只汇总"
                             f"confirmation/lockbox canonical方向解码；严格修正后的比较 `{json.dumps(result['comparison'], ensure_ascii=False)}`，"
                             f"裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`。\n")
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 同词反关系—异表达同关系的全坐标锁箱裁决（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 每个关系族/语言用discovery canonical拟合全坐标方差缩放质心方向，在confirmation选择独立句field/checkpoint，fresh-unit锁箱同时测试：(1) canonical同主要词汇、相反角色/顺序；(2) paraphrase异句法同关系的跨表面泛化。以embedding token均值q0为冻结基线，并将embedding对深层每个坐标的对角预测扣除后复测残差。对锁箱按family-language-unit聚类bootstrap 400次，并做128次分层训练标签置乱。另冻结输出句前checkpoint，比较canonical/paraphrase独立句到句前场的单一对角映射桥。

$$s(x)=\left(x-\frac{{\mu_0+\mu_1}}{{2}}\right)^\top\Sigma_{{diag}}^{{-1}}(\mu_0-\mu_1),$$

$$R_q=H_q-(a_q\odot H_0+b_q),\qquad
\Delta_{{deep}}=Acc_{{deep,para}}-Acc_{{embed,para}}.$$

**结果汇总。** 模型比较 `{json.dumps(result['comparison'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2391_c19761_c20080_semantic_lexical_adjudication.py`；逐checkpoint结果、bootstrap/null和全物理坐标关系响应位于 `tests/glm5/result/phase2391_c19761_c20080_semantic_lexical_adjudication`。

**理论进展、问题硬伤与结论。** 同词反关系排除了纯bag-of-words均值解释；跨表面关系泛化若超过embedding，才支持深层关系响应候选。即使通过，也只能说明冻结读出可复用，不等于模型运行时执行了该线性判别或仿射桥。若总体通过但个别族失败，理论必须保留族差异；若残差失败，则深层优势仍可由embedding可预测部分解释。下一Phase只围绕这里重复通过的响应建立坐标指纹和动态，不用Top-K替代全场结论。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        cached = json.loads(final.read_text(encoding="utf-8"))
        if cached.get("analysis_version") == 3:
            append_memo(cached); print(json.dumps(cached, ensure_ascii=False, indent=2)); return
    models = {key: analyze_model(key) for key in MODELS}
    comparison = {key: {"field": value["selected"]["field"], "qpoint": value["selected"]["qpoint"],
        "same_bag_direction": value["selected"]["lockbox_same_bag_direction"], "cross_surface": value["selected"]["lockbox_cross_surface"],
        "embedding": value["selected"]["embedding_cross_surface"], "deep_gain": value["selected"]["deep_gain_over_embedding"],
        "residual": value["selected"]["embedding_residual_cross_surface"], "families_ge_055": value["selected"]["family_pass_count"],
        "bridge": value["bridge"], "semantic_supported": value["semantic_relation_supported"]} for key,value in models.items()}
    adjudication = {"universal_semantic_relation_field": all(value["semantic_relation_supported"] for value in models.values()),
                    "supported_models": [key for key,value in models.items() if value["semantic_relation_supported"]],
                    "deep_semantic_relation_not_supported": all(not value["semantic_relation_supported"] for value in models.values()),
                    "prior_identity_matching_likely_lexically_dominated": all(value["selected"]["deep_gain_over_embedding"] < .05 for value in models.values()),
                    "contextual_relation_signal_candidate": any(value["output_selected"]["lockbox_accuracy"] >= .65 for value in models.values()),
                    "claim_boundary": "cross-surface relation readout, not autonomous internal gear or complete semantic representation"}
    checks = {"two_models": set(models) == set(MODELS), "finite": all(math.isfinite(value["selected"]["lockbox_cross_surface"]) for value in models.values()),
              "full_coordinate_response": all((OUT / key / "derived/selected_relation_response.float32.npy").exists() for key in MODELS)}
    result = {"analysis_version": 3, "phase": PHASE, "campaign": CAMPAIGN, "models": models, "comparison": comparison, "adjudication": adjudication,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
