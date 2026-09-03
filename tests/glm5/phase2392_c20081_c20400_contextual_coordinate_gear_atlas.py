#!/usr/bin/env python3
"""Build an all-coordinate contextual relation-response atlas and generation bridge."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2389 = RESULT / "phase2389_c19121_c19440_crossmodel_autonomous_capability"
P2390 = RESULT / "phase2390_c19441_c19760_qwen_semantic_lexical_fullfield"
P2391 = RESULT / "phase2391_c19761_c20080_semantic_lexical_adjudication"
OUT = RESULT / "phase2392_c20081_c20400_contextual_coordinate_gear_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2392
CAMPAIGN = "C20081-C20400"
MODELS = ("qwen4b", "qwen14b")
FAMILIES = ("preference", "taxonomy", "temporal", "causal", "comparison", "spatial", "role_binding", "ownership_transfer")
LANGUAGES = ("en", "zh")
PARTITIONS = ("discovery", "confirmation", "fresh_unit_lockbox")
EPS = 1e-8

sys.path.insert(0, str(TESTS))
import phase2391_c19761_c20080_semantic_lexical_adjudication as adjudicate  # noqa: E402


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


def relation_response(field: np.ndarray, rows: list[dict]) -> np.ndarray:
    result = np.empty((field.shape[1], len(PARTITIONS), len(FAMILIES), len(LANGUAGES), field.shape[2]), dtype=np.float32)
    for qpoint in range(field.shape[1]):
        values = np.asarray(field[:, qpoint], dtype=np.float32)
        for pi, partition in enumerate(PARTITIONS):
            for fi, family in enumerate(FAMILIES):
                for li, language in enumerate(LANGUAGES):
                    groups = [i for i,row in enumerate(rows) if row["partition"] == partition and row["family"] == family and row["language"] == language]
                    bit0 = values[[i for i in groups if int(rows[i]["relation_bit"]) == 0]]
                    bit1 = values[[i for i in groups if int(rows[i]["relation_bit"]) == 1]]
                    result[qpoint, pi, fi, li] = bit0.mean(0) - bit1.mean(0)
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def layer_dynamics(response: np.ndarray, field: np.ndarray, rows: list[dict]) -> list[dict]:
    dynamics = []
    for qpoint in range(response.shape[0]):
        disc, conf, lock = response[qpoint, 0], response[qpoint, 1], response[qpoint, 2]
        confirm_cos = [cosine(disc[fi,li], conf[fi,li]) for fi in range(len(FAMILIES)) for li in range(len(LANGUAGES))]
        lock_cos = [cosine(disc[fi,li], lock[fi,li]) for fi in range(len(FAMILIES)) for li in range(len(LANGUAGES))]
        language_cos = [cosine(disc[fi,0], disc[fi,1]) for fi in range(len(FAMILIES))]
        decoded = adjudicate.evaluate_boundary(np.asarray(field[:, qpoint], dtype=np.float32), rows)
        norm = float(np.mean(np.linalg.norm(disc, axis=-1) / math.sqrt(disc.shape[-1])))
        dynamics.append({"qpoint": qpoint, "response_rms": norm, "confirmation_response_cosine": float(np.mean(confirm_cos)),
                         "lockbox_response_cosine": float(np.mean(lock_cos)), "cross_language_cosine": float(np.mean(language_cos)),
                         "confirmation_accuracy": decoded["confirmation_accuracy"], "lockbox_accuracy": decoded["lockbox_accuracy"]})
    increments = [0.0] + [dynamics[i]["response_rms"] - dynamics[i-1]["response_rms"] for i in range(1, len(dynamics))]
    for item, increment in zip(dynamics, increments): item["response_rms_increment"] = increment
    return dynamics


def group_signature(response: np.ndarray) -> tuple[np.ndarray, dict]:
    # Average languages, retain all coordinates, and group by eight-family sign signatures.
    family = response.mean(1); scale = np.median(np.abs(family), axis=1, keepdims=True) + EPS
    normalized = (family / scale).T.astype(np.float32)
    bits = (normalized > 0).astype(np.int32); ids = sum(bits[:, fi] << fi for fi in range(bits.shape[1])).astype(np.int32)
    counts = Counter(ids.tolist()); largest = [{"group_id": int(group), "coordinates": int(count)} for group,count in counts.most_common(20)]
    return np.concatenate((normalized, np.abs(normalized)), axis=1), {"groups_present": len(counts), "largest_groups": largest,
        "definition": "eight-family discovery response sign; grouping is descriptive and not a semantic ontology"}, ids


def subset_tournament(values: np.ndarray, rows: list[dict], disc: np.ndarray, conf: np.ndarray, seed: int) -> dict:
    mean_abs = np.mean(np.abs(disc), axis=(0,1)); sign_stability = np.mean(np.sign(disc) == np.sign(conf), axis=(0,1))
    score = mean_abs * (0.25 + sign_stability); order = np.argsort(-score); dimension = values.shape[1]
    fractions = (1.0, .75, .50, .25, .10, .05, .01); results = []
    for fraction in fractions:
        count = dimension if fraction == 1.0 else max(8, int(round(dimension * fraction))); coords = order[:count]
        evaluated = adjudicate.evaluate_boundary(values[:, coords], rows)
        results.append({"fraction": fraction, "coordinates": count, "confirmation": evaluated["confirmation_accuracy"], "lockbox": evaluated["lockbox_accuracy"]})
    rng = np.random.default_rng(seed); count = max(8, int(round(dimension * .10))); random_scores = []
    for _ in range(32):
        coords = rng.choice(dimension, size=count, replace=False); random_scores.append(adjudicate.evaluate_boundary(values[:, coords], rows)["lockbox_accuracy"])
    return {"primary": "all coordinates", "rank_basis": "discovery magnitude times confirmation sign stability", "nested": results,
            "random_10pct": {"repeats": 32, "mean": float(np.mean(random_scores)), "q95": float(np.quantile(random_scores, .95))},
            "rank_score_path": None, "order": order, "score": score}


def fit_binary(values: np.ndarray, rows: list[dict], train: list[int]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([int(rows[i]["relation_bit"]) for i in train]); x = values[train]; means = [x[labels == bit].mean(0) for bit in (0,1)]
    variance = .5 * (x[labels == 0].var(0) + x[labels == 1].var(0)); floor = max(float(np.median(variance)) * .01, EPS)
    return (.5 * (means[0] + means[1])).astype(np.float32), ((means[0] - means[1]) / np.maximum(variance, floor)).astype(np.float32)


def cross_condition_generalization(values: np.ndarray, rows: list[dict]) -> dict:
    cross_language = {}; heldout_family = {}
    for family in FAMILIES:
        directions = []
        for train_language, test_language in (("en","zh"),("zh","en")):
            train = [i for i,r in enumerate(rows) if r["partition"] == "discovery" and r["family"] == family and r["language"] == train_language]
            test = [i for i,r in enumerate(rows) if r["partition"] == "fresh_unit_lockbox" and r["family"] == family and r["language"] == test_language]
            midpoint, direction = fit_binary(values, rows, train); predictions = [int(np.dot(values[i]-midpoint,direction) < 0) for i in test]
            directions.append(float(np.mean([prediction == int(rows[i]["relation_bit"]) for prediction,i in zip(predictions,test)])))
        cross_language[family] = float(np.mean(directions))
        train = [i for i,r in enumerate(rows) if r["partition"] == "discovery" and r["family"] != family]
        test = [i for i,r in enumerate(rows) if r["partition"] == "fresh_unit_lockbox" and r["family"] == family]
        midpoint, direction = fit_binary(values, rows, train); predictions = [int(np.dot(values[i]-midpoint,direction) < 0) for i in test]
        heldout_family[family] = float(np.mean([prediction == int(rows[i]["relation_bit"]) for prediction,i in zip(predictions,test)]))
    return {"cross_language_by_family": cross_language, "cross_language_mean": float(np.mean(list(cross_language.values()))),
            "heldout_family": heldout_family, "heldout_family_mean": float(np.mean(list(heldout_family.values())))}


def behavior_bridge(key: str, values: np.ndarray, rows: list[dict]) -> dict:
    params = adjudicate.fit_discriminants(values, [{**row, "form": "canonical"} for row in rows])
    _, lock = adjudicate.predict(values, [{**row, "form": "canonical"} for row in rows], params, "fresh_unit_lockbox", "canonical")
    confidence = {rows[item["index"]]["case_id"]: item["score"] * (1 if item["relation_bit"] == 0 else -1) for item in lock}
    decisions = np.load(P2390 / key / "raw/semantic_selection_sequence_scores.float32.npy", mmap_mode="r")
    lock_indices = [i for i,row in enumerate(rows) if row["partition"] == "fresh_unit_lockbox"]
    x = np.array([confidence[rows[i]["case_id"]] for i in lock_indices]); y = np.asarray(decisions[lock_indices,2], dtype=np.float32)
    corr = float(np.corrcoef(x,y)[0,1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0; close(decisions)
    generated = read_rows(P2389 / key / "generation/semantic_selection.jsonl")
    success = {row["case_id"]: bool(row["target_first_line_exact"]) for row in generated}
    good = [value for case,value in confidence.items() if success.get(case, False)]; bad = [value for case,value in confidence.items() if case in success and not success[case]]
    return {"lockbox_rows": len(lock_indices), "confidence_logprob_margin_correlation": corr,
            "autonomous_success_rows": len(good), "autonomous_failure_rows": len(bad),
            "mean_confidence_success": float(np.mean(good)) if good else None, "mean_confidence_failure": float(np.mean(bad)) if bad else None,
            "success_minus_failure": float(np.mean(good)-np.mean(bad)) if good and bad else None}


def analyze_model(key: str) -> dict:
    source = json.loads((P2391 / key / "analysis/final.json").read_text(encoding="utf-8")); qpoint = int(source["output_selected"]["qpoint"])
    base = P2390 / key; rows = read_rows(base / "index/selection_rows.jsonl"); field = np.load(base / "raw/semantic_selection_prompt_boundary.float16.npy", mmap_mode="r")
    response = relation_response(field, rows); values = np.asarray(field[:,qpoint], dtype=np.float32); dynamics = layer_dynamics(response, field, rows)
    selected_response = response[qpoint,0]; fingerprints, group_summary, group_ids = group_signature(selected_response)
    disc, conf, lock = response[qpoint,0], response[qpoint,1], response[qpoint,2]
    tournament = subset_tournament(values, rows, disc, conf, PHASE + (0 if key == "qwen4b" else 100)); order, score = tournament.pop("order"), tournament.pop("score")
    derived = OUT / key / "derived"; derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "all_layer_partition_relation_response.float32.npy", response, allow_pickle=False)
    np.save(derived / "selected_coordinate_fingerprint.float32.npy", fingerprints, allow_pickle=False)
    np.save(derived / "selected_coordinate_group_ids.int32.npy", group_ids, allow_pickle=False)
    np.save(derived / "confirmation_frozen_coordinate_rank.int32.npy", order.astype(np.int32), allow_pickle=False)
    np.save(derived / "confirmation_frozen_coordinate_score.float32.npy", score.astype(np.float32), allow_pickle=False)
    family_mean = selected_response.mean(1); reuse = np.array([[cosine(family_mean[i],family_mean[j]) for j in range(len(FAMILIES))] for i in range(len(FAMILIES))], dtype=np.float32)
    np.save(derived / "family_response_cosine.float32.npy", reuse, allow_pickle=False)
    response_stability = {"confirmation_cosine": float(np.mean([cosine(disc[fi,li],conf[fi,li]) for fi in range(8) for li in range(2)])),
                          "lockbox_cosine": float(np.mean([cosine(disc[fi,li],lock[fi,li]) for fi in range(8) for li in range(2)])),
                          "cross_language_cosine": float(np.mean([cosine(disc[fi,0],disc[fi,1]) for fi in range(8)])),
                          "coordinate_sign_confirmation": float(np.mean(np.sign(disc)==np.sign(conf))),
                          "coordinate_sign_lockbox": float(np.mean(np.sign(disc)==np.sign(lock)))}
    generalization = cross_condition_generalization(values, rows); bridge = behavior_bridge(key, values, rows)
    change = max(dynamics[1:], key=lambda item: item["response_rms_increment"])
    result = {"model": key, "selected_qpoint": qpoint, "full_coordinate_primary_lockbox": source["output_selected"]["lockbox_accuracy"],
              "full_coordinate_by_family": source["output_selected"]["lockbox_by_family"], "response_stability": response_stability,
              "layer_dynamics": dynamics, "largest_positive_change": change, "coordinate_groups": group_summary,
              "subset_diagnostic": tournament, "generalization": generalization, "behavior_bridge": bridge,
              "flagships": {"preference_like_like_apples": source["output_selected"]["lockbox_by_family"]["preference"],
                            "taxonomy_like_apple_fruit": source["output_selected"]["lockbox_by_family"]["taxonomy"]},
              "claim_boundary": "context-conditioned coordinate response atlas; not a causal gear, semantic neuron list, or ontology"}
    close(field); save(OUT / key / "analysis/final.json", result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 上下文关系响应的单坐标—坐标群—层轨迹图谱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2391否决静态独立句语义场后，转向真正行为阳性的句前上下文场。对每个checkpoint、partition、8关系族、双语和全部物理坐标计算bit0−bit1响应；confirmation冻结checkpoint和逐坐标排序，fresh-unit锁箱裁决。全坐标是主结果；之后才以100%→1%嵌套坐标群和32个随机10%组诊断冗余。保存每坐标的16维family-language方向指纹、八族符号群、八族复用矩阵、层间出现/增强轨迹。另做跨语言、留一关系族外推，并把冻结判别置信度连接到整句logprob margin与自主生成成功/失败。

$$R_{{q,f,\ell,j}}=\mathbb E[H_{{q,j}}|d=0,f,\ell]-\mathbb E[H_{{q,j}}|d=1,f,\ell],$$

$$r_j=[R_{{f_1,en,j}},R_{{f_1,zh,j}},\ldots,R_{{f_8,zh,j}}].$$

**结果汇总。** 双模型图谱 `{json.dumps(result['summary'], ensure_ascii=False)}`；普遍性裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2392_c20081_c20400_contextual_coordinate_gear_atlas.py`；全层/全partition/全坐标响应、坐标指纹、群ID、复用矩阵与分析位于 `tests/glm5/result/phase2392_c20081_c20400_contextual_coordinate_gear_atlas`。

**理论进展、问题硬伤与结论。** 关系信息在query和两个反关系候选共同存在的上下文中形成，不能倒写成独立句的静态语义向量。跨语言或留族外推失败意味着响应是族/表面条件化的，不是普适方向齿轮。坐标子群只诊断冗余和协同，不能用少量坐标覆盖全场结论；生成相关性也不是因果性。只有confirmation冻结后仍跨unit稳定、并连接行为的候选进入下一Phase定点干预。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result=json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result,ensure_ascii=False,indent=2)); return
    models = {key: analyze_model(key) for key in MODELS}
    summary = {key: {"qpoint": value["selected_qpoint"], "full_coordinate_lockbox": value["full_coordinate_primary_lockbox"],
        "response_stability": value["response_stability"], "generalization": value["generalization"],
        "behavior_bridge": value["behavior_bridge"], "subset_diagnostic": value["subset_diagnostic"], "flagships": value["flagships"]} for key,value in models.items()}
    adjudication = {"contextual_relation_signal_replicated": all(value["full_coordinate_primary_lockbox"] >= .70 for value in models.values()),
                    "universal_cross_language_direction": all(value["generalization"]["cross_language_mean"] >= .65 for value in models.values()),
                    "unseen_family_operator": all(value["generalization"]["heldout_family_mean"] >= .65 for value in models.values()),
                    "generation_bridge_replicated": all((value["behavior_bridge"]["success_minus_failure"] or 0) > 0 for value in models.values()),
                    "theory": "contextual relation responses are family-conditioned distributed fields unless cross-condition gates pass"}
    checks = {"two_models": set(models)==set(MODELS), "all_coordinate_arrays": all((OUT/key/"derived/all_layer_partition_relation_response.float32.npy").exists() for key in MODELS),
              "finite_primary": all(math.isfinite(value["full_coordinate_primary_lockbox"]) for value in models.values())}
    result={"phase":PHASE,"campaign":CAMPAIGN,"models":models,"summary":summary,"adjudication":adjudication,"checks":checks,"all_checks_passed":all(checks.values())}
    save(final,result); append_memo(result)
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
