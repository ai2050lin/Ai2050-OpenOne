#!/usr/bin/env python3
"""Compete full-coordinate conditional-equivalence routes on prompt and generation fields."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
P2352 = RESULT / "phase2352_c9241_c9400_natural_multifuture_transient_field"
OUT = RESULT / "phase2353_c9401_c9560_conditional_equivalence_route_competition"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"; VIS = ROOT / "frontend/public/vis_data/research_kernel"
STATES = P2352 / "raw/prompt_boundary_all_checkpoints.float16.npy"
GEN_BINARY = VIS / "c9242_qwen4b_natural_generation_token_trajectory.float16.npy"
PHASE = 2353; CAMPAIGN = "C9401-C9560"; EPS = 1e-8
FACTORS = ("language", "surface", "query", "depth", "state")
TRANSFERS = (("language", "en", "zh"), ("language", "zh", "en"),
             ("surface", "direct", "natural"), ("surface", "natural", "direct"),
             ("query", "source", "terminal"), ("query", "terminal", "source"),
             ("query", "first", "penultimate"), ("query", "penultimate", "first"),
             ("depth", 2, 5), ("depth", 5, 2), ("depth", 3, 4), ("depth", 4, 3),
             ("state", 0, 1), ("state", 1, 0))

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2352_c9241_c9400_natural_multifuture_transient_field as source  # noqa: E402

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def fit_residual(field: np.ndarray, rows: list[dict], train_mask: np.ndarray, factors: tuple[str, ...]) -> np.ndarray:
    field64 = field.astype(np.float64); center = field64[train_mask].mean(axis=0)
    out = field64 - center
    for factor in factors:
        levels = sorted({str(row[factor]) for i, row in enumerate(rows) if train_mask[i]})
        effects = {}
        for level in levels:
            idx = np.asarray([i for i, row in enumerate(rows) if train_mask[i] and str(row[factor]) == level])
            effects[level] = field64[idx].mean(axis=0) - center
        for i, row in enumerate(rows):
            effect = effects.get(str(row[factor]))
            if effect is not None: out[i] -= effect
    return out.astype(np.float32)


def row_normalize(field: np.ndarray) -> np.ndarray:
    return (field / np.maximum(np.linalg.norm(field, axis=1, keepdims=True), EPS)).astype(np.float32)


def classify(prototypes: np.ndarray, actual: np.ndarray, correct: np.ndarray) -> dict:
    distances = np.maximum(np.sum(actual * actual, axis=1, keepdims=True) + np.sum(prototypes * prototypes, axis=1)[None, :]
                           - 2 * actual @ prototypes.T, 0)
    order = np.argsort(distances, axis=1); hit = order[:, 0] == correct
    own = distances[np.arange(len(correct)), correct]
    nearest_wrong = np.min(np.where(np.eye(len(prototypes), dtype=bool)[correct], np.inf, distances), axis=1)
    return {"rows": len(correct), "accuracy": float(np.mean(hit)),
            "median_distance_ratio": float(np.median(own / np.maximum(nearest_wrong, EPS)))}


def transfer(field: np.ndarray, rows: list[dict], labels: list[str], factor: str, source_value: Any,
             target_value: Any, partition: str) -> dict:
    prototypes = []
    for label in labels:
        idx = [i for i, row in enumerate(rows) if row["family"] == label and row["partition"] in ("discovery", "confirmation")
               and row[factor] == source_value]
        prototypes.append(field[idx].mean(axis=0, dtype=np.float64))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["family"] in labels and row["partition"] == partition and row[factor] == target_value:
            groups[(row["family"], row["unit"])].append(i)
    keys = sorted(groups); actual = np.stack([field[groups[key]].mean(axis=0, dtype=np.float64) for key in keys])
    return classify(np.stack(prototypes), actual, np.asarray([labels.index(key[0]) for key in keys]))


def evaluate(field: np.ndarray, rows: list[dict], labels: list[str], partition: str) -> dict:
    values = {f"{factor}:{a}->{b}": transfer(field, rows, labels, factor, a, b, partition) for factor, a, b in TRANSFERS}
    return {"transfers": values, "minimum_accuracy": min(v["accuracy"] for v in values.values()),
            "mean_accuracy": float(np.mean([v["accuracy"] for v in values.values()])),
            "maximum_distance_ratio": max(v["median_distance_ratio"] for v in values.values())}


def prompt_competition(rows: list[dict], labels: list[str]) -> tuple[dict, dict[str, np.ndarray]]:
    states = np.load(STATES, mmap_mode="r"); train = np.asarray([r["partition"] in ("discovery", "confirmation") for r in rows])
    trajectory = []; candidates = []
    for qpoint in range(states.shape[1]):
        signed = states[:, qpoint].astype(np.float32); absolute = np.abs(signed)
        route_fields = {
            "absolute_hidden": absolute,
            "row_normalized_absolute_hidden": row_normalize(absolute),
            "factorial_residual_absolute_hidden": fit_residual(absolute, rows, train, FACTORS),
            "factorial_residual_signed_hidden": fit_residual(signed, rows, train, FACTORS),
        }
        record = {"qpoint": qpoint, "routes": {}}
        for name, field in route_fields.items():
            score = evaluate(field, rows, labels, "fresh_confirmation")
            record["routes"][name] = {k: v for k, v in score.items() if k != "transfers"}
            candidates.append((score["minimum_accuracy"], -score["maximum_distance_ratio"], qpoint, name))
        trajectory.append(record)
    _, _, qpoint, route_name = max(candidates)
    signed = states[:, qpoint].astype(np.float32); absolute = np.abs(signed)
    route_fields = {"absolute_hidden": absolute, "row_normalized_absolute_hidden": row_normalize(absolute),
                    "factorial_residual_absolute_hidden": fit_residual(absolute, rows, train, FACTORS),
                    "factorial_residual_signed_hidden": fit_residual(signed, rows, train, FACTORS)}
    selected_field = route_fields[route_name]; sorted_field = np.sort(selected_field, axis=1)
    selected = evaluate(selected_field, rows, labels, "fresh_confirmation")
    lockbox = evaluate(selected_field, rows, labels, "fresh_lockbox"); sorted_lockbox = evaluate(sorted_field, rows, labels, "fresh_lockbox")
    result = {"selection_trajectory": trajectory, "selected_qpoint": qpoint, "selected_route": route_name,
              "selected": selected, "lockbox": lockbox, "sorted_lockbox": sorted_lockbox,
              "coordinate_advantage": lockbox["minimum_accuracy"] - sorted_lockbox["minimum_accuracy"]}
    fields = {"selected_route": selected_field, "row_sorted_control": sorted_field,
              "absolute_hidden_control": absolute, "signed_hidden_control": signed}
    close_memmap(states); return result, fields


def gen_transfer(field: np.ndarray, rows: list[dict], labels: list[str], factor: str, a: Any, b: Any, split: str) -> dict:
    train_units = (12, 13); eval_unit = 14 if split == "selection" else 15
    prototypes = []
    for label in labels:
        idx = [i for i, r in enumerate(rows) if r["family"] == label and r["unit"] in train_units and r[factor] == a]
        prototypes.append(field[idx].mean(axis=0, dtype=np.float64))
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["family"] in labels and r["unit"] == eval_unit and r[factor] == b: groups[r["family"]].append(i)
    keys = sorted(groups); actual = np.stack([field[groups[k]].mean(axis=0, dtype=np.float64) for k in keys])
    return classify(np.stack(prototypes), actual, np.asarray([labels.index(k) for k in keys]))


def gen_evaluate(field: np.ndarray, rows: list[dict], labels: list[str], split: str) -> dict:
    axes = (("language", "en", "zh"), ("language", "zh", "en"), ("query", "source", "terminal"),
            ("query", "terminal", "source"), ("query", "first", "penultimate"), ("query", "penultimate", "first"))
    values = {f"{f}:{a}->{b}": gen_transfer(field, rows, labels, f, a, b, split) for f, a, b in axes}
    return {"transfers": values, "minimum_accuracy": min(v["accuracy"] for v in values.values()),
            "mean_accuracy": float(np.mean([v["accuracy"] for v in values.values()])),
            "maximum_distance_ratio": max(v["median_distance_ratio"] for v in values.values())}


def generation_competition(labels: list[str]) -> tuple[dict, dict[str, np.ndarray]]:
    rows = io.read_rows(P2352 / "raw/fresh_lockbox_generation.jsonl")
    shape = (384, 12, 38, 2560); source_field = np.load(GEN_BINARY, mmap_mode="r").reshape(shape)
    train = np.asarray([r["unit"] in (12, 13) for r in rows]); factors = ("language", "query")
    candidates = []; trajectory = []
    for step in range(shape[1]):
        for qpoint in range(shape[2]):
            absolute = np.abs(source_field[:, step, qpoint].astype(np.float32))
            residual = fit_residual(absolute, rows, train, factors)
            score = gen_evaluate(residual, rows, labels, "selection")
            candidates.append((score["minimum_accuracy"], -score["maximum_distance_ratio"], step, qpoint))
            trajectory.append({"step": step, "qpoint": qpoint, "minimum_accuracy": score["minimum_accuracy"],
                               "mean_accuracy": score["mean_accuracy"], "maximum_distance_ratio": score["maximum_distance_ratio"]})
    _, _, step, qpoint = max(candidates)
    absolute = np.abs(source_field[:, step, qpoint].astype(np.float32)); residual = fit_residual(absolute, rows, train, factors)
    sorted_control = np.sort(residual, axis=1)
    selection = gen_evaluate(residual, rows, labels, "selection"); lockbox = gen_evaluate(residual, rows, labels, "lockbox")
    sorted_lockbox = gen_evaluate(sorted_control, rows, labels, "lockbox")
    close_memmap(source_field)
    return {"selection_trajectory": trajectory, "selected_step": step, "selected_qpoint": qpoint,
            "selection": selection, "lockbox": lockbox, "sorted_lockbox": sorted_lockbox,
            "coordinate_advantage": lockbox["minimum_accuracy"] - sorted_lockbox["minimum_accuracy"]}, {
                "generation_residual": residual, "generation_sorted_control": sorted_control,
                "generation_absolute_hidden": absolute}


def publish(rows: list[dict], prompt: dict, prompt_fields: dict[str, np.ndarray], gen: dict, gen_fields: dict[str, np.ndarray]) -> list[dict]:
    datasets = []
    dataset_id = "c9401_qwen4b_conditional_equivalence_prompt_passport"; matrix = np.concatenate(list(prompt_fields.values()))
    binary = VIS / f"{dataset_id}.float32.npy"; out = atlas.create_binary(binary.name, *matrix.shape, np.float32); out[:] = matrix
    out.flush(); close_memmap(out); metadata = []
    for view in prompt_fields:
        metadata.extend({"case_id": r["case_id"], "family": r["family"], "language": r["language"], "surface": r["surface"],
                         "query": r["query"], "depth": r["depth"], "partition": r["partition"], "unit": r["unit"],
                         "state": r["state"], "qpoint": prompt["selected_qpoint"], "view": view} for r in rows)
    datasets.append(atlas.write_metadata(dataset_id, "Qwen3-4B conditional-equivalence prompt passport", binary, metadata,
        "Qwen3-4B-FP16", "conditional_equivalence_prompt_passport_v1", "descriptive route competition; not causal",
        "qualified families, fresh-confirmation selection and untouched fresh-lockbox adjudication",
        "all 2560 physical coordinates retained with sorted-address control",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "qpoint": prompt["selected_qpoint"], "views": list(prompt_fields)}))
    gen_rows = io.read_rows(P2352 / "raw/fresh_lockbox_generation.jsonl")
    dataset_id = "c9402_qwen4b_generation_conditioned_equivalence_passport"; matrix = np.concatenate(list(gen_fields.values()))
    binary = VIS / f"{dataset_id}.float32.npy"; out = atlas.create_binary(binary.name, *matrix.shape, np.float32); out[:] = matrix
    out.flush(); close_memmap(out); metadata = []
    for view in gen_fields:
        metadata.extend({"case_id": r["case_id"], "family": r["family"], "language": r["language"], "query": r["query"],
                         "depth": r["depth"], "unit": r["unit"], "step": gen["selected_step"],
                         "qpoint": gen["selected_qpoint"], "view": view} for r in gen_rows)
    datasets.append(atlas.write_metadata(dataset_id, "Qwen3-4B generation-conditioned equivalence passport", binary, metadata,
        "Qwen3-4B-FP16", "generation_conditioned_equivalence_passport_v1", "descriptive self-generated-history atlas; not causal",
        "fresh-lockbox generations split by held-out units 12-13/14/15",
        "all 2560 physical coordinates retained with sorted-address control",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "step": gen["selected_step"],
         "qpoint": gen["selected_qpoint"], "views": list(gen_fields)}))
    return datasets


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 条件等价全坐标路线竞赛与生成瞬态锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Phase2352的12族、6144条prompt场上，不选Top-K，比较原始绝对值、逐行L2绝对值、去除语言/表述/查询/图深/状态训练集主效应的绝对残差和有符号残差；仅用discovery+confirmation拟合，fresh_confirmation选路线与层，fresh_lockbox裁决。跨语言、表述、查询、图深、状态共14个双向轴。另对384条模型自生成轨迹，以units12–13建原型、unit14选生成位置/层、unit15锁箱，并保留坐标排序对照。

$$
R_i=X_i-\mu_{{train}}-\sum_{{a\in A}}(\mu_{{a(i),train}}-\mu_{{train}}),\qquad
\Delta_{{coord}}=A_{{physical}}-A_{{sort(X)}}.
$$

**结果汇总。** prompt竞赛 `{json.dumps(result['prompt'], ensure_ascii=False)}`；生成瞬态竞赛 `{json.dumps(result['generation'], ensure_ascii=False)}`；门槛 `{json.dumps(result['gate'], ensure_ascii=False)}`；可视化 `{json.dumps(result['datasets'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2353_c9401_c9560_conditional_equivalence_route_competition.py`；结果 `tests/glm5/result/phase2353_c9401_c9560_conditional_equivalence_route_competition`；客户端`c9401/c9402`。

**理论进展、问题硬伤与结论。** “条件等价”在这里严格指：同族在未见unit上跨指定外部条件仍被同一模型局部坐标图谱识别；它不是范畴等价、流形同胚或计算算子。坐标排序控制用于判断具体地址是否必要。即使prompt和生成瞬态均通过，仍只是相关图谱；只有下一Phase的保范数匹配干预同时强于错族、错时刻、置乱方向并可独立救援，才进入因果候选。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    p2352 = json.loads((P2352 / "analysis/final.json").read_text(encoding="utf-8")); rows = io.read_rows(P2352 / "material/natural_multifuture_graphs.jsonl")
    labels = list(p2352["behavior"]["qualified"])
    if len(labels) < 3: raise RuntimeError(("too_few_behavior_qualified_families", labels))
    prompt, prompt_fields = prompt_competition(rows, labels); generation, gen_fields = generation_competition(labels)
    gate = {"qualified_families": len(labels), "prompt_lockbox_minimum": prompt["lockbox"]["minimum_accuracy"],
            "prompt_coordinate_advantage": prompt["coordinate_advantage"], "generation_lockbox_minimum": generation["lockbox"]["minimum_accuracy"],
            "generation_coordinate_advantage": generation["coordinate_advantage"],
            "prompt_descriptive_pass": prompt["lockbox"]["minimum_accuracy"] >= 0.30 and prompt["coordinate_advantage"] >= 0.10,
            "generation_descriptive_pass": generation["selected_step"] > 0 and generation["lockbox"]["minimum_accuracy"] >= 0.25 and generation["coordinate_advantage"] >= 0.10,
            "post_first_token_required": True,
            "causal_status": False}
    datasets = publish(rows, prompt, prompt_fields, generation, gen_fields); verification = [atlas.verify(r) for r in datasets]
    verified = all(all(v for k, v in row.items() if k != "id") for row in verification); catalog = atlas.update_catalog(datasets)
    build = atlas.frontend_build()
    if not (verified and build["passed"]): raise RuntimeError((verification, build))
    raw_size = STATES.stat().st_size; STATES.unlink()
    cleanup = {"deleted_unpublished_full_prompt_field": str(STATES), "bytes_reclaimed": raw_size, "deleted_ok": not STATES.exists()}
    checks = {"qualified": len(labels) >= 3, "assets": verified, "frontend_build": build["passed"], "raw_deleted": cleanup["deleted_ok"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "labels": labels, "prompt": prompt, "generation": generation,
              "gate": gate, "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
              "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
              "frontend_build": build, "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2353_failed", checks))
    append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
