#!/usr/bin/env python3
"""Map full-coordinate HiddenState and semantic interaction fields into first-token output margins."""
from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2423 = RESULT / "phase2423_c30001_c30320_semantic_validity_behavior_contract"
P2424 = RESULT / "phase2424_c30321_c30640_semantic_validity_multievent_fullfield"
P2426 = RESULT / "phase2426_c30961_c31280_coordinate_identity_multinull"
OUT = RESULT / "phase2430_c32241_c32560_output_coordinate_compilation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE = 2430
CAMPAIGN = "C32241-C32560"
SPLITS = ("confirmation", "fresh_unit", "template", "joint", "language", "family")

sys.path.insert(0, str(TESTS))
import phase2389_c19121_c19440_crossmodel_autonomous_capability as capability  # noqa: E402
import phase2425_c30641_c30960_semantic_specific_interaction_atlas as atlas  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def close(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def collect_readout(rows: list[dict], state_path: str) -> dict:
    contribution_path = OUT / "derived/readout_coordinate_contribution.float32.npy"
    weight_path = OUT / "derived/readout_weight_difference.float16.npy"
    contribution_path.parent.mkdir(parents=True, exist_ok=True)
    n, dim = len(rows), 2560
    contributions = np.lib.format.open_memmap(contribution_path, mode="r+" if contribution_path.exists() else "w+",
                                               dtype=np.float32, shape=(n, dim))
    weights = np.lib.format.open_memmap(weight_path, mode="r+" if weight_path.exists() else "w+",
                                       dtype=np.float16, shape=(n, dim))
    progress = OUT / "derived/readout_progress.json"
    completed = int(json.loads(progress.read_text(encoding="utf-8"))["completed"]) if progress.exists() else 0
    state = np.load(state_path, mmap_mode="r")
    model = tokenizer = output_weight = None
    label = "Qwen3-4B-BF16"
    if completed < n:
        model, tokenizer, label = capability.load_model("qwen4b")
        output_weight = model.get_output_embeddings().weight.detach()
    metadata = []
    try:
        for start in range(completed, n, 128):
            batch = rows[start:start + 128]
            target = torch.tensor([row["target_ids"][0] for row in batch], dtype=torch.long, device=output_weight.device)
            foil = torch.tensor([row["foil_ids"][0] for row in batch], dtype=torch.long, device=output_weight.device)
            difference = (output_weight[target] - output_weight[foil]).float().cpu().numpy()
            hidden = np.asarray(state[start:start + len(batch), 37, 2], dtype=np.float32)
            weights[start:start + len(batch)] = difference.astype(np.float16)
            contributions[start:start + len(batch)] = hidden * difference
            weights.flush(); contributions.flush(); save(progress, {"completed": start + len(batch)})
            print(f"[phase2430 readout] {start + len(batch)}/{n}", flush=True)
        for index, row in enumerate(rows):
            metadata.append({"case_id": row["case_id"], "first_divergence_index": int(row.get("first_divergence_index", 0)),
                             "target_first_token": int(row["target_ids"][0]), "foil_first_token": int(row["foil_ids"][0])})
    finally:
        del model, tokenizer, output_weight; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        close(state); close(contributions); close(weights)
    write_rows(OUT / "index/readout_rows.jsonl", metadata)
    return {"model": label, "contribution": {"path": str(contribution_path), "shape": [n, dim], "bytes": contribution_path.stat().st_size},
            "weight_difference": {"path": str(weight_path), "shape": [n, dim], "bytes": weight_path.stat().st_size}}


def behavior_arrays(rows: list[dict], teacher: list[dict]) -> tuple[np.ndarray, dict]:
    margins = np.asarray([record["first_divergence_logit_margin"] for record in teacher], dtype=np.float32)
    divergence = np.asarray([int(record["first_divergence_index"]) for record in teacher])
    return margins, {"rows": len(rows), "divergence_zero": int(np.sum(divergence == 0)),
                     "divergence_zero_rate": float(np.mean(divergence == 0)), "finite_margin": bool(np.isfinite(margins).all())}


def output_closure(readout: dict, margins: np.ndarray) -> dict:
    contributions = np.load(readout["contribution"]["path"], mmap_mode="r")
    predicted = np.asarray(contributions, dtype=np.float64).sum(axis=1)
    residual = predicted - margins
    result = {"rows": len(margins), "coordinate_sum_margin_correlation": pearson(predicted, margins),
              "rmse": float(np.sqrt(np.mean(residual * residual))), "mae": float(np.mean(np.abs(residual))),
              "max_abs": float(np.max(np.abs(residual))), "predicted_margin_mean": float(np.mean(predicted)),
              "teacher_margin_mean": float(np.mean(margins)),
              "relative_rmse": float(np.sqrt(np.mean(residual * residual)) /
                                     max(np.sqrt(np.mean(np.asarray(margins, dtype=np.float64) ** 2)), 1e-30)),
              "numeric_boundary": "FP16 archived HiddenState versus BF16 runtime logits; this is an approximate reconstruction, not bit-exact closure"}
    close(contributions)
    return result


def interaction_output(rows: list[dict], readout: dict, collection: dict, corrected: dict) -> dict:
    meta, index = atlas.configuration_index(rows)
    families = np.asarray([row["family"] for row in meta], dtype=object)
    specs = atlas.split_specs(meta, families)
    contribution = np.load(readout["contribution"]["path"], mmap_mode="r")
    state = np.load(collection["state"]["path"], mmap_mode="r")
    attention = np.load(collection["attention"]["path"], mmap_mode="r")
    mlp = np.load(collection["mlp"]["path"], mmap_mode="r")
    output_semantic, output_lexical = atlas.interactions(contribution[:, None, None, :], 0, 0, index)
    outputs = (output_semantic, output_lexical)
    results = {}
    selections = corrected["analysis"]["selections"]
    for ii, interaction in enumerate(atlas.INTERACTIONS):
        results[interaction] = {}
        for ci, component in enumerate(atlas.COMPONENTS):
            layer, event = selections[interaction][component]
            h_pair = atlas.interactions(state, layer, event, index)
            a_pair = atlas.interactions(attention, layer, event, index)
            m_pair = atlas.interactions(mlp, layer, event, index)
            internal = ((a_pair[ii] + m_pair[ii]), a_pair[ii], m_pair[ii])[ci]
            results[interaction][component] = {}
            for split in SPLITS:
                train, test, conditioned = specs[split]
                if split == "family":
                    values = atlas.family_holdout(meta, families, train, test, internal, outputs[ii])
                else:
                    fitted = atlas.fit(train, families, internal, outputs[ii], family_conditioned=conditioned)
                    global_p, family_p, state_p = atlas.predict(test, families, internal, fitted)
                    _, _, mismatch_p = atlas.predict(test, families, internal, fitted, mismatch=True)
                    values = atlas.gains(outputs[ii][test], global_p, family_p, state_p, mismatch_p)
                results[interaction][component][split] = {"family_gain": values[1], "state_gain": values[2],
                                                          "mismatch_gain": values[3], "physical_advantage": values[2] - values[3]}
    summary = {interaction: {component: {"mean_state_gain": float(np.mean([value["state_gain"] for value in splits.values()])),
                                         "mean_physical_advantage": float(np.mean([value["physical_advantage"] for value in splits.values()])),
                                         "positive_physical_split_rate": float(np.mean([value["physical_advantage"] > 0 for value in splits.values()]))}
                             for component, splits in components.items()} for interaction, components in results.items()}
    for value in (contribution, state, attention, mlp):
        close(value)
    return {"selections": selections, "results": results, "summary": summary}


def autonomous_bridge(rows: list[dict], margins: np.ndarray) -> dict:
    autonomous = read_rows(P2423 / "qwen4b/behavior/autonomous_lockbox.jsonl")
    lockbox_indices = [i for i, row in enumerate(rows) if row["variant"] == "valid" and int(row["unit"]) >= 6]
    if len(lockbox_indices) != len(autonomous):
        raise RuntimeError((len(lockbox_indices), len(autonomous)))
    selected = margins[lockbox_indices]
    exact = np.asarray([record["exact"] for record in autonomous], dtype=bool)
    present = np.asarray([record["target_present"] for record in autonomous], dtype=bool)
    return {"rows": len(autonomous), "exact_rate": float(exact.mean()), "target_present_rate": float(present.mean()),
            "margin_exact_correlation": pearson(selected, exact.astype(float)), "margin_exact_auc": auc(selected, exact),
            "margin_target_present_correlation": pearson(selected, present.astype(float)),
            "margin_target_present_auc": auc(selected, present)}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 词嵌入—HiddenState逐坐标输出编译与自主行为桥（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2423全部6144条，读取answer boundary的final norm HiddenState全部2560坐标，并从Qwen4B输出嵌入矩阵取得目标/foil第一分歧token的逐坐标权重差。保存每条的权重差与$H_i\Delta W_i$贡献，验证坐标和能否重建教师强制logit margin。随后把Phase2426冻结的query-end语义/词项内部交互映射到输出贡献交互，在六锁箱比较同坐标与+791错位。最后用fresh valid的教师margin预测自主exact/target-present，不把教师强制闭合冒充自主生成。

$$m=\ell_{{target}}-\ell_{{foil}}=\sum_{{i=1}}^{{2560}}H_i(W_{{target,i}}-W_{{foil,i}}),$$

$$C_i=H_i\Delta W_i,\qquad I_{{sem}}^C=(C_{{v,t}}-C_{{v,s}})-(C_{{a,t}}-C_{{a,s}}).$$

**结果汇总。** 逐参数文件 `{json.dumps(result['readout'], ensure_ascii=False)}`；第一token资格 `{json.dumps(result['behavior_coverage'], ensure_ascii=False)}`；输出恒等闭合 `{json.dumps(result['output_closure'], ensure_ascii=False)}`；内部场到输出贡献 `{json.dumps(result['interaction_output'], ensure_ascii=False)}`；自主桥 `{json.dumps(result['autonomous_bridge'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2430_c32241_c32560_output_coordinate_compilation.py`；6144×2560输出嵌入权重差、逐坐标logit贡献、逐行token索引及六锁箱映射位于`tests/glm5/result/phase2430_c32241_c32560_output_coordinate_compilation`。HiddenState源为Phase2424完整场。未修改其他Markdown。

**分析与理论进展。** 输出层恒等式提供真实参数级编译接口：每个HiddenState坐标怎样乘上具体token嵌入权重并累加成margin。它本身是架构恒等式，不是发现；新问题是较早query-end的语义交互能否跨内容/语言/家族稳定预测这些逐坐标贡献，以及该margin是否对应自主输出。只有两者都成立，内部纹理才与语言行为闭合。

**问题硬伤与结论。** 只对第一分歧token且要求分歧index为0时，answer-boundary状态才是精确对应；若候选共享前缀必须另走共享前缀后的状态。float16 HiddenState/权重差带来小重建误差。自主exact受模型输出格式严重影响，target-present更宽松但可能包含解释文本。线性输出恒等式不证明内部形成机制，六锁箱内部→输出映射仍是离线关联。
"""
    with MEMO.open("a", encoding="utf-8", newline="") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows = read_rows(P2423 / "qwen4b/index/semantic_validity_rows.jsonl")
    teacher = read_rows(P2423 / "qwen4b/behavior/teacher_scores.jsonl")
    p2424 = json.loads((P2424 / "analysis/final.json").read_text(encoding="utf-8"))
    corrected = json.loads((P2426 / "analysis/final.json").read_text(encoding="utf-8"))
    margins, coverage = behavior_arrays(rows, teacher)
    readout = collect_readout(rows, p2424["collection"]["state"]["path"])
    closure = output_closure(readout, margins)
    interaction = interaction_output(rows, readout, p2424["collection"], corrected)
    autonomous = autonomous_bridge(rows, margins)
    semantic_output = all(interaction["results"]["semantic_validity"][component][split]["physical_advantage"] > 0 and
                          interaction["results"]["semantic_validity"][component][split]["state_gain"] > 0
                          for component in atlas.COMPONENTS for split in SPLITS)
    autonomous_closed = autonomous["margin_target_present_auc"] > .7 and autonomous["margin_target_present_correlation"] > .2
    numeric_qualified = closure["coordinate_sum_margin_correlation"] > .995 and closure["relative_rmse"] < .08
    adjudication = {"architectural_linear_readout_identity": True,
                    "archived_fp16_readout_numeric_approximation_qualified": numeric_qualified,
                    "semantic_internal_to_output_physical_map_all_components_splits": semantic_output,
                    "teacher_margin_predicts_autonomous_target_all_gates": autonomous_closed,
                    "output_behavior_bridge_closed": semantic_output and autonomous_closed,
                    "language_encoding_mechanism_closed": False}
    checks = {"rows_6144": coverage["rows"] == 6144, "all_first_divergence_zero": coverage["divergence_zero_rate"] == 1.0,
              "full_coordinate_readout": readout["contribution"]["shape"] == [6144, 2560],
              "fp16_readout_numeric_approximation": numeric_qualified,
              "six_splits": all(set(value) == set(SPLITS) for interaction_value in interaction["results"].values() for value in interaction_value.values()),
              "finite": all(math.isfinite(value) for value in autonomous.values() if isinstance(value, float)),
              "raw_retained": all(Path(item["path"]).exists() for item in p2424["collection"].values()),
              "claim_boundary": not adjudication["language_encoding_mechanism_closed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "readout": readout, "behavior_coverage": coverage,
              "output_closure": closure, "interaction_output": interaction, "autonomous_bridge": autonomous,
              "adjudication": adjudication, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
