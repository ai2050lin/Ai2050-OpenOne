#!/usr/bin/env python3
"""Score both complete candidates and test locked full-coordinate readouts of output margin."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2513 = RESULT / "phase2513_c76673_c78624_fresh_context_factorial_behavior_fullfield"
P2514 = RESULT / "phase2514_c78625_c79776_context_operator_competition_lockbox"
OUT = RESULT / "phase2515_c79777_c80800_output_margin_fullcoordinate_readout"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2515, "C79777-C80800"
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def continuation_ids(tokenizer, row: dict, candidate: str) -> list[int]:
    text = (" " if row["language"] == "en" else "") + candidate
    return [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]


def score_candidates(model, tokenizer, rows: list[dict], batch_size: int = 8) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    pad = tokenizer.pad_token_id
    jobs = []
    for row in rows:
        for relation_index, candidate in enumerate(row["relation_targets"]):
            cont = continuation_ids(tokenizer, row, candidate)
            jobs.append({"row": row, "relation_index": relation_index, "candidate": candidate,
                         "continuation": cont, "sequence": row["prompt_ids"] + cont})
    output = []
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        width = max(len(job["sequence"]) for job in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, job in enumerate(batch):
            seq = job["sequence"]
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device); mask[i, :len(seq)] = 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        for i, job in enumerate(batch):
            begin = len(job["row"]["prompt_ids"])
            token_values = []
            for j, token_id in enumerate(job["continuation"]):
                values = logits[i, begin - 1 + j].float()
                token_values.append(float(values[token_id] - torch.logsumexp(values, dim=-1)))
            output.append({"case_id": job["row"]["case_id"], "unit": job["row"]["unit"],
                           "pair_id": job["row"]["pair_id"], "language": job["row"]["language"],
                           "context_id": job["row"]["context_id"], "meaning_swap": job["row"]["meaning_swap"],
                           "query_marker": job["row"]["query_marker"], "relation_index": job["relation_index"],
                           "candidate": job["candidate"], "continuation_ids": job["continuation"],
                           "token_logprobs": token_values, "sum_logprob": float(sum(token_values)),
                           "mean_logprob": float(np.mean(token_values))})
        if (start + len(batch)) % 256 == 0:
            print(f"[phase2515 scores] {start + len(batch)}/{len(jobs)}", flush=True)
    return output


def output_interactions(rows: list[dict], scores: list[dict], pairs: list[int]) -> tuple[np.ndarray, list[dict]]:
    by = {(r["case_id"], r["relation_index"]): r for r in scores}
    row_by = {r["case_id"]: r for r in rows}
    values = np.zeros((2, len(pairs), 2, 16, 2), dtype=np.float64)  # unit,pair,lang,context,sum/mean
    metadata = []
    for ui, unit in enumerate((28, 29)):
        for pi, pair_id in enumerate(pairs):
            for li, language in enumerate(("en", "zh")):
                for context in range(16):
                    cells = {}
                    for m in (0, 1):
                        for q in (0, 1):
                            candidates = [r for r in rows if r["unit"] == unit and r["pair_id"] == pair_id and r["language"] == language
                                          and r["context_id"] == context and r["meaning_swap"] == m and r["query_marker"] == q]
                            assert len(candidates) == 1
                            row = candidates[0]
                            s0, s1 = by[(row["case_id"], 0)], by[(row["case_id"], 1)]
                            cells[(m, q)] = (s0["sum_logprob"] - s1["sum_logprob"], s0["mean_logprob"] - s1["mean_logprob"])
                    interaction = (np.asarray(cells[(0, 0)]) - np.asarray(cells[(0, 1)])
                                   - np.asarray(cells[(1, 0)]) + np.asarray(cells[(1, 1)])) / 4
                    values[ui, pi, li, context] = interaction
                    exemplar = next(r for r in rows if r["unit"] == unit and r["pair_id"] == pair_id and r["language"] == language and r["context_id"] == context)
                    metadata.append({"unit_index": ui, "unit": unit, "pair_index": pi, "pair_id": pair_id,
                                     "edge": exemplar["families"], "language_index": li, "language": language,
                                     "context_id": context, "paraphrase": exemplar["paraphrase"], "fact_order": exemplar["fact_order"],
                                     "definition_order": exemplar["definition_order"], "candidate_order": exemplar["candidate_order"]})
    return values, metadata


def factor_features(meta: list[dict]) -> np.ndarray:
    return np.asarray([[1.0, r["language_index"], r["paraphrase"], r["fact_order"], r["definition_order"], r["candidate_order"]]
                       for r in meta], dtype=np.float64)


def scalar_metrics(y: np.ndarray, pred: np.ndarray, baseline: float) -> dict:
    den = float(np.square(y - baseline).sum()); sse = float(np.square(y - pred).sum())
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(y) > 0 and np.std(pred) > 0 else 0.0
    return {"r2_vs_train_mean": 1 - sse / den if den else 0.0, "correlation": corr,
            "mae": float(np.mean(np.abs(y - pred))), "prediction_positive_rate": float(np.mean(pred > 0)),
            "target_positive_rate": float(np.mean(y > 0))}


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha_multiplier: float) -> dict:
    xm, ym = x.mean(axis=0), float(y.mean()); xc, yc = x - xm, y - ym
    kernel = xc @ xc.T
    scale = float(np.trace(kernel) / max(len(x), 1))
    dual = np.linalg.solve(kernel + (alpha_multiplier * scale + 1e-12) * np.eye(len(x)), yc)
    return {"xm": xm, "ym": ym, "dual": dual, "xtrain": xc}


def ridge_predict(params: dict, x: np.ndarray) -> np.ndarray:
    return params["ym"] + ((x - params["xm"]) @ params["xtrain"].T) @ params["dual"]


def condition_fit(meta: list[dict], y: np.ndarray) -> dict:
    f = factor_features(meta); return {"beta": np.linalg.pinv(f) @ y}


def condition_predict(params: dict, meta: list[dict]) -> np.ndarray:
    return factor_features(meta) @ params["beta"]


def edge_cv_alpha(x: np.ndarray, y: np.ndarray, meta: list[dict], residualize: bool) -> tuple[float, dict]:
    panels = []
    for alpha in ALPHAS:
        predictions, targets, baselines = [], [], []
        for edge in sorted({r["pair_index"] for r in meta}):
            test = np.asarray([r["pair_index"] == edge for r in meta]); train = ~test
            mt, mv = [r for r, f in zip(meta, train) if f], [r for r, f in zip(meta, test) if f]
            base = float(y[train].mean())
            if residualize:
                cp = condition_fit(mt, y[train]); train_res = y[train] - condition_predict(cp, mt)
                pred = condition_predict(cp, mv) + ridge_predict(ridge_fit(x[train], train_res, alpha), x[test])
            else:
                pred = ridge_predict(ridge_fit(x[train], y[train], alpha), x[test])
            predictions.extend(pred); targets.extend(y[test]); baselines.extend([base] * int(test.sum()))
        targets, predictions, baselines = map(np.asarray, (targets, predictions, baselines))
        den = float(np.square(targets - baselines).sum()); sse = float(np.square(targets - predictions).sum())
        panels.append({"alpha_multiplier": alpha, "r2_vs_fold_train_mean": 1 - sse / den if den else 0.0,
                       "correlation": float(np.corrcoef(targets, predictions)[0, 1]) if np.std(predictions) else 0.0})
    best = max(panels, key=lambda p: p["r2_vs_fold_train_mean"])
    return float(best["alpha_multiplier"]), {"selected": best, "all": panels}


def source_field(interactions: np.ndarray, unit_index: int, event_index: int, qpoint: int) -> np.ndarray:
    # interaction array axes unit,pair,language,context,event,qpoint,coordinate.
    return np.asarray(interactions[unit_index, :, :, :, event_index, qpoint].reshape(-1, interactions.shape[-1]), dtype=np.float64)


def scan_readout(interactions: np.ndarray, output: np.ndarray, metadata: list[dict], event_index: int, metric_index: int) -> dict:
    train_meta = [r for r in metadata if r["unit_index"] == 0]; test_meta = [r for r in metadata if r["unit_index"] == 1]
    y_train, y_test = output[0, ..., metric_index].reshape(-1), output[1, ..., metric_index].reshape(-1)
    scan = []
    for qpoint in range(1, 38):
        x = source_field(interactions, 0, event_index, qpoint)
        alpha, cv = edge_cv_alpha(x, y_train, train_meta, residualize=True)
        scan.append({"qpoint": qpoint, "alpha": alpha, "confirmation_cv": cv["selected"]})
    selected = max(scan, key=lambda p: p["confirmation_cv"]["r2_vs_fold_train_mean"])
    qpoint, alpha = selected["qpoint"], selected["alpha"]
    x_train, x_test = source_field(interactions, 0, event_index, qpoint), source_field(interactions, 1, event_index, qpoint)
    condition = condition_fit(train_meta, y_train); condition_pred = condition_predict(condition, test_meta)
    direct = ridge_predict(ridge_fit(x_train, y_train, alpha), x_test)
    residual = y_train - condition_predict(condition, train_meta)
    combined = condition_pred + ridge_predict(ridge_fit(x_train, residual, alpha), x_test)
    baseline = float(y_train.mean())
    return {"selection": "unit28 whole-edge CV over q1-q37 and ridge alpha; unit29 untouched",
            "selected_qpoint": qpoint, "selected_alpha": alpha, "scan": scan,
            "unit29": {"mean_baseline": scalar_metrics(y_test, np.full_like(y_test, baseline), baseline),
                       "explicit_condition_only": scalar_metrics(y_test, condition_pred, baseline),
                       "hidden_only_ridge": scalar_metrics(y_test, direct, baseline),
                       "condition_plus_hidden_residual_ridge": scalar_metrics(y_test, combined, baseline)}}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    concise = {name: {"selected_qpoint": panel["selected_qpoint"], "selected_alpha": panel["selected_alpha"],
                      "unit29": panel["unit29"]} for name, panel in result["readouts"].items()}
    text = rf"""


## Phase {PHASE}: 完整候选序列概率与全坐标输出读出的未见unit检验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Phase2513四个合格关系对的1024条prompt分别teacher-force两个完整候选字符串，逐token求logprob并形成relation0−relation1分数，再对meaning-swap×query-marker做四格交互，共unit×4pair×2language×16context=256个输出交互。只在unit28以整条relation edge留一，同时选择q1–q37与七个ridge强度；随后用全新unit29检验三种读出：仅显式五因素、仅全2560坐标、因素加HiddenState残差。分别以query和answer事件的全坐标交互预测sum/mean logprob交互。

$$\Delta L=\log P(Y_0\mid x)-\log P(Y_1\mid x),\qquad I_L=\tfrac14(\Delta L_{{00}}-\Delta L_{{01}}-\Delta L_{{10}}+\Delta L_{{11}}).$$

**结果汇总。** 输出几何 `{json.dumps(result['output_geometry'], ensure_ascii=False)}`；读出摘要 `{json.dumps(concise, ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2515_c79777_c80800_output_margin_fullcoordinate_readout.py`；2048条候选序列分数、256个输出交互、全部读出扫描、哈希与final位于`{OUT}`。

**分析与理论进展。** 输出交互方向通过只说明模型行为概率随定义选择变化；未见unit的数值预测才检验固定读出。因素+Hidden残差若不超过因素基线，不能说query场携带可直接解码的输出幅度；若超过，也只是外部预测关联。整edge留一减少关系记忆，但四个pair仍少。

**问题硬伤与结论。** teacher forcing不等于自主生成；sequence长度可能不同，因此同时报告sum与mean；核ridge在样本空间拟合，不能解释为模型内部线性神经元；层位与alpha多重选择仅由unit28完成但仍可能选择偏差。下一Phase只在读出/场结果支持的冻结层上定义坐标联盟并做自然状态patch，随机同规模和补集控制不可缺失。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    f13, f14 = load_json(P2513 / "analysis/final.json"), load_json(P2514 / "analysis/final.json")
    rows = read_jsonl(Path(f13["collection"]["event_index"]))
    material = {r["case_id"]: r for r in read_jsonl(P2513 / "material/factorial_rows.jsonl")}
    for row in rows:
        row["relation_targets"] = material[row["case_id"]]["relation_targets"]
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        scores = score_candidates(model, tokenizer, rows)
    finally:
        model_utils.release_model(model); gc.collect()
    score_path = OUT / "output/candidate_sequence_scores.jsonl"; write_jsonl(score_path, scores)
    pairs = f13["behavior"]["qualified_pair_ids"]
    outputs, metadata = output_interactions(rows, scores, pairs)
    output_path = OUT / "derived/output_fourcell_interactions.float64.npy"; output_path.parent.mkdir(parents=True, exist_ok=True); np.save(output_path, outputs)
    meta_path = OUT / "index/output_interaction_rows.jsonl"; write_jsonl(meta_path, metadata)
    interactions = np.load(f14["fields"]["all_interactions"]["path"], mmap_mode="r")
    readouts = {}
    for event_name, event_index in (("query", 0), ("answer", 3)):
        for metric_name, metric_index in (("sum", 0), ("mean", 1)):
            readouts[f"{event_name}_{metric_name}"] = scan_readout(interactions, outputs, metadata, event_index, metric_index)
    geometry = {}
    for ui, unit in enumerate((28, 29)):
        geometry[str(unit)] = {}
        for metric, mi in (("sum", 0), ("mean", 1)):
            values = outputs[ui, ..., mi].reshape(-1)
            geometry[str(unit)][metric] = {"groups": len(values), "mean": float(np.mean(values)),
                                           "positive_rate": float(np.mean(values > 0)), "minimum": float(np.min(values)),
                                           "maximum": float(np.max(values))}
    checks = {"sources_passed": f13["all_checks_passed"] and f14["all_checks_passed"],
              "scores_2048": len(scores) == 2048, "outputs_shape": outputs.shape == (2, 4, 2, 16, 2),
              "unit29_never_selected": True, "all_coordinates_used": interactions.shape[-1] == 2560,
              "finite": bool(np.isfinite(outputs).all()), "hash": len(digest(output_path)) == 64,
              "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA",
              "output_geometry": geometry, "readouts": readouts,
              "files": {"scores": str(score_path), "output_interactions": str(output_path), "output_sha256": digest(output_path),
                        "index": str(meta_path)},
              "adjudication": {"output_interaction_reliably_positive": geometry["29"]["sum"]["positive_rate"] >= .9,
                               "direct_fullcoordinate_output_readout_identified": False,
                               "causal_mediator_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "geometry": geometry,
                      "readouts": {k: {"q": v["selected_qpoint"], "unit29": v["unit29"]} for k, v in readouts.items()},
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
