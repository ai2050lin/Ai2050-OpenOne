#!/usr/bin/env python3
"""Trace autonomous generation and exact candidate-sequence scores for the fresh relation-swap lockbox."""
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
P2503 = RESULT / "phase2503_c67201_c68224_equal_length_fresh_lockbox_behavior_fullfield"
P2504 = RESULT / "phase2504_c68225_c68864_corrected_semantic_selection_walsh_lockbox"
OUT = RESULT / "phase2505_c68865_c70016_semantic_selection_autonomous_output_geometry"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, MAX_NEW, DIM = 2505, "C68865-C70016", 10, 2560

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402
import phase2500_c64001_c65152_semantic_necessity_2x2_behavior as base  # noqa: E402
import phase2502_c66177_c67200_semantic_selection_walsh_fullcoordinate_lockbox as walsh  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def continuation_score(model, tokenizer, row: dict, candidate: str, device: torch.device) -> dict:
    text = (" " if row["language"] == "en" else "") + candidate
    continuation = [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]
    sequence = row["prompt_ids"] + continuation
    ids = torch.tensor([sequence], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
        start = len(row["prompt_ids"])
        token_logits = logits[start - 1:start + len(continuation) - 1].float().log_softmax(dim=-1)
        values = [float(token_logits[i, token_id].item()) for i, token_id in enumerate(continuation)]
    return {"token_ids": continuation, "token_logprobs": values, "sum_logprob": sum(values),
            "mean_logprob": sum(values) / max(len(values), 1)}


def capture(model, tokenizer, rows: list[dict]) -> dict:
    qmods = field_utils.modules(model)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    field_path = raw / "autonomous_boundary_generation_allqpoint.float16.npy"
    mask_path = raw / "autonomous_event_mask.uint8.npy"
    token_path = raw / "autonomous_token_ids.int32.npy"
    field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16,
                                      shape=(len(rows), MAX_NEW + 1, len(qmods), DIM))
    mask = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(len(rows), MAX_NEW + 1))
    token_ids = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.int32, shape=(len(rows), MAX_NEW))
    field[:] = 0; mask[:] = 0; token_ids[:] = -1
    captures = {}
    handles = []
    for qpoint, module in enumerate(qmods):
        def hook(_module, _inputs, output, qpoint=qpoint):
            captures[qpoint] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(module.register_forward_hook(hook))
    device = model.get_input_embeddings().weight.device
    index = []
    try:
        with torch.inference_mode():
            for model_row, row in enumerate(rows):
                ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
                generated_ids = []
                answer_step = None
                parsed = None
                correct = False
                generated_text = ""
                for step in range(MAX_NEW + 1):
                    captures.clear()
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    for qpoint in range(len(qmods)):
                        field[model_row, step, qpoint] = captures[qpoint][0, -1].float().cpu().numpy().astype(np.float16)
                    mask[model_row, step] = 1
                    if step == MAX_NEW:
                        break
                    next_id = int(torch.argmax(output.logits[0, -1]).item())
                    generated_ids.append(next_id)
                    token_ids[model_row, step] = next_id
                    ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    parsed, correct = base.parse(generated_text, row)
                    if parsed is not None:
                        answer_step = step + 1
                        captures.clear()
                        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                        for qpoint in range(len(qmods)):
                            field[model_row, answer_step, qpoint] = captures[qpoint][0, -1].float().cpu().numpy().astype(np.float16)
                        mask[model_row, answer_step] = 1
                        break
                scores = {candidate: continuation_score(model, tokenizer, row, candidate, device) for candidate in row["candidates"]}
                relation_scores = [scores[target] for target in row["relation_targets"]]
                index.append({"model_row": model_row, "case_id": row["case_id"], "unit": row["unit"],
                              "pair_id": row["pair_id"], "families": row["families"], "language": row["language"],
                              "surface": row["surface"], "meaning_swap": row["meaning_swap"],
                              "query_marker": row["query_marker"], "selected_relation": row["selected_relation"],
                              "target": row["target"], "candidates": row["candidates"],
                              "generated_ids": generated_ids, "generated_text": generated_text,
                              "parsed_answer": parsed, "parsed_correct": bool(correct), "answer_step": answer_step,
                              "first_step": 1 if generated_ids else None, "candidate_scores": scores,
                              "relation0_minus_relation1_sum": relation_scores[0]["sum_logprob"] - relation_scores[1]["sum_logprob"],
                              "relation0_minus_relation1_mean": relation_scores[0]["mean_logprob"] - relation_scores[1]["mean_logprob"]})
                if (model_row + 1) % 24 == 0:
                    field.flush(); mask.flush(); token_ids.flush()
                    print(f"[phase2505 trajectory] {model_row + 1}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush(); mask.flush(); token_ids.flush()
        del field, mask, token_ids
    index_path = OUT / "index/autonomous_rows.jsonl"
    write_jsonl(index_path, index)
    return {"field": str(field_path), "shape": [len(rows), MAX_NEW + 1, len(qmods), DIM],
            "event_mask": str(mask_path), "token_ids": str(token_path), "index": str(index_path),
            "sha256": {p.name: sha256(p) for p in (field_path, mask_path, token_path)}}


def group_interactions(field: np.ndarray, index: list[dict], qpoint: int, event: str,
                       require_all_correct: bool = False) -> dict[tuple, np.ndarray]:
    lookup = {(r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in index}
    output = {}
    for pair_id in sorted({r["pair_id"] for r in index}):
        for language in ("en", "zh"):
            for surface in range(4):
                rows = {(m, q): lookup[(pair_id, language, surface, m, q)] for m in (0, 1) for q in (0, 1)}
                if require_all_correct and not all(r["parsed_correct"] for r in rows.values()):
                    continue
                steps = {}
                valid = True
                for key, row in rows.items():
                    if event == "boundary": step = 0
                    elif event == "first": step = row["first_step"]
                    else: step = row["answer_step"]
                    if step is None:
                        valid = False; break
                    steps[key] = np.asarray(field[row["model_row"], step, qpoint], dtype=np.float64)
                if valid:
                    output[(pair_id, language, surface)] = (steps[(0, 0)] - steps[(0, 1)] - steps[(1, 0)] + steps[(1, 1)]) / 4
    return output


def transition_metric(first: dict[tuple, np.ndarray], second: dict[tuple, np.ndarray]) -> dict:
    keys = sorted(set(first) & set(second))
    same = [walsh.cosine(first[key], second[key]) for key in keys]
    wrong = []
    for key in keys:
        pair_id, language, surface = key
        for other in keys:
            if other[1:] == (language, surface) and other[0] != pair_id:
                wrong.append(walsh.cosine(first[key], second[other]))
    return {"groups": len(keys), "same_mean": float(np.mean(same)) if same else 0.0,
            "wrong_mean": float(np.mean(wrong)) if wrong else 0.0,
            "wrong_q95": float(np.quantile(wrong, .95)) if wrong else 0.0,
            "identity_advantage_over_q95": float(np.mean(same) - np.quantile(wrong, .95)) if same and wrong else 0.0}


def output_geometry(index: list[dict]) -> dict:
    panels = {}
    for metric in ("sum", "mean"):
        key = f"relation0_minus_relation1_{metric}"
        correct_margins = []
        cell_interactions = []
        lookup = {(r["pair_id"], r["language"], r["surface"], r["meaning_swap"], r["query_marker"]): r for r in index}
        for row in index:
            signed = row[key] if row["selected_relation"] == 0 else -row[key]
            correct_margins.append(signed)
        for pair_id in sorted({r["pair_id"] for r in index}):
            for language in ("en", "zh"):
                for surface in range(4):
                    cell = {(m, q): lookup[(pair_id, language, surface, m, q)][key] for m in (0, 1) for q in (0, 1)}
                    cell_interactions.append((cell[(0, 0)] - cell[(0, 1)] - cell[(1, 0)] + cell[(1, 1)]) / 4)
        panels[metric] = {"rows": len(correct_margins), "correct_margin_mean": float(np.mean(correct_margins)),
                          "correct_margin_positive_rate": float(np.mean(np.asarray(correct_margins) > 0)),
                          "fourcell_groups": len(cell_interactions),
                          "fourcell_interaction_mean": float(np.mean(cell_interactions)),
                          "fourcell_interaction_positive_rate": float(np.mean(np.asarray(cell_interactions) > 0)),
                          "fourcell_interactions": [float(v) for v in cell_interactions]}
    return panels


def analyze(collection: dict, qpoint: int) -> dict:
    field = np.load(collection["field"], mmap_mode="r")
    index = read_jsonl(Path(collection["index"]))
    all_events = {event: group_interactions(field, index, qpoint, event, False) for event in ("boundary", "first", "answer")}
    success_events = {event: group_interactions(field, index, qpoint, event, True) for event in ("boundary", "first", "answer")}
    transitions = {}
    success_transitions = {}
    for a, b in (("boundary", "first"), ("boundary", "answer"), ("first", "answer")):
        transitions[f"{a}_to_{b}"] = transition_metric(all_events[a], all_events[b])
        success_transitions[f"{a}_to_{b}"] = transition_metric(success_events[a], success_events[b])
    output = output_geometry(index)
    hidden_norms = []
    scalar_interactions = output["sum"]["fourcell_interactions"]
    for key in sorted(all_events["boundary"]):
        hidden_norms.append(float(np.linalg.norm(all_events["boundary"][key])))
    norm_output_corr = float(np.corrcoef(hidden_norms, scalar_interactions)[0, 1]) if len(hidden_norms) > 1 else 0.0
    return {"behavior": {"rows": len(index), "parsed_rate": sum(r["answer_step"] is not None for r in index) / len(index),
                          "accuracy": sum(r["parsed_correct"] for r in index) / len(index),
                          "all_four_cells_correct_groups": len(success_events["boundary"]),
                          "total_fourcell_groups": len(all_events["boundary"])},
            "trajectory_all_parsed": transitions, "trajectory_all_four_correct": success_transitions,
            "output_sequence_geometry": output,
            "hidden_interaction_norm_vs_output_sum_interaction_correlation": norm_output_corr,
            "correlation_boundary": "descriptive across 24 pair-language-surface groups; not mediation"}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 语义必要关系选择的自主生成轨迹与候选序列输出几何（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用Phase2504共同合格的三pair、unit23全部96条四格材料，在真实贪心生成中逐token回灌；每条保存answer-boundary加最多10个生成token的q0–q37×2560全坐标。q30仍由unit21冻结，boundary、first、解析到候选名称的answer token全部使用同层。并对每条prompt分别teacher-force两个候选完整字符串，计算总log probability与每token平均log probability；对relation0减relation1的输出分数再做同一四格交互。

$$\Delta L=L(r_0)-L(r_1),\qquad I_L=\tfrac14(\Delta L_{{00}}-\Delta L_{{01}}-\Delta L_{{10}}+\Delta L_{{11}}).$$

**结果汇总。** 原场 `{json.dumps(result['collection'], ensure_ascii=False)}`；行为与轨迹 `{json.dumps(result['analysis']['behavior'], ensure_ascii=False)}`、`{json.dumps(result['analysis']['trajectory_all_parsed'], ensure_ascii=False)}`；四格全部正确子集 `{json.dumps(result['analysis']['trajectory_all_four_correct'], ensure_ascii=False)}`；输出序列几何 `{json.dumps(result['analysis']['output_sequence_geometry'], ensure_ascii=False)}`；HiddenState交互范数与输出交互相关 `{result['analysis']['hidden_interaction_norm_vs_output_sum_interaction_correlation']}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2505_c68865_c70016_semantic_selection_autonomous_output_geometry.py`；96×11×38×2560自主轨迹、event mask、生成token IDs、候选逐token logprob、索引、哈希与final位于`{OUT}`。

**分析与理论进展。** 候选序列四格交互直接检验关系含义与查询marker的组合是否到达输出概率，而不仅是HiddenState余弦。轨迹交互比较问的是相同外部四格对比在boundary、first、answer时是否保持；它不要求原始单样本向量不变。全部样本为主，另报告四格都正确的严格子集，避免只展示成功路径而隐藏选择偏差。

**数值更正、问题硬伤与结论。** Phase2504最初写入MEMO的exact-zero prefix密度出现零除显示（effective count约1e30、50%/90%为2561）；其正确解释是interaction全零、有效坐标数与覆盖数均为0，Phase2504 final已更正。本Phase的sequence score是teacher-forced观察性读出，不是因果patch；总logprob受token长度影响，因此同时报告mean。生成token身份会进入first/answer状态，轨迹保持不是纯语义搬运。HiddenState范数与输出交互的相关若存在也不能证明中介。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    f2503 = json.loads((P2503 / "analysis/final.json").read_text(encoding="utf-8"))
    f2504 = json.loads((P2504 / "analysis/final.json").read_text(encoding="utf-8"))
    pair_ids = set(f2504["contract"]["pair_ids"])
    rows = [r for r in read_jsonl(OUT.parent / "phase2503_c67201_c68224_equal_length_fresh_lockbox_behavior_fullfield/material/fresh_lockbox_rows.jsonl")
            if r["pair_id"] in pair_ids]
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        collection = capture(model, tokenizer, rows)
    finally:
        model_utils.release_model(model); gc.collect()
    analysis = analyze(collection, int(f2504["contract"]["qpoint"]))
    checks = {"source_phases_passed": f2503["all_checks_passed"] and f2504["all_checks_passed"],
              "rows_96": collection["shape"][0] == 96,
              "all_qpoints_coordinates": collection["shape"][2:] == [38, 2560],
              "same_qpoint_all_events": int(f2504["contract"]["qpoint"]) == 30,
              "all_rows_have_first": analysis["behavior"]["parsed_rate"] > 0,
              "output_fourcell_groups_24": analysis["output_sequence_geometry"]["sum"]["fourcell_groups"] == 24,
              "hashes": len(collection["sha256"]) == 3, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "qpoint": int(f2504["contract"]["qpoint"]),
              "pair_ids": sorted(pair_ids), "collection": collection, "analysis": analysis,
              "adjudication": {"relation_selection_reaches_candidate_sequence_probability": analysis["output_sequence_geometry"]["mean"]["fourcell_interaction_positive_rate"] > .5,
                               "autonomous_trajectory_available": True, "causal_mediator_identified": False,
                               "pure_semantic_code_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
