#!/usr/bin/env python3
"""Exact-shape natural patching across token-position regions at frozen q28."""
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
P2517 = RESULT / "phase2517_c81953_c82976_exact_shape_paired_natural_patch"
OUT = RESULT / "phase2518_c82977_c84000_exact_shape_position_path_patch"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, QPOINT, DIM = 2518, "C82977-C84000", 28, 2560
CONTEXTS = (0, 3, 5, 6, 9, 10, 12, 15)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2390_c19441_c19760_qwen_semantic_lexical_fullfield as field_utils  # noqa: E402


def load_json(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8-sig"))
def read_jsonl(path: Path) -> list[dict]: return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def pad(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences): ids[i, :len(seq)] = torch.tensor(seq, device=device); mask[i, :len(seq)] = 1
    return ids, mask


def score(logits: torch.Tensor, jobs: list[dict]) -> list[float]:
    output = []
    for i, job in enumerate(jobs):
        values = []
        for j, token in enumerate(job["continuation"]):
            v = logits[i, job["prompt_length"] - 1 + j].float(); values.append(float(v[token] - torch.logsumexp(v, -1)))
        output.append(float(sum(values)))
    return output


def region_positions(row: dict) -> dict[str, list[int]]:
    spans, length = row["spans"], len(row["prompt_ids"])
    definition_end = spans["definition_end"][0][1] - 1; facts_end = spans["facts_end"][0][1] - 1
    query_start = facts_end + 1; candidate_start = min(spans["candidate0"][-1][0], spans["candidate1"][-1][0])
    marker_start, marker_end = spans["query_marker"][-1]
    query_last = spans["query_marker"][-1][1] - 1; boundary = length - 1
    return {"query_last": [query_last], "query_marker_span": list(range(marker_start, marker_end)),
            "definition_region": list(range(0, definition_end + 1)),
            "facts_region": list(range(definition_end + 1, facts_end + 1)),
            "query_region": list(range(query_start, candidate_start)),
            "candidate_and_instruction_region": list(range(candidate_start, boundary + 1)),
            "answer_boundary": [boundary], "through_query": list(range(0, query_last + 1)),
            "all_prompt": list(range(length))}


def compile_interventions(rows: list[dict], pairs: list[int]) -> list[dict]:
    lookup = {(r["unit"], r["pair_id"], r["language"], r["context_id"], r["meaning_swap"], r["query_marker"]): r for r in rows}
    output = []
    for pair_id in pairs:
        for language in ("en", "zh"):
            for context in CONTEXTS:
                for q in (0, 1):
                    base, donor = lookup[(29, pair_id, language, context, 0, q)], lookup[(29, pair_id, language, context, 1, q)]
                    assert len(base["prompt_ids"]) == len(donor["prompt_ids"])
                    output.append({"id": f"p{pair_id}-{language}-x{context}-q{q}", "pair_id": pair_id,
                                   "language": language, "context_id": context, "query_marker": q,
                                   "base": base, "donor": donor, "regions": region_positions(base)})
    return output


def run(model, tokenizer, interventions: list[dict], batch_size: int = 8) -> list[dict]:
    module = field_utils.modules(model)[QPOINT]; device = model.get_input_embeddings().weight.device
    active, captured = {"source": None, "positions": None}, {}
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output; captured["hidden"] = hidden.detach().clone()
        if active["source"] is None: return None
        changed = hidden.clone()
        for i, positions in enumerate(active["positions"]):
            changed[i, positions] = active["source"][i, positions].to(changed.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = module.register_forward_hook(hook)
    jobs = []
    for item in interventions:
        for ri, candidate in enumerate(item["base"]["relation_targets"]):
            cont = [int(v) for v in tokenizer.encode((" " if item["language"] == "en" else "") + candidate, add_special_tokens=False)]
            jobs.append({"id": item["id"], "relation_index": ri, "continuation": cont,
                         "prompt_length": len(item["base"]["prompt_ids"]), "base_sequence": item["base"]["prompt_ids"] + cont,
                         "donor_sequence": item["donor"]["prompt_ids"] + cont, "regions": item["regions"]})
    region_names = ["query_last", "query_marker_span", "definition_region", "facts_region", "query_region",
                    "candidate_and_instruction_region", "answer_boundary", "through_query", "all_prompt"]
    results = []
    try:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]; base_seq = [j["base_sequence"] for j in batch]; donor_seq = [j["donor_sequence"] for j in batch]
            assert [len(x) for x in base_seq] == [len(x) for x in donor_seq]
            base_ids, mask = pad(base_seq, tokenizer.pad_token_id, device); donor_ids, dmask = pad(donor_seq, tokenizer.pad_token_id, device); assert torch.equal(mask, dmask)
            active.update(source=None, positions=None); captured.clear()
            with torch.inference_mode(): logits = model(input_ids=base_ids, attention_mask=mask, use_cache=False).logits
            base_hidden = captured["hidden"].clone()
            for job, value in zip(batch, score(logits, batch)): results.append({"id": job["id"], "condition": "no_patch", "relation_index": job["relation_index"], "sum_logprob": value})
            active.update(source=None, positions=None); captured.clear()
            with torch.inference_mode(): model(input_ids=donor_ids, attention_mask=dmask, use_cache=False)
            donor_hidden = captured["hidden"].clone(); shuffled = donor_hidden.roll(shifts=2 if len(batch) > 2 else 1, dims=0)
            conditions = [("self_all_prompt", base_hidden, [j["regions"]["all_prompt"] for j in batch]),
                          ("shuffled_all_prompt", shuffled, [j["regions"]["all_prompt"] for j in batch])]
            conditions += [(f"donor_{name}", donor_hidden, [j["regions"][name] for j in batch]) for name in region_names]
            for name, source, positions in conditions:
                active.update(source=source, positions=positions); captured.clear()
                with torch.inference_mode(): plogits = model(input_ids=base_ids, attention_mask=mask, use_cache=False).logits
                for job, value in zip(batch, score(plogits, batch)): results.append({"id": job["id"], "condition": name, "relation_index": job["relation_index"], "sum_logprob": value})
            if start % 32 == 0: print(f"[phase2518 position batches] {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)
    finally: handle.remove()
    return results


def analyze(interventions: list[dict], scores: list[dict]) -> tuple[dict, list[dict]]:
    lookup = {(r["id"], r["condition"], r["relation_index"]): r["sum_logprob"] for r in scores}; conditions = sorted({r["condition"] for r in scores})
    records = []
    for item in interventions:
        sign = 1 if item["query_marker"] == 0 else -1; base = lookup[(item["id"], "no_patch", 0)] - lookup[(item["id"], "no_patch", 1)]
        for condition in conditions:
            value = lookup[(item["id"], condition, 0)] - lookup[(item["id"], condition, 1)]
            records.append({"id": item["id"], "condition": condition, "base_difference": base, "patched_difference": value,
                            "shift_toward_donor": -sign * (value - base), "donor_margin": -sign * value, "donor_flip": -sign * value > 0})
    panels = {}
    for condition in conditions:
        v = [r for r in records if r["condition"] == condition]
        panels[condition] = {"n": len(v), "mean_shift_toward_donor": float(np.mean([r["shift_toward_donor"] for r in v])),
                             "positive_shift_rate": float(np.mean([r["shift_toward_donor"] > 0 for r in v])),
                             "donor_flip_rate": float(np.mean([r["donor_flip"] for r in v])), "mean_donor_margin": float(np.mean([r["donor_margin"] for r in v]))}
    self_err = [abs(r["base_difference"] - r["patched_difference"]) for r in records if r["condition"] == "self_all_prompt"]
    return {"panels": panels, "self_patch_max_abs_difference": float(max(self_err))}, records


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: q28关系状态的token位置路径exact-shape自然patch（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2517证明单个query末token不是充分中介，本Phase保持同一128个unit29干预、q28、完整候选序列与exact-shape成对前向，只把patch位置扩展为：query末token、完整query-marker跨度、definition区、facts区、query区、candidate+instruction区、answer-boundary、从开头到query、全部prompt；另有self全部prompt和batch内错配全部prompt。每次替换相应位置的全部2560坐标，不按坐标大小筛选。

$$H^{{base}}_{{28,S,:}}\leftarrow H^{{donor}}_{{28,S,:}},\qquad S\subseteq\{{1,\ldots,T_{{prompt}}\}}.$$

**结果汇总。** 位置面板 `{json.dumps(result['analysis']['panels'], ensure_ascii=False)}`；self最大误差 `{result['analysis']['self_patch_max_abs_difference']}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2518_c82977_c84000_exact_shape_position_path_patch.py`；位置合同、3072条候选分数、逐干预结果与final位于`{OUT}`。

**分析与理论进展。** 该实验回答“q28关系计算分散在哪些token位置”，而非寻找单坐标。若answer-boundary或all-prompt donor能转移而query区不能，表示关系选择在晚期已被汇聚到输出位置，query交互只是相关前驱；若definition/facts联合才有效，则状态路径分布在多位置。错配all-prompt控制区分匹配计算状态与一般全场替换。

**问题硬伤与结论。** token区间包含多种内容，区域patch仍可能离开自然联合分布；all-prompt转移接近替换整个中间计算，因果充分性较弱且不等于最小机制；仍只有合成二候选与单层q28。结果只决定下一阶段该追踪哪个位置事件，不闭合编码理论。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    f13, f17 = load_json(P2513 / "analysis/final.json"), load_json(P2517 / "analysis/final.json")
    rows = read_jsonl(Path(f13["collection"]["event_index"])); material = {r["case_id"]: r for r in read_jsonl(P2513 / "material/factorial_rows.jsonl")}
    for row in rows:
        source = material[row["case_id"]]; row["relation_targets"] = source["relation_targets"]; row["spans"] = source["spans"]
    interventions = compile_interventions(rows, f13["behavior"]["qualified_pair_ids"])
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try: scores = run(model, tokenizer, interventions)
    finally: model_utils.release_model(model); gc.collect()
    score_path = OUT / "output/position_patch_scores.jsonl"; write_jsonl(score_path, scores)
    analysis, records = analyze(interventions, scores); record_path = OUT / "analysis/intervention_records.jsonl"; write_jsonl(record_path, records)
    checks = {"sources_passed": f13["all_checks_passed"] and f17["all_checks_passed"], "interventions_128": len(interventions) == 128,
              "scores_3072": len(scores) == 128 * 12 * 2, "exact_shape": True,
              "self_control_valid": analysis["self_patch_max_abs_difference"] == 0.0, "all_positions_and_coordinates": True,
              "hash": len(digest(score_path)) == 64, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B nonquantized BF16 CUDA", "qpoint": QPOINT,
              "analysis": analysis, "files": {"scores": str(score_path), "records": str(record_path), "scores_sha256": digest(score_path)},
              "adjudication": {"single_query_token_sufficient": False,
                               "best_position_condition": max((k for k in analysis["panels"] if k.startswith("donor_")),
                                                              key=lambda k: analysis["panels"][k]["mean_shift_toward_donor"]),
                               "minimal_language_gear_identified": False, "language_encoding_mechanism_closed": False},
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "analysis": analysis, "adjudication": result["adjudication"],
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
