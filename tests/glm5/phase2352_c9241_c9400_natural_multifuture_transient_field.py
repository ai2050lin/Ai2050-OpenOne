#!/usr/bin/env python3
"""Collect a broad natural multi-future prompt and generation-token full-coordinate field."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2352_c9241_c9400_natural_multifuture_transient_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
MATERIAL = OUT / "material/natural_multifuture_graphs.jsonl"
STATES = OUT / "raw/prompt_boundary_all_checkpoints.float16.npy"
DECISIONS = OUT / "raw/teacher_forced_decisions.float32.npy"
TRAJECTORY = OUT / "raw/generation_all_checkpoint_trajectory.float16.npy"
GENERATION = OUT / "raw/fresh_lockbox_generation.jsonl"
PROGRESS = OUT / "raw/progress.json"
PHASE = 2352
CAMPAIGN = "C9241-C9400"
FAMILIES = (
    "taxonomy", "attitude", "event", "causal", "spatial", "possession",
    "partwhole", "temporal", "coreference", "negation", "translation", "grammar",
)
RELATIONS = {
    "taxonomy": ("is_a", "subtype_of", "belongs_to", "classified_under", "contained_by"),
    "attitude": ("likes", "prefers", "seeks", "values", "remembers"),
    "event": ("triggers", "enables", "precedes", "changes", "completes"),
    "causal": ("causes", "produces", "moves", "opens", "reveals"),
    "spatial": ("inside", "north_of", "near", "behind", "within"),
    "possession": ("owns", "stores", "contains", "guards", "holds"),
    "partwhole": ("part_of", "section_of", "component_of", "member_of", "unit_of"),
    "temporal": ("before", "during", "after", "until", "follows"),
    "coreference": ("names", "refers_to", "describes", "identifies", "tracks"),
    "negation": ("excludes", "blocks", "avoids", "rejects", "cancels"),
    "translation": ("translates_to", "means", "paraphrases", "renders_as", "denotes"),
    "grammar": ("modifies", "coordinates", "governs", "attaches_to", "scopes_over"),
}
LANGUAGES = ("en", "zh")
SURFACES = ("direct", "natural")
QUERIES = ("source", "first", "penultimate", "terminal")
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
UNITS = 16
GEN_STEPS = 12
NAMES_A = ("Aster", "Birch", "Cedar", "Dahlia", "Ember", "Flint", "Grove", "Hazel",
           "Indigo", "Juniper", "Kestrel", "Lumen", "Maple", "Nectar", "Olive", "Pine")
NAMES_B = ("Vale", "Crown", "Harbor", "Field", "Stone", "River", "Meadow", "Bridge")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def node_names(family_index: int, unit: int, state: int, count: int = 7) -> list[str]:
    base = family_index * 37 + unit + state * 211
    return [f"{NAMES_A[(base + k * 3) % len(NAMES_A)]} {NAMES_B[(base * 3 + k * 5) % len(NAMES_B)]}"
            for k in range(count)]


def compile_material(tokenizer) -> tuple[list[dict], dict]:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for language in LANGUAGES:
            for surface in SURFACES:
                for unit in range(UNITS):
                    partition = PARTITIONS[unit // 4]
                    depth = 2 + unit % 4
                    for state in (0, 1):
                        nodes = node_names(family_index, unit, state)
                        rels = RELATIONS[family]
                        edges = [[nodes[k], rels[k], nodes[k + 1]] for k in range(depth)]
                        foil = nodes[depth + 1]
                        graph_hash = hashlib.sha256(json.dumps(edges, sort_keys=True).encode()).hexdigest()
                        if language == "en":
                            direct = " ".join(f"{a} {r} {b}." for a, r, b in edges)
                            natural = "A report says " + "; then ".join(f"{a} is linked by {r} to {b}" for a, r, b in edges) + "."
                            questions = {"source": "Who starts the chain?", "first": "What is the first reached name?",
                                         "penultimate": "What name appears just before the endpoint?",
                                         "terminal": "What is the final endpoint?"}
                            suffix = "Reply with only <answer>the exact two-word name</answer> and stop.\nAnswer:"
                        else:
                            direct = "".join(f"{a}通过{r}连接{b}。" for a, r, b in edges)
                            natural = "记录显示：" + "；随后".join(f"{a}经由{r}连接到{b}" for a, r, b in edges) + "。"
                            questions = {"source": "链条从哪个名称开始？", "first": "第一步到达哪个名称？",
                                         "penultimate": "终点之前的名称是什么？", "terminal": "最终终点是什么？"}
                            suffix = "只用<answer>精确的双词名称</answer>作答，然后停止。\n答案："
                        facts = direct if surface == "direct" else natural
                        targets = {"source": nodes[0], "first": nodes[1], "penultimate": nodes[depth - 1],
                                   "terminal": nodes[depth]}
                        for query in QUERIES:
                            target = targets[query]
                            prompt = f"{facts}\n{questions[query]}\n{suffix}"
                            prompt_ids = [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]
                            answer_text = f" <answer>{target}</answer>"
                            wrong_text = f" <answer>{foil}</answer>"
                            rows.append({"case_id": f"c9241-{family}-{language}-{surface}-u{unit:02d}-s{state}-{query}",
                                         "design_index": len(rows), "family": family, "family_index": family_index,
                                         "language": language, "surface": surface, "unit": unit, "state": state,
                                         "partition": partition, "depth": depth, "query": query, "graph": edges,
                                         "graph_hash": graph_hash, "prompt": prompt, "prompt_ids": prompt_ids,
                                         "target": target, "foil": foil,
                                         "target_ids": [int(x) for x in tokenizer.encode(answer_text, add_special_tokens=False)],
                                         "wrong_ids": [int(x) for x in tokenizer.encode(wrong_text, add_special_tokens=False)]})
    hashes = defaultdict(set)
    for row in rows:
        hashes[(row["family"], row["surface"], row["unit"], row["state"])].add(row["graph_hash"])
    return rows, {"rows": len(rows), "families": len(FAMILIES), "units": UNITS, "languages": list(LANGUAGES),
                  "surfaces": list(SURFACES), "queries": list(QUERIES), "depths": [2, 3, 4, 5],
                  "parallel_graphs": all(len(v) == 1 for v in hashes.values()),
                  "unique_cases": len({r["case_id"] for r in rows}) == len(rows),
                  "prompt_token_range": [min(len(r["prompt_ids"]) for r in rows), max(len(r["prompt_ids"]) for r in rows)]}


def candidate_score(model, device, batch: list[dict], key: str, capture: dict[int, torch.Tensor], pad: int) -> np.ndarray:
    combined = [row["prompt_ids"] + row[key] for row in batch]
    ids, mask, positions = baseline.pad_right(combined, device, pad)
    capture.clear()
    output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
    scores = []
    for local, row in enumerate(batch):
        answer = row[key]; start = len(row["prompt_ids"])
        pos = torch.arange(start - 1, start + len(answer) - 1, device=device)
        logits = model.lm_head(output.last_hidden_state[local, pos]).float()
        token_ids = torch.tensor(answer, dtype=torch.long, device=device)
        scores.append(float(F.log_softmax(logits, dim=-1)[torch.arange(len(answer), device=device), token_ids].mean()))
    return np.asarray(scores, dtype=np.float32)


def collect_prompt_field(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    qmodules = modules(model); shape = (len(rows), len(qmodules), int(model.config.hidden_size))
    completed = 0
    if STATES.exists() and DECISIONS.exists() and PROGRESS.exists():
        completed = int(json.loads(PROGRESS.read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(STATES, mode="r+"); decisions = np.lib.format.open_memmap(DECISIONS, mode="r+")
    else:
        STATES.parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(STATES, mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(DECISIONS, mode="w+", dtype=np.float32, shape=(len(rows), 4))
    capture: dict[int, torch.Tensor] = {}; handles = []
    for qpoint, module in enumerate(qmodules):
        def hook(_m, _i, value, qpoint=qpoint): capture[qpoint] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                good = candidate_score(model, device, batch, "target_ids", capture, pad)
                for qpoint in range(len(qmodules)):
                    states[start:start + len(batch), qpoint] = torch.stack(
                        [capture[qpoint][i, len(row["prompt_ids"]) - 1] for i, row in enumerate(batch)]
                    ).float().cpu().numpy().astype(np.float16)
                bad = candidate_score(model, device, batch, "wrong_ids", capture, pad); margin = good - bad
                decisions[start:start + len(batch)] = np.stack([good, bad, margin, margin > 0], axis=1)
                states.flush(); decisions.flush(); save(PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                if (start + len(batch)) % 192 == 0 or start + len(batch) == len(rows):
                    print(f"[phase2352 prompt] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        states.flush(); decisions.flush(); close_memmap(states); close_memmap(decisions)
    return {"shape": list(shape), "batch_size": batch_size}


def generate_trajectory(model, tokenizer, device, rows: list[dict], batch_size: int = 8) -> dict:
    selected = [row for row in rows if row["partition"] == "fresh_lockbox" and row["surface"] == "natural" and row["state"] == 0]
    qmodules = modules(model); shape = (len(selected), GEN_STEPS, len(qmodules), int(model.config.hidden_size))
    trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="w+", dtype=np.float16, shape=shape)
    records = []; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    capture: dict[int, torch.Tensor] = {}; handles = []
    for qpoint, module in enumerate(qmodules):
        def hook(_m, _i, value, qpoint=qpoint): capture[qpoint] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for start in range(0, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                max_len = max(len(row["prompt_ids"]) for row in batch)
                ids = torch.full((len(batch), max_len), pad, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for i, row in enumerate(batch):
                    seq = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
                    ids[i, -len(seq):] = seq; mask[i, -len(seq):] = 1
                generated = []
                past = None; current = ids
                for step in range(GEN_STEPS):
                    capture.clear()
                    output = model(input_ids=current, attention_mask=mask, past_key_values=past, use_cache=True, return_dict=True)
                    for qpoint in range(len(qmodules)):
                        trajectory[start:start + len(batch), step, qpoint] = capture[qpoint][:, -1].float().cpu().numpy().astype(np.float16)
                    token = output.logits[:, -1].argmax(dim=-1); generated.append(token)
                    past = output.past_key_values; current = token[:, None]
                    mask = torch.cat([mask, torch.ones((len(batch), 1), dtype=mask.dtype, device=device)], dim=1)
                tokens = torch.stack(generated, dim=1).cpu().tolist()
                for row, token_ids in zip(batch, tokens):
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    tag = re_search_answer(text)
                    records.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                                    "query": row["query"], "depth": row["depth"], "unit": row["unit"],
                                    "target": row["target"], "generated": text, "parsed_answer": tag,
                                    "raw_exact": text == f"<answer>{row['target']}</answer>", "parsed_exact": tag == row["target"],
                                    "token_ids": token_ids})
                trajectory.flush(); print(f"[phase2352 generation] {min(start + len(batch), len(selected))}/{len(selected)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        trajectory.flush(); close_memmap(trajectory)
    io.write_rows(GENERATION, records)
    return {"rows": len(selected), "shape": list(shape), "steps": GEN_STEPS}


def re_search_answer(text: str) -> str:
    import re
    match = re.search(r"<answer>\s*([^<\n]+?)\s*</answer>", text)
    return match.group(1).strip() if match else ""


def behavior(rows: list[dict]) -> dict:
    decisions = np.load(DECISIONS, mmap_mode="r")
    qualified = []; family_cells = {}
    for family in FAMILIES:
        cells = {}
        for language in LANGUAGES:
            for surface in SURFACES:
                for partition in PARTITIONS:
                    idx = [i for i, row in enumerate(rows) if row["family"] == family and row["language"] == language
                           and row["surface"] == surface and row["partition"] == partition]
                    cells[f"{language}:{surface}:{partition}"] = float(np.mean(decisions[idx, 3]))
        family_cells[family] = {"minimum": min(cells.values()), "cells": cells}
        if min(cells.values()) >= 0.75: qualified.append(family)
    overall = float(np.mean(decisions[:, 3])); close_memmap(decisions)
    generated = io.read_rows(GENERATION)
    return {"teacher_forced_overall": overall, "family_cells": family_cells, "qualified": qualified,
            "generation": {"rows": len(generated), "raw_exact": float(np.mean([r["raw_exact"] for r in generated])),
                           "parsed_tag_exact": float(np.mean([r["parsed_exact"] for r in generated])),
                           "first_line_exact": float(np.mean([r["generated"].splitlines()[0].strip() == r["target"] for r in generated])),
                           "target_anywhere": float(np.mean([r["target"] in r["generated"] for r in generated]))}}


def publish(rows: list[dict], generation_info: dict) -> list[dict]:
    datasets = []
    source = np.load(STATES, mmap_mode="r"); qpoints = [0, 12, 24, 35, 37]
    dataset_id = "c9241_qwen4b_natural_multifuture_prompt_field"
    binary = VIS / f"{dataset_id}.float16.npy"; out = atlas.create_binary(binary.name, len(rows) * len(qpoints), 2560, np.float16)
    metadata = []; cursor = 0
    for qpoint in qpoints:
        out[cursor:cursor + len(rows)] = source[:, qpoint]
        metadata.extend({"case_id": r["case_id"], "family": r["family"], "language": r["language"], "surface": r["surface"],
                         "query": r["query"], "depth": r["depth"], "partition": r["partition"], "unit": r["unit"],
                         "state": r["state"], "qpoint": qpoint, "field": "hiddenstate"} for r in rows)
        cursor += len(rows)
    out.flush(); close_memmap(out); close_memmap(source)
    datasets.append(atlas.write_metadata(dataset_id, "Qwen3-4B natural multi-future prompt-boundary field", binary, metadata,
        "Qwen3-4B-FP16", "natural_multifuture_prompt_field_v1", "observational full-coordinate prompt field",
        "12 families x bilingual x two surfaces x variable 2-5 edge depth x four queries",
        "embedding and four concrete HiddenState checkpoints, every one of 2560 coordinates retained",
        {"phase": PHASE, "campaign": CAMPAIGN, "qpoints": qpoints, "coordinate_count": 2560, "no_topk": True}))
    traj = np.load(TRAJECTORY, mmap_mode="r"); flat = traj.reshape(-1, traj.shape[-1])
    dataset_id = "c9242_qwen4b_natural_generation_token_trajectory"
    binary = VIS / f"{dataset_id}.float16.npy"; out = atlas.create_binary(binary.name, flat.shape[0], 2560, np.float16); out[:] = flat
    out.flush(); close_memmap(out)
    generated_rows = io.read_rows(GENERATION); metadata = []
    for row in generated_rows:
        for step in range(GEN_STEPS):
            for qpoint in range(38):
                metadata.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"],
                                 "query": row["query"], "depth": row["depth"], "unit": row["unit"], "step": step,
                                 "qpoint": qpoint, "generated_token_id": row["token_ids"][step], "field": "hiddenstate"})
    close_memmap(traj)
    datasets.append(atlas.write_metadata(dataset_id, "Qwen3-4B generation-token all-checkpoint trajectory", binary, metadata,
        "Qwen3-4B-FP16", "natural_generation_token_trajectory_v1", "observational transient generation trajectory",
        "384 fresh-lockbox natural prompts x 12 generated positions x embedding+36 blocks+final norm",
        "every generated position, checkpoint and all 2560 coordinates retained",
        {"phase": PHASE, "campaign": CAMPAIGN, "coordinate_count": 2560, "no_topk": True,
         "shape4d": generation_info["shape"]}))
    return datasets


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 十二族变深度自然多未来指令与生成token瞬态全坐标场（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 建立12种语言模式族，每族16 units、两个状态、中英、direct/natural、四种查询；图深随unit在2–5条边间轮换，避免把族固定绑定为三边模板，共6144条。答案改成双词自然名称并要求`<answer>...</answer>`后停止；对完整目标/错答序列计算teacher-forced平均对数概率。对384条fresh-lockbox自然提示自主生成12 token，并保存每个生成位置的embedding、36 block和final norm的全部2560坐标。

$$
\ell(y|x)=|y|^{{-1}}\sum_k\log p(y_k|x,y_{{<k}}),\qquad
T_{{i,s,q}}=H_q(x_i,\hat y_{{i,<s}})\in\mathbb R^{{2560}}.
$$

**结果汇总。** 材料 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；采集 `{json.dumps(result['collection'], ensure_ascii=False)}`；行为与生成 `{json.dumps(result['behavior'], ensure_ascii=False)}`；可视化 `{json.dumps(result['datasets'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2352_c9241_c9400_natural_multifuture_transient_field.py`；结果 `tests/glm5/result/phase2352_c9241_c9400_natural_multifuture_transient_field`；客户端数据`c9241/c9242`。

**理论进展、问题硬伤与结论。** 该场首次把prompt边界静态图谱与模型自身生成历史下的逐token瞬态放在同一大样本合同中，并交叉图深、语言、表述和查询角色；但双词匿名名称仍非开放世界回答，teacher forcing仍只比较一个冻结foil，自主生成的标签解析正确也不等于自然解释正确。瞬态场是观测对象，不自动构成“流形”或因果齿轮。下一Phase只在行为合格族上，以未见unit锁箱比较条件残差、配对交互、有符号变化和具体坐标排序控制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    rows, material_audit = compile_material(tokenizer); io.write_rows(MATERIAL, rows)
    freeze = {"frozen_before_model_load": True, "behavior_threshold": 0.75, "families": list(FAMILIES),
              "coordinate_policy": "all coordinates; no Top-K/PCA/projection", "generation_steps": GEN_STEPS,
              "selection": "fresh_confirmation", "adjudication": "fresh_lockbox"}
    save(OUT / "config/frozen_contract.json", freeze)
    model = None
    try:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        prompt_info = collect_prompt_field(model, device, rows)
        generation_info = generate_trajectory(model, tokenizer, device, rows)
    finally:
        if model is not None: model_utils.release_model(model)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    behavior_result = behavior(rows); datasets = publish(rows, generation_info)
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(v for k, v in row.items() if k != "id") for row in verification)
    catalog = atlas.update_catalog(datasets); build = atlas.frontend_build()
    if not (verified and build["passed"]): raise RuntimeError((verification, build))
    traj_size = TRAJECTORY.stat().st_size; TRAJECTORY.unlink()
    cleanup = {"deleted_visualized_trajectory_duplicate": str(TRAJECTORY), "bytes_reclaimed": traj_size,
               "prompt_field_retained_for_phase2353": str(STATES)}
    checks = {"rows": material_audit["rows"] == 6144, "parallel_graphs": material_audit["parallel_graphs"],
              "prompt_shape": prompt_info["shape"] == [6144, 38, 2560],
              "generation_shape": generation_info["shape"] == [384, 12, 38, 2560], "assets": verified,
              "frontend_build": build["passed"], "trajectory_duplicate_deleted": not TRAJECTORY.exists()}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "freeze": freeze, "material_audit": material_audit,
              "collection": {"prompt": prompt_info, "generation": generation_info}, "behavior": behavior_result,
              "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)), "verification": verification,
              "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)), "frontend_build": build,
              "cleanup": cleanup, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2352_failed", checks))
    append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
