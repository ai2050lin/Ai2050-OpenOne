#!/usr/bin/env python3
"""Map and causally test multi-layer attention-head routing into the answer boundary."""
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
P2520 = RESULT / "phase2520_c85025_c86176_natural_language_counterfactual_fullfield"
P2523 = RESULT / "phase2523_c88577_c89952_boundary_component_residual_accounting"
P2524 = RESULT / "phase2524_c89953_c91328_event_path_visualization_retention_audit"
OUT = RESULT / "phase2525_c91329_c92704_multilayer_attention_route_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, NLAYERS, NHEADS, HDIM = 2525, "C91329-C92704", 36, 32, 128
REGIONS = ("facts_prefix", "query_property", "post_query_candidates_instruction", "answer_boundary_self")
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pad(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, :len(sequence)] = 1
    return ids, mask


def region_indices(row: dict) -> list[list[int]]:
    boundary = len(row["prompt_ids"]) - 1
    facts_end = int(row["spans"]["facts_end"][-1][-1])
    query_span = row["spans"]["query_property"][-1]
    query = list(range(int(query_span[0]), int(query_span[1]) + 1))
    post = list(range(int(query_span[1]) + 1, boundary))
    return [list(range(0, facts_end + 1)), query, post, [boundary]]


def collect_attention(model, tokenizer, rows: list[dict], path: Path) -> tuple[list[dict], dict]:
    selected = [r for r in rows if r["unit"] in (30, 31) and r["surface"] == 0 and r["output_mode"] == "candidate"]
    selected.sort(key=lambda r: (r["unit"], r["family_id"], r["language"], r["meaning_swap"], r["query_property"]))
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(len(selected), NLAYERS, NHEADS, len(REGIONS)))
    device = model.get_input_embeddings().weight.device
    index_rows = []
    for start in range(0, len(selected), 4):
        batch = selected[start:start + 4]
        ids, mask = pad([r["prompt_ids"] for r in batch], tokenizer.pad_token_id, device)
        with torch.inference_mode():
            output = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                 output_attentions=True, return_dict=True)
        if output.attentions is None or len(output.attentions) != NLAYERS:
            raise RuntimeError("attention matrices unavailable")
        for bi, row in enumerate(batch):
            boundary = len(row["prompt_ids"]) - 1
            regions = region_indices(row)
            for layer, attention in enumerate(output.attentions):
                line = attention[bi, :, boundary]
                for region_index, positions in enumerate(regions):
                    field[start + bi, layer, :, region_index] = line[:, positions].sum(dim=-1).float().cpu().numpy()
            index_rows.append({k: row[k] for k in ("case_id", "unit", "family_id", "family", "language", "surface",
                                                               "output_mode", "meaning_swap", "query_property")} |
                              {"attention_row": start + bi, "answer_boundary_token": boundary,
                               "region_positions": {name: positions for name, positions in zip(REGIONS, regions)}})
    field.flush()
    return index_rows, {"path": str(path), "shape": list(field.shape), "dtype": "float32", "sha256": sha(path),
                        "regions": list(REGIONS)}


def build_interaction(field_path: Path, index_rows: list[dict], family_ids: list[int], path: Path) -> tuple[np.ndarray, dict]:
    field = np.load(field_path, mmap_mode="r")
    answer = np.zeros((2, len(family_ids), 2, NLAYERS, NHEADS, len(REGIONS)), np.float32)
    index = {(r["unit"], r["family_id"], r["language"], r["meaning_swap"], r["query_property"]): r for r in index_rows}
    for ui, unit in enumerate((30, 31)):
        for fi, family_id in enumerate(family_ids):
            for li, language in enumerate(("en", "zh")):
                cells = {(m, q): np.asarray(field[index[(unit, family_id, language, m, q)]["attention_row"]], np.float32)
                         for m in (0, 1) for q in (0, 1)}
                answer[ui, fi, li] = (cells[(0, 0)] - cells[(0, 1)] - cells[(1, 0)] + cells[(1, 1)]) / 4
    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, answer)
    return answer, {"path": str(path), "shape": list(answer.shape), "dtype": "float32", "sha256": sha(path)}


def freeze_routes(interaction: np.ndarray) -> dict:
    # Unit30 only; late layers 20-35; facts/query/post-query regions. Unit31 remains untouched lockbox.
    energy = np.square(interaction[0, :, :, 20:, :, :3]).sum(axis=(0, 1, 4))  # layer x head
    pairs = [(layer + 20, head, float(energy[layer, head])) for layer in range(16) for head in range(NHEADS)]
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    top = pairs[:32]
    top_keys = {(layer, head) for layer, head, _ in top}
    rng = np.random.default_rng(2525)
    pool = [(layer, head) for layer in range(20, 36) for head in range(NHEADS) if (layer, head) not in top_keys]
    random_keys = [pool[int(v)] for v in rng.choice(len(pool), size=32, replace=False)]
    by_region = np.square(interaction[0, :, :, 20:, :, :]).sum(axis=(0, 1, 2, 3))
    return {"selection_unit": 30, "late_layers": [20, 35], "top_k": 32,
            "top": [{"layer": l, "head": h, "energy": e} for l, h, e in top],
            "random": [{"layer": l, "head": h} for l, h in sorted(random_keys)],
            "unit30_region_energy": {name: float(by_region[i]) for i, name in enumerate(REGIONS)}}


def score(logits: torch.Tensor, jobs: list[dict]) -> list[float]:
    answer = []
    for i, job in enumerate(jobs):
        values = []
        for offset, token in enumerate(job["continuation"]):
            z = logits[i, job["prompt_length"] - 1 + offset].float()
            values.append(float(z[token] - torch.logsumexp(z, -1)))
        answer.append(float(sum(values)))
    return answer


def causal(model, tokenizer, rows: list[dict], family_ids: list[int], routes: dict) -> list[dict]:
    index = {(r["unit"], r["family_id"], r["language"], r["surface"], r["output_mode"], r["meaning_swap"], r["query_property"]): r for r in rows}
    jobs = []
    for family_id in family_ids:
        for language in ("en", "zh"):
            for query in (0, 1):
                base = index[(31, family_id, language, 0, "candidate", 0, query)]
                donor = index[(31, family_id, language, 0, "candidate", 1, query)]
                for candidate_index, entity in enumerate(base["entities"]):
                    prefix = " " if language == "en" else ""
                    continuation = [int(v) for v in tokenizer.encode(prefix + entity, add_special_tokens=False)]
                    jobs.append({"id": f"f{family_id}-{language}-q{query}", "family_id": family_id,
                                 "family": base["family"], "language": language, "query": query,
                                 "candidate_index": candidate_index, "continuation": continuation,
                                 "prompt_length": len(base["prompt_ids"]), "position": len(base["prompt_ids"]) - 1,
                                 "base_sequence": base["prompt_ids"] + continuation,
                                 "donor_sequence": donor["prompt_ids"] + continuation})
    layers = model_utils.get_layers(model)
    active: dict[int, dict] = {}
    captured: dict[int, torch.Tensor] = {}
    positions: list[int] = []
    handles = []
    for layer_index in range(20, 36):
        def pre_hook(_module, inputs, layer_index=layer_index):
            hidden = inputs[0]
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            pos_index = torch.tensor(positions, device=hidden.device)
            captured[layer_index] = hidden[batch_index, pos_index].detach().clone()
            if layer_index not in active:
                return None
            changed = hidden.clone()
            spec = active[layer_index]
            source = spec["source"].to(device=hidden.device, dtype=hidden.dtype)
            for head in spec["heads"]:
                lo, hi = head * HDIM, (head + 1) * HDIM
                changed[batch_index, pos_index, lo:hi] = source[:, lo:hi]
            return (changed, *inputs[1:])
        handles.append(layers[layer_index].self_attn.o_proj.register_forward_pre_hook(pre_hook))
    top_by_layer: dict[int, list[int]] = {}
    random_by_layer: dict[int, list[int]] = {}
    for item in routes["top"]: top_by_layer.setdefault(item["layer"], []).append(item["head"])
    for item in routes["random"]: random_by_layer.setdefault(item["layer"], []).append(item["head"])
    device = model.get_input_embeddings().weight.device
    output_rows = []
    try:
        for start in range(0, len(jobs), 8):
            batch = jobs[start:start + 8]
            positions[:] = [j["position"] for j in batch]
            base_ids, base_mask = pad([j["base_sequence"] for j in batch], tokenizer.pad_token_id, device)
            donor_ids, donor_mask = pad([j["donor_sequence"] for j in batch], tokenizer.pad_token_id, device)
            if not torch.equal(base_mask, donor_mask): raise RuntimeError("exact-shape mask mismatch")
            active.clear(); captured.clear()
            with torch.inference_mode(): logits = model(input_ids=base_ids, attention_mask=base_mask, use_cache=False).logits
            base_states = {k: v.clone() for k, v in captured.items()}
            for job, value in zip(batch, score(logits, batch)):
                output_rows.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                   {"condition": "no_patch", "value": value})
            active.clear(); captured.clear()
            with torch.inference_mode(): model.model(input_ids=donor_ids, attention_mask=donor_mask, use_cache=False)
            donor_states = {k: v.clone() for k, v in captured.items()}
            conditions = [
                ("self_all_late", {l: {"heads": list(range(NHEADS)), "source": base_states[l]} for l in range(20, 36)}),
                ("donor_top32", {l: {"heads": h, "source": donor_states[l]} for l, h in top_by_layer.items()}),
                ("donor_random32", {l: {"heads": h, "source": donor_states[l]} for l, h in random_by_layer.items()}),
                ("donor_all_late", {l: {"heads": list(range(NHEADS)), "source": donor_states[l]} for l in range(20, 36)}),
                ("shuffled_all_late", {l: {"heads": list(range(NHEADS)), "source": donor_states[l].roll(2, 0)} for l in range(20, 36)}),
            ]
            for condition, specs in conditions:
                active.clear(); active.update(specs); captured.clear()
                with torch.inference_mode(): logits = model(input_ids=base_ids, attention_mask=base_mask, use_cache=False).logits
                for job, value in zip(batch, score(logits, batch)):
                    output_rows.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                       {"condition": condition, "value": value})
    finally:
        for handle in handles: handle.remove()
    return output_rows


def panels(rows: list[dict]) -> dict:
    index = {(r["id"], r["condition"], r["candidate_index"]): r for r in rows}
    ids = sorted({r["id"] for r in rows}); answer = {}
    for condition in sorted({r["condition"] for r in rows}):
        values = []
        for item_id in ids:
            query = index[(item_id, condition, 0)]["query"]; sign = 1 if query == 0 else -1
            base = index[(item_id, "no_patch", 0)]["value"] - index[(item_id, "no_patch", 1)]["value"]
            patched = index[(item_id, condition, 0)]["value"] - index[(item_id, condition, 1)]["value"]
            values.append((-sign * (patched - base), -sign * patched, abs(patched - base)))
        answer[condition] = {"n": len(values), "mean_shift": float(np.mean([v[0] for v in values])),
                             "positive_shift_rate": float(np.mean([v[0] > 0 for v in values])),
                             "donor_flip_rate": float(np.mean([v[1] > 0 for v in values])),
                             "max_absolute_change": float(max(v[2] for v in values))}
    return answer


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 多晚层Attention来源路由图谱与head锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 对Qwen3-4B、双unit、九自然模式族、英中双语、双meaning-swap、双query的144条surface0-candidate提示，保存36层×32 query-head从answer-boundary指向facts-prefix、query-property、post-query候选/指令、boundary-self四区域的完整注意力质量。unit30按layer20–35的三类外部来源Walsh能量冻结32个layer×head路由，unit31在36组exact-shape样本上比较top32、等量随机、全部晚层head、self与batch错配。

$$R_{{lhr}}=\sum_{{j\in r}}\alpha_{{lh}}(t_{{answer}},j),\qquad I_R=\tfrac14(R_{{00}}-R_{{01}}-R_{{10}}+R_{{11}}).$$

**结果汇总。** 来源能量与冻结路由 `{json.dumps(result['routes'], ensure_ascii=False)}`；注意力质量审计 `{json.dumps(result['attention_mass_audit'], ensure_ascii=False)}`；因果 `{json.dumps(result['causal'], ensure_ascii=False)}`；字段 `{json.dumps(result['fields'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2525_c91329_c92704_multilayer_attention_route_lockbox.py`；144×36×32×4注意力质量、Walsh路由、冻结head清单、因果分数、哈希与final位于`{OUT}`。

**分析与理论进展。** 注意力质量回答“答案边界从哪里读取”，o_proj前的head切片移植回答“这些晚层读取结果是否足以搬动输出”。top32优于随机才支持可复用条件路由；全部晚层仍弱于整block则说明Attention路径只是残差编译的一部分，不能把注意力图直接解释成语言算法。

**问题硬伤与结论。** 区域按提示结构粗分，property span与实体内容仍同变；attention weight不是贡献；head输出patch保留base的MLP和早层残差；只测试Qwen3-4B。该Phase不会因为全晚层门失败而丢弃来源图谱，但必须把“可见路由”和“充分机制”分开表述。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    f2520, f2523, f2524 = load(P2520 / "analysis/final.json"), load(P2523 / "analysis/final.json"), load(P2524 / "analysis/final.json")
    family_ids = f2520["behavior"]["qualified_family_ids"]
    material = [r for r in read(P2520 / "material/natural_rows.jsonl") if r["family_id"] in family_ids]
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        attention_path = OUT / "fields/answer_boundary_attention_region_mass.float32.npy"
        attention_path.parent.mkdir(parents=True, exist_ok=True)
        index_rows, attention_meta = collect_attention(model, tokenizer, material, attention_path)
        index_path = OUT / "material/attention_rows.jsonl"; write(index_path, index_rows)
        interaction_path = OUT / "derived/attention_region_walsh.float32.npy"
        interaction, interaction_meta = build_interaction(attention_path, index_rows, family_ids, interaction_path)
        routes = freeze_routes(interaction); save(OUT / "analysis/frozen_routes.json", routes)
        causal_rows = causal(model, tokenizer, material, family_ids, routes)
    finally:
        model_utils.release_model(model); gc.collect()
    causal_path = OUT / "output/head_route_causal_scores.jsonl"; write(causal_path, causal_rows)
    causal_result = panels(causal_rows)
    maximum_region_mass = float(np.max(np.load(attention_path, mmap_mode="r").sum(axis=-1)))
    checks = {"sources_passed": f2520["all_checks_passed"] and f2523["all_checks_passed"] and f2524["all_checks_passed"],
              "attention_shape": attention_meta["shape"] == [144, 36, 32, 4],
              "interaction_shape": interaction_meta["shape"] == [2, 9, 2, 36, 32, 4],
              "attention_mass_bounded": maximum_region_mass <= 1.01,
              "frozen_32": len(routes["top"]) == 32 and len(routes["random"]) == 32,
              "causal_36": causal_result["donor_all_late"]["n"] == 36,
              "self_exact": causal_result["self_all_late"]["max_absolute_change"] == 0,
              "hashes": all(len(v) == 64 for v in (attention_meta["sha256"], interaction_meta["sha256"], sha(index_path), sha(causal_path))),
              "claim_boundary": True}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B BF16 CUDA nonquantized",
             "scope": {"prompts": 144, "families": f2520["behavior"]["qualified_families"], "causal_pairs": 36},
             "routes": routes, "attention_mass_audit": {"maximum_four_region_sum": maximum_region_mass,
                                                          "bf16_tolerance": 0.01, "regions_are_disjoint": True},
             "causal": causal_result,
             "fields": {"attention_mass": attention_meta, "interaction": interaction_meta,
                        "index": {"path": str(index_path), "sha256": sha(index_path)},
                        "causal_scores": {"path": str(causal_path), "sha256": sha(causal_path)}},
             "adjudication": {"top_route_advantage_over_random": causal_result["donor_top32"]["mean_shift"] > causal_result["donor_random32"]["mean_shift"],
                              "all_late_attention_sufficient_for_donor_flip": causal_result["donor_all_late"]["donor_flip_rate"] >= .75,
                              "attention_route_is_complete_compiler": False, "language_encoding_mechanism_closed": False},
             "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", final)
    if final["all_checks_passed"]: append_memo(final)
    print(json.dumps({"phase": PHASE, "regions": routes["unit30_region_energy"], "top": routes["top"][:8],
                      "causal": causal_result, "checks": checks, "all_checks_passed": final["all_checks_passed"]},
                     ensure_ascii=False, indent=2))
    if not final["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
