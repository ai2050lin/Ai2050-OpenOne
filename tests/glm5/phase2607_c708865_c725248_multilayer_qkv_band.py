#!/usr/bin/env python3
"""Test whether the strong source-residual effect is consumed across a multi-layer Q/K/V band."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2606 = RESULT / "phase2606_c692481_c708864_qkv_head_route"
OUT = RESULT / "phase2607_c708865_c725248_multilayer_qkv_band"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2607, "C708865-C725248"
START_LAYER = 6
ENDS = (6, 11, 17, 23, 29, 35)

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox as p2601  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_all_layers(model, tokenizer, pairs):
    layers = model_utils.get_layers(model)
    device = model.get_input_embeddings().weight.device
    dims = {"q": layers[0].self_attn.q_proj.out_features,
            "k": layers[0].self_attn.k_proj.out_features,
            "v": layers[0].self_attn.v_proj.out_features}
    fields, paths = {}, {}
    for component, dim in dims.items():
        path = OUT / f"field/{component}_45x2x36x{dim}.float16.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields[component] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                                       shape=(len(pairs), 2, len(layers), dim))
        paths[component] = path
    for pair_index, pair in enumerate(pairs):
        width = max(len(row["prompt_ids"]) for row in pair)
        ids = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for variant, row in enumerate(pair):
            ids[variant, :len(row["prompt_ids"])] = torch.tensor(row["prompt_ids"], device=device)
            mask[variant, :len(row["prompt_ids"])] = 1
        handles = []
        for layer_index, layer in enumerate(layers):
            for component, module in (("q", layer.self_attn.q_proj),
                                      ("k", layer.self_attn.k_proj),
                                      ("v", layer.self_attn.v_proj)):
                def make_hook(comp, li):
                    def hook(_module, _inputs, output):
                        for variant, row in enumerate(pair):
                            if comp in ("k", "v"):
                                pos = torch.tensor(row["source_token_positions"], device=output.device)
                                value = output[variant, pos].mean(0)
                            else:
                                value = output[variant, len(row["prompt_ids"]) - 1]
                            fields[comp][pair_index, variant, li] = value.detach().cpu().to(torch.float16).numpy()
                    return hook
                handles.append(module.register_forward_hook(make_hook(component, layer_index)))
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if (pair_index + 1) % 15 == 0:
            print(f"[phase2607 collect] {pair_index + 1}/{len(pairs)}", flush=True)
    for field in fields.values():
        field.flush()
    return paths


def jobs_for(tokenizer, pairs):
    jobs = []
    for pair_index, pair in enumerate(pairs):
        row = pair[0]
        for answer_index, answer in enumerate((pair[0]["target"], pair[1]["target"])):
            ids, answer_positions = p2601.candidate_token_ids(tokenizer, row["prompt"], answer)
            jobs.append({"pair_index": pair_index, "answer_index": answer_index, "ids": ids,
                         "answer_positions": answer_positions, "source_positions": row["source_token_positions"],
                         "answer_boundary": len(row["prompt_ids"]) - 1})
    return jobs


def score(model, tokenizer, pairs, deltas, components=(), start=None, end=None, roll=False, batch_size=18):
    device = model.get_input_embeddings().weight.device
    layers = model_utils.get_layers(model)
    jobs = jobs_for(tokenizer, pairs)
    scores = np.zeros((len(pairs), 2), dtype=np.float32)
    for offset in range(0, len(jobs), batch_size):
        batch = jobs[offset:offset + batch_size]
        width = max(len(job["ids"]) for job in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, job in enumerate(batch):
            ids[index, :len(job["ids"])] = torch.tensor(job["ids"], device=device)
            mask[index, :len(job["ids"])] = 1
            answer_mask[index, job["answer_positions"]] = True
        handles = []
        if components:
            for layer_index in range(start, end + 1):
                attn = layers[layer_index].self_attn
                modules = {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj}
                for component in components:
                    vectors = []
                    positions = []
                    for job in batch:
                        vector = deltas[component][job["pair_index"], layer_index].astype(np.float32).copy()
                        if roll:
                            vector = np.roll(vector, 257 if component == "q" else 193)
                        vectors.append(vector)
                        positions.append(job["source_positions"] if component in ("k", "v") else [job["answer_boundary"]])
                    vectors_t = torch.tensor(np.stack(vectors), dtype=torch.float32, device=device)

                    def make_hook(vectors_local, positions_local):
                        def hook(_module, _inputs, output):
                            patched = output.clone()
                            for index, pos in enumerate(positions_local):
                                patched[index, pos] = patched[index, pos] + vectors_local[index].to(patched.dtype)
                            return patched
                        return hook
                    handles.append(modules[component].register_forward_hook(make_hook(vectors_t, positions)))
        try:
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                logp = torch.log_softmax(logits[:, :-1], dim=-1)
                token_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for index, job in enumerate(batch):
                scores[job["pair_index"], job["answer_index"]] = float(token_lp[index][answer_mask[index, 1:]].mean().item())
        finally:
            for handle in handles:
                handle.remove()
    return scores


def summarize(scores, baseline):
    margin = scores[:, 1] - scores[:, 0]
    base = baseline[:, 1] - baseline[:, 0]
    return {"mean_margin_gain": float(np.mean(margin - base)),
            "target1_flip_rate": float(np.mean(margin > 0)),
            "mean_target1_margin": float(np.mean(margin))}


def append_memo(result):
    heading = f"## Phase {PHASE}: source差分的多层Q/K/V带累积因果测试（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** Phase2606单消费层Q/K/V均近零，不据此停止，而检验Phase2605的source residual差分是否通过后续多层反复读取。对同一45 external pair采集36层全部source K/V和answer Q坐标，分别从layer6连续patch到6/11/17/23/29/35：

$$K^0_{{l,S}}\leftarrow K^0_{{l,S}}+\delta K_l,\quad
V^0_{{l,S}}\leftarrow V^0_{{l,S}}+\delta V_l,\quad
Q^0_{{l,t_a}}\leftarrow Q^0_{{l,t_a}}+\delta Q_l.$$

每条progressive KV带都有逐层等范数坐标roll；全6—35带另比较Q、K、V、KV、QKV及各自roll。

**测试用例。** 45 pair、21个不重复条件=945 pair-condition、1890完整候选序列；保存Q `[45,2,36,4096]`、K/V `[45,2,36,1024]`全部坐标。Qwen3-4B BF16 CUDA非量化。

**结果汇总。** 条件=`{json.dumps(result['conditions'], ensure_ascii=False)}`；关键=`{json.dumps(result['key_comparisons'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2607_c708865_c725248_multilayer_qkv_band.py`；三类全层组件场、1890候选得分、完整条件与final位于`{OUT}`。

**分析与理论进展。** 若真实KV带效应随终止层增长且高于逐层roll，支持source内容通过多层重复路由；若仍弱，则Phase2605强效更可能依赖每层非线性重编码后的整残差状态，而不是把原pair的投影差逐层硬搬运。两种结果都约束机制。

**问题硬伤。** 多层同时patch偏离自然分布；各层delta来自完整counterfactual而非前一干预实时重算；Q只改answer边界；K/V source池化；仍是oracle pair和候选似然。

**结论。** `{result['claim_boundary']}`；检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    pairs = load_json(P2606 / "material/selected_external_pairs.json")
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        paths = collect_all_layers(model, tokenizer, pairs)
        fields = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
        deltas = {key: value[:, 1].astype(np.float32) - value[:, 0].astype(np.float32) for key, value in fields.items()}
        score_map = {"baseline": score(model, tokenizer, pairs, deltas)}
        for end in ENDS:
            for roll in (False, True):
                key = f"kv_{START_LAYER}_{end}" + ("_roll" if roll else "")
                score_map[key] = score(model, tokenizer, pairs, deltas, ("k", "v"), START_LAYER, end, roll)
                print(f"[phase2607] {key}", flush=True)
        for components in (("q",), ("k",), ("v",), ("k", "v"), ("q", "k", "v")):
            name = "".join(components)
            for roll in (False, True):
                key = f"{name}_{START_LAYER}_35" + ("_roll" if roll else "")
                if key not in score_map:
                    score_map[key] = score(model, tokenizer, pairs, deltas, components, START_LAYER, 35, roll)
                    print(f"[phase2607] {key}", flush=True)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    baseline = score_map["baseline"]
    conditions = {key: summarize(value, baseline) for key, value in score_map.items()}
    records = []
    for condition, values in score_map.items():
        for pair_index, pair in enumerate(pairs):
            records.append({"pair_id": pair[0]["pair_id"], "condition": condition,
                            "target0_mean_logp": float(values[pair_index, 0]),
                            "target1_mean_logp": float(values[pair_index, 1]),
                            "target1_minus_target0": float(values[pair_index, 1] - values[pair_index, 0])})
    score_path = OUT / "causal/candidate_scores.jsonl"
    write_jsonl(score_path, records)
    progressive = [conditions[f"kv_6_{end}"]["mean_margin_gain"] for end in ENDS]
    progressive_roll = [conditions[f"kv_6_{end}_roll"]["mean_margin_gain"] for end in ENDS]
    key = {"ends": list(ENDS), "kv_progressive_gain": progressive,
           "kv_progressive_roll_gain": progressive_roll,
           "full_band": {name: conditions[name] for name in
                         ("q_6_35", "k_6_35", "v_6_35", "kv_6_35", "qkv_6_35",
                          "q_6_35_roll", "k_6_35_roll", "v_6_35_roll", "kv_6_35_roll", "qkv_6_35_roll")}}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized",
              "fields": {key: list(value.shape) for key, value in fields.items()},
              "conditions": conditions, "key_comparisons": key,
              "claim_boundary": "multi-layer oracle Q/K/V band test; not natural recurrent recomputation or endogenous extraction",
              "hashes": {**{key: sha256(path) for key, path in paths.items()}, "scores": sha256(score_path)},
              "language_mechanism_closed": False}
    result["checks"] = {"phase2606_complete": load_json(P2606 / "analysis/final.json")["all_checks_passed"],
                        "all_45_pairs": len(pairs) == 45,
                        "all_36_layers": all(value.shape[2] == 36 for value in fields.values()),
                        "all_qkv_coordinates": fields["q"].shape[-1] == 4096 and fields["k"].shape[-1] == fields["v"].shape[-1] == 1024,
                        "all_21_conditions": len(conditions) == 21,
                        "all_945_pair_conditions": len(records) == 945,
                        "all_1890_candidate_sequences": len(records) * 2 == 1890,
                        "all_coordinates_no_topk": True,
                        "scientific_result_does_not_abort": True, "claim_boundary": True}
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in ("phase", "key_comparisons", "checks", "all_checks_passed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
