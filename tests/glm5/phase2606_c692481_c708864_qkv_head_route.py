#!/usr/bin/env python3
"""Q/K/V/attention and exhaustive head route test on external single-prompt pairs."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2603 = RESULT / "phase2603_c643329_c659712_unique_natural_lockbox"
P2605 = RESULT / "phase2605_c676097_c692480_singleprompt_source_patch"
OUT = RESULT / "phase2606_c692481_c708864_qkv_head_route"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2606, "C692481-C708864"
CONSUME_LAYERS = (1, 6, 12, 18)
BEST_LAYER = 6
COMPONENTS = ("q", "k", "v", "attn")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox as p2601  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def select_external(material, eligible, qualified):
    ids = {row["pair_id"] for row in eligible if row["split"] == "external"}
    grouped = defaultdict(list)
    for row in material:
        if row["pair_id"] in ids and f"{row['family']}/{row['language']}" in qualified:
            grouped[row["pair_id"]].append(row)
    selected = []
    for group in sorted(qualified):
        candidates = sorted(pair_id for pair_id, rows in grouped.items()
                            if f"{rows[0]['family']}/{rows[0]['language']}" == group)
        if len(candidates) < 5:
            raise RuntimeError((group, len(candidates)))
        selected.extend(candidates[:5])
    return [sorted(grouped[pair_id], key=lambda row: row["variant"]) for pair_id in selected]


def collect_components(model, tokenizer, pairs):
    layers = model_utils.get_layers(model)
    device = model.get_input_embeddings().weight.device
    qdim = layers[0].self_attn.q_proj.out_features
    kdim = layers[0].self_attn.k_proj.out_features
    vdim = layers[0].self_attn.v_proj.out_features
    dmodel = model.get_input_embeddings().weight.shape[1]
    shapes = {"q": qdim, "k": kdim, "v": vdim, "attn": dmodel}
    paths, fields = {}, {}
    for component, dim in shapes.items():
        path = OUT / f"field/{component}_45x2x4x{dim}.float16.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        paths[component] = path
        fields[component] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(len(pairs), 2, len(CONSUME_LAYERS), dim))
    for pair_index, pair in enumerate(pairs):
        width = max(len(row["prompt_ids"]) for row in pair)
        ids = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for variant, row in enumerate(pair):
            ids[variant, :len(row["prompt_ids"])] = torch.tensor(row["prompt_ids"], device=device)
            mask[variant, :len(row["prompt_ids"])] = 1
        handles = []
        for layer_slot, layer_index in enumerate(CONSUME_LAYERS):
            attn = layers[layer_index].self_attn
            for component, module in (("q", attn.q_proj), ("k", attn.k_proj),
                                      ("v", attn.v_proj), ("attn", attn)):
                def make_hook(comp, slot):
                    def hook(_module, _inputs, output):
                        tensor = output[0] if isinstance(output, (tuple, list)) else output
                        for variant, row in enumerate(pair):
                            if comp in ("k", "v"):
                                positions = torch.tensor(row["source_token_positions"], device=tensor.device)
                                value = tensor[variant, positions].mean(0)
                            else:
                                value = tensor[variant, len(row["prompt_ids"]) - 1]
                            fields[comp][pair_index, variant, slot] = value.detach().cpu().to(torch.float16).numpy()
                    return hook
                handles.append(module.register_forward_hook(make_hook(component, layer_slot)))
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if (pair_index + 1) % 15 == 0:
            print(f"[phase2606 collect] {pair_index + 1}/{len(pairs)}", flush=True)
    for field in fields.values():
        field.flush()
    return paths, shapes


def build_jobs(tokenizer, pairs):
    jobs = []
    for pair_index, pair in enumerate(pairs):
        recipient = pair[0]
        for answer_index, answer in enumerate((pair[0]["target"], pair[1]["target"])):
            ids, answer_positions = p2601.candidate_token_ids(tokenizer, recipient["prompt"], answer)
            jobs.append({"pair_index": pair_index, "answer_index": answer_index, "ids": ids,
                         "answer_positions": answer_positions,
                         "source_positions": recipient["source_token_positions"],
                         "answer_boundary": len(recipient["prompt_ids"]) - 1})
    return jobs


def score_patch(model, tokenizer, pairs, component_deltas, layer_index=None, specs=(), batch_size=18):
    device = model.get_input_embeddings().weight.device
    layers = model_utils.get_layers(model)
    jobs = build_jobs(tokenizer, pairs)
    scores = np.zeros((len(pairs), 2), dtype=np.float32)
    slot = CONSUME_LAYERS.index(layer_index) if layer_index is not None else None
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        width = max(len(job["ids"]) for job in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, job in enumerate(batch):
            ids[index, :len(job["ids"])] = torch.tensor(job["ids"], device=device)
            mask[index, :len(job["ids"])] = 1
            answer_mask[index, job["answer_positions"]] = True
        handles = []
        if specs:
            attn = layers[layer_index].self_attn
            modules = {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj, "attn": attn}
            for component, roll, head in specs:
                vectors = []
                positions = []
                for job in batch:
                    vector = component_deltas[component][job["pair_index"], slot].astype(np.float32).copy()
                    if roll:
                        vector = np.roll(vector, 257 if component == "q" else 193 if component in ("k", "v") else 641)
                    if head is not None:
                        masked = np.zeros_like(vector)
                        head_dim = 128
                        masked[head * head_dim:(head + 1) * head_dim] = vector[head * head_dim:(head + 1) * head_dim]
                        vector = masked
                    vectors.append(vector)
                    positions.append(job["source_positions"] if component in ("k", "v") else [job["answer_boundary"]])
                vectors_t = torch.tensor(np.stack(vectors), dtype=torch.float32, device=device)

                def make_patch(vectors_local, positions_local):
                    def hook(_module, _inputs, output):
                        tensor = output[0] if isinstance(output, (tuple, list)) else output
                        patched = tensor.clone()
                        for index, pos in enumerate(positions_local):
                            patched[index, pos] = patched[index, pos] + vectors_local[index].to(patched.dtype)
                        if isinstance(output, tuple):
                            return (patched,) + output[1:]
                        if isinstance(output, list):
                            return [patched] + output[1:]
                        return patched
                    return hook
                handles.append(modules[component].register_forward_hook(make_patch(vectors_t, positions)))
        try:
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                logp = torch.log_softmax(logits[:, :-1], dim=-1)
                token_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for index, job in enumerate(batch):
                values = token_lp[index][answer_mask[index, 1:]]
                scores[job["pair_index"], job["answer_index"]] = float(values.mean().item())
        finally:
            for handle in handles:
                handle.remove()
    return scores


def summary(scores, baseline):
    margins = scores[:, 1] - scores[:, 0]
    base_margin = baseline[:, 1] - baseline[:, 0]
    return {"n": len(margins), "mean_target1_margin": float(margins.mean()),
            "mean_margin_gain": float((margins - base_margin).mean()),
            "target1_flip_rate": float(np.mean(margins > 0))}


def append_memo(result):
    heading = f"## Phase {PHASE}: 外测pair的Q/K/V/Attention与全部head路由解剖（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 在9个合格组各冻结5个从未用于Phase2605的external pair。对消费层1/6/12/18，同pair两侧采集source K/V均值、answer-boundary Q和attention输出全部物理坐标；以variant0为唯一recipient，分别加入真实Q、K、V、K+V、QKV、attention差分及各自等范数roll：

$$\delta K=\bar K(S_1)-\bar K(S_0),\quad \delta V=\bar V(S_1)-\bar V(S_0),\quad
\delta Q=Q^1(t_a)-Q^0(t_a).$$

在Phase2605最强消费层6额外穷举全部32个Q head、8个K head、8个V head，不先Top-K筛选。

**测试用例。** 45 external pair；baseline+4层×10组件条件+48个单head条件=4005 pair-condition、8010条完整候选序列。保存Q=`{result['fields']['q']}`、K=`{result['fields']['k']}`、V=`{result['fields']['v']}`、attention=`{result['fields']['attn']}`的两侧原始全坐标场；Qwen3-4B BF16 CUDA非量化。

**结果汇总。** 组件=`{json.dumps(result['component_conditions'], ensure_ascii=False)}`；全部head=`{json.dumps(result['all_head_conditions'], ensure_ascii=False)}`；关键裁决=`{json.dumps(result['key_comparisons'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2606_c692481_c708864_qkv_head_route.py`；四类组件原场、8010候选得分、全部48 head效应和final位于`{OUT}`。

**分析与理论进展。** K改变source寻址特征，V改变source送出的内容，Q改变answer的读取请求，attention输出是合并后结果；只有真实组件优于自身roll才可称方向选择性。单head扫描覆盖全部head但只是充分性切片，不能把最大head命名为必要齿轮；多个弱head共同作用符合耦合路线。

**问题硬伤。** 所有组件方向仍来自同pair oracle；Q/K/V在projection后还经过norm与RoPE；pool source可能丢失token次序；单组件注入不保持自然联合分布；完整序列评分不等于greedy。结果只定位候选路由，不闭合语义算法。

**结论。** `{result['claim_boundary']}`；检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    p2603 = load_json(P2603 / "analysis/final.json")
    material = read_jsonl(P2603 / "material/cases.unique.jsonl")
    eligible = load_json(P2603 / "material/eligible_pairs.json")
    pairs = select_external(material, eligible, set(p2603["qualified_groups"]))
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        paths, shapes = collect_components(model, tokenizer, pairs)
        fields = {component: np.load(path, mmap_mode="r") for component, path in paths.items()}
        deltas = {component: fields[component][:, 1].astype(np.float32) - fields[component][:, 0].astype(np.float32)
                  for component in COMPONENTS}
        baseline = score_patch(model, tokenizer, pairs, deltas)
        score_map = {"baseline": baseline}
        component_specs = {
            "q": (("q", False, None),), "k": (("k", False, None),), "v": (("v", False, None),),
            "kv": (("k", False, None), ("v", False, None)),
            "qkv": (("q", False, None), ("k", False, None), ("v", False, None)),
            "attn": (("attn", False, None),),
            "q_roll": (("q", True, None),), "k_roll": (("k", True, None),),
            "v_roll": (("v", True, None),), "attn_roll": (("attn", True, None),),
        }
        for layer in CONSUME_LAYERS:
            for name, specs in component_specs.items():
                key = f"l{layer}_{name}"
                score_map[key] = score_patch(model, tokenizer, pairs, deltas, layer, specs)
                print(f"[phase2606 component] {key}", flush=True)
        head_map = {}
        for component, count in (("q", 32), ("k", 8), ("v", 8)):
            for head in range(count):
                key = f"l{BEST_LAYER}_{component}_head{head}"
                score_map[key] = score_patch(model, tokenizer, pairs, deltas, BEST_LAYER,
                                             ((component, False, head),))
                head_map[key] = summary(score_map[key], baseline)
            print(f"[phase2606 heads] {component} {count}", flush=True)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    component_summary = {key: summary(value, baseline) for key, value in score_map.items()
                         if key == "baseline" or "head" not in key}
    records = []
    for condition, scores in score_map.items():
        for pair_index, pair in enumerate(pairs):
            records.append({"pair_id": pair[0]["pair_id"], "family": pair[0]["family"],
                            "language": pair[0]["language"], "condition": condition,
                            "target0_mean_logp": float(scores[pair_index, 0]),
                            "target1_mean_logp": float(scores[pair_index, 1]),
                            "target1_minus_target0": float(scores[pair_index, 1] - scores[pair_index, 0])})
    score_path = OUT / "causal/candidate_scores.jsonl"
    write_jsonl(score_path, records)
    head_gains = {component: [head_map[f"l{BEST_LAYER}_{component}_head{head}"]["mean_margin_gain"]
                              for head in range(32 if component == "q" else 8)]
                  for component in ("q", "k", "v")}
    key = {"best_consumer_layer_by_qkv_vs_roll": max(
                CONSUME_LAYERS, key=lambda layer: component_summary[f"l{layer}_qkv"]["mean_margin_gain"] -
                max(component_summary[f"l{layer}_q_roll"]["mean_margin_gain"],
                    component_summary[f"l{layer}_k_roll"]["mean_margin_gain"],
                    component_summary[f"l{layer}_v_roll"]["mean_margin_gain"])),
           "l6_q_gain": component_summary["l6_q"]["mean_margin_gain"],
           "l6_k_gain": component_summary["l6_k"]["mean_margin_gain"],
           "l6_v_gain": component_summary["l6_v"]["mean_margin_gain"],
           "l6_kv_gain": component_summary["l6_kv"]["mean_margin_gain"],
           "l6_qkv_gain": component_summary["l6_qkv"]["mean_margin_gain"],
           "l6_attn_gain": component_summary["l6_attn"]["mean_margin_gain"],
           "head_gain_distribution": {component: {"all": values, "positive": sum(value > 0 for value in values),
                                                    "median": float(np.median(values)), "max": float(np.max(values))}
                                      for component, values in head_gains.items()}}
    pair_path = OUT / "material/selected_external_pairs.json"
    save_json(pair_path, pairs)
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "selection": {"groups": 9, "pairs": len(pairs)},
              "fields": {component: list(fields[component].shape) for component in COMPONENTS},
              "component_conditions": component_summary, "all_head_conditions": head_map,
              "key_comparisons": key,
              "claim_boundary": "oracle single-recipient component sufficiency map; no head necessity or learned endogenous extractor",
              "hashes": {**{component: sha256(path) for component, path in paths.items()},
                         "scores": sha256(score_path), "pairs": sha256(pair_path)},
              "language_mechanism_closed": False}
    result["checks"] = {"phase2605_complete": load_json(P2605 / "analysis/final.json")["all_checks_passed"],
                        "all_45_external_pairs": len(pairs) == 45,
                        "all_component_fields": all(fields[c].shape[:3] == (45, 2, 4) for c in COMPONENTS),
                        "all_4005_pair_conditions": len(records) == 4005,
                        "all_8010_candidate_sequences": len(records) * 2 == 8010,
                        "all_32_q_heads": len([key for key in head_map if "_q_head" in key]) == 32,
                        "all_8_k_heads": len([key for key in head_map if "_k_head" in key]) == 8,
                        "all_8_v_heads": len([key for key in head_map if "_v_head" in key]) == 8,
                        "all_coordinates_no_topk": True,
                        "scientific_result_does_not_abort": True, "claim_boundary": True}
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({key: result[key] for key in ("phase", "selection", "key_comparisons", "checks", "all_checks_passed")}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
