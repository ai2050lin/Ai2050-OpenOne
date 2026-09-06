#!/usr/bin/env python3
"""Full-coordinate binding field for all corrected relation-necessary lockbox pairs."""
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
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2556 = RESULT / "phase2556_c190721_c198912_form_id_collision_erratum_recompute"
P2557 = RESULT / "phase2557_c198913_c207104_corrected_relation_recipient_lockbox"
OUT = RESULT / "phase2558_c207105_c215296_full_coordinate_recipient_field"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2558, "C207105-C215296"
BASE_REGIONS = ("frame", "facts_entity", "facts_relation", "facts_value", "query_context",
                "query_relation", "query_value", "candidate", "instruction", "answer_boundary")
CELL_REGIONS = tuple(f"cell_value_{index}" for index in range(4))
REGIONS = BASE_REGIONS + CELL_REGIONS
RECIPIENTS = ("query_relation", "query_value", "candidate", "instruction", "answer_boundary")
SOURCES = ("facts_entity", "facts_relation", "facts_value")

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def region_positions(row: dict, name: str) -> list[int]:
    if name.startswith("cell_value_"):
        return list(row["fact_cells"][int(name.rsplit("_", 1)[1])]["value_positions"])
    return list(row["regions"][name])


def compile_pairs() -> list[tuple[dict, dict]]:
    material = [row for row in read(P2556 / "material/phase2554_corrected_token_atomic.jsonl")
                if row["ablation"] == "full_scaffold"]
    behavior = [row for row in read(P2556 / "behavior/phase2554_recomputed.jsonl")
                if row["ablation"] == "full_scaffold"]
    correct = {row["base_case_id"]: row["correct"] for row in behavior}
    index = {(row["family_id"], row["relation_form"], row["value_form"], row["query_relation"],
              row["query_value"], row["binding"]): row for row in material}
    pairs = []
    for family_id in range(32):
        for relation_form in ("natural", "nonce"):
            for value_form in ("natural", "nonce"):
                for query_relation in (0, 1):
                    for query_value in (0, 1):
                        key = (family_id, relation_form, value_form, query_relation, query_value)
                        base, donor = index[key + (0,)], index[key + (1,)]
                        if correct[base["base_case_id"]] and correct[donor["base_case_id"]]:
                            pairs.append((base, donor))
    return pairs


class Capture:
    def __init__(self, model):
        self.layers = model_utils.get_layers(model)
        self.residual_in: dict[int, torch.Tensor] = {}
        self.layer_out: dict[int, torch.Tensor] = {}
        self.norm_input: dict[int, torch.Tensor] = {}
        self.position_embedding: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = []
        for layer_index, layer in enumerate(self.layers):
            def layer_pre(_module, args, layer_index=layer_index):
                self.residual_in[layer_index] = args[0].detach()
            def layer_post(_module, _args, output, layer_index=layer_index):
                self.layer_out[layer_index] = (output[0] if isinstance(output, tuple) else output).detach()
            def attention_pre(_module, args, kwargs, layer_index=layer_index):
                self.norm_input[layer_index] = (args[0] if args else kwargs["hidden_states"]).detach()
                self.position_embedding[layer_index] = tuple(item.detach() for item in kwargs["position_embeddings"])
            self.handles.append(layer.register_forward_pre_hook(layer_pre))
            self.handles.append(layer.register_forward_hook(layer_post))
            self.handles.append(layer.self_attn.register_forward_pre_hook(attention_pre, with_kwargs=True))

    def clear(self) -> None:
        self.residual_in.clear()
        self.layer_out.clear()
        self.norm_input.clear()
        self.position_embedding.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def mean_tokens(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    return tensor[..., positions, :].float().mean(dim=-2)


def capture_one(model, row: dict, controller: Capture, dimensions: dict) -> dict[str, np.ndarray]:
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    controller.clear()
    with torch.inference_mode():
        output = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                             output_attentions=True, return_dict=True)
        embedding = model.model.embed_tokens(ids)[0]
    n_layers, n_heads, n_kv, head_dim = dimensions["layers"], dimensions["heads"], dimensions["kv_heads"], dimensions["head_dim"]
    hidden = np.zeros((n_layers + 1, len(REGIONS), dimensions["hidden"]), dtype=np.float32)
    embedding_regions = np.zeros((len(REGIONS), dimensions["hidden"]), dtype=np.float32)
    q_regions = np.zeros((n_layers, len(RECIPIENTS), n_heads, head_dim), dtype=np.float32)
    k_regions = np.zeros((n_layers, len(REGIONS), n_kv, head_dim), dtype=np.float32)
    v_regions = np.zeros_like(k_regions)
    attention_mass = np.zeros((n_layers, n_heads, len(RECIPIENTS), len(SOURCES)), dtype=np.float32)
    weighted_query_value = np.zeros((n_layers, n_heads, len(SOURCES), head_dim), dtype=np.float32)
    region_map = {name: region_positions(row, name) for name in REGIONS}
    for region_index, name in enumerate(REGIONS):
        embedding_regions[region_index] = embedding[region_map[name]].float().mean(dim=0).cpu().numpy()
        hidden[0, region_index] = controller.residual_in[0][0, region_map[name]].float().mean(dim=0).cpu().numpy()
    for layer_index, layer in enumerate(controller.layers):
        for region_index, name in enumerate(REGIONS):
            hidden[layer_index + 1, region_index] = controller.layer_out[layer_index][0, region_map[name]].float().mean(dim=0).cpu().numpy()
        sa = layer.self_attn
        normalized = controller.norm_input[layer_index]
        shape = (*normalized.shape[:-1], -1, head_dim)
        with torch.inference_mode():
            query = sa.q_norm(sa.q_proj(normalized).view(shape)).transpose(1, 2)
            key = sa.k_norm(sa.k_proj(normalized).view(shape)).transpose(1, 2)
            value = sa.v_proj(normalized).view(shape).transpose(1, 2)
            cos, sin = controller.position_embedding[layer_index]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query, key, value = query[0], key[0], value[0]
        repeated_value = value.repeat_interleave(n_heads // n_kv, dim=0).float()
        attention = output.attentions[layer_index][0].float()
        for region_index, name in enumerate(REGIONS):
            k_regions[layer_index, region_index] = mean_tokens(key, region_map[name]).cpu().numpy()
            v_regions[layer_index, region_index] = mean_tokens(value, region_map[name]).cpu().numpy()
        for recipient_index, recipient in enumerate(RECIPIENTS):
            recipient_positions = region_map[recipient]
            q_regions[layer_index, recipient_index] = mean_tokens(query, recipient_positions).cpu().numpy()
            for source_index, source in enumerate(SOURCES):
                source_positions = region_map[source]
                sub = attention[:, recipient_positions][:, :, source_positions]
                attention_mass[layer_index, :, recipient_index, source_index] = sub.sum(dim=-1).mean(dim=-1).cpu().numpy()
                if recipient == "query_value":
                    weighted = torch.einsum("hrs,hsd->hrd", sub, repeated_value[:, source_positions])
                    weighted_query_value[layer_index, :, source_index] = weighted.mean(dim=1).cpu().numpy()
    del output
    return {"embedding": embedding_regions, "hidden": hidden, "q": q_regions, "k": k_regions,
            "v": v_regions, "attention": attention_mass, "weighted": weighted_query_value}


def summarize_field(paths: dict[str, Path], pairs: list[tuple[dict, dict]], dimensions: dict) -> dict:
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    region_index = {name: index for index, name in enumerate(REGIONS)}
    recipient_index = {name: index for index, name in enumerate(RECIPIENTS)}
    source_index = {name: index for index, name in enumerate(SOURCES)}

    def rms(value: np.ndarray, axes: tuple[int, ...]) -> list:
        return np.sqrt(np.mean(np.asarray(value, dtype=np.float32) ** 2, axis=axes)).tolist()

    layer_region_rms = {
        "hidden": rms(arrays["hidden"], (0, 3)),
        "q": rms(arrays["q"], (0, 3, 4)),
        "k": rms(arrays["k"], (0, 3, 4)),
        "v": rms(arrays["v"], (0, 3, 4)),
        "weighted_query_value": rms(arrays["weighted"], (0, 2, 4)),
    }
    early_v = np.asarray(arrays["v"][:, 0:9, region_index["facts_value"]], dtype=np.float32)
    middle_kv = np.concatenate((np.asarray(arrays["k"][:, 9:18, region_index["facts_value"]], dtype=np.float32),
                                np.asarray(arrays["v"][:, 9:18, region_index["facts_value"]], dtype=np.float32)), axis=2)
    midlate_qv = np.concatenate((np.asarray(arrays["k"][:, 18:27, region_index["query_value"]], dtype=np.float32),
                                 np.asarray(arrays["v"][:, 18:27, region_index["query_value"]], dtype=np.float32)), axis=2)
    late_q = np.asarray(arrays["q"][:, 27:36, recipient_index["answer_boundary"]], dtype=np.float32)

    def coordinate_stats(value: np.ndarray) -> dict:
        mean = value.mean(axis=0)
        positive = (value > 0).mean(axis=0)
        consistency = np.maximum(positive, 1.0 - positive)
        return {"shape": list(mean.shape), "mean_abs": float(np.mean(np.abs(mean))),
                "median_sign_consistency": float(np.median(consistency)),
                "fraction_sign_consistency_ge_075": float(np.mean(consistency >= .75)),
                "fraction_sign_consistency_ge_090": float(np.mean(consistency >= .90))}

    form_indices: dict[tuple[str, str], list[int]] = {}
    for form in (("natural", "natural"), ("natural", "nonce"), ("nonce", "natural"), ("nonce", "nonce")):
        form_indices[form] = [index for index, (base, _) in enumerate(pairs)
                              if base["relation_form"] == form[0] and base["value_form"] == form[1]]

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        av, bv = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
        return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12))

    components = {"early_value_v": early_v, "middle_value_kv": middle_kv,
                  "middlelate_query_value_kv": midlate_qv, "late_answer_q": late_q}
    cross_form = {}
    for component, values in components.items():
        means = {f"r={form[0]},v={form[1]}": values[indexes].mean(axis=0) for form, indexes in form_indices.items()}
        reference = means["r=natural,v=natural"]
        cross_form[component] = {name: cosine(reference, value) for name, value in means.items()}
    return {"layer_region_rms": layer_region_rms,
            "coordinate_statistics": {name: coordinate_stats(value) for name, value in components.items()},
            "cross_form_mean_field_cosine_to_natural_natural": cross_form,
            "attention_binding_difference_mean_abs": float(np.mean(np.abs(np.asarray(arrays["attention"], dtype=np.float32)))),
            "query_value_from_facts_value_weighted_difference_rms_by_layer": rms(
                arrays["weighted"][:, :, :, source_index["facts_value"]], (0, 2, 3)),
            "coordinate_contract": {"hidden_coordinates": dimensions["hidden"], "q_heads": dimensions["heads"],
                                    "kv_heads": dimensions["kv_heads"], "head_coordinates": dimensions["head_dim"],
                                    "regions": list(REGIONS), "recipients": list(RECIPIENTS), "sources": list(SOURCES)}}


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 392对全坐标source—recipient绑定场（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** 对Phase2557全部392个修正eligible binding对分别运行base与donor，共784条Qwen3-4B BF16非量化CUDA前向。不是Top-K采样：保存每对donor−base的全部2560维输入embedding与layer0–36 HiddenState region/cell场、全部36层五个recipient的32×128维post-RoPE Q、14个region/cell的8×128维post-RoPE K和V、五recipient×三source的全32-head注意力质量差，以及query-value从三source实际搬运的32×128维weighted-V差。四个事实value cell分开保存，避免value多重集相同导致region均值抵消。

$$
\Delta h_{{i,l,r,c}}=h^{{(b=1)}}_{{i,l,r,c}}-h^{{(b=0)}}_{{i,l,r,c}},
\quad
\Delta w_{{i,l,h,s,c}}=\frac1{{|R_q|}}\sum_{{a\in R_q}}\sum_{{j\in S_s}}\left(\alpha'v'-\alpha v\right)_{{i,l,h,a,j,c}}.
$$

**结果汇总。** 字段维度与全坐标统计为`{json.dumps(result['summary'], ensure_ascii=False)}`；文件元数据为`{json.dumps(result['fields'], ensure_ascii=False)}`；检查为`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2558_c207105_c215296_full_coordinate_recipient_field.py`；pair索引、全部float16 NPY差分场、派生全坐标统计和final位于`{OUT}`。

**分析与理论进展。** sign-consistency按每个物理layer×head/KV-head×coordinate在全部pair上计算，不按幅值挑Top-K。跨form余弦只比较冻结后的整体平均场，若nonce/nonce与自然场低相似，就与Phase2557中query-value和late-Q的因果分化相互印证；若early-V高相似，则说明早期内容绑定比晚期输出编译更具词面不变性。cell级embedding与HiddenState允许区分“token词面变化”与“同一value在entity/relation角色中重新绑定”。

**问题硬伤与结论。** region/cell内部使用token均值，虽保留全部坐标但不是逐token序列；float16落盘有舍入；post-RoPE Q/K含位置；Attention质量仍不能单独当因果；字段只覆盖行为合格英文表格任务。此Phase提供坐标联盟的发现材料，不以观察相关性命名齿轮；联盟规则必须在下一Phase冻结后做独立干预。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2557 / "analysis/final.json")
    pairs = compile_pairs()
    model = tokenizer = None
    arrays = {}
    controller = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        config = model.config
        dimensions = {"layers": len(model_utils.get_layers(model)), "hidden": int(config.hidden_size),
                      "heads": int(config.num_attention_heads), "kv_heads": int(config.num_key_value_heads),
                      "head_dim": int(config.head_dim)}
        n = len(pairs)
        paths = {"embedding": OUT / "fields/embedding_delta.float16.npy",
                 "hidden": OUT / "fields/hidden_delta.float16.npy",
                 "q": OUT / "fields/recipient_q_delta.float16.npy",
                 "k": OUT / "fields/region_k_delta.float16.npy",
                 "v": OUT / "fields/region_v_delta.float16.npy",
                 "attention": OUT / "fields/recipient_source_attention_delta.float16.npy",
                 "weighted": OUT / "fields/query_value_source_weighted_v_delta.float16.npy"}
        arrays = {"embedding": allocate(paths["embedding"], (n, len(REGIONS), dimensions["hidden"])),
                  "hidden": allocate(paths["hidden"], (n, dimensions["layers"] + 1, len(REGIONS), dimensions["hidden"])),
                  "q": allocate(paths["q"], (n, dimensions["layers"], len(RECIPIENTS), dimensions["heads"], dimensions["head_dim"])),
                  "k": allocate(paths["k"], (n, dimensions["layers"], len(REGIONS), dimensions["kv_heads"], dimensions["head_dim"])),
                  "v": allocate(paths["v"], (n, dimensions["layers"], len(REGIONS), dimensions["kv_heads"], dimensions["head_dim"])),
                  "attention": allocate(paths["attention"], (n, dimensions["layers"], dimensions["heads"], len(RECIPIENTS), len(SOURCES))),
                  "weighted": allocate(paths["weighted"], (n, dimensions["layers"], dimensions["heads"], len(SOURCES), dimensions["head_dim"]))}
        controller = Capture(model)
        index_rows = []
        for pair_index, (base, donor) in enumerate(pairs):
            base_field = capture_one(model, base, controller, dimensions)
            donor_field = capture_one(model, donor, controller, dimensions)
            for name in arrays:
                arrays[name][pair_index] = (donor_field[name] - base_field[name]).astype(np.float16)
            index_rows.append({"pair_index": pair_index, "case_id": base["base_case_id"],
                               "family_id": base["family_id"], "family": base["family"],
                               "relation_form": base["relation_form"], "value_form": base["value_form"],
                               "query_relation": base["query_relation"], "query_value": base["query_value"],
                               "base_target": base["target_index"], "donor_target": donor["target_index"]})
            if (pair_index + 1) % 16 == 0 or pair_index + 1 == len(pairs):
                for array in arrays.values():
                    array.flush()
                print(f"[phase2558] {pair_index + 1}/{len(pairs)} pairs", flush=True)
        index_path = OUT / "fields/pair_index.jsonl"
        p2552.write(index_path, index_rows)
        for array in arrays.values():
            array.flush()
        del arrays
        arrays = {}
        summary = summarize_field(paths, pairs, dimensions)
    finally:
        if controller is not None:
            controller.close()
        arrays.clear()
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
    field_meta = {name: {"path": str(path), "shape": list(np.load(path, mmap_mode="r").shape),
                         "dtype": "float16", "bytes": path.stat().st_size, "sha256": sha(path)}
                  for name, path in paths.items()}
    field_meta["index"] = {"path": str(index_path), "bytes": index_path.stat().st_size, "sha256": sha(index_path)}
    checks = {"phase2557_passed": prior["all_checks_passed"], "pairs_392": len(pairs) == 392,
              "all_hidden_coordinates": field_meta["hidden"]["shape"][-1] == 2560,
              "all_q_coordinates": field_meta["q"]["shape"][-2:] == [32, 128],
              "all_kv_coordinates": field_meta["k"]["shape"][-2:] == [8, 128]
              and field_meta["v"]["shape"][-2:] == [8, 128],
              "all_layers": field_meta["hidden"]["shape"][1] == 37 and field_meta["q"]["shape"][1] == 36,
              "cell_values_separated": len(CELL_REGIONS) == 4, "no_topk_primary_analysis": True,
              "all_hashes": all(len(meta["sha256"]) == 64 for meta in field_meta.values()), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "model": "Qwen3-4B BF16 CUDA nonquantized", "design": {"pairs": len(pairs),
              "forward_passes": 2 * len(pairs), "regions": list(REGIONS), "recipients": list(RECIPIENTS),
              "sources": list(SOURCES)}, "summary": summary, "fields": field_meta,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/full_coordinate_summary.json", summary)
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps({"phase": PHASE, "design": result["design"], "summary": summary,
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
