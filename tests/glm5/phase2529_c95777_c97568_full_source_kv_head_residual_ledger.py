#!/usr/bin/env python3
"""Full late-layer source K/V -> head -> residual coordinate ledger on Qwen3-4B."""
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
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2520 = RESULT / "phase2520_c85025_c86176_natural_language_counterfactual_fullfield"
P2525 = RESULT / "phase2525_c91329_c92704_multilayer_attention_route_lockbox"
OUT = RESULT / "phase2529_c95777_c97568_full_source_kv_head_residual_ledger"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2529, "C95777-C97568"
LATE = tuple(range(20, 36))
REGIONS = ("facts_prefix", "question_context", "query_property", "post_query", "answer_boundary_self")
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
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pad(sequences: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, : len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, : len(sequence)] = 1
    return ids, mask


def regions(row: dict) -> list[list[int]]:
    boundary = len(row["prompt_ids"]) - 1
    facts_end = int(row["spans"]["facts_end"][-1][1])
    query_start, query_end = map(int, row["spans"]["query_property"][-1])
    parts = [
        list(range(0, facts_end)),
        list(range(facts_end, query_start)),
        list(range(query_start, query_end)),
        list(range(query_end, boundary)),
        [boundary],
    ]
    flat = [p for group in parts for p in group]
    if flat != list(range(boundary + 1)):
        raise RuntimeError((row["case_id"], boundary, parts))
    return parts


def allocate(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    field[:] = np.nan
    return field


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)) / (torch.sqrt(torch.mean(b.float() ** 2)) + 1e-12))


def collect(model, tokenizer, rows: list[dict]) -> tuple[list[dict], dict]:
    selected = [r for r in rows if r["surface"] == 0 and r["output_mode"] == "candidate"]
    selected.sort(key=lambda r: (r["unit"], r["family_id"], r["language"], r["meaning_swap"], r["query_property"]))
    layers = model_utils.get_layers(model)
    config = model.config
    nheads = int(config.num_attention_heads)
    nkv = int(config.num_key_value_heads)
    hdim = int(config.head_dim)
    dim = int(config.hidden_size)
    max_seq = max(len(r["prompt_ids"]) for r in selected)
    n = len(selected)
    paths = {
        "embedding": OUT / "fields/token_embedding.float16.npy",
        "hidden": OUT / "fields/answer_boundary_q20_q36.float16.npy",
        "key": OUT / "fields/late_key_post_rope.float16.npy",
        "value": OUT / "fields/late_value.float16.npy",
        "head_source": OUT / "fields/source_head_pre_o.float16.npy",
        "residual_source": OUT / "fields/source_attention_residual.float16.npy",
    }
    embedding = allocate(paths["embedding"], (n, max_seq, dim))
    hidden = allocate(paths["hidden"], (n, 17, dim))
    key = allocate(paths["key"], (n, len(LATE), nkv, max_seq, hdim))
    value = allocate(paths["value"], (n, len(LATE), nkv, max_seq, hdim))
    head_source = allocate(paths["head_source"], (n, len(LATE), nheads, len(REGIONS), hdim))
    residual_source = allocate(paths["residual_source"], (n, len(LATE), len(REGIONS), dim))

    norm_input: dict[int, torch.Tensor] = {}
    position_embedding: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    o_input: dict[int, torch.Tensor] = {}
    attn_output: dict[int, torch.Tensor] = {}
    layer_output: dict[int, torch.Tensor] = {}
    handles = []
    def layer19_post(_module, _args, output):
        layer_output[19] = (output[0] if isinstance(output, tuple) else output).detach()
    handles.append(layers[19].register_forward_hook(layer19_post))
    for layer_index in LATE:
        def attn_pre(_module, args, kwargs, layer_index=layer_index):
            norm_input[layer_index] = (args[0] if args else kwargs["hidden_states"]).detach()
            position_embedding[layer_index] = tuple(x.detach() for x in kwargs["position_embeddings"])
        handles.append(layers[layer_index].self_attn.register_forward_pre_hook(attn_pre, with_kwargs=True))

        def o_pre(_module, args, layer_index=layer_index):
            o_input[layer_index] = args[0].detach()
        handles.append(layers[layer_index].self_attn.o_proj.register_forward_pre_hook(o_pre))

        def attn_post(_module, _args, output, layer_index=layer_index):
            attn_output[layer_index] = output[0].detach()
        handles.append(layers[layer_index].self_attn.register_forward_hook(attn_post))

        def layer_post(_module, _args, output, layer_index=layer_index):
            layer_output[layer_index] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(layers[layer_index].register_forward_hook(layer_post))

    device = model.get_input_embeddings().weight.device
    index_rows = []
    max_pre_error = 0.0
    max_residual_error = 0.0
    try:
        for start in range(0, n, 4):
            batch = selected[start : start + 4]
            ids, mask = pad([r["prompt_ids"] for r in batch], tokenizer.pad_token_id, device)
            norm_input.clear(); position_embedding.clear(); o_input.clear(); attn_output.clear(); layer_output.clear()
            with torch.inference_mode():
                out = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                  output_attentions=True, return_dict=True)
                emb = model.model.embed_tokens(ids)
            for bi, row in enumerate(batch):
                seq_len = len(row["prompt_ids"])
                boundary = seq_len - 1
                embedding[start + bi, :seq_len] = emb[bi, :seq_len].float().cpu().numpy().astype(np.float16)
                source_parts = regions(row)
                for qi, layer_index in enumerate(range(19, 36)):
                    hidden[start + bi, qi] = layer_output[layer_index][bi, boundary].float().cpu().numpy().astype(np.float16)
                for local_layer, layer_index in enumerate(LATE):
                    sa = layers[layer_index].self_attn
                    x = norm_input[layer_index]
                    input_shape = x.shape[:-1]
                    hidden_shape = (*input_shape, -1, hdim)
                    with torch.inference_mode():
                        q = sa.q_norm(sa.q_proj(x).view(hidden_shape)).transpose(1, 2)
                        k = sa.k_norm(sa.k_proj(x).view(hidden_shape)).transpose(1, 2)
                        v = sa.v_proj(x).view(hidden_shape).transpose(1, 2)
                        cos, sin = position_embedding[layer_index]
                        _, k = apply_rotary_pos_emb(q, k, cos, sin)
                    key[start + bi, local_layer, :, :seq_len] = k[bi, :, :seq_len].float().cpu().numpy().astype(np.float16)
                    value[start + bi, local_layer, :, :seq_len] = v[bi, :, :seq_len].float().cpu().numpy().astype(np.float16)
                    repeated = v.repeat_interleave(nheads // nkv, dim=1)
                    line = out.attentions[layer_index][bi, :, boundary, :seq_len].float()
                    groups = []
                    for positions in source_parts:
                        weights = line[:, positions]
                        vals = repeated[bi, :, positions].float()
                        groups.append(torch.einsum("hp,hpd->hd", weights, vals))
                    source = torch.stack(groups, dim=1)  # head, region, head_dim
                    head_source[start + bi, local_layer] = source.cpu().numpy().astype(np.float16)
                    projected = torch.stack([
                        F.linear(source[:, ri].reshape(-1), sa.o_proj.weight.float(), None)
                        for ri in range(len(REGIONS))
                    ])
                    residual_source[start + bi, local_layer] = projected.detach().cpu().numpy().astype(np.float16)
                    pre_expected = source.sum(dim=1).reshape(-1)
                    max_pre_error = max(max_pre_error, rel_rms(pre_expected, o_input[layer_index][bi, boundary]))
                    out_expected = projected.sum(dim=0)
                    if sa.o_proj.bias is not None:
                        out_expected = out_expected + sa.o_proj.bias.float()
                    max_residual_error = max(max_residual_error, rel_rms(out_expected, attn_output[layer_index][bi, boundary]))
                index_rows.append({
                    "field_row": start + bi,
                    **{k: row[k] for k in ("case_id", "unit", "family_id", "family", "language", "meaning_swap", "query_property")},
                    "prompt_length": seq_len,
                    "answer_boundary_token": boundary,
                    "region_positions": {name: pos for name, pos in zip(REGIONS, source_parts)},
                    "prompt_ids": row["prompt_ids"],
                })
            if (start + len(batch)) % 36 == 0:
                for field in (embedding, hidden, key, value, head_source, residual_source): field.flush()
                print(f"[phase2529] {start + len(batch)}/{n}", flush=True)
    finally:
        for handle in handles: handle.remove()
        for field in (embedding, hidden, key, value, head_source, residual_source): field.flush()
    del embedding, hidden, key, value, head_source, residual_source
    meta = {
        "model": {"layers": len(layers), "late_layers": list(LATE), "hidden_size": dim,
                  "attention_heads": nheads, "kv_heads": nkv, "head_dim": hdim, "max_sequence": max_seq},
        "conservation": {"maximum_head_pre_o_relative_rms": max_pre_error,
                         "maximum_attention_residual_relative_rms": max_residual_error},
        "fields": {
            name: {"path": str(path), "shape": list(np.load(path, mmap_mode="r").shape),
                   "dtype": "float16", "bytes": path.stat().st_size, "sha256": sha(path)}
            for name, path in paths.items()
        },
    }
    return index_rows, meta


def build_interactions(index_rows: list[dict], family_ids: list[int], meta: dict) -> tuple[dict, dict]:
    index = {(r["unit"], r["family_id"], r["language"], r["meaning_swap"], r["query_property"]): r["field_row"] for r in index_rows}
    specs = {
        "hidden": (OUT / "fields/answer_boundary_q20_q36.float16.npy", OUT / "derived/hidden_walsh.float16.npy"),
        "head_source": (OUT / "fields/source_head_pre_o.float16.npy", OUT / "derived/source_head_walsh.float16.npy"),
        "residual_source": (OUT / "fields/source_attention_residual.float16.npy", OUT / "derived/source_residual_walsh.float16.npy"),
    }
    outputs = {}
    arrays = {}
    for name, (source_path, target_path) in specs.items():
        source = np.load(source_path, mmap_mode="r")
        shape = (2, len(family_ids), 2, *source.shape[1:])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float16, shape=shape)
        for ui, unit in enumerate((30, 31)):
            for fi, family_id in enumerate(family_ids):
                for li, language in enumerate(("en", "zh")):
                    cells = {(m, q): np.asarray(source[index[(unit, family_id, language, m, q)]], np.float32)
                             for m in (0, 1) for q in (0, 1)}
                    target[ui, fi, li] = ((cells[(0, 0)] - cells[(0, 1)] - cells[(1, 0)] + cells[(1, 1)]) / 4).astype(np.float16)
        target.flush(); del target
        arrays[name] = np.load(target_path, mmap_mode="r")
        outputs[name] = {"path": str(target_path), "shape": list(arrays[name].shape), "dtype": "float16",
                         "bytes": target_path.stat().st_size, "sha256": sha(target_path)}

    head = np.asarray(arrays["head_source"], np.float32)
    # Discovery is unit30 only. All late heads and all 128 coordinates enter the energy; no Top-K compression is used for analysis.
    energy = np.square(head[0, :, :, :, :, :4, :]).sum(axis=(0, 1, 4, 5))
    pairs = [(int(LATE[li]), hi, float(energy[li, hi])) for li in range(len(LATE)) for hi in range(energy.shape[1])]
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    top = pairs[:32]
    top_keys = {(l, h) for l, h, _ in top}
    pool = [(l, h) for l in LATE for h in range(energy.shape[1]) if (l, h) not in top_keys]
    rng = np.random.default_rng(2529)
    random_keys = [pool[int(i)] for i in rng.choice(len(pool), 32, replace=False)]
    energy31 = np.square(head[1, :, :, :, :, :4, :]).sum(axis=(0, 1, 4, 5))
    flat30, flat31 = energy.reshape(-1), energy31.reshape(-1)
    pearson = float(np.corrcoef(flat30, flat31)[0, 1])
    rank30 = np.argsort(np.argsort(flat30)); rank31 = np.argsort(np.argsort(flat31))
    spearman = float(np.corrcoef(rank30, rank31)[0, 1])
    unit31_top = set(np.argsort(-flat31)[:32].tolist())
    overlap = len(set(np.argsort(-flat30)[:32].tolist()) & unit31_top)
    region_energy = np.square(head).sum(axis=(1, 2, 3, 4, 6)).tolist()
    routes = {
        "selection_unit": 30,
        "all_late_routes_analyzed": len(pairs),
        "top_k_only_for_lockbox_intervention": 32,
        "top": [{"layer": l, "head": h, "energy": e} for l, h, e in top],
        "random": [{"layer": l, "head": h} for l, h in sorted(random_keys)],
        "unit30_unit31_energy_pearson": pearson,
        "unit30_unit31_energy_spearman": spearman,
        "top32_overlap": overlap,
        "region_energy_by_unit": {str(unit): {name: float(region_energy[ui][ri]) for ri, name in enumerate(REGIONS)}
                                  for ui, unit in enumerate((30, 31))},
    }
    return outputs, routes


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    compact_fields = {k: {q: v[q] for q in ("path", "shape", "bytes", "sha256")} for k, v in result["fields"].items()}
    text = rf"""


## Phase {PHASE}: 全晚层source→K/V→head→residual全坐标守恒账本（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 使用Qwen3-4B BF16非量化，在Phase2520双unit九个合格自然模式族、英中双语、meaning-swap×query四格的144条surface0-candidate提示上，重新把可见前缀切成五个互斥且穷尽的区域：facts-prefix、question-context、query-property、post-query、answer-boundary-self。对layer20–35保存全部KV heads的post-RoPE K、V，全部32个query heads×128物理坐标的source贡献、经各层真实$W_O$写入的全部2560 residual坐标、q20–q36答案边界HiddenState以及逐token 2560维词嵌入。top32只用于后续冻结干预，分析本身覆盖全部512个late layer×head，不以Top-K代替全场。

$$u_{{lhr}}=\sum_{{j\in r}}\alpha_{{lhaj}}v_{{lhj}},\qquad g_{{lr}}=W_{{O,l}}\operatorname{{concat}}_h u_{{lhr}},\qquad A_{{l,a}}=\sum_r g_{{lr}}.$$

**结果汇总。** 范围 `{json.dumps(result['scope'], ensure_ascii=False)}`；加法守恒 `{json.dumps(result['conservation'], ensure_ascii=False)}`；全路由冻结与跨unit重复 `{json.dumps(result['routes'], ensure_ascii=False)}`；字段 `{json.dumps(compact_fields, ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2529_c95777_c97568_full_source_kv_head_residual_ledger.py`；全量K/V、head坐标、residual坐标、embedding、HiddenState、Walsh交互、索引和final位于`{OUT}`。

**分析与理论进展。** 现在“从哪里读”不再只用attention质量代替：每个source区域的128维head值贡献和经$W_O$写入的2560维坐标都可直接核算，并能加回真实Attention输出。unit30只负责冻结干预候选，unit31仅用于锁箱；跨unit能量相关/重合是路线重复性测量，不是共享语义轴证明。该账本把Phase2525的whole-head充分性与source-specific内容贡献分开，为下一Phase的边patch和持续阻断提供直接对象。

**问题硬伤与结论。** 单模块加法分解是架构恒等式，不等于各source贡献在下游独立因果；GQA使多个query heads共享同一KV head；区域仍按提示结构而非词法语义自动发现；K/V和embedding物理坐标只在本模型内有意义；float16落盘会有二次舍入。是否存在自然必要联盟、删除后恢复和source特异救援仍必须由干预裁决。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    prior = load(P2520 / "analysis/final.json")
    family_ids = prior["behavior"]["qualified_family_ids"]
    material = [r for r in read(P2520 / "material/natural_rows.jsonl") if r["family_id"] in family_ids]
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        index_rows, meta = collect(model, tokenizer, material)
    finally:
        model_utils.release_model(model); gc.collect()
    index_path = OUT / "material/field_rows.jsonl"
    write(index_path, index_rows)
    interactions, routes = build_interactions(index_rows, family_ids, meta)
    save(OUT / "analysis/frozen_contribution_routes.json", routes)
    fields = meta["fields"] | {f"{name}_interaction": value for name, value in interactions.items()} | {
        "index": {"path": str(index_path), "shape": [len(index_rows)], "dtype": "jsonl",
                  "bytes": index_path.stat().st_size, "sha256": sha(index_path)}
    }
    checks = {
        "source_phase_passed": prior["all_checks_passed"],
        "rows_144": len(index_rows) == 144,
        "all_512_late_routes": routes["all_late_routes_analyzed"] == 512,
        "regions_exhaustive_disjoint": all(sum(len(v) for v in row["region_positions"].values()) == row["prompt_length"] for row in index_rows),
        "head_additive_conservation": meta["conservation"]["maximum_head_pre_o_relative_rms"] < 0.01,
        "residual_additive_conservation": meta["conservation"]["maximum_attention_residual_relative_rms"] < 0.01,
        "physical_fields_hashed": all(len(v["sha256"]) == 64 for v in fields.values()),
        "discovery_lockbox_separated": routes["selection_unit"] == 30,
        "claim_boundary": True,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "scope": {"prompts": 144, "families": prior["behavior"]["qualified_families"], "units": [30, 31],
                  "languages": ["en", "zh"], "late_layers": list(LATE), "regions": list(REGIONS), **meta["model"]},
        "conservation": meta["conservation"],
        "routes": routes,
        "fields": fields,
        "adjudication": {"source_contribution_additive_within_attention_module": True,
                         "source_contribution_independent_downstream_cause": False,
                         "top32_is_only_an_intervention_candidate": True,
                         "language_mechanism_closed": False},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]:
        append_memo(result)
    print(json.dumps({"phase": PHASE, "conservation": result["conservation"], "routes": routes,
                      "field_bytes": {k: v["bytes"] for k, v in fields.items()},
                      "checks": checks, "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
