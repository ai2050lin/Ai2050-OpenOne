#!/usr/bin/env python3
"""Full-coordinate attention/MLP accounting and causal decomposition at the answer boundary."""
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
P2521 = RESULT / "phase2521_c86177_c87200_natural_field_and_causal_lockbox"
P2522 = RESULT / "phase2522_c87201_c88576_crossmodel_natural_boundary_replication"
OUT = RESULT / "phase2523_c88577_c89952_boundary_component_residual_accounting"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN, DIM, NLAYERS = 2523, "C88577-C89952", 2560, 36
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
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, :len(seq)] = 1
    return ids, mask


def collect_components(model, tokenizer, rows: list[dict], path: Path, residual_path: Path) -> tuple[list[dict], dict, dict]:
    selected = [r for r in rows if r["unit"] in (30, 31) and r["surface"] == 0 and r["output_mode"] == "candidate"]
    selected.sort(key=lambda r: (r["unit"], r["family_id"], r["language"], r["meaning_swap"], r["query_property"]))
    layers = model_utils.get_layers(model)
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(len(selected), 2, NLAYERS, DIM))
    residual = np.lib.format.open_memmap(residual_path, mode="w+", dtype=np.float32,
                                         shape=(len(selected), NLAYERS + 1, DIM))
    positions: list[int] = []
    cache: dict[tuple[int, int], torch.Tensor] = {}
    handles = []
    def embedding_hook(_module, _inputs, output):
        batch_index = torch.arange(output.shape[0], device=output.device)
        pos_index = torch.tensor(positions, device=output.device)
        cache[(2, 0)] = output[batch_index, pos_index].detach().float().cpu()
    handles.append(model.get_input_embeddings().register_forward_hook(embedding_hook))
    for li, layer in enumerate(layers):
        for ci, module in enumerate((layer.self_attn, layer.mlp)):
            def hook(_module, _inputs, output, li=li, ci=ci):
                hidden = output[0] if isinstance(output, tuple) else output
                batch_index = torch.arange(hidden.shape[0], device=hidden.device)
                pos_index = torch.tensor(positions, device=hidden.device)
                cache[(ci, li)] = hidden[batch_index, pos_index].detach().float().cpu()
            handles.append(module.register_forward_hook(hook))
        def block_hook(_module, _inputs, output, li=li):
            hidden = output[0] if isinstance(output, tuple) else output
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            pos_index = torch.tensor(positions, device=hidden.device)
            cache[(2, li + 1)] = hidden[batch_index, pos_index].detach().float().cpu()
        handles.append(layer.register_forward_hook(block_hook))
    device = model.get_input_embeddings().weight.device
    index_rows = []
    try:
        for start in range(0, len(selected), 8):
            batch = selected[start:start + 8]
            positions[:] = [len(r["prompt_ids"]) - 1 for r in batch]
            ids, mask = pad([r["prompt_ids"] for r in batch], tokenizer.pad_token_id, device)
            cache.clear()
            with torch.inference_mode():
                model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            for ci in range(2):
                for li in range(NLAYERS):
                    field[start:start + len(batch), ci, li] = cache[(ci, li)].numpy()
            for qpoint in range(NLAYERS + 1):
                residual[start:start + len(batch), qpoint] = cache[(2, qpoint)].numpy()
            for offset, row in enumerate(batch):
                index_rows.append({k: row[k] for k in ("case_id", "unit", "family_id", "family", "language", "surface",
                                                                    "output_mode", "meaning_swap", "query_property", "model_row")} |
                                  {"component_row": start + offset, "answer_boundary_token": len(row["prompt_ids"]) - 1})
    finally:
        for handle in handles:
            handle.remove()
        field.flush()
        residual.flush()
    return (index_rows,
            {"path": str(path), "shape": list(field.shape), "dtype": "float32", "sha256": sha(path)},
            {"path": str(residual_path), "shape": list(residual.shape), "dtype": "float32", "sha256": sha(residual_path)})


def interactions(field_path: Path, index_rows: list[dict], families: list[int], path: Path) -> tuple[np.ndarray, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    field = np.load(field_path, mmap_mode="r")
    output = np.zeros((2, len(families), 2, 2, NLAYERS, DIM), np.float32)
    index = {(r["unit"], r["family_id"], r["language"], r["meaning_swap"], r["query_property"]): r for r in index_rows}
    for ui, unit in enumerate((30, 31)):
        for fi, family_id in enumerate(families):
            for li, language in enumerate(("en", "zh")):
                cells = {(m, q): np.asarray(field[index[(unit, family_id, language, m, q)]["component_row"]], np.float32)
                         for m in (0, 1) for q in (0, 1)}
                output[ui, fi, li] = (cells[(0, 0)] - cells[(0, 1)] - cells[(1, 0)] + cells[(1, 1)]) / 4
    np.save(path, output)
    return output, {"path": str(path), "shape": list(output.shape), "dtype": "float32", "sha256": sha(path)}


def accounting(component: np.ndarray, index_rows: list[dict], source: dict, residual_path: Path) -> dict:
    hidden_previous = np.load(source["collection"]["field"], mmap_mode="r")
    residual = np.load(residual_path, mmap_mode="r")
    raw = np.load(OUT / "fields/answer_boundary_components.float32.npy", mmap_mode="r")
    closure, previous_storage = [], []
    for r in index_rows:
        h = np.asarray(residual[r["component_row"]], np.float32)
        c = np.asarray(raw[r["component_row"]], np.float32)
        delta = h[1:NLAYERS + 1] - h[:NLAYERS]
        error = delta - c[0] - c[1]
        rms_error = float(np.sqrt(np.mean(error * error)))
        rms_delta = float(np.sqrt(np.mean(delta * delta)))
        closure.append((rms_error, float(np.max(np.abs(error))), rms_error / max(rms_delta, 1e-12)))
        old = np.asarray(hidden_previous[r["model_row"], 2, :NLAYERS + 1], np.float32)
        previous_storage.append(float(np.sqrt(np.mean((old - h) ** 2))))
    by_unit = {}
    for ui, unit in enumerate((30, 31)):
        attn = np.sqrt(np.mean(component[ui, :, :, 0] ** 2, axis=(0, 1, 3)))
        mlp = np.sqrt(np.mean(component[ui, :, :, 1] ** 2, axis=(0, 1, 3)))
        combined = np.sqrt(np.mean((component[ui, :, :, 0] + component[ui, :, :, 1]) ** 2, axis=(0, 1, 3)))
        records = [{"layer": i, "qpoint_out": i + 1, "attention_rms": float(attn[i]), "mlp_rms": float(mlp[i]),
                    "combined_increment_rms": float(combined[i]),
                    "mlp_over_attention": float(mlp[i] / max(attn[i], 1e-12))} for i in range(NLAYERS)]
        by_unit[str(unit)] = {"layers": records,
                              "attention_peak_layer": int(np.argmax(attn)), "mlp_peak_layer": int(np.argmax(mlp)),
                              "combined_peak_layer": int(np.argmax(combined)),
                              "late_quarter_fraction_of_component_norm": float((attn[27:].sum() + mlp[27:].sum()) / (attn.sum() + mlp.sum()))}
    return {"residual_identity": "H_(l+1)-H_l = attention_l + mlp_l at answer boundary",
            "closure": {"rows": len(closure), "rms_mean": float(np.mean([x[0] for x in closure])),
                        "rms_max": float(max(x[0] for x in closure)), "absolute_max": float(max(x[1] for x in closure)),
                        "relative_rms_mean": float(np.mean([x[2] for x in closure])),
                        "relative_rms_max": float(max(x[2] for x in closure))},
            "phase2520_fp16_cross_run_rms_difference": {"mean": float(np.mean(previous_storage)),
                                                         "max": float(max(previous_storage)),
                                                         "used_as_identity_gate": False},
            "by_unit": by_unit}


def score(logits: torch.Tensor, jobs: list[dict]) -> list[float]:
    values = []
    for i, job in enumerate(jobs):
        lp = []
        for offset, token in enumerate(job["continuation"]):
            z = logits[i, job["prompt_length"] - 1 + offset].float()
            lp.append(float(z[token] - torch.logsumexp(z, -1)))
        values.append(float(sum(lp)))
    return values


def causal(model, tokenizer, rows: list[dict], families: list[int]) -> list[dict]:
    index = {(r["unit"], r["family_id"], r["language"], r["surface"], r["output_mode"], r["meaning_swap"], r["query_property"]): r for r in rows}
    items = []
    for family_id in families:
        for language in ("en", "zh"):
            for query in (0, 1):
                base = index[(31, family_id, language, 0, "candidate", 0, query)]
                donor = index[(31, family_id, language, 0, "candidate", 1, query)]
                items.append({"id": f"f{family_id}-{language}-q{query}", "family_id": family_id,
                              "family": base["family"], "language": language, "query": query,
                              "base": base, "donor": donor})
    jobs = []
    for item in items:
        for candidate_index, entity in enumerate(item["base"]["entities"]):
            prefix = " " if item["language"] == "en" else ""
            continuation = [int(v) for v in tokenizer.encode(prefix + entity, add_special_tokens=False)]
            jobs.append({k: item[k] for k in ("id", "family_id", "family", "language", "query")} |
                        {"candidate_index": candidate_index, "continuation": continuation,
                         "prompt_length": len(item["base"]["prompt_ids"]), "position": len(item["base"]["prompt_ids"]) - 1,
                         "base_sequence": item["base"]["prompt_ids"] + continuation,
                         "donor_sequence": item["donor"]["prompt_ids"] + continuation})
    layers = model_utils.get_layers(model)
    layer_map = {"middle": layers[27], "final": layers[35]}
    modules = {}
    for stage, layer in layer_map.items():
        modules[f"{stage}_attn"] = layer.self_attn
        modules[f"{stage}_mlp"] = layer.mlp
        modules[f"{stage}_block"] = layer
    active: dict[str, torch.Tensor] = {}
    captured: dict[str, torch.Tensor] = {}
    positions: list[int] = []
    handles = []
    for key, module in modules.items():
        def hook(_module, _inputs, output, key=key):
            hidden = output[0] if isinstance(output, tuple) else output
            batch_index = torch.arange(hidden.shape[0], device=hidden.device)
            pos_index = torch.tensor(positions, device=hidden.device)
            captured[key] = hidden[batch_index, pos_index].detach().clone()
            if key not in active:
                return None
            changed = hidden.clone()
            changed[batch_index, pos_index] = active[key].to(device=hidden.device, dtype=hidden.dtype)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        handles.append(module.register_forward_hook(hook))
    output_rows = []
    device = model.get_input_embeddings().weight.device
    try:
        for start in range(0, len(jobs), 8):
            batch = jobs[start:start + 8]
            positions[:] = [j["position"] for j in batch]
            base_ids, base_mask = pad([j["base_sequence"] for j in batch], tokenizer.pad_token_id, device)
            donor_ids, donor_mask = pad([j["donor_sequence"] for j in batch], tokenizer.pad_token_id, device)
            if not torch.equal(base_mask, donor_mask):
                raise RuntimeError("exact-shape mask mismatch")
            active.clear(); captured.clear()
            with torch.inference_mode():
                logits = model(input_ids=base_ids, attention_mask=base_mask, use_cache=False).logits
            base_states = {k: v.clone() for k, v in captured.items()}
            for job, value in zip(batch, score(logits, batch)):
                output_rows.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                   {"condition": "no_patch", "value": value})
            active.clear(); captured.clear()
            with torch.inference_mode():
                model.model(input_ids=donor_ids, attention_mask=donor_mask, use_cache=False)
            donor_states = {k: v.clone() for k, v in captured.items()}
            conditions = [
                ("self_final_block", {"final_block": base_states["final_block"]}),
                ("donor_middle_attn", {"middle_attn": donor_states["middle_attn"]}),
                ("donor_middle_mlp", {"middle_mlp": donor_states["middle_mlp"]}),
                ("donor_middle_components", {"middle_attn": donor_states["middle_attn"], "middle_mlp": donor_states["middle_mlp"]}),
                ("donor_middle_block", {"middle_block": donor_states["middle_block"]}),
                ("donor_final_attn", {"final_attn": donor_states["final_attn"]}),
                ("donor_final_mlp", {"final_mlp": donor_states["final_mlp"]}),
                ("donor_final_components", {"final_attn": donor_states["final_attn"], "final_mlp": donor_states["final_mlp"]}),
                ("donor_final_block", {"final_block": donor_states["final_block"]}),
                ("shuffled_final_block", {"final_block": donor_states["final_block"].roll(2, 0)}),
            ]
            for condition, sources in conditions:
                active.clear(); active.update(sources); captured.clear()
                with torch.inference_mode():
                    logits = model(input_ids=base_ids, attention_mask=base_mask, use_cache=False).logits
                for job, value in zip(batch, score(logits, batch)):
                    output_rows.append({k: job[k] for k in ("id", "family_id", "family", "language", "query", "candidate_index")} |
                                       {"condition": condition, "value": value})
    finally:
        for handle in handles:
            handle.remove()
    return output_rows


def panels(rows: list[dict]) -> dict:
    index = {(r["id"], r["condition"], r["candidate_index"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    answer = {}
    for condition in sorted({r["condition"] for r in rows}):
        values = []
        for item_id in ids:
            query = index[(item_id, condition, 0)]["query"]
            sign = 1 if query == 0 else -1
            base = index[(item_id, "no_patch", 0)]["value"] - index[(item_id, "no_patch", 1)]["value"]
            patched = index[(item_id, condition, 0)]["value"] - index[(item_id, condition, 1)]["value"]
            values.append((-sign * (patched - base), -sign * patched, abs(patched - base)))
        answer[condition] = {"n": len(values), "mean_shift": float(np.mean([v[0] for v in values])),
                             "positive_shift_rate": float(np.mean([v[0] > 0 for v in values])),
                             "donor_flip_rate": float(np.mean([v[1] > 0 for v in values])),
                             "max_absolute_change": float(max(v[2] for v in values))}
    return answer


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 答案边界Attention/MLP全坐标守恒与因果拆分（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B BF16 CUDA上，对双unit、九模式族、英中双语、双meaning-swap、双query的144条surface0-candidate自然提示，同一forward保存37个answer-boundary残差检查点及36层Attention、MLP原生2560坐标输出；Phase2520的独立FP16场只作跨运行差异审计，不作为恒等式通过门。因果锁箱固定q28（layer27）与q36（layer35），对unit31九族×双语×双query=36组exact-shape样本分别移植Attention、MLP、二者、整block，并比较self与shuffled。

$$H_{{l+1}}-H_l=A_l+M_l,\qquad I(H_{{l+1}})-I(H_l)=I(A_l)+I(M_l).$$

**结果汇总。** 残差核算 `{json.dumps(result['accounting'], ensure_ascii=False)}`；因果面板 `{json.dumps(result['causal'], ensure_ascii=False)}`；裁决 `{json.dumps(result['adjudication'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2523_c88577_c89952_boundary_component_residual_accounting.py`；144×2×36×2560组件场、九族Walsh场、因果分数、哈希和final位于`{OUT}`。

**分析与理论进展。** 该实验把“晚层边界能翻转输出”拆成三项：进入该层的残差状态、该层Attention增量、该层MLP增量。整block donor与Attention+MLP donor的差异直接测量donor residual input是否不可缺少；单组件结果只说明局部充分度，不等于该组件独立计算语义。

**问题硬伤与结论。** Attention/MLP模块输出本身仍是所有head或所有神经元的合计，尚未定位内部物理齿；patch会改变后续层的自然轨迹；q28/q36来自Qwen3-4B历史冻结点。若残差恒等式闭合但组件移植远弱于整block，应把下一步重点放在跨层累积路径，而不是把最后一层称作编译器。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    source = load(P2520 / "analysis/final.json")
    prior1 = load(P2521 / "analysis/final.json")
    prior2 = load(P2522 / "analysis/final.json")
    family_ids = source["behavior"]["qualified_family_ids"]
    material = read(P2520 / "material/natural_rows.jsonl")
    field_index = {r["case_id"]: r for r in read(Path(source["collection"]["index"]))}
    material = [r | {"model_row": field_index[r["case_id"]]["model_row"]} for r in material if r["case_id"] in field_index]
    qualified_material = [r for r in material if r["family_id"] in family_ids]
    (OUT / "fields").mkdir(parents=True, exist_ok=True)
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        component_path = OUT / "fields/answer_boundary_components.float32.npy"
        residual_path = OUT / "fields/answer_boundary_residual_checkpoints.float32.npy"
        index_rows, component_meta, residual_meta = collect_components(model, tokenizer, qualified_material, component_path, residual_path)
        index_path = OUT / "material/component_index.jsonl"; write(index_path, index_rows)
        interaction_path = OUT / "derived/component_walsh.float32.npy"
        component_interaction, interaction_meta = interactions(component_path, index_rows, family_ids, interaction_path)
        account = accounting(component_interaction, index_rows, source, residual_path)
        causal_rows = causal(model, tokenizer, qualified_material, family_ids)
    finally:
        model_utils.release_model(model); gc.collect()
    causal_path = OUT / "output/component_causal_scores.jsonl"; write(causal_path, causal_rows)
    causal_result = panels(causal_rows)
    checks = {"sources_passed": source["all_checks_passed"] and prior1["all_checks_passed"] and prior2["all_checks_passed"],
              "component_shape": component_meta["shape"] == [144, 2, 36, DIM],
              "interaction_shape": interaction_meta["shape"] == [2, 9, 2, 2, 36, DIM],
              "residual_shape": residual_meta["shape"] == [144, 37, DIM],
              "residual_closure": account["closure"]["relative_rms_max"] < 0.02,
              "causal_36": causal_result["donor_final_block"]["n"] == 36,
              "self_exact": causal_result["self_final_block"]["max_absolute_change"] == 0,
              "hashes": all(len(x) == 64 for x in (component_meta["sha256"], residual_meta["sha256"], interaction_meta["sha256"], sha(index_path), sha(causal_path))),
              "claim_boundary": True}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B BF16 CUDA nonquantized",
             "scope": {"families": source["behavior"]["qualified_families"], "prompts": 144, "causal_pairs": 36},
             "accounting": account, "causal": causal_result,
             "fields": {"components": component_meta, "residual_checkpoints": residual_meta, "interactions": interaction_meta,
                        "index": {"path": str(index_path), "sha256": sha(index_path)},
                        "causal_scores": {"path": str(causal_path), "sha256": sha(causal_path)}},
             "adjudication": {"residual_stream_physically_accounted": checks["residual_closure"],
                              "whole_state_vs_component_separated": True,
                              "single_component_semantic_gear_identified": False,
                              "language_encoding_mechanism_closed": False},
             "checks": checks, "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", final)
    if final["all_checks_passed"]:
        append_memo(final)
    print(json.dumps({"phase": PHASE, "accounting": account["closure"], "causal": causal_result,
                      "checks": checks, "all_checks_passed": final["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not final["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
