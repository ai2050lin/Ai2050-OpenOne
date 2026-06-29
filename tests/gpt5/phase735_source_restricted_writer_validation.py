#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import MODEL_CONFIGS, get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads, get_v_proj  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase733_prompt_type_skeleton_source_localization import (  # noqa: E402
    MODELS,
    get_tensor,
    select_prompt_pairs,
    site_kind_layer,
    site_module,
)


OUT_ROOT = Path("results/glm5_phase735_source_restricted_writer_validation")
PHASE734_ROOT = Path("results/glm5_phase734_prompt_type_skeleton_writer_decomposition")

SOURCE_GROUPS = [
    "instruction",
    "records_all",
    "target_record_line",
    "records_other",
    "question",
    "object_tokens",
    "relation_tokens",
    "target_value_tokens",
    "answer_prefix",
    "all_pre_answer",
    "self_last",
]

FALLBACK_CANDIDATES = {
    "qwen3": {
        "target_site": "hidden_36",
        "attention": [{"component_id": "L35H0", "layer": 35, "head": 0}],
        "mlp": [{"component_id": "L28:mlp[256:512]", "layer": 28, "start": 256, "end": 512}],
    },
    "glm4": {
        "target_site": "hidden_40",
        "attention": [{"component_id": "L39H21", "layer": 39, "head": 21}],
        "mlp": [{"component_id": "L38:mlp[2870:3280]", "layer": 38, "start": 2870, "end": 3280}],
    },
    "deepseek7b": {
        "target_site": "hidden_28",
        "attention": [{"component_id": "L22H24", "layer": 22, "head": 24}],
        "mlp": [
            {"component_id": "L27:mlp[2872:3231]", "layer": 27, "start": 2872, "end": 3231},
            {"component_id": "L22:mlp[718:1077]", "layer": 22, "start": 718, "end": 1077},
        ],
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec.float()).item())


def dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.float().flatten(), b.float().flatten()).item())


def safe_mean(vals: list[float | int | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def select_evenly(n_items: int, max_items: int | None) -> list[int]:
    if max_items is None or max_items >= n_items:
        return list(range(n_items))
    if max_items <= 1:
        return [0]
    idxs = []
    for i in range(max_items):
        idx = round(i * (n_items - 1) / (max_items - 1))
        if idx not in idxs:
            idxs.append(idx)
    return idxs


def load_model_bf16_eager(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"[load] {model_name}: bf16 device_map=auto attn=eager quantization=off")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    if hasattr(model, "hf_device_map"):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if "cuda" in str(v))
        cpu_count = sum(1 for v in dmap.values() if "cpu" in str(v))
        log(f"[load] {model_name}: loaded eager, gpu_components={gpu_count}, cpu_components={cpu_count}, gpu={gpu_mem:.2f}GB")
    else:
        log(f"[load] {model_name}: loaded eager, device={next(model.parameters()).device}, gpu={gpu_mem:.2f}GB")
    return model, tokenizer, next(model.parameters()).device, "eager"


def split_range(start: int, end: int, n_parts: int) -> list[tuple[int, int]]:
    width = max(1, end - start)
    n_parts = max(1, min(n_parts, width))
    out = []
    for i in range(n_parts):
        a = start + round(i * width / n_parts)
        b = start + round((i + 1) * width / n_parts)
        if b > a:
            out.append((a, b))
    return out


def parse_component_id(component_id: str) -> dict[str, int] | None:
    head = re.match(r"L(\d+)H(\d+)$", component_id)
    if head:
        return {"layer": int(head.group(1)), "head": int(head.group(2))}
    mlp = re.match(r"L(\d+):mlp\[(\d+):(\d+)\]$", component_id)
    if mlp:
        return {"layer": int(mlp.group(1)), "start": int(mlp.group(2)), "end": int(mlp.group(3))}
    return None


def load_phase734_candidates(model_name: str, round_name: str, top_attn: int, top_mlp: int) -> dict[str, Any]:
    fallback = FALLBACK_CANDIDATES[model_name]
    path = PHASE734_ROOT / round_name / f"phase734_{model_name}_summary.json"
    if not path.exists():
        return {"target_site": fallback["target_site"], "attention": fallback["attention"][:top_attn], "mlp": fallback["mlp"][:top_mlp]}
    data = json.loads(path.read_text(encoding="utf-8"))

    def pick(rows: list[dict[str, Any]], top_k: int, kind: str) -> list[dict[str, Any]]:
        chosen: list[dict[str, Any]] = []
        ordered = [r for r in rows if r.get("role_guess") == "writer_candidate"] + [r for r in rows if r.get("role_guess") != "writer_candidate"]
        seen: set[str] = set()
        for row in ordered:
            cid = row.get("component_id")
            if not cid or cid in seen:
                continue
            parsed = parse_component_id(cid)
            if not parsed:
                continue
            rec = {"component_id": cid, **parsed}
            for k in ["mean_explicit_skeleton_loss", "mean_explicit_logprob_delta", "role_guess"]:
                if k in row:
                    rec[f"phase734_{k}"] = row[k]
            chosen.append(rec)
            seen.add(cid)
            if len(chosen) >= top_k:
                break
        if len(chosen) < top_k:
            for row in fallback[kind]:
                if row["component_id"] not in seen:
                    chosen.append(row)
                    seen.add(row["component_id"])
                if len(chosen) >= top_k:
                    break
        return chosen[:top_k]

    return {
        "target_site": data.get("target_site") or fallback["target_site"],
        "attention": pick(data.get("top_attention_writer_candidates", []), top_attn, "attention"),
        "mlp": pick(data.get("top_mlp_writer_candidates", []), top_mlp, "mlp"),
    }


def select_balanced_pairs(max_pairs: int | None) -> list[dict[str, Any]]:
    pairs = select_prompt_pairs(None)
    if not max_pairs or max_pairs >= len(pairs):
        return pairs
    return [pairs[i] for i in select_evenly(len(pairs), max_pairs)]


def input_device(model) -> torch.device:
    return next(model.parameters()).device


def first_token_diag(logits: torch.Tensor, tokenizer, answer: str) -> dict[str, Any]:
    tid = target_token_ids(tokenizer, answer)[0]
    return logit_diag(logits, int(tid))


def char_span_positions(tokenizer, prompt: str, start: int, end: int) -> list[int]:
    if start < 0 or end <= start:
        return []
    before = tokenizer.encode(prompt[:start], add_special_tokens=False)
    upto = tokenizer.encode(prompt[:end], add_special_tokens=False)
    return list(range(len(before), len(upto)))


def find_token_subseq_positions(ids: list[int], pattern: list[int]) -> list[int]:
    if not pattern:
        return []
    out: list[int] = []
    plen = len(pattern)
    for i in range(0, len(ids) - plen + 1):
        if ids[i:i + plen] == pattern:
            out.extend(range(i, i + plen))
    return out


def phrase_positions(tokenizer, ids: list[int], phrases: list[str]) -> list[int]:
    pos: set[int] = set()
    for phrase in phrases:
        variants = [phrase]
        if not phrase.startswith((" ", "\n", ".", "=", ":")):
            variants.extend([f" {phrase}", f".{phrase}", f"={phrase}", f" = {phrase}"])
        for var in variants:
            pat = tokenizer.encode(var, add_special_tokens=False)
            pos.update(find_token_subseq_positions(ids, pat))
    return sorted(pos)


def bare_phrase_positions(tokenizer, ids: list[int], phrases: list[str]) -> list[int]:
    pos: set[int] = set()
    for phrase in phrases:
        for var in [phrase, f" {phrase}"]:
            pat = tokenizer.encode(var, add_special_tokens=False)
            pos.update(find_token_subseq_positions(ids, pat))
    return sorted(pos)


def line_span_positions(tokenizer, prompt: str, predicate: Callable[[str], bool]) -> list[int]:
    positions: set[int] = set()
    cursor = 0
    for line in prompt.splitlines(keepends=True):
        bare = line.rstrip("\n")
        if predicate(bare):
            positions.update(char_span_positions(tokenizer, prompt, cursor, cursor + len(line)))
        cursor += len(line)
    return sorted(positions)


def build_source_groups(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    answer_pos = len(ids) - 1
    obj = case["object"]
    relation = case["relation"]
    answer = case["answer"]
    target_line = f"{obj}.{relation} = {answer}"

    record_all = line_span_positions(tokenizer, prompt, lambda s: bool(re.match(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+ = ", s)))
    target_record = line_span_positions(tokenizer, prompt, lambda s: s.strip() == target_line)
    question = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Question:"))
    answer_prefix = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Answer:"))
    instruction = line_span_positions(
        tokenizer,
        prompt,
        lambda s: (
            s.startswith("Facts:")
            or s.startswith("Temporary world facts:")
            or s.startswith("Use the facts above.")
            or s.startswith("Answer using common everyday knowledge.")
            or s.startswith("Use exactly one short value.")
        ),
    )
    relation_phrases = [relation]
    if relation == "grows_on_tree":
        relation_phrases.extend(["tree", "grow"])
    else:
        relation_phrases.append(relation.replace("_", " "))
    answer_hits = bare_phrase_positions(tokenizer, ids, [answer])
    value_positions = sorted(set(answer_hits) & set(target_record))
    if not value_positions:
        value_positions = answer_hits

    groups = {
        "instruction": instruction,
        "records_all": record_all,
        "target_record_line": target_record,
        "records_other": sorted(set(record_all) - set(target_record)),
        "question": question,
        "object_tokens": phrase_positions(tokenizer, ids, [obj]),
        "relation_tokens": phrase_positions(tokenizer, ids, relation_phrases),
        "target_value_tokens": value_positions,
        "answer_prefix": [p for p in answer_prefix if p < answer_pos],
        "all_pre_answer": list(range(0, max(0, answer_pos))),
        "self_last": [answer_pos],
    }
    return {k: [p for p in sorted(set(v)) if 0 <= p < len(ids)] for k, v in groups.items()}


def forward_site_logits(
    model,
    device,
    ids: list[int],
    target_site: str,
    install_hooks: Callable[[], list[Any]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = install_hooks() if install_hooks else []
    kind, _layer_idx = site_kind_layer(target_site)
    module = site_module(model, target_site)
    if kind == "layer_input":
        def pre_hook(_module, inputs):
            captured["vec"] = inputs[0][0, -1].detach().float().cpu()
        handles.append(module.register_forward_pre_hook(pre_hook))
    else:
        def hook(_module, _inputs, output):
            captured["vec"] = get_tensor(output)[0, -1].detach().float().cpu()
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return captured["vec"], out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def normalize_attention(attn: torch.Tensor | np.ndarray, num_heads: int) -> np.ndarray:
    arr = attn.detach().float().cpu().numpy() if isinstance(attn, torch.Tensor) else np.asarray(attn, dtype=np.float32)
    if arr.ndim != 4:
        raise RuntimeError(f"unexpected attention rank={arr.ndim}")
    if arr.shape[1] == num_heads:
        return arr
    if arr.shape[2] == num_heads:
        return np.transpose(arr, (0, 2, 1, 3))
    raise RuntimeError(f"cannot normalize attention shape={arr.shape}, heads={num_heads}")


def forward_base_with_attention(
    model,
    device,
    ids: list[int],
    target_site: str,
    candidate_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, np.ndarray], dict[int, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    value_store: dict[int, torch.Tensor] = {}
    handles = []
    kind, _layer_idx = site_kind_layer(target_site)
    target_module = site_module(model, target_site)
    if kind == "layer_input":
        def target_pre_hook(_module, inputs):
            captured["vec"] = inputs[0][0, -1].detach().float().cpu()
        handles.append(target_module.register_forward_pre_hook(target_pre_hook))
    else:
        def target_hook(_module, _inputs, output):
            captured["vec"] = get_tensor(output)[0, -1].detach().float().cpu()
        handles.append(target_module.register_forward_hook(target_hook))
    for layer_idx in sorted(set(candidate_layers)):
        attn = get_attention_module(get_layers(model)[layer_idx])
        v_proj = get_v_proj(attn)

        def v_hook(_module, _inputs, output, layer_idx=layer_idx):
            value_store[layer_idx] = get_tensor(output).detach().float().cpu()

        handles.append(v_proj.register_forward_hook(v_hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )
        if out.attentions is None:
            raise RuntimeError("model did not return attentions; Phase735 requires eager attention")
        attn_store: dict[int, np.ndarray] = {}
        for layer_idx in sorted(set(candidate_layers)):
            _o_proj, n_heads, _head_dim = head_meta(model, layer_idx)
            attn_store[layer_idx] = normalize_attention(out.attentions[layer_idx], n_heads)
        return captured["vec"], out.logits[0, -1].detach().float().cpu(), attn_store, value_store
    finally:
        for h in handles:
            h.remove()


def install_source_contribution_erasure(model, layer_idx: int, head_idx: int, contribution: torch.Tensor):
    o_proj, n_heads, head_dim = head_meta(model, layer_idx)
    contrib = contribution.detach().float().cpu()

    def pre_hook(_module, inputs):
        x = inputs[0]
        y = x.clone()
        yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
        yv[0, -1, head_idx, :] = yv[0, -1, head_idx, :] - contrib.to(device=y.device, dtype=y.dtype)
        return (y,) + tuple(inputs[1:])

    return [o_proj.register_forward_pre_hook(pre_hook)]


def install_mlp_group_ablation(model, layer_idx: int, start: int, end: int):
    module = get_layers(model)[layer_idx].mlp

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            y = output[0].clone()
            y[0, -1, start:end] = 0
            return (y,) + output[1:]
        y = output.clone()
        y[0, -1, start:end] = 0
        return y

    return [module.register_forward_hook(hook)]


def object_bucket(obj: str) -> str:
    if obj in {"apple", "banana", "pear"}:
        return "anchor"
    if obj in {"grape", "orange", "lemon"}:
        return "holdout_fruit"
    return "other"


def scan_attention_source_groups(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    attention_specs: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    log_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_layers = sorted({int(s["layer"]) for s in attention_specs})
    layer_meta = {}
    for layer_idx in candidate_layers:
        attn = get_attention_module(get_layers(model)[layer_idx])
        o_proj, n_heads, head_dim = head_meta(model, layer_idx)
        n_kv_heads = get_num_kv_heads(model, attn, n_heads)
        layer_meta[layer_idx] = {"o_proj": o_proj, "n_heads": n_heads, "head_dim": head_dim, "n_kv_heads": n_kv_heads}
    log(f"{model_name}: attention source specs={len(attention_specs)} layers={candidate_layers}")

    for pair_idx, pair in enumerate(pairs, 1):
        common = pair["commonsense"]
        explicit = pair["explicit_profile"]
        c_ids = tokenizer.encode(prompt_for(common), add_special_tokens=False)
        e_prompt = prompt_for(explicit)
        e_ids = tokenizer.encode(e_prompt, add_special_tokens=False)
        answer = common["answer"]
        c_base_vec, _c_base_logits = forward_site_logits(model, device, c_ids, target_site)
        e_base_vec, e_base_logits, attn_store, value_store = forward_base_with_attention(
            model, device, e_ids, target_site, candidate_layers
        )
        skeleton = e_base_vec - c_base_vec
        skeleton_norm = norm(skeleton)
        if skeleton_norm <= 1e-9:
            continue
        d_hat = skeleton / skeleton_norm
        e_base_diag = first_token_diag(e_base_logits, tokenizer, answer)
        source_groups = build_source_groups(tokenizer, e_prompt, explicit, e_ids)
        answer_pos = len(e_ids) - 1
        for spec in attention_specs:
            layer_idx = int(spec["layer"])
            head_idx = int(spec["head"])
            meta = layer_meta[layer_idx]
            attn_np = attn_store[layer_idx]
            values = value_store[layer_idx]
            o_weight = meta["o_proj"].weight.detach().float().cpu()
            h_start = head_idx * meta["head_dim"]
            h_end = h_start + meta["head_dim"]
            for source_group in SOURCE_GROUPS:
                source_pos = source_groups.get(source_group, [])
                if not source_pos:
                    continue
                contribution = compute_source_contribution(
                    attn_np,
                    values,
                    [answer_pos],
                    [source_pos],
                    meta["n_heads"],
                    meta["n_kv_heads"],
                )[0, head_idx]
                direct_vec = torch.mv(o_weight[:, h_start:h_end], contribution)
                direct_projection = dot(direct_vec, d_hat)
                direct_norm = norm(direct_vec)
                attn_mass = float(np.asarray(attn_np[0, head_idx, answer_pos, source_pos], dtype=np.float32).sum())

                def install(layer_idx=layer_idx, head_idx=head_idx, contribution=contribution):
                    return install_source_contribution_erasure(model, layer_idx, head_idx, contribution)

                ab_vec, ab_logits = forward_site_logits(model, device, e_ids, target_site, install)
                ab_diag = first_token_diag(ab_logits, tokenizer, answer)
                shift = ab_vec - e_base_vec
                proj_delta = dot(shift, d_hat)
                rows.append(
                    {
                        "model": model_name,
                        "component_type": "attention_head_source_group",
                        "component_id": spec["component_id"],
                        "layer": layer_idx,
                        "head": head_idx,
                        "source_group": source_group,
                        "source_token_count": len(source_pos),
                        "target_site": target_site,
                        "pair_id": pair["pair_id"],
                        "object": common["object"],
                        "object_bucket": object_bucket(common["object"]),
                        "relation": common["relation"],
                        "answer": answer,
                        "baseline_skeleton_norm": skeleton_norm,
                        "attention_mass": attn_mass,
                        "direct_source_projection": direct_projection,
                        "direct_source_norm": direct_norm,
                        "explicit_projection_delta": proj_delta,
                        "explicit_skeleton_loss": -proj_delta,
                        "explicit_target_delta_norm": norm(shift),
                        "explicit_logprob_delta": ab_diag["target_logprob"] - e_base_diag["target_logprob"],
                        "explicit_rank_delta": ab_diag["target_rank"] - e_base_diag["target_rank"],
                    }
                )
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: attention source validation {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def scan_mlp_fine_groups(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    mlp_specs: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    subgroups: int,
    log_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fine_specs = []
    for spec in mlp_specs:
        for start, end in split_range(int(spec["start"]), int(spec["end"]), subgroups):
            fine_specs.append({**spec, "sub_start": start, "sub_end": end, "sub_component_id": f"L{spec['layer']}:mlp[{start}:{end}]"})
    log(f"{model_name}: mlp fine specs={len(fine_specs)} from coarse={len(mlp_specs)}, subgroups={subgroups}")
    for pair_idx, pair in enumerate(pairs, 1):
        common = pair["commonsense"]
        explicit = pair["explicit_profile"]
        c_ids = tokenizer.encode(prompt_for(common), add_special_tokens=False)
        e_ids = tokenizer.encode(prompt_for(explicit), add_special_tokens=False)
        answer = common["answer"]
        c_base_vec, _c_base_logits = forward_site_logits(model, device, c_ids, target_site)
        e_base_vec, e_base_logits = forward_site_logits(model, device, e_ids, target_site)
        skeleton = e_base_vec - c_base_vec
        skeleton_norm = norm(skeleton)
        if skeleton_norm <= 1e-9:
            continue
        d_hat = skeleton / skeleton_norm
        e_base_diag = first_token_diag(e_base_logits, tokenizer, answer)
        for spec in fine_specs:
            layer_idx = int(spec["layer"])
            start = int(spec["sub_start"])
            end = int(spec["sub_end"])

            def install(layer_idx=layer_idx, start=start, end=end):
                return install_mlp_group_ablation(model, layer_idx, start, end)

            ab_vec, ab_logits = forward_site_logits(model, device, e_ids, target_site, install)
            ab_diag = first_token_diag(ab_logits, tokenizer, answer)
            shift = ab_vec - e_base_vec
            proj_delta = dot(shift, d_hat)
            rows.append(
                {
                    "model": model_name,
                    "component_type": "mlp_fine_output_group",
                    "coarse_component_id": spec["component_id"],
                    "component_id": spec["sub_component_id"],
                    "layer": layer_idx,
                    "start": start,
                    "end": end,
                    "width": end - start,
                    "target_site": target_site,
                    "pair_id": pair["pair_id"],
                    "object": common["object"],
                    "object_bucket": object_bucket(common["object"]),
                    "relation": common["relation"],
                    "answer": answer,
                    "baseline_skeleton_norm": skeleton_norm,
                    "explicit_projection_delta": proj_delta,
                    "explicit_skeleton_loss": -proj_delta,
                    "explicit_target_delta_norm": norm(shift),
                    "explicit_logprob_delta": ab_diag["target_logprob"] - e_base_diag["target_logprob"],
                    "explicit_rank_delta": ab_diag["target_rank"] - e_base_diag["target_rank"],
                }
            )
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: mlp fine decomposition {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def summarize_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["component_id"], row["source_group"])].append(row)
    out = []
    for (cid, source_group), vals in groups.items():
        loss = safe_mean([v["explicit_skeleton_loss"] for v in vals]) or 0.0
        logprob = safe_mean([v["explicit_logprob_delta"] for v in vals]) or 0.0
        direct = safe_mean([v["direct_source_projection"] for v in vals]) or 0.0
        attn_mass = safe_mean([v["attention_mass"] for v in vals]) or 0.0
        holdout_vals = [v for v in vals if v["object_bucket"] == "holdout_fruit"]
        if loss > 0 and logprob < 0:
            role = "source_restricted_writer_path"
        elif loss > 0:
            role = "source_state_contributor"
        elif logprob < 0:
            role = "source_likelihood_support"
        else:
            role = "weak_or_mixed"
        out.append(
            {
                "component_type": "attention_head_source_group",
                "component_id": cid,
                "layer": vals[0]["layer"],
                "head": vals[0]["head"],
                "source_group": source_group,
                "n": len(vals),
                "mean_source_token_count": safe_mean([v["source_token_count"] for v in vals]),
                "mean_attention_mass": attn_mass,
                "mean_direct_source_projection": direct,
                "mean_direct_source_norm": safe_mean([v["direct_source_norm"] for v in vals]),
                "mean_explicit_skeleton_loss": loss,
                "mean_explicit_logprob_delta": logprob,
                "mean_explicit_rank_delta": safe_mean([v["explicit_rank_delta"] for v in vals]),
                "mean_explicit_target_delta_norm": safe_mean([v["explicit_target_delta_norm"] for v in vals]),
                "holdout_n": len(holdout_vals),
                "holdout_mean_explicit_skeleton_loss": safe_mean([v["explicit_skeleton_loss"] for v in holdout_vals]),
                "holdout_mean_explicit_logprob_delta": safe_mean([v["explicit_logprob_delta"] for v in holdout_vals]),
                "role_guess": role,
            }
        )
    return sorted(out, key=lambda r: (r["mean_explicit_skeleton_loss"], -abs(r["mean_explicit_logprob_delta"])), reverse=True)


def summarize_mlp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["component_id"]].append(row)
    out = []
    for cid, vals in groups.items():
        loss = safe_mean([v["explicit_skeleton_loss"] for v in vals]) or 0.0
        logprob = safe_mean([v["explicit_logprob_delta"] for v in vals]) or 0.0
        if loss > 0 and logprob < 0:
            role = "fine_mlp_writer_candidate"
        elif loss > 0:
            role = "fine_mlp_state_contributor"
        elif logprob < 0:
            role = "fine_mlp_likelihood_support"
        else:
            role = "weak_or_mixed"
        holdout_vals = [v for v in vals if v["object_bucket"] == "holdout_fruit"]
        out.append(
            {
                "component_type": "mlp_fine_output_group",
                "component_id": cid,
                "coarse_component_id": vals[0]["coarse_component_id"],
                "layer": vals[0]["layer"],
                "start": vals[0]["start"],
                "end": vals[0]["end"],
                "width": vals[0]["width"],
                "n": len(vals),
                "mean_explicit_skeleton_loss": loss,
                "mean_explicit_logprob_delta": logprob,
                "mean_explicit_rank_delta": safe_mean([v["explicit_rank_delta"] for v in vals]),
                "mean_explicit_target_delta_norm": safe_mean([v["explicit_target_delta_norm"] for v in vals]),
                "holdout_n": len(holdout_vals),
                "holdout_mean_explicit_skeleton_loss": safe_mean([v["explicit_skeleton_loss"] for v in holdout_vals]),
                "holdout_mean_explicit_logprob_delta": safe_mean([v["explicit_logprob_delta"] for v in holdout_vals]),
                "role_guess": role,
            }
        )
    return sorted(out, key=lambda r: (r["mean_explicit_skeleton_loss"], -abs(r["mean_explicit_logprob_delta"])), reverse=True)


def build_summary(
    model_name: str,
    round_name: str,
    attn_impl: str,
    candidate_payload: dict[str, Any],
    source_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_summary = summarize_source_rows(source_rows)
    mlp_summary = summarize_mlp_rows(mlp_rows)
    return {
        "phase": 735,
        "title": "Source-Restricted Writer Validation and MLP Fine Decomposition",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "attention_note": "eager attention is required because source-restricted attribution needs output_attentions",
        "quantization": "off",
        "dtype": "bfloat16",
        "target_site": candidate_payload["target_site"],
        "phase734_round": args.phase734_round,
        "attention_specs": candidate_payload["attention"],
        "mlp_specs": candidate_payload["mlp"],
        "max_pairs": args.max_pairs,
        "mlp_subgroups": args.mlp_subgroups,
        "n_attention_source_rows": len(source_rows),
        "n_mlp_fine_rows": len(mlp_rows),
        "top_attention_source_paths": source_summary[:24],
        "top_mlp_fine_candidates": mlp_summary[:24],
        "attention_source_role_counts": dict((r, sum(1 for x in source_summary if x["role_guess"] == r)) for r in sorted({x["role_guess"] for x in source_summary})),
        "mlp_fine_role_counts": dict((r, sum(1 for x in mlp_summary if x["role_guess"] == r)) for r in sorted({x["role_guess"] for x in mlp_summary})),
        "strict_interpretation": "source-restricted erasure validates source-token contribution paths for candidate heads; MLP fine groups remain output-channel groups, not single-neuron proof",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_balanced_pairs(args.max_pairs)
    candidates = load_phase734_candidates(args.model, args.phase734_round, args.top_attn, args.top_mlp)
    log(
        f"{args.model}/{args.round_name}: pairs={len(pairs)} target_site={candidates['target_site']} "
        f"attention={len(candidates['attention'])} mlp={len(candidates['mlp'])}"
    )
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        source_rows = scan_attention_source_groups(
            model,
            tokenizer,
            device,
            args.model,
            candidates["target_site"],
            candidates["attention"],
            pairs,
            args.log_every,
        )
        mlp_rows = scan_mlp_fine_groups(
            model,
            tokenizer,
            device,
            args.model,
            candidates["target_site"],
            candidates["mlp"],
            pairs,
            args.mlp_subgroups,
            args.log_every,
        )
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args.model, args.round_name, attn_impl, candidates, source_rows, mlp_rows, args)
    write_jsonl(out_dir / f"phase735_{args.model}_attention_source_rows.jsonl", source_rows)
    write_jsonl(out_dir / f"phase735_{args.model}_mlp_fine_rows.jsonl", mlp_rows)
    write_json(out_dir / f"phase735_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "target_site": candidates["target_site"],
                "top_attention_source_paths": summary["top_attention_source_paths"][:5],
                "top_mlp_fine_candidates": summary["top_mlp_fine_candidates"][:5],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def build_atlas_graph(payload: dict[str, Any], round_name: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_node(node: dict[str, Any]) -> None:
        if node["id"] in seen:
            return
        seen.add(node["id"])
        nodes.append(node)

    models = payload.get("models", [])
    for model_index, model in enumerate(models):
        lane_z = (model_index - (len(models) - 1) / 2) * 8
        summary = payload["by_model"][model]
        model_node = f"{model}:model"
        phase_node = f"{model}:phase:735:{round_name}"
        target_node = f"{model}:target:{summary['target_site']}"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-24, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": f"Phase 735 {round_name}", "model": model, "position": [-18, 2, lane_z], "role": "source_restricted_validation"})
        add_node({"id": target_node, "type": "layer", "label": summary["target_site"], "model": model, "role": "downstream_prompt_type_carrier", "evidence_level": "phase733_target_site"})
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "phase": 735})
        edges.append({"source": phase_node, "target": target_node, "relation": "measures_downstream_site", "phase": 735})
        for rec in summary.get("top_attention_source_paths", [])[:12]:
            head_node = f"{model}:writer:{round_name}:{rec['component_id']}"
            source_node = f"{model}:source:{round_name}:{rec['component_id']}:{rec['source_group']}"
            add_node(
                {
                    "id": head_node,
                    "type": "head",
                    "label": rec["component_id"],
                    "model": model,
                    "layer": rec["layer"],
                    "head": rec["head"],
                    "role": "candidate_writer_head",
                    "score": rec["mean_explicit_skeleton_loss"],
                }
            )
            add_node(
                {
                    "id": source_node,
                    "type": "token_group",
                    "label": rec["source_group"],
                    "model": model,
                    "role": rec["role_guess"],
                    "attention_mass": rec["mean_attention_mass"],
                    "score": rec["mean_explicit_skeleton_loss"],
                }
            )
            edges.append({"source": source_node, "target": head_node, "relation": "source_group_contributes_to", "weight": rec["mean_explicit_skeleton_loss"], "phase": 735})
            edges.append({"source": head_node, "target": target_node, "relation": "source_restricted_writer_path_to", "weight": rec["mean_explicit_skeleton_loss"], "phase": 735})
        for rec in summary.get("top_mlp_fine_candidates", [])[:12]:
            node_id = f"{model}:mlp_fine:{round_name}:{rec['component_id']}"
            add_node(
                {
                    "id": node_id,
                    "type": "channel_group",
                    "label": rec["component_id"],
                    "model": model,
                    "layer": rec["layer"],
                    "role": rec["role_guess"],
                    "score": rec["mean_explicit_skeleton_loss"],
                    "logprob_delta": rec["mean_explicit_logprob_delta"],
                }
            )
            edges.append({"source": node_id, "target": target_node, "relation": "fine_mlp_group_perturbs", "weight": rec["mean_explicit_skeleton_loss"], "phase": 735})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 735 Source-Restricted Writer Validation ({round_name})",
        "model_info": {"model": "cross_model", "models": models, "phase": 735, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "source group -> writer -> downstream carrier", "y": "layer index", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 735},
        "source_files": [str(OUT_ROOT / round_name / "phase735_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase735_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 735,
        "title": "Source-Restricted Writer Validation and MLP Fine Decomposition",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "source-restricted attention contribution erasure plus MLP output-channel subgroup ablation",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase735_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase735_atlas_graph.json", graph)
    lines = [
        f"# Phase 735 Source-Restricted Writer Validation ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: source-token-group contribution erasure for candidate attention heads; fine MLP subgroup ablation.",
        "",
        "| model | target site | top source path | source loss | source logprob | attention mass | top MLP fine | MLP loss | MLP logprob |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for model, summary in payload["by_model"].items():
        src = (summary.get("top_attention_source_paths") or [{}])[0]
        mlp = (summary.get("top_mlp_fine_candidates") or [{}])[0]
        src_label = f"{src.get('component_id')}<-{src.get('source_group')}"
        lines.append(
            f"| {model} | {summary.get('target_site')} | {src_label} | "
            f"{(src.get('mean_explicit_skeleton_loss') or 0):.3f} | {(src.get('mean_explicit_logprob_delta') or 0):.3f} | "
            f"{(src.get('mean_attention_mass') or 0):.3f} | {mlp.get('component_id')} | "
            f"{(mlp.get('mean_explicit_skeleton_loss') or 0):.3f} | {(mlp.get('mean_explicit_logprob_delta') or 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Source-restricted erasure shows which source token group contributed through a candidate head to the downstream skeleton direction.",
            "- This is stronger than a head ranking, but it is still not a full neuron-level proof.",
            "- MLP fine decomposition narrows output-channel groups; it does not yet identify individual hidden neurons.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase735_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_balanced_pairs(args.max_pairs)), "models": {}}
    for model in MODELS:
        payload["models"][model] = load_phase734_candidates(model, args.phase734_round, args.top_attn, args.top_mlp)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase734-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--top-attn", type=int, default=2)
    parser.add_argument("--top-mlp", type=int, default=2)
    parser.add_argument("--mlp-subgroups", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only or --dry-run is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
