from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hf_probe_env import get_layers, load_probe_model, release_loaded
from phase289_contract_scan import parse_csv, tokenize
from phase290_contract_break_scan import compute_metrics
from phase301_passive_factor_closure import (
    PassiveBase,
    build_bases,
    mean,
    max_seq_len_for_texts,
    select_bases,
    split_bases,
    state_texts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[phase302] {message}", flush=True)


def parse_layers(value: str, n_layers: int) -> list[int]:
    out = set()
    for item in parse_csv(value):
        out.add(max(0, min(int(item), n_layers - 1)))
    return sorted(out)


def encode_ids(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=add_special_tokens)]


def find_subsequence(haystack: list[int], needle: list[int]) -> list[int] | None:
    if not needle:
        return None
    n = len(needle)
    for idx in range(0, len(haystack) - n + 1):
        if haystack[idx : idx + n] == needle:
            return list(range(idx, idx + n))
    return None


def find_word_span(tokenizer: Any, text: str, word: str) -> list[int] | None:
    ids = encode_ids(tokenizer, text, add_special_tokens=True)
    candidates = [
        encode_ids(tokenizer, f" {word}", add_special_tokens=False),
        encode_ids(tokenizer, word, add_special_tokens=False),
        encode_ids(tokenizer, f" {word.lower()}", add_special_tokens=False),
        encode_ids(tokenizer, word.lower(), add_special_tokens=False),
    ]
    for needle in candidates:
        span = find_subsequence(ids, needle)
        if span is not None:
            return span
    return None


def token_positions(base: PassiveBase, state: str, text: str, tokenizer: Any) -> dict[str, list[int]]:
    a = base.agent
    p = base.patient
    v = base.verb
    raw: dict[str, str] = {"verb": v}
    if state == "active_ab":
        raw.update({"subject": a, "object": p, "semantic_agent": a, "semantic_patient": p, "last": p})
    elif state == "active_ba":
        raw.update({"subject": p, "object": a, "semantic_agent": p, "semantic_patient": a, "last": a})
    elif state == "passive_ab_by":
        raw.update({"subject": p, "by_agent": a, "semantic_agent": a, "semantic_patient": p, "last": a})
    elif state == "passive_ba_by":
        raw.update({"subject": a, "by_agent": p, "semantic_agent": p, "semantic_patient": a, "last": p})
    elif state == "passive_ab_no":
        raw.update({"subject": p, "semantic_patient": p, "last": v})
    elif state == "passive_ba_no":
        raw.update({"subject": a, "semantic_patient": a, "last": v})
    else:
        raise ValueError(f"unknown state={state}")

    out: dict[str, list[int]] = {}
    for key, word in raw.items():
        span = find_word_span(tokenizer, text, word)
        if span is not None:
            out[key] = span
    return out


def capture_token_vectors(
    loaded: Any,
    text: str,
    positions: dict[str, list[int]],
    target_layers: list[int],
    modules: list[str],
    seq_len: int,
) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    layers = get_layers(loaded.model)
    captured: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    hooks = []

    def store(layer_idx: int, module_name: str, value: torch.Tensor) -> None:
        value_f = value.detach().float().squeeze(0).cpu()
        for pos_name, span in positions.items():
            valid = [idx for idx in span if 0 <= idx < value_f.shape[0]]
            if valid:
                captured.setdefault(layer_idx, {}).setdefault(module_name, {})[pos_name] = value_f[valid, :].mean(dim=0).clone()

    def make_pre_hook(layer_idx: int):
        def hook(_module: Any, inputs: Any) -> None:
            if isinstance(inputs, tuple) and inputs:
                store(layer_idx, "resid_in", inputs[0])
        return hook

    def make_output_hook(layer_idx: int, module_name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            store(layer_idx, module_name, value)
        return hook

    for layer_idx in target_layers:
        layer = layers[layer_idx]
        if "resid_in" in modules:
            hooks.append(layer.register_forward_pre_hook(make_pre_hook(layer_idx)))
        if "resid_out" in modules:
            hooks.append(layer.register_forward_hook(make_output_hook(layer_idx, "resid_out")))
        if "mlp_out" in modules:
            hooks.append(layer.mlp.register_forward_hook(make_output_hook(layer_idx, "mlp_out")))

    with torch.no_grad():
        loaded.model(**tokenize(loaded, text, seq_len))

    for hook in hooks:
        hook.remove()
    return captured


def baseline_logits(loaded: Any, text: str, seq_len: int) -> torch.Tensor:
    with torch.no_grad():
        out = loaded.model(**tokenize(loaded, text, seq_len))
    return out.logits[0, -1, :].detach().cpu().float().clone()


def build_token_cache(
    loaded: Any,
    bases: list[PassiveBase],
    target_layers: list[int],
    modules: list[str],
    max_seq_len: int,
    progress_every: int,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    start = time.time()
    for idx, base in enumerate(bases):
        texts = state_texts(base)
        seq_len = max_seq_len_for_texts(loaded, texts, max_seq_len)
        cache[base.name] = {}
        for state, text in texts.items():
            positions = token_positions(base, state, text, loaded.tokenizer)
            cache[base.name][state] = {
                "text": text,
                "positions": positions,
                "vectors": capture_token_vectors(loaded, text, positions, target_layers, modules, seq_len),
            }
        if (idx + 1) % progress_every == 0:
            log(f"captured token bases={idx + 1}/{len(bases)} elapsed={time.time() - start:.1f}s")
    return cache


def get_vec(cache: dict[str, dict[str, Any]], base: PassiveBase, state: str, layer: int, module: str, position: str) -> torch.Tensor | None:
    return (
        cache.get(base.name, {})
        .get(state, {})
        .get("vectors", {})
        .get(layer, {})
        .get(module, {})
        .get(position)
    )


def direction_mean(items: list[torch.Tensor]) -> torch.Tensor | None:
    return torch.stack(items).mean(dim=0) if items else None


def compute_token_directions(
    cache: dict[str, dict[str, Any]],
    train_bases: list[PassiveBase],
    target_layers: list[int],
    modules: list[str],
) -> dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]]:
    directions: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for layer in target_layers:
        for module in modules:
            for pos in ["subject", "object", "by_agent", "verb", "last"]:
                voice_diffs = []
                by_diffs = []
                role_diffs = []
                for base in train_bases:
                    if pos == "subject":
                        pairs = [("active_ab", "passive_ab_by"), ("active_ba", "passive_ba_by")]
                    elif pos == "object":
                        pairs = [("active_ab", "passive_ab_by"), ("active_ba", "passive_ba_by")]
                    elif pos == "by_agent":
                        pairs = [("active_ab", "passive_ab_by"), ("active_ba", "passive_ba_by")]
                    else:
                        pairs = [("active_ab", "passive_ab_by"), ("active_ba", "passive_ba_by")]
                    for src, dst in pairs:
                        src_pos = pos
                        dst_pos = pos
                        if pos == "object" and dst.startswith("passive"):
                            dst_pos = "subject"
                        if pos == "by_agent" and src.startswith("active"):
                            src_pos = "subject"
                        src_vec = get_vec(cache, base, src, layer, module, src_pos)
                        dst_vec = get_vec(cache, base, dst, layer, module, dst_pos)
                        if src_vec is not None and dst_vec is not None:
                            voice_diffs.append(dst_vec - src_vec)

                    if pos in {"subject", "verb", "last"}:
                        for src, dst in [("passive_ab_no", "passive_ab_by"), ("passive_ba_no", "passive_ba_by")]:
                            src_vec = get_vec(cache, base, src, layer, module, pos)
                            dst_vec = get_vec(cache, base, dst, layer, module, pos)
                            if src_vec is not None and dst_vec is not None:
                                by_diffs.append(dst_vec - src_vec)

                    role_pairs = [
                        ("active_ab", "subject", "object"),
                        ("active_ba", "subject", "object"),
                        ("passive_ab_by", "by_agent", "subject"),
                        ("passive_ba_by", "by_agent", "subject"),
                    ]
                    for state, agent_pos, patient_pos in role_pairs:
                        agent_vec = get_vec(cache, base, state, layer, module, agent_pos)
                        patient_vec = get_vec(cache, base, state, layer, module, patient_pos)
                        if agent_vec is not None and patient_vec is not None:
                            role_diffs.append(patient_vec - agent_vec)

                voice = direction_mean(voice_diffs)
                by_phrase = direction_mean(by_diffs)
                agent_to_patient = direction_mean(role_diffs)
                if voice is not None:
                    directions["voice"][layer][module][pos] = voice
                if by_phrase is not None:
                    directions["by_phrase"][layer][module][pos] = by_phrase
                if agent_to_patient is not None:
                    directions["agent_to_patient"][layer][module][pos] = agent_to_patient
                    directions["patient_to_agent"][layer][module][pos] = -agent_to_patient
    return directions


def probe_rows(
    cache: dict[str, dict[str, Any]],
    train_bases: list[PassiveBase],
    test_bases: list[PassiveBase],
    directions: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]],
    target_layers: list[int],
    modules: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def score_items(bases: list[PassiveBase], variable: str, layer: int, module: str, pos: str) -> list[tuple[float, int]]:
        d = directions.get(variable, {}).get(layer, {}).get(module, {}).get(pos)
        if d is None or float(d.norm()) <= 1e-12:
            return []
        out: list[tuple[float, int]] = []
        if variable == "voice":
            labels = [
                ("active_ab", 0),
                ("active_ba", 0),
                ("passive_ab_by", 1),
                ("passive_ba_by", 1),
                ("passive_ab_no", 1),
                ("passive_ba_no", 1),
            ]
            for base in bases:
                for state, label in labels:
                    vec = get_vec(cache, base, state, layer, module, pos)
                    if vec is not None:
                        out.append((float(torch.dot(vec, d)), label))
        elif variable == "by_phrase":
            labels = [("passive_ab_no", 0), ("passive_ba_no", 0), ("passive_ab_by", 1), ("passive_ba_by", 1)]
            for base in bases:
                for state, label in labels:
                    vec = get_vec(cache, base, state, layer, module, pos)
                    if vec is not None:
                        out.append((float(torch.dot(vec, d)), label))
        elif variable == "agent_to_patient":
            role_items = [
                ("active_ab", "subject", 0),
                ("active_ab", "object", 1),
                ("active_ba", "subject", 0),
                ("active_ba", "object", 1),
                ("passive_ab_by", "by_agent", 0),
                ("passive_ab_by", "subject", 1),
                ("passive_ba_by", "by_agent", 0),
                ("passive_ba_by", "subject", 1),
            ]
            for base in bases:
                for state, role_pos, label in role_items:
                    vec = get_vec(cache, base, state, layer, module, role_pos)
                    if vec is not None:
                        out.append((float(torch.dot(vec, d)), label))
        return out

    for variable in ["voice", "by_phrase", "agent_to_patient"]:
        for layer in target_layers:
            for module in modules:
                positions = sorted(directions.get(variable, {}).get(layer, {}).get(module, {}))
                for pos in positions:
                    train = score_items(train_bases, variable, layer, module, pos)
                    test = score_items(test_bases, variable, layer, module, pos)
                    if not train or not test:
                        continue
                    pos_scores = [score for score, label in train if label == 1]
                    neg_scores = [score for score, label in train if label == 0]
                    threshold = 0.5 * (mean(pos_scores) + mean(neg_scores))
                    correct = 0
                    margins = []
                    for score, label in test:
                        pred = 1 if score >= threshold else 0
                        correct += int(pred == label)
                        margins.append((score - threshold) if label else (threshold - score))
                    rows.append({
                        "variable": variable,
                        "layer": layer,
                        "module": module,
                        "token_position": pos,
                        "direction_norm": float(directions[variable][layer][module][pos].norm()),
                        "threshold": threshold,
                        "test_accuracy": correct / max(len(test), 1),
                        "test_total": len(test),
                        "mean_signed_margin": mean(margins),
                    })
    return rows


def patch_token_direction_forward(
    loaded: Any,
    text: str,
    seq_len: int,
    layer_idx: int,
    module: str,
    token_span: list[int],
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    layers = get_layers(loaded.model)
    layer = layers[layer_idx]
    hooks = []

    def add_direction(ref: torch.Tensor) -> torch.Tensor:
        patched = ref.clone()
        d = direction.to(device=ref.device, dtype=ref.dtype)
        for idx in token_span:
            if 0 <= idx < patched.shape[1]:
                patched[:, idx, :] = patched[:, idx, :] + alpha * d
        return patched

    def pre_hook(_module: Any, inputs: Any) -> Any:
        if not (isinstance(inputs, tuple) and inputs):
            return inputs
        return (add_direction(inputs[0]),) + tuple(inputs[1:])

    def output_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        ref = output[0] if isinstance(output, tuple) else output
        patched = add_direction(ref)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    if module == "resid_in":
        hooks.append(layer.register_forward_pre_hook(pre_hook))
    elif module == "resid_out":
        hooks.append(layer.register_forward_hook(output_hook))
    elif module == "mlp_out":
        hooks.append(layer.mlp.register_forward_hook(output_hook))
    else:
        raise ValueError(f"unknown module={module}")

    try:
        with torch.no_grad():
            out = loaded.model(**tokenize(loaded, text, seq_len))
        return out.logits[0, -1, :].detach().cpu().float().clone()
    finally:
        for hook in hooks:
            hook.remove()


def patch_token_multi_direction_forward(
    loaded: Any,
    text: str,
    seq_len: int,
    layer_idx: int,
    module: str,
    patches: list[tuple[list[int], torch.Tensor, float]],
) -> torch.Tensor:
    layers = get_layers(loaded.model)
    layer = layers[layer_idx]
    hooks = []

    def add_directions(ref: torch.Tensor) -> torch.Tensor:
        patched = ref.clone()
        for token_span, direction, alpha in patches:
            d = direction.to(device=ref.device, dtype=ref.dtype)
            for idx in token_span:
                if 0 <= idx < patched.shape[1]:
                    patched[:, idx, :] = patched[:, idx, :] + alpha * d
        return patched

    def pre_hook(_module: Any, inputs: Any) -> Any:
        if not (isinstance(inputs, tuple) and inputs):
            return inputs
        return (add_directions(inputs[0]),) + tuple(inputs[1:])

    def output_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        ref = output[0] if isinstance(output, tuple) else output
        patched = add_directions(ref)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    if module == "resid_in":
        hooks.append(layer.register_forward_pre_hook(pre_hook))
    elif module == "resid_out":
        hooks.append(layer.register_forward_hook(output_hook))
    elif module == "mlp_out":
        hooks.append(layer.mlp.register_forward_hook(output_hook))
    else:
        raise ValueError(f"unknown module={module}")

    try:
        with torch.no_grad():
            out = loaded.model(**tokenize(loaded, text, seq_len))
        return out.logits[0, -1, :].detach().cpu().float().clone()
    finally:
        for hook in hooks:
            hook.remove()


def intervention_specs() -> list[dict[str, Any]]:
    return [
        {
            "variable": "voice",
            "source_state": "active_ab",
            "target_state": "passive_ab_by",
            "patch_map": {"subject": "voice:by_agent", "object": "voice:subject", "verb": "voice:verb"},
        },
        {
            "variable": "voice",
            "source_state": "active_ba",
            "target_state": "passive_ba_by",
            "patch_map": {"subject": "voice:by_agent", "object": "voice:subject", "verb": "voice:verb"},
        },
        {
            "variable": "role_swap",
            "source_state": "active_ab",
            "target_state": "active_ba",
            "patch_map": {"subject": "agent_to_patient:subject", "object": "patient_to_agent:object"},
        },
        {
            "variable": "role_swap",
            "source_state": "active_ba",
            "target_state": "active_ab",
            "patch_map": {"subject": "agent_to_patient:subject", "object": "patient_to_agent:object"},
        },
        {
            "variable": "by_phrase",
            "source_state": "passive_ab_no",
            "target_state": "passive_ab_by",
            "patch_map": {"subject": "by_phrase:subject", "verb": "by_phrase:verb", "last": "by_phrase:last"},
        },
        {
            "variable": "by_phrase",
            "source_state": "passive_ba_no",
            "target_state": "passive_ba_by",
            "patch_map": {"subject": "by_phrase:subject", "verb": "by_phrase:verb", "last": "by_phrase:last"},
        },
    ]


def patch_modes_for(spec: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    patch_map = dict(spec["patch_map"])
    modes = [(f"{pos}_only", {pos: direction_key}) for pos, direction_key in patch_map.items()]
    modes.append(("all_positions", patch_map))
    if spec["variable"] == "role_swap":
        modes.append(("both_roles", patch_map))
    return modes


def summarize(rows: list[dict[str, Any]], probe: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            groups[(row["variable"], row["patch_mode"], int(row["layer"]), row["module"], row["source_position"])].append(row)
    curves = []
    for (variable, patch_mode, layer, module, source_position), items in sorted(groups.items()):
        curves.append({
            "variable": variable,
            "patch_mode": patch_mode,
            "layer": layer,
            "module": module,
            "source_position": source_position,
            "mean_progress": mean([float(x["progress"]) for x in items]),
            "mean_kl_ratio": mean([float(x["kl_ratio"]) for x in items]),
            "mean_logit_delta_ratio": mean([float(x["logit_delta_ratio"]) for x in items]),
            "n": len(items),
        })
    best = {}
    for variable in sorted({row["variable"] for row in curves}):
        items = [row for row in curves if row["variable"] == variable]
        if items:
            best[variable] = max(items, key=lambda row: row["mean_progress"])
    probe_best = {}
    for variable in sorted({row["variable"] for row in probe}):
        items = [row for row in probe if row["variable"] == variable]
        if items:
            probe_best[variable] = max(items, key=lambda row: row["test_accuracy"])
    return {
        "probe_best": probe_best,
        "token_patch_curve": curves,
        "best_by_variable": best,
        "nonfinite_rows": sum(1 for row in rows if float(row.get("finite", 1.0)) < 0.5),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bases = select_bases(args.max_bases, args.seed)
    train_bases, all_test_bases = split_bases(bases, args.train_fraction, args.seed)
    test_start = max(0, int(args.test_start))
    test_end = len(all_test_bases) if args.test_count < 0 else min(len(all_test_bases), test_start + int(args.test_count))
    test_bases = all_test_bases[test_start:test_end]
    shard_label = args.shard_label or f"test{test_start:03d}-{test_end:03d}"
    loaded = None
    try:
        loaded = load_probe_model(args.model)
        layers = get_layers(loaded.model)
        target_layers = parse_layers(args.layers, len(layers))
        modules = parse_csv(args.modules)
        alphas = [float(item) for item in parse_csv(args.alphas)]
        log(f"model={args.model} class={type(loaded.model).__name__} layers={len(layers)}")
        log(f"env dtype={os.environ.get('PROBE_TORCH_DTYPE')} attn={os.environ.get('PROBE_ATTN_IMPLEMENTATION')} auto={os.environ.get('PROBE_DEVICE_MAP_AUTO_MODELS')}")
        log(
            f"bases={len(bases)} train={len(train_bases)} test_total={len(all_test_bases)} "
            f"test_shard={test_start}:{test_end} layers={target_layers} modules={modules} alphas={alphas}"
        )

        cache = build_token_cache(loaded, bases, target_layers, modules, args.max_seq_len, args.progress_every)
        directions = compute_token_directions(cache, train_bases, target_layers, modules)
        probe = probe_rows(cache, train_bases, all_test_bases, directions, target_layers, modules)
        log(f"probe rows={len(probe)} best_acc={max([row['test_accuracy'] for row in probe], default=0):.4f}")

        rows: list[dict[str, Any]] = []
        start = time.time()
        specs = intervention_specs()
        partial_dir = output_dir / "partials" / args.model
        partial_dir.mkdir(parents=True, exist_ok=True)

        def build_data(complete: bool) -> dict[str, Any]:
            return {
                "model": args.model,
                "class": type(loaded.model).__name__,
                "complete": complete,
                "num_bases": len(bases),
                "num_train_bases": len(train_bases),
                "num_test_bases": len(test_bases),
                "num_all_test_bases": len(all_test_bases),
                "test_start": test_start,
                "test_end": test_end,
                "shard_label": shard_label,
                "num_results": len(rows),
                "target_layers": target_layers,
                "modules": modules,
                "alphas": alphas,
                "bases": [base.__dict__ for base in bases],
                "train_bases": [base.name for base in train_bases],
                "all_test_bases": [base.name for base in all_test_bases],
                "test_bases": [base.name for base in test_bases],
                "probe_rows": probe,
                "results": rows,
                "summary": summarize(rows, probe),
            }

        for idx, base in enumerate(test_bases):
            texts = state_texts(base)
            seq_len = max_seq_len_for_texts(loaded, texts, args.max_seq_len)
            logits = {state: baseline_logits(loaded, text, seq_len) for state, text in texts.items()}
            for spec in specs:
                src = spec["source_state"]
                dst = spec["target_state"]
                kl = float(F.kl_div(F.log_softmax(logits[src], dim=-1), F.softmax(logits[dst], dim=-1), reduction="sum"))
                if kl < 1e-8:
                    continue
                src_positions = cache[base.name][src]["positions"]
                for patch_mode, patch_map in patch_modes_for(spec):
                    source_label = "+".join(sorted(patch_map))
                    direction_label = "+".join(patch_map[pos] for pos in sorted(patch_map))
                    for layer in target_layers:
                        for module in modules:
                            patch_items: list[tuple[str, str, list[int], torch.Tensor]] = []
                            for source_position, direction_key in patch_map.items():
                                if source_position not in src_positions:
                                    continue
                                direction_variable, direction_position = direction_key.split(":", 1)
                                direction = directions.get(direction_variable, {}).get(layer, {}).get(module, {}).get(direction_position)
                                if direction is not None:
                                    patch_items.append((direction_variable, direction_position, src_positions[source_position], direction))
                            if not patch_items:
                                continue
                            for alpha in alphas:
                                patched = patch_token_multi_direction_forward(
                                    loaded,
                                    texts[src],
                                    seq_len,
                                    layer,
                                    module,
                                    [(span, direction, alpha) for _var, _pos, span, direction in patch_items],
                                )
                                metrics = compute_metrics(patched, logits[src], logits[dst], kl) or {}
                                rows.append({
                                    "base": base.name,
                                    "agent": base.agent,
                                    "patient": base.patient,
                                    "verb": base.verb,
                                    "variable": spec["variable"],
                                    "source_state": src,
                                    "target_state": dst,
                                    "patch_mode": patch_mode,
                                    "source_position": source_label,
                                    "direction_variable": direction_label,
                                    "direction_position": direction_label,
                                    "num_token_patches": len(patch_items),
                                    "layer": layer,
                                    "module": module,
                                    "alpha": alpha,
                                    **metrics,
                                })
            if (idx + 1) % args.progress_every == 0:
                log(f"intervention bases={idx + 1}/{len(test_bases)} rows={len(rows)} elapsed={time.time() - start:.1f}s")
            partial_file = partial_dir / f"{args.model}_phase302_{shard_label}.partial.json"
            partial_file.write_text(json.dumps(build_data(False), indent=2), encoding="utf-8")
            log(f"partial saved {partial_file}")

        data = build_data(True)
        out_file = output_dir / f"{args.model}_phase302_passive_token_role_closure_{shard_label}.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase302_passive_token_role_closure"))
    parser.add_argument("--max-bases", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--modules", default="resid_in,resid_out,mlp_out")
    parser.add_argument("--alphas", default="0,1.0")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=302)
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=-1)
    parser.add_argument("--shard-label", default="")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        data = run(args)
        log(f"done rows={data['num_results']} nonfinite={data['summary']['nonfinite_rows']}")
    finally:
        if args.hard_exit_after_model:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
