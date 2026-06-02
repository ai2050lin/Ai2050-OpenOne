from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from hf_probe_env import get_layers, load_probe_model, release_loaded
from phase289_contract_scan import parse_csv, tokenize
from phase301_passive_factor_closure import PassiveBase, mean, select_bases, split_bases, state_texts
from phase302_passive_token_role_closure import find_word_span, token_positions


REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[phase303] {message}", flush=True)


def parse_layers(value: str, n_layers: int) -> list[int]:
    out = set()
    for item in parse_csv(value):
        out.add(max(0, min(int(item), n_layers - 1)))
    return sorted(out)


def first_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"empty tokenization for {text!r}")
    return int(ids[0])


def state_roles(base: PassiveBase, state: str) -> dict[str, str]:
    a = base.agent
    p = base.patient
    if state == "active_ab":
        return {"agent": a, "patient": p}
    if state == "active_ba":
        return {"agent": p, "patient": a}
    if state == "passive_ab_by":
        return {"agent": a, "patient": p}
    if state == "passive_ba_by":
        return {"agent": p, "patient": a}
    raise ValueError(f"unsupported state={state}")


def query_prompt(sentence: str, query_type: str) -> str:
    if query_type == "agent":
        return f"{sentence}. the one who did the action was the"
    if query_type == "patient":
        return f"{sentence}. the one that received the action was the"
    raise ValueError(f"unknown query_type={query_type}")


def score_margin_from_logits(logits: torch.Tensor, tokenizer: Any, correct: str, wrong: str) -> dict[str, Any]:
    correct_id = first_token_id(tokenizer, f" {correct}")
    wrong_id = first_token_id(tokenizer, f" {wrong}")
    correct_logit = float(logits[correct_id])
    wrong_logit = float(logits[wrong_id])
    return {
        "correct": correct,
        "wrong": wrong,
        "correct_token_id": correct_id,
        "wrong_token_id": wrong_id,
        "correct_token_piece": tokenizer.decode([correct_id]),
        "wrong_token_piece": tokenizer.decode([wrong_id]),
        "correct_logit": correct_logit,
        "wrong_logit": wrong_logit,
        "margin": correct_logit - wrong_logit,
        "correct_choice": correct_logit > wrong_logit,
    }


def baseline_logits(loaded: Any, prompt: str, seq_len: int) -> torch.Tensor:
    with torch.no_grad():
        out = loaded.model(**tokenize(loaded, prompt, seq_len))
    return out.logits[0, -1, :].detach().cpu().float().clone()


def capture_prompt_tensors(
    loaded: Any,
    prompt: str,
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
        loaded.model(**tokenize(loaded, prompt, seq_len))

    for hook in hooks:
        hook.remove()
    return captured


def prompt_positions(base: PassiveBase, state: str, sentence: str, prompt: str, tokenizer: Any) -> dict[str, list[int]]:
    positions = token_positions(base, state, sentence, tokenizer)
    # token_positions uses the sentence text; spans are the same prefix positions inside prompt.
    for label, word in {"agent": state_roles(base, state)["agent"], "patient": state_roles(base, state)["patient"]}.items():
        span = find_word_span(tokenizer, prompt, word)
        if span is not None:
            positions[label] = span
    return positions


def build_prompt_cache(
    loaded: Any,
    bases: list[PassiveBase],
    target_layers: list[int],
    modules: list[str],
    max_seq_len: int,
    progress_every: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    states = ["active_ab", "active_ba", "passive_ab_by", "passive_ba_by"]
    query_types = ["agent", "patient"]
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    start = time.time()
    for idx, base in enumerate(bases):
        texts = state_texts(base)
        cache[base.name] = {}
        for state in states:
            sentence = texts[state]
            cache[base.name][state] = {}
            for query_type in query_types:
                prompt = query_prompt(sentence, query_type)
                seq_len = min(max(len(loaded.tokenizer.encode(prompt, add_special_tokens=True)), 8), max_seq_len)
                positions = prompt_positions(base, state, sentence, prompt, loaded.tokenizer)
                cache[base.name][state][query_type] = {
                    "prompt": prompt,
                    "positions": positions,
                    "vectors": capture_prompt_tensors(loaded, prompt, positions, target_layers, modules, seq_len),
                    "seq_len": seq_len,
                }
        if (idx + 1) % progress_every == 0:
            log(f"captured query bases={idx + 1}/{len(bases)} elapsed={time.time() - start:.1f}s")
    return cache


def get_vec(cache: dict[str, dict[str, dict[str, Any]]], base: PassiveBase, state: str, query_type: str, layer: int, module: str, position: str) -> torch.Tensor | None:
    return (
        cache.get(base.name, {})
        .get(state, {})
        .get(query_type, {})
        .get("vectors", {})
        .get(layer, {})
        .get(module, {})
        .get(position)
    )


def role_swap_specs() -> list[tuple[str, str]]:
    return [
        ("active_ab", "active_ba"),
        ("active_ba", "active_ab"),
        ("passive_ab_by", "passive_ba_by"),
        ("passive_ba_by", "passive_ab_by"),
    ]


def patch_modes(src_state: str) -> list[tuple[str, dict[str, str]]]:
    if src_state.startswith("active"):
        return [
            ("subject_only", {"subject": "subject"}),
            ("object_only", {"object": "object"}),
            ("verb_only", {"verb": "verb"}),
            ("subject_object", {"subject": "subject", "object": "object"}),
            ("all_roles", {"subject": "subject", "object": "object", "verb": "verb"}),
        ]
    return [
        ("subject_only", {"subject": "subject"}),
        ("by_agent_only", {"by_agent": "by_agent"}),
        ("verb_only", {"verb": "verb"}),
        ("subject_by_agent", {"subject": "subject", "by_agent": "by_agent"}),
        ("all_roles", {"subject": "subject", "by_agent": "by_agent", "verb": "verb"}),
    ]


def patch_transplant_forward(
    loaded: Any,
    prompt: str,
    seq_len: int,
    layer_idx: int,
    module: str,
    src_positions: dict[str, list[int]],
    target_items: list[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    layers = get_layers(loaded.model)
    layer = layers[layer_idx]
    hooks = []

    def transplant(ref: torch.Tensor) -> torch.Tensor:
        patched = ref.clone()
        for src_pos, target_vec in target_items:
            span = src_positions.get(src_pos)
            if not span:
                continue
            value = target_vec.to(device=ref.device, dtype=ref.dtype)
            for idx in span:
                if 0 <= idx < patched.shape[1]:
                    patched[:, idx, :] = value
        return patched

    def pre_hook(_module: Any, inputs: Any) -> Any:
        if not (isinstance(inputs, tuple) and inputs):
            return inputs
        return (transplant(inputs[0]),) + tuple(inputs[1:])

    def output_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        ref = output[0] if isinstance(output, tuple) else output
        patched = transplant(ref)
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
            out = loaded.model(**tokenize(loaded, prompt, seq_len))
        return out.logits[0, -1, :].detach().cpu().float().clone()
    finally:
        for hook in hooks:
            hook.remove()


def summarize(rows: list[dict[str, Any]], baselines: list[dict[str, Any]]) -> dict[str, Any]:
    base_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in baselines:
        base_groups[(row["query_type"], row["state"])].append(row)
    baseline_summary = []
    for (query_type, state), items in sorted(base_groups.items()):
        baseline_summary.append({
            "query_type": query_type,
            "state": state,
            "accuracy": mean([1.0 if item["correct_choice"] else 0.0 for item in items]),
            "mean_margin": mean([float(item["margin"]) for item in items]),
            "n": len(items),
        })

    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["query_type"], row["patch_mode"], int(row["layer"]), row["module"])].append(row)
    curves = []
    for (query_type, patch_mode, layer, module), items in sorted(groups.items()):
        curves.append({
            "query_type": query_type,
            "patch_mode": patch_mode,
            "layer": layer,
            "module": module,
            "mean_target_margin": mean([float(item["patched_target_margin"]) for item in items]),
            "mean_source_margin": mean([float(item["source_target_margin"]) for item in items]),
            "mean_target_clean_margin": mean([float(item["target_clean_margin"]) for item in items]),
            "mean_margin_progress": mean([float(item["margin_progress"]) for item in items]),
            "flip_rate": mean([1.0 if item["patched_target_margin"] > 0 else 0.0 for item in items]),
            "n": len(items),
        })
    best = {}
    for query_type in sorted({row["query_type"] for row in curves}):
        items = [row for row in curves if row["query_type"] == query_type]
        if items:
            best[query_type] = max(items, key=lambda row: row["mean_margin_progress"])
    return {
        "baseline_summary": baseline_summary,
        "intervention_curve": curves,
        "best_by_query": best,
        "nonfinite_rows": sum(1 for row in rows if not bool(row.get("finite", True))),
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
        log(f"model={args.model} class={type(loaded.model).__name__} layers={len(layers)}")
        log(
            f"bases={len(bases)} train={len(train_bases)} test_total={len(all_test_bases)} "
            f"test_shard={test_start}:{test_end} layers={target_layers} modules={modules}"
        )

        cache = build_prompt_cache(loaded, bases, target_layers, modules, args.max_seq_len, args.progress_every)
        baseline_rows: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
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
                "target_layers": target_layers,
                "modules": modules,
                "train_bases": [base.name for base in train_bases],
                "all_test_bases": [base.name for base in all_test_bases],
                "test_bases": [base.name for base in test_bases],
                "baseline_rows": baseline_rows,
                "results": rows,
                "num_results": len(rows),
                "summary": summarize(rows, baseline_rows),
            }

        states = ["active_ab", "active_ba", "passive_ab_by", "passive_ba_by"]
        queries = ["agent", "patient"]
        start_time = time.time()
        for base_idx, base in enumerate(test_bases, start=1):
            texts = state_texts(base)
            clean_logits: dict[tuple[str, str], torch.Tensor] = {}
            for state in states:
                roles = state_roles(base, state)
                for query_type in queries:
                    prompt = cache[base.name][state][query_type]["prompt"]
                    seq_len = int(cache[base.name][state][query_type]["seq_len"])
                    logits = baseline_logits(loaded, prompt, seq_len)
                    clean_logits[(state, query_type)] = logits
                    correct = roles[query_type]
                    wrong = roles["patient" if query_type == "agent" else "agent"]
                    scored = score_margin_from_logits(logits, loaded.tokenizer, correct, wrong)
                    baseline_rows.append({
                        "base": base.name,
                        "state": state,
                        "query_type": query_type,
                        **scored,
                    })

            for src_state, dst_state in role_swap_specs():
                for query_type in queries:
                    src_roles = state_roles(base, src_state)
                    dst_roles = state_roles(base, dst_state)
                    target_correct = dst_roles[query_type]
                    target_wrong = dst_roles["patient" if query_type == "agent" else "agent"]
                    source_logits = clean_logits[(src_state, query_type)]
                    target_logits = clean_logits[(dst_state, query_type)]
                    source_target_margin = score_margin_from_logits(source_logits, loaded.tokenizer, target_correct, target_wrong)["margin"]
                    target_clean_margin = score_margin_from_logits(target_logits, loaded.tokenizer, target_correct, target_wrong)["margin"]
                    denom = target_clean_margin - source_target_margin
                    src_prompt = cache[base.name][src_state][query_type]["prompt"]
                    src_seq_len = int(cache[base.name][src_state][query_type]["seq_len"])
                    src_positions = cache[base.name][src_state][query_type]["positions"]
                    for patch_mode, mapping in patch_modes(src_state):
                        for layer in target_layers:
                            for module in modules:
                                target_items: list[tuple[str, torch.Tensor]] = []
                                for src_pos, dst_pos in mapping.items():
                                    vec = get_vec(cache, base, dst_state, query_type, layer, module, dst_pos)
                                    if vec is not None:
                                        target_items.append((src_pos, vec))
                                if not target_items:
                                    continue
                                patched_logits = patch_transplant_forward(
                                    loaded,
                                    src_prompt,
                                    src_seq_len,
                                    layer,
                                    module,
                                    src_positions,
                                    target_items,
                                )
                                finite = bool(torch.isfinite(patched_logits).all().item())
                                patched_score = score_margin_from_logits(patched_logits, loaded.tokenizer, target_correct, target_wrong)
                                patched_margin = float(patched_score["margin"])
                                margin_progress = 0.0 if abs(denom) < 1e-8 else (patched_margin - source_target_margin) / denom
                                rows.append({
                                    "base": base.name,
                                    "src_state": src_state,
                                    "dst_state": dst_state,
                                    "query_type": query_type,
                                    "patch_mode": patch_mode,
                                    "layer": layer,
                                    "module": module,
                                    "target_correct": target_correct,
                                    "target_wrong": target_wrong,
                                    "source_target_margin": source_target_margin,
                                    "target_clean_margin": target_clean_margin,
                                    "patched_target_margin": patched_margin,
                                    "margin_progress": margin_progress,
                                    "patched_correct": patched_margin > 0,
                                    "finite": finite,
                                    "num_token_patches": len(target_items),
                                })
            partial_file = partial_dir / f"{args.model}_phase303_{shard_label}.partial.json"
            partial_file.write_text(json.dumps(build_data(False), indent=2), encoding="utf-8")
            log(f"base {base_idx}/{len(test_bases)} rows={len(rows)} partial={partial_file} elapsed={time.time() - start_time:.1f}s")

        data = build_data(True)
        out_file = output_dir / f"{args.model}_phase303_role_query_closure_{shard_label}.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase303_role_query_closure"))
    parser.add_argument("--max-bases", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--modules", default="resid_in,resid_out,mlp_out")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=303)
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
