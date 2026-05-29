from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hf_probe_env import get_layers, load_probe_model, release_loaded
from phase289_contract_scan import parse_csv, tokenize
from phase290_contract_break_scan import compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PassiveBase:
    name: str
    agent: str
    patient: str
    verb: str


def log(message: str) -> None:
    print(f"[phase301] {message}", flush=True)


def mean(values: list[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else 0.0


def build_bases() -> list[PassiveBase]:
    rows = [
        ("dog_cat", "dog", "cat", "chased"),
        ("teacher_student", "teacher", "student", "praised"),
        ("artist_portrait", "artist", "portrait", "painted"),
        ("chef_meal", "chef", "meal", "prepared"),
        ("storm_roof", "storm", "roof", "damaged"),
        ("doctor_patient", "doctor", "patient", "helped"),
        ("guard_thief", "guard", "thief", "caught"),
        ("manager_worker", "manager", "worker", "promoted"),
        ("nurse_child", "nurse", "child", "comforted"),
        ("coach_player", "coach", "player", "trained"),
        ("police_driver", "police", "driver", "stopped"),
        ("author_book", "author", "book", "wrote"),
        ("workers_bridge", "workers", "bridge", "built"),
        ("host_guests", "host", "guests", "invited"),
        ("committee_proposal", "committee", "proposal", "selected"),
        ("movers_piano", "movers", "piano", "moved"),
        ("lawyer_witness", "lawyer", "witness", "questioned"),
        ("scientist_sample", "scientist", "sample", "tested"),
        ("mechanic_engine", "mechanic", "engine", "repaired"),
        ("director_actor", "director", "actor", "hired"),
        ("farmer_horse", "farmer", "horse", "guided"),
        ("tailor_jacket", "tailor", "jacket", "fixed"),
        ("editor_article", "editor", "article", "revised"),
        ("gardener_flowers", "gardener", "flowers", "watered"),
        ("judge_case", "judge", "case", "reviewed"),
        ("engineer_machine", "engineer", "machine", "designed"),
        ("clerk_package", "clerk", "package", "delivered"),
        ("soldier_gate", "soldier", "gate", "guarded"),
        ("neighbor_window", "neighbor", "window", "closed"),
        ("parent_baby", "parent", "baby", "carried"),
        ("reporter_story", "reporter", "story", "published"),
        ("banker_account", "banker", "account", "opened"),
    ]
    return [PassiveBase(*row) for row in rows]


def state_texts(base: PassiveBase) -> dict[str, str]:
    a = base.agent
    p = base.patient
    v = base.verb
    return {
        "active_ab": f"the {a} {v} the {p}",
        "active_ba": f"the {p} {v} the {a}",
        "passive_ab_by": f"the {p} was {v} by the {a}",
        "passive_ba_by": f"the {a} was {v} by the {p}",
        "passive_ab_no": f"the {p} was {v}",
        "passive_ba_no": f"the {a} was {v}",
    }


def select_bases(max_bases: int, seed: int) -> list[PassiveBase]:
    bases = build_bases()
    rng = random.Random(seed)
    rng.shuffle(bases)
    selected = bases[:max_bases]
    selected.sort(key=lambda item: item.name)
    return selected


def split_bases(bases: list[PassiveBase], train_fraction: float, seed: int) -> tuple[list[PassiveBase], list[PassiveBase]]:
    rng = random.Random(seed + 301)
    items = list(bases)
    rng.shuffle(items)
    split = int(round(len(items) * train_fraction))
    split = max(1, min(len(items) - 1, split))
    train = sorted(items[:split], key=lambda item: item.name)
    test = sorted(items[split:], key=lambda item: item.name)
    return train, test


def parse_layers(value: str, n_layers: int) -> list[int]:
    out = set()
    for item in parse_csv(value):
        out.add(max(0, min(int(item), n_layers - 1)))
    return sorted(out)


def capture_vectors(
    loaded: Any,
    text: str,
    target_layers: list[int],
    modules: list[str],
    seq_len: int,
) -> dict[int, dict[str, torch.Tensor]]:
    layers = get_layers(loaded.model)
    captured: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []

    def store(layer_idx: int, name: str, value: torch.Tensor) -> None:
        captured.setdefault(layer_idx, {})[name] = value.detach().float().mean(dim=1).squeeze(0).cpu().clone()

    def make_pre_hook(layer_idx: int):
        def hook(_module: Any, inputs: Any) -> None:
            if isinstance(inputs, tuple) and inputs:
                store(layer_idx, "resid_in", inputs[0])
        return hook

    def make_output_hook(layer_idx: int, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            store(layer_idx, name, value)
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


def max_seq_len_for_texts(loaded: Any, texts: dict[str, str], max_seq_len: int) -> int:
    return min(
        max(len(loaded.tokenizer.encode(text, add_special_tokens=True)) for text in texts.values()),
        max_seq_len,
    )


def direction_specs() -> dict[str, list[tuple[str, str]]]:
    return {
        "voice": [
            ("active_ab", "passive_ab_by"),
            ("active_ba", "passive_ba_by"),
        ],
        "role_swap": [
            ("active_ab", "active_ba"),
            ("passive_ab_by", "passive_ba_by"),
            ("passive_ab_no", "passive_ba_no"),
        ],
        "by_phrase": [
            ("passive_ab_no", "passive_ab_by"),
            ("passive_ba_no", "passive_ba_by"),
        ],
    }


def compute_directions(
    vectors: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]],
    train_bases: list[PassiveBase],
    target_layers: list[int],
    modules: list[str],
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    directions: dict[str, dict[int, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    specs = direction_specs()
    for variable, pairs in specs.items():
        for layer in target_layers:
            for module in modules:
                diffs = []
                for base in train_bases:
                    states = vectors[base.name]
                    for src, dst in pairs:
                        src_vec = states[src].get(layer, {}).get(module)
                        dst_vec = states[dst].get(layer, {}).get(module)
                        if src_vec is not None and dst_vec is not None:
                            diffs.append(dst_vec - src_vec)
                if diffs:
                    directions[variable][layer][module] = torch.stack(diffs).mean(dim=0)
    return directions


def probe_rows(
    vectors: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]],
    train_bases: list[PassiveBase],
    test_bases: list[PassiveBase],
    directions: dict[str, dict[int, dict[str, torch.Tensor]]],
    target_layers: list[int],
    modules: list[str],
) -> list[dict[str, Any]]:
    state_labels = {
        "voice": [("active_ab", 0), ("active_ba", 0), ("passive_ab_by", 1), ("passive_ba_by", 1)],
        "role_swap": [("active_ab", 0), ("passive_ab_by", 0), ("passive_ab_no", 0), ("active_ba", 1), ("passive_ba_by", 1), ("passive_ba_no", 1)],
        "by_phrase": [("passive_ab_no", 0), ("passive_ba_no", 0), ("passive_ab_by", 1), ("passive_ba_by", 1)],
    }
    rows: list[dict[str, Any]] = []
    for variable, labels in state_labels.items():
        for layer in target_layers:
            for module in modules:
                direction = directions.get(variable, {}).get(layer, {}).get(module)
                if direction is None or float(direction.norm()) <= 1e-12:
                    continue
                train_pos = []
                train_neg = []
                for base in train_bases:
                    for state, expected in labels:
                        vec = vectors[base.name][state].get(layer, {}).get(module)
                        if vec is None:
                            continue
                        score = float(torch.dot(vec, direction))
                        (train_pos if expected else train_neg).append(score)
                threshold = 0.5 * (mean(train_pos) + mean(train_neg))
                correct = 0
                total = 0
                margins = []
                for base in test_bases:
                    for state, expected in labels:
                        vec = vectors[base.name][state].get(layer, {}).get(module)
                        if vec is None:
                            continue
                        score = float(torch.dot(vec, direction))
                        pred = 1 if score >= threshold else 0
                        correct += int(pred == expected)
                        total += 1
                        margins.append((score - threshold) if expected else (threshold - score))
                rows.append({
                    "variable": variable,
                    "layer": layer,
                    "module": module,
                    "direction_norm": float(direction.norm()),
                    "threshold": threshold,
                    "test_accuracy": correct / max(total, 1),
                    "test_total": total,
                    "mean_signed_margin": mean(margins),
                })
    return rows


def intervention_specs() -> list[tuple[str, str, str, str]]:
    specs: list[tuple[str, str, str, str]] = []
    for src, dst in direction_specs()["voice"]:
        specs.append(("voice", src, dst, "forward"))
        specs.append(("voice", dst, src, "reverse"))
    for src, dst in direction_specs()["role_swap"]:
        specs.append(("role_swap", src, dst, "forward"))
        specs.append(("role_swap", dst, src, "reverse"))
    for src, dst in direction_specs()["by_phrase"]:
        specs.append(("by_phrase", src, dst, "forward"))
        specs.append(("by_phrase", dst, src, "reverse"))
    return specs


def patch_direction_forward(
    loaded: Any,
    text: str,
    seq_len: int,
    layer_idx: int,
    module: str,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    layers = get_layers(loaded.model)
    layer = layers[layer_idx]
    hooks = []

    def add_direction(ref: torch.Tensor) -> torch.Tensor:
        d = direction.to(device=ref.device, dtype=ref.dtype)
        while d.dim() < ref.dim():
            d = d.unsqueeze(0)
        return ref + alpha * d

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


def summarize(rows: list[dict[str, Any]], probe: list[dict[str, Any]]) -> dict[str, Any]:
    by_variable_layer_module: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_variable_direction: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(float(row["alpha"]) - 1.0) > 1e-9:
            continue
        by_variable_layer_module[(str(row["variable"]), str(row["direction"]), int(row["layer"]), str(row["module"]))].append(row)
        by_variable_direction[(str(row["variable"]), str(row["direction"]))].append(row)
    curves = []
    for (variable, direction, layer, module), items in sorted(by_variable_layer_module.items()):
        curves.append({
            "variable": variable,
            "direction": direction,
            "layer": layer,
            "module": module,
            "mean_progress": mean([float(x["progress"]) for x in items]),
            "mean_kl_ratio": mean([float(x["kl_ratio"]) for x in items]),
            "mean_logit_delta_ratio": mean([float(x["logit_delta_ratio"]) for x in items]),
            "n": len(items),
        })
    variable_direction = []
    for (variable, direction), items in sorted(by_variable_direction.items()):
        variable_direction.append({
            "variable": variable,
            "direction": direction,
            "mean_progress": mean([float(x["progress"]) for x in items]),
            "mean_kl_ratio": mean([float(x["kl_ratio"]) for x in items]),
            "n": len(items),
        })
    best = {}
    for variable in sorted({row["variable"] for row in curves}):
        for direction in sorted({row["direction"] for row in curves if row["variable"] == variable}):
            items = [row for row in curves if row["variable"] == variable and row["direction"] == direction]
            if items:
                best[f"{variable}:{direction}"] = max(items, key=lambda row: row["mean_progress"])
    probe_best = {}
    for variable in sorted({row["variable"] for row in probe}):
        items = [row for row in probe if row["variable"] == variable]
        if items:
            probe_best[variable] = max(items, key=lambda row: row["test_accuracy"])
    return {
        "probe_best": probe_best,
        "layer_module_curve": curves,
        "variable_direction_curve": variable_direction,
        "best_by_variable_direction": best,
        "nonfinite_rows": sum(1 for row in rows if float(row.get("finite", 1.0)) < 0.5),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bases = select_bases(args.max_bases, args.seed)
    train_bases, test_bases = split_bases(bases, args.train_fraction, args.seed)

    loaded = None
    try:
        loaded = load_probe_model(args.model)
        layers = get_layers(loaded.model)
        target_layers = parse_layers(args.layers, len(layers))
        modules = parse_csv(args.modules)
        alphas = [float(item) for item in parse_csv(args.alphas)]
        log(f"model={args.model} class={type(loaded.model).__name__} layers={len(layers)}")
        log(f"env dtype={os.environ.get('PROBE_TORCH_DTYPE')} attn={os.environ.get('PROBE_ATTN_IMPLEMENTATION')} auto={os.environ.get('PROBE_DEVICE_MAP_AUTO_MODELS')}")
        log(f"bases={len(bases)} train={len(train_bases)} test={len(test_bases)} layers={target_layers} modules={modules} alphas={alphas}")

        vectors: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]] = {}
        start = time.time()
        for idx, base in enumerate(bases):
            texts = state_texts(base)
            seq_len = max_seq_len_for_texts(loaded, texts, args.max_seq_len)
            vectors[base.name] = {
                state: capture_vectors(loaded, text, target_layers, modules, seq_len)
                for state, text in texts.items()
            }
            if (idx + 1) % args.progress_every == 0:
                log(f"captured bases={idx + 1}/{len(bases)} elapsed={time.time() - start:.1f}s")

        directions = compute_directions(vectors, train_bases, target_layers, modules)
        probe = probe_rows(vectors, train_bases, test_bases, directions, target_layers, modules)
        log(f"probe rows={len(probe)} best_acc={max([row['test_accuracy'] for row in probe], default=0):.4f}")

        rows: list[dict[str, Any]] = []
        specs = intervention_specs()
        for idx, base in enumerate(test_bases):
            texts = state_texts(base)
            seq_len = max_seq_len_for_texts(loaded, texts, args.max_seq_len)
            logits = {state: baseline_logits(loaded, text, seq_len) for state, text in texts.items()}
            for variable, src, dst, direction_name in specs:
                kl = float(F.kl_div(F.log_softmax(logits[src], dim=-1), F.softmax(logits[dst], dim=-1), reduction="sum"))
                if kl < 1e-8:
                    continue
                for layer in target_layers:
                    for module in modules:
                        direction = directions.get(variable, {}).get(layer, {}).get(module)
                        if direction is None:
                            continue
                        sign = 1.0 if direction_name == "forward" else -1.0
                        for alpha in alphas:
                            patched = patch_direction_forward(
                                loaded,
                                texts[src],
                                seq_len,
                                layer,
                                module,
                                direction,
                                sign * alpha,
                            )
                            metrics = compute_metrics(patched, logits[src], logits[dst], kl) or {}
                            rows.append({
                                "base": base.name,
                                "agent": base.agent,
                                "patient": base.patient,
                                "verb": base.verb,
                                "variable": variable,
                                "source_state": src,
                                "target_state": dst,
                                "direction": direction_name,
                                "layer": layer,
                                "module": module,
                                "alpha": alpha,
                                **metrics,
                            })
            if (idx + 1) % args.progress_every == 0:
                log(f"intervention bases={idx + 1}/{len(test_bases)} rows={len(rows)} elapsed={time.time() - start:.1f}s")

        data = {
            "model": args.model,
            "class": type(loaded.model).__name__,
            "complete": True,
            "num_bases": len(bases),
            "num_train_bases": len(train_bases),
            "num_test_bases": len(test_bases),
            "num_results": len(rows),
            "target_layers": target_layers,
            "modules": modules,
            "alphas": alphas,
            "bases": [base.__dict__ for base in bases],
            "train_bases": [base.name for base in train_bases],
            "test_bases": [base.name for base in test_bases],
            "probe_rows": probe,
            "results": rows,
            "summary": summarize(rows, probe),
        }
        out_file = output_dir / f"{args.model}_phase301_passive_factor_closure.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase301_passive_factor_closure"))
    parser.add_argument("--max-bases", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--modules", default="resid_in,resid_out,mlp_out")
    parser.add_argument("--alphas", default="0,0.5,1.0")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=301)
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
