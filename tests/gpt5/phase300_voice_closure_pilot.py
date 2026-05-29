from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hf_probe_env import get_layers, load_probe_model, release_loaded
from phase289_contract_scan import parse_csv, tokenize
from phase290_contract_break_scan import compute_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES = ["resid_in", "resid_out", "mlp_out"]


@dataclass(frozen=True)
class VoicePair:
    name: str
    subtype: str
    active: str
    passive: str
    agent: str
    patient: str
    verb: str


def log(message: str) -> None:
    print(f"[phase300] {message}", flush=True)


def mean(values: list[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else 0.0


def build_voice_pairs() -> list[VoicePair]:
    rows: list[VoicePair] = []
    transitive = [
        ("dog", "cat", "chased"),
        ("teacher", "student", "praised"),
        ("artist", "portrait", "painted"),
        ("chef", "meal", "prepared"),
        ("storm", "roof", "damaged"),
        ("doctor", "patient", "helped"),
        ("guard", "thief", "caught"),
        ("manager", "worker", "promoted"),
        ("nurse", "child", "comforted"),
        ("coach", "player", "trained"),
        ("police", "driver", "stopped"),
        ("author", "book", "wrote"),
        ("workers", "bridge", "built"),
        ("host", "guests", "invited"),
        ("committee", "proposal", "selected"),
        ("movers", "piano", "moved"),
        ("lawyer", "witness", "questioned"),
        ("scientist", "sample", "tested"),
        ("mechanic", "engine", "repaired"),
        ("director", "actor", "hired"),
        ("farmer", "horse", "guided"),
        ("pilot", "plane", "landed"),
        ("tailor", "jacket", "fixed"),
        ("editor", "article", "revised"),
        ("gardener", "flowers", "watered"),
        ("judge", "case", "reviewed"),
        ("engineer", "machine", "designed"),
        ("clerk", "package", "delivered"),
        ("soldier", "gate", "guarded"),
        ("neighbor", "window", "closed"),
    ]
    modifiers = [
        "near the old tree",
        "beside the quiet road",
        "inside the small room",
        "after the meeting",
        "during the storm",
        "with great care",
    ]

    for idx, (agent, patient, verb) in enumerate(transitive):
        active = f"the {agent} {verb} the {patient}"
        passive = f"the {patient} was {verb} by the {agent}"
        rows.append(VoicePair(f"by_{idx:03d}", "by_phrase", active, passive, agent, patient, verb))

        active_get = f"the {agent} {verb} the {patient}"
        passive_get = f"the {patient} got {verb} by the {agent}"
        rows.append(VoicePair(f"get_{idx:03d}", "get_passive", active_get, passive_get, agent, patient, verb))

        modifier = modifiers[idx % len(modifiers)]
        active_long = f"the {agent} {verb} the {patient} {modifier}"
        passive_long = f"the {patient} {modifier} was {verb} by the {agent}"
        rows.append(VoicePair(f"long_{idx:03d}", "long_passive", active_long, passive_long, agent, patient, verb))

    datives = [
        ("teacher", "student", "book"),
        ("committee", "scientist", "award"),
        ("company", "worker", "offer"),
        ("manager", "assistant", "task"),
        ("agent", "customer", "ticket"),
        ("coach", "player", "chance"),
        ("officer", "driver", "warning"),
        ("director", "actor", "role"),
        ("host", "guest", "gift"),
        ("doctor", "patient", "note"),
        ("lawyer", "client", "document"),
        ("parent", "child", "snack"),
        ("clerk", "visitor", "form"),
        ("guide", "tourist", "map"),
        ("nurse", "doctor", "report"),
        ("engineer", "team", "plan"),
        ("artist", "friend", "sketch"),
        ("chef", "waiter", "recipe"),
        ("pilot", "crew", "signal"),
        ("judge", "lawyer", "order"),
    ]
    for idx, (agent, recipient, obj) in enumerate(datives):
        active = f"the {agent} gave the {recipient} a {obj}"
        passive = f"the {recipient} was given a {obj} by the {agent}"
        rows.append(VoicePair(f"dative_{idx:03d}", "dative_passive", active, passive, agent, recipient, "gave"))
    return rows


def select_pairs(pairs: list[VoicePair], max_pairs_per_subtype: int, seed: int) -> list[VoicePair]:
    by_subtype: dict[str, list[VoicePair]] = defaultdict(list)
    for pair in pairs:
        by_subtype[pair.subtype].append(pair)
    rng = random.Random(seed)
    selected: list[VoicePair] = []
    for subtype in sorted(by_subtype):
        items = list(by_subtype[subtype])
        rng.shuffle(items)
        selected.extend(items[:max_pairs_per_subtype])
    selected.sort(key=lambda item: (item.subtype, item.name))
    return selected


def stratified_train_test_split(
    pairs: list[VoicePair],
    train_fraction: float,
    seed: int,
) -> tuple[list[VoicePair], list[VoicePair]]:
    by_subtype: dict[str, list[VoicePair]] = defaultdict(list)
    for pair in pairs:
        by_subtype[pair.subtype].append(pair)
    rng = random.Random(seed + 1009)
    train: list[VoicePair] = []
    test: list[VoicePair] = []
    for subtype in sorted(by_subtype):
        items = list(by_subtype[subtype])
        rng.shuffle(items)
        split = int(round(len(items) * train_fraction))
        split = max(1, min(len(items) - 1, split))
        train.extend(items[:split])
        test.extend(items[split:])
    train.sort(key=lambda item: (item.subtype, item.name))
    test.sort(key=lambda item: (item.subtype, item.name))
    return train, test


def parse_layers(value: str, n_layers: int) -> list[int]:
    out = set()
    for item in parse_csv(value):
        layer = int(item)
        out.add(max(0, min(layer, n_layers - 1)))
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
        vec = value.detach().float().mean(dim=1).squeeze(0).cpu().clone()
        captured.setdefault(layer_idx, {})[name] = vec

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


def compute_voice_directions(
    vectors: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]],
    train_pairs: list[VoicePair],
    target_layers: list[int],
    modules: list[str],
) -> dict[int, dict[str, torch.Tensor]]:
    directions: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    for layer in target_layers:
        for module in modules:
            diffs = []
            for pair in train_pairs:
                active_vec = vectors[pair.name]["active"].get(layer, {}).get(module)
                passive_vec = vectors[pair.name]["passive"].get(layer, {}).get(module)
                if active_vec is None or passive_vec is None:
                    continue
                diffs.append(passive_vec - active_vec)
            if diffs:
                directions[layer][module] = torch.stack(diffs).mean(dim=0)
    return directions


def probe_accuracy(
    vectors: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]],
    train_pairs: list[VoicePair],
    test_pairs: list[VoicePair],
    directions: dict[int, dict[str, torch.Tensor]],
    target_layers: list[int],
    modules: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in target_layers:
        for module in modules:
            direction = directions.get(layer, {}).get(module)
            if direction is None or float(direction.norm()) <= 1e-12:
                continue
            train_scores_active = []
            train_scores_passive = []
            for pair in train_pairs:
                active_vec = vectors[pair.name]["active"].get(layer, {}).get(module)
                passive_vec = vectors[pair.name]["passive"].get(layer, {}).get(module)
                if active_vec is not None:
                    train_scores_active.append(float(torch.dot(active_vec, direction)))
                if passive_vec is not None:
                    train_scores_passive.append(float(torch.dot(passive_vec, direction)))
            threshold = 0.5 * (mean(train_scores_active) + mean(train_scores_passive))
            correct = 0
            total = 0
            margin_values = []
            for pair in test_pairs:
                for label, expected in [("active", 0), ("passive", 1)]:
                    vec = vectors[pair.name][label].get(layer, {}).get(module)
                    if vec is None:
                        continue
                    score = float(torch.dot(vec, direction))
                    pred = 1 if score >= threshold else 0
                    correct += int(pred == expected)
                    total += 1
                    signed_margin = (score - threshold) if expected == 1 else (threshold - score)
                    margin_values.append(signed_margin)
            rows.append({
                "layer": layer,
                "module": module,
                "direction_norm": float(direction.norm()),
                "threshold": threshold,
                "test_accuracy": correct / max(total, 1),
                "test_total": total,
                "mean_signed_margin": mean(margin_values),
            })
    return rows


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


def summarize_interventions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_module_direction: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_subtype_direction: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            by_layer_module_direction[(int(row["layer"]), str(row["module"]), str(row["direction"]))].append(row)
            by_subtype_direction[(str(row["subtype"]), str(row["direction"]))].append(row)

    layer_module_curve = []
    for (layer, module, direction), items in sorted(by_layer_module_direction.items()):
        layer_module_curve.append({
            "layer": layer,
            "module": module,
            "direction": direction,
            "mean_progress": mean([float(x["progress"]) for x in items]),
            "mean_kl_ratio": mean([float(x["kl_ratio"]) for x in items]),
            "mean_logit_delta_ratio": mean([float(x["logit_delta_ratio"]) for x in items]),
            "n": len(items),
        })
    subtype_curve = []
    for (subtype, direction), items in sorted(by_subtype_direction.items()):
        subtype_curve.append({
            "subtype": subtype,
            "direction": direction,
            "mean_progress": mean([float(x["progress"]) for x in items]),
            "mean_kl_ratio": mean([float(x["kl_ratio"]) for x in items]),
            "n": len(items),
        })
    best = {}
    for direction in sorted({row["direction"] for row in layer_module_curve}):
        items = [row for row in layer_module_curve if row["direction"] == direction]
        if items:
            best[direction] = max(items, key=lambda row: row["mean_progress"])
    return {
        "layer_module_curve": layer_module_curve,
        "subtype_curve": subtype_curve,
        "best_by_direction": best,
        "nonfinite_rows": sum(1 for row in rows if float(row.get("finite", 1.0)) < 0.5),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = select_pairs(build_voice_pairs(), args.max_pairs_per_subtype, args.seed)
    train_pairs, test_pairs = stratified_train_test_split(pairs, args.train_fraction, args.seed)
    if not test_pairs:
        raise SystemExit("Need at least one test pair")

    loaded = None
    try:
        loaded = load_probe_model(args.model)
        layers = get_layers(loaded.model)
        target_layers = parse_layers(args.layers, len(layers))
        modules = parse_csv(args.modules)
        alphas = [float(x) for x in parse_csv(args.alphas)]
        log(f"model={args.model} class={type(loaded.model).__name__} layers={len(layers)}")
        log(f"env dtype={os.environ.get('PROBE_TORCH_DTYPE')} attn={os.environ.get('PROBE_ATTN_IMPLEMENTATION')} auto={os.environ.get('PROBE_DEVICE_MAP_AUTO_MODELS')}")
        log(f"pairs={len(pairs)} train={len(train_pairs)} test={len(test_pairs)} layers={target_layers} modules={modules} alphas={alphas}")

        vectors: dict[str, dict[str, dict[int, dict[str, torch.Tensor]]]] = {}
        start = time.time()
        for idx, pair in enumerate(pairs):
            seq_len = min(
                max(
                    len(loaded.tokenizer.encode(pair.active, add_special_tokens=True)),
                    len(loaded.tokenizer.encode(pair.passive, add_special_tokens=True)),
                ),
                args.max_seq_len,
            )
            vectors[pair.name] = {
                "active": capture_vectors(loaded, pair.active, target_layers, modules, seq_len),
                "passive": capture_vectors(loaded, pair.passive, target_layers, modules, seq_len),
            }
            if (idx + 1) % args.progress_every == 0:
                log(f"captured pairs={idx + 1}/{len(pairs)} elapsed={time.time() - start:.1f}s")

        directions = compute_voice_directions(vectors, train_pairs, target_layers, modules)
        probe_rows = probe_accuracy(vectors, train_pairs, test_pairs, directions, target_layers, modules)
        log(f"probe rows={len(probe_rows)} best_acc={max([row['test_accuracy'] for row in probe_rows], default=0):.4f}")

        rows: list[dict[str, Any]] = []
        for idx, pair in enumerate(test_pairs):
            seq_len = min(
                max(
                    len(loaded.tokenizer.encode(pair.active, add_special_tokens=True)),
                    len(loaded.tokenizer.encode(pair.passive, add_special_tokens=True)),
                ),
                args.max_seq_len,
            )
            logits_active = baseline_logits(loaded, pair.active, seq_len)
            logits_passive = baseline_logits(loaded, pair.passive, seq_len)
            kl_ap = float(F.kl_div(F.log_softmax(logits_active, dim=-1), F.softmax(logits_passive, dim=-1), reduction="sum"))
            kl_pa = float(F.kl_div(F.log_softmax(logits_passive, dim=-1), F.softmax(logits_active, dim=-1), reduction="sum"))
            if kl_ap < 1e-8 or kl_pa < 1e-8:
                continue
            for layer in target_layers:
                for module in modules:
                    direction = directions.get(layer, {}).get(module)
                    if direction is None:
                        continue
                    for alpha in alphas:
                        patched_ap = patch_direction_forward(
                            loaded, pair.active, seq_len, layer, module, direction, alpha
                        )
                        metrics_ap = compute_metrics(patched_ap, logits_active, logits_passive, kl_ap) or {}
                        rows.append({
                            "pair": pair.name,
                            "subtype": pair.subtype,
                            "agent": pair.agent,
                            "patient": pair.patient,
                            "verb": pair.verb,
                            "layer": layer,
                            "module": module,
                            "alpha": alpha,
                            "direction": "active_to_passive",
                            **metrics_ap,
                        })

                        patched_pa = patch_direction_forward(
                            loaded, pair.passive, seq_len, layer, module, direction, -alpha
                        )
                        metrics_pa = compute_metrics(patched_pa, logits_passive, logits_active, kl_pa) or {}
                        rows.append({
                            "pair": pair.name,
                            "subtype": pair.subtype,
                            "agent": pair.agent,
                            "patient": pair.patient,
                            "verb": pair.verb,
                            "layer": layer,
                            "module": module,
                            "alpha": alpha,
                            "direction": "passive_to_active",
                            **metrics_pa,
                        })
            if (idx + 1) % args.progress_every == 0:
                log(f"intervention test_pairs={idx + 1}/{len(test_pairs)} rows={len(rows)} elapsed={time.time() - start:.1f}s")

        data = {
            "model": args.model,
            "class": type(loaded.model).__name__,
            "complete": True,
            "num_pairs": len(pairs),
            "num_train_pairs": len(train_pairs),
            "num_test_pairs": len(test_pairs),
            "num_results": len(rows),
            "target_layers": target_layers,
            "modules": modules,
            "alphas": alphas,
            "pairs": [pair.__dict__ for pair in pairs],
            "probe_rows": probe_rows,
            "results": rows,
            "summary": summarize_interventions(rows),
        }
        out_file = output_dir / f"{args.model}_phase300_voice_closure_pilot.json"
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log(f"saved {out_file}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase300_voice_closure_pilot"))
    parser.add_argument("--max-pairs-per-subtype", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--modules", default="resid_in,resid_out,mlp_out")
    parser.add_argument("--alphas", default="0,0.5,1.0,1.5")
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=300)
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
