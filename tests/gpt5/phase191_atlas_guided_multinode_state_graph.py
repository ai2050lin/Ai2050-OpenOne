#!/usr/bin/env python3
"""Phase191: atlas-guided multi-node state graph intervention.

This phase tests whether candidate-ranking repair is better explained by a
small graph of MLP z-state updates than by a single node/channel group.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, compute_full_string_logprob_batch, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import answer_vectors, candidate_delta_metric  # noqa: E402
from phase595_mlp_update_causal_validation import load_phase594_mlp_nodes  # noqa: E402
from phase596_mlp_internal_gate_path_audit import collect_mlp_internal, get_mlp  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase191_atlas_guided_multinode_state_graph")
CONTROL_KINDS = {"random_control", "wrong_relation_control", "shuffled_node_control"}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: list[str]) -> dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def patch_full_logprob_multi_z(model, tokenizer, device, prompt: str, answer: str,
                               patches: list[dict[str, Any]], alpha: float) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        return -100.0

    all_ids = prompt_ids + answer_ids
    by_layer: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for patch in patches:
        pos = int(patch["patch_pos"])
        if pos < 0 or pos >= len(prompt_ids):
            continue
        by_layer.setdefault(int(patch["layer"]), []).append((pos, patch["vec"].to(device=device)))
    if not by_layer:
        return -100.0

    handles = []
    layers = get_layers(model)
    try:
        for layer_idx, layer_patches in by_layer.items():
            mlp = get_mlp(layers[layer_idx])

            def make_hook(items):
                def pre_hook(_module, inputs):
                    z = inputs[0]
                    z_new = z.clone()
                    for patch_pos, vec in items:
                        if patch_pos < z_new.shape[1]:
                            z_new[0, patch_pos, :] = z_new[0, patch_pos, :] + alpha * vec.to(dtype=z_new.dtype)
                    return (z_new,) + tuple(inputs[1:])

                return pre_hook

            handles.append(mlp.down_proj.register_forward_pre_hook(make_hook(layer_patches)))

        total = 0.0
        with torch.inference_mode():
            full_input = torch.tensor([all_ids], device=device)
            out = model(input_ids=full_input, return_dict=True)
            logits = out.logits[0].float()
            start = len(prompt_ids) - 1
            for i, tid in enumerate(answer_ids):
                pos = start + i
                if pos >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[pos], dim=-1)[tid].cpu())
        return total
    finally:
        for handle in handles:
            handle.remove()


def patched_score_map_multi_z(model, tokenizer, device, prompt: str, candidates: list[str],
                              patches: list[dict[str, Any]], alpha: float) -> dict[str, float]:
    return {
        ans: patch_full_logprob_multi_z(model, tokenizer, device, prompt, ans, patches, alpha)
        for ans in candidates
    }


def channel_contribs(mlp, E: torch.Tensor, correct: str, old_top_wrong: str,
                     values: list[str], delta_z: torch.Tensor) -> dict[str, torch.Tensor]:
    weight = mlp.down_proj.weight.detach().float().cpu()
    ci = values.index(correct)
    wi = values.index(old_top_wrong)
    mean_e = E.mean(dim=0)
    margin = (E[ci] - E[wi]).float().cpu() @ weight
    correct_specific = (E[ci] - mean_e).float().cpu() @ weight
    old_specific = (E[wi] - mean_e).float().cpu() @ weight
    dz = delta_z.float().cpu()
    return {
        "margin": margin * dz,
        "correct_specific": correct_specific * dz,
        "old_specific": old_specific * dz,
    }


def top_mask(contrib: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    k = min(k, contrib.numel())
    mask = torch.zeros_like(contrib, dtype=torch.bool)
    if k > 0:
        mask[torch.topk(contrib, k=k, largest=largest).indices] = True
    return mask


def masked(delta_z: torch.Tensor, mask: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
    out = torch.zeros_like(delta_z.float().cpu())
    out[mask] = sign * delta_z.float().cpu()[mask]
    return out


def node_vectors(delta_z: torch.Tensor, mlp, E: torch.Tensor, correct: str, old_top_wrong: str,
                 values: list[str], topk: int, seed: int) -> dict[str, torch.Tensor]:
    dz = delta_z.float().cpu()
    contrib = channel_contribs(mlp, E, correct, old_top_wrong, values, dz)
    margin_pos = top_mask(contrib["margin"], topk, largest=True)
    margin_neg = top_mask(contrib["margin"], topk, largest=False)
    correct_pos = top_mask(contrib["correct_specific"], topk, largest=True)
    old_pos = top_mask(contrib["old_specific"], topk, largest=True)

    combo_margin = masked(dz, margin_pos, +1.0) + masked(dz, margin_neg, -1.0)
    combo_correct_minus_old = masked(dz, correct_pos, +1.0) + masked(dz, old_pos, -1.0)
    return {
        "raw_delta": dz,
        f"combo_margin_top{topk}": combo_margin,
        f"combo_correct_minus_old_top{topk}": combo_correct_minus_old,
        f"random_same_norm_top{topk}": random_same_norm(combo_margin, seed=seed),
    }


def make_node_sets(valid_nodes: list[dict[str, Any]], max_set_size: int) -> list[tuple[dict[str, Any], ...]]:
    sets: list[tuple[dict[str, Any], ...]] = []
    for size in range(1, min(max_set_size, len(valid_nodes)) + 1):
        sets.extend(itertools.combinations(valid_nodes, size))
    return sets


def graph_key(node_set: tuple[dict[str, Any], ...], mode: str, kind: str, alpha: float) -> str:
    nodes = "+".join(f"{n['position']}@L{n['layer']}" for n in node_set)
    return f"{nodes}|{mode}|{kind}|a{alpha:g}"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for row in rows for k in row["patches"]})
    by_key: dict[str, Any] = {}
    for key in keys:
        items = [row["patches"][key] for row in rows if key in row["patches"]]
        entry = {
            "key": key,
            "nodes": items[0]["nodes"],
            "set_size": items[0]["set_size"],
            "mode": items[0]["mode"],
            "kind": items[0]["kind"],
            "alpha": items[0]["alpha"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_specific_margin_gain": 0.0,
            "mean_common_delta": 0.0,
            "mean_correct_specific": 0.0,
            "mean_old_top_wrong_specific": 0.0,
            "positive_margin": 0,
            "mean_vec_norm": 0.0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            entry["mean_vec_norm"] += item["vec_norm"]
            metric = item["metric"]
            entry["mean_margin_gain"] += metric["margin_gain"]
            entry["mean_specific_margin_gain"] += metric["specific_margin_gain"]
            entry["mean_common_delta"] += metric["common_delta"]
            entry["mean_correct_specific"] += metric["correct_specific"]
            entry["mean_old_top_wrong_specific"] += metric["old_top_wrong_specific"]
            entry["positive_margin"] += int(metric["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_key[key] = entry

    best = sorted(
        by_key.values(),
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["positive_margin_rate"]),
        reverse=True,
    )[:60]
    by_set_size: dict[str, dict[str, Any]] = {}
    for size in sorted({v["set_size"] for v in by_key.values()}):
        vals = [v for v in by_key.values() if v["set_size"] == size]
        sig = [v for v in vals if v["kind"] not in CONTROL_KINDS]
        ctrl = [v for v in vals if v["kind"] in CONTROL_KINDS]
        sig_best = sorted(sig, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:5]
        ctrl_best = sorted(ctrl, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)[:5]
        by_set_size[str(size)] = {"signal_best": sig_best, "control_best": ctrl_best}
    return {"by_key": by_key, "best": best, "by_set_size": by_set_size}


def run_model(args) -> dict[str, Any]:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = load_phase594_mlp_nodes(args.model, args.top_nodes)
        node_layers = sorted({node["layer"] for node in nodes})
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        signal_modes = ["raw_delta", f"combo_margin_top{args.topk}", f"combo_correct_minus_old_top{args.topk}"]
        log(
            f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes="
            f"{[(n['position'], n['layer']) for n in nodes]}, topk={args.topk}, "
            f"alphas={alphas}, max_set={args.max_set_size}"
        )

        rows: list[dict[str, Any]] = []
        target_seen = 0
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            wrong_scores = score_map(model, tokenizer, device, case["wrong_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"])
            base_cap = collect_mlp_internal(model, tokenizer, device, case["base_prompt"], node_layers)
            repair_cap = collect_mlp_internal(model, tokenizer, device, case["repair_prompt"], node_layers)
            wrong_cap = collect_mlp_internal(model, tokenizer, device, case["wrong_prompt"], node_layers)

            valid_nodes: list[dict[str, Any]] = []
            for node in nodes:
                pos_name = node["position"]
                li = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name)
                if bp is None or rp is None or li not in base_cap["z"] or li not in repair_cap["z"]:
                    continue
                if bp >= base_cap["z"][li].shape[1] or rp >= repair_cap["z"][li].shape[1]:
                    continue
                mlp = get_mlp(get_layers(model)[li])
                delta_z = repair_cap["z"][li][0, rp] - base_cap["z"][li][0, bp]
                vecs = node_vectors(delta_z, mlp, E, correct, old_top_wrong, values, args.topk, seed=si * 1009 + li)
                if wp is not None and li in wrong_cap["z"] and wp < wrong_cap["z"][li].shape[1]:
                    wrong_delta = wrong_cap["z"][li][0, wp] - base_cap["z"][li][0, bp]
                    vecs[f"wrong_relation_top{args.topk}"] = masked(
                        wrong_delta,
                        top_mask(torch.abs(wrong_delta.float().cpu()), args.topk, largest=True),
                        +1.0,
                    )
                valid_nodes.append({
                    "position": pos_name,
                    "layer": li,
                    "patch_pos": bp,
                    "phase594": node.get("phase594", {}),
                    "vecs": vecs,
                })

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "wrong_prompt": winner_stats(wrong_scores, correct),
                "patches": {},
            }

            for node_set in make_node_sets(valid_nodes, args.max_set_size):
                for alpha in alphas:
                    for mode in signal_modes:
                        patches = [
                            {"layer": n["layer"], "patch_pos": n["patch_pos"], "vec": n["vecs"][mode]}
                            for n in node_set if mode in n["vecs"]
                        ]
                        if len(patches) != len(node_set):
                            continue
                        key = graph_key(node_set, mode, "signal", alpha)
                        scores = patched_score_map_multi_z(model, tokenizer, device, case["base_prompt"], values, patches, alpha)
                        patched = winner_stats(scores, correct)
                        row["patches"][key] = {
                            "nodes": [{"position": n["position"], "layer": n["layer"]} for n in node_set],
                            "set_size": len(node_set),
                            "mode": mode,
                            "kind": "signal",
                            "alpha": alpha,
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                            "vec_norm": float(sum(torch.linalg.vector_norm(p["vec"].float()).cpu() for p in patches)),
                        }

                    mode = f"random_same_norm_top{args.topk}"
                    patches = [
                        {"layer": n["layer"], "patch_pos": n["patch_pos"], "vec": n["vecs"][mode]}
                        for n in node_set if mode in n["vecs"]
                    ]
                    if len(patches) == len(node_set):
                        key = graph_key(node_set, mode, "random_control", alpha)
                        scores = patched_score_map_multi_z(model, tokenizer, device, case["base_prompt"], values, patches, alpha)
                        patched = winner_stats(scores, correct)
                        row["patches"][key] = {
                            "nodes": [{"position": n["position"], "layer": n["layer"]} for n in node_set],
                            "set_size": len(node_set),
                            "mode": mode,
                            "kind": "random_control",
                            "alpha": alpha,
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                            "vec_norm": float(sum(torch.linalg.vector_norm(p["vec"].float()).cpu() for p in patches)),
                        }

                    mode = f"wrong_relation_top{args.topk}"
                    patches = [
                        {"layer": n["layer"], "patch_pos": n["patch_pos"], "vec": n["vecs"][mode]}
                        for n in node_set if mode in n["vecs"]
                    ]
                    if len(patches) == len(node_set):
                        key = graph_key(node_set, mode, "wrong_relation_control", alpha)
                        scores = patched_score_map_multi_z(model, tokenizer, device, case["base_prompt"], values, patches, alpha)
                        patched = winner_stats(scores, correct)
                        row["patches"][key] = {
                            "nodes": [{"position": n["position"], "layer": n["layer"]} for n in node_set],
                            "set_size": len(node_set),
                            "mode": mode,
                            "kind": "wrong_relation_control",
                            "alpha": alpha,
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                            "vec_norm": float(sum(torch.linalg.vector_norm(p["vec"].float()).cpu() for p in patches)),
                        }

                    if len(node_set) > 1:
                        vecs = [n["vecs"].get(f"combo_margin_top{args.topk}") for n in node_set]
                        if all(v is not None for v in vecs):
                            shifted = vecs[1:] + vecs[:1]
                            patches = [
                                {"layer": n["layer"], "patch_pos": n["patch_pos"], "vec": vec}
                                for n, vec in zip(node_set, shifted)
                            ]
                            key = graph_key(node_set, f"shuffled_combo_margin_top{args.topk}", "shuffled_node_control", alpha)
                            scores = patched_score_map_multi_z(model, tokenizer, device, case["base_prompt"], values, patches, alpha)
                            patched = winner_stats(scores, correct)
                            row["patches"][key] = {
                                "nodes": [{"position": n["position"], "layer": n["layer"]} for n in node_set],
                                "set_size": len(node_set),
                                "mode": f"shuffled_combo_margin_top{args.topk}",
                                "kind": "shuffled_node_control",
                                "alpha": alpha,
                                "winner": patched,
                                "metric": candidate_delta_metric(base_scores, scores, correct, old_top_wrong),
                                "vec_norm": float(sum(torch.linalg.vector_norm(p["vec"].float()).cpu() for p in patches)),
                            }
            rows.append(row)

        summary = summarize(rows)
        for item in summary["best"][:14]:
            log(
                f"best {item['key']}: switch={item['switch']}/{item['n']} "
                f"mgain={item['mean_margin_gain']:.3f} common={item['mean_common_delta']:.3f} "
                f"pm={item['positive_margin_rate']:.3f}"
            )
        return {
            "phase": 191,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "atlas_target": {
                "node_source": "Phase594 MLP update nodes",
                "edge": "candidate-specific ranking repair",
                "old_level": "Level3 strong transition / weak Level4 channel contribution",
                "target_level": "Level4/5 if signal graph beats controls",
                "stop_condition": "if multi-node signal does not beat matched controls, stop static z-patch route and move to forward dynamics",
            },
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "topk": args.topk,
            "alphas": alphas,
            "max_set_size": args.max_set_size,
            "nodes": nodes,
            "summary": summary,
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--alphas", default="1.0,2.0")
    parser.add_argument("--max-set-size", type=int, default=3)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 8
        args.top_nodes = min(args.top_nodes, 2)
        args.topk = min(args.topk, 32)
        args.alphas = "1.0"
        args.max_set_size = min(args.max_set_size, 2)
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 8)
        args.max_samples = max(args.max_samples, 128)
        args.top_nodes = max(args.top_nodes, 3)
        args.max_set_size = max(args.max_set_size, 3)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase191_{args.model}_atlas_guided_multinode_state_graph_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
