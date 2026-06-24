#!/usr/bin/env python3
"""Phase190: channel-state candidate ranking intervention.

This is the first CUDA follow-up after Atlas v0.  It tests whether MLP z-channel
groups can improve correct-vs-old-top-wrong ranking at the strongest Phase594
candidate nodes.
"""
from __future__ import annotations

import argparse
import gc
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


OUT_ROOT = Path("results/gpt5_phase190_channel_state_candidate_ranking")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_map(model, tokenizer, device, prompt: str, candidates: list[str]) -> dict[str, float]:
    return {k: v[0] for k, v in compute_full_string_logprob_batch(model, tokenizer, device, prompt, candidates).items()}


def patch_full_logprob_z(model, tokenizer, device, prompt: str, answer: str, layer_idx: int,
                         patch_pos: int, z_delta: torch.Tensor, alpha: float) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids or patch_pos < 0 or patch_pos >= len(prompt_ids):
        return -100.0
    all_ids = prompt_ids + answer_ids
    mlp = get_mlp(get_layers(model)[layer_idx])
    vec = z_delta.to(device=device)

    def pre_hook(_module, inputs):
        z = inputs[0]
        z_new = z.clone()
        z_new[0, patch_pos, :] = z_new[0, patch_pos, :] + alpha * vec.to(dtype=z_new.dtype)
        return (z_new,) + tuple(inputs[1:])

    handle = mlp.down_proj.register_forward_pre_hook(pre_hook)
    try:
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
        handle.remove()


def patched_score_map_z(model, tokenizer, device, prompt: str, candidates: list[str],
                        layer_idx: int, patch_pos: int, z_delta: torch.Tensor,
                        alpha: float) -> dict[str, float]:
    return {
        ans: patch_full_logprob_z(model, tokenizer, device, prompt, ans, layer_idx, patch_pos, z_delta, alpha)
        for ans in candidates
    }


def channel_readouts(mlp, E: torch.Tensor, correct: str, old_top_wrong: str, values: list[str]) -> dict[str, torch.Tensor]:
    weight = mlp.down_proj.weight.detach().float().cpu()  # hidden x intermediate
    ci = values.index(correct)
    wi = values.index(old_top_wrong)
    mean_e = E.mean(dim=0)
    return {
        "margin": (E[ci] - E[wi]).float().cpu() @ weight,
        "correct_specific": (E[ci] - mean_e).float().cpu() @ weight,
        "old_specific": (E[wi] - mean_e).float().cpu() @ weight,
        "common": mean_e.float().cpu() @ weight,
    }


def top_mask(contrib: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    k = min(k, contrib.numel())
    mask = torch.zeros_like(contrib, dtype=torch.bool)
    if k <= 0:
        return mask
    idx = torch.topk(contrib, k=k, largest=largest).indices
    mask[idx] = True
    return mask


def vec_from_mask(delta_z: torch.Tensor, mask: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
    out = torch.zeros_like(delta_z.float().cpu())
    out[mask] = sign * delta_z.float().cpu()[mask]
    return out


def make_channel_specs(delta_z: torch.Tensor, mlp, E: torch.Tensor, correct: str, old_top_wrong: str,
                       values: list[str], topks: list[int], seed: int) -> list[dict[str, Any]]:
    delta_z = delta_z.float().cpu()
    readouts = channel_readouts(mlp, E, correct, old_top_wrong, values)
    contrib = {name: readout * delta_z for name, readout in readouts.items()}
    specs: list[dict[str, Any]] = []
    for k in topks:
        margin_pos = top_mask(contrib["margin"], k, largest=True)
        correct_pos = top_mask(contrib["correct_specific"], k, largest=True)
        old_pos = top_mask(contrib["old_specific"], k, largest=True)
        common_pos = top_mask(contrib["common"].abs(), k, largest=True)
        margin_neg = top_mask(contrib["margin"], k, largest=False)

        boost_correct = vec_from_mask(delta_z, correct_pos, +1.0)
        suppress_old = vec_from_mask(delta_z, old_pos, -1.0)
        boost_margin = vec_from_mask(delta_z, margin_pos, +1.0)
        remove_bad_margin = vec_from_mask(delta_z, margin_neg, -1.0)
        common_only = vec_from_mask(delta_z, common_pos, +1.0)
        combo = boost_correct + suppress_old
        combo_margin = boost_margin + remove_bad_margin
        random_vec = random_same_norm(boost_margin, seed=seed + k * 17)

        specs.extend([
            {"name": f"boost_correct_top{k}", "kind": "boost_correct", "k": k, "vec": boost_correct},
            {"name": f"suppress_old_top{k}", "kind": "suppress_old", "k": k, "vec": suppress_old},
            {"name": f"boost_margin_top{k}", "kind": "boost_margin", "k": k, "vec": boost_margin},
            {"name": f"remove_bad_margin_top{k}", "kind": "remove_bad_margin", "k": k, "vec": remove_bad_margin},
            {"name": f"combo_correct_minus_old_top{k}", "kind": "combo_correct_minus_old", "k": k, "vec": combo},
            {"name": f"combo_margin_top{k}", "kind": "combo_margin", "k": k, "vec": combo_margin},
            {"name": f"common_control_top{k}", "kind": "common_control", "k": k, "vec": common_only},
            {"name": f"random_margin_norm_top{k}", "kind": "random_control", "k": k, "vec": random_vec},
        ])
    return specs


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for row in rows for k in row["patches"]})
    by_key: dict[str, Any] = {}
    for key in keys:
        items = [row["patches"][key] for row in rows if key in row["patches"]]
        entry = {
            "key": key,
            "position": items[0]["node"]["position"],
            "layer": items[0]["node"]["layer"],
            "kind": items[0]["kind"],
            "k": items[0]["k"],
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
        key=lambda x: (x["switch"], x["mean_margin_gain"], x["mean_specific_margin_gain"]),
        reverse=True,
    )[:40]
    return {"by_key": by_key, "best": best}


def run_model(args) -> dict[str, Any]:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        E = answer_vectors(model, tokenizer, values)
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = load_phase594_mlp_nodes(args.model, args.top_nodes)
        node_layers = sorted({node["layer"] for node in nodes})
        topks = [int(x) for x in args.topks.split(",") if x.strip()]
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}, topks={topks}, alphas={alphas}")

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

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "wrong_prompt": winner_stats(wrong_scores, correct),
                "patches": {},
            }

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
                wrong_delta_z = None
                if wp is not None and wrong_cap is not None and li in wrong_cap["z"] and wp < wrong_cap["z"][li].shape[1]:
                    wrong_delta_z = wrong_cap["z"][li][0, wp] - base_cap["z"][li][0, bp]

                specs = make_channel_specs(delta_z, mlp, E, correct, old_top_wrong, values, topks, seed=si * 1009 + li)
                if wrong_delta_z is not None:
                    for k in topks:
                        specs.append({
                            "name": f"wrong_relation_delta_top{k}",
                            "kind": "wrong_relation_control",
                            "k": k,
                            "vec": vec_from_mask(wrong_delta_z, top_mask(torch.abs(wrong_delta_z), k, largest=True), +1.0),
                        })

                for spec in specs:
                    for alpha in alphas:
                        key = f"{pos_name}|L{li}|{spec['name']}|a{alpha:g}"
                        patched_scores = patched_score_map_z(
                            model, tokenizer, device, case["base_prompt"], values, li, bp, spec["vec"], alpha
                        )
                        patched = winner_stats(patched_scores, correct)
                        row["patches"][key] = {
                            "node": {"position": pos_name, "layer": li},
                            "kind": spec["kind"],
                            "k": spec["k"],
                            "alpha": alpha,
                            "vec_norm": float(torch.linalg.vector_norm(spec["vec"].float()).cpu()),
                            "winner": patched,
                            "metric": candidate_delta_metric(base_scores, patched_scores, correct, old_top_wrong),
                        }
            rows.append(row)

        summary = summarize(rows)
        for item in summary["best"][:12]:
            log(
                f"best {item['key']}: switch={item['switch']}/{item['n']} "
                f"mgain={item['mean_margin_gain']:.3f} spec={item['mean_specific_margin_gain']:.3f} "
                f"common={item['mean_common_delta']:.3f}"
            )
        return {
            "phase": 190,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "topks": topks,
            "alphas": alphas,
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
    parser.add_argument("--topks", default="16,64,256")
    parser.add_argument("--alphas", default="0.5,1.0,2.0")
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
        args.topks = "16"
        args.alphas = "1.0"
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 8)
        args.max_samples = max(args.max_samples, 128)
        args.top_nodes = max(args.top_nodes, 3)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase190_{args.model}_channel_state_candidate_ranking_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
