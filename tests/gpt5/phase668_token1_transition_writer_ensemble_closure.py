#!/usr/bin/env python3
"""
Phase 668: Token1 Transition Writer Ensemble Closure.

Phase 667 suggests the value-specific token1 transition state is stronger at
layer_out / layer_input boundaries than in single attention-head slices. This
phase tests whether small writer ensembles can approach full-boundary restore.
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
from typing import Dict, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_num_heads, get_o_proj  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase599_final_layer_washout_decomposition import get_attn  # noqa: E402
from phase665_autoregressive_continuation_controller_localization import collect_id_components, install_id_patch_hooks, token_top_metric  # noqa: E402
from phase666_token1_transition_boundary_remove_restore import (  # noqa: E402
    compute_source_patches,
    logits_with_hooks,
    phase665_path,
    pick_mismatch,
)
from phase667_value_specific_token1_transition_writer_localization import collect_o_input  # noqa: E402
from phase656_format_prior_writer_localization_audit import SITE_SPECS  # noqa: E402


OUT_ROOT = Path("results/glm5_phase668_token1_transition_writer_ensemble_closure")
ENSEMBLES = {
    "qwen3": [
        {"name": "full_L23_layer_input", "kind": "component_set", "components": [(23, "layer_input")]},
        {"name": "full_L22_layer_out", "kind": "component_set", "components": [(22, "layer_out")]},
        {"name": "L22_attn_mlp", "kind": "component_set", "components": [(22, "attn_out"), (22, "mlp_out")]},
        {"name": "L22_heads10_11", "kind": "head_set", "layer": 22, "heads": [10, 11]},
    ],
    "glm4": [
        {"name": "full_L22_attn_out", "kind": "component_set", "components": [(22, "attn_out")]},
        {"name": "full_L22_layer_input", "kind": "component_set", "components": [(22, "layer_input")]},
        {"name": "L21_layer_out_L22_attn_mlp", "kind": "component_set", "components": [(21, "layer_out"), (22, "attn_out"), (22, "mlp_out")]},
        {"name": "L22_heads7_13", "kind": "head_set", "layer": 22, "heads": [7, 13]},
    ],
    "deepseek7b": [
        {"name": "full_L21_layer_out", "kind": "component_set", "components": [(21, "layer_out")]},
        {"name": "full_L22_layer_input", "kind": "component_set", "components": [(22, "layer_input")]},
        {"name": "L21_attn_mlp", "kind": "component_set", "components": [(21, "attn_out"), (21, "mlp_out")]},
        {"name": "L21_heads14_17", "kind": "head_set", "layer": 21, "heads": [14, 17]},
    ],
}
INTERVENTIONS = ["zero_remove", "mismatch_restore", "correct_restore"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def head_patch_logits(
    model,
    tokenizer,
    device,
    ids: List[int],
    prompt: str,
    source_patches,
    combo,
    last_writers,
    layer_idx: int,
    head_ids: List[int],
    target_vec: torch.Tensor,
) -> torch.Tensor:
    layers = get_layers(model)
    attn = get_attn(layers[layer_idx])
    o_proj = get_o_proj(attn)
    n_heads = get_num_heads(model, attn)
    head_dim = target_vec.numel() // max(1, n_heads)
    valid_heads = [h for h in head_ids if 0 <= h < n_heads]

    def hook(_module, inputs):
        x = inputs[0]
        y = x.clone()
        pos = len(ids) - 1
        if 0 <= pos < y.shape[1]:
            for head in valid_heads:
                start = head * head_dim
                end = target_vec.numel() if head == n_heads - 1 else (head + 1) * head_dim
                y[0, pos, start:end] = target_vec[start:end].to(device=y.device, dtype=y.dtype)
        return (y,) + tuple(inputs[1:])

    h = o_proj.register_forward_pre_hook(hook)
    try:
        return logits_with_hooks(model, tokenizer, device, ids, prompt, source_patches, combo, last_writers)
    finally:
        h.remove()


def component_patch_logits(
    model,
    tokenizer,
    device,
    ids: List[int],
    prompt: str,
    source_patches,
    combo,
    last_writers,
    patch_items: List[Tuple[int, str, torch.Tensor]],
) -> torch.Tensor:
    pos = len(ids) - 1
    patches = [(li, comp, [pos], [target]) for li, comp, target in patch_items]
    return logits_with_hooks(model, tokenizer, device, ids, prompt, source_patches, combo, last_writers, patches)


def summarize(rows: List[Dict]) -> Dict:
    groups: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (row["pair_task"], row["site"], row["combo_name"], row["ensemble"], row["intervention"])
        g = groups.setdefault(key, {
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "ensemble": row["ensemble"],
            "kind": row["kind"],
            "intervention": row["intervention"],
            "n": 0,
            "top1": 0,
            "rank_delta": [],
            "margin_delta": [],
            "top1_text": {},
        })
        g["n"] += 1
        g["top1"] += int(row["expected_rank"] == 1)
        g["rank_delta"].append(float(row["baseline_expected_rank"] - row["expected_rank"]))
        g["margin_delta"].append(float(row["expected_minus_top1"] - row["baseline_expected_minus_top1"]))
        text = row["top1_text"].replace("\n", "\\n")
        g["top1_text"][text] = g["top1_text"].get(text, 0) + 1

    intervention = []
    for g in groups.values():
        r = dict(g)
        r["expected_top1_rate"] = g["top1"] / max(1, g["n"])
        r["mean_rank_delta_vs_baseline"] = mean(g["rank_delta"])
        r["mean_margin_delta_vs_baseline"] = mean(g["margin_delta"])
        r["top1_text"] = dict(sorted(g["top1_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        for k in ["top1", "rank_delta", "margin_delta"]:
            r.pop(k, None)
        intervention.append(r)

    by_ens: Dict[Tuple, Dict] = {}
    for r in intervention:
        key = (r["pair_task"], r["site"], r["combo_name"], r["ensemble"])
        e = by_ens.setdefault(key, {
            "pair_task": r["pair_task"],
            "site": r["site"],
            "combo_name": r["combo_name"],
            "ensemble": r["ensemble"],
            "kind": r["kind"],
            "n": r["n"],
            "interventions": {},
        })
        e["interventions"][r["intervention"]] = r

    ensemble_out = []
    for e in by_ens.values():
        correct = e["interventions"].get("correct_restore", {})
        mismatch = e["interventions"].get("mismatch_restore", {})
        zero = e["interventions"].get("zero_remove", {})
        e["correct_top1_rate"] = correct.get("expected_top1_rate", 0.0)
        e["mismatch_top1_rate"] = mismatch.get("expected_top1_rate", 0.0)
        e["correct_minus_mismatch_margin_delta"] = (
            correct.get("mean_margin_delta_vs_baseline", 0.0) - mismatch.get("mean_margin_delta_vs_baseline", 0.0)
        )
        e["correct_minus_zero_margin_delta"] = (
            correct.get("mean_margin_delta_vs_baseline", 0.0) - zero.get("mean_margin_delta_vs_baseline", 0.0)
        )
        ensemble_out.append(e)
    ensemble_out.sort(
        key=lambda r: (
            -r["correct_minus_mismatch_margin_delta"],
            -r["correct_top1_rate"],
            r["mismatch_top1_rate"],
            r["ensemble"],
        )
    )
    intervention.sort(key=lambda r: (r["pair_task"], r["site"], r["combo_name"], r["ensemble"], r["intervention"]))
    return {"ensemble_specificity": ensemble_out, "intervention_summary": intervention}


def run_model(args) -> Dict:
    data_path = phase665_path(args.model)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing Phase 665 confirm result: {data_path}")
    phase665 = json.loads(data_path.read_text(encoding="utf-8"))
    failures = phase665["selected_failures"][: args.max_failures]
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        ensembles = []
        for spec in ENSEMBLES[args.model]:
            if spec["kind"] == "component_set":
                comps = [(li, comp) for li, comp in spec["components"] if 0 <= li < info.n_layers]
                if comps:
                    s = dict(spec)
                    s["components"] = comps
                    ensembles.append(s)
            else:
                if 0 <= spec["layer"] < info.n_layers:
                    ensembles.append(spec)
        rows = []
        site_specs = {s["name"]: s for s in SITE_SPECS[args.model]}
        log(f"{args.model}: failures={len(failures)}, ensembles={ensembles}")
        for i, failure in enumerate(failures):
            correct_ids = failure["correct_ids"]
            if len(correct_ids) < 2:
                continue
            expected_id = int(correct_ids[1])
            forced_prev = [int(correct_ids[0])]
            task_ids = tokenizer.encode(failure["task_prompt"], add_special_tokens=False) + forced_prev
            value_ids = tokenizer.encode(failure["value_prompt"], add_special_tokens=False) + forced_prev
            pos = len(task_ids) - 1
            mismatch = pick_mismatch(failures, i)
            mismatch_ids = tokenizer.encode(mismatch["value_prompt"], add_special_tokens=False) + forced_prev
            source_patches, _stats = compute_source_patches(model, tokenizer, device, info, failure, site_specs)
            combo = failure["combo"]
            last_writers = failure["last_writers"]
            baseline_logits = logits_with_hooks(
                model, tokenizer, device, task_ids, failure["task_prompt"], source_patches, combo, last_writers
            )
            baseline = token_top_metric(tokenizer, baseline_logits, expected_id, args.top_k)

            for spec in ensembles:
                for intervention in INTERVENTIONS:
                    if spec["kind"] == "component_set":
                        patch_items = []
                        for layer, comp in spec["components"]:
                            correct_cache = collect_id_components(model, device, value_ids, pos, [layer], [comp])
                            task_cache = collect_id_components(model, device, task_ids, pos, [layer], [comp])
                            mismatch_cache = collect_id_components(model, device, mismatch_ids, pos, [layer], [comp])
                            task_vec = task_cache.get(layer, {}).get(comp)
                            correct_vec = correct_cache.get(layer, {}).get(comp)
                            mismatch_vec = mismatch_cache.get(layer, {}).get(comp)
                            if task_vec is None or correct_vec is None or mismatch_vec is None:
                                continue
                            target = {
                                "zero_remove": torch.zeros_like(task_vec),
                                "mismatch_restore": mismatch_vec,
                                "correct_restore": correct_vec,
                            }[intervention]
                            patch_items.append((layer, comp, target))
                        if not patch_items:
                            continue
                        logits = component_patch_logits(
                            model,
                            tokenizer,
                            device,
                            task_ids,
                            failure["task_prompt"],
                            source_patches,
                            combo,
                            last_writers,
                            patch_items,
                        )
                    else:
                        layer = int(spec["layer"])
                        correct_o = collect_o_input(model, device, value_ids, pos, layer)
                        task_o = collect_o_input(model, device, task_ids, pos, layer)
                        mismatch_o = collect_o_input(model, device, mismatch_ids, pos, layer)
                        if correct_o is None or task_o is None or mismatch_o is None:
                            continue
                        target = {
                            "zero_remove": torch.zeros_like(task_o),
                            "mismatch_restore": mismatch_o,
                            "correct_restore": correct_o,
                        }[intervention]
                        logits = head_patch_logits(
                            model,
                            tokenizer,
                            device,
                            task_ids,
                            failure["task_prompt"],
                            source_patches,
                            combo,
                            last_writers,
                            layer,
                            list(spec["heads"]),
                            target,
                        )
                    met = token_top_metric(tokenizer, logits, expected_id, args.top_k)
                    rows.append({
                        "sample_idx": failure["sample_idx"],
                        "pair_task": failure["pair_task"],
                        "site": failure["site"],
                        "combo_name": failure["combo_name"],
                        "ensemble": spec["name"],
                        "kind": spec["kind"],
                        "intervention": intervention,
                        "expected_rank": met["expected_rank"],
                        "expected_minus_top1": met["expected_minus_top1"],
                        "baseline_expected_rank": baseline["expected_rank"],
                        "baseline_expected_minus_top1": baseline["expected_minus_top1"],
                        "top1_text": met["top1"]["text"],
                    })
        summary = summarize(rows)
        log("Top ensemble specificity:")
        for r in summary["ensemble_specificity"][:20]:
            log(
                f"  {r['kind']} {r['pair_task']} {r['site']} {r['combo_name']} {r['ensemble']} "
                f"correct_minus_mismatch={r['correct_minus_mismatch_margin_delta']:.3f} "
                f"correct_top1={r['correct_top1_rate']:.2f} mismatch_top1={r['mismatch_top1_rate']:.2f}"
            )
        return {
            "phase": 668,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_phase665": str(data_path),
            "n_layers": info.n_layers,
            "ensembles": ensembles,
            "top_k": args.top_k,
            "n_failures_tested": len(failures),
            "n_rows": len(rows),
            "summary": summary,
            "rows": rows if args.save_rows else rows[: args.example_limit],
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--max-failures", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--save-rows", action="store_true")
    parser.add_argument("--example-limit", type=int, default=240)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_failures = min(args.max_failures, 1)
        args.top_k = min(args.top_k, 20)
        log("SMOKE TEST MODE")
    if args.confirm:
        args.max_failures = max(args.max_failures, 12)
        args.top_k = max(args.top_k, 30)
        args.example_limit = max(args.example_limit, 320)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase668_{args.model}_token1_transition_writer_ensemble_closure_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
