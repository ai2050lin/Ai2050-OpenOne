#!/usr/bin/env python3
"""
Phase 667: Value-Specific Token1 Transition Writer Localization.

Phase 666 separated general continuation state from value-specific token1
transition state. This phase asks whether that state is written by a localized
component/head or by a distributed mixture.

It reuses Phase 665 failures and the Phase 666 controls:
  baseline / zero_remove / mismatch_restore / correct_restore

Writer candidates are deliberately narrow:
  qwen3:      L22 attn/mlp/layer_out + L22 attention head o_proj slices
  GLM4:       L21 layer_out, L22 attn/mlp/layer_out + L22 head slices
  DS7B:       L21 attn/mlp/layer_out + L21 head slices
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
from phase665_autoregressive_continuation_controller_localization import collect_id_components, token_top_metric  # noqa: E402
from phase666_token1_transition_boundary_remove_restore import (  # noqa: E402
    INTERVENTIONS,
    compute_source_patches,
    logits_with_hooks,
    phase665_path,
    pick_mismatch,
)


OUT_ROOT = Path("results/glm5_phase667_value_specific_token1_transition_writer_localization")
COMPONENT_WRITERS = {
    "qwen3": [
        {"layer": 22, "component": "attn_out", "label": "L22_attn_out"},
        {"layer": 22, "component": "mlp_out", "label": "L22_mlp_out"},
        {"layer": 22, "component": "layer_out", "label": "L22_layer_out"},
        {"layer": 23, "component": "layer_input", "label": "L23_layer_input"},
    ],
    "glm4": [
        {"layer": 21, "component": "layer_out", "label": "L21_layer_out"},
        {"layer": 22, "component": "attn_out", "label": "L22_attn_out"},
        {"layer": 22, "component": "mlp_out", "label": "L22_mlp_out"},
        {"layer": 22, "component": "layer_out", "label": "L22_layer_out"},
    ],
    "deepseek7b": [
        {"layer": 21, "component": "attn_out", "label": "L21_attn_out"},
        {"layer": 21, "component": "mlp_out", "label": "L21_mlp_out"},
        {"layer": 21, "component": "layer_out", "label": "L21_layer_out"},
        {"layer": 22, "component": "layer_input", "label": "L22_layer_input"},
    ],
}
HEAD_LAYERS = {"qwen3": [22], "glm4": [22], "deepseek7b": [21]}
HEAD_INTERVENTIONS = ["zero_remove", "mismatch_restore", "correct_restore"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_o_input(model, device, ids: List[int], pos: int, layer_idx: int) -> torch.Tensor | None:
    layers = get_layers(model)
    if not (0 <= layer_idx < len(layers)):
        return None
    attn = get_attn(layers[layer_idx])
    if attn is None:
        return None
    o_proj = get_o_proj(attn)
    captured = {}

    def hook(_module, inputs):
        x = inputs[0]
        if 0 <= pos < x.shape[1]:
            captured["o_input"] = x[0, pos].detach().float().cpu()

    handle = o_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True)
    finally:
        handle.remove()
    return captured.get("o_input")


def logits_with_head_patch(
    model,
    tokenizer,
    device,
    ids: List[int],
    prompt: str,
    source_patches,
    combo,
    last_writers,
    layer_idx: int,
    head_idx: int,
    n_heads: int,
    target_vec: torch.Tensor,
) -> torch.Tensor:
    layers = get_layers(model)
    attn = get_attn(layers[layer_idx])
    o_proj = get_o_proj(attn)
    handles = []
    head_dim = target_vec.numel() // max(1, n_heads)
    start = head_idx * head_dim
    end = target_vec.numel() if head_idx == n_heads - 1 else (head_idx + 1) * head_dim

    def hook(_module, inputs):
        x = inputs[0]
        y = x.clone()
        pos = len(ids) - 1
        if 0 <= pos < y.shape[1]:
            y[0, pos, start:end] = target_vec[start:end].to(device=y.device, dtype=y.dtype)
        return (y,) + tuple(inputs[1:])

    try:
        handles.append(o_proj.register_forward_pre_hook(hook))
        return logits_with_hooks(model, tokenizer, device, ids, prompt, source_patches, combo, last_writers)
    finally:
        for h in handles:
            h.remove()


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: List[Dict]) -> Dict:
    groups: Dict[Tuple, Dict] = {}
    for row in rows:
        key = (row["kind"], row["pair_task"], row["site"], row["combo_name"], row["writer"], row["intervention"])
        g = groups.setdefault(key, {
            "kind": row["kind"],
            "pair_task": row["pair_task"],
            "site": row["site"],
            "combo_name": row["combo_name"],
            "writer": row["writer"],
            "layer": row["layer"],
            "component": row.get("component"),
            "head": row.get("head"),
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

    intervention_out = []
    for g in groups.values():
        r = dict(g)
        r["expected_top1_rate"] = g["top1"] / max(1, g["n"])
        r["mean_rank_delta_vs_baseline"] = mean(g["rank_delta"])
        r["mean_margin_delta_vs_baseline"] = mean(g["margin_delta"])
        r["top1_text"] = dict(sorted(g["top1_text"].items(), key=lambda kv: kv[1], reverse=True)[:8])
        for k in ["top1", "rank_delta", "margin_delta"]:
            r.pop(k, None)
        intervention_out.append(r)

    writer_map: Dict[Tuple, Dict] = {}
    for item in intervention_out:
        key = (item["kind"], item["pair_task"], item["site"], item["combo_name"], item["writer"])
        w = writer_map.setdefault(key, {
            "kind": item["kind"],
            "pair_task": item["pair_task"],
            "site": item["site"],
            "combo_name": item["combo_name"],
            "writer": item["writer"],
            "layer": item["layer"],
            "component": item.get("component"),
            "head": item.get("head"),
            "n": item["n"],
            "interventions": {},
        })
        w["interventions"][item["intervention"]] = item

    writer_out = []
    for w in writer_map.values():
        correct = w["interventions"].get("correct_restore", {})
        mismatch = w["interventions"].get("mismatch_restore", {})
        zero = w["interventions"].get("zero_remove", {})
        w["correct_minus_mismatch_margin_delta"] = (
            correct.get("mean_margin_delta_vs_baseline", 0.0) - mismatch.get("mean_margin_delta_vs_baseline", 0.0)
        )
        w["correct_minus_zero_margin_delta"] = (
            correct.get("mean_margin_delta_vs_baseline", 0.0) - zero.get("mean_margin_delta_vs_baseline", 0.0)
        )
        w["correct_top1_rate"] = correct.get("expected_top1_rate", 0.0)
        w["mismatch_top1_rate"] = mismatch.get("expected_top1_rate", 0.0)
        writer_out.append(w)
    writer_out.sort(
        key=lambda r: (
            -r["correct_minus_mismatch_margin_delta"],
            -r["correct_top1_rate"],
            r["mismatch_top1_rate"],
            r["kind"],
            r["writer"],
        )
    )
    intervention_out.sort(key=lambda r: (r["kind"], r["pair_task"], r["site"], r["combo_name"], r["writer"], r["intervention"]))
    return {"writer_specificity": writer_out, "intervention_summary": intervention_out}


def run_model(args) -> Dict:
    data_path = phase665_path(args.model)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing Phase 665 confirm result: {data_path}")
    phase665 = json.loads(data_path.read_text(encoding="utf-8"))
    failures = phase665["selected_failures"][: args.max_failures]
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        component_writers = [w for w in COMPONENT_WRITERS[args.model] if 0 <= w["layer"] < info.n_layers]
        head_layers = [li for li in HEAD_LAYERS[args.model] if 0 <= li < info.n_layers]
        rows = []
        log(f"{args.model}: failures={len(failures)}, component_writers={component_writers}, head_layers={head_layers}")
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
            source_patches, _stats = compute_source_patches(
                model,
                tokenizer,
                device,
                info,
                failure,
                {s["name"]: s for s in __import__("phase656_format_prior_writer_localization_audit").SITE_SPECS[args.model]},
            )
            combo = failure["combo"]
            last_writers = failure["last_writers"]
            baseline_logits = logits_with_hooks(
                model, tokenizer, device, task_ids, failure["task_prompt"], source_patches, combo, last_writers
            )
            baseline = token_top_metric(tokenizer, baseline_logits, expected_id, args.top_k)

            for writer in component_writers:
                layer = int(writer["layer"])
                comp = writer["component"]
                correct_cache = collect_id_components(model, device, value_ids, pos, [layer], [comp])
                task_cache = collect_id_components(model, device, task_ids, pos, [layer], [comp])
                mismatch_cache = collect_id_components(model, device, mismatch_ids, pos, [layer], [comp])
                correct_vec = correct_cache.get(layer, {}).get(comp)
                task_vec = task_cache.get(layer, {}).get(comp)
                mismatch_vec = mismatch_cache.get(layer, {}).get(comp)
                if correct_vec is None or task_vec is None or mismatch_vec is None:
                    continue
                vectors = {
                    "zero_remove": torch.zeros_like(task_vec),
                    "mismatch_restore": mismatch_vec,
                    "correct_restore": correct_vec,
                }
                for intervention, target in vectors.items():
                    logits = logits_with_hooks(
                        model,
                        tokenizer,
                        device,
                        task_ids,
                        failure["task_prompt"],
                        source_patches,
                        combo,
                        last_writers,
                        [(layer, comp, [pos], [target])],
                    )
                    met = token_top_metric(tokenizer, logits, expected_id, args.top_k)
                    rows.append({
                        "kind": "component",
                        "sample_idx": failure["sample_idx"],
                        "pair_task": failure["pair_task"],
                        "site": failure["site"],
                        "combo_name": failure["combo_name"],
                        "writer": writer["label"],
                        "layer": layer,
                        "component": comp,
                        "intervention": intervention,
                        "expected_rank": met["expected_rank"],
                        "expected_minus_top1": met["expected_minus_top1"],
                        "baseline_expected_rank": baseline["expected_rank"],
                        "baseline_expected_minus_top1": baseline["expected_minus_top1"],
                        "top1_text": met["top1"]["text"],
                    })

            for layer in head_layers:
                attn = get_attn(get_layers(model)[layer])
                if attn is None:
                    continue
                n_heads = get_num_heads(model, attn)
                if args.max_heads > 0:
                    head_ids = list(range(min(args.max_heads, n_heads)))
                else:
                    head_ids = list(range(n_heads))
                correct_o = collect_o_input(model, device, value_ids, pos, layer)
                task_o = collect_o_input(model, device, task_ids, pos, layer)
                mismatch_o = collect_o_input(model, device, mismatch_ids, pos, layer)
                if correct_o is None or task_o is None or mismatch_o is None:
                    continue
                vectors = {
                    "zero_remove": torch.zeros_like(task_o),
                    "mismatch_restore": mismatch_o,
                    "correct_restore": correct_o,
                }
                for head in head_ids:
                    for intervention, target in vectors.items():
                        logits = logits_with_head_patch(
                            model,
                            tokenizer,
                            device,
                            task_ids,
                            failure["task_prompt"],
                            source_patches,
                            combo,
                            last_writers,
                            layer,
                            head,
                            n_heads,
                            target,
                        )
                        met = token_top_metric(tokenizer, logits, expected_id, args.top_k)
                        rows.append({
                            "kind": "head_o_input",
                            "sample_idx": failure["sample_idx"],
                            "pair_task": failure["pair_task"],
                            "site": failure["site"],
                            "combo_name": failure["combo_name"],
                            "writer": f"L{layer}_head{head}_o_input",
                            "layer": layer,
                            "head": head,
                            "intervention": intervention,
                            "expected_rank": met["expected_rank"],
                            "expected_minus_top1": met["expected_minus_top1"],
                            "baseline_expected_rank": baseline["expected_rank"],
                            "baseline_expected_minus_top1": baseline["expected_minus_top1"],
                            "top1_text": met["top1"]["text"],
                        })
        summary = summarize(rows)
        log("Top writer specificity:")
        for r in summary["writer_specificity"][:25]:
            log(
                f"  {r['kind']} {r['pair_task']} {r['site']} {r['combo_name']} {r['writer']} "
                f"correct_minus_mismatch={r['correct_minus_mismatch_margin_delta']:.3f} "
                f"correct_top1={r['correct_top1_rate']:.2f} mismatch_top1={r['mismatch_top1_rate']:.2f}"
            )
        return {
            "phase": 667,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_phase665": str(data_path),
            "n_layers": info.n_layers,
            "component_writers": component_writers,
            "head_layers": head_layers,
            "max_heads": args.max_heads,
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
    parser.add_argument("--max-heads", type=int, default=0, help="0 means all heads")
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
        args.max_heads = 4
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
    out_path = out_dir / f"phase667_{args.model}_value_specific_token1_transition_writer_localization_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
