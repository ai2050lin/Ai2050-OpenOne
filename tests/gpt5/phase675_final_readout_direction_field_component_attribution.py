#!/usr/bin/env python3
"""
Phase 675: Final Readout Direction Field Component Attribution.

Uses the Phase 674 same_format_random_value rows and measures where the
competitor-vs-expected readout gap changes across late-layer natural trajectory
components. No patching is used.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn, get_final_norm, get_mlp  # noqa: E402


PHASE674_ROOT = Path("results/glm5_phase674_synthetic_value_readout_competitor_source_localization")
OUT_ROOT = Path("results/glm5_phase675_final_readout_direction_field_component_attribution")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_phase674_rows(model_name: str, max_cases: int) -> list[dict]:
    path = PHASE674_ROOT / f"phase674_{model_name}_synthetic_value_readout_source_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_cases] if max_cases > 0 else rows


def lm_gap(model, state: torch.Tensor | None, expected_id: int, competitor_id: int) -> float | None:
    if state is None:
        return None
    emb = model.get_output_embeddings()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        logits = emb(state.to(device=device, dtype=dtype).unsqueeze(0)).squeeze(0).float().cpu()
    return float(logits[competitor_id].item() - logits[expected_id].item())


def capture_late_trajectory(model, tokenizer, device, prompt: str, layer_indices: list[int]) -> dict:
    layers = get_layers(model)
    final_norm = get_final_norm(model)
    captured: dict[str, Any] = {"layers": {}}
    handles = []

    for li in layer_indices:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)
        captured["layers"][li] = {}

        def layer_pre(_module, inputs, layer_idx=li):
            captured["layers"][layer_idx]["layer_input"] = inputs[0].detach().float().cpu()

        def layer_out(_module, _inputs, output, layer_idx=li):
            captured["layers"][layer_idx]["layer_out"] = extract_tensor(output).detach().float().cpu()

        handles.append(layer.register_forward_pre_hook(layer_pre))
        handles.append(layer.register_forward_hook(layer_out))

        if attn is not None:
            def attn_out(_module, _inputs, output, layer_idx=li):
                captured["layers"][layer_idx]["attn_out"] = extract_tensor(output).detach().float().cpu()

            handles.append(attn.register_forward_hook(attn_out))

        if mlp is not None:
            def mlp_pre(_module, inputs, layer_idx=li):
                captured["layers"][layer_idx]["mlp_input"] = inputs[0].detach().float().cpu()

            def mlp_out(_module, _inputs, output, layer_idx=li):
                captured["layers"][layer_idx]["mlp_out"] = extract_tensor(output).detach().float().cpu()

            handles.append(mlp.register_forward_pre_hook(mlp_pre))
            handles.append(mlp.register_forward_hook(mlp_out))

    if final_norm is not None:
        def norm_pre(_module, inputs):
            captured["final_norm_input"] = inputs[0].detach().float().cpu()

        def norm_out(_module, _inputs, output):
            captured["final_norm_output"] = extract_tensor(output).detach().float().cpu()

        handles.append(final_norm.register_forward_pre_hook(norm_pre))
        handles.append(final_norm.register_forward_hook(norm_out))

    try:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, return_dict=True)
        captured["logits"] = out.logits.detach().float().cpu()
        captured["pos"] = len(ids) - 1
    finally:
        for handle in handles:
            handle.remove()
    return captured


def at_pos(tensor: torch.Tensor | None, pos: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 3:
        if pos >= tensor.shape[1]:
            return None
        return tensor[0, pos]
    if tensor.ndim == 2:
        if pos >= tensor.shape[0]:
            return None
        return tensor[pos]
    return tensor


def case_prompt(row: dict) -> str:
    control = json.loads(
        Path("results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json")
        .read_text(encoding="utf-8")
    )
    by_id = {case["case_id"]: case["prompt"] for case in control["cases"]}
    return by_id[row["case_id"]]


def summarize(rows: list[dict]) -> dict:
    component_groups: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "sum_delta": 0.0,
        "positive": 0,
        "sum_before": 0.0,
        "sum_after": 0.0,
    })
    final_groups: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "sum_gap": 0.0,
        "expected_top1": 0,
        "top1_category": {},
    })
    for row in rows:
        for group in ["overall", row["top1_category"], row["relation"]]:
            fg = final_groups[group]
            fg["n"] += 1
            fg["sum_gap"] += row["final_gap"]
            fg["expected_top1"] += int(row["expected_rank"] == 1)
            cat = row["top1_category"]
            fg["top1_category"][cat] = fg["top1_category"].get(cat, 0) + 1
        for item in row["component_deltas"]:
            for group in ["overall", row["top1_category"], row["relation"]]:
                key = f"{group}|{item['component']}"
                cg = component_groups[key]
                cg["n"] += 1
                cg["sum_delta"] += item["delta_gap"]
                cg["positive"] += int(item["delta_gap"] > 0)
                cg["sum_before"] += item["before_gap"]
                cg["sum_after"] += item["after_gap"]

    final = {}
    for key, item in final_groups.items():
        n = max(1, item["n"])
        final[key] = {
            "n": item["n"],
            "mean_final_gap": item["sum_gap"] / n,
            "expected_top1_rate": item["expected_top1"] / n,
            "top1_category": dict(sorted(item["top1_category"].items(), key=lambda kv: kv[1], reverse=True)),
        }

    components = []
    for key, item in component_groups.items():
        n = max(1, item["n"])
        group, component = key.split("|", 1)
        components.append({
            "group": group,
            "component": component,
            "n": item["n"],
            "mean_delta_gap": item["sum_delta"] / n,
            "positive_rate": item["positive"] / n,
            "mean_before_gap": item["sum_before"] / n,
            "mean_after_gap": item["sum_after"] / n,
        })
    components.sort(key=lambda x: (x["group"] != "overall", -x["mean_delta_gap"], -x["positive_rate"]))
    return {"final": final, "components": components}


def run_model(args) -> dict:
    phase674_rows = load_phase674_rows(args.model, args.max_cases)
    prompt_by_case = json.loads(
        Path("results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json")
        .read_text(encoding="utf-8")
    )
    prompt_by_case = {case["case_id"]: case["prompt"] for case in prompt_by_case["cases"]}

    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        n_layers = len(get_layers(model))
        start = max(0, n_layers - args.last_layers)
        layer_indices = list(range(start, n_layers))
        for i, row674 in enumerate(phase674_rows):
            prompt = prompt_by_case[row674["case_id"]]
            expected_id = int(row674["expected_id"])
            competitor_id = int(row674["competitor"]["id"])
            captured = capture_late_trajectory(model, tokenizer, device, prompt, layer_indices)
            pos = captured["pos"]
            component_deltas = []
            last_layer_out_gap = None
            for li in layer_indices:
                lc = captured["layers"][li]
                layer_input = at_pos(lc.get("layer_input"), pos)
                mlp_input = at_pos(lc.get("mlp_input"), pos)
                layer_out = at_pos(lc.get("layer_out"), pos)
                attn_out = at_pos(lc.get("attn_out"), pos)
                mlp_out = at_pos(lc.get("mlp_out"), pos)
                g_layer_in = lm_gap(model, layer_input, expected_id, competitor_id)
                g_mlp_in = lm_gap(model, mlp_input, expected_id, competitor_id)
                g_layer_out = lm_gap(model, layer_out, expected_id, competitor_id)
                g_attn_out = lm_gap(model, attn_out, expected_id, competitor_id)
                g_mlp_out = lm_gap(model, mlp_out, expected_id, competitor_id)
                if g_layer_in is not None and g_mlp_in is not None:
                    component_deltas.append({
                        "layer": li,
                        "component": f"L{li}.attn_plus_residual",
                        "before_gap": g_layer_in,
                        "after_gap": g_mlp_in,
                        "delta_gap": g_mlp_in - g_layer_in,
                        "component_direct_gap": g_attn_out,
                    })
                if g_mlp_in is not None and g_layer_out is not None:
                    component_deltas.append({
                        "layer": li,
                        "component": f"L{li}.mlp_plus_residual",
                        "before_gap": g_mlp_in,
                        "after_gap": g_layer_out,
                        "delta_gap": g_layer_out - g_mlp_in,
                        "component_direct_gap": g_mlp_out,
                    })
                if li == layer_indices[-1]:
                    last_layer_out_gap = g_layer_out

            final_in = at_pos(captured.get("final_norm_input"), pos)
            final_out = at_pos(captured.get("final_norm_output"), pos)
            g_final_in = lm_gap(model, final_in, expected_id, competitor_id)
            g_final_out = lm_gap(model, final_out, expected_id, competitor_id)
            if g_final_in is not None and g_final_out is not None:
                component_deltas.append({
                    "layer": n_layers,
                    "component": "final_norm",
                    "before_gap": g_final_in,
                    "after_gap": g_final_out,
                    "delta_gap": g_final_out - g_final_in,
                    "component_direct_gap": None,
                })
            if last_layer_out_gap is not None and g_final_in is not None:
                component_deltas.append({
                    "layer": n_layers,
                    "component": "post_last_to_final_norm_input",
                    "before_gap": last_layer_out_gap,
                    "after_gap": g_final_in,
                    "delta_gap": g_final_in - last_layer_out_gap,
                    "component_direct_gap": None,
                })

            logits = captured["logits"][0, pos].float()
            final_gap = float(logits[competitor_id].item() - logits[expected_id].item())
            rows.append({
                "case_id": row674["case_id"],
                "relation": row674["relation"],
                "top1_category": row674["top1_category"],
                "expected_rank": row674["expected_rank"],
                "expected_id": expected_id,
                "competitor_id": competitor_id,
                "competitor_text": row674["competitor"]["text"],
                "final_gap": final_gap,
                "final_norm_proxy_gap": g_final_out,
                "component_deltas": component_deltas,
            })
            if (i + 1) % 12 == 0 or i + 1 == len(phase674_rows):
                log(f"{args.model}: {i + 1}/{len(phase674_rows)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase675_{args.model}_component_attribution_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    result = {
        "phase": 675,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(rows),
        "last_layers": args.last_layers,
        "summary": summary,
    }
    out_path = OUT_ROOT / f"phase675_{args.model}_component_attribution_summary.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    log(f"Wrote {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def write_cross_summary() -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase675_*_component_attribution_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    result = {
        "phase": 675,
        "title": "Final Readout Direction Field Component Attribution Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase675_cross_model_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 675 Final Readout Direction Field Component Attribution",
        "",
        f"- generated: `{result['timestamp']}`",
        "",
        "| model | cases | top1_rate | final_gap | strongest overall delta | positive_rate |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for item in models:
        overall = item["summary"]["final"].get("overall", {})
        comps = [c for c in item["summary"]["components"] if c["group"] == "overall"]
        strongest = comps[0] if comps else {"component": "NA", "mean_delta_gap": 0.0, "positive_rate": 0.0}
        lines.append(
            f"| {item['model']} | {item['n_cases']} | "
            f"{overall.get('expected_top1_rate', 0.0):.3f} | "
            f"{overall.get('mean_final_gap', 0.0):.3f} | "
            f"{strongest['component']} ({strongest['mean_delta_gap']:.3f}) | "
            f"{strongest['positive_rate']:.3f} |"
        )
    (OUT_ROOT / "phase675_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--max-cases", type=int, default=72)
    parser.add_argument("--last-layers", type=int, default=8)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
