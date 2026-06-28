#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
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

from model_utils import get_layers, load_model, release_model  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase724_fruit_route_channel_group_drilldown import install_head_channel_ablation  # noqa: E402
from phase727_category_fruit_cluster_intervention import build_interventions  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402


OUT_ROOT = Path("results/glm5_phase729_full_head_vs_cluster_residual_propagation")
MODELS = ["qwen3", "glm4", "deepseek7b"]
INTERVENTIONS = ["category_cluster", "category_full_head", "category_plus_fruit_cluster"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def category_cases(max_cases: int | None = None) -> list[dict[str, Any]]:
    rows = [c for c in build_cases(None) if c["relation"] == "category"]
    return rows[:max_cases] if max_cases else rows


def norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec.float()).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa = a.float()
    bb = b.float()
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(denom.item()) <= 1e-9:
        return None
    return float(torch.dot(aa.flatten(), bb.flatten()).div(denom).item())


def get_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    return output


def install_multi_ablation(model, ranges: list[dict[str, int]]):
    handles = []
    for r in ranges:
        hs, _head_dim = install_head_channel_ablation(
            model,
            int(r["layer"]),
            int(r["head"]),
            int(r["start"]),
            int(r["end"]),
        )
        handles.extend(hs)
    return handles


def capture_forward(
    model,
    tokenizer,
    device,
    prompt: str,
    ranges: list[dict[str, int]],
    capture_layers: list[int],
) -> dict[str, torch.Tensor]:
    layers = get_layers(model)
    captured: dict[str, torch.Tensor] = {}
    handles = install_multi_ablation(model, ranges) if ranges else []

    def make_layer_input_hook(li: int):
        def hook(_module, inputs):
            captured[f"L{li}_layer_input"] = inputs[0][0, -1].detach().float().cpu()
        return hook

    def make_component_hook(li: int, component: str):
        def hook(_module, _inputs, output):
            captured[f"L{li}_{component}"] = get_tensor(output)[0, -1].detach().float().cpu()
        return hook

    try:
        for li in capture_layers:
            if 0 <= li < len(layers):
                handles.append(layers[li].register_forward_pre_hook(make_layer_input_hook(li)))
                if hasattr(layers[li], "self_attn"):
                    handles.append(layers[li].self_attn.register_forward_hook(make_component_hook(li, "attn_out")))
                if hasattr(layers[li], "mlp"):
                    handles.append(layers[li].mlp.register_forward_hook(make_component_hook(li, "mlp_out")))

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_hidden_states=True,
            )

        for hs_idx, hs in enumerate(out.hidden_states):
            # hidden_states[0] is embedding output; hidden_states[k+1] is after layer k.
            captured[f"hidden_{hs_idx}"] = hs[0, -1].detach().float().cpu()
        captured["final_hidden"] = out.hidden_states[-1][0, -1].detach().float().cpu()
        captured["final_logits"] = out.logits[0, -1].detach().float().cpu()
        return captured
    finally:
        for h in handles:
            h.remove()


def site_records(
    case: dict[str, Any],
    intervention: str,
    base: dict[str, torch.Tensor],
    patched: dict[str, torch.Tensor],
    source_hidden_idx: int,
    monitor_hidden_idxs: list[int],
    component_layers: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_key = f"hidden_{source_hidden_idx}"
    final_delta = patched["final_hidden"] - base["final_hidden"]
    source_delta = patched[source_key] - base[source_key]
    source_norm = max(norm(source_delta), 1e-9)
    final_norm = norm(final_delta)

    for hs_idx in monitor_hidden_idxs:
        key = f"hidden_{hs_idx}"
        if key not in patched or key not in base:
            continue
        delta = patched[key] - base[key]
        delta_norm = norm(delta)
        rows.append(
            {
                "case_id": case["case_id"],
                "object": case["object"],
                "object_group": case["object_group"],
                "prompt_type": case["prompt_type"],
                "answer": case["answer"],
                "intervention": intervention,
                "site": f"hidden_{hs_idx}",
                "site_kind": "layer_out" if hs_idx > 0 else "embedding",
                "layer": hs_idx - 1,
                "delta_norm": delta_norm,
                "source_delta_norm": source_norm,
                "amplification_vs_source": delta_norm / source_norm,
                "final_delta_norm": final_norm,
                "cos_with_final_delta": cosine(delta, final_delta),
            }
        )

    for li in component_layers:
        input_key = f"L{li}_layer_input"
        input_delta = patched.get(input_key, torch.zeros_like(final_delta)) - base.get(input_key, torch.zeros_like(final_delta))
        input_norm = max(norm(input_delta), 1e-9)
        for component in ["attn_out", "mlp_out"]:
            key = f"L{li}_{component}"
            if key not in patched or key not in base:
                continue
            delta = patched[key] - base[key]
            delta_norm = norm(delta)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "object": case["object"],
                    "object_group": case["object_group"],
                    "prompt_type": case["prompt_type"],
                    "answer": case["answer"],
                    "intervention": intervention,
                    "site": key,
                    "site_kind": component,
                    "layer": li,
                    "delta_norm": delta_norm,
                    "source_delta_norm": source_norm,
                    "input_delta_norm": input_norm,
                    "amplification_vs_source": delta_norm / source_norm,
                    "component_vs_layer_input": delta_norm / input_norm,
                    "final_delta_norm": final_norm,
                    "cos_with_final_delta": cosine(delta, final_delta),
                }
            )
    return rows


def mean(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows: list[dict[str, Any]], model_name: str, interventions: dict[str, list[dict[str, int]]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["intervention"], row["site"])].append(row)

    site_summary = []
    for (intervention, site), vals in sorted(by_key.items()):
        site_summary.append(
            {
                "model": model_name,
                "intervention": intervention,
                "site": site,
                "site_kind": vals[0]["site_kind"],
                "layer": vals[0]["layer"],
                "n": len(vals),
                "mean_delta_norm": mean([v["delta_norm"] for v in vals]),
                "mean_amplification_vs_source": mean([v["amplification_vs_source"] for v in vals]),
                "mean_final_delta_norm": mean([v["final_delta_norm"] for v in vals]),
                "mean_cos_with_final_delta": mean([v["cos_with_final_delta"] for v in vals if v["cos_with_final_delta"] is not None]),
                "mean_component_vs_layer_input": mean([v.get("component_vs_layer_input") for v in vals if v.get("component_vs_layer_input") is not None]),
            }
        )

    by_intervention = {}
    for intervention in sorted({r["intervention"] for r in rows}):
        vals = [r for r in site_summary if r["intervention"] == intervention]
        layer_out = [r for r in vals if r["site_kind"] == "layer_out"]
        mlp = [r for r in vals if r["site_kind"] == "mlp_out"]
        attn = [r for r in vals if r["site_kind"] == "attn_out"]
        by_intervention[intervention] = {
            "n_sites": len(vals),
            "max_layer_out_amplification": max([r["mean_amplification_vs_source"] for r in layer_out], default=0.0),
            "final_layer_out_amplification": next((r["mean_amplification_vs_source"] for r in layer_out if r["site"] == "hidden_final"), None),
            "mean_mlp_component_vs_input": mean([r["mean_component_vs_layer_input"] for r in mlp]),
            "mean_attn_component_vs_input": mean([r["mean_component_vs_layer_input"] for r in attn]),
            "top_sites_by_delta": sorted(vals, key=lambda r: r["mean_delta_norm"], reverse=True)[:10],
            "top_sites_by_amplification": sorted(vals, key=lambda r: r["mean_amplification_vs_source"], reverse=True)[:10],
        }

    return {
        "phase": 729,
        "title": "Full-Head vs Channel-Cluster Residual Propagation",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "interventions": {k: interventions[k] for k in INTERVENTIONS if k in interventions},
        "site_summary": site_summary,
        "by_intervention": by_intervention,
    }


def run_model(args) -> dict[str, Any]:
    cases = category_cases(args.max_cases)
    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        layers = get_layers(model)
        interventions = build_interventions(model, args.model)
        source_layer = int(interventions["category_full_head"][0]["layer"])
        source_hidden_idx = source_layer + 1
        max_hidden_idx = min(len(layers), source_layer + args.downstream_layers + 1)
        monitor_hidden_idxs = list(range(source_hidden_idx, max_hidden_idx + 1))
        # Add the final hidden state as a stable readout endpoint.
        if len(layers) not in monitor_hidden_idxs:
            monitor_hidden_idxs.append(len(layers))
        component_layers = [li for li in range(source_layer + 1, min(source_layer + args.component_layers + 1, len(layers)))]
        capture_layers = sorted(set([source_layer] + component_layers))
        log(
            f"{args.model}: cases={len(cases)}, source=L{source_layer}, "
            f"monitor_hidden={monitor_hidden_idxs}, component_layers={component_layers}"
        )
        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            base = capture_forward(model, tokenizer, device, prompt, [], capture_layers)
            for intervention in INTERVENTIONS:
                patched = capture_forward(model, tokenizer, device, prompt, interventions[intervention], capture_layers)
                rows.extend(
                    site_records(
                        case,
                        intervention,
                        base,
                        patched,
                        source_hidden_idx,
                        monitor_hidden_idxs,
                        component_layers,
                    )
                )
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: {idx}/{len(cases)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows, args.model, interventions)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase729_{args.model}_propagation_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase729_{args.model}_propagation_summary.json", summary)
    print(json.dumps({"model": args.model, "n_cases": summary["n_cases"], "by_intervention": summary["by_intervention"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase729_{model}_propagation_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 729,
        "title": "Full-Head vs Channel-Cluster Residual Propagation",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "residual and component propagation under cluster/full-head ablation",
        "by_model": {
            s["model"]: {
                "n_cases": s["n_cases"],
                "interventions": s["interventions"],
                "by_intervention": s["by_intervention"],
                "site_summary": s["site_summary"],
            }
            for s in summaries
        },
    }
    write_json(OUT_ROOT / "phase729_cross_model_summary.json", payload)
    lines = [
        "# Phase 729 Full-Head vs Channel-Cluster Residual Propagation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: residual/component propagation.",
        "",
        "| model | intervention | max layer amp | mean MLP/input | mean attn/input | top site | top delta |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for model_name, item in payload["by_model"].items():
        for intervention, rec in item["by_intervention"].items():
            top = rec["top_sites_by_delta"][0] if rec["top_sites_by_delta"] else {}
            lines.append(
                f"| {model_name} | {intervention} | {rec['max_layer_out_amplification']:.3f} | "
                f"{rec['mean_mlp_component_vs_input']:.3f} | {rec['mean_attn_component_vs_input']:.3f} | "
                f"{top.get('site', '-')} | {top.get('mean_delta_norm', 0):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- This is a propagation measurement, not a generation closure test.",
            "- A large full-head residual trajectory with weak cluster trajectory supports the Phase 727 boundary.",
            "- MLP/input and attention/input ratios are diagnostic ratios, not a proof of module-level causality.",
            "",
        ]
    )
    (OUT_ROOT / "phase729_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"]}, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--downstream-layers", type=int, default=8)
    parser.add_argument("--component-layers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=8)
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
        os._exit(0)


if __name__ == "__main__":
    main()
