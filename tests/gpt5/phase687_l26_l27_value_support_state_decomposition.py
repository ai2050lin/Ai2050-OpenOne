#!/usr/bin/env python3
"""
Phase 687: L26/L27 Value-Support State Decomposition.

Phase 686 showed that same-case L26/L27 layer_out restore completely repairs
DS7B short_only paired failures. This phase decomposes that effective state:

1. Same-case component restore at L26/L27:
   layer_input / attn_out / mlp_out / layer_out.
2. Cross-case donor delta controls for L26/L27 layer_out:
   same_value, same_relation_diff_value, same_family_diff_value, unrelated.

No learned classifier or PCA is used. The output metrics are expected token
rank/top1 and simple route margins.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
from phase597_state_conditioned_mlp_generation_audit import get_mlp  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_attn  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_diag,
    route_id_sets,
    route_scores,
    select_base_cases,
    value_phrase,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    SHORT_VARIANT,
    TERSE_VARIANT,
    select_paired_cases,
)


OUT_ROOT = Path("results/glm5_phase687_l26_l27_value_support_state_decomposition")
COMPONENTS = ["layer_input", "attn_out", "mlp_out", "layer_out"]
DONOR_KINDS = ["same_value", "same_relation_diff_value", "same_family_diff_value", "unrelated"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_layers(model_name: str, n_layers: int) -> list[int]:
    if model_name == "deepseek7b":
        raw = [26, 27]
    elif model_name == "glm4":
        raw = [38, 39]
    else:
        raw = [33, 34]
    return [li for li in raw if 0 <= li < n_layers]


def get_module(model, layer_idx: int, component: str):
    layer = get_layers(model)[layer_idx]
    if component in {"layer_input", "layer_out"}:
        return layer
    if component == "attn_out":
        return get_attn(layer)
    if component == "mlp_out":
        return get_mlp(layer)
    raise ValueError(component)


def capture_states(
    model,
    tokenizer,
    device,
    prompt: str,
    sites: list[tuple[int, str]],
) -> tuple[torch.Tensor, dict[tuple[int, str], torch.Tensor]]:
    captured: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    for site in sites:
        li, component = site
        module = get_module(model, li, component)
        if module is None:
            continue
        if component == "layer_input":
            def pre_hook(_module, inputs, site=site):
                captured[site] = inputs[0][0, -1].detach()

            handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            def out_hook(_module, _inputs, output, site=site):
                y = extract_tensor(output)
                captured[site] = y[0, -1].detach()

            handles.append(module.register_forward_hook(out_hook))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        logits = out.logits[0, -1].detach()
    finally:
        for h in handles:
            h.remove()
    return logits, captured


def patch_tensor_at_last(output_or_input, new_vec: torch.Tensor, component: str):
    y = output_or_input.clone()
    y[0, -1] = new_vec.to(device=y.device, dtype=y.dtype)
    return y


def install_patch_hooks(
    model,
    patches: list[dict[str, Any]],
):
    handles = []
    for patch in patches:
        li = patch["layer"]
        component = patch["component"]
        new_vec = patch["new_vec"]
        module = get_module(model, li, component)
        if module is None:
            continue
        if component == "layer_input":
            def pre_hook(_module, inputs, new_vec=new_vec):
                y_new = patch_tensor_at_last(inputs[0], new_vec, "layer_input")
                return (y_new,) + tuple(inputs[1:])

            handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            def out_hook(_module, _inputs, output, new_vec=new_vec):
                y = extract_tensor(output)
                y_new = patch_tensor_at_last(y, new_vec, component)
                if isinstance(output, tuple):
                    return (y_new,) + output[1:]
                return y_new

            handles.append(module.register_forward_hook(out_hook))
    return handles


def best_expected_rank(logits: torch.Tensor, expected_ids: set[int]) -> tuple[int, int, bool]:
    logits_cpu = logits.detach().float().cpu()
    valid = [tid for tid in expected_ids if 0 <= tid < logits_cpu.numel()]
    best_id = max(valid, key=lambda tid: float(logits_cpu[tid].item()))
    rank = int((logits_cpu > logits_cpu[best_id]).sum().item()) + 1
    return int(best_id), rank, rank == 1


def classify(logits: torch.Tensor, routes: dict[str, set[int]], expected_ids: set[int]) -> dict[str, Any]:
    logits_cpu = logits.detach().float().cpu()
    expected_id, expected_rank, expected_top1 = best_expected_rank(logits_cpu, expected_ids)
    diag = route_diag(route_scores(logits_cpu, routes), "value")
    return {
        "expected_id": expected_id,
        "expected_rank": expected_rank,
        "expected_top1": expected_top1,
        "top1_id": int(torch.argmax(logits_cpu).item()),
        "prose_minus_value": diag["prose_minus_value"],
        "target_margin": diag["target_margin"],
        "best_other_route": diag["best_other_route"],
    }


def paired_case_metadata(case_map: dict[str, dict], paired_ids: list[str]) -> dict[str, dict[str, Any]]:
    return {
        cid: {
            "case_id": cid,
            "family": case_map[cid]["family"],
            "relation": case_map[cid].get("relation"),
            "value": value_phrase(case_map[cid]),
        }
        for cid in paired_ids
    }


def choose_donors(meta: dict[str, dict[str, Any]], case_id: str) -> dict[str, str | None]:
    cur = meta[case_id]
    ids = [cid for cid in sorted(meta) if cid != case_id]

    def first(pred):
        for cid in ids:
            if pred(meta[cid]):
                return cid
        return None

    return {
        "same_value": first(lambda m: m["value"] == cur["value"]),
        "same_relation_diff_value": first(lambda m: m["relation"] == cur["relation"] and m["value"] != cur["value"]),
        "same_family_diff_value": first(lambda m: m["family"] == cur["family"] and m["value"] != cur["value"]),
        "unrelated": first(lambda m: m["family"] != cur["family"] and m["relation"] != cur["relation"] and m["value"] != cur["value"]),
    }


def random_same_norm(vec: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    noise = torch.randn(vec.shape, generator=gen, dtype=torch.float32)
    noise = noise / (noise.norm() + 1e-8) * float(vec.float().norm().detach().cpu().item())
    return noise.to(device=vec.device, dtype=vec.dtype)


def run_patched(model, tokenizer, device, prompt: str, patches: list[dict[str, Any]], routes, expected_ids):
    handles = install_patch_hooks(model, patches)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    model, tokenizer, device = load_model_flash(args.model)
    rows = []
    try:
        layers = model_layers(args.model, len(get_layers(model)))
        sites = [(li, comp) for li in layers for comp in COMPONENTS]
        cache: dict[str, dict[str, Any]] = {}
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            prompts = {
                "short": prompt_for(case, SHORT_VARIANT),
                "terse": prompt_for(case, TERSE_VARIANT),
            }
            short_logits, short_states = capture_states(model, tokenizer, device, prompts["short"], sites)
            terse_logits, terse_states = capture_states(model, tokenizer, device, prompts["terse"], sites)
            cache[case_id] = {
                "short_states": short_states,
                "terse_states": terse_states,
                "short_diag": classify(short_logits, routes, expected_ids),
                "terse_diag": classify(terse_logits, routes, expected_ids),
                "routes": routes,
                "expected_ids": expected_ids,
                "short_prompt": prompts["short"],
            }
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: cached {idx}/{len(paired_ids)} paired cases")

        for idx, case_id in enumerate(paired_ids, 1):
            cur = cache[case_id]
            short_states = cur["short_states"]
            terse_states = cur["terse_states"]
            short_diag = cur["short_diag"]
            terse_diag = cur["terse_diag"]
            # Same-case component decomposition.
            for li in layers:
                for component in COMPONENTS:
                    site = (li, component)
                    if site not in short_states or site not in terse_states:
                        continue
                    delta = terse_states[site] - short_states[site]
                    conditions = [
                        ("same_case_add_delta", short_states[site] + delta),
                        ("same_case_replace", terse_states[site]),
                        ("random_same_norm", short_states[site] + random_same_norm(delta, seed=idx * 1009 + li * 17)),
                    ]
                    for mode, new_vec in conditions:
                        patched = run_patched(
                            model, tokenizer, device, cur["short_prompt"],
                            [{"layer": li, "component": component, "new_vec": new_vec}],
                            cur["routes"], cur["expected_ids"],
                        )
                        rows.append(make_row(meta, case_id, "component", mode, f"L{li}_{component}", None, short_diag, terse_diag, patched))

            # Cross-case donor delta controls for layer_out only.
            donors = choose_donors(meta, case_id)
            for donor_kind, donor_id in donors.items():
                if donor_id is None or donor_id not in cache:
                    continue
                donor = cache[donor_id]
                for li in layers:
                    site = (li, "layer_out")
                    delta = donor["terse_states"][site] - donor["short_states"][site]
                    for mode, new_vec in [
                        (f"{donor_kind}_add_delta", short_states[site] + delta),
                        (f"{donor_kind}_replace", donor["terse_states"][site]),
                    ]:
                        patched = run_patched(
                            model, tokenizer, device, cur["short_prompt"],
                            [{"layer": li, "component": "layer_out", "new_vec": new_vec}],
                            cur["routes"], cur["expected_ids"],
                        )
                        rows.append(make_row(meta, case_id, "cross_donor", mode, f"L{li}_layer_out", donor_id, short_diag, terse_diag, patched))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: patched {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase687_{args.model}_state_decomposition_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 687,
        "title": "L26/L27 Value-Support State Decomposition",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "layers": layers,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase687_{args.model}_state_decomposition_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def make_row(meta, case_id, kind, mode, site, donor_id, short_diag, terse_diag, patched):
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "kind": kind,
        "mode": mode,
        "site": site,
        "donor_id": donor_id,
        "donor_family": meta[donor_id]["family"] if donor_id else None,
        "donor_relation": meta[donor_id]["relation"] if donor_id else None,
        "donor_value": meta[donor_id]["value"] if donor_id else None,
        "short_rank": short_diag["expected_rank"],
        "terse_rank": terse_diag["expected_rank"],
        "patched_rank": patched["expected_rank"],
        "short_top1": short_diag["expected_top1"],
        "terse_top1": terse_diag["expected_top1"],
        "patched_top1": patched["expected_top1"],
        "short_pmv": short_diag["prose_minus_value"],
        "terse_pmv": terse_diag["prose_minus_value"],
        "patched_pmv": patched["prose_minus_value"],
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "repair_rate": sum(1 for r in rows if (not r["short_top1"]) and r["patched_top1"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_short_rank": sum(r["short_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_terse_rank": sum(r["terse_rank"] for r in rows) / n,
        "mean_rank_delta": sum(r["short_rank"] - r["patched_rank"] for r in rows) / n,
        "mean_short_pmv": sum(r["short_pmv"] for r in rows) / n,
        "mean_patched_pmv": sum(r["patched_pmv"] for r in rows) / n,
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["kind"], r["mode"], r["site"])].append(r)
    by_condition = {f"{k}|{m}|{s}": summarize_group(v) for (k, m, s), v in grouped.items()}
    component_rows = [r for r in rows if r["kind"] == "component"]
    cross_rows = [r for r in rows if r["kind"] == "cross_donor"]
    best_component = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("component|")),
        key=lambda kv: (kv[1].get("repair_rate", 0.0), -kv[1].get("mean_patched_rank", 1e9)),
        reverse=True,
    )[:16]
    best_cross = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("cross_donor|")),
        key=lambda kv: (kv[1].get("repair_rate", 0.0), -kv[1].get("mean_patched_rank", 1e9)),
        reverse=True,
    )[:16]
    by_kind = {
        "component": summarize_group(component_rows),
        "cross_donor": summarize_group(cross_rows),
    }
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "by_kind": by_kind,
        "by_condition": by_condition,
        "best_component_conditions": [{"condition": k, **v} for k, v in best_component],
        "best_cross_donor_conditions": [{"condition": k, **v} for k, v in best_cross],
    }


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase687_*_state_decomposition_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 687,
        "title": "L26/L27 Value-Support State Decomposition Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase687_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 687 L26/L27 Value-Support State Decomposition",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | layers | best_component | comp_repair | comp_rank | best_cross | cross_repair | cross_rank |",
        "|---|---:|---|---|---:|---:|---|---:|---:|",
    ]
    for item in models:
        bc = item["summary"]["best_component_conditions"][0] if item["summary"]["best_component_conditions"] else {}
        bx = item["summary"]["best_cross_donor_conditions"][0] if item["summary"]["best_cross_donor_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {item['layers']} | "
            f"{bc.get('condition', '')} | {bc.get('repair_rate', 0.0):.3f} | {bc.get('mean_patched_rank', 0.0):.2f} | "
            f"{bx.get('condition', '')} | {bx.get('repair_rate', 0.0):.3f} | {bx.get('mean_patched_rank', 0.0):.2f} |"
        )
    lines.extend(["", "## Best Component Conditions", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in item["summary"]["best_component_conditions"][:12]:
            lines.append(
                f"| {row['condition']} | {row['repair_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                f"{row['mean_patched_rank']:.2f} | {row['mean_rank_delta']:.2f} | {row['mean_patched_pmv']:.3f} |"
            )
        lines.append("")
    lines.append("## Best Cross-Donor Conditions")
    lines.append("")
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| condition | repair | top1 | patched_rank | rank_delta | patched_pmv |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in item["summary"]["best_cross_donor_conditions"][:12]:
            lines.append(
                f"| {row['condition']} | {row['repair_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                f"{row['mean_patched_rank']:.2f} | {row['mean_rank_delta']:.2f} | {row['mean_patched_pmv']:.3f} |"
            )
        lines.append("")
    (OUT_ROOT / "phase687_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=12)
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
