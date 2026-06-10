from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import get_module, load_model, parse_csv  # noqa: E402
from phase70_object_relation_value_closure import parse_layer_pairs  # noqa: E402
from phase72_object_relation_value_fullseq_closure import capture_state  # noqa: E402
from phase75_relation_frame_token_intervention import get_frame_positions  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items  # noqa: E402
from phase83_suffix_token_decomposition import build_suffix_token_bases  # noqa: E402
from phase79_rank_sweep_remainder_audit import project_state  # noqa: E402


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def norm_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[\\s:;,.!?\\-]+", "", text)
    text = re.sub(r"\\s+", " ", text)
    return text


def first_span(text: str) -> str:
    text = norm_text(text)
    return re.split(r"[\\n\\r\\.;,]", text, maxsplit=1)[0].strip()


def target_hit(generated: str, target: str) -> tuple[bool, bool]:
    g = first_span(generated)
    t = norm_text(target)
    return g.startswith(t), t in g


def zero_source(dim: int) -> torch.Tensor:
    return torch.zeros(dim, dtype=torch.float32)


def generate_with_patches(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    max_length: int,
    module_name: str,
    max_new_tokens: int,
    patch_layer: int | None = None,
    patches: list[tuple[int, torch.Tensor, torch.Tensor, str]] | None = None,
) -> str:
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] > max_length:
        return ""
    attention_mask = torch.ones_like(input_ids)
    handles = []

    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
        for pos_raw, source_cpu, basis_cpu, mode in patches or []:
            pos = int(pos_raw) if pos_raw >= 0 else hs.shape[1] + int(pos_raw)
            if 0 <= pos < hs.shape[1]:
                hs[0, pos, :] = project_state(hs[0, pos, :], source_cpu, basis_cpu, mode)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    try:
        if patch_layer is not None and patches:
            handles.append(get_module(layers[patch_layer], module_name).register_forward_hook(hook_fn))
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, input_ids.shape[1]:].detach().cpu().tolist()
        return tokenizer.decode(gen_ids, skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["base_prefix_hit"]]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "prefix_hit": avg([1.0 if v["prefix_hit"] else 0.0 for v in vals]),
        "contains_hit": avg([1.0 if v["contains_hit"] else 0.0 for v in vals]),
        "eligible_prefix_hit": avg([1.0 if v["prefix_hit"] else 0.0 for v in eligible]),
        "eligible_contains_hit": avg([1.0 if v["contains_hit"] else 0.0 for v in eligible]),
        "prefix_drop": avg([float(v["base_prefix_hit"]) - float(v["prefix_hit"]) for v in vals]),
        "eligible_prefix_drop": avg([float(v["base_prefix_hit"]) - float(v["prefix_hit"]) for v in eligible]),
        "changed": avg([1.0 if norm_text(v["generated"]) != norm_text(v["base_generated"]) else 0.0 for v in vals]),
        "eligible_changed": avg([1.0 if norm_text(v["generated"]) != norm_text(v["base_generated"]) else 0.0 for v in eligible]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition_path: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_condition_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        c = str(row["condition"])
        rel = str(row["relation"])
        layer = int(row["layer"])
        by_condition[c].append(row)
        by_condition_path[(c, layer)].append(row)
        by_condition_relation[(c, rel)].append(row)
    return {
        "by_condition": {k: group_summary(v) for k, v in by_condition.items()},
        "by_condition_path": {f"{c}:L{layer}": group_summary(v) for (c, layer), v in by_condition_path.items()},
        "by_condition_relation": {f"{c}:{rel}": group_summary(v) for (c, rel), v in by_condition_relation.items()},
    }


def add_row(
    results: dict[str, Any],
    item: dict[str, Any],
    idx: int,
    layer: int,
    cond: str,
    base_generated: str,
    generated: str,
) -> None:
    bp, bc = target_hit(base_generated, item["target"])
    p, c = target_hit(generated, item["target"])
    results["rows"].append({
        "item_idx": idx,
        "layer": layer,
        "condition": cond,
        "relation": item["relation"],
        "frame_key": item["frame_key"],
        "object": item["object"],
        "target": item["target"],
        "base_generated": base_generated,
        "generated": generated,
        "base_prefix_hit": bp,
        "base_contains_hit": bc,
        "prefix_hit": p,
        "contains_hit": c,
    })


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE85_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    audit_layers = sorted({x for pair in layer_pairs for x in pair})
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase85 model={args.model} items={len(items)} layers={audit_layers} component_rank={args.component_rank}")

    results: dict[str, Any] = {
        "phase": 85,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "readout_open_generation_audit",
        "audit_layers": audit_layers,
        "source_layer_pairs": layer_pairs,
        "module": args.module,
        "component_rank": args.component_rank,
        "max_basis_items": args.max_basis_items,
        "max_new_tokens": args.max_new_tokens,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase85_readout_open_generation_audit.json"
    partial_path = out_dir / f"{args.model}_phase85_readout_open_generation_audit.partial.json"
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 85 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed: set[tuple[int, int]] = set()
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in results["rows"]:
        counts[(int(row["layer"]), int(row["item_idx"]))] += 1
    for k, v in counts.items():
        if v >= 11:
            completed.add(k)

    t0 = time.time()
    for layer_idx in audit_layers:
        log(f"building suffix bases for L{layer_idx}")
        bases = build_suffix_token_bases(model, tokenizer, device, layers, items, layer_idx, args.module, args.max_length, args.contrast_rank, args.component_rank, args.max_basis_items)
        log(f"bases ready for L{layer_idx}")
        for idx, item in enumerate(items):
            if (layer_idx, idx) in completed:
                continue
            clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
            if clean_pos.get("frame_last") is None:
                continue
            fp = int(clean_pos["frame_last"])
            h_clean = capture_state(model, tokenizer, device, layers[layer_idx], args.module, item["clean_prompt"], args.max_length)
            dim = int(h_clean.shape[-1])
            z = zero_source(dim)
            restore_frame = h_clean[fp]
            base_generated = generate_with_patches(model, tokenizer, device, layers, item["clean_prompt"], args.max_length, args.module, args.max_new_tokens)
            add_row(results, item, idx, layer_idx, "base", base_generated, base_generated)
            for label in ["suffix_final", "suffix_all", "suffix_function", "suffix_lexical", "all_suffix_tokens"]:
                bkey = f"{label}_basis" if label != "all_suffix_tokens" else "all_suffix_token_basis"
                basis = bases[bkey]
                erase_patches = [(fp, z, basis, "subspace")]
                restore_patches = [(fp, restore_frame, basis, "subspace")]
                erased = generate_with_patches(model, tokenizer, device, layers, item["clean_prompt"], args.max_length, args.module, args.max_new_tokens, layer_idx, erase_patches)
                restored = generate_with_patches(model, tokenizer, device, layers, item["clean_prompt"], args.max_length, args.module, args.max_new_tokens, layer_idx, erase_patches + restore_patches)
                add_row(results, item, idx, layer_idx, f"erase_frame_{label}", base_generated, erased)
                add_row(results, item, idx, layer_idx, f"restore_frame_{label}", base_generated, restored)
            if (idx + 1) % args.progress_every == 0:
                log(f"layer=L{layer_idx} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
                partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cuda()
        partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    results["summary"] = summarize(results["rows"])
    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {final_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--relations", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-items", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--contrast-rank", type=int, default=64)
    parser.add_argument("--component-rank", type=int, default=24)
    parser.add_argument("--max-basis-items", type=int, default=224)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=28)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        release_loaded(None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
