from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
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
from phase72_object_relation_value_fullseq_closure import capture_state, stats_from_scores  # noqa: E402
from phase75_relation_frame_token_intervention import get_frame_positions  # noqa: E402
from phase76_object_frame_joint_closure import uniq  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items  # noqa: E402
from phase79_rank_sweep_remainder_audit import fullseq_logprob_rank_patch, project_state  # noqa: E402
from phase83_suffix_token_decomposition import build_suffix_token_bases  # noqa: E402
from phase84_clean_suffix_erase_restore import zero_source  # noqa: E402
from phase87_reader_stack_calibration import (  # noqa: E402
    choice_templates,
    option_letters,
    option_orders,
    parse_choice,
    render_options,
)


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


def make_hook(patches: list[tuple[int, torch.Tensor, torch.Tensor, str]]):
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
        for pos_raw, source_cpu, basis_cpu, mode in patches:
            pos = int(pos_raw) if pos_raw >= 0 else hs.shape[1] + int(pos_raw)
            if 0 <= pos < hs.shape[1]:
                hs[0, pos, :] = project_state(hs[0, pos, :], source_cpu, basis_cpu, mode)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    return hook_fn


def generate_text_patch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    module_name: str,
    destroy_layer: int | None = None,
    destroy_patches: list[tuple[int, torch.Tensor, torch.Tensor, str]] | None = None,
    restore_layer: int | None = None,
    restore_patches: list[tuple[int, torch.Tensor, torch.Tensor, str]] | None = None,
) -> str:
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] > max_length:
        return ""
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if destroy_layer is not None and destroy_patches:
            handles.append(get_module(layers[destroy_layer], module_name).register_forward_hook(make_hook(destroy_patches)))
        if restore_layer is not None and restore_patches:
            handles.append(get_module(layers[restore_layer], module_name).register_forward_hook(make_hook(restore_patches)))
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
    gen_ids = out[0, input_ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def build_prompt(template: str, clean_prompt: str, candidates: list[str]) -> str:
    return template.format(clean_prompt=clean_prompt, options=render_options(candidates))


def closed_stats(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    item: dict[str, Any],
    candidates: list[str],
    max_length: int,
    module: str,
    destroy_layer: int | None = None,
    destroy_patches: list[tuple[int, torch.Tensor, torch.Tensor, str]] | None = None,
    restore_layer: int | None = None,
    restore_patches: list[tuple[int, torch.Tensor, torch.Tensor, str]] | None = None,
) -> dict[str, Any]:
    scores = {
        v: fullseq_logprob_rank_patch(
            model,
            tokenizer,
            device,
            layers,
            item["clean_prompt"],
            v,
            max_length,
            module,
            destroy_layer,
            destroy_patches,
            restore_layer,
            restore_patches,
        )
        for v in candidates
    }
    return stats_from_scores(scores, item["target"], [v for v in candidates if v != item["target"]])


def choose_condition_specs(
    conditions: list[str],
    op: int,
    fp: int,
    z: torch.Tensor,
    restore_obj: torch.Tensor,
    restore_frame: torch.Tensor,
    bases_d: dict[str, torch.Tensor],
    bases_r: dict[str, torch.Tensor],
) -> dict[str, tuple[list[tuple[int, torch.Tensor, torch.Tensor, str]], list[tuple[int, torch.Tensor, torch.Tensor, str]]]]:
    basis_keys = {
        "suffix_final": "suffix_final_basis",
        "suffix_all": "suffix_all_basis",
        "suffix_function": "suffix_function_basis",
        "suffix_lexical": "suffix_lexical_basis",
        "all_suffix_tokens": "all_suffix_token_basis",
    }
    specs = {}
    for cond in conditions:
        if cond.startswith("frame_"):
            label = cond.removeprefix("frame_")
            bkey = basis_keys[label]
            specs[cond] = (
                [(fp, z, bases_d[bkey], "subspace")],
                [(fp, restore_frame, bases_r[bkey], "subspace")],
            )
        elif cond.startswith("object_"):
            label = cond.removeprefix("object_")
            bkey = basis_keys[label]
            specs[cond] = (
                [(op, z, bases_d[bkey], "subspace")],
                [(op, restore_obj, bases_r[bkey], "subspace")],
            )
        else:
            raise ValueError(f"unknown condition={cond}")
    return specs


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["base_choice_correct"]]
    closed_eligible = [v for v in vals if v["closed_base_rank"] == 1]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "closed_eligible_n": len(closed_eligible),
        "base_choice_top1": avg([float(v["base_choice_correct"]) for v in vals]),
        "erase_choice_top1": avg([float(v["erase_choice_correct"]) for v in vals]),
        "restore_choice_top1": avg([float(v["restore_choice_correct"]) for v in vals]),
        "choice_drop": avg([float(v["choice_drop"]) for v in vals]),
        "choice_restore_gain": avg([float(v["choice_restore_gain"]) for v in vals]),
        "choice_restore_gap": avg([float(v["choice_restore_gap"]) for v in vals]),
        "base_choice_valid": avg([float(v["base_choice_valid"]) for v in vals]),
        "erase_choice_valid": avg([float(v["erase_choice_valid"]) for v in vals]),
        "restore_choice_valid": avg([float(v["restore_choice_valid"]) for v in vals]),
        "eligible_erase_choice_top1": avg([float(v["erase_choice_correct"]) for v in eligible]),
        "eligible_restore_choice_top1": avg([float(v["restore_choice_correct"]) for v in eligible]),
        "eligible_choice_drop": avg([float(v["choice_drop"]) for v in eligible]),
        "eligible_choice_restore_gain": avg([float(v["choice_restore_gain"]) for v in eligible]),
        "closed_base_top1": avg([float(v["closed_base_rank"] == 1) for v in vals]),
        "closed_erase_top1": avg([float(v["closed_erase_rank"] == 1) for v in vals]),
        "closed_restore_top1": avg([float(v["closed_restore_rank"] == 1) for v in vals]),
        "closed_drop": avg([float(v["closed_drop"]) for v in vals]),
        "closed_restore_gain": avg([float(v["closed_restore_gain"]) for v in vals]),
        "closed_restore_gap": avg([float(v["closed_restore_gap"]) for v in vals]),
        "closed_base_margin": avg([float(v["closed_base_margin"]) for v in vals]),
        "closed_erase_margin": avg([float(v["closed_erase_margin"]) for v in vals]),
        "closed_restore_margin": avg([float(v["closed_restore_margin"]) for v in vals]),
        "closed_eligible_erase_top1": avg([float(v["closed_erase_rank"] == 1) for v in closed_eligible]),
        "closed_eligible_restore_top1": avg([float(v["closed_restore_rank"] == 1) for v in closed_eligible]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_condition": defaultdict(list),
        "by_condition_path": defaultdict(list),
        "by_condition_template": defaultdict(list),
        "by_condition_order": defaultdict(list),
        "by_condition_relation": defaultdict(list),
        "by_path": defaultdict(list),
    }
    for row in rows:
        c = row["condition"]
        dl, rl = int(row["destroy_layer"]), int(row["restore_layer"])
        groups["by_condition"][c].append(row)
        groups["by_condition_path"][(c, dl, rl)].append(row)
        groups["by_condition_template"][(c, row["template_key"])].append(row)
        groups["by_condition_order"][(c, row["order_key"])].append(row)
        groups["by_condition_relation"][(c, row["relation"])].append(row)
        groups["by_path"][(dl, rl)].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE88_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    templates = choice_templates()
    if args.choice_templates:
        keep = set(parse_csv(args.choice_templates))
        templates = {k: v for k, v in templates.items() if k in keep}
    order_keep = set(parse_csv(args.choice_orders)) if args.choice_orders else {"rotating", "target_last"}
    conditions = parse_csv(args.conditions)
    log(
        f"Phase88 model={args.model} items={len(items)} pairs={layer_pairs} "
        f"templates={list(templates)} orders={sorted(order_keep)} conditions={conditions}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase88_choice_reader_erase_restore.json"
    partial_path = out_dir / f"{args.model}_phase88_choice_reader_erase_restore.partial.json"
    results: dict[str, Any] = {
        "phase": 88,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "choice_reader_erase_restore",
        "num_items": len(items),
        "layer_pairs": layer_pairs,
        "module": args.module,
        "conditions": conditions,
        "choice_templates": list(templates),
        "choice_orders": sorted(order_keep),
        "max_distractors": args.max_distractors,
        "contrast_rank": args.contrast_rank,
        "component_rank": args.component_rank,
        "max_basis_items": args.max_basis_items,
        "relations": sorted({x["relation"] for x in items}),
        "rows": [],
        "summary": {},
        "samples": [],
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 88 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results.setdefault("samples", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed = {
        (
            int(r["destroy_layer"]),
            int(r["restore_layer"]),
            int(r["item_idx"]),
            r["condition"],
            r["template_key"],
            r["order_key"],
        )
        for r in results["rows"]
    }
    base_choice_cache: dict[tuple[int, str, str], dict[str, Any]] = {}
    base_closed_cache: dict[int, dict[str, Any]] = {}
    t0 = time.time()
    for destroy_layer, restore_layer in layer_pairs:
        log(f"building suffix bases for L{destroy_layer} and L{restore_layer}")
        bases_d = build_suffix_token_bases(
            model,
            tokenizer,
            device,
            layers,
            items,
            destroy_layer,
            args.module,
            args.max_length,
            args.contrast_rank,
            args.component_rank,
            args.max_basis_items,
        )
        bases_r = build_suffix_token_bases(
            model,
            tokenizer,
            device,
            layers,
            items,
            restore_layer,
            args.module,
            args.max_length,
            args.contrast_rank,
            args.component_rank,
            args.max_basis_items,
        )
        log(f"bases ready for {destroy_layer}->{restore_layer}")
        for idx, item in enumerate(items):
            clean_pos = get_frame_positions(tokenizer, item["clean_prompt"], item["object"])
            if any(x is None for x in (clean_pos.get("object_last"), clean_pos.get("frame_last"))):
                continue
            orders = {k: v for k, v in option_orders(item, idx, args.max_distractors).items() if k in order_keep}
            closed_candidates = option_orders(item, idx, args.max_distractors)["target_first"]
            if idx not in base_closed_cache:
                base_closed_cache[idx] = closed_stats(
                    model,
                    tokenizer,
                    device,
                    layers,
                    item,
                    closed_candidates,
                    args.max_length,
                    args.module,
                )
            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            op, fp = int(clean_pos["object_last"]), int(clean_pos["frame_last"])
            dim = int(h_clean_r.shape[-1])
            z = zero_source(dim)
            specs = choose_condition_specs(
                conditions,
                op,
                fp,
                z,
                h_clean_r[op],
                h_clean_r[fp],
                bases_d,
                bases_r,
            )
            closed_by_condition = {}
            for condition, (destroy_patches, restore_patches) in specs.items():
                closed_erase = closed_stats(
                    model,
                    tokenizer,
                    device,
                    layers,
                    item,
                    closed_candidates,
                    args.max_length,
                    args.module,
                    destroy_layer,
                    destroy_patches,
                )
                closed_restore = closed_stats(
                    model,
                    tokenizer,
                    device,
                    layers,
                    item,
                    closed_candidates,
                    args.max_length,
                    args.module,
                    destroy_layer,
                    destroy_patches,
                    restore_layer,
                    restore_patches,
                )
                closed_by_condition[condition] = (closed_erase, closed_restore, destroy_patches, restore_patches)

            for order_key, candidates in orders.items():
                target_letter = option_letters(len(candidates))[candidates.index(item["target"])]
                for template_key, template in templates.items():
                    base_key = (idx, template_key, order_key)
                    if base_key not in base_choice_cache:
                        base_prompt = build_prompt(template, item["clean_prompt"], candidates)
                        generated = generate_text_patch(
                            model,
                            tokenizer,
                            device,
                            layers,
                            base_prompt,
                            args.max_length,
                            args.choice_max_new_tokens,
                            args.module,
                        )
                        parsed = parse_choice(generated, candidates)
                        base_choice_cache[base_key] = {
                            "generated": generated,
                            **parsed,
                            "choice_correct": parsed["selected_value"] == item["target"],
                            "choice_valid": bool(parsed["choice_valid"]),
                        }
                    base_choice = base_choice_cache[base_key]
                    prompt = build_prompt(template, item["clean_prompt"], candidates)
                    for condition, (closed_erase, closed_restore, destroy_patches, restore_patches) in closed_by_condition.items():
                        comp_key = (destroy_layer, restore_layer, idx, condition, template_key, order_key)
                        if comp_key in completed:
                            continue
                        erase_generated = generate_text_patch(
                            model,
                            tokenizer,
                            device,
                            layers,
                            prompt,
                            args.max_length,
                            args.choice_max_new_tokens,
                            args.module,
                            destroy_layer,
                            destroy_patches,
                        )
                        restore_generated = generate_text_patch(
                            model,
                            tokenizer,
                            device,
                            layers,
                            prompt,
                            args.max_length,
                            args.choice_max_new_tokens,
                            args.module,
                            destroy_layer,
                            destroy_patches,
                            restore_layer,
                            restore_patches,
                        )
                        erase_parsed = parse_choice(erase_generated, candidates)
                        restore_parsed = parse_choice(restore_generated, candidates)
                        base_correct = bool(base_choice["choice_correct"])
                        erase_correct = erase_parsed["selected_value"] == item["target"]
                        restore_correct = restore_parsed["selected_value"] == item["target"]
                        base_closed = base_closed_cache[idx]
                        row = {
                            "item_idx": idx,
                            "destroy_layer": destroy_layer,
                            "restore_layer": restore_layer,
                            "condition": condition,
                            "template_key": template_key,
                            "order_key": order_key,
                            "relation": item["relation"],
                            "frame_key": item["frame_key"],
                            "object": item["object"],
                            "target": item["target"],
                            "target_letter": target_letter,
                            "candidates": candidates,
                            "base_generated": base_choice["generated"],
                            "erase_generated": erase_generated,
                            "restore_generated": restore_generated,
                            "base_selected_value": base_choice["selected_value"],
                            "erase_selected_value": erase_parsed["selected_value"],
                            "restore_selected_value": restore_parsed["selected_value"],
                            "base_selected_letter": base_choice["selected_letter"],
                            "erase_selected_letter": erase_parsed["selected_letter"],
                            "restore_selected_letter": restore_parsed["selected_letter"],
                            "base_choice_valid": bool(base_choice["choice_valid"]),
                            "erase_choice_valid": bool(erase_parsed["choice_valid"]),
                            "restore_choice_valid": bool(restore_parsed["choice_valid"]),
                            "base_choice_correct": base_correct,
                            "erase_choice_correct": bool(erase_correct),
                            "restore_choice_correct": bool(restore_correct),
                            "choice_drop": float(base_correct) - float(erase_correct),
                            "choice_restore_gain": float(restore_correct) - float(erase_correct),
                            "choice_restore_gap": float(base_correct) - float(restore_correct),
                            "closed_base_rank": base_closed["rank"],
                            "closed_erase_rank": closed_erase["rank"],
                            "closed_restore_rank": closed_restore["rank"],
                            "closed_base_margin": base_closed["margin"],
                            "closed_erase_margin": closed_erase["margin"],
                            "closed_restore_margin": closed_restore["margin"],
                            "closed_base_top": base_closed["top"],
                            "closed_erase_top": closed_erase["top"],
                            "closed_restore_top": closed_restore["top"],
                            "closed_drop": float(base_closed["margin"]) - float(closed_erase["margin"]),
                            "closed_restore_gain": float(closed_restore["margin"]) - float(closed_erase["margin"]),
                            "closed_restore_gap": float(base_closed["margin"]) - float(closed_restore["margin"]),
                            "closed_choice_agreement_base": base_closed["top"] == base_choice["selected_value"],
                            "closed_choice_agreement_erase": closed_erase["top"] == erase_parsed["selected_value"],
                            "closed_choice_agreement_restore": closed_restore["top"] == restore_parsed["selected_value"],
                        }
                        results["rows"].append(row)
                        if len(results["samples"]) < args.max_samples and (row["choice_drop"] or row["choice_restore_gain"] or not row["base_choice_valid"]):
                            results["samples"].append(row)

            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
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
    parser.add_argument("--max-items", type=int, default=336)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--choice-max-new-tokens", type=int, default=4)
    parser.add_argument("--choice-templates", default="choice_json_letter")
    parser.add_argument("--choice-orders", default="rotating,target_last")
    parser.add_argument("--conditions", default="frame_suffix_final,frame_suffix_all,frame_suffix_function,frame_suffix_lexical,frame_all_suffix_tokens,object_suffix_final,object_all_suffix_tokens")
    parser.add_argument("--contrast-rank", type=int, default=64)
    parser.add_argument("--component-rank", type=int, default=24)
    parser.add_argument("--max-basis-items", type=int, default=224)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=84)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        release_loaded(None)
        cleanup_cuda()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
