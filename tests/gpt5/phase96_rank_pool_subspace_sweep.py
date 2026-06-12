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
from phase68_object_attribute_natural_exchange import load_model, parse_csv  # noqa: E402
from phase87_reader_stack_calibration import option_letters  # noqa: E402
from phase90_component_margin_reader_alignment import build_items, uniq  # noqa: E402
from phase92_cross_item_component_transplant import select_donors  # noqa: E402
from phase94_factor_subspace_closure import (  # noqa: E402
    build_factor_bases,
    condition_score,
    module_name,
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


def parse_nodes(text: str) -> list[tuple[int, str]]:
    out = []
    for raw in parse_csv(text):
        layer, comp = raw.split(":", 1)
        if comp not in {"attn", "mlp"}:
            raise ValueError(f"unknown node component={raw}")
        out.append((int(layer), comp))
    return out


def slice_basis(basis: torch.Tensor, factor: str, rank: int) -> torch.Tensor:
    if basis.numel() == 0:
        return basis
    if factor == "pc1":
        return basis[:, :1].contiguous()
    return basis[:, : min(rank, basis.shape[1])].contiguous()


def finite(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return v == v and abs(v) != float("inf")


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "value_delta": avg([float(v["value_delta"]) for v in vals]),
        "letter_delta": avg([float(v["letter_delta"]) for v in vals]),
        "value_top1_delta": avg([float(v["value_top1_delta"]) for v in vals]),
        "letter_top1_delta": avg([float(v["letter_top1_delta"]) for v in vals]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_node": defaultdict(list),
        "by_pool": defaultdict(list),
        "by_rank": defaultdict(list),
        "by_factor": defaultdict(list),
        "by_node_pool_factor": defaultdict(list),
        "by_node_rank_factor": defaultdict(list),
        "by_node_pool_rank_factor": defaultdict(list),
    }
    for row in rows:
        node = f"L{row['layer']}:{row['component']}"
        groups["by_node"][node].append(row)
        groups["by_pool"][row["pool_mode"]].append(row)
        groups["by_rank"][row["rank"]].append(row)
        groups["by_factor"][row["factor"]].append(row)
        groups["by_node_pool_factor"][(node, row["pool_mode"], row["factor"])].append(row)
        groups["by_node_rank_factor"][(node, row["rank"], row["factor"])].append(row)
        groups["by_node_pool_rank_factor"][(node, row["pool_mode"], row["rank"], row["factor"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def compact_summary(data: dict[str, Any], limit: int = 240) -> dict[str, Any]:
    out = {}
    for k, v in sorted(data.items())[:limit]:
        out[k] = {
            "n": v.get("n", 0),
            "value_delta": round(float(v.get("value_delta", 0.0)), 4),
            "letter_delta": round(float(v.get("letter_delta", 0.0)), 4),
            "value_top1_delta": round(float(v.get("value_top1_delta", 0.0)), 4),
            "letter_top1_delta": round(float(v.get("letter_top1_delta", 0.0)), 4),
        }
    return out


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE96_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    parsed_nodes = parse_nodes(args.nodes)
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors = select_donors(items)
    ranks = [int(x) for x in parse_csv(args.ranks)]
    pools = parse_csv(args.pool_modes)
    factors = parse_csv(args.factors)
    max_rank = max([r for r in ranks if r > 0] or [1])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase96_rank_pool_subspace_sweep.json"
    partial_path = out_dir / f"{args.model}_phase96_rank_pool_subspace_sweep.partial.json"

    results: dict[str, Any] = {
        "phase": 96,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "rank_pool_subspace_sweep_destroy_only",
        "nodes": [f"{l}:{c}" for l, c in parsed_nodes],
        "num_items": len(items),
        "ranks": ranks,
        "pool_modes": pools,
        "factors": factors,
        "rows": [],
        "basis_dims": {},
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 96 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results.setdefault("basis_dims", {})
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed = {
        (
            int(r["layer"]),
            r["component"],
            r["pool_mode"],
            int(r["rank"]),
            r["factor"],
            int(r["item_idx"]),
        )
        for r in results["rows"]
    }
    t0 = time.time()
    log(
        f"Phase96 model={args.model} items={len(items)} nodes={parsed_nodes} "
        f"pools={pools} ranks={ranks} factors={factors}"
    )
    for layer_idx, component in parsed_nodes:
        clean_cache: dict[int, dict[str, Any]] = {}
        for pool in pools:
            log(f"building max-rank bases model={args.model} node=L{layer_idx}:{component} pool={pool} max_rank={max_rank}")
            bases = build_factor_bases(
                model,
                tokenizer,
                device,
                layers,
                items,
                donors,
                layer_idx,
                component,
                args.choice_template,
                max_rank,
                pool,
                args.max_length,
            )
            results["basis_dims"][f"L{layer_idx}:{component}:{pool}"] = {k: int(v.shape[1]) for k, v in bases.items()}
            partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            for rank in ranks:
                rank_bases = {factor: slice_basis(bases[factor], factor, rank) for factor in factors}
                for idx, item in enumerate(items):
                    if idx not in clean_cache:
                        candidates = uniq([item["target"], *item["distractors"][: args.max_distractors]])
                        letters = option_letters(len(candidates))
                        target_letter = letters[candidates.index(item["target"])]
                        clean_cache[idx] = {
                            "candidates": candidates,
                            "letters": letters,
                            "target_letter": target_letter,
                            "score": condition_score(
                                model,
                                tokenizer,
                                device,
                                layers,
                                item,
                                None,
                                candidates,
                                letters,
                                target_letter,
                                args.choice_template,
                                args.max_length,
                                layer_idx,
                                component,
                                None,
                                "clean",
                                "both",
                            ),
                        }
                    cache = clean_cache[idx]
                    for factor in factors:
                        key = (layer_idx, component, pool, rank, factor, idx)
                        if key in completed:
                            continue
                        patched = condition_score(
                            model,
                            tokenizer,
                            device,
                            layers,
                            item,
                            None,
                            cache["candidates"],
                            cache["letters"],
                            cache["target_letter"],
                            args.choice_template,
                            args.max_length,
                            layer_idx,
                            component,
                            rank_bases[factor],
                            "destroy",
                            "both",
                        )
                        clean = cache["score"]
                        row = {
                            "item_idx": idx,
                            "condition": f"destroy_{factor}",
                            "factor": factor,
                            "op": "destroy",
                            "layer": layer_idx,
                            "component": component,
                            "module_name": module_name(component),
                            "slot": item["slot"],
                            "template_key": item["template_key"],
                            "object": item["object"],
                            "target": item["target"],
                            "pool_mode": pool,
                            "rank": rank,
                            "basis_dim": int(rank_bases[factor].shape[1]),
                            "clean_value_margin": clean["value_margin"],
                            "patched_value_margin": patched["value_margin"],
                            "clean_letter_margin": clean["letter_margin"],
                            "patched_letter_margin": patched["letter_margin"],
                            "clean_value_top1": clean["value_top1"],
                            "patched_value_top1": patched["value_top1"],
                            "clean_letter_top1": clean["letter_top1"],
                            "patched_letter_top1": patched["letter_top1"],
                            "value_delta": patched["value_margin"] - clean["value_margin"],
                            "letter_delta": patched["letter_margin"] - clean["letter_margin"],
                            "value_top1_delta": float(patched["value_top1"]) - float(clean["value_top1"]),
                            "letter_top1_delta": float(patched["letter_top1"]) - float(clean["letter_top1"]),
                        }
                        results["rows"].append(row)
                        completed.add(key)
                    if (idx + 1) % args.progress_every == 0:
                        log(
                            f"model={args.model} node=L{layer_idx}:{component} pool={pool} rank={rank} "
                            f"item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s"
                        )
                        partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                        cleanup_cuda()
    results["summary"] = summarize(results["rows"])
    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {final_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--slots", default="category,color,function,material,location")
    parser.add_argument("--slot-templates", default="")
    parser.add_argument("--max-items", type=int, default=210)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--ranks", default="1,4,16")
    parser.add_argument("--pool-modes", default="tail,prefix,mean")
    parser.add_argument("--factors", default="pc1,object,target,slot,choice")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=35)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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
