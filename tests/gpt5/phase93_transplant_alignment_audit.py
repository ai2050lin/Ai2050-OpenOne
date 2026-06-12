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
from phase90_component_margin_reader_alignment import build_items  # noqa: E402
from phase92_cross_item_component_transplant import (  # noqa: E402
    donor_kinds,
    run_item_node,
    select_donors,
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
            raise ValueError(f"unknown component in node: {raw}")
        out.append((int(layer), comp))
    return out


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "value_drop": avg([float(v["value_drop"]) for v in vals]),
        "value_patch_gain": avg([float(v["value_patch_gain"]) for v in vals]),
        "value_patch_gap": avg([float(v["value_patch_gap"]) for v in vals]),
        "letter_drop": avg([float(v["letter_drop"]) for v in vals]),
        "letter_patch_gain": avg([float(v["letter_patch_gain"]) for v in vals]),
        "letter_patch_gap": avg([float(v["letter_patch_gap"]) for v in vals]),
        "value_top1_patch_gain": avg([float(v["value_top1_patch_gain"]) for v in vals]),
        "letter_top1_patch_gain": avg([float(v["letter_top1_patch_gain"]) for v in vals]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_copy_mode": defaultdict(list),
        "by_node_copy_mode": defaultdict(list),
        "by_node_copy_mode_donor_kind": defaultdict(list),
        "by_copy_mode_donor_kind": defaultdict(list),
    }
    for row in rows:
        node = f"L{row['layer']}:{row['component']}"
        groups["by_copy_mode"][row["copy_mode"]].append(row)
        groups["by_node_copy_mode"][(node, row["copy_mode"])].append(row)
        groups["by_node_copy_mode_donor_kind"][(node, row["copy_mode"], row["donor_kind"])].append(row)
        groups["by_copy_mode_donor_kind"][(row["copy_mode"], row["donor_kind"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE93_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    nodes = parse_nodes(args.nodes)
    copy_modes = parse_csv(args.copy_modes)
    wanted_donor_kinds = set(parse_csv(args.donor_kinds)) if args.donor_kinds else set(donor_kinds())
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors_by_idx = select_donors(items)
    log(f"Phase93 model={args.model} items={len(items)} nodes={nodes} copy_modes={copy_modes} donor_kinds={sorted(wanted_donor_kinds)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase93_transplant_alignment_audit.json"
    partial_path = out_dir / f"{args.model}_phase93_transplant_alignment_audit.partial.json"
    results: dict[str, Any] = {
        "phase": 93,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "transplant_alignment_audit",
        "nodes": [f"{l}:{c}" for l, c in nodes],
        "num_items": len(items),
        "slots": sorted({x["slot"] for x in items}),
        "choice_template": args.choice_template,
        "copy_modes": copy_modes,
        "donor_kinds": sorted(wanted_donor_kinds),
        "rows": [],
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 93 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed = {
        (int(r["layer"]), r["component"], int(r["item_idx"]), r["donor_kind"], r["copy_mode"])
        for r in results["rows"]
    }
    t0 = time.time()
    for copy_mode in copy_modes:
        for layer_idx, component in nodes:
            for idx, _item in enumerate(items):
                pending = [
                    k for k in wanted_donor_kinds
                    if (layer_idx, component, idx, k, copy_mode) not in completed
                ]
                if not pending:
                    continue
                item_rows = run_item_node(
                    model, tokenizer, device, layers, items, donors_by_idx, idx,
                    layer_idx, component, args.choice_template, args.max_distractors,
                    args.max_length, copy_mode,
                )
                for row in item_rows:
                    if row["donor_kind"] not in wanted_donor_kinds:
                        continue
                    key = (int(row["layer"]), row["component"], int(row["item_idx"]), row["donor_kind"], row["copy_mode"])
                    if key not in completed:
                        results["rows"].append(row)
                        completed.add(key)
                if (idx + 1) % args.progress_every == 0:
                    log(
                        f"mode={copy_mode} node=L{layer_idx}:{component} "
                        f"item={idx + 1}/{len(items)} rows={len(results['rows'])} "
                        f"elapsed={time.time() - t0:.0f}s"
                    )
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
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--slots", default="category,color,function,material,location")
    parser.add_argument("--slot-templates", default="")
    parser.add_argument("--max-items", type=int, default=420)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--copy-modes", default="tail,prefix,both")
    parser.add_argument("--donor-kinds", default="self_restore,same_slot_same_target,same_slot_diff_target,diff_slot_same_object,diff_slot_diff_object")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=70)
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
