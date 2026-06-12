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
import torch.nn.functional as F


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import get_module, load_model, parse_csv  # noqa: E402
from phase87_reader_stack_calibration import choice_templates, option_letters, render_options  # noqa: E402
from phase90_component_margin_reader_alignment import build_items, uniq  # noqa: E402
from phase92_cross_item_component_transplant import select_donors  # noqa: E402


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


def module_name(component: str) -> str:
    return "attn_out" if component == "attn" else "mlp_out"


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def pool_tensor(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    x = tensor[0].float()
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "tail":
        return x[-1]
    if mode == "prefix":
        n = max(1, min(8, x.shape[0]))
        return x[:n].mean(dim=0)
    raise ValueError(f"unknown pool mode={mode}")


def capture_output(
    model: Any,
    device: torch.device,
    layers: list[Any],
    text_ids: list[int],
    layer_idx: int,
    component: str,
) -> torch.Tensor:
    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        captured["h"] = hs.detach().float().cpu()

    handle = get_module(layers[layer_idx], module_name(component)).register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["h"]


def orthonormal_basis(mat: torch.Tensor, rank: int) -> torch.Tensor:
    if mat.numel() == 0 or mat.shape[0] == 0:
        return torch.empty((mat.shape[-1] if mat.ndim == 2 else 0, 0), dtype=torch.float32)
    x = mat.float()
    x = x - x.mean(dim=0, keepdim=True)
    keep = torch.isfinite(x).all(dim=1)
    x = x[keep]
    if x.shape[0] == 0:
        return torch.empty((mat.shape[-1], 0), dtype=torch.float32)
    try:
        _u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    except Exception:
        return torch.empty((mat.shape[-1], 0), dtype=torch.float32)
    k = max(0, min(rank, vh.shape[0]))
    return vh[:k].T.contiguous().float()


def project(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[1] == 0:
        return torch.zeros_like(x)
    b = basis.to(device=x.device, dtype=x.dtype)
    return (x @ b) @ b.T


def align_like(target: torch.Tensor, donor_cpu: torch.Tensor, copy_mode: str) -> torch.Tensor:
    donor = donor_cpu.to(device=target.device, dtype=target.dtype)
    if donor.shape == target.shape:
        return donor
    if donor.ndim != 3 or target.ndim != 3 or donor.shape[0] != target.shape[0] or donor.shape[2] != target.shape[2]:
        return target
    out = target.clone()
    n = min(int(donor.shape[1]), int(target.shape[1]))
    if n <= 0:
        return out
    if copy_mode == "tail":
        out[:, -n:, :] = donor[:, -n:, :]
    elif copy_mode == "prefix":
        out[:, :n, :] = donor[:, :n, :]
    elif copy_mode == "both":
        k = max(1, n // 2)
        out[:, :k, :] = donor[:, :k, :]
        out[:, -k:, :] = donor[:, -k:, :]
    else:
        raise ValueError(copy_mode)
    return out


def make_factor_hook(
    basis: torch.Tensor,
    op: str,
    donor_output: torch.Tensor | None,
    copy_mode: str,
):
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        if op == "destroy":
            patched = hs - project(hs, basis)
        elif op == "transplant":
            if donor_output is None:
                patched = hs
            else:
                donor = align_like(hs, donor_output, copy_mode)
                patched = hs - project(hs, basis) + project(donor, basis)
        else:
            raise ValueError(op)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    return hook_fn


def score_stats(scores: dict[str, float], target: str, candidates: list[str]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    competitors = [x for x in candidates if x != target]
    max_comp = max((scores[x] for x in competitors), default=-1e9)
    return {
        "top": ordered[0][0] if ordered else "",
        "top1": bool(ordered and ordered[0][0] == target),
        "margin": float(scores.get(target, -1e9) - max_comp),
    }


def fullseq_logprob_factor(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    donor_prompt: str | None,
    continuation: str,
    max_length: int,
    layer_idx: int,
    component: str,
    basis: torch.Tensor | None,
    op: str,
    copy_mode: str,
) -> float:
    prompt_ids = token_ids(tokenizer, prompt)
    cont_ids = token_ids(tokenizer, continuation)
    if not cont_ids:
        return float("-inf")
    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        return float("-inf")
    donor_output = None
    if op == "transplant":
        if donor_prompt is None:
            return float("-inf")
        donor_ids = token_ids(tokenizer, donor_prompt) + cont_ids
        if len(donor_ids) > max_length:
            return float("-inf")
        donor_output = capture_output(model, device, layers, donor_ids, layer_idx, component)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if basis is not None:
            handles.append(
                get_module(layers[layer_idx], module_name(component)).register_forward_hook(
                    make_factor_hook(basis, op, donor_output, copy_mode)
                )
            )
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
    finally:
        for handle in handles:
            handle.remove()
    start = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(cont_ids):
        pos = start + i - 1
        if pos < 0 or pos >= log_probs.shape[0]:
            return float("-inf")
        total += float(log_probs[pos, tok].detach().cpu())
    return total


def condition_score(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    item: dict[str, Any],
    donor_item: dict[str, Any] | None,
    candidates: list[str],
    letters: list[str],
    target_letter: str,
    choice_template: str,
    max_length: int,
    layer_idx: int,
    component: str,
    basis: torch.Tensor | None,
    op: str,
    copy_mode: str,
) -> dict[str, Any]:
    value_scores = {}
    for value in candidates:
        value_scores[value] = fullseq_logprob_factor(
            model, tokenizer, device, layers,
            item["prompt"],
            donor_item["prompt"] if donor_item else None,
            " " + value,
            max_length, layer_idx, component, basis, op, copy_mode,
        )
    choice_prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
    donor_choice_prompt = build_choice_prompt(choice_template, donor_item["prompt"], candidates) if donor_item else None
    letter_scores = {}
    for letter in letters:
        letter_scores[letter] = fullseq_logprob_factor(
            model, tokenizer, device, layers,
            choice_prompt, donor_choice_prompt, letter,
            max_length, layer_idx, component, basis, op, copy_mode,
        )
    vs = score_stats(value_scores, item["target"], candidates)
    ls = score_stats(letter_scores, target_letter, letters)
    return {
        "value_margin": vs["margin"],
        "letter_margin": ls["margin"],
        "value_top1": vs["top1"],
        "letter_top1": ls["top1"],
    }


def build_factor_bases(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    donors: dict[int, dict[str, int | None]],
    layer_idx: int,
    component: str,
    choice_template: str,
    rank: int,
    pool_mode: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    value_vecs: list[torch.Tensor] = []
    choice_vecs: list[torch.Tensor] = []
    for item in items:
        candidates = uniq([item["target"], *item["distractors"][:4]])
        letters = option_letters(len(candidates))
        target_letter = letters[candidates.index(item["target"])]
        value_ids = token_ids(tokenizer, item["prompt"]) + token_ids(tokenizer, " " + item["target"])
        choice_prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
        choice_ids = token_ids(tokenizer, choice_prompt) + token_ids(tokenizer, target_letter)
        if len(value_ids) <= max_length:
            value_vecs.append(pool_tensor(capture_output(model, device, layers, value_ids, layer_idx, component), pool_mode))
        if len(choice_ids) <= max_length:
            choice_vecs.append(pool_tensor(capture_output(model, device, layers, choice_ids, layer_idx, component), pool_mode))
    value_mat = torch.stack(value_vecs) if value_vecs else torch.empty((0, 0))
    choice_mat = torch.stack(choice_vecs) if choice_vecs else torch.empty((0, value_mat.shape[1] if value_mat.ndim == 2 else 0))

    def diffs(kind: str) -> torch.Tensor:
        rows = []
        for idx, item in enumerate(items):
            j = donors[idx].get(kind)
            if j is None or idx >= len(value_vecs) or j >= len(value_vecs):
                continue
            rows.append(value_vecs[idx] - value_vecs[j])
        if not rows:
            return torch.empty((0, value_mat.shape[1]))
        return torch.stack(rows)

    choice_rows = []
    for i in range(min(len(value_vecs), len(choice_vecs))):
        choice_rows.append(choice_vecs[i] - value_vecs[i])
    choice_diff = torch.stack(choice_rows) if choice_rows else torch.empty((0, value_mat.shape[1]))

    bases = {
        "pc1": orthonormal_basis(value_mat, 1),
        "slot": orthonormal_basis(diffs("diff_slot_same_object"), rank),
        "target": orthonormal_basis(diffs("same_slot_diff_target"), rank),
        "object": orthonormal_basis(diffs("same_slot_same_target"), rank),
        "choice": orthonormal_basis(choice_diff, rank),
    }
    return bases


def intervention_specs() -> list[tuple[str, str, str, str | None]]:
    return [
        ("destroy_pc1", "pc1", "destroy", None),
        ("destroy_slot", "slot", "destroy", None),
        ("destroy_target", "target", "destroy", None),
        ("destroy_object", "object", "destroy", None),
        ("destroy_choice", "choice", "destroy", None),
        ("trans_slot", "slot", "transplant", "same_slot_diff_target"),
        ("trans_target_same", "target", "transplant", "same_slot_same_target"),
        ("trans_target_diff", "target", "transplant", "same_slot_diff_target"),
        ("trans_object", "object", "transplant", "diff_slot_same_object"),
        ("trans_choice", "choice", "transplant", "same_slot_same_target"),
    ]


def run_item(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    donors: dict[int, dict[str, int | None]],
    bases: dict[str, torch.Tensor],
    item_idx: int,
    layer_idx: int,
    component: str,
    choice_template: str,
    max_distractors: int,
    max_length: int,
    copy_mode: str,
) -> list[dict[str, Any]]:
    item = items[item_idx]
    candidates = uniq([item["target"], *item["distractors"][:max_distractors]])
    letters = option_letters(len(candidates))
    target_letter = letters[candidates.index(item["target"])]
    clean = condition_score(
        model, tokenizer, device, layers, item, None, candidates, letters, target_letter,
        choice_template, max_length, layer_idx, component, None, "clean", copy_mode,
    )
    rows = []
    for condition, factor, op, donor_kind in intervention_specs():
        donor_item = None
        donor_idx = None
        if donor_kind:
            donor_idx = donors[item_idx].get(donor_kind)
            if donor_idx is None:
                continue
            donor_item = items[donor_idx]
        patched = condition_score(
            model, tokenizer, device, layers, item, donor_item, candidates, letters, target_letter,
            choice_template, max_length, layer_idx, component, bases[factor], op, copy_mode,
        )
        rows.append({
            "item_idx": item_idx,
            "donor_idx": donor_idx,
            "condition": condition,
            "factor": factor,
            "op": op,
            "donor_kind": donor_kind or "",
            "layer": layer_idx,
            "component": component,
            "slot": item["slot"],
            "template_key": item["template_key"],
            "object": item["object"],
            "target": item["target"],
            "copy_mode": copy_mode,
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
        })
    return rows


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
        "by_condition": defaultdict(list),
        "by_factor": defaultdict(list),
        "by_node_condition": defaultdict(list),
        "by_node_factor": defaultdict(list),
    }
    for row in rows:
        node = f"L{row['layer']}:{row['component']}"
        groups["by_node"][node].append(row)
        groups["by_condition"][row["condition"]].append(row)
        groups["by_factor"][row["factor"]].append(row)
        groups["by_node_condition"][(node, row["condition"])].append(row)
        groups["by_node_factor"][(node, row["factor"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE94_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    nodes = parse_csv(args.nodes)
    parsed_nodes = []
    for raw in nodes:
        layer, comp = raw.split(":", 1)
        parsed_nodes.append((int(layer), comp))
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors = select_donors(items)
    log(f"Phase94 model={args.model} items={len(items)} nodes={parsed_nodes} rank={args.rank} copy_mode={args.copy_mode}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase94_factor_subspace_closure.json"
    partial_path = out_dir / f"{args.model}_phase94_factor_subspace_closure.partial.json"
    results: dict[str, Any] = {
        "phase": 94,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "factor_subspace_closure",
        "nodes": [f"{l}:{c}" for l, c in parsed_nodes],
        "num_items": len(items),
        "rank": args.rank,
        "pool_mode": args.pool_mode,
        "copy_mode": args.copy_mode,
        "rows": [],
        "basis_dims": {},
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 94 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results.setdefault("basis_dims", {})
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed = {(int(r["layer"]), r["component"], int(r["item_idx"]), r["condition"]) for r in results["rows"]}
    t0 = time.time()
    for layer_idx, component in parsed_nodes:
        log(f"building bases node=L{layer_idx}:{component}")
        bases = build_factor_bases(
            model, tokenizer, device, layers, items, donors, layer_idx, component,
            args.choice_template, args.rank, args.pool_mode, args.max_length,
        )
        results["basis_dims"][f"L{layer_idx}:{component}"] = {k: int(v.shape[1]) for k, v in bases.items()}
        partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        for idx, _item in enumerate(items):
            pending = [
                spec[0] for spec in intervention_specs()
                if (layer_idx, component, idx, spec[0]) not in completed
            ]
            if not pending:
                continue
            item_rows = run_item(
                model, tokenizer, device, layers, items, donors, bases, idx,
                layer_idx, component, args.choice_template, args.max_distractors,
                args.max_length, args.copy_mode,
            )
            for row in item_rows:
                key = (int(row["layer"]), row["component"], int(row["item_idx"]), row["condition"])
                if key not in completed:
                    results["rows"].append(row)
                    completed.add(key)
            if (idx + 1) % args.progress_every == 0:
                log(f"node=L{layer_idx}:{component} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
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
    parser.add_argument("--max-items", type=int, default=420)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pool-mode", default="tail")
    parser.add_argument("--copy-mode", choices=["tail", "prefix", "both"], default="both")
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
