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
from phase68_object_attribute_natural_exchange import load_model, parse_csv  # noqa: E402
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


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    except Exception:
        return token_ids(tokenizer, text), []


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def local_positions(tokenizer: Any, prompt: str, position_kind: str) -> list[int]:
    ids = token_ids(tokenizer, prompt)
    if not ids:
        return []
    if position_kind == "prompt_tail":
        return [len(ids) - 1]
    if position_kind == "last4":
        return list(range(max(0, len(ids) - 4), len(ids)))
    raise ValueError(position_kind)


def attention_meta(layer: Any, model: Any) -> tuple[int, int]:
    sa = layer.self_attn
    n_heads = int(getattr(sa, "num_heads", getattr(model.config, "num_attention_heads")))
    in_features = int(sa.o_proj.in_features)
    if in_features % n_heads != 0:
        raise ValueError(f"o_proj.in_features={in_features} not divisible by heads={n_heads}")
    return n_heads, in_features // n_heads


def parse_head_sets(text: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        name, value = raw.split("=", 1)
        out[name.strip()] = [int(x) for x in parse_csv(value)]
    if not out:
        raise ValueError("--head-sets is empty")
    return out


def capture_oproj_input(
    model: Any,
    device: torch.device,
    layers: list[Any],
    text_ids: list[int],
    layer_idx: int,
) -> torch.Tensor:
    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    captured: dict[str, torch.Tensor] = {}
    sa = layers[layer_idx].self_attn

    def pre_hook(_module: Any, inputs: Any):
        captured["h"] = inputs[0].detach().float().cpu()
        return None

    handle = sa.o_proj.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["h"]


def make_restore_pre_hook(
    condition: str,
    head_indices: list[int],
    head_dim: int,
    target_positions: list[int],
    clean_input: torch.Tensor | None,
    donor_input: torch.Tensor | None,
    donor_positions: list[int],
):
    spans = [(h * head_dim, (h + 1) * head_dim) for h in head_indices]

    def pre_hook(_module: Any, inputs: Any):
        x = inputs[0]
        patched = x.clone()
        tpos = [p for p in target_positions if 0 <= p < patched.shape[1]]
        if not tpos:
            return inputs
        clean = clean_input.to(device=patched.device, dtype=patched.dtype) if clean_input is not None else None
        donor = donor_input.to(device=patched.device, dtype=patched.dtype) if donor_input is not None else None
        dpos = [p for p in donor_positions if donor is not None and 0 <= p < donor.shape[1]]

        def copy_from(src: torch.Tensor, start: int | None = None, end: int | None = None):
            block = src[:, tpos, :] if start is None else src[:, tpos, start:end]
            if start is None:
                patched[:, tpos, :] = block
            else:
                patched[:, tpos, start:end] = block

        def copy_donor_all():
            if donor is None or not dpos:
                return
            block = donor[:, dpos, :]
            patched[:, tpos, :] = block if len(dpos) == len(tpos) else block.mean(dim=1, keepdim=True).expand(-1, len(tpos), -1)

        def copy_donor_span(start: int, end: int):
            if donor is None or not dpos:
                return
            block = donor[:, dpos, start:end]
            patched[:, tpos, start:end] = block if len(dpos) == len(tpos) else block.mean(dim=1, keepdim=True).expand(-1, len(tpos), -1)

        if condition == "zero_all":
            patched[:, tpos, :] = 0
        elif condition == "transplant_all":
            copy_donor_all()
        elif condition.startswith("zero_all_restore_clean_heads"):
            patched[:, tpos, :] = 0
            if clean is not None:
                for start, end in spans:
                    copy_from(clean, start, end)
        elif condition.startswith("transplant_all_restore_clean_heads"):
            copy_donor_all()
            if clean is not None:
                for start, end in spans:
                    copy_from(clean, start, end)
        elif condition.startswith("transplant_heads"):
            for start, end in spans:
                copy_donor_span(start, end)
        elif condition.startswith("zero_heads"):
            for start, end in spans:
                patched[:, tpos, start:end] = 0
        else:
            raise ValueError(condition)
        return (patched,) + tuple(inputs[1:])

    return pre_hook


def fullseq_logprob(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    donor_prompt: str | None,
    continuation: str,
    max_length: int,
    layer_idx: int,
    condition: str,
    head_indices: list[int],
    head_dim: int,
    position_kind: str,
) -> float:
    prompt_ids = token_ids(tokenizer, prompt)
    cont_ids = token_ids(tokenizer, continuation)
    if not cont_ids:
        return float("-inf")
    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        return float("-inf")
    target_positions = local_positions(tokenizer, prompt, position_kind)
    clean_input = None
    donor_input = None
    donor_positions: list[int] = []
    if condition != "clean":
        clean_input = capture_oproj_input(model, device, layers, full_ids, layer_idx)
    if "transplant" in condition:
        if donor_prompt is None:
            return float("-inf")
        donor_ids = token_ids(tokenizer, donor_prompt) + cont_ids
        if len(donor_ids) > max_length:
            return float("-inf")
        donor_input = capture_oproj_input(model, device, layers, donor_ids, layer_idx)
        donor_positions = local_positions(tokenizer, donor_prompt, position_kind)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if condition != "clean":
            handles.append(
                layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                    make_restore_pre_hook(condition, head_indices, head_dim, target_positions, clean_input, donor_input, donor_positions)
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


def score_stats(scores: dict[str, float], target: str, candidates: list[str]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    competitors = [x for x in candidates if x != target]
    max_comp = max((scores[x] for x in competitors), default=-1e9)
    return {"top1": bool(ordered and ordered[0][0] == target), "margin": float(scores.get(target, -1e9) - max_comp)}


def joint_score(
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
    condition: str,
    head_indices: list[int],
    head_dim: int,
    position_kind: str,
) -> dict[str, Any]:
    value_scores = {}
    for value in candidates:
        value_scores[value] = fullseq_logprob(
            model, tokenizer, device, layers, item["prompt"], donor_item["prompt"] if donor_item else None, value,
            max_length, layer_idx, condition, head_indices, head_dim, position_kind,
        )
    letter_prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
    donor_letter_prompt = build_choice_prompt(choice_template, donor_item["prompt"], candidates) if donor_item else None
    letter_scores = {}
    for letter in letters:
        letter_scores[letter] = fullseq_logprob(
            model, tokenizer, device, layers, letter_prompt, donor_letter_prompt, letter,
            max_length, layer_idx, condition, head_indices, head_dim, position_kind,
        )
    vs = score_stats(value_scores, item["target"], candidates)
    ls = score_stats(letter_scores, target_letter, letters)
    return {"value_margin": vs["margin"], "value_top1": vs["top1"], "letter_margin": ls["margin"], "letter_top1": ls["top1"]}


def build_conditions(head_sets: dict[str, list[int]]) -> list[tuple[str, str, list[int], bool]]:
    out: list[tuple[str, str, list[int], bool]] = [("zero_all", "all", [], False), ("transplant_all", "all", [], True)]
    for name, heads in head_sets.items():
        out.append((f"zero_heads:{name}", name, heads, False))
        out.append((f"transplant_heads:{name}", name, heads, True))
        out.append((f"zero_all_restore_clean_heads:{name}", name, heads, False))
        out.append((f"transplant_all_restore_clean_heads:{name}", name, heads, True))
    return out


def char_span_indices(tokenizer: Any, text: str, start: int, end: int) -> set[int]:
    ids, offs = offsets(tokenizer, text)
    if not ids or not offs:
        return set()
    return {i for i, (a, b) in enumerate(offs) if b > start and a < end}


def source_labels(tokenizer: Any, prompt: str, item: dict[str, Any], candidates: list[str], letters: list[str]) -> list[str]:
    ids, offs = offsets(tokenizer, prompt)
    labels = ["other"] * len(ids)
    if not ids:
        return labels
    clean_start = prompt.find(item["prompt"])
    clean_end = clean_start + len(item["prompt"]) if clean_start >= 0 else -1
    obj_start = prompt.find(item["object"], clean_start if clean_start >= 0 else 0)
    obj_end = obj_start + len(item["object"]) if obj_start >= 0 else -1
    return_start = prompt.find("Return JSON")
    option_start = prompt.find("Options:")
    target = item["target"]
    for i, (a, b) in enumerate(offs):
        frag = prompt[a:b]
        if i >= len(ids) - 4:
            labels[i] = "readout_tail"
        elif obj_start >= 0 and b > obj_start and a < obj_end:
            labels[i] = "object"
        elif clean_start >= 0 and b > clean_start and a < clean_end:
            labels[i] = "clean_relation"
        elif return_start >= 0 and b > return_start:
            labels[i] = "format_return"
        elif frag.strip(". ") in letters:
            labels[i] = "letter_label"
        elif any((cand in frag or (prompt.find(cand) >= 0 and b > prompt.find(cand) and a < prompt.find(cand) + len(cand))) for cand in [target]):
            labels[i] = "target_option"
        elif option_start >= 0 and b > option_start:
            labels[i] = "distractor_option"
    return labels


def attention_source_summary(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer_idx: int,
    items: list[dict[str, Any]],
    head_sets: dict[str, list[int]],
    choice_template: str,
    max_items: int,
    max_distractors: int,
) -> dict[str, Any]:
    if max_items <= 0:
        return {}
    acc: dict[str, list[float]] = defaultdict(list)
    n = 0
    for item in items[:max_items]:
        candidates = uniq([item["target"], *item["distractors"][:max_distractors]])
        prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
        ids = token_ids(tokenizer, prompt)
        if not ids:
            continue
        labels = source_labels(tokenizer, prompt, item, candidates, option_letters(len(candidates)))
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, use_cache=False)
            attn = out.attentions[layer_idx].detach().float().cpu()[0]
        except Exception as exc:
            return {"error": str(exc)}
        tpos = [len(ids) - 1]
        for set_name, heads in head_sets.items():
            for head in heads:
                weights = attn[head, tpos, :].mean(dim=0)
                by_label: dict[str, float] = defaultdict(float)
                for idx, label in enumerate(labels):
                    by_label[label] += float(weights[idx])
                for label, value in by_label.items():
                    acc[f"{set_name}:H{head}:{label}"].append(value)
        n += 1
        cleanup_cuda()
    summary: dict[str, Any] = {"items": n, "by_head_label": {}}
    for key, vals in acc.items():
        summary["by_head_label"][key] = round(avg(vals), 6)
    return summary


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
        "by_condition": defaultdict(list),
        "by_head_set": defaultdict(list),
        "by_position": defaultdict(list),
        "by_position_condition": defaultdict(list),
    }
    for row in rows:
        groups["by_condition"][row["condition"]].append(row)
        groups["by_head_set"][row["head_set_name"]].append(row)
        groups["by_position"][row["position_kind"]].append(row)
        groups["by_position_condition"][(row["position_kind"], row["condition"])].append(row)
    return {key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()} for key, group in groups.items()}


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE100_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_idx = int(args.layer)
    n_heads, head_dim = attention_meta(layers[layer_idx], model)
    head_sets = parse_head_sets(args.head_sets)
    for name, heads in head_sets.items():
        bad = [h for h in heads if h < 0 or h >= n_heads]
        if bad:
            raise ValueError(f"{args.model} head set {name} invalid heads {bad}; n_heads={n_heads}")
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors = select_donors(items)
    positions = parse_csv(args.positions)
    conditions = build_conditions(head_sets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase100_head_route_restore_source.json"
    partial_path = out_dir / f"{args.model}_phase100_head_route_restore_source.partial.json"
    results: dict[str, Any] = {
        "phase": 100,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "head_route_restore_source",
        "layer": layer_idx,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "head_sets": head_sets,
        "num_items": len(items),
        "positions": positions,
        "rows": [],
        "summary": {},
        "source_attention": {},
    }
    if args.resume or args.source_only:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 100 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    if args.source_only:
        log(f"Phase100 source-only model={args.model} layer=L{layer_idx} source_items={args.source_attn_items}")
        results["source_attention"] = attention_source_summary(
            model, tokenizer, device, layer_idx, items, head_sets, args.choice_template, args.source_attn_items, args.max_distractors
        )
        results["summary"] = summarize(results.get("rows", []))
        final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Wrote {final_path}")
        return results
    completed = {(int(r["item_idx"]), r["position_kind"], r["condition"], r["head_set_name"]) for r in results["rows"]}
    clean_cache: dict[int, dict[str, Any]] = {}
    t0 = time.time()
    log(f"Phase100 model={args.model} layer=L{layer_idx} items={len(items)} positions={positions} head_sets={head_sets}")
    clean_heads = next(iter(head_sets.values()))
    for idx, item in enumerate(items):
        candidates = uniq([item["target"], *item["distractors"][: args.max_distractors]])
        letters = option_letters(len(candidates))
        target_letter = letters[candidates.index(item["target"])]
        clean_cache[idx] = {
            "candidates": candidates,
            "letters": letters,
            "target_letter": target_letter,
            "score": joint_score(
                model, tokenizer, device, layers, item, None, candidates, letters, target_letter,
                args.choice_template, args.max_length, layer_idx, "clean", clean_heads, head_dim, "prompt_tail",
            ),
        }
    for position_kind in positions:
        for idx, item in enumerate(items):
            cache = clean_cache[idx]
            donor_idx = donors[idx].get(args.donor_kind)
            donor_item = items[donor_idx] if donor_idx is not None else None
            for condition, head_set_name, heads, needs_donor in conditions:
                key = (idx, position_kind, condition, head_set_name)
                if key in completed:
                    continue
                if needs_donor and donor_item is None:
                    continue
                patched = joint_score(
                    model, tokenizer, device, layers, item, donor_item, cache["candidates"], cache["letters"],
                    cache["target_letter"], args.choice_template, args.max_length, layer_idx, condition,
                    heads, head_dim, position_kind,
                )
                clean = cache["score"]
                row = {
                    "item_idx": idx,
                    "donor_idx": donor_idx,
                    "position_kind": position_kind,
                    "condition": condition,
                    "head_set_name": head_set_name,
                    "heads": heads,
                    "layer": layer_idx,
                    "slot": item["slot"],
                    "template_key": item["template_key"],
                    "object": item["object"],
                    "target": item["target"],
                    "clean_value_margin": clean["value_margin"],
                    "patched_value_margin": patched["value_margin"],
                    "value_delta": patched["value_margin"] - clean["value_margin"],
                    "clean_letter_margin": clean["letter_margin"],
                    "patched_letter_margin": patched["letter_margin"],
                    "letter_delta": patched["letter_margin"] - clean["letter_margin"],
                    "clean_value_top1": clean["value_top1"],
                    "patched_value_top1": patched["value_top1"],
                    "value_top1_delta": float(patched["value_top1"]) - float(clean["value_top1"]),
                    "clean_letter_top1": clean["letter_top1"],
                    "patched_letter_top1": patched["letter_top1"],
                    "letter_top1_delta": float(patched["letter_top1"]) - float(clean["letter_top1"]),
                }
                results["rows"].append(row)
                completed.add(key)
            if (idx + 1) % args.progress_every == 0:
                log(f"model={args.model} pos={position_kind} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
                partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cuda()
    results["summary"] = summarize(results["rows"])
    results["source_attention"] = attention_source_summary(
        model, tokenizer, device, layer_idx, items, head_sets, args.choice_template, args.source_attn_items, args.max_distractors
    )
    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {final_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer", required=True)
    parser.add_argument("--head-sets", required=True)
    parser.add_argument("--slots", default="category,color,function,material,location")
    parser.add_argument("--slot-templates", default="")
    parser.add_argument("--max-items", type=int, default=210)
    parser.add_argument("--source-attn-items", type=int, default=60)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--positions", default="prompt_tail,last4")
    parser.add_argument("--donor-kind", default="same_slot_diff_target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=35)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-only", action="store_true")
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
