#!/usr/bin/env python3
"""
Phase 618: Attention Source and Pattern/Content Decomposition
注意力 source 与 pattern/content 分解

Phase 617 localized DS7B value-gate repair to multi-layer attention head slots.
This phase asks what those heads read: source groups and pattern/content roles.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_layers, get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import random_same_norm  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, n_heads_for, parse_layers  # noqa: E402
from phase610_head_cumulative_mixture import TOP_HEADS  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402


OUT_ROOT = Path("results/glm5_phase618_attention_source_pattern_content")
SOURCE_GROUPS = [
    "self_answer",
    "question_line",
    "final_object_category_line",
    "object_rule_lines",
    "value_rule_lines",
    "punct_format",
    "other",
    "all_source",
]
MODES = ["rb_pattern", "br_content", "rr_pattern_content"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def full_ids(tokenizer, prompt: str, answer: str) -> List[int]:
    return tokenizer.encode(prompt, add_special_tokens=False) + answer_ids(tokenizer, answer)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def default_layers(model: str, n_layers: int) -> List[int]:
    if model == "qwen3":
        return [li for li in [27, 28, 29] if li < n_layers]
    if model == "glm4":
        return [li for li in [32, 33, 34] if li < n_layers]
    if model == "deepseek7b":
        return [li for li in [20, 21, 22] if li < n_layers]
    return list(range(max(0, n_layers - 4), n_layers))


def top_heads(model: str, n_heads: int, k: int) -> List[int]:
    heads = [h for h in TOP_HEADS.get(model, []) if 0 <= h < n_heads]
    if not heads:
        heads = list(range(min(k, n_heads)))
    return heads[:k]


def offset_tokens(tokenizer, prompt: str) -> List[Tuple[int, int]]:
    try:
        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping")
        if offsets is not None:
            return [(int(a), int(b)) for a, b in offsets]
    except Exception:
        pass

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    offsets = []
    cursor = 0
    for i in range(len(ids)):
        piece = tokenizer.decode([ids[i]], skip_special_tokens=False)
        clean = piece.replace("Ġ", " ").replace("▁", " ")
        idx = prompt.find(clean, cursor)
        if idx < 0:
            idx = cursor
        offsets.append((idx, min(len(prompt), idx + len(clean))))
        cursor = offsets[-1][1]
    return offsets


def line_spans(prompt: str) -> List[Tuple[str, int, int]]:
    out = []
    start = 0
    for line in prompt.splitlines(keepends=True):
        end = start + len(line)
        out.append((line.rstrip("\n"), start, end))
        start = end
    return out


def token_source_groups(tokenizer, prompt: str, answer_pos: int) -> Dict[str, List[int]]:
    offsets = offset_tokens(tokenizer, prompt)
    spans = line_spans(prompt)
    char_group = {}
    in_rules = False
    for line, start, end in spans:
        stripped = line.strip()
        if stripped == "Rules:":
            in_rules = True
            continue
        if stripped.startswith("Question:"):
            in_rules = False
            for i in range(start, end):
                char_group[i] = "question_line"
            continue
        if stripped.startswith("Answer:"):
            in_rules = False
            for i in range(start, end):
                char_group[i] = "punct_format"
            continue
        if not stripped:
            continue
        if in_rules and " belongs to " in stripped:
            for i in range(start, end):
                char_group[i] = "object_rule_lines"
        elif in_rules:
            for i in range(start, end):
                char_group[i] = "value_rule_lines"
        elif " belongs to " in stripped:
            for i in range(start, end):
                char_group[i] = "final_object_category_line"

    groups = {name: [] for name in SOURCE_GROUPS}
    punctuation = set(string.punctuation)
    for ti, (a, b) in enumerate(offsets):
        if ti >= answer_pos:
            continue
        text = prompt[a:b]
        if text and all((ch.isspace() or ch in punctuation) for ch in text):
            groups["punct_format"].append(ti)
            continue
        label = None
        for ci in range(a, max(a + 1, b)):
            label = char_group.get(ci)
            if label:
                break
        if label is None:
            label = "other"
        groups[label].append(ti)
    groups["self_answer"] = [answer_pos]
    groups["all_source"] = list(range(answer_pos + 1))
    return groups


def repeat_v_to_heads(v_proj: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    seq, dim = v_proj.shape
    n_kv = max(1, dim // head_dim)
    v = v_proj[:, : n_kv * head_dim].reshape(seq, n_kv, head_dim).transpose(0, 1).contiguous()
    if n_kv < n_heads:
        rep = max(1, n_heads // n_kv)
        v = v.unsqueeze(1).expand(n_kv, rep, seq, head_dim).reshape(n_kv * rep, seq, head_dim)
        if v.shape[0] < n_heads:
            v = torch.cat([v, v[-1:].expand(n_heads - v.shape[0], seq, head_dim)], dim=0)
        v = v[:n_heads]
    elif n_kv > n_heads:
        v = v[:n_heads]
    return v.float().cpu()


def collect_parts_multi(model, tokenizer, device, prompt: str, answer: str, layers_to_scan: List[int]) -> Dict[int, Dict]:
    layers = get_layers(model)
    pos = answer_prefix_pos(tokenizer, prompt)
    captured: Dict[int, Dict[str, torch.Tensor]] = {li: {} for li in layers_to_scan}
    handles = []

    for li in layers_to_scan:
        attn = layers[li].self_attn

        def make_v_hook(layer_idx):
            def hook(_module, _inputs, output):
                captured[layer_idx]["v_proj"] = output[0].detach().float().cpu()
            return hook

        def make_o_pre_hook(layer_idx):
            def hook(_module, inputs):
                x = inputs[0]
                if pos < x.shape[1]:
                    captured[layer_idx]["o_input"] = x[0, pos].detach().float().cpu()
            return hook

        handles.append(attn.v_proj.register_forward_hook(make_v_hook(li)))
        handles.append(attn.o_proj.register_forward_pre_hook(make_o_pre_hook(li)))

    try:
        ids = full_ids(tokenizer, prompt, answer)
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                output_attentions=True,
                return_dict=True,
            )
        if out.attentions is None:
            raise RuntimeError("output_attentions did not return attention weights")
        for li in layers_to_scan:
            captured[li]["alpha"] = out.attentions[li][0, :, pos, :].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return captured


def grouped_z(parts: Dict, heads: List[int], n_heads: int, group_tokens: List[int], source: str) -> Dict[int, torch.Tensor]:
    o = parts["o_input"].float().cpu()
    width = o.numel()
    head_dim = width // max(1, n_heads)
    v_heads = repeat_v_to_heads(parts["v_proj"], n_heads, head_dim)
    alpha = parts["alpha"].float().cpu()
    max_seq = min(alpha.shape[-1], v_heads.shape[1])
    toks = [t for t in group_tokens if 0 <= t < max_seq]
    out = {}
    for hi in heads:
        start = hi * head_dim
        end = width if hi == n_heads - 1 else (hi + 1) * head_dim
        dim = end - start
        if not toks:
            out[hi] = torch.zeros(dim)
            continue
        a = alpha[hi, toks].float()
        v = v_heads[hi, toks, :dim].float()
        if source == "base":
            out[hi] = (a[:, None] * v).sum(dim=0).float().cpu()
        else:
            out[hi] = (a[:, None] * v).sum(dim=0).float().cpu()
    return out


def slot_deltas_for_mode(
    base_parts: Dict,
    repair_parts: Dict,
    heads: List[int],
    n_heads: int,
    group_tokens: List[int],
    mode: str,
) -> Dict[int, torch.Tensor]:
    width = base_parts["o_input"].numel()
    head_dim = width // max(1, n_heads)
    base_v = repeat_v_to_heads(base_parts["v_proj"], n_heads, head_dim)
    repair_v = repeat_v_to_heads(repair_parts["v_proj"], n_heads, head_dim)
    base_alpha = base_parts["alpha"].float().cpu()
    repair_alpha = repair_parts["alpha"].float().cpu()
    max_seq = min(base_alpha.shape[-1], repair_alpha.shape[-1], base_v.shape[1], repair_v.shape[1])
    toks = [t for t in group_tokens if 0 <= t < max_seq]
    deltas = {}
    for hi in heads:
        start = hi * head_dim
        end = width if hi == n_heads - 1 else (hi + 1) * head_dim
        dim = end - start
        if not toks:
            deltas[hi] = torch.zeros(dim)
            continue
        bb = (base_alpha[hi, toks, None] * base_v[hi, toks, :dim]).sum(dim=0)
        if mode == "rb_pattern":
            target = (repair_alpha[hi, toks, None] * base_v[hi, toks, :dim]).sum(dim=0)
        elif mode == "br_content":
            target = (base_alpha[hi, toks, None] * repair_v[hi, toks, :dim]).sum(dim=0)
        elif mode == "rr_pattern_content":
            target = (repair_alpha[hi, toks, None] * repair_v[hi, toks, :dim]).sum(dim=0)
        else:
            raise ValueError(mode)
        deltas[hi] = (target - bb).float().cpu()
    return deltas


def make_specs(model_name: str, layers_to_scan: List[int], heads_by_layer: Dict[int, int], top_k: int) -> List[Dict]:
    min_heads = min(heads_by_layer.values())
    heads = top_heads(model_name, min_heads, top_k)
    specs = []
    for group in SOURCE_GROUPS:
        for mode in MODES:
            specs.append({
                "name": f"top{len(heads)}_midlate_{group}_{mode}",
                "group": group,
                "mode": mode,
                "ops": [{"layer": li, "heads": heads} for li in layers_to_scan],
            })
    for li in layers_to_scan:
        for h in heads[: min(3, len(heads))]:
            for group in ["all_source", "question_line", "final_object_category_line", "value_rule_lines"]:
                specs.append({
                    "name": f"L{li}_H{h}_{group}_rr",
                    "group": group,
                    "mode": "rr_pattern_content",
                    "ops": [{"layer": li, "heads": [h]}],
                })
    return specs


def patch_answer_score(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    spec: Dict,
    cache: Dict[str, Dict],
    heads_by_layer: Dict[int, int],
    group_tokens: Dict[str, List[int]],
    random_mode: bool,
    seed: int,
) -> float:
    layers = get_layers(model)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = answer_ids(tokenizer, answer)
    ids = prompt_ids + ans_ids
    if not ans_ids:
        return -100.0
    pos = len(prompt_ids)
    handles = []
    group = spec["group"]
    mode = spec["mode"]

    for op_idx, op in enumerate(spec["ops"]):
        li = op["layer"]
        heads = op["heads"]
        attn = layers[li].self_attn
        n_heads = heads_by_layer[li]
        base_parts = cache[answer]["base"][li]
        repair_parts = cache[answer]["repair"][li]
        width = base_parts["o_input"].numel()
        head_dim = width // max(1, n_heads)
        deltas = slot_deltas_for_mode(base_parts, repair_parts, heads, n_heads, group_tokens[group], mode)
        if random_mode:
            deltas = {
                hi: random_same_norm(d, seed=seed + op_idx * 1009 + li * 101 + hi)
                for hi, d in deltas.items()
            }

        def hook(_module, inputs, deltas=deltas, n_heads=n_heads, width=width):
            x = inputs[0]
            x_new = x.clone()
            if pos < x_new.shape[1]:
                head_dim = width // max(1, n_heads)
                for hi, d in deltas.items():
                    start = hi * head_dim
                    end = width if hi == n_heads - 1 else (hi + 1) * head_dim
                    x_new[0, pos, start:end] = x_new[0, pos, start:end] + d.to(
                        device=x_new.device, dtype=x_new.dtype
                    )
            return (x_new,) + tuple(inputs[1:])

        handles.append(attn.o_proj.register_forward_pre_hook(hook))

    try:
        total = 0.0
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0].float()
            start_pos = len(prompt_ids) - 1
            for i, tid in enumerate(ans_ids):
                p = start_pos + i
                if p >= logits.shape[0]:
                    break
                total += float(torch.log_softmax(logits[p], dim=-1)[tid].cpu())
        return total
    finally:
        for h in handles:
            h.remove()


def patched_scores(model, tokenizer, device, case: Dict, values: List[str], spec: Dict,
                   cache: Dict[str, Dict], heads_by_layer: Dict[int, int],
                   group_tokens: Dict[str, List[int]], random_mode: bool, seed: int) -> Dict[str, float]:
    scores = {}
    for ai, ans in enumerate(values):
        scores[ans] = patch_answer_score(
            model, tokenizer, device, case["base_prompt"], ans, spec, cache,
            heads_by_layer, group_tokens, random_mode=random_mode, seed=seed + ai * 997,
        )
    return scores


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["patches"]})
    by_patch = {}
    for key in keys:
        items = [r["patches"][key] for r in rows if key in r["patches"]]
        entry = {
            "key": key,
            "name": items[0]["name"],
            "group": items[0]["group"],
            "mode": items[0]["mode"],
            "random": items[0]["random"],
            "n_ops": len(items[0]["ops"]),
            "n_slots": sum(len(op["heads"]) for op in items[0]["ops"]),
            "ops": items[0]["ops"],
            "n": len(items),
            "switch": 0,
            "mean_margin_gain": 0.0,
            "mean_correct_delta": 0.0,
            "mean_wrong_delta": 0.0,
            "positive_margin": 0,
        }
        for item in items:
            entry["switch"] += int(item["winner"]["correct"])
            m = item["metric"]
            entry["mean_margin_gain"] += m["margin_gain"]
            entry["mean_correct_delta"] += m["correct_delta"]
            entry["mean_wrong_delta"] += m["old_top_wrong_delta"]
            entry["positive_margin"] += int(m["margin_gain"] > 0)
        n = max(1, len(items))
        for name in list(entry):
            if name.startswith("mean_"):
                entry[name] /= n
        entry["switch_rate"] = entry["switch"] / n
        entry["positive_margin_rate"] = entry["positive_margin"] / n
        by_patch[key] = entry
    best = sorted(by_patch.values(), key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
    log("Best source pattern/content patches:")
    for item in best[:24]:
        flag = "random" if item["random"] else "real"
        log(
            f"  {item['name']} {flag}: switch={item['switch']}/{item['n']} "
            f"margin={item['mean_margin_gain']:.3f} slots={item['n_slots']}"
        )
    return {"by_patch": by_patch, "best": best[:200]}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers_to_scan = parse_layers(args.layers) if args.layers else default_layers(args.model, info.n_layers)
        layers_to_scan = [li for li in layers_to_scan if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in layers_to_scan}
        specs = make_specs(args.model, layers_to_scan, heads_by_layer, args.top_k)
        if args.compact:
            specs = [
                s for s in specs
                if s["group"] in ["all_source", "question_line", "final_object_category_line", "value_rule_lines"]
                or s["name"].startswith("L")
            ]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0}
        target_seen = 0
        log(
            f"{args.model}: layers={info.n_layers}, scan_layers={layers_to_scan}, "
            f"heads={heads_by_layer}, specs={len(specs)}, raw_cases={len(raw_cases)}"
        )

        for si, case in enumerate(raw_cases):
            base_len = len(tokenizer.encode(case["base_prompt"], add_special_tokens=False))
            repair_len = len(tokenizer.encode(case["repair_prompt"], add_special_tokens=False))
            if base_len != repair_len:
                filtered["token_len_mismatch"] += 1
                continue
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, case["correct"])
            repair = winner_stats(repair_scores, case["correct"])
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                filtered["not_target"] += 1
                continue
            target_seen += int(target_case)
            old_top_wrong = base["top_wrong"]
            group_tokens = token_source_groups(tokenizer, case["base_prompt"], base_len)

            cache: Dict[str, Dict] = {}
            for ans in values:
                if len(full_ids(tokenizer, case["base_prompt"], ans)) != len(full_ids(tokenizer, case["repair_prompt"], ans)):
                    raise RuntimeError("Full prompt+answer length mismatch after prompt alignment")
                cache[ans] = {
                    "base": collect_parts_multi(model, tokenizer, device, case["base_prompt"], ans, layers_to_scan),
                    "repair": collect_parts_multi(model, tokenizer, device, case["repair_prompt"], ans, layers_to_scan),
                }

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "base_prompt_len": base_len,
                "target_case": target_case,
                "group_sizes": {k: len(v) for k, v in group_tokens.items()},
                "base": base,
                "repair_prompt": repair,
                "repair_metric": candidate_delta_metric(base_scores, repair_scores, case["correct"], old_top_wrong),
                "patches": {},
            }
            for spec in specs:
                for random_mode in [False, True]:
                    suffix = "random" if random_mode else "real"
                    key = f"{spec['name']}|{suffix}"
                    scores = patched_scores(
                        model, tokenizer, device, case, values, spec, cache,
                        heads_by_layer, group_tokens, random_mode=random_mode,
                        seed=si * 1009 + len(spec["name"]),
                    )
                    patched = winner_stats(scores, case["correct"])
                    row["patches"][key] = {
                        "name": spec["name"],
                        "group": spec["group"],
                        "mode": spec["mode"],
                        "ops": spec["ops"],
                        "random": random_mode,
                        "winner": patched,
                        "metric": candidate_delta_metric(base_scores, scores, case["correct"], old_top_wrong),
                    }
            rows.append(row)

        return {
            "phase": 618,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "top_k": args.top_k,
            "top_heads": top_heads(args.model, min(heads_by_layer.values()), args.top_k),
            "source_groups": SOURCE_GROUPS,
            "modes": MODES,
            "n_specs": len(specs),
            "compact": args.compact,
            "n_raw_cases": len(raw_cases),
            "n_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "summary": summarize(rows),
            "rows": rows,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--layers", default="")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_k = min(args.top_k, 2)
        if not args.layers:
            layers = default_layers(args.model, 40 if args.model == "glm4" else 36 if args.model == "qwen3" else 28)
            args.layers = str(layers[-1])
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 16)
        args.max_samples = max(args.max_samples, 128)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase618_{args.model}_attention_source_pattern_content_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
