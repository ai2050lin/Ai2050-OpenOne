#!/usr/bin/env python3
"""
Phase 619: Rule-Line Token Micro Atlas
值规则行 token 微图谱

Phase 618 localized the DS7B value-gate repair path to value_rule_lines and
showed pattern-dominant restoration. This phase splits those rule lines into
category / relation / value / punctuation / wrong-line token groups.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase593_atlas_guided_causal_patch import candidate_delta_metric  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids, n_heads_for, parse_layers  # noqa: E402
from phase610_head_cumulative_mixture import TOP_HEADS  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase618_attention_source_pattern_content import (  # noqa: E402
    MODES,
    answer_prefix_pos,
    collect_parts_multi,
    default_layers,
    full_ids,
    line_spans,
    offset_tokens,
    patch_answer_score,
    patched_scores,
    summarize,
)


OUT_ROOT = Path("results/glm5_phase619_rule_line_token_micro_atlas")
MICRO_GROUPS = [
    "all_value_rule_lines",
    "correct_rule_line",
    "correct_category_token",
    "correct_relation_token",
    "correct_value_token",
    "correct_punct_token",
    "wrong_same_relation_lines",
    "wrong_same_category_lines",
    "other_value_rule_lines",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def top_heads(model: str, n_heads: int, k: int) -> List[int]:
    heads = [h for h in TOP_HEADS.get(model, []) if 0 <= h < n_heads]
    if not heads:
        heads = list(range(min(k, n_heads)))
    return heads[:k]


def tokens_for_char_span(offsets: List[Tuple[int, int]], start: int, end: int, answer_pos: int) -> List[int]:
    toks = []
    for ti, (a, b) in enumerate(offsets):
        if ti >= answer_pos:
            continue
        if max(a, start) < min(b, end):
            toks.append(ti)
    return toks


def find_part_span(line: str, line_start: int, text: str, start_after: int = 0) -> Tuple[int, int]:
    local = line.find(text, start_after)
    if local < 0:
        return (-1, -1)
    return (line_start + local, line_start + local + len(text))


def parse_value_rule_line(line: str) -> Tuple[str, str, str] | None:
    stripped = line.strip()
    if not stripped or " belongs to " in stripped or stripped == "Rules:":
        return None
    if stripped.startswith("Question:") or stripped.startswith("Answer:"):
        return None
    body = stripped[:-1] if stripped.endswith(".") else stripped
    parts = body.split()
    if len(parts) < 3:
        return None
    return (parts[0], parts[1], " ".join(parts[2:]))


def rule_micro_groups(tokenizer, prompt: str, case: Dict, answer_pos: int) -> Dict[str, List[int]]:
    offsets = offset_tokens(tokenizer, prompt)
    groups: Dict[str, List[int]] = {name: [] for name in MICRO_GROUPS}
    category = case["category"]
    relation = case["relation"]
    correct = case["correct"]
    in_rules = False

    for line, start, end in line_spans(prompt):
        stripped = line.strip()
        if stripped == "Rules:":
            in_rules = True
            continue
        if stripped.startswith("Question:") or stripped.startswith("Answer:"):
            in_rules = False
            continue
        if not in_rules:
            continue
        parsed = parse_value_rule_line(line)
        if parsed is None:
            continue
        cat, rel, val = parsed
        line_toks = tokens_for_char_span(offsets, start, end, answer_pos)
        groups["all_value_rule_lines"].extend(line_toks)

        is_correct = cat == category and rel == relation and val == correct
        if is_correct:
            groups["correct_rule_line"].extend(line_toks)
            cat_span = find_part_span(line, start, category)
            rel_span = find_part_span(line, start, relation, max(0, cat_span[1] - start))
            val_span = find_part_span(line, start, correct, max(0, rel_span[1] - start))
            punct_local = line.rfind(".")
            punct_span = (start + punct_local, start + punct_local + 1) if punct_local >= 0 else (-1, -1)
            for name, span in [
                ("correct_category_token", cat_span),
                ("correct_relation_token", rel_span),
                ("correct_value_token", val_span),
                ("correct_punct_token", punct_span),
            ]:
                if span[0] >= 0:
                    groups[name].extend(tokens_for_char_span(offsets, span[0], span[1], answer_pos))
        elif rel == relation:
            groups["wrong_same_relation_lines"].extend(line_toks)
            groups["other_value_rule_lines"].extend(line_toks)
        elif cat == category:
            groups["wrong_same_category_lines"].extend(line_toks)
            groups["other_value_rule_lines"].extend(line_toks)
        else:
            groups["other_value_rule_lines"].extend(line_toks)

    return {k: sorted(set(v)) for k, v in groups.items()}


def make_specs(model_name: str, layers_to_scan: List[int], heads_by_layer: Dict[int, int], top_k: int) -> List[Dict]:
    min_heads = min(heads_by_layer.values())
    heads = top_heads(model_name, min_heads, top_k)
    specs: List[Dict] = []
    for group in MICRO_GROUPS:
        for mode in MODES:
            specs.append({
                "name": f"top{len(heads)}_midlate_{group}_{mode}",
                "group": group,
                "mode": mode,
                "ops": [{"layer": li, "heads": heads} for li in layers_to_scan],
            })
    single_groups = [
        "correct_rule_line",
        "correct_category_token",
        "correct_relation_token",
        "correct_value_token",
        "wrong_same_relation_lines",
    ]
    for li in layers_to_scan:
        for h in heads[: min(3, len(heads))]:
            for group in single_groups:
                specs.append({
                    "name": f"L{li}_H{h}_{group}_rr",
                    "group": group,
                    "mode": "rr_pattern_content",
                    "ops": [{"layer": li, "heads": [h]}],
                })
    return specs


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers_to_scan = parse_layers(args.layers) if args.layers else default_layers(args.model, info.n_layers)
        layers_to_scan = [li for li in layers_to_scan if 0 <= li < info.n_layers]
        heads_by_layer = {li: n_heads_for(model, get_layers(model)[li].self_attn) for li in layers_to_scan}
        specs = make_specs(args.model, layers_to_scan, heads_by_layer, args.top_k)
        if args.compact:
            keep = {
                "all_value_rule_lines",
                "correct_rule_line",
                "correct_category_token",
                "correct_relation_token",
                "correct_value_token",
                "wrong_same_relation_lines",
                "wrong_same_category_lines",
            }
            specs = [s for s in specs if s["group"] in keep or s["name"].startswith("L")]
        values = CANDIDATE_VALUES[:4]
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "empty_correct_rule_line": 0}
        target_seen = 0
        log(
            f"{args.model}: layers={info.n_layers}, scan_layers={layers_to_scan}, "
            f"heads={heads_by_layer}, specs={len(specs)}, raw_cases={len(raw_cases)}"
        )

        for si, case in enumerate(raw_cases):
            base_len = answer_prefix_pos(tokenizer, case["base_prompt"])
            repair_len = answer_prefix_pos(tokenizer, case["repair_prompt"])
            if base_len != repair_len:
                filtered["token_len_mismatch"] += 1
                continue
            group_tokens = rule_micro_groups(tokenizer, case["base_prompt"], case, base_len)
            if not group_tokens["correct_rule_line"]:
                filtered["empty_correct_rule_line"] += 1
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
            "phase": 619,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "layers_to_scan": layers_to_scan,
            "heads_by_layer": heads_by_layer,
            "top_k": args.top_k,
            "top_heads": top_heads(args.model, min(heads_by_layer.values()), args.top_k),
            "micro_groups": MICRO_GROUPS,
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
    out_path = out_dir / f"phase619_{args.model}_rule_line_token_micro_atlas_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
