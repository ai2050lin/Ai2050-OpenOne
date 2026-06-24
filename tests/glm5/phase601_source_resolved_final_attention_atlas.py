#!/usr/bin/env python3
"""
Phase 601: Source-Resolved Final Attention Acceptance Atlas
最后层源词元注意力接受图谱

Phase 600 showed that natural correct trajectories change final-layer attention
more than artificial MLP-input repair. This phase resolves that attention shift
by semantic source token groups.
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
from typing import Dict, List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase586_distributed_value_path_patch import build_cases, case_positions, random_same_norm, token_pos_after_substring  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import (  # noqa: E402
    collect_mlp_input_output,
    get_mlp,
    replace_input,
    score_map,
)
from phase598_downstream_trajectory_acceptance_audit import select_nodes  # noqa: E402
from phase600_final_layer_acceptance_rule_audit import collect_final_block  # noqa: E402
from model_utils import get_layers  # noqa: E402


OUT_ROOT = Path("results/glm5_phase601_source_resolved_final_attention_atlas")
SOURCE_GROUPS = [
    "rule_relation",
    "rule_value",
    "object",
    "category_first",
    "query_relation",
    "query_category",
    "prompt_last",
    "punct_newline",
    "other",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_span_after_substring(tokenizer, prompt: str, needle: str, occurrence: str = "last") -> List[int]:
    if not needle:
        return []
    idx = prompt.find(needle) if occurrence == "first" else prompt.rfind(needle)
    if idx < 0:
        return []
    start = len(tokenizer.encode(prompt[:idx], add_special_tokens=False))
    end = len(tokenizer.encode(prompt[:idx + len(needle)], add_special_tokens=False))
    return list(range(max(0, start), max(start, end)))


def punct_newline_positions(tokenizer, prompt: str) -> List[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    out = []
    for i, tid in enumerate(ids):
        s = tokenizer.decode([tid])
        if any(ch in s for ch in ["\n", ".", ":", "?", ",", ";", "!", "-", "(", ")"]):
            out.append(i)
    return out


def source_groups(tokenizer, prompt: str, case: Dict, relation_for_rule: str) -> Dict[str, List[int]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    n = len(ids)
    groups = {
        "rule_relation": token_span_after_substring(tokenizer, prompt, relation_for_rule, "first"),
        "rule_value": token_span_after_substring(tokenizer, prompt, case["correct"], "first"),
        "object": token_span_after_substring(tokenizer, prompt, case["object"], "last"),
        "category_first": token_span_after_substring(tokenizer, prompt, case["category"], "first"),
        "query_relation": token_span_after_substring(tokenizer, prompt, case["relation"], "last"),
        "query_category": token_span_after_substring(tokenizer, prompt, case["category"], "last"),
        "prompt_last": [n - 1] if n else [],
        "punct_newline": punct_newline_positions(tokenizer, prompt),
    }
    used = set()
    for name, vals in groups.items():
        if name != "punct_newline":
            used.update(vals)
    groups["other"] = [i for i in range(n) if i not in used]
    return {k: sorted(set(v for v in vals if 0 <= v < n)) for k, vals in groups.items()}


def attn_slice(attn: Optional[torch.Tensor], target_pos: Optional[int]) -> Optional[torch.Tensor]:
    if attn is None or target_pos is None or target_pos < 0:
        return None
    if attn.dim() != 4 or target_pos >= attn.shape[2]:
        return None
    return attn[0, :, target_pos, :].float().cpu()


def mass_by_group(attn_for_pos: Optional[torch.Tensor], groups: Dict[str, List[int]]) -> Dict[str, float]:
    if attn_for_pos is None:
        return {g: 0.0 for g in SOURCE_GROUPS}
    src_len = attn_for_pos.shape[-1]
    out = {}
    for name in SOURCE_GROUPS:
        idxs = [i for i in groups.get(name, []) if 0 <= i < src_len]
        if not idxs:
            out[name] = 0.0
        else:
            out[name] = float(attn_for_pos[:, idxs].sum(dim=-1).mean().cpu())
    return out


def entropy_top(attn_for_pos: Optional[torch.Tensor]) -> Dict[str, float]:
    if attn_for_pos is None:
        return {"entropy": 0.0, "top_mass": 0.0}
    eps = 1e-8
    p = attn_for_pos / attn_for_pos.sum(dim=-1, keepdim=True).clamp_min(eps)
    return {
        "entropy": float((-(p * torch.log(p.clamp_min(eps))).sum(dim=-1).mean()).cpu()),
        "top_mass": float(p.max(dim=-1).values.mean().cpu()),
    }


def collect_prompt_attention(model, tokenizer, device, prompt: str, probe_layer: int,
                             target_pos: Optional[int],
                             groups: Dict[str, List[int]],
                             source_layer: Optional[int] = None,
                             patch_pos: Optional[int] = None,
                             target_input: Optional[torch.Tensor] = None) -> Dict:
    cap = collect_final_block(
        model,
        tokenizer,
        device,
        prompt,
        probe_layer,
        source_layer=source_layer,
        patch_pos=patch_pos,
        target_input=target_input,
        capture_attn=True,
    )
    a = attn_slice(cap.get("attention_pattern"), target_pos)
    return {"mass": mass_by_group(a, groups), "stats": entropy_top(a)}


def target_specs(base_x: torch.Tensor, repair_x: torch.Tensor, wrong_x: Optional[torch.Tensor],
                 seed: int, alpha: float) -> List[Dict]:
    d = repair_x.float().cpu() - base_x.float().cpu()
    specs = [
        {"name": "artificial_repair", "kind": "artificial_repair", "target": base_x.float().cpu() + alpha * d},
        {"name": "artificial_random", "kind": "artificial_random", "target": base_x.float().cpu() + alpha * random_same_norm(d, seed=seed)},
    ]
    if wrong_x is not None:
        specs.append({"name": "artificial_wrong", "kind": "artificial_wrong", "target": base_x.float().cpu() + alpha * (wrong_x.float().cpu() - base_x.float().cpu())})
    return specs


def summarize(rows: List[Dict]) -> Dict:
    keys = sorted({k for r in rows for k in r["attention"]})
    by_key = {}
    for key in keys:
        items = [r["attention"][key] for r in rows if key in r["attention"]]
        entry = {
            "key": key,
            "trajectory": items[0]["trajectory"],
            "position": items[0]["node"]["position"],
            "source_layer": items[0]["node"]["source_layer"],
            "probe_layer": items[0]["node"]["probe_layer"],
            "n": len(items),
            "entropy": 0.0,
            "top_mass": 0.0,
        }
        for g in SOURCE_GROUPS:
            entry[f"mass_{g}"] = 0.0
            entry[f"delta_{g}"] = 0.0
        for item in items:
            entry["entropy"] += item["stats"]["entropy"]
            entry["top_mass"] += item["stats"]["top_mass"]
            for g in SOURCE_GROUPS:
                entry[f"mass_{g}"] += item["mass"].get(g, 0.0)
                entry[f"delta_{g}"] += item["delta"].get(g, 0.0)
        n = max(1, len(items))
        entry["entropy"] /= n
        entry["top_mass"] /= n
        for g in SOURCE_GROUPS:
            entry[f"mass_{g}"] /= n
            entry[f"delta_{g}"] /= n
        by_key[key] = entry

    contrast = {}
    for key, entry in by_key.items():
        if "|natural_correct" not in key:
            continue
        base = key.replace("|natural_correct", "")
        art_key = base + "|artificial_repair"
        nat_wrong_key = base + "|natural_wrong"
        if art_key in by_key:
            c = {
                "key": base,
                "position": entry["position"],
                "source_layer": entry["source_layer"],
                "n": min(entry["n"], by_key[art_key]["n"]),
            }
            score = 0.0
            for g in SOURCE_GROUPS:
                v = entry[f"delta_{g}"] - by_key[art_key][f"delta_{g}"]
                c[f"nat_minus_art_{g}"] = v
                score += abs(v)
            if nat_wrong_key in by_key:
                for g in SOURCE_GROUPS:
                    c[f"nat_minus_wrong_{g}"] = entry[f"delta_{g}"] - by_key[nat_wrong_key][f"delta_{g}"]
            c["l1_nat_minus_artificial"] = score
            contrast[base] = c

    best_deltas = sorted(
        by_key.values(),
        key=lambda x: max(abs(x[f"delta_{g}"]) for g in SOURCE_GROUPS),
        reverse=True,
    )[:80]
    best_contrast = sorted(
        contrast.values(),
        key=lambda x: x["l1_nat_minus_artificial"],
        reverse=True,
    )[:40]
    log("Largest source attention deltas:")
    for item in best_deltas[:10]:
        best_g = max(SOURCE_GROUPS, key=lambda g: abs(item[f"delta_{g}"]))
        log(f"  {item['key']}: {best_g} delta={item[f'delta_{best_g}']:.4f}, entropy={item['entropy']:.3f}")
    log("Largest natural-vs-artificial contrasts:")
    for item in best_contrast[:8]:
        best_g = max(SOURCE_GROUPS, key=lambda g: abs(item[f"nat_minus_art_{g}"]))
        log(f"  {item['key']}: {best_g} nat-art={item[f'nat_minus_art_{best_g}']:.4f}")
    return {"by_key": by_key, "contrast": contrast, "best_deltas": best_deltas, "best_contrast": best_contrast}


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        cases = list(build_cases(args.n_tables, args.max_samples))
        nodes = select_nodes(args.model, args.top_nodes)
        source_layers = sorted({n["layer"] for n in nodes})
        probe_layer = info.n_layers - 1
        log(f"{args.model}: layers={info.n_layers}, cases={len(cases)}, nodes={[(n['position'], n['layer']) for n in nodes]}, probe=L{probe_layer}")

        rows = []
        target_seen = 0
        for si, case in enumerate(cases):
            correct = case["correct"]
            base_scores = score_map(model, tokenizer, device, case["base_prompt"], values)
            repair_scores = score_map(model, tokenizer, device, case["repair_prompt"], values)
            base = winner_stats(base_scores, correct)
            repair = winner_stats(repair_scores, correct)
            target_case = (not base["correct"]) and repair["correct"]
            if args.target_only and not target_case:
                continue
            target_seen += int(target_case)

            base_pos = case_positions(tokenizer, case, case["base_prompt"], case["relation"])
            repair_pos = case_positions(tokenizer, case, case["repair_prompt"], case["repair_rel"])
            wrong_pos = case_positions(tokenizer, case, case["wrong_prompt"], case["wrong_rel"])
            groups_base = source_groups(tokenizer, case["base_prompt"], case, case["relation"])
            groups_repair = source_groups(tokenizer, case["repair_prompt"], case, case["repair_rel"])
            groups_wrong = source_groups(tokenizer, case["wrong_prompt"], case, case["wrong_rel"])

            base_cap = collect_mlp_input_output(model, tokenizer, device, case["base_prompt"], source_layers)
            repair_cap = collect_mlp_input_output(model, tokenizer, device, case["repair_prompt"], source_layers)
            wrong_cap = collect_mlp_input_output(model, tokenizer, device, case["wrong_prompt"], source_layers)

            row = {
                "sample_idx": si,
                "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                "target_case": target_case,
                "base": base,
                "repair_prompt": repair,
                "attention": {},
            }

            for node in nodes:
                pos_name = node["position"]
                source_layer = node["layer"]
                bp = base_pos.get(pos_name)
                rp = repair_pos.get(pos_name)
                wp = wrong_pos.get(pos_name)
                if bp is None or rp is None or wp is None:
                    continue
                if source_layer not in base_cap["mlp_input"] or source_layer not in repair_cap["mlp_input"]:
                    continue
                if bp >= base_cap["mlp_input"][source_layer].shape[1] or rp >= repair_cap["mlp_input"][source_layer].shape[1]:
                    continue
                base_x = base_cap["mlp_input"][source_layer][0, bp]
                repair_x = repair_cap["mlp_input"][source_layer][0, rp]
                wrong_x = None
                if source_layer in wrong_cap["mlp_input"] and wp < wrong_cap["mlp_input"][source_layer].shape[1]:
                    wrong_x = wrong_cap["mlp_input"][source_layer][0, wp]

                base_attn = collect_prompt_attention(model, tokenizer, device, case["base_prompt"], probe_layer, bp, groups_base)
                for traj, prompt, tpos, groups, kwargs in [
                    ("natural_correct", case["repair_prompt"], rp, groups_repair, {}),
                    ("natural_wrong", case["wrong_prompt"], wp, groups_wrong, {}),
                ]:
                    attn = collect_prompt_attention(model, tokenizer, device, prompt, probe_layer, tpos, groups, **kwargs)
                    key = f"{pos_name}|L{source_layer}|{traj}"
                    row["attention"][key] = {
                        "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                        "trajectory": traj,
                        "mass": attn["mass"],
                        "delta": {g: attn["mass"].get(g, 0.0) - base_attn["mass"].get(g, 0.0) for g in SOURCE_GROUPS},
                        "base_mass": base_attn["mass"],
                        "stats": attn["stats"],
                    }

                for spec in target_specs(base_x, repair_x, wrong_x, si * 1009 + source_layer, args.alpha):
                    attn = collect_prompt_attention(
                        model,
                        tokenizer,
                        device,
                        case["base_prompt"],
                        probe_layer,
                        bp,
                        groups_base,
                        source_layer=source_layer,
                        patch_pos=bp,
                        target_input=spec["target"],
                    )
                    key = f"{pos_name}|L{source_layer}|{spec['kind']}"
                    row["attention"][key] = {
                        "node": {"position": pos_name, "source_layer": source_layer, "probe_layer": probe_layer},
                        "trajectory": spec["kind"],
                        "mass": attn["mass"],
                        "delta": {g: attn["mass"].get(g, 0.0) - base_attn["mass"].get(g, 0.0) for g in SOURCE_GROUPS},
                        "base_mass": base_attn["mass"],
                        "stats": attn["stats"],
                    }
            rows.append(row)

        return {
            "phase": 601,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "probe_layer": probe_layer,
            "n_cases": len(cases),
            "n_target_cases_seen": target_seen,
            "n_rows": len(rows),
            "target_only": args.target_only,
            "alpha": args.alpha,
            "nodes": nodes,
            "source_groups": SOURCE_GROUPS,
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
    parser.add_argument("--top-nodes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--target-only", action="store_true", default=True)
    parser.add_argument("--include-nontarget", dest="target_only", action="store_false")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 1
        args.max_samples = 4
        args.top_nodes = min(args.top_nodes, 2)
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 16)
        args.max_samples = max(args.max_samples, 128)
        args.top_nodes = max(args.top_nodes, 3)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase601_{args.model}_source_resolved_final_attention_atlas_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
