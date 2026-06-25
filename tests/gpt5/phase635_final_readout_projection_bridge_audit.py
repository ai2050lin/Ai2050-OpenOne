#!/usr/bin/env python3
"""
Phase 635: Final Readout Projection Bridge Audit
最终读出投影桥审计

Phase 634 ruled out the tested multi-position source/format field as the
missing DS7B token0 prefix gate. This phase audits the last bridge:
final_norm input -> final_norm output -> lm_head token competition.
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
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_model_info, release_model  # noqa: E402
from phase584_gate_repair import CANDIDATE_VALUES, load_model_flash  # noqa: E402
from phase587_value_winner_competition import winner_stats  # noqa: E402
from phase597_state_conditioned_mlp_generation_audit import score_map  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase609_query_oproj_head_decomposition import answer_ids  # noqa: E402
from phase612_source_aligned_pattern_content_split import build_aligned_cases  # noqa: E402
from phase615_residual_state_builder_scan import collect_components as collect_answer_components  # noqa: E402
from phase624_result_state_downstream_propagation_atlas import default_downstream_layers  # noqa: E402
from phase628_prefix_format_semantic_integration import generation_eval, make_cumulative_patches, token_strings  # noqa: E402
from phase630_distributed_format_route_multisource import (  # noqa: E402
    collect_positions_components,
    install_source_patch_hooks,
    source_groups,
)
from phase631_token0_prefix_readout_competition import get_unembed  # noqa: E402
from phase632_natural_prefix_readout_writer_backtrace import logit_metrics  # noqa: E402
from phase634_multi_position_format_source_field_closure import (  # noqa: E402
    COMPONENT,
    group_layer_defaults,
    make_group_patch,
    merge_patches,
)


OUT_ROOT = Path("results/glm5_phase635_final_readout_projection_bridge_audit")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_prefix_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def final_state_probe(
    model,
    tokenizer,
    device,
    prompt: str,
    source_patches: List[Tuple[int, str, List[int], List[torch.Tensor]]] | None = None,
    final_patch: Dict | None = None,
) -> Dict:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    final_norm = get_final_norm(model)
    source_handles = []
    handles = []
    captured: Dict[str, torch.Tensor] = {}
    if source_patches:
        source_handles = install_source_patch_hooks(model, tokenizer, prompt, source_patches)

    if final_norm is not None:
        def pre_hook(_module, inputs):
            x = inputs[0]
            captured["final_norm_input_orig"] = x[0, -1].detach().float().cpu()
            if final_patch is None or final_patch["kind"] != "input":
                return None
            x_new = x.clone()
            target = final_patch["target"].to(device=x_new.device, dtype=x_new.dtype)
            x_new[0, -1, :] = target
            captured["final_norm_input_patched"] = x_new[0, -1].detach().float().cpu()
            return (x_new,) + tuple(inputs[1:])

        def out_hook(_module, _inputs, output):
            y = extract_tensor(output)
            captured["final_norm_output_orig"] = y[0, -1].detach().float().cpu()
            if final_patch is None:
                return output
            y_new = y.clone()
            if final_patch["kind"] == "output":
                target = final_patch["target"].to(device=y_new.device, dtype=y_new.dtype)
                y_new[0, -1, :] = target
            elif final_patch["kind"] == "delta":
                delta = final_patch["delta"].to(device=y_new.device, dtype=y_new.dtype)
                y_new[0, -1, :] = y_new[0, -1, :] + delta
            else:
                return output
            captured["final_norm_output_patched"] = y_new[0, -1].detach().float().cpu()
            if isinstance(output, tuple):
                return (y_new,) + output[1:]
            return y_new

        handles.append(final_norm.register_forward_pre_hook(pre_hook))
        handles.append(final_norm.register_forward_hook(out_hook))

    try:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
        return {
            "logits": logits.detach().cpu(),
            "final_norm_input": captured.get("final_norm_input_patched", captured.get("final_norm_input_orig")),
            "final_norm_output": captured.get("final_norm_output_patched", captured.get("final_norm_output_orig")),
            "final_norm_output_orig": captured.get("final_norm_output_orig"),
        }
    finally:
        for h in handles:
            h.remove()
        for h in source_handles:
            h.remove()


def greedy_generate_bridge(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    source_patches: List[Tuple[int, str, List[int], List[torch.Tensor]]] | None = None,
    answer_patches: List[Tuple[int, str, torch.Tensor]] | None = None,
    final_patch: Dict | None = None,
) -> Dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = list(prompt_ids)
    gen = []
    top5 = []
    source_handles = []
    answer_handles = []
    if source_patches:
        source_handles = install_source_patch_hooks(model, tokenizer, prompt, source_patches)
    if answer_patches:
        from phase629_format_prefix_gate_localization import install_answer_pos_layer_patch_hooks  # noqa: E402
        answer_handles = install_answer_pos_layer_patch_hooks(model, tokenizer, prompt, answer_patches)
    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                h_pre = None
                h_out = None
                if final_patch is not None and step == 0:
                    final_norm = get_final_norm(model)
                    if final_norm is not None:
                        if final_patch["kind"] == "input":
                            def pre_hook(_module, inputs):
                                x = inputs[0]
                                x_new = x.clone()
                                x_new[0, -1, :] = final_patch["target"].to(device=x_new.device, dtype=x_new.dtype)
                                return (x_new,) + tuple(inputs[1:])
                            h_pre = final_norm.register_forward_pre_hook(pre_hook)
                        else:
                            def out_hook(_module, _inputs, output):
                                y = extract_tensor(output)
                                y_new = y.clone()
                                if final_patch["kind"] == "output":
                                    y_new[0, -1, :] = final_patch["target"].to(device=y_new.device, dtype=y_new.dtype)
                                elif final_patch["kind"] == "delta":
                                    y_new[0, -1, :] = y_new[0, -1, :] + final_patch["delta"].to(device=y_new.device, dtype=y_new.dtype)
                                if isinstance(output, tuple):
                                    return (y_new,) + output[1:]
                                return y_new
                            h_out = final_norm.register_forward_hook(out_hook)
                try:
                    logits = model(input_ids=torch.tensor([ids], device=device), return_dict=True).logits[0, -1].float()
                finally:
                    if h_pre is not None:
                        h_pre.remove()
                    if h_out is not None:
                        h_out.remove()
                topv, topi = torch.topk(torch.log_softmax(logits, dim=-1), k=5)
                top5.append([
                    {"id": int(i), "text": tokenizer.decode([int(i)]), "logprob": float(v)}
                    for v, i in zip(topv.cpu(), topi.cpu())
                ])
                tid = int(torch.argmax(logits).item())
                gen.append(tid)
                ids.append(tid)
        return {"ids": gen, "tokens": token_strings(tokenizer, gen), "text": tokenizer.decode(gen), "top5": top5}
    finally:
        for h in source_handles:
            h.remove()
        for h in answer_handles:
            h.remove()


def make_all6_source_patch(model, tokenizer, device, case: Dict, layers_needed: List[int], seed: int):
    layer_map = group_layer_defaults(case["model_name"])
    base_groups = source_groups(tokenizer, case, case["base_prompt"], use_repair_subject=False)
    repair_groups = source_groups(tokenizer, case, case["repair_prompt"], use_repair_subject=True)
    patches = []
    missing = 0
    for gi, group in enumerate(["prompt_last", "answer_label", "question_mark_answer", "relation_tail", "question_subject", "question_all"]):
        li = layer_map[group]
        if li not in layers_needed:
            continue
        base_pos = base_groups.get(group, [])
        repair_pos = repair_groups.get(group, [])
        if not base_pos or len(base_pos) != len(repair_pos):
            missing += 1
            continue
        base_cache = collect_positions_components(model, tokenizer, device, case["base_prompt"], base_pos, [li], [COMPONENT])
        repair_cache = collect_positions_components(model, tokenizer, device, case["repair_prompt"], repair_pos, [li], [COMPONENT])
        patches.extend(make_group_patch(
            base_cache,
            repair_cache,
            base_pos,
            li,
            COMPONENT,
            "restore",
            seed + gi * 173,
        ))
    return merge_patches(patches), missing


def vector_metrics(base_vec: torch.Tensor | None, vec: torch.Tensor | None, readout_unit: torch.Tensor) -> Dict:
    if base_vec is None or vec is None:
        return {"delta_norm": None, "readout_projection": None, "readout_cos": None}
    delta = vec.float().cpu() - base_vec.float().cpu()
    norm = float(delta.norm().item())
    proj = float(torch.dot(delta, readout_unit).item())
    cos = float(F.cosine_similarity(delta, readout_unit, dim=0).item()) if norm > 1e-8 else 0.0
    return {"delta_norm": norm, "readout_projection": proj, "readout_cos": cos}


def summarize(rows: List[Dict]) -> Dict:
    stats = {}
    for row in rows:
        item = stats.setdefault(row["mode"], {
            "mode": row["mode"],
            "n": 0,
            "tok0_hit": 0,
            "exact": 0,
            "wrong_exact": 0,
            "sum_margin": 0.0,
            "sum_rank": 0.0,
            "sum_out_proj": 0.0,
            "sum_out_cos": 0.0,
            "sum_in_proj": 0.0,
            "projection_count": 0,
            "top0_text": {},
        })
        item["n"] += 1
        item["tok0_hit"] += int(row["tok0_id"] == row["prefix_id"])
        item["exact"] += int(row["eval"]["exact_correct"])
        item["wrong_exact"] += int(row["eval"]["exact_wrong"])
        item["sum_margin"] += row["prefix_margin"]
        item["sum_rank"] += row["prefix_rank"]
        out_proj = row["final_output_delta"].get("readout_projection")
        out_cos = row["final_output_delta"].get("readout_cos")
        in_proj = row["final_input_delta"].get("readout_projection")
        if out_proj is not None:
            item["sum_out_proj"] += out_proj
            item["sum_out_cos"] += out_cos or 0.0
            item["sum_in_proj"] += in_proj or 0.0
            item["projection_count"] += 1
        item["top0_text"].setdefault(row["tok0_text"], 0)
        item["top0_text"][row["tok0_text"]] += 1
    out = []
    for item in stats.values():
        n = max(1, item["n"])
        p = max(1, item["projection_count"])
        row = dict(item)
        row["tok0_rate"] = item["tok0_hit"] / n
        row["exact_rate"] = item["exact"] / n
        row["wrong_exact_rate"] = item["wrong_exact"] / n
        row["mean_prefix_margin"] = item["sum_margin"] / n
        row["mean_prefix_rank"] = item["sum_rank"] / n
        row["mean_final_output_readout_projection"] = item["sum_out_proj"] / p
        row["mean_final_output_readout_cos"] = item["sum_out_cos"] / p
        row["mean_final_input_readout_projection"] = item["sum_in_proj"] / p
        row["top0_text"] = dict(sorted(row["top0_text"].items(), key=lambda kv: kv[1], reverse=True)[:6])
        out.append(row)
    out.sort(key=lambda x: (x["exact"], x["tok0_hit"], x["mean_prefix_margin"]), reverse=True)
    return {"by_mode": out}


def prefix_rank(logits: torch.Tensor, prefix_id: int) -> int:
    return int((logits > logits[prefix_id]).sum().item()) + 1


def run_model(args) -> Dict:
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        values = CANDIDATE_VALUES[:4]
        downstream_layers = default_downstream_layers(args.model, info.n_layers)
        downstream_layers = [li for li in downstream_layers if 0 <= li < info.n_layers]
        layer_map = group_layer_defaults(args.model)
        layers_needed = sorted(set(layer_map.values()))
        tokenization = {v: {"ids": answer_ids(tokenizer, v), "tokens": token_strings(tokenizer, answer_ids(tokenizer, v))} for v in values}
        max_new_tokens = max(len(v["ids"]) for v in tokenization.values())
        raw_cases = list(build_aligned_cases(args.n_tables, args.max_samples))
        W = get_unembed(model).detach().float().cpu()
        rows = []
        filtered = {"token_len_mismatch": 0, "not_target": 0, "source_missing": 0}
        target_seen = 0
        log(f"{args.model}: downstream={downstream_layers}, source_layers={layers_needed}, raw_cases={len(raw_cases)}")

        for si, case0 in enumerate(raw_cases):
            case = dict(case0)
            case["model_name"] = args.model
            if answer_prefix_pos(tokenizer, case["base_prompt"]) != answer_prefix_pos(tokenizer, case["repair_prompt"]):
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

            correct_ids = answer_ids(tokenizer, case["correct"])
            old_wrong_ids = answer_ids(tokenizer, base["top_wrong"])
            prefix_id = correct_ids[0]
            base_state = final_state_probe(model, tokenizer, device, case["base_prompt"])
            repair_state = final_state_probe(model, tokenizer, device, case["repair_prompt"])
            base_logits = base_state["logits"]
            top_id = int(torch.argmax(base_logits).item())
            competitor_id = top_id if top_id != prefix_id else int(torch.topk(base_logits, k=2).indices[1].item())
            direction = W[prefix_id] - W[competitor_id]
            readout_unit = direction / max(float(direction.norm().item()), 1e-8)
            hidden_norm = float(base_state["final_norm_output"].norm().item()) if base_state["final_norm_output"] is not None else 1.0

            answer_cache = {
                "base": collect_answer_components(model, tokenizer, device, case["base_prompt"], case["correct"], downstream_layers),
                "repair": collect_answer_components(model, tokenizer, device, case["repair_prompt"], case["correct"], downstream_layers),
            }
            semantic_patches = make_cumulative_patches(answer_cache, downstream_layers, "layer_out", False, si * 1009 + 23)
            source_patches, missing = make_all6_source_patch(model, tokenizer, device, case, layers_needed, si * 1009 + 41)
            filtered["source_missing"] += missing
            source_state = final_state_probe(model, tokenizer, device, case["base_prompt"], source_patches=source_patches)

            readout_delta = readout_unit * (hidden_norm * args.readout_scale)
            mode_specs = {
                "base": {"prompt": case["base_prompt"], "source": [], "answer": [], "final": None, "state": base_state},
                "repair_prompt": {"prompt": case["repair_prompt"], "source": [], "answer": [], "final": None, "state": repair_state},
                "semantic_cumulative": {"prompt": case["base_prompt"], "source": [], "answer": semantic_patches, "final": None, "state": base_state},
                "source_all6": {"prompt": case["base_prompt"], "source": source_patches, "answer": [], "final": None, "state": source_state},
                "source_all6_semantic": {"prompt": case["base_prompt"], "source": source_patches, "answer": semantic_patches, "final": None, "state": source_state},
                "final_input_repair": {"prompt": case["base_prompt"], "source": [], "answer": [], "final": {"kind": "input", "target": repair_state["final_norm_input"]}, "state": None},
                "final_input_repair_semantic": {"prompt": case["base_prompt"], "source": [], "answer": semantic_patches, "final": {"kind": "input", "target": repair_state["final_norm_input"]}, "state": None},
                "final_output_repair": {"prompt": case["base_prompt"], "source": [], "answer": [], "final": {"kind": "output", "target": repair_state["final_norm_output"]}, "state": None},
                "final_output_repair_semantic": {"prompt": case["base_prompt"], "source": [], "answer": semantic_patches, "final": {"kind": "output", "target": repair_state["final_norm_output"]}, "state": None},
                "final_output_source": {"prompt": case["base_prompt"], "source": [], "answer": [], "final": {"kind": "output", "target": source_state["final_norm_output"]}, "state": None},
                "final_output_source_semantic": {"prompt": case["base_prompt"], "source": [], "answer": semantic_patches, "final": {"kind": "output", "target": source_state["final_norm_output"]}, "state": None},
                "readout_delta": {"prompt": case["base_prompt"], "source": [], "answer": [], "final": {"kind": "delta", "delta": readout_delta}, "state": None},
                "readout_delta_semantic": {"prompt": case["base_prompt"], "source": [], "answer": semantic_patches, "final": {"kind": "delta", "delta": readout_delta}, "state": None},
            }

            for mode, spec in mode_specs.items():
                probe = spec["state"] or final_state_probe(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    source_patches=spec["source"],
                    final_patch=spec["final"],
                )
                logits = probe["logits"]
                metrics = logit_metrics(logits, prefix_id, competitor_id)
                gen = greedy_generate_bridge(
                    model,
                    tokenizer,
                    device,
                    spec["prompt"],
                    max_new_tokens,
                    source_patches=spec["source"],
                    answer_patches=spec["answer"],
                    final_patch=spec["final"],
                )
                ev = generation_eval(gen, correct_ids, old_wrong_ids)
                rows.append({
                    "sample_idx": si,
                    "mode": mode,
                    "case": {k: case[k] for k in ["tt_idx", "object", "relation", "category", "correct"]},
                    "prefix_id": prefix_id,
                    "prefix_text": tokenizer.decode([prefix_id]),
                    "competitor_id": competitor_id,
                    "competitor_text": tokenizer.decode([competitor_id]),
                    "tok0_id": metrics["tok0_id"],
                    "tok0_text": tokenizer.decode([metrics["tok0_id"]]),
                    "prefix_margin": metrics["prefix_margin"],
                    "prefix_rank": prefix_rank(logits, prefix_id),
                    "eval": ev,
                    "final_input_delta": vector_metrics(base_state["final_norm_input"], probe.get("final_norm_input"), readout_unit),
                    "final_output_delta": vector_metrics(base_state["final_norm_output"], probe.get("final_norm_output"), readout_unit),
                    "generation_text": gen["text"] if len(rows) < 260 else "",
                })

        summary = summarize(rows)
        log("Best final bridge modes:")
        for item in summary["by_mode"][:20]:
            log(
                f"  {item['mode']}: tok0={item['tok0_hit']}/{item['n']} exact={item['exact']}/{item['n']} "
                f"rank={item['mean_prefix_rank']:.1f} margin={item['mean_prefix_margin']:.3f} "
                f"out_proj={item['mean_final_output_readout_projection']:.3f}"
            )
        return {
            "phase": 635,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": info.n_layers,
            "downstream_layers": downstream_layers,
            "source_layer_map": layer_map,
            "readout_scale": args.readout_scale,
            "tokenization": tokenization,
            "max_new_tokens": max_new_tokens,
            "n_raw_cases": len(raw_cases),
            "n_rows": len({r["sample_idx"] for r in rows}),
            "n_mode_rows": len(rows),
            "n_target_cases_seen": target_seen,
            "filtered": filtered,
            "target_only": args.target_only,
            "summary": summary,
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
    parser.add_argument("--n-tables", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--readout-scale", type=float, default=0.25)
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
        log("SMOKE TEST MODE")
    if args.confirm:
        args.n_tables = max(args.n_tables, 32)
        args.max_samples = max(args.max_samples, 256)
        log("CONFIRMATION TEST MODE")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = (time.time() - t0) / 60.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "confirm" if args.confirm else "run"
    out_path = out_dir / f"phase635_{args.model}_final_readout_projection_bridge_audit_{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']:.2f} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
