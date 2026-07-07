#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase204_global_trajectory_stop_execution_atlas as p204  # noqa: E402
import phase209_pattern_running_contrast_atlas as p209  # noqa: E402
import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402
from model_utils import get_layers  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402


PHASE = 217
SOURCE_PHASE = 216
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase217_reproducible_headset_validation")


HEAD_SETS = {
    "qwen3": [
        {
            "headset_id": "qwen3_explain_route_set",
            "pattern_id": "answer_explain",
            "anchor_label": "mixed",
            "heads": [
                {"layer_idx": 3, "head_idx": 15},
                {"layer_idx": 11, "head_idx": 3},
                {"layer_idx": 29, "head_idx": 11},
            ],
        },
        {
            "headset_id": "qwen3_repeat_route_set",
            "pattern_id": "answer_repeat",
            "anchor_label": "mixed",
            "heads": [
                {"layer_idx": 31, "head_idx": 26},
                {"layer_idx": 29, "head_idx": 11},
            ],
        },
    ],
    "glm4": [
        {
            "headset_id": "glm4_target_seeded_l29_route_set",
            "pattern_id": "answer_target_seeded",
            "anchor_label": "gen_after_step_6",
            "heads": [
                {"layer_idx": 29, "head_idx": 28},
                {"layer_idx": 29, "head_idx": 10},
                {"layer_idx": 29, "head_idx": 18},
                {"layer_idx": 29, "head_idx": 11},
                {"layer_idx": 29, "head_idx": 25},
            ],
        },
        {
            "headset_id": "glm4_repeat_route_set",
            "pattern_id": "answer_repeat",
            "anchor_label": "gen_after_step_3",
            "heads": [{"layer_idx": 12, "head_idx": 21}],
        },
        {
            "headset_id": "glm4_explain_competition_route_set",
            "pattern_id": "answer_explain",
            "anchor_label": "gen_after_step_3",
            "heads": [{"layer_idx": 12, "head_idx": 18}],
        },
    ],
    "deepseek7b": [
        {
            "headset_id": "deepseek7b_explain_l24_route_set",
            "pattern_id": "answer_explain",
            "anchor_label": "gen_after_step_1",
            "weak_sample": True,
            "heads": [
                {"layer_idx": 24, "head_idx": 20},
                {"layer_idx": 24, "head_idx": 16},
            ],
        },
        {
            "headset_id": "deepseek7b_list_l24_route_set",
            "pattern_id": "answer_list",
            "anchor_label": "gen_after_step_1",
            "weak_sample": True,
            "heads": [{"layer_idx": 24, "head_idx": 20}],
        },
    ],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def select_rows(rows: list[dict[str, Any]], pattern_id: str, max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success = [row for row in rows if row.get("pattern_id") == pattern_id and row.get("pattern_match")]
    drift = [row for row in rows if row.get("pattern_id") == pattern_id and not row.get("pattern_match")]
    return success[: int(max_rows)], drift[: int(max_rows)]


def anchor_step(anchor_label: str) -> int | None:
    if anchor_label == "prompt_last":
        return 1
    m = re.search(r"gen_after_step_(\d+)", anchor_label)
    if not m:
        return None
    return max(1, int(m.group(1)))


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def heads_by_layer(headset: dict[str, Any]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for item in headset.get("heads") or []:
        out[int(item["layer_idx"])].append(int(item["head_idx"]))
    return {layer: sorted(set(heads)) for layer, heads in out.items()}


def install_headset_hooks(model, headset: dict[str, Any], pos: int, ablate: bool, norm_sink: list[dict[str, Any]]):
    layers = get_layers(model)
    handles = []
    for layer_idx, head_ids in heads_by_layer(headset).items():
        attn = get_attention_module(layers[int(layer_idx)])
        o_proj = get_o_proj(attn)
        num_heads = get_num_heads(model, attn)
        valid_heads = [h for h in head_ids if 0 <= h < int(num_heads)]
        if not valid_heads:
            continue

        def make_hook(li: int, hids: list[int], nheads: int):
            def hook(_module: Any, inputs: tuple[Any, ...]):
                x = inputs[0]
                if x.shape[-1] % nheads != 0:
                    raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {nheads}")
                y = x.clone() if ablate else x
                view = y.view(y.shape[0], y.shape[1], nheads, y.shape[-1] // nheads)
                safe_pos = min(max(0, int(pos)), y.shape[1] - 1)
                for hid in hids:
                    vec = view[0, safe_pos, int(hid), :].detach().float()
                    norm_sink.append(
                        {
                            "layer_idx": int(li),
                            "head_idx": int(hid),
                            "pos": int(safe_pos),
                            "head_output_norm": float(torch.linalg.vector_norm(vec).item()),
                            "head_output_mean": float(vec.mean().item()),
                            "ablated": bool(ablate),
                        }
                    )
                    if ablate:
                        view[:, safe_pos, int(hid), :] = 0
                return (y,) + inputs[1:]

            return hook

        handles.append(o_proj.register_forward_pre_hook(make_hook(layer_idx, valid_heads, int(num_heads))))
    return handles


def forward_logits(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    headset: dict[str, Any] | None,
    ablate: bool,
    capture_norms: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    norm_rows: list[dict[str, Any]] = []
    handles = []
    try:
        if headset is not None and (ablate or capture_norms):
            handles = install_headset_hooks(model, headset, last_pos, ablate, norm_rows)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = out.logits[0, last_pos].detach().float().cpu()
        del out
    finally:
        for handle in handles:
            handle.remove()
        del input_ids, attention_mask
    return logits, norm_rows


def generate_condition(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    row: dict[str, Any],
    headset: dict[str, Any],
    condition: str,
    max_steps: int,
) -> dict[str, Any]:
    generated = ""
    emitted_ids: list[int] = []
    emitted_tokens: list[str] = []
    token_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    patch_step = anchor_step(str(headset.get("anchor_label") or ""))
    for step in range(1, int(max_steps) + 1):
        ablate = condition == "headset_all_steps" or (condition == "headset_anchor_step" and step == patch_step)
        capture = condition == "none"
        logits, norms = forward_logits(
            model,
            tokenizer,
            device,
            str(row.get("prompt") or "") + generated,
            headset if (ablate or capture) else None,
            ablate,
            capture,
        )
        for nrow in norms:
            norm_rows.append({**nrow, "step": int(step), "condition": condition})
        metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
        next_id = int(metrics["top_token_id"])
        next_text = str(metrics.get("top_token") or token_text(tokenizer, next_id))
        emitted_ids.append(next_id)
        emitted_tokens.append(next_text)
        generated += next_text
        token_rows.append(
            {
                "step": int(step),
                "patched": bool(ablate),
                "top_token": next_text,
                "top_token_id": next_id,
                "target_rank": metrics.get("target_rank"),
                "prose_margin": metrics.get("prose_margin"),
                "echo_margin": metrics.get("echo_margin"),
                "stop_margin": metrics.get("stop_margin"),
            }
        )
        if next_id in eos_ids:
            break
    expected = p209.expected_output_pattern(str(row.get("pattern_id")))
    classification = p209.classify_pattern(generated, row, emitted_ids, eos_ids)
    return {
        "generated": generated,
        "emitted_ids": emitted_ids,
        "emitted_tokens": emitted_tokens,
        "steps_generated": len(emitted_ids),
        "expected_output_pattern": expected,
        "pattern_match": classification.get("output_pattern") == expected,
        "pattern_drift": classification.get("output_pattern") != expected,
        "failure_mode": "match" if classification.get("output_pattern") == expected else classification.get("output_pattern"),
        "token_rows": token_rows,
        "norm_rows": norm_rows,
        **classification,
    }


def summarize_rollouts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["headset_id", "source_group", "condition"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, items in buckets.items():
        rec = {name: value for name, value in zip(keys, key)}
        rec.update(
            {
                "rows": len(items),
                "pattern_match": sum(1 for item in items if item.get("pattern_match")),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "output_patterns": dict(Counter(str(item.get("output_pattern")) for item in items).most_common()),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
            }
        )
        out.append(rec)
    out.sort(key=lambda row: (str(row.get("headset_id")), str(row.get("source_group")), str(row.get("condition"))))
    return out


def summarize_norms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["headset_id", "source_group", "layer_idx", "head_idx", "step"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, items in buckets.items():
        rec = {name: value for name, value in zip(keys, key)}
        vals = [float(item.get("head_output_norm") or 0.0) for item in items]
        rec.update({"rows": len(items), "head_output_norm_mean": sum(vals) / len(vals) if vals else None})
        out.append(rec)
    out.sort(key=lambda row: (str(row.get("headset_id")), str(row.get("source_group")), int(row.get("layer_idx")), int(row.get("head_idx")), int(row.get("step"))))
    return out


def effect_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        buckets[str(row.get("headset_id"))][(str(row.get("source_group")), str(row.get("condition")))] = row
    out = []
    for hid, by in buckets.items():
        success_none = by.get(("success_repro", "none"), {})
        drift_none = by.get(("drift_repro", "none"), {})
        for condition in ["headset_anchor_step", "headset_all_steps"]:
            success_patch = by.get(("success_repro", condition), {})
            drift_patch = by.get(("drift_repro", condition), {})
            out.append(
                {
                    "headset_id": hid,
                    "condition": condition,
                    "success_rows": success_none.get("rows", 0),
                    "drift_rows": drift_none.get("rows", 0),
                    "success_base_match": finite_int(success_none.get("pattern_match")),
                    "success_patch_match": finite_int(success_patch.get("pattern_match")),
                    "drift_base_match": finite_int(drift_none.get("pattern_match")),
                    "drift_patch_match": finite_int(drift_patch.get("pattern_match")),
                    "damage_match_loss": finite_int(success_none.get("pattern_match")) - finite_int(success_patch.get("pattern_match")),
                    "repair_match_gain": finite_int(drift_patch.get("pattern_match")) - finite_int(drift_none.get("pattern_match")),
                    "success_base_outputs": success_none.get("output_patterns", {}),
                    "success_patch_outputs": success_patch.get("output_patterns", {}),
                    "drift_base_outputs": drift_none.get("output_patterns", {}),
                    "drift_patch_outputs": drift_patch.get("output_patterns", {}),
                }
            )
    out.sort(key=lambda row: abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0)), reverse=True)
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    rollout_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        rows = load_rows(args.model, args.phase210_round)
        for headset in HEAD_SETS[args.model]:
            success_rows, drift_rows = select_rows(rows, str(headset["pattern_id"]), int(args.max_filter_rows))
            kept_success: list[dict[str, Any]] = []
            kept_drift: list[dict[str, Any]] = []
            for source_group, source_rows, target_list in [
                ("success", success_rows, kept_success),
                ("drift", drift_rows, kept_drift),
            ]:
                for row in source_rows:
                    result = generate_condition(model, tokenizer, device, groups, row, headset, "none", int(args.max_steps))
                    reproducible = bool(result.get("pattern_match")) if source_group == "success" else not bool(result.get("pattern_match"))
                    filter_rows.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase217_baseline_filter_row",
                            "model": args.model,
                            "headset_id": headset["headset_id"],
                            "pattern_id": headset["pattern_id"],
                            "source_group": source_group,
                            "trajectory_id": row.get("trajectory_id"),
                            "reproducible": reproducible,
                            "output_pattern": result.get("output_pattern"),
                            "pattern_match": result.get("pattern_match"),
                        }
                    )
                    if reproducible:
                        target_list.append(row)
            kept_success = kept_success[: int(args.max_eval_rows)]
            kept_drift = kept_drift[: int(args.max_eval_rows)]
            for source_group, eval_rows in [("success_repro", kept_success), ("drift_repro", kept_drift)]:
                for row in eval_rows:
                    for condition in ["none", "headset_anchor_step", "headset_all_steps"]:
                        result = generate_condition(model, tokenizer, device, groups, row, headset, condition, int(args.max_steps))
                        for nrow in result.pop("norm_rows", []):
                            norm_rows.append(
                                {
                                    "phase": PHASE,
                                    "source_phase": SOURCE_PHASE,
                                    "row_kind": "phase217_head_output_norm_row",
                                    "model": args.model,
                                    "headset_id": headset["headset_id"],
                                    "pattern_id": headset["pattern_id"],
                                    "source_group": source_group,
                                    "trajectory_id": row.get("trajectory_id"),
                                    **nrow,
                                }
                            )
                        rollout_rows.append(
                            {
                                "phase": PHASE,
                                "source_phase": SOURCE_PHASE,
                                "row_kind": "phase217_headset_rollout_row",
                                "model": args.model,
                                "headset_id": headset["headset_id"],
                                "pattern_id": headset["pattern_id"],
                                "anchor_label": headset.get("anchor_label"),
                                "heads": headset.get("heads"),
                                "weak_sample": bool(headset.get("weak_sample")),
                                "source_group": source_group,
                                "condition": condition,
                                "trajectory_id": row.get("trajectory_id"),
                                "target_label": row.get("target_label"),
                                "object": row.get("object"),
                                **result,
                            }
                        )
            log(f"{args.model}|{headset['headset_id']}: kept success={len(kept_success)} drift={len(kept_drift)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary_rows = summarize_rollouts(rollout_rows)
    norm_summary_rows = summarize_norms(norm_rows)
    effects = effect_rows(summary_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Reproducible Baseline and Headset Validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "headset_count": len(HEAD_SETS[args.model]),
        "filter_rows": len(filter_rows),
        "reproducible_success_rows": sum(1 for row in filter_rows if row.get("source_group") == "success" and row.get("reproducible")),
        "reproducible_drift_rows": sum(1 for row in filter_rows if row.get("source_group") == "drift" and row.get("reproducible")),
        "rollout_rows": len(rollout_rows),
        "norm_rows": len(norm_rows),
        "summary_rows": len(summary_rows),
        "norm_summary_rows": len(norm_summary_rows),
        "effect_rows": effects,
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effects),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effects),
    }
    write_json(out_dir / f"phase217_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase217_{args.model}_filter_rows.jsonl", filter_rows)
    write_jsonl(out_dir / f"phase217_{args.model}_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase217_{args.model}_head_output_norm_rows.jsonl", norm_rows)
    write_jsonl(out_dir / f"phase217_{args.model}_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase217_{args.model}_norm_summary_rows.jsonl", norm_summary_rows)
    write_jsonl(out_dir / f"phase217_{args.model}_effect_rows.jsonl", effects)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "filter_rows": len(filter_rows),
                "rollout_rows": len(rollout_rows),
                "damage": payload["total_damage_match_loss"],
                "repair": payload["total_repair_match_gain"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase217_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    effects = []
    for model in MODELS:
        effects.extend(list(p214.iter_jsonl(out_dir / f"phase217_{model}_effect_rows.jsonl") or []))
    payload = {
        "schema_version": "phase217_reproducible_headset_validation_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_name,
        "models": [summary.get("model") for summary in summaries],
        "headset_count": sum(int(summary.get("headset_count") or 0) for summary in summaries),
        "filter_rows": sum(int(summary.get("filter_rows") or 0) for summary in summaries),
        "reproducible_success_rows": sum(int(summary.get("reproducible_success_rows") or 0) for summary in summaries),
        "reproducible_drift_rows": sum(int(summary.get("reproducible_drift_rows") or 0) for summary in summaries),
        "rollout_rows": sum(int(summary.get("rollout_rows") or 0) for summary in summaries),
        "norm_rows": sum(int(summary.get("norm_rows") or 0) for summary in summaries),
        "effect_rows": len(effects),
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effects),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effects),
        "top_effect_rows": sorted(
            effects,
            key=lambda row: abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0)),
            reverse=True,
        )[:60],
    }
    write_json(out_dir / "phase217_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase217_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 217 reproducible baseline and headset validation", ""]
    lines.append(f"Headset count: {payload.get('headset_count')}")
    lines.append(f"Filter rows: {payload.get('filter_rows')}")
    lines.append(f"Reproducible success rows: {payload.get('reproducible_success_rows')}")
    lines.append(f"Reproducible drift rows: {payload.get('reproducible_drift_rows')}")
    lines.append(f"Rollout rows: {payload.get('rollout_rows')}")
    lines.append(f"Norm rows: {payload.get('norm_rows')}")
    lines.append(f"Total damage match loss: {payload.get('total_damage_match_loss')}")
    lines.append(f"Total repair match gain: {payload.get('total_repair_match_gain')}")
    lines.append("")
    lines.append("| headset | condition | success | drift | damage | repair | success outputs | drift outputs |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload.get("top_effect_rows") or []:
        lines.append(
            f"| {row.get('headset_id')} | {row.get('condition')} | {row.get('success_rows')} | {row.get('drift_rows')} | "
            f"{row.get('damage_match_loss')} | {row.get('repair_match_gain')} | {row.get('success_patch_outputs')} | {row.get('drift_patch_outputs')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="reproducible_headset_validation")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-filter-rows", type=int, default=8)
    parser.add_argument("--max-eval-rows", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "status": payload.get("status"),
                    "models": payload.get("models"),
                    "headsets": payload.get("headset_count"),
                    "filter_rows": payload.get("filter_rows"),
                    "rollout_rows": payload.get("rollout_rows"),
                    "damage": payload.get("total_damage_match_loss"),
                    "repair": payload.get("total_repair_match_gain"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
