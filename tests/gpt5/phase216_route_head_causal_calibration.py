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
from phase112_attention_transport_head_mapping_cuda import (  # noqa: E402
    get_attention_module,
    get_num_heads,
    get_o_proj,
    make_head_ablation_pre_hook,
)


PHASE = 216
SOURCE_PHASE = 215
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase216_route_head_causal_calibration")


CANDIDATES = {
    "qwen3": [
        {"pattern_id": "answer_explain", "anchor_label": "gen_after_step_3", "layer_idx": 3, "head_idx": 15, "route_delta": -0.4903},
        {"pattern_id": "answer_explain", "anchor_label": "gen_after_step_6", "layer_idx": 29, "head_idx": 11, "route_delta": 0.3799},
        {"pattern_id": "answer_explain", "anchor_label": "gen_after_step_1", "layer_idx": 11, "head_idx": 3, "route_delta": 0.3744},
        {"pattern_id": "answer_repeat", "anchor_label": "prompt_last", "layer_idx": 31, "head_idx": 26, "route_delta": -0.3544},
        {"pattern_id": "answer_repeat", "anchor_label": "gen_after_step_1", "layer_idx": 29, "head_idx": 11, "route_delta": 0.3533},
    ],
    "glm4": [
        {"pattern_id": "answer_target_seeded", "anchor_label": "gen_after_step_6", "layer_idx": 29, "head_idx": 28, "route_delta": 0.6162},
        {"pattern_id": "answer_target_seeded", "anchor_label": "gen_after_step_6", "layer_idx": 29, "head_idx": 10, "route_delta": 0.5537},
        {"pattern_id": "answer_target_seeded", "anchor_label": "gen_after_step_6", "layer_idx": 29, "head_idx": 18, "route_delta": 0.5504},
        {"pattern_id": "answer_target_seeded", "anchor_label": "gen_after_step_6", "layer_idx": 29, "head_idx": 11, "route_delta": 0.5383},
        {"pattern_id": "answer_target_seeded", "anchor_label": "gen_after_step_6", "layer_idx": 29, "head_idx": 25, "route_delta": 0.5132},
        {"pattern_id": "answer_repeat", "anchor_label": "gen_after_step_3", "layer_idx": 12, "head_idx": 21, "route_delta": 0.5219},
        {"pattern_id": "answer_explain", "anchor_label": "gen_after_step_3", "layer_idx": 12, "head_idx": 18, "route_delta": -0.4386},
    ],
    "deepseek7b": [
        {"pattern_id": "answer_explain", "anchor_label": "gen_after_step_1", "layer_idx": 24, "head_idx": 20, "route_delta": -0.8713, "weak_sample": True},
        {"pattern_id": "answer_explain", "anchor_label": "gen_after_step_1", "layer_idx": 24, "head_idx": 16, "route_delta": -0.8330, "weak_sample": True},
        {"pattern_id": "answer_list", "anchor_label": "gen_after_step_1", "layer_idx": 24, "head_idx": 20, "route_delta": -0.6579, "weak_sample": True},
    ],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def install_head_ablation(model, layer_idx: int, head_idx: int, pos: int):
    layers = get_layers(model)
    attn = get_attention_module(layers[int(layer_idx)])
    o_proj = get_o_proj(attn)
    num_heads = get_num_heads(model, attn)
    if int(head_idx) >= int(num_heads):
        raise ValueError(f"head {head_idx} >= num_heads {num_heads}")
    positions = torch.tensor([int(pos)], dtype=torch.long)
    return o_proj.register_forward_pre_hook(make_head_ablation_pre_hook(int(num_heads), int(head_idx), positions))


def forward_logits(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    candidate: dict[str, Any] | None = None,
) -> torch.Tensor:
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    handle = None
    try:
        if candidate is not None:
            handle = install_head_ablation(model, int(candidate["layer_idx"]), int(candidate["head_idx"]), last_pos)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = out.logits[0, last_pos].detach().float().cpu()
        del out
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention_mask
    return logits


def generate_condition(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    row: dict[str, Any],
    candidate: dict[str, Any],
    condition: str,
    max_steps: int,
) -> dict[str, Any]:
    generated = ""
    emitted_ids: list[int] = []
    emitted_tokens: list[str] = []
    token_rows: list[dict[str, Any]] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    patch_step = anchor_step(str(candidate.get("anchor_label") or ""))
    for step in range(1, int(max_steps) + 1):
        active_patch = condition == "ablate_all_steps" or (condition == "ablate_anchor_step" and step == patch_step)
        logits = forward_logits(
            model,
            tokenizer,
            device,
            str(row.get("prompt") or "") + generated,
            candidate if active_patch else None,
        )
        metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
        next_id = int(metrics["top_token_id"])
        next_text = str(metrics.get("top_token") or token_text(tokenizer, next_id))
        emitted_ids.append(next_id)
        emitted_tokens.append(next_text)
        generated += next_text
        token_rows.append(
            {
                "step": int(step),
                "patched": bool(active_patch),
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
        **classification,
    }


def candidate_key(model_name: str, candidate: dict[str, Any]) -> str:
    return (
        f"{model_name}|{candidate['pattern_id']}|{candidate['anchor_label']}|"
        f"L{candidate['layer_idx']}H{candidate['head_idx']}"
    )


def summarize_rollouts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["candidate_key", "source_group", "condition"]
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
    out.sort(key=lambda row: (str(row.get("candidate_key")), str(row.get("source_group")), str(row.get("condition"))))
    return out


def candidate_effects(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        buckets[str(row.get("candidate_key"))][(str(row.get("source_group")), str(row.get("condition")))] = row
    out = []
    for key, by in buckets.items():
        success_none = by.get(("success", "none"), {})
        drift_none = by.get(("drift", "none"), {})
        for condition in ["ablate_anchor_step", "ablate_all_steps"]:
            success_patch = by.get(("success", condition), {})
            drift_patch = by.get(("drift", condition), {})
            out.append(
                {
                    "candidate_key": key,
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
    out.sort(key=lambda row: (abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0))), reverse=True)
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    rollout_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        rows = load_rows(args.model, args.phase210_round)
        for candidate in CANDIDATES[args.model]:
            success_rows, drift_rows = select_rows(rows, str(candidate["pattern_id"]), int(args.max_eval_rows))
            ckey = candidate_key(args.model, candidate)
            for source_group, eval_rows in [("success", success_rows), ("drift", drift_rows)]:
                for row in eval_rows:
                    for condition in ["none", "ablate_anchor_step", "ablate_all_steps"]:
                        result = generate_condition(
                            model, tokenizer, device, groups, row, candidate, condition, int(args.max_steps)
                        )
                        rollout_rows.append(
                            {
                                "phase": PHASE,
                                "source_phase": SOURCE_PHASE,
                                "row_kind": "phase216_route_head_ablation_rollout_row",
                                "model": args.model,
                                "candidate_key": ckey,
                                "pattern_id": candidate["pattern_id"],
                                "anchor_label": candidate["anchor_label"],
                                "layer_idx": int(candidate["layer_idx"]),
                                "head_idx": int(candidate["head_idx"]),
                                "route_delta": candidate.get("route_delta"),
                                "weak_sample": bool(candidate.get("weak_sample")),
                                "source_group": source_group,
                                "condition": condition,
                                "trajectory_id": row.get("trajectory_id"),
                                "target_label": row.get("target_label"),
                                "object": row.get("object"),
                                **result,
                            }
                        )
            log(f"{ckey}: success={len(success_rows)} drift={len(drift_rows)}")
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
    effect_rows = candidate_effects(summary_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Route Head Causal Calibration",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "candidate_count": len(CANDIDATES[args.model]),
        "rollout_rows": len(rollout_rows),
        "summary_rows": len(summary_rows),
        "effect_rows": effect_rows,
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effect_rows),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effect_rows),
    }
    write_json(out_dir / f"phase216_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase216_{args.model}_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase216_{args.model}_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase216_{args.model}_effect_rows.jsonl", effect_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "rollout_rows": len(rollout_rows), "damage": payload["total_damage_match_loss"], "repair": payload["total_repair_match_gain"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase216_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    effect_rows = []
    for model in MODELS:
        effect_rows.extend(list(p214.iter_jsonl(out_dir / f"phase216_{model}_effect_rows.jsonl") or []))
    payload = {
        "schema_version": "phase216_route_head_causal_calibration_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_name,
        "models": [summary.get("model") for summary in summaries],
        "candidate_count": sum(int(summary.get("candidate_count") or 0) for summary in summaries),
        "rollout_rows": sum(int(summary.get("rollout_rows") or 0) for summary in summaries),
        "effect_rows": len(effect_rows),
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in effect_rows),
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in effect_rows),
        "top_effect_rows": sorted(
            effect_rows,
            key=lambda row: abs(int(row.get("damage_match_loss") or 0)) + abs(int(row.get("repair_match_gain") or 0)),
            reverse=True,
        )[:60],
    }
    write_json(out_dir / "phase216_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase216_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 216 route head causal calibration", ""]
    lines.append(f"Candidate count: {payload.get('candidate_count')}")
    lines.append(f"Rollout rows: {payload.get('rollout_rows')}")
    lines.append(f"Total damage match loss: {payload.get('total_damage_match_loss')}")
    lines.append(f"Total repair match gain: {payload.get('total_repair_match_gain')}")
    lines.append("")
    lines.append("| candidate | condition | success | drift | damage | repair | success outputs | drift outputs |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload.get("top_effect_rows") or []:
        lines.append(
            f"| {row.get('candidate_key')} | {row.get('condition')} | {row.get('success_rows')} | {row.get('drift_rows')} | "
            f"{row.get('damage_match_loss')} | {row.get('repair_match_gain')} | {row.get('success_patch_outputs')} | {row.get('drift_patch_outputs')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="route_head_causal_calibration")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-eval-rows", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload.get("status"), "models": payload.get("models"), "candidates": payload.get("candidate_count"), "rollout_rows": payload.get("rollout_rows"), "damage": payload.get("total_damage_match_loss"), "repair": payload.get("total_repair_match_gain")}, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
