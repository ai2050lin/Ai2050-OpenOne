#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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
from model_utils import get_layers  # noqa: E402


PHASE = 212
SOURCE_PHASE = 211
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase212_switchpoint_causal_validation")


CANDIDATES = {
    "qwen3": [
        {"pattern_id": "answer_list", "failure_mode": "other_or_wrong", "layer_idx": 32, "step": 11},
        {"pattern_id": "answer_list", "failure_mode": "short_answer", "layer_idx": 32, "step": 9},
    ],
    "glm4": [
        {"pattern_id": "answer_list", "failure_mode": "repeat_answer", "layer_idx": 29, "step": 8},
        {"pattern_id": "answer_list", "failure_mode": "echo_then_answer", "layer_idx": 35, "step": 8},
    ],
    "deepseek7b": [
        {"pattern_id": "answer_explain", "failure_mode": "other_or_wrong", "layer_idx": 26, "step": 7},
        {"pattern_id": "answer_list", "failure_mode": "other_or_wrong", "layer_idx": 24, "step": 7},
    ],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_trajectories(input_dir: Path, model: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(input_dir / f"phase210_{model}_trajectory_rows.jsonl") or [])


def select_rows(rows: list[dict[str, Any]], pattern_id: str, failure_mode: str, max_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success = [row for row in rows if row.get("pattern_id") == pattern_id and row.get("pattern_match")]
    drift = [
        row
        for row in rows
        if row.get("pattern_id") == pattern_id and not row.get("pattern_match") and str(row.get("failure_mode")) == failure_mode
    ]
    return success[:max_rows], drift[:max_rows]


def prefix_for_step(row: dict[str, Any], step: int) -> str:
    emitted = row.get("emitted_tokens") or []
    prefix_tokens = emitted[: max(0, int(step) - 1)]
    return str(row.get("prompt") or "") + "".join(str(tok) for tok in prefix_tokens)


def hidden_at_prefix(model, tokenizer, device: torch.device, text: str, layer_idx: int) -> torch.Tensor:
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_idx = int(layer_idx) + 1
    vec = result.hidden_states[hidden_idx][0, last_pos].detach().float().cpu()
    del result, input_ids, attention_mask
    return vec


def mean_hidden_for_rows(model, tokenizer, device: torch.device, rows: list[dict[str, Any]], layer_idx: int, step: int) -> torch.Tensor | None:
    vectors = []
    for row in rows:
        try:
            vectors.append(hidden_at_prefix(model, tokenizer, device, prefix_for_step(row, step), layer_idx))
        except Exception as exc:
            log(f"skip donor vector {row.get('trajectory_id')}: {exc}")
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def token_text(tokenizer, token_id: int | None) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return str(token_id)


def forward_logits_with_optional_patch(
    model,
    tokenizer,
    device: torch.device,
    text: str,
    layer_idx: int | None,
    patch_vector: torch.Tensor | None,
) -> torch.Tensor:
    handle = None
    if layer_idx is not None and patch_vector is not None:
        layers = get_layers(model)
        patch = patch_vector.to(device=device, dtype=next(model.parameters()).dtype)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1, :] = patch
                return (hidden, *output[1:])
            hidden = output.clone()
            hidden[:, -1, :] = patch
            return hidden

        handle = layers[int(layer_idx)].register_forward_hook(hook)
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = int(attention_mask.sum(dim=1).item()) - 1
    try:
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = result.logits[0, last_pos].detach().float().cpu()
        del result
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention_mask
    return logits


def generate_manual(
    model,
    tokenizer,
    device: torch.device,
    row: dict[str, Any],
    groups: dict[str, list[int]],
    max_steps: int,
    layer_idx: int | None = None,
    patch_step: int | None = None,
    patch_vector: torch.Tensor | None = None,
) -> dict[str, Any]:
    generated = ""
    emitted_ids: list[int] = []
    emitted_tokens: list[str] = []
    eos_ids = set(int(x) for x in groups.get("eos") or [])
    token_metrics: list[dict[str, Any]] = []
    for step in range(1, int(max_steps) + 1):
        use_patch = patch_vector is not None and patch_step is not None and int(step) == int(patch_step)
        logits = forward_logits_with_optional_patch(
            model,
            tokenizer,
            device,
            str(row.get("prompt") or "") + generated,
            layer_idx if use_patch else None,
            patch_vector if use_patch else None,
        )
        metrics = p204.metric_for_logits(tokenizer, logits, row, groups)
        next_id = int(metrics["top_token_id"])
        next_text = str(metrics.get("top_token") or token_text(tokenizer, next_id))
        emitted_ids.append(next_id)
        emitted_tokens.append(next_text)
        token_metrics.append(
            {
                "step": step,
                "top_token_id": next_id,
                "top_token": next_text,
                "target_rank": metrics.get("target_rank"),
                "stop_margin": metrics.get("stop_margin"),
                "prose_margin": metrics.get("prose_margin"),
                "echo_margin": metrics.get("echo_margin"),
                "eos_rank": metrics.get("eos_rank"),
                "period_rank": metrics.get("period_rank"),
                "patched_step": bool(use_patch),
            }
        )
        generated += next_text
        if next_id in eos_ids:
            break
    classification = p209.classify_pattern(generated, row, emitted_ids, eos_ids)
    expected = p209.expected_output_pattern(str(row.get("pattern_id")))
    return {
        "generated": generated,
        "emitted_ids": emitted_ids,
        "emitted_tokens": emitted_tokens,
        "steps_generated": len(emitted_ids),
        "expected_output_pattern": expected,
        "pattern_match": classification.get("output_pattern") == expected,
        "pattern_drift": classification.get("output_pattern") != expected,
        "failure_mode": "match" if classification.get("output_pattern") == expected else classification.get("output_pattern"),
        "token_metrics": token_metrics,
        **classification,
    }


def eval_candidate(
    model,
    tokenizer,
    device: torch.device,
    groups: dict[str, list[int]],
    model_name: str,
    candidate: dict[str, Any],
    all_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pattern_id = str(candidate["pattern_id"])
    failure_mode = str(candidate["failure_mode"])
    layer_idx = int(candidate["layer_idx"])
    step = int(candidate["step"])
    success_rows, drift_rows = select_rows(all_rows, pattern_id, failure_mode, int(args.max_rows_per_group))
    donor_success = success_rows[: int(args.max_donor_rows)]
    donor_drift = drift_rows[: int(args.max_donor_rows)]
    eval_success = success_rows[: int(args.max_eval_rows)]
    eval_drift = drift_rows[: int(args.max_eval_rows)]
    success_mean = mean_hidden_for_rows(model, tokenizer, device, donor_success, layer_idx, step)
    drift_mean = mean_hidden_for_rows(model, tokenizer, device, donor_drift, layer_idx, step)
    rows: list[dict[str, Any]] = []
    if success_mean is None or drift_mean is None:
        return rows, {
            "model": model_name,
            "candidate": candidate,
            "status": "skipped",
            "reason": "missing donor vectors",
            "success_rows": len(success_rows),
            "drift_rows": len(drift_rows),
        }
    for source_group, eval_rows, patch_name, patch_vec in [
        ("drift", eval_drift, "none", None),
        ("drift", eval_drift, "success_mean", success_mean),
        ("success", eval_success, "none", None),
        ("success", eval_success, "drift_mean", drift_mean),
    ]:
        for item in eval_rows:
            result = generate_manual(
                model,
                tokenizer,
                device,
                item,
                groups,
                int(args.max_steps),
                layer_idx=layer_idx,
                patch_step=step,
                patch_vector=patch_vec,
            )
            rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase212_patch_rollout_row",
                    "model": model_name,
                    "candidate_key": f"{model_name}|{pattern_id}|{failure_mode}|L{layer_idx}|S{step}",
                    "pattern_id": pattern_id,
                    "failure_mode_target": failure_mode,
                    "layer_idx": layer_idx,
                    "patch_step": step,
                    "source_group": source_group,
                    "patch_condition": patch_name,
                    "trajectory_id": item.get("trajectory_id"),
                    "source_sample_id": item.get("source_sample_id"),
                    "prompt": item.get("prompt"),
                    "target_label": item.get("target_label"),
                    "object": item.get("object"),
                    **result,
                }
            )
    summary = summarize_candidate_rows(model_name, candidate, rows, len(success_rows), len(drift_rows))
    return rows, summary


def summarize_candidate_rows(model_name: str, candidate: dict[str, Any], rows: list[dict[str, Any]], success_count: int, drift_count: int) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("source_group"), row.get("patch_condition"))].append(row)
    group_rows = []
    for key, items in buckets.items():
        source_group, patch_condition = key
        group_rows.append(
            {
                "source_group": source_group,
                "patch_condition": patch_condition,
                "rows": len(items),
                "pattern_match": sum(1 for item in items if item.get("pattern_match")),
                "answer_present": sum(1 for item in items if item.get("answer_present")),
                "ended_with_eos": sum(1 for item in items if item.get("ended_with_eos")),
                "output_patterns": dict(Counter(str(item.get("output_pattern")) for item in items).most_common()),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
                "top_generated_prefixes": dict(Counter(str(item.get("generated"))[:80] for item in items).most_common(5)),
            }
        )
    by = {(row["source_group"], row["patch_condition"]): row for row in group_rows}
    drift_base = by.get(("drift", "none"), {})
    drift_patch = by.get(("drift", "success_mean"), {})
    success_base = by.get(("success", "none"), {})
    success_patch = by.get(("success", "drift_mean"), {})
    return {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "row_kind": "phase212_candidate_summary_row",
        "model": model_name,
        "candidate": candidate,
        "candidate_key": f"{model_name}|{candidate['pattern_id']}|{candidate['failure_mode']}|L{candidate['layer_idx']}|S{candidate['step']}",
        "available_success_rows": success_count,
        "available_drift_rows": drift_count,
        "group_rows": group_rows,
        "repair_match_gain": int(drift_patch.get("pattern_match") or 0) - int(drift_base.get("pattern_match") or 0),
        "damage_match_loss": int(success_base.get("pattern_match") or 0) - int(success_patch.get("pattern_match") or 0),
        "status": "complete",
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = INPUT_ROOT / args.phase210_round
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = p201.token_groups(tokenizer)
        all_rows = load_trajectories(input_dir, args.model)
        for candidate in CANDIDATES[args.model]:
            cand_rows, summary = eval_candidate(model, tokenizer, device, groups, args.model, candidate, all_rows, args)
            rows.extend(cand_rows)
            candidate_summaries.append(summary)
            log(f"{args.model} {summary.get('candidate_key')} rows={len(cand_rows)} repair={summary.get('repair_match_gain')} damage={summary.get('damage_match_loss')}")
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Switchpoint Causal Validation",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "phase210_round": args.phase210_round,
        "rollout_rows": len(rows),
        "candidate_summaries": candidate_summaries,
        "boundary": "Single layer-step hidden-state mean patch. Positive result is directional causal evidence, not full closure.",
    }
    write_json(out_dir / f"phase212_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase212_{args.model}_patch_rollout_rows.jsonl", rows)
    write_jsonl(out_dir / f"phase212_{args.model}_candidate_summary_rows.jsonl", candidate_summaries)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "rollout_rows": len(rows), "candidate_summaries": candidate_summaries}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase212_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    candidate_summaries = []
    for summary in summaries:
        candidate_summaries.extend(summary.get("candidate_summaries") or [])
    payload = {
        "schema_version": "phase212_switchpoint_causal_validation_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_name,
        "models": [summary.get("model") for summary in summaries],
        "rollout_rows": sum(int(summary.get("rollout_rows") or 0) for summary in summaries),
        "candidate_summaries": candidate_summaries,
        "total_repair_match_gain": sum(int(row.get("repair_match_gain") or 0) for row in candidate_summaries),
        "total_damage_match_loss": sum(int(row.get("damage_match_loss") or 0) for row in candidate_summaries),
    }
    write_json(out_dir / "phase212_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase212_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 212 switchpoint causal validation", ""]
    lines.append(f"Rollout rows: {payload.get('rollout_rows')}")
    lines.append(f"Total repair match gain: {payload.get('total_repair_match_gain')}")
    lines.append(f"Total damage match loss: {payload.get('total_damage_match_loss')}")
    lines.append("")
    lines.append("| model | candidate | success rows | drift rows | repair gain | damage loss |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload.get("candidate_summaries") or []:
        cand = row.get("candidate") or {}
        label = f"{cand.get('pattern_id')}->{cand.get('failure_mode')} L{cand.get('layer_idx')} S{cand.get('step')}"
        lines.append(
            f"| {row.get('model')} | {label} | {row.get('available_success_rows')} | {row.get('available_drift_rows')} | "
            f"{row.get('repair_match_gain')} | {row.get('damage_match_loss')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="switchpoint_causal_validation")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-rows-per-group", type=int, default=8)
    parser.add_argument("--max-donor-rows", type=int, default=6)
    parser.add_argument("--max-eval-rows", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload.get("status"), "models": payload.get("models"), "rollout_rows": payload.get("rollout_rows"), "repair_gain": payload.get("total_repair_match_gain"), "damage_loss": payload.get("total_damage_match_loss")}, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
