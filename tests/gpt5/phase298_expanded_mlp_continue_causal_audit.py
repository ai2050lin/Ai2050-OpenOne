#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase269_mlp_continuation_writer_necessity_audit as p269  # noqa: E402


PHASE = "Phase298"
SCHEMA_VERSION = "2.25.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
RESULT_ROOT = ROOT / "tests/result/phase298_expanded_mlp_continue_causal_audit"
ROUND_DEFAULT = "expanded_mlp_continue_causal_audit"
PATCHES = [
    {"patch_type": "mlp_zero_last_token", "scale": 0.0},
    {"patch_type": "mlp_quarter_last_token", "scale": 0.25},
    {"patch_type": "mlp_half_last_token", "scale": 0.5},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def load_prompt_bank() -> dict[tuple[str, str], dict[str, Any]]:
    bank: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(V2 / "phase292_feature_priority_queue_rows.jsonl"):
        bank[(str(row.get("model")), str(row.get("case_id")))] = row
    return bank


def selected_cases(model: str, limit: int) -> list[dict[str, Any]]:
    rows = [
        r
        for r in read_jsonl(V2 / "phase296_component_summary_rows.jsonl")
        if r.get("model") == model and r.get("dominant_positive_component") == "mlp"
    ]
    rows.sort(
        key=lambda r: (
            -safe_float(r.get("sum_positive_mlp_delta")),
            -safe_float(r.get("final_continue_stop_margin")),
            str(r.get("family_id")),
            str(r.get("case_id")),
        )
    )
    if limit:
        rows = rows[:limit]
    return rows


def base_row(case: dict[str, Any], prompt_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": case["model"],
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "mode_id": prompt_row.get("mode_id") or case.get("mode_id") or "phase292_queue",
        "variant_id": case.get("variant_id") or prompt_row.get("variant_id"),
        "target": case.get("target") or prompt_row.get("target"),
        "expected_pattern": prompt_row.get("expected_pattern"),
        "channel_focus": case.get("channel_focus") or prompt_row.get("channel_focus"),
        "source_component_summary_id": case.get("component_summary_id"),
        "dominant_positive_component_phase296": case.get("dominant_positive_component"),
        "strongest_mlp_layer_phase296": case.get("strongest_mlp_layer"),
        "strongest_mlp_delta_phase296": case.get("strongest_mlp_delta"),
        "sum_positive_mlp_delta_phase296": case.get("sum_positive_mlp_delta"),
        "final_continue_stop_margin_phase296": case.get("final_continue_stop_margin"),
    }


def audit_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_bank = load_prompt_bank()
    cases = selected_cases(args.model, args.limit_per_model)
    audit_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    model_obj = tokenizer = None
    try:
        model_obj, tokenizer, device, _attn_impl = p269.p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, case in enumerate(cases, start=1):
            prompt_row = prompt_bank.get((args.model, str(case.get("case_id"))))
            if not prompt_row:
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "missing_id": f"phase298:missing:{args.model}:{case.get('case_id')}",
                        "model": args.model,
                        "case_id": case.get("case_id"),
                        "reason": "prompt row not found in phase292 priority queue",
                    }
                )
                continue
            prompt = str(prompt_row["prompt"])
            aliases = [str(prompt_row.get("target") or case.get("target") or "")]
            layer_idx = int(case["strongest_mlp_layer"])
            base = base_row(case, prompt_row)
            try:
                base_logits = p269.forward_logits(model_obj, tokenizer, device, prompt)
                base_scores = p269.score_logits(tokenizer, base_logits, aliases)
                base_text, base_tokens = p269.generate_text(model_obj, tokenizer, device, prompt, args.rollout_tokens)
                for patch in PATCHES:
                    patched_logits = p269.forward_logits(model_obj, tokenizer, device, prompt, layer_idx, float(patch["scale"]))
                    patched_scores = p269.score_logits(tokenizer, patched_logits, aliases)
                    patched_text, patched_tokens = p269.generate_text(
                        model_obj, tokenizer, device, prompt, args.rollout_tokens, layer_idx, float(patch["scale"])
                    )
                    delta_margin = safe_float(patched_scores.get("continue_stop_margin")) - safe_float(base_scores.get("continue_stop_margin"))
                    delta_target = safe_float(patched_scores.get("target_logit")) - safe_float(base_scores.get("target_logit"))
                    winner_changed = base_scores.get("tri_winner") != patched_scores.get("tri_winner")
                    support = bool(delta_margin < -1.0 or winner_changed)
                    audit_id = f"phase298:audit:{args.model}:{case['case_id']}:L{layer_idx}:{patch['patch_type']}"
                    row = {
                        **base,
                        "mlp_causal_audit_id": audit_id,
                        "patch_type": patch["patch_type"],
                        "patch_scale": patch["scale"],
                        "patched_layer": layer_idx,
                        "base_continue_stop_margin": round(safe_float(base_scores.get("continue_stop_margin")), 6),
                        "patched_continue_stop_margin": round(safe_float(patched_scores.get("continue_stop_margin")), 6),
                        "delta_continue_stop_margin": round(delta_margin, 6),
                        "base_winner": base_scores.get("tri_winner"),
                        "patched_winner": patched_scores.get("tri_winner"),
                        "winner_changed": winner_changed,
                        "base_target_logit": round(safe_float(base_scores.get("target_logit")), 6),
                        "patched_target_logit": round(safe_float(patched_scores.get("target_logit")), 6),
                        "delta_target_logit": round(delta_target, 6),
                        "causal_support_level": "strong" if winner_changed else ("weak" if delta_margin < -1.0 else "not_supported"),
                        "necessity_supported": support,
                    }
                    audit_rows.append(row)
                    effect_rows.append(
                        {
                            **base,
                            "causal_effect_id": audit_id.replace(":audit:", ":effect:"),
                            "patch_type": patch["patch_type"],
                            "patch_scale": patch["scale"],
                            "patched_layer": layer_idx,
                            "effect_metric": "continue_stop_margin",
                            "effect_value": row["delta_continue_stop_margin"],
                            "winner_changed": winner_changed,
                            "necessity_supported": support,
                            "causal_support_level": row["causal_support_level"],
                        }
                    )
                    rollout_rows.append(
                        {
                            **base,
                            "rollout_effect_id": audit_id.replace(":audit:", ":rollout:"),
                            "patch_type": patch["patch_type"],
                            "patch_scale": patch["scale"],
                            "patched_layer": layer_idx,
                            "base_text": base_text[:300],
                            "patched_text": patched_text[:300],
                            "base_new_tokens": base_tokens,
                            "patched_new_tokens": patched_tokens,
                            "rollout_changed": base_text != patched_text,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "missing_id": f"phase298:error:{args.model}:{case.get('case_id')}",
                        "model": args.model,
                        "case_id": case.get("case_id"),
                        "family_id": case.get("family_id"),
                        "reason": repr(exc),
                    }
                )
            log(f"{args.model}: audited {idx}/{len(cases)} MLP-dominant cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p269.p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    summary = summarize_model(args.model, cases, audit_rows, effect_rows, rollout_rows, missing_rows)
    write_json(out_dir / f"phase298_{args.model}_summary.json", summary)
    write_jsonl(out_dir / f"phase298_{args.model}_mlp_causal_audit_rows.jsonl", audit_rows)
    write_jsonl(out_dir / f"phase298_{args.model}_causal_effect_rows.jsonl", effect_rows)
    write_jsonl(out_dir / f"phase298_{args.model}_rollout_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase298_{args.model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize_model(
    model: str,
    cases: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "model": model,
        "selected_mlp_dominant_cases": len(cases),
        "audit_rows": len(audit),
        "causal_effect_rows": len(effects),
        "rollout_rows": len(rollouts),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in audit)),
        "necessity_supported_counts": dict(Counter(str(r["necessity_supported"]) for r in audit)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in audit)),
        "causal_support_level_counts": dict(Counter(str(r["causal_support_level"]) for r in audit)),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollouts)),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in cases)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r.get("delta_continue_stop_margin")) for r in audit]),
        "mean_delta_target_logit": mean_safe([safe_float(r.get("delta_target_logit")) for r in audit]),
    }


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase298_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    audit: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        audit.extend(read_jsonl(out_dir / f"phase298_{model}_mlp_causal_audit_rows.jsonl"))
        effects.extend(read_jsonl(out_dir / f"phase298_{model}_causal_effect_rows.jsonl"))
        rollouts.extend(read_jsonl(out_dir / f"phase298_{model}_rollout_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase298_{model}_missing_rows.jsonl"))
    by_model = defaultdict(list)
    by_family = defaultdict(list)
    for row in audit:
        by_model[str(row.get("model"))].append(row)
        by_family[str(row.get("family_id"))].append(row)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "round_name": round_name,
        "status": "complete",
        "model_summaries": summaries,
        "selected_mlp_dominant_cases": sum(int(s.get("selected_mlp_dominant_cases", 0)) for s in summaries),
        "audit_rows": len(audit),
        "causal_effect_rows": len(effects),
        "rollout_rows": len(rollouts),
        "missing_rows": len(missing),
        "patch_counts": dict(Counter(str(r["patch_type"]) for r in audit)),
        "necessity_supported_counts": dict(Counter(str(r["necessity_supported"]) for r in audit)),
        "winner_changed_counts": dict(Counter(str(r["winner_changed"]) for r in audit)),
        "causal_support_level_counts": dict(Counter(str(r["causal_support_level"]) for r in audit)),
        "rollout_changed_counts": dict(Counter(str(r["rollout_changed"]) for r in rollouts)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r.get("delta_continue_stop_margin")) for r in audit]),
        "mean_delta_target_logit": mean_safe([safe_float(r.get("delta_target_logit")) for r in audit]),
        "model_mean_delta_continue_stop_margin": {
            model: mean_safe([safe_float(r.get("delta_continue_stop_margin")) for r in rows]) for model, rows in sorted(by_model.items())
        },
        "family_mean_delta_continue_stop_margin": {
            family: mean_safe([safe_float(r.get("delta_continue_stop_margin")) for r in rows]) for family, rows in sorted(by_family.items())
        },
    }
    write_json(out_dir / "phase298_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase298_mlp_causal_audit_rows.jsonl", audit)
    write_jsonl(out_dir / "phase298_causal_effect_rows.jsonl", effects)
    write_jsonl(out_dir / "phase298_rollout_rows.jsonl", rollouts)
    write_jsonl(out_dir / "phase298_missing_rows.jsonl", missing)
    write_json(V2 / "phase298_cross_model_summary.json", payload)
    write_jsonl(V2 / "phase298_mlp_causal_audit_rows.jsonl", audit)
    write_jsonl(V2 / "phase298_causal_effect_rows.jsonl", effects)
    write_jsonl(V2 / "phase298_rollout_rows.jsonl", rollouts)
    write_jsonl(V2 / "phase298_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase298 Expanded MLP Continue Causal Audit",
        "",
        f"- status: {payload['status']}",
        f"- selected_mlp_dominant_cases: {payload['selected_mlp_dominant_cases']}",
        f"- audit_rows: {payload['audit_rows']}",
        f"- causal_effect_rows: {payload['causal_effect_rows']}",
        f"- rollout_rows: {payload['rollout_rows']}",
        f"- missing_rows: {payload['missing_rows']}",
        f"- patch_counts: {json.dumps(payload['patch_counts'], ensure_ascii=False)}",
        f"- necessity_supported_counts: {json.dumps(payload['necessity_supported_counts'], ensure_ascii=False)}",
        f"- winner_changed_counts: {json.dumps(payload['winner_changed_counts'], ensure_ascii=False)}",
        f"- causal_support_level_counts: {json.dumps(payload['causal_support_level_counts'], ensure_ascii=False)}",
        f"- rollout_changed_counts: {json.dumps(payload['rollout_changed_counts'], ensure_ascii=False)}",
        f"- mean_delta_continue_stop_margin: {payload['mean_delta_continue_stop_margin']}",
        f"- mean_delta_target_logit: {payload['mean_delta_target_logit']}",
        "",
        "This is a low-side-effect causal audit on expanded Phase296 MLP-dominant samples, not closure.",
    ]
    (out_dir / "phase298_expanded_mlp_continue_causal_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--limit-per-model", type=int, default=0)
    parser.add_argument("--rollout-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        audit_model(args)
        return
    for model in MODELS:
        args.model = model
        audit_model(args)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
