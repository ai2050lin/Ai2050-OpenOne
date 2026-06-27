#!/usr/bin/env python3
"""
Phase 711: Global Mechanism Atlas v0.

This is the first atlas-building phase. It does not run model inference. It
consolidates already-tested evidence from Phase 698-710 into a queryable
mechanism-unit table.

The goal is intentionally modest:
  - start with a language-core micro-atlas, not an all-model/all-capability map;
  - label mechanism roles before claiming semantic neurons;
  - keep model specificity and evidence level visible.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


OUT_ROOT = Path("results/glm5_phase711_global_mechanism_atlas_v0")
MODELS = ["deepseek7b", "glm4", "qwen3"]
PHASE_ROOTS = {
    "p698": Path("results/glm5_phase698_answer_last_attention_head_source_audit"),
    "p707": Path("results/glm5_phase707_full_value_phrase_likelihood_audit"),
    "p709": Path("results/glm5_phase709_natural_generation_writein_closure"),
    "p710": Path("results/glm5_phase710_natural_writein_factor_split"),
}


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(dct: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    return default if dct is None else dct.get(key, default)


def get_condition(summary: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not summary:
        return None
    return summary.get("summary", {}).get("by_condition", {}).get(key)


def get_phase707_condition(summary: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not summary:
        return None
    by_split = summary.get("summary", {}).get("by_split_condition", {})
    if key in by_split:
        return by_split[key]
    # Older Phase707 summaries also expose aggregate rows via best lists.
    for group_name in ("best_restore_conditions", "best_degradation_conditions"):
        for row in summary.get("summary", {}).get(group_name, []):
            if row.get("condition") == key:
                return row
    return None


def status_from_scores(natural: dict[str, Any] | None, phrase: dict[str, Any] | None) -> str:
    target = float(safe_get(natural, "target_value_rate", 0.0) or 0.0)
    prose = float(safe_get(natural, "prose_target_rate", 0.0) or 0.0)
    donor = float(safe_get(natural, "donor_value_rate", 0.0) or 0.0)
    phrase_target = float(safe_get(phrase, "target_phrase_win_rate", 0.0) or 0.0)
    phrase_donor = float(safe_get(phrase, "donor_phrase_win_rate", 0.0) or 0.0)

    if donor >= 0.05 or phrase_donor >= 0.05:
        return "mixed_value_residue"
    if target >= 0.35 and target >= prose:
        return "short_value_route_carrier"
    if prose > target and (prose >= 0.35 or phrase_target >= 0.35):
        return "prose_or_format_route_carrier"
    if target + prose >= 0.70:
        return "target_side_route_carrier"
    return "unresolved_or_weak"


def evidence_level(natural: dict[str, Any] | None, phrase: dict[str, Any] | None) -> str:
    if natural and natural.get("n", 0) and natural.get("target_or_target_prose_rate", 0.0) >= 0.7:
        return "level6_partial_natural_generation"
    if phrase and phrase.get("n", 0):
        return "level5_phrase_likelihood"
    return "level4_component_or_lower"


def role_scores(natural: dict[str, Any] | None, phrase: dict[str, Any] | None, phase710: dict[str, Any] | None) -> dict[str, Any]:
    target = float(safe_get(natural, "target_value_rate", 0.0) or 0.0)
    prose = float(safe_get(natural, "prose_target_rate", 0.0) or 0.0)
    donor = float(safe_get(natural, "donor_value_rate", 0.0) or 0.0)
    target_side = float(safe_get(natural, "target_or_target_prose_rate", target + prose) or 0.0)
    phrase_td = safe_get(phrase, "mean_phrase_target_minus_donor")
    phrase_tp = safe_get(phrase, "mean_phrase_target_minus_prose")
    post_layer_target = float(safe_get(phase710, "target_value_rate", target) or 0.0)
    return {
        "route_gain_score": target_side,
        "identity_score": max(0.0, min(1.0, target - donor)),
        "format_or_prose_score": prose,
        "donor_residue_score": donor,
        "phrase_target_minus_donor": phrase_td,
        "phrase_target_minus_prose": phrase_tp,
        "post_layer_target_value_rate": post_layer_target,
    }


def build_model_atlas(model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p698_scores = read_json(PHASE_ROOTS["p698"] / f"phase698_{model}_candidate_scores.json") or []
    p707 = read_json(PHASE_ROOTS["p707"] / f"phase707_{model}_full_phrase_likelihood_summary.json")
    p709 = read_json(PHASE_ROOTS["p709"] / f"phase709_{model}_natural_generation_summary.json")
    p710 = read_json(PHASE_ROOTS["p710"] / f"phase710_{model}_factor_split_summary.json")

    natural_top = get_condition(p709, "unrelated|restore|source_top_channel_512")
    natural_all = get_condition(p709, "unrelated|restore|all_positive_source_channels")
    phrase_top = get_phase707_condition(p707, "unrelated|restore|source_top_channel_512")
    p710_pre = get_condition(p710, "pre_o_input|unrelated|restore|source_top_channel_512")
    p710_post_o = get_condition(p710, "post_o_output|unrelated|restore|source_top_channel_512")
    p710_post_layer = get_condition(p710, "post_layer_output|unrelated|restore|source_top_channel_512")

    for rank, head in enumerate(p698_scores[:32], 1):
        layer = int(head["layer"])
        h = int(head["head"])
        rows.append({
            "unit_id": f"{model}_L{layer}_H{h}",
            "unit_type": "attention_head",
            "model": model,
            "layer": layer,
            "head": h,
            "channel": None,
            "rank_in_phase698": rank,
            "source_group": "target_value+answer_line+self_last",
            "target_position": "answer_start",
            "mechanism_scope": "language_core_micro_atlas_v0",
            "phase698_mean_direct_effect": head.get("mean_direct_effect"),
            "phase698_mean_delta_norm": head.get("mean_delta_norm"),
            "role_scores": role_scores(natural_top, phrase_top, p710_post_layer),
            "status": status_from_scores(natural_top, phrase_top),
            "evidence_level": evidence_level(natural_top, phrase_top),
            "phase709_unrelated_restore_top512": natural_top,
            "phase710_pre_o_input_top512": p710_pre,
            "phase710_post_o_output_top512": p710_post_o,
            "phase710_post_layer_output_top512": p710_post_layer,
            "cross_model_status": "model_specific_primary" if model == "deepseek7b" else "weak_cross_model_reference",
        })

    top_channels = []
    donor_sets = p707.get("summary", {}).get("donor_sets", []) if p707 else []
    if donor_sets:
        top_channels = donor_sets[0].get("top_channel_scores", [])

    for rank, ch in enumerate(top_channels[:64], 1):
        layer = int(ch["layer"])
        h = int(ch["head"])
        c = int(ch["channel"])
        rows.append({
            "unit_id": f"{model}_L{layer}_H{h}_C{c}",
            "unit_type": "attention_channel",
            "model": model,
            "layer": layer,
            "head": h,
            "channel": c,
            "rank_in_phase707_channel_scores": rank,
            "source_group": "target_value+answer_line+self_last",
            "target_position": "answer_start",
            "mechanism_scope": "language_core_micro_atlas_v0",
            "mean_direct_effect": ch.get("mean_direct_effect"),
            "mean_abs_effect": ch.get("mean_abs_effect"),
            "mean_combo_delta_value": ch.get("mean_combo_delta_value"),
            "mean_output_dir_proj": ch.get("mean_output_dir_proj"),
            "n_train_cases": ch.get("n_train_cases"),
            "role_scores": role_scores(natural_top, phrase_top, p710_post_layer),
            "status": status_from_scores(natural_top, phrase_top),
            "evidence_level": evidence_level(natural_top, phrase_top),
            "phase709_unrelated_restore_top512": natural_top,
            "phase709_unrelated_restore_all_positive": natural_all,
            "phase710_pre_o_input_top512": p710_pre,
            "phase710_post_o_output_top512": p710_post_o,
            "phase710_post_layer_output_top512": p710_post_layer,
            "cross_model_status": "model_specific_primary" if model == "deepseek7b" else "weak_cross_model_reference",
        })

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = Counter(row["model"] for row in rows)
    by_type = Counter(row["unit_type"] for row in rows)
    by_status = Counter(row["status"] for row in rows)
    by_level = Counter(row["evidence_level"] for row in rows)
    model_status = defaultdict(Counter)
    for row in rows:
        model_status[row["model"]][row["status"]] += 1
    return {
        "n_units": len(rows),
        "by_model": dict(by_model),
        "by_type": dict(by_type),
        "by_status": dict(by_status),
        "by_evidence_level": dict(by_level),
        "model_status": {model: dict(counter) for model, counter in model_status.items()},
    }


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 711 Global Mechanism Atlas v0",
        "",
        f"- generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- units: `{summary['n_units']}`",
        "",
        "## Summary",
        "",
        f"- by_model: `{summary['by_model']}`",
        f"- by_type: `{summary['by_type']}`",
        f"- by_status: `{summary['by_status']}`",
        f"- by_evidence_level: `{summary['by_evidence_level']}`",
        "",
        "## Top Units",
        "",
        "| unit_id | type | status | evidence | route | identity | prose | donor | phase698/direct |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        rows,
        key=lambda r: (
            r["model"] != "deepseek7b",
            r["unit_type"] != "attention_channel",
            -(r.get("mean_direct_effect") or r.get("phase698_mean_direct_effect") or 0.0),
        ),
    )
    for row in ordered[:80]:
        scores = row["role_scores"]
        direct = row.get("mean_direct_effect", row.get("phase698_mean_direct_effect"))
        lines.append(
            f"| {row['unit_id']} | {row['unit_type']} | {row['status']} | {row['evidence_level']} | "
            f"{scores['route_gain_score']:.3f} | {scores['identity_score']:.3f} | {scores['format_or_prose_score']:.3f} | "
            f"{scores['donor_residue_score']:.3f} | {(direct or 0.0):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.extend(build_model_atlas(model))
    summary = summarize(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (OUT_ROOT / "phase711_atlas_units.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 711,
        "title": "Global Mechanism Atlas v0",
        "timestamp": timestamp,
        "models": models,
        "summary": summary,
        "sources": {k: str(v) for k, v in PHASE_ROOTS.items()},
        "notes": [
            "Atlas v0 is a mechanism-role atlas, not a full semantic neuron atlas.",
            "Rows inherit condition-level causal/generation evidence; unit-local labels remain provisional.",
            "DS7B is primary; qwen3/GLM4 are weak cross-model references because paired cases are sparse.",
        ],
    }
    (OUT_ROOT / "phase711_atlas_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(rows, summary, OUT_ROOT / "phase711_atlas_summary.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
