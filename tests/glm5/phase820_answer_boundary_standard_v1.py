#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase819_automatic_answer_equivalence_boundary_discovery as p819  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402


PHASE = 820
SOURCE_818 = Path("tests/result/phase818_alias_span_candidate_scoring_benchmark")
SOURCE_819 = Path("tests/result/phase819_automatic_answer_equivalence_boundary_discovery")
RESULT_ROOT = Path("tests/result/phase820_answer_boundary_standard_v1")


OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    ("p816_cactus_desert_plant", "cactus"): ("object_echo", "Echoes the object rather than giving the requested category phrase."),
    ("p816_cactus_desert_plant", "cactus plant"): ("close_near_miss", "Correct object-specific plant phrase, but misses desert-plant granularity."),
    ("p816_cactus_desert_plant", "cactus plants"): ("close_near_miss", "Plural object-specific plant phrase; related but not strict desert-plant category."),
    ("p816_carrot_root_vegetable", "vegetable"): ("close_near_miss", "Correct broad category, but misses root-vegetable granularity."),
    ("p816_carrot_root_vegetable", "vegetables"): ("close_near_miss", "Plural broad category; acceptable only as medium/loose, not strict target."),
    ("p816_cactus_desert_plant", "plant life cycle"): ("unknown_other", "Related plant process, but not a category phrase for cactus."),
    ("p816_hammer_hand_tool", "hammer is a hand tool"): ("format_with_target", "Contains the target phrase but violates write-only-the-phrase protocol."),
    ("p816_heart_body_organ", "circulatory system"): ("broad_near_miss", "Related biological system, not the requested organ category."),
    ("p816_heart_body_organ", "human body part"): ("close_near_miss", "Close parent category; broader than body organ."),
    ("p816_laptop_electronic_device", "electronics"): ("close_near_miss", "Related electronics category; slightly broader than electronic device."),
    ("p816_laptop_electronic_device", "personal computing device"): ("target_equivalent", "Acceptable specific category for laptop under electronic-device task."),
    ("p816_oxygen_chemical_element", "gas"): ("broad_near_miss", "True physical state here, but not the requested chemical-element category."),
    ("p816_oxygen_chemical_element", "o2"): ("object_echo", "Chemical formula/object identity, not category phrase."),
    ("p816_red_warm_color", "color"): ("broad_near_miss", "Correct parent category, but misses warm-color granularity."),
    ("p816_red_warm_color", "color category"): ("broad_near_miss", "Parent category with category suffix; not warm-color equivalent."),
    ("p816_salmon_aquatic_animal", "freshwater fish"): ("target_equivalent", "Specific correct category for salmon under aquatic-animal task."),
    ("p816_salmon_aquatic_animal", "salmon is best described as a freshwater fish"): (
        "format_with_target",
        "Semantically answers the question but violates phrase-only protocol.",
    ),
    ("p816_salmon_aquatic_animal", "the correct phrase is aquatic animal"): (
        "format_with_target",
        "Contains target phrase but adds explanation/prefix.",
    ),
    ("p816_triangle_geometric_shape", "geometry"): ("broad_near_miss", "Related mathematical domain, not the requested shape category."),
    ("p816_triangle_geometric_shape", "triangle"): ("object_echo", "Echoes the object rather than giving its category."),
    ("p816_winter_cold_season", "cold weather"): ("broad_near_miss", "Related condition, not a season category."),
    ("p816_winter_cold_season", "seasonal weather patterns"): ("unknown_other", "Related weather phrase, but not a category phrase for winter."),
    ("p816_winter_cold_season", "winter"): ("object_echo", "Echoes the object rather than giving its category."),
}


ACCEPTANCE = {
    "target_equivalent": (True, True, True, True, True),
    "close_near_miss": (False, True, True, False, True),
    "broad_near_miss": (False, False, True, False, True),
    "wrong": (False, False, False, False, False),
    "generic_blocker": (False, False, False, False, False),
    "format_echo": (False, False, False, False, False),
    "format_with_target": (False, False, False, True, False),
    "object_echo": (False, False, False, False, False),
    "unknown_other": (False, False, False, False, False),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_text(value: Any) -> str:
    return p819.norm_text(value)


def finalize_class(row: dict[str, Any]) -> tuple[str, str]:
    key = (str(row.get("case_id")), str(row.get("generated_norm")))
    if key in OVERRIDES:
        return OVERRIDES[key]
    cls = str(row.get("dominant_boundary_class") or "unknown_other")
    return cls, "Inherited from Phase 819 boundary atlas v1 rule."


def standard_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_819 / "phase819_phrase_aggregates.jsonl")
    out = []
    for row in rows:
        final_class, reason = finalize_class(row)
        strict, medium, loose, semantic_target_present, protocol_valid = ACCEPTANCE.get(final_class, ACCEPTANCE["unknown_other"])
        out.append(
            {
                "row_kind": "phase820_answer_boundary_standard_v1",
                "phase": PHASE,
                "case_id": row.get("case_id"),
                "target_answer": row.get("target_answer"),
                "generated_norm": row.get("generated_norm"),
                "examples": row.get("examples", []),
                "n_observations": row.get("n_observations"),
                "models": row.get("models", {}),
                "prompt_variants": row.get("prompt_variants", {}),
                "phase819_class": row.get("dominant_boundary_class"),
                "final_boundary_class": final_class,
                "strict_accept": strict,
                "medium_accept": medium,
                "loose_accept": loose,
                "semantic_target_present": semantic_target_present,
                "protocol_valid": protocol_valid,
                "review_reason": reason,
            }
        )
    out.sort(key=lambda r: (str(r["case_id"]), str(r["generated_norm"])))
    return out


def standard_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["case_id"]), str(row["generated_norm"])): row for row in rows}


def fallback_standard(case_id: str, phrase: Any) -> dict[str, Any]:
    override = OVERRIDES.get((case_id, norm_text(phrase)))
    if override:
        cls, reason = override
    else:
        cls = ""
        reason = ""
    case = p819.phase818_case(case_id)
    if cls:
        pass
    elif not case:
        cls = "unknown_other"
        reason = "No case metadata found."
    else:
        boundary = p819.classify_phrase(case, phrase, None)
        cls = boundary["boundary_class"]
        reason = "Fallback classified by Phase 819 boundary rule."
    strict, medium, loose, semantic_target_present, protocol_valid = ACCEPTANCE.get(cls, ACCEPTANCE["unknown_other"])
    return {
        "case_id": case_id,
        "generated_norm": norm_text(phrase),
        "final_boundary_class": cls,
        "strict_accept": strict,
        "medium_accept": medium,
        "loose_accept": loose,
        "semantic_target_present": semantic_target_present,
        "protocol_valid": protocol_valid,
        "review_reason": reason,
    }


def class_for_phrase(lookup: dict[tuple[str, str], dict[str, Any]], case_id: str, phrase: Any) -> dict[str, Any]:
    key = (case_id, norm_text(phrase))
    return lookup.get(key) or fallback_standard(case_id, phrase)


def best_score(row: dict[str, Any], key: str) -> float | None:
    data = row.get(key) or {}
    try:
        return float(data.get("score_mean_logprob"))
    except (TypeError, ValueError):
        return None


def max_or_none(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def gt(lhs: float | None, rhs: float | None) -> bool:
    return lhs is not None and rhs is not None and lhs > rhs


def reanalyze_phase818(round_name: str, standards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = standard_lookup(standards)
    out = []
    for model in MODELS:
        for row in read_jsonl(SOURCE_818 / round_name / f"phase818_{model}_rows.jsonl"):
            case_id = str(row.get("case_id"))
            gen_std = class_for_phrase(lookup, case_id, row.get("generated_clean"))
            near_phrase = ((row.get("best_near_miss") or {}).get("phrase") or (row.get("best_near_miss") or {}).get("variant_text") or "")
            near_std = class_for_phrase(lookup, case_id, near_phrase) if near_phrase else None
            alias_score = best_score(row, "best_target_alias_class")
            near_score = best_score(row, "best_near_miss")
            wrong_score = best_score(row, "best_wrong")
            generic_score = best_score(row, "best_generic_blocker")
            strict_score = gt(alias_score, max_or_none([near_score, wrong_score, generic_score]))
            if near_std and near_std.get("final_boundary_class") == "close_near_miss":
                medium_accept_score = max_or_none([alias_score, near_score])
                medium_compete_score = max_or_none([wrong_score, generic_score])
            else:
                medium_accept_score = alias_score
                medium_compete_score = max_or_none([near_score, wrong_score, generic_score])
            if near_std and near_std.get("final_boundary_class") in {"close_near_miss", "broad_near_miss"}:
                loose_accept_score = max_or_none([alias_score, near_score])
                loose_compete_score = max_or_none([wrong_score, generic_score])
            else:
                loose_accept_score = alias_score
                loose_compete_score = max_or_none([near_score, wrong_score, generic_score])
            medium_score = gt(medium_accept_score, medium_compete_score)
            loose_score = gt(loose_accept_score, loose_compete_score)
            wrong_cleared = bool(row.get("wrong_cleared"))
            generic_cleared = bool(row.get("generic_blocker_cleared"))
            out.append(
                {
                    "row_kind": "phase820_reanalysis_row",
                    "phase": PHASE,
                    "source_phase": 818,
                    "round": round_name,
                    "model": model,
                    "case_id": case_id,
                    "prompt_variant": row.get("prompt_variant"),
                    "target_answer": row.get("target_answer"),
                    "generated_clean": row.get("generated_clean"),
                    "generated_norm": norm_text(row.get("generated_clean")),
                    "final_boundary_class": gen_std.get("final_boundary_class"),
                    "phase818_generation_class": row.get("generation_class"),
                    "strict_rollout": bool(gen_std.get("strict_accept")),
                    "medium_rollout": bool(gen_std.get("medium_accept")),
                    "loose_rollout": bool(gen_std.get("loose_accept")),
                    "semantic_target_present": bool(gen_std.get("semantic_target_present")),
                    "protocol_valid": bool(gen_std.get("protocol_valid")),
                    "strict_score_closure_v1": strict_score,
                    "medium_score_closure_v1": medium_score,
                    "loose_score_closure_v1": loose_score,
                    "wrong_cleared": wrong_cleared,
                    "generic_cleared": generic_cleared,
                    "strict_full_v1": bool(strict_score and gen_std.get("strict_accept") and wrong_cleared and generic_cleared),
                    "medium_full_v1": bool(medium_score and gen_std.get("medium_accept") and wrong_cleared and generic_cleared),
                    "loose_full_v1": bool(loose_score and gen_std.get("loose_accept") and wrong_cleared and generic_cleared),
                    "best_near_phrase": near_phrase,
                    "best_near_final_class": near_std.get("final_boundary_class") if near_std else None,
                }
            )
    return out


def summarize(standards: list[dict[str, Any]], rows: list[dict[str, Any]], round_name: str) -> dict[str, Any]:
    by_model_prompt = {}
    for model in MODELS:
        for prompt in ["exact_choices", "no_choices"]:
            vals = [r for r in rows if r["model"] == model and r["prompt_variant"] == prompt]
            if not vals:
                continue
            by_model_prompt[f"{model}/{prompt}"] = {
                "n": len(vals),
                "strict_rollout": sum(1 for r in vals if r["strict_rollout"]),
                "medium_rollout": sum(1 for r in vals if r["medium_rollout"]),
                "loose_rollout": sum(1 for r in vals if r["loose_rollout"]),
                "semantic_target_present": sum(1 for r in vals if r["semantic_target_present"]),
                "protocol_valid": sum(1 for r in vals if r["protocol_valid"]),
                "strict_full_v1": sum(1 for r in vals if r["strict_full_v1"]),
                "medium_full_v1": sum(1 for r in vals if r["medium_full_v1"]),
                "loose_full_v1": sum(1 for r in vals if r["loose_full_v1"]),
                "classes": dict(Counter(r["final_boundary_class"] for r in vals)),
                "score_closures": {
                    "strict": sum(1 for r in vals if r["strict_score_closure_v1"]),
                    "medium": sum(1 for r in vals if r["medium_score_closure_v1"]),
                    "loose": sum(1 for r in vals if r["loose_score_closure_v1"]),
                },
            }
    changed = [row for row in standards if row["phase819_class"] != row["final_boundary_class"]]
    payload = {
        "phase": PHASE,
        "title": "Answer Boundary Standard v1",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_phase": 819,
        "reanalyzed_phase": 818,
        "round": round_name,
        "n_standard_rows": len(standards),
        "n_reanalysis_rows": len(rows),
        "standard_class_distribution": dict(Counter(row["final_boundary_class"] for row in standards)),
        "changed_from_phase819": len(changed),
        "changed_rows": changed,
        "model_prompt_summary": by_model_prompt,
        "boundary": (
            "This is a v1 external evaluation standard. It is reproducible and reviewable, but not a final internal model equivalence class."
        ),
    }
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 820 Answer Boundary Standard v1",
        "",
        "- Source: Phase 819 phrase aggregates.",
        "- Boundary: external review standard for strict / medium / loose closure; no model loading.",
        "",
        "## Standard Distribution",
        "",
        f"- Standard rows: {payload['n_standard_rows']}",
        f"- Changed from Phase 819 heuristic class: {payload['changed_from_phase819']}",
        f"- Classes: `{json.dumps(payload['standard_class_distribution'], ensure_ascii=False)}`",
        "",
        "## Phase 818 Confirm Reanalysis",
        "",
        "| model/prompt | n | strict rollout | medium rollout | loose rollout | semantic target | protocol valid | strict full | medium full | loose full | classes | scores |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for key, row in sorted(payload["model_prompt_summary"].items()):
        lines.append(
            f"| {key} | {row['n']} | {row['strict_rollout']} | {row['medium_rollout']} | {row['loose_rollout']} | "
            f"{row['semantic_target_present']} | {row['protocol_valid']} | {row['strict_full_v1']} | "
            f"{row['medium_full_v1']} | {row['loose_full_v1']} | "
            f"`{json.dumps(row['classes'], ensure_ascii=False)}` | "
            f"`{json.dumps(row['score_closures'], ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Changed Rows",
        "",
        "| case | target | phrase | phase819 | final | reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["changed_rows"]:
        lines.append(
            f"| {row['case_id']} | `{row['target_answer']}` | `{row['generated_norm']}` | "
            f"`{row['phase819_class']}` | `{row['final_boundary_class']}` | {row['review_reason']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    standards = standard_rows()
    rows = reanalyze_phase818(args.round_name, standards)
    payload = summarize(standards, rows, args.round_name)
    write_jsonl(RESULT_ROOT / "answer_boundary_standard_v1.jsonl", standards)
    write_jsonl(RESULT_ROOT / "phase820_reanalysis_rows.jsonl", rows)
    write_json(RESULT_ROOT / "phase820_summary.json", payload)
    write_markdown(RESULT_ROOT / "phase820_summary.md", payload)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "standard_rows": payload["n_standard_rows"],
                "changed": payload["changed_from_phase819"],
                "classes": payload["standard_class_distribution"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="confirm")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
