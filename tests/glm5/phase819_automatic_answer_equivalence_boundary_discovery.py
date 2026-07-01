#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase818_alias_span_candidate_scoring_benchmark as p818  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402


PHASE = 819
SOURCE_816 = Path("tests/result/phase816_multi_token_answer_span_rollout_closure")
SOURCE_817 = Path("tests/result/phase817_alias_aware_answer_span_audit")
SOURCE_818 = Path("tests/result/phase818_alias_span_candidate_scoring_benchmark")
RESULT_ROOT = Path("tests/result/phase819_automatic_answer_equivalence_boundary_discovery")


FORMAT_PATTERNS = [
    re.compile(r"^_+$"),
    re.compile(r"^\?+$"),
    re.compile(r"^\[?\s*answer\s*\]?$", re.I),
    re.compile(r"^</?think>?$", re.I),
    re.compile(r"answer must be", re.I),
    re.compile(r"choose exactly", re.I),
    re.compile(r"the correct phrase is", re.I),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_text(value: Any) -> str:
    return p818.norm_text(value)


def token_set(text: str) -> set[str]:
    return {tok for tok in norm_text(text).split() if tok}


def overlap_score(phrase: str, references: list[str]) -> float:
    toks = token_set(phrase)
    if not toks:
        return 0.0
    best = 0.0
    for ref in references:
        rtoks = token_set(ref)
        if not rtoks:
            continue
        inter = len(toks & rtoks)
        union = len(toks | rtoks)
        score = inter / union if union else 0.0
        best = max(best, score)
    return best


def is_format_echo(text: Any) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if any(pat.search(raw) for pat in FORMAT_PATTERNS):
        return True
    if raw.count("_") >= 3:
        return True
    return False


def phase818_case(case_id: str) -> dict[str, Any] | None:
    for case in p818.p816.CASES:
        if case["case_id"] == case_id:
            return case
    return None


def classify_phrase(case: dict[str, Any], phrase: Any, source_class: str | None = None) -> dict[str, Any]:
    text = str(phrase or "").strip()
    norm = norm_text(text)
    target_aliases = p818.case_target_aliases(case)
    near_misses = p818.case_near_misses(case)
    wrongs = p818.unique_phrases([case["contrast_answer"], *case.get("distractors", [])])
    generic = p818.p816.GENERIC_BLOCKERS

    target_ok, target_match, target_kind = p818.class_match(text, target_aliases)
    if not target_ok:
        gen_norm = norm_text(text)
        for alias in target_aliases:
            alias_norm = norm_text(alias)
            if gen_norm.endswith("s") and gen_norm[:-1] == alias_norm:
                target_ok, target_match, target_kind = True, alias, "simple_plural_alias"
                break
            if alias_norm.endswith("s") and alias_norm[:-1] == gen_norm:
                target_ok, target_match, target_kind = True, alias, "simple_singular_alias"
                break
    near_ok, near_match, near_kind = p818.class_match(text, near_misses)
    wrong_ok, wrong_match, wrong_kind = p818.class_match(text, wrongs)
    generic_ok, generic_match, generic_kind = p818.class_match(text, generic)
    if target_ok or source_class == "target_alias":
        boundary_class = "target_equivalent"
        matched = target_match
        match_kind = target_kind
    elif near_ok or source_class == "near_miss":
        score = overlap_score(text, target_aliases)
        if score >= 0.34:
            boundary_class = "close_near_miss"
        else:
            boundary_class = "broad_near_miss"
        matched = near_match
        match_kind = near_kind
    elif wrong_ok or source_class == "wrong":
        boundary_class = "wrong"
        matched = wrong_match
        match_kind = wrong_kind
    elif generic_ok or source_class == "generic_blocker":
        boundary_class = "generic_blocker"
        matched = generic_match
        match_kind = generic_kind
    elif is_format_echo(text):
        boundary_class = "format_echo"
        matched = None
        match_kind = "format_pattern"
    else:
        boundary_class = "unknown_other"
        matched = None
        match_kind = "no_boundary_match"

    strict_accept = boundary_class == "target_equivalent"
    medium_accept = boundary_class in {"target_equivalent", "close_near_miss"}
    loose_accept = boundary_class in {"target_equivalent", "close_near_miss", "broad_near_miss"}
    return {
        "generated_clean": text,
        "generated_norm": norm,
        "boundary_class": boundary_class,
        "matched_boundary_phrase": matched,
        "boundary_match_kind": match_kind,
        "target_overlap_score": overlap_score(text, target_aliases),
        "strict_accept": strict_accept,
        "medium_accept": medium_accept,
        "loose_accept": loose_accept,
    }


def best_score(row: dict[str, Any], key: str) -> float | None:
    data = row.get(key) or {}
    val = data.get("score_mean_logprob")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def gt(lhs: float | None, rhs: float | None) -> bool:
    return lhs is not None and rhs is not None and lhs > rhs


def max_score(*values: float | None) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def score_reanalysis(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    alias = best_score(row, "best_target_alias_class")
    near = best_score(row, "best_near_miss")
    wrong = best_score(row, "best_wrong")
    generic = best_score(row, "best_generic_blocker")
    best_near_phrase = ((row.get("best_near_miss") or {}).get("phrase") or (row.get("best_near_miss") or {}).get("variant_text") or "")
    near_boundary = classify_phrase(case, best_near_phrase, "near_miss")["boundary_class"] if near is not None else "none"

    strict_score = gt(alias, max_score(near, wrong, generic))
    if near_boundary == "close_near_miss":
        medium_accept = max_score(alias, near)
        medium_compete = max_score(wrong, generic)
        loose_accept = medium_accept
        loose_compete = medium_compete
    else:
        medium_accept = alias
        medium_compete = max_score(near, wrong, generic)
        loose_accept = max_score(alias, near)
        loose_compete = max_score(wrong, generic)
    return {
        "best_alias_score": alias,
        "best_near_score": near,
        "best_wrong_score": wrong,
        "best_generic_score": generic,
        "best_near_boundary_class": near_boundary,
        "strict_score_closure": strict_score,
        "medium_score_closure": gt(medium_accept, medium_compete),
        "loose_score_closure": gt(loose_accept, loose_compete),
    }


def collect_phase816(rounds: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rnd in rounds:
        for model in MODELS:
            for row in read_jsonl(SOURCE_816 / rnd / f"phase816_{model}_rows.jsonl"):
                case = phase818_case(str(row.get("case_id")))
                if not case:
                    continue
                cls = "target_alias" if row.get("rollout_closure") else row.get("generation_match_class")
                boundary = classify_phrase(case, row.get("generated_clean"), cls)
                out.append(
                    {
                        "row_kind": "phase819_boundary_observation",
                        "phase": PHASE,
                        "source_phase": 816,
                        "round": rnd,
                        "model": model,
                        "case_id": row.get("case_id"),
                        "object": row.get("object"),
                        "prompt_variant": row.get("prompt_variant"),
                        "target_answer": row.get("target_answer"),
                        "source_generation_class": row.get("generation_match_class"),
                        **boundary,
                    }
                )
    return out


def collect_phase817(rounds: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rnd in rounds:
        for model in MODELS:
            for row in read_jsonl(SOURCE_817 / rnd / f"phase817_{model}_rows.jsonl"):
                case = phase818_case(str(row.get("case_id")))
                if not case:
                    continue
                boundary = classify_phrase(case, row.get("generated_clean"), None)
                out.append(
                    {
                        "row_kind": "phase819_boundary_observation",
                        "phase": PHASE,
                        "source_phase": 817,
                        "round": rnd,
                        "model": model,
                        "case_id": row.get("case_id"),
                        "prompt_variant": row.get("prompt_variant"),
                        "target_answer": row.get("target_answer"),
                        "source_generation_class": "alias_rollout" if row.get("alias_rollout_closure") else "not_alias",
                        **boundary,
                    }
                )
    return out


def collect_phase818(rounds: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rnd in rounds:
        for model in MODELS:
            for row in read_jsonl(SOURCE_818 / rnd / f"phase818_{model}_rows.jsonl"):
                case = phase818_case(str(row.get("case_id")))
                if not case:
                    continue
                boundary = classify_phrase(case, row.get("generated_clean"), row.get("generation_class"))
                scores = score_reanalysis(row, case)
                out.append(
                    {
                        "row_kind": "phase819_boundary_observation",
                        "phase": PHASE,
                        "source_phase": 818,
                        "round": rnd,
                        "model": model,
                        "case_id": row.get("case_id"),
                        "object": row.get("object"),
                        "prompt_variant": row.get("prompt_variant"),
                        "target_answer": row.get("target_answer"),
                        "source_generation_class": row.get("generation_class"),
                        "source_phase818_label": row.get("phase818_label"),
                        "strict_full_reanalysis": bool(scores["strict_score_closure"] and boundary["strict_accept"] and row.get("wrong_cleared") and row.get("generic_blocker_cleared")),
                        "medium_full_reanalysis": bool(scores["medium_score_closure"] and boundary["medium_accept"] and row.get("wrong_cleared") and row.get("generic_blocker_cleared")),
                        "loose_full_reanalysis": bool(scores["loose_score_closure"] and boundary["loose_accept"] and row.get("wrong_cleared") and row.get("generic_blocker_cleared")),
                        **scores,
                        **boundary,
                    }
                )
    return out


def phrase_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("generated_norm"):
            continue
        groups[(str(row.get("case_id")), str(row.get("generated_norm")))].append(row)
    out = []
    for (case_id, norm), vals in sorted(groups.items()):
        classes = Counter(v.get("boundary_class") for v in vals)
        models = Counter(v.get("model") for v in vals)
        prompts = Counter(v.get("prompt_variant") for v in vals)
        rounds = Counter(v.get("round") for v in vals)
        phases = Counter(str(v.get("source_phase")) for v in vals)
        exemplar = vals[0]
        out.append(
            {
                "row_kind": "phase819_phrase_aggregate",
                "phase": PHASE,
                "case_id": case_id,
                "target_answer": exemplar.get("target_answer"),
                "generated_norm": norm,
                "examples": sorted({str(v.get("generated_clean")) for v in vals if v.get("generated_clean")})[:12],
                "n_observations": len(vals),
                "boundary_classes": dict(classes),
                "dominant_boundary_class": classes.most_common(1)[0][0] if classes else None,
                "models": dict(models),
                "prompt_variants": dict(prompts),
                "rounds": dict(rounds),
                "source_phases": dict(phases),
                "strict_accept": any(v.get("strict_accept") for v in vals),
                "medium_accept": any(v.get("medium_accept") for v in vals),
                "loose_accept": any(v.get("loose_accept") for v in vals),
                "target_overlap_score_max": max(float(v.get("target_overlap_score") or 0.0) for v in vals),
            }
        )
    return out


def summarize_phase818_primary(rows: list[dict[str, Any]], primary_round: str) -> dict[str, Any]:
    vals = [r for r in rows if r.get("source_phase") == 818 and r.get("round") == primary_round]
    by_model_prompt: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for prompt in ["exact_choices", "no_choices"]:
            sub = [r for r in vals if r.get("model") == model and r.get("prompt_variant") == prompt]
            if not sub:
                continue
            key = f"{model}/{prompt}"
            by_model_prompt[key] = {
                "n": len(sub),
                "strict_rollout": sum(1 for r in sub if r.get("strict_accept")),
                "medium_rollout": sum(1 for r in sub if r.get("medium_accept")),
                "loose_rollout": sum(1 for r in sub if r.get("loose_accept")),
                "strict_full_reanalysis": sum(1 for r in sub if r.get("strict_full_reanalysis")),
                "medium_full_reanalysis": sum(1 for r in sub if r.get("medium_full_reanalysis")),
                "loose_full_reanalysis": sum(1 for r in sub if r.get("loose_full_reanalysis")),
                "boundary_classes": dict(Counter(r.get("boundary_class") for r in sub)),
                "score_boundaries": {
                    "strict_score": sum(1 for r in sub if r.get("strict_score_closure")),
                    "medium_score": sum(1 for r in sub if r.get("medium_score_closure")),
                    "loose_score": sum(1 for r in sub if r.get("loose_score_closure")),
                },
            }
    return by_model_prompt


def case_summary(rows: list[dict[str, Any]], phrase_aggs: list[dict[str, Any]], primary_round: str) -> list[dict[str, Any]]:
    out = []
    by_case_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case_phrases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case_rows[str(row.get("case_id"))].append(row)
    for row in phrase_aggs:
        by_case_phrases[str(row.get("case_id"))].append(row)
    for case_id, vals in sorted(by_case_rows.items()):
        case = phase818_case(case_id)
        primary = [r for r in vals if r.get("source_phase") == 818 and r.get("round") == primary_round]
        out.append(
            {
                "case_id": case_id,
                "object": case.get("object") if case else None,
                "target_answer": case.get("answer") if case else None,
                "target_aliases": p818.case_target_aliases(case) if case else [],
                "near_misses": p818.case_near_misses(case) if case else [],
                "n_observations": len(vals),
                "n_unique_generated_phrases": len(by_case_phrases.get(case_id, [])),
                "phrase_boundary_classes": dict(Counter(p.get("dominant_boundary_class") for p in by_case_phrases.get(case_id, []))),
                "primary_boundary_classes": dict(Counter(r.get("boundary_class") for r in primary)),
                "primary_strict_rollout": sum(1 for r in primary if r.get("strict_accept")),
                "primary_medium_rollout": sum(1 for r in primary if r.get("medium_accept")),
                "primary_loose_rollout": sum(1 for r in primary if r.get("loose_accept")),
                "boundary_ambiguous_phrases": [
                    p
                    for p in by_case_phrases.get(case_id, [])
                    if p.get("dominant_boundary_class") in {"close_near_miss", "broad_near_miss", "unknown_other"}
                ][:20],
            }
        )
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 819 Automatic Answer Equivalence Boundary Discovery",
        "",
        "- Source: offline reanalysis of Phase 816-818 generated phrases and Phase 818 candidate-score rows.",
        "- Boundary: no model loading; this phase fixes strict / medium / loose answer-boundary standards before returning to causal localization.",
        "",
        "## Primary Confirm Summary",
        "",
        "| model/prompt | n | strict rollout | medium rollout | loose rollout | strict full | medium full | loose full | boundary classes | score closures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for key, row in sorted(payload["primary_model_prompt_summary"].items()):
        lines.append(
            f"| {key} | {row['n']} | {row['strict_rollout']} | {row['medium_rollout']} | {row['loose_rollout']} | "
            f"{row['strict_full_reanalysis']} | {row['medium_full_reanalysis']} | {row['loose_full_reanalysis']} | "
            f"`{json.dumps(row['boundary_classes'], ensure_ascii=False)}` | "
            f"`{json.dumps(row['score_boundaries'], ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Phrase Boundary Distribution",
        "",
        f"- Total observations: {payload['n_boundary_rows']}",
        f"- Unique generated phrase aggregates: {payload['n_phrase_aggregates']}",
        f"- Boundary classes: `{json.dumps(payload['boundary_class_distribution'], ensure_ascii=False)}`",
        "",
        "## Most Frequent Phrase Aggregates",
        "",
        "| case | target | phrase | n | class | models | prompts |",
        "|---|---|---|---:|---|---|---|",
    ]
    top = sorted(payload["phrase_aggregates"], key=lambda r: (-int(r["n_observations"]), r["case_id"], r["generated_norm"]))[:40]
    for row in top:
        lines.append(
            f"| {row['case_id']} | `{row.get('target_answer')}` | `{row.get('generated_norm')}` | "
            f"{row['n_observations']} | `{row.get('dominant_boundary_class')}` | "
            f"`{json.dumps(row.get('models') or {}, ensure_ascii=False)}` | "
            f"`{json.dumps(row.get('prompt_variants') or {}, ensure_ascii=False)}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rounds = [x.strip() for x in str(args.rounds).split(",") if x.strip()]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(collect_phase816(rounds))
    rows.extend(collect_phase817(rounds))
    rows.extend(collect_phase818(rounds))
    phrases = phrase_aggregates(rows)
    cases = case_summary(rows, phrases, args.primary_round)
    primary = summarize_phase818_primary(rows, args.primary_round)
    payload = {
        "phase": PHASE,
        "title": "Automatic Answer Equivalence Boundary Discovery",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_phases": [816, 817, 818],
        "rounds": rounds,
        "primary_round": args.primary_round,
        "n_boundary_rows": len(rows),
        "n_phrase_aggregates": len(phrases),
        "n_cases": len(cases),
        "boundary_class_distribution": dict(Counter(row.get("boundary_class") for row in rows)),
        "primary_model_prompt_summary": primary,
        "phrase_aggregates": phrases[:200],
        "case_summary": cases,
        "boundary": (
            "This phase does not claim final semantic equivalence. It builds a reproducible boundary table and strict/medium/loose closure reanalysis."
        ),
    }
    write_jsonl(RESULT_ROOT / "phase819_boundary_rows.jsonl", rows)
    write_jsonl(RESULT_ROOT / "phase819_phrase_aggregates.jsonl", phrases)
    write_json(RESULT_ROOT / "phase819_case_summary.json", cases)
    write_json(RESULT_ROOT / "phase819_summary.json", payload)
    write_markdown(RESULT_ROOT / "phase819_summary.md", payload)
    print(json.dumps({"phase": PHASE, "rows": len(rows), "phrases": len(phrases), "classes": payload["boundary_class_distribution"]}, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="smoke,main,confirm")
    parser.add_argument("--primary-round", default="confirm")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
