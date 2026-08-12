#!/usr/bin/env python3
"""Freeze the Phase1122 lexical-coherence null audit for Phase1121."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1121_wordnet_adjective_double_orthogonal_protocol as source


PHASE = 1122
PROTOCOL_REVISION = 1
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1122_adjective_lexical_coherence_audit"
SOURCE_ROOT = source.OUT_ROOT
MODELS = tuple(source.MODELS)
REFERENCE_MODELS = tuple(source.REFERENCE_MODELS)
PRIMARY_METRIC = "target_ablated_content_tfidf_cosine"
SECONDARY_METRICS = (
    "target_ablated_raw_unigram_jaccard",
    "target_ablated_content_unigram_jaccard",
    "target_ablated_character_trigram_cosine",
    "full_content_tfidf_cosine",
)


STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does doing for from had has have "
    "he her hers herself him himself his how i if in into is it its itself may me might more most "
    "my myself no nor not of on once only or other our ours ourselves out over own same she should "
    "so some such than that the their theirs them themselves then there these they this those through "
    "to too under until up very was we were what when where which while who whom why will with would "
    "you your yours yourself yourselves".split()
)


THRESHOLDS = {
    "maximum_primary_null_direction_rate": 0.65,
    "maximum_any_secondary_null_direction_rate": 0.75,
    "minimum_behavior_advantage_over_primary": 0.20,
    "minimum_primary_adversarial_direction_accuracy": 0.75,
    "minimum_primary_adversarial_interaction_count": 36,
    "minimum_qualified_reference_models": 2,
}


PREDICTIONS = {
    "P1": "The Phase1121 protocol and result digests remain intact and all source rows map uniquely.",
    "P2": "The target-ablated primary lexical null has direction accuracy at most 0.65.",
    "P3": "No frozen secondary lexical null has direction accuracy above 0.75.",
    "P4": "At least two chat models exceed the primary null by 0.20 and retain 0.75 direction accuracy where that null is non-positive.",
    "P5": "A pass rules out only this frozen family of token-overlap explanations; it does not establish a hidden semantic mechanism.",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold())


def content_tokens(text: str) -> list[str]:
    return [token for token in word_tokens(text) if token not in STOPWORDS]


def remove_exact_term(text: str, term: str) -> str:
    return re.sub(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", " ", text, flags=re.IGNORECASE)


def validate_source() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source_prereg = read_json(SOURCE_ROOT / "protocol" / "preregistration.json")
    source_audit = read_json(SOURCE_ROOT / "protocol" / "audit.json")
    source_final = read_json(SOURCE_ROOT / "analysis" / "final_summary.json")
    selected = read_json(SOURCE_ROOT / "protocol" / "selected_concepts.json")["selected"]
    if not source_audit["all_checks_passed"]:
        raise RuntimeError("Phase1121 source protocol audit failed")
    if source_final["protocol_digest"] != source_prereg["protocol_digest"]:
        raise RuntimeError("Phase1121 source protocol digest mismatch")
    if len(selected) != 24 or len({row["concept_id"] for row in selected}) != 24:
        raise RuntimeError("Phase1121 selected concept identity mismatch")
    return source_prereg, source_final, selected


def main() -> None:
    source_prereg, source_final, selected = validate_source()
    material_rows: list[dict[str, Any]] = []
    for row in selected:
        for surface in source.SURFACES:
            for sense in source.SENSES:
                sentence = row["base_examples"][sense] if surface == "base" else row["synonym_examples"][sense]
                term = row["base"] if surface == "base" else row["synonym_surfaces"][sense]
                ablated = remove_exact_term(sentence, term)
                material_rows.append({
                    "concept_id": row["concept_id"],
                    "deranged_control_concept_id": row["deranged_control_concept_id"],
                    "split": row["split"],
                    "surface": surface,
                    "context_sense": sense,
                    "term": term,
                    "sentence": sentence,
                    "target_ablated_sentence": ablated,
                    "raw_tokens": word_tokens(ablated),
                    "content_tokens": content_tokens(ablated),
                    "definitions": row["definitions"],
                })
    material_core = {
        "schema_version": "phase1122_lexical_material.v1",
        "phase": PHASE,
        "source_selected_digest": source_prereg["selected_digest"],
        "rows": material_rows,
    }
    material = dict(material_core)
    material["material_digest"] = digest(material_core)
    write_json(OUT_ROOT / "protocol" / "material.json", material)

    prereg_core = {
        "schema_version": "phase1122_lexical_coherence_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": source.PHASE,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "source_final_digest": source_final["final_digest"],
        "source_selected_digest": source_prereg["selected_digest"],
        "material_digest": material["material_digest"],
        "models": list(MODELS),
        "reference_models": list(REFERENCE_MODELS),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "thresholds": THRESHOLDS,
        "predictions": PREDICTIONS,
        "tokenization": "lowercase ASCII word tokens; primary target term removed exactly; fixed stopword list",
        "interaction_formula": "0.5*((s_context0_definition0-s_context0_definition1)-(s_context1_definition0-s_context1_definition1))",
        "adversarial_subset": "source model interactions whose primary lexical-null interaction is non-positive",
        "model_outputs_read_during_protocol": False,
        "scope_limit": "This audit can reject frozen lexical-overlap nulls only; passing does not prove abstract semantics, hidden invariance, use, or causality.",
        "forbidden_actions": [
            "change tokenization, stopwords, metrics, thresholds, or target ablation after reading the audit result",
            "select a lexical metric by whichever best matches a model after reading model outputs",
            "drop a model, split, surface, template, or concept",
            "upgrade K57 beyond behavior level from this audit",
        ],
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)

    checks = {
        "source_protocol_passed": True,
        "source_protocol_digest_matches": source_final["protocol_digest"] == source_prereg["protocol_digest"],
        "source_final_digest_matches": source_final["final_digest"] == prereg["source_final_digest"],
        "concept_count_24": len(selected) == 24,
        "material_row_count_96": len(material_rows) == 96,
        "all_targets_removed": all(row["term"].casefold() not in word_tokens(row["target_ablated_sentence"]) for row in material_rows),
        "two_surfaces_per_concept_sense": len({(row["concept_id"], row["surface"], row["context_sense"]) for row in material_rows}) == 96,
        "three_splits_present": {row["split"] for row in material_rows} == set(source.SPLITS),
        "primary_not_secondary": PRIMARY_METRIC not in SECONDARY_METRICS,
        "protocol_digest": digest(prereg_core) == prereg["protocol_digest"],
    }
    audit_core = {
        "schema_version": "phase1122_lexical_coherence_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1122 protocol audit failed")
    print(json.dumps({"preregistration": prereg, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
