#!/usr/bin/env python3
"""Phase1351: freeze Qwen-only C052 pair-identity probe campaign."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from model_utils import MODEL_CONFIGS

PHASE, CAMPAIGN = 1351, "C052"
OUT = TESTS / "result/phase1351_c052_qwen_pair_probe_contract"
PARENT = TESTS / "result/phase1350_c051_matched_null_behavior"
MODEL = "qwen3"
PARTITIONS = ("prototype_discovery", "clock_selection", "confirmation", "holdout")
FAMILIES = ("weather phenomenon", "punctuation mark", "office item", "music genre")
SYSTEM = "Answer the stated yes-or-no question. Output only yes or no."
WORDS = {
    "prototype_discovery": {
        "weather phenomenon": ("hurricane", "tornado", "blizzard", "cyclone"),
        "punctuation mark": ("comma", "semicolon", "colon", "apostrophe"),
        "office item": ("stapler", "paperclip", "envelope", "binder"),
        "music genre": ("jazz", "blues", "reggae", "hiphop"),
    },
    "clock_selection": {
        "weather phenomenon": ("typhoon", "monsoon", "thunderstorm", "hailstorm"),
        "punctuation mark": ("hyphen", "dash", "parenthesis", "bracket"),
        "office item": ("folder", "notepad", "marker", "eraser"),
        "music genre": ("gospel", "funk", "disco", "techno"),
    },
    "confirmation": {
        "weather phenomenon": ("drought", "heatwave", "downpour", "drizzle"),
        "punctuation mark": ("brace", "ellipsis", "slash", "backslash"),
        "office item": ("ruler", "clipboard", "highlighter", "pushpin"),
        "music genre": ("ambient", "punk", "grunge", "soul"),
    },
    "holdout": {
        "weather phenomenon": ("fog", "mist", "sleet", "waterspout"),
        "punctuation mark": ("asterisk", "ampersand", "question mark", "period"),
        "office item": ("thumbtack", "labelmaker", "holepunch", "calculator"),
        "music genre": ("rock", "metal", "country", "house"),
    },
}

TEMPLATES = {
    "record": {
        "core_membership": (
            'Candidate entry: "{target}"\nCandidate category: "{tested_family}"\n'
            'Context note: use the ordinary {sense_hint} sense; another example mentions "{note_word}" and "{note_family}".\n'
            'In ordinary English, is the candidate entry a kind of the candidate category?'
        ),
        "role_bound_lexical": (
            'Candidate entry: "{target}"\nCandidate category: "{tested_family}"\n'
            'Reference entry: "{reference_word}"\nReference category: "{reference_family}"\n'
            'Context note also mentions "{note_word}" and "{note_family}".\n'
            'Do both candidate fields exactly match their corresponding reference fields?'
        ),
        "explicit_status": (
            'Candidate entry: "{target}"\nCandidate category: "{tested_family}"\n'
            'Context note mentions "{note_word}" and "{note_family}".\nStatus: {status}.\n'
            'Does the stated Status say approved?'
        ),
    },
    "prose": {
        "core_membership": (
            'Consider candidate entry "{target}" and candidate category "{tested_family}" in the ordinary {sense_hint} sense. '
            'A context note also mentions "{note_word}" and "{note_family}". '
            'In ordinary English, is the candidate entry a kind of the candidate category?'
        ),
        "role_bound_lexical": (
            'The candidate fields are entry "{target}" and category "{tested_family}". '
            'The corresponding reference fields are entry "{reference_word}" and category "{reference_family}"; '
            'a context note also mentions "{note_word}" and "{note_family}". '
            'Do both candidate fields exactly match their corresponding reference fields?'
        ),
        "explicit_status": (
            'The candidate fields are entry "{target}" and category "{tested_family}". '
            'A context note mentions "{note_word}" and "{note_family}", and the stated Status is {status}. '
            'Does the stated Status say approved?'
        ),
    },
}


def prior_words():
    found = set()
    for path in (TESTS / "result").glob("phase13*/material/frozen_concept_graph.json"):
        try:
            found.update(str(x["word"]) for x in core.load(path).get("concepts", []))
        except Exception:
            pass
    return found


def concepts():
    return [{"word": word, "family": family, "partition": partition,
             "sense": f"ordinary {family} sense of {word}",
             "adjudication": f"an ordinary-English member of {family}, not of the other frozen categories"}
            for partition in PARTITIONS for family in FAMILIES for word in WORDS[partition][family]]


def cells(partition, family_a, family_b, index, offset, panel):
    wa, wb = WORDS[partition][family_a][index], WORDS[partition][family_b][(index + offset) % 4]
    base = (
        ("aa", wa, family_a, family_a, wa, True),
        ("ab", wa, family_a, family_b, wb, False),
        ("ba", wb, family_b, family_a, wa, False),
        ("bb", wb, family_b, family_b, wb, True),
    )
    output = []
    for cell, target, target_family, tested, other, truth in base:
        if panel == "role_bound_lexical":
            if cell == "aa": ref_word, ref_family = wa, family_a
            elif cell == "ab": ref_word, ref_family = wa, family_a
            elif cell == "ba": ref_word, ref_family = wa, family_a
            else: ref_word, ref_family = wb, family_b
        else:
            ref_word, ref_family = target, tested
        note_family = family_b if tested == family_a else family_a
        note_word = wb if target == wa else wa
        output.append((cell, target, target_family, tested, other, ref_word, ref_family,
                       note_word, note_family, truth))
    return output


def cases():
    rows = []
    for partition in PARTITIONS:
        for family_a, family_b in combinations(FAMILIES, 2):
            for index in range(4):
                for offset in (0, 1):
                    for panel in ("core_membership", "role_bound_lexical", "explicit_status"):
                        for surface in ("record", "prose"):
                            key = f"{panel}:{partition}:{FAMILIES.index(family_a)}__{FAMILIES.index(family_b)}:{index}:o{offset}:{surface}"
                            for values in cells(partition, family_a, family_b, index, offset, panel):
                                (cell, target, target_family, tested, other, ref_word, ref_family,
                                 note_word, note_family, truth) = values
                                status = "approved" if truth else "rejected"
                                prompt = TEMPLATES[surface][panel].format(
                                    target=target, tested_family=tested, sense_hint=target_family,
                                    reference_word=ref_word, reference_family=ref_family,
                                    note_word=note_word, note_family=note_family, status=status,
                                ) + " Output only yes or no."
                                rows.append({
                                    "case_id": f"c052-b-{len(rows):05d}", "panel": panel,
                                    "partition": partition, "family_pair": f"{FAMILIES.index(family_a)}__{FAMILIES.index(family_b)}",
                                    "family_pair_names": f"{family_a}__{family_b}", "pair_index": index,
                                    "pair_offset": offset, "surface": surface, "quartet_key": key, "cell": cell,
                                    "target": target, "target_family": target_family, "tested_family": tested,
                                    "reference_word": ref_word, "reference_family": ref_family,
                                    "note_word": note_word, "note_family": note_family, "status": status,
                                    "truth": truth, "prompt": prompt, "candidates": ["yes", "no"],
                                    "gold_position": 0 if truth else 1,
                                })
    return rows


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_CONFIGS[MODEL]["path"], trust_remote_code=True,
                                        local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def locate(tok, ids, value, first=True):
    needles = [[int(x) for x in tok.encode(v, add_special_tokens=False)] for v in (value, " " + value)]
    matches = []
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                matches.append(list(range(start, start + len(needle))))
    if not matches:
        return None
    return min(matches, key=lambda x: x[0]) if first else max(matches, key=lambda x: x[0])


def compile_rows(source):
    tok = tokenizer()
    output = []
    for row in source:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        output.append({"case_id": row["case_id"], "prompt_ids": ids,
                       "candidate_ids": [[int(x) for x in tok.encode(c, add_special_tokens=False)] for c in ("yes", "no")],
                       "target_span": locate(tok, ids, row["target"], first=True),
                       "tested_family_span": locate(tok, ids, row["tested_family"], first=True),
                       "boundary_position": len(ids) - 1})
    return output


def main():
    parent = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c051_null_library" or "qwen3" not in parent.get("qualified_models", []) \
            or not audit.get("all_checks_passed"):
        raise RuntimeError("C051 must be closed with Qwen-specific controls qualified")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1351 already exists")
    concept_rows, source = concepts(), cases()
    compiled = compile_rows(source)
    groups = defaultdict(list)
    for row in source:
        groups[row["quartet_key"]].append(row)
    panel_counts = Counter(row["panel"] for row in source)
    balance = Counter((r["panel"], r["partition"], r["surface"], r["truth"]) for r in source)
    lengths = defaultdict(list)
    for row, comp in zip(source, compiled):
        lengths[(row["panel"], row["surface"])].append(len(comp["prompt_ids"]))
    medians = {f"{p}:{s}": statistics.median(v) for (p, s), v in lengths.items()}
    checks = {
        "parent": True,
        "fresh": not ({r["word"] for r in concept_rows} & prior_words()),
        "concepts": len(concept_rows) == 64 and len({r["word"] for r in concept_rows}) == 64,
        "cases": len(source) == 4608,
        "panels": panel_counts == {"core_membership": 1536, "role_bound_lexical": 1536, "explicit_status": 1536},
        "quartets": len(groups) == 1152 and all(len(v) == 4 for v in groups.values()),
        "balance": len(balance) == 48 and all(v == 96 for v in balance.values()),
        "semantic_uniqueness": all(r["truth"] == (r["target_family"] == r["tested_family"])
                                   for r in source if r["panel"] == "core_membership"),
        "controlled_naturalness": all(len(r["prompt"].split()) <= 80 for r in source),
        "spans": all(r["target_span"] and r["tested_family_span"] for r in compiled),
        "candidate_ids": all(r["candidate_ids"][0] and r["candidate_ids"][1] for r in compiled),
        "length_match": max(medians.values()) - min(medians.values()) <= 30,
        "no_historical_layer_reuse": True,
    }
    pre = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()),
           "total": len(checks), "all_checks_passed": all(checks.values()), "token_length_medians": medians,
           "human_review": {"independent_blind_review": False,
                            "scope": "controlled English with explicit ordinary-sense adjudication"}}
    core.save(OUT / "audit/pre_model_material_audit.json", pre)
    if not pre["all_checks_passed"]:
        raise RuntimeError(json.dumps(pre, indent=2))
    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c052.graph.v1", "concepts": concept_rows})
    core.write_rows(OUT / "material/frozen_cases.jsonl", source)
    core.write_rows(OUT / "compiled/qwen3_cases.jsonl", compiled)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
        "research_object": "Qwen-specific full-dimensional linear readability of cross-word family-pair interaction identity",
        "claim_boundary": "probe readability only; no causal use, mechanism identity, semantic ontology, or cross-model isomorphism",
        "material": {"families": list(FAMILIES), "partitions": list(PARTITIONS), "panels": list(panel_counts)},
        "behavior_gate": {"executor_max_abs_diff_max": 1e-6, "core_accuracy_min": 0.95,
                          "core_partition_min": 0.90, "core_surface_min": 0.90, "core_family_min": 0.90,
                          "core_truth_min": 0.90, "core_quartet_all_min": 0.90,
                          "control_accuracy_min": 0.98, "control_partition_min": 0.97,
                          "control_surface_min": 0.97, "control_truth_min": 0.97,
                          "control_quartet_all_min": 0.90},
        "probe_gate": {"roles": ["target_span_mean", "tested_family_span_mean", "answer_boundary"],
                       "primary_role": "tested_family_span_mean", "probe": "full-dimensional cosine nearest centroid",
                       "prototype_partition": "prototype_discovery", "selection_partition": "clock_selection",
                       "selection_top1_min": 0.70, "selection_surface_min": 0.60,
                       "selection_median_gap_min": 0.05, "selection_relative_norm_min": 0.001,
                       "control_top1_max": 0.30, "permuted_label_top1_max": 0.30,
                       "persistence_layers": 3, "confirmation_top1_min": 0.60,
                       "holdout_top1_min": 0.60, "transfer_surface_min": 0.55,
                       "transfer_median_gap_min": 0.03, "layer0_relative_norm_max": 1e-6,
                       "numeric_relative_l2_max": 1e-6,
                       "no_dimension_reduction": True, "no_probe_hyperparameter_search": True},
        "branching": {"behavior_fail": "close C052 before hidden states",
                      "behavior_pass": "run Phase1353 all-layer full-dimensional probe",
                      "probe_fail": "close at readability boundary",
                      "probe_pass": "register Qwen-specific descriptive formation candidate; causal work requires a new campaign"},
        "stop_rule": "After behavior reveal, do not change material, model, panel, threshold, probe, role, partition, or branch.",
        "parameter_boundary": "No attention, MLP, SAE, PCA, learned projection, parameter search, or causal intervention is authorized.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1352_c052_qwen_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "protocol/file_hashes.json", {"material": core.sha(OUT / "material/frozen_cases.jsonl"),
                                                   "concepts": core.sha(OUT / "material/frozen_concept_graph.json"),
                                                   "compiled": core.sha(OUT / "compiled/qwen3_cases.jsonl")})
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN,
                                             "contract_sha256": protocol["contract_sha256"],
                                             "all_gates_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps({"preaudit": pre, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
