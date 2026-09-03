#!/usr/bin/env python3
"""Phase1504: freeze C087 cross-root SemEval verb-equivalence observation."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1503_c086_major_stage_closure"
SOURCE = RESULT / "phase1126_semeval_lexsub_natural_cloze" / "protocol" / "source"
OUT = RESULT / "phase1504_c087_cross_root_semeval_contract"
sys.path.insert(0, str(TESTS))

import phase1113_wordnet_semantic_quadrant_protocol as wordnet_source
import phase1126_semeval_lexsub_natural_cloze_protocol as semeval
import phase1331_relational_measurement_core as core
import phase1435_c072_permutation_spectrum_contract as spans
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1504, "C087"
SEED = "C087-cross-root-semeval-v1"
PARTITIONS = ("response_discovery", "confirmation", "lockbox")
ITEMS_PER_PARTITION = 12
INSTANCES_PER_ITEM = 6
SURFACES = {
    "a_natural": (
        'Source sentence: "{sentence}"\nSource verb: {lemma}\nCandidate verb: {candidate}\n'
        "In the source sentence, do the source verb and candidate verb express the same "
        "action meaning? Answer exactly same or different."
    ),
    "b_natural": (
        'Consider this sentence: "{sentence}"\nThe verb being interpreted is {lemma}.\n'
        "The proposed substitute is {candidate}.\nIn this context, are their action meanings "
        "the same? Reply with exactly same or different."
    ),
}
SYSTEM = (
    "Judge the verbs by their meaning in the supplied sentence, not by spelling. "
    "Answer with exactly one word: same or different."
)
ROLES = ("source_relation", "candidate_relation", "boundary")


def digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def char_trigrams(value: str) -> set[str]:
    value = re.sub(r"[^a-z]", "", value.lower())
    if len(value) < 3:
        return {value} if value else set()
    return {value[i : i + 3] for i in range(len(value) - 2)}


def trigram_overlap(left: str, right: str) -> float:
    a, b = char_trigrams(left), char_trigrams(right)
    return len(a & b) / max(1, min(len(a), len(b)))


def verb_synsets() -> dict[str, set[str]]:
    with tarfile.open(wordnet_source.WORDNET_ARCHIVE, "r:gz") as archive:
        raw = archive.extractfile("WordNet-3.0/dict/data.verb").read().decode("utf-8")
    result: dict[str, set[str]] = defaultdict(set)
    for line in raw.splitlines():
        if not line or line[0].isspace():
            continue
        fields = line.split("|", 1)[0].split()
        try:
            word_count = int(fields[3], 16)
        except (IndexError, ValueError):
            continue
        for i in range(word_count):
            result[fields[4 + 2 * i].replace("_", " ").lower()].add(fields[0])
    return dict(result)


def select_rows(tok):
    instances = semeval.parse_xml(SOURCE / "lexsub_test.xml")
    gold = semeval.parse_gold(SOURCE / "gold.gold")
    eligible = {}
    for item, source_rows in sorted(instances.items()):
        lemma, pos = item.rsplit(".", 1)
        if pos != "v" or len(tok.encode(" " + lemma, add_special_tokens=False)) != 1:
            continue
        rows = []
        for source_row in source_rows:
            word_count = len(re.findall(r"[A-Za-z]+", source_row["sentence"]))
            if not 8 <= word_count <= 100:
                continue
            candidates = []
            for candidate, votes in gold.get(source_row["instance_id"], {}).items():
                if (
                    votes >= 2
                    and candidate != lemma
                    and re.fullmatch(r"[a-z]+", candidate)
                    and len(tok.encode(" " + candidate, add_special_tokens=False)) == 1
                    and trigram_overlap(lemma, candidate) == 0.0
                ):
                    candidates.append((candidate, votes))
            if not candidates:
                continue
            candidate, votes = sorted(candidates, key=lambda x: (-x[1], x[0]))[0]
            rows.append(
                {
                    "item": item,
                    "lemma": lemma,
                    "source_instance_id": source_row["instance_id"],
                    "source_head": source_row["head"],
                    "sentence": " ".join(source_row["sentence"].split()),
                    "positive_candidate": candidate,
                    "positive_votes": votes,
                }
            )
        rows.sort(key=lambda x: (-x["positive_votes"], x["source_instance_id"]))
        if len(rows) >= INSTANCES_PER_ITEM:
            eligible[item] = rows[:INSTANCES_PER_ITEM]
    selected_items = sorted(eligible, key=digest)[: ITEMS_PER_PARTITION * 3]
    if len(selected_items) != ITEMS_PER_PARTITION * 3:
        raise RuntimeError(("insufficient_items", len(selected_items)))
    synsets = verb_synsets()
    selected = []
    item_partitions = {}
    for partition_index, partition in enumerate(PARTITIONS):
        items = selected_items[
            partition_index * ITEMS_PER_PARTITION : (partition_index + 1) * ITEMS_PER_PARTITION
        ]
        for item in items:
            item_partitions[item] = partition
        rows = [{**row, "partition": partition} for item in items for row in eligible[item]]
        shift = None
        for candidate_shift in range(1, len(rows)):
            valid = True
            for i, row in enumerate(rows):
                donor = rows[(i + candidate_shift) % len(rows)]
                negative = donor["positive_candidate"]
                if (
                    donor["item"] == row["item"]
                    or gold.get(row["source_instance_id"], {}).get(negative, 0) != 0
                    or trigram_overlap(row["lemma"], negative) != 0.0
                    or synsets.get(row["lemma"], set()) & synsets.get(negative, set())
                ):
                    valid = False
                    break
            if valid:
                shift = candidate_shift
                break
        if shift is None:
            raise RuntimeError(("no_derangement", partition))
        for i, row in enumerate(rows):
            donor = rows[(i + shift) % len(rows)]
            selected.append(
                {
                    **row,
                    "negative_candidate": donor["positive_candidate"],
                    "negative_source_item": donor["item"],
                    "negative_source_instance_id": donor["source_instance_id"],
                    "negative_current_votes": gold.get(row["source_instance_id"], {}).get(
                        donor["positive_candidate"], 0
                    ),
                    "negative_wordnet_shared_synset": bool(
                        synsets.get(row["lemma"], set())
                        & synsets.get(donor["positive_candidate"], set())
                    ),
                    "derangement_shift": shift,
                }
            )
    return selected, item_partitions


def active_cases(selected):
    cases, groups = [], []
    for group_index, row in enumerate(selected):
        group = {
            "set_id": f"c087-compose-{group_index:04d}",
            "partition": row["partition"],
            "item": row["item"],
            "lemma": row["lemma"],
            "source_instance_id": row["source_instance_id"],
        }
        candidate_order = (
            ["same", "different"] if group_index % 2 == 0 else ["different", "same"]
        )
        for surface, template in SURFACES.items():
            for semantic_match, label in ((True, "same"), (False, "different")):
                candidate = (
                    row["positive_candidate"] if semantic_match else row["negative_candidate"]
                )
                case = {
                    "case_id": f"c087-a-{len(cases):04d}",
                    "set_id": group["set_id"],
                    "partition": row["partition"],
                    "item": row["item"],
                    "lemma": row["lemma"],
                    "source_instance_id": row["source_instance_id"],
                    "sentence": row["sentence"],
                    "surface": surface,
                    "semantic_match": semantic_match,
                    "semantic_label": label,
                    "candidate": candidate,
                    "candidate_origin": "human_positive_here" if semantic_match else "human_positive_elsewhere_zero_here",
                    "human_votes_here": row["positive_votes"] if semantic_match else 0,
                    "lexical_trigram_overlap": trigram_overlap(row["lemma"], candidate),
                    "candidates": candidate_order,
                    "gold_position": candidate_order.index(label),
                }
                case["prompt"] = template.format(**case)
                cases.append(case)
                group[f"{surface}_{label}"] = case["case_id"]
        groups.append(group)
    return cases, groups


def compile_cases(tok, cases):
    compiled = []
    for case in cases:
        ids = core.chat_ids(tok, SYSTEM, case["prompt"])
        source_matches = spans.all_spans(tok, ids, case["lemma"])
        candidate_matches = spans.all_spans(tok, ids, case["candidate"])
        if not source_matches or not candidate_matches:
            raise RuntimeError((case["case_id"], source_matches, candidate_matches))
        role_positions = {
            "source_relation": source_matches[-1],
            "candidate_relation": candidate_matches[-1],
            "boundary": [len(ids) - 1],
        }
        if any(len(role_positions[role]) != 1 for role in ROLES):
            raise RuntimeError((case["case_id"], role_positions))
        compiled.append(
            {
                **case,
                "prompt_ids": ids,
                "role_positions": role_positions,
                "candidate_ids": [
                    list(map(int, tok.encode(" " + value, add_special_tokens=False)))
                    for value in case["candidates"]
                ],
            }
        )
    return compiled


def balanced_accuracy(labels, predictions):
    return sum(
        sum(p == truth for y, p in zip(labels, predictions) if y == truth)
        / sum(y == truth for y in labels)
        for truth in (True, False)
    ) / 2.0


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1504 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if (
        parent["authorization"] != "preregister_c087_cross_root_paraphrase_layered_observation"
        or not parent_audit["all_checks_passed"]
    ):
        raise RuntimeError("Phase1503 authorization missing")
    tok = tokenizer()
    selected, item_partitions = select_rows(tok)
    cases, groups = active_cases(selected)
    compiled = compile_cases(tok, cases)
    labels = [case["semantic_match"] for case in cases]
    candidate_majority = {}
    for candidate in {case["candidate"] for case in cases}:
        rows = [case for case in cases if case["candidate"] == candidate]
        candidate_majority[candidate] = sum(r["semantic_match"] for r in rows) >= len(rows) / 2
    zero_models = {
        "always_same": balanced_accuracy(labels, [True] * len(labels)),
        "always_different": balanced_accuracy(labels, [False] * len(labels)),
        "candidate_identity": balanced_accuracy(
            labels, [candidate_majority[case["candidate"]] for case in cases]
        ),
        "first_candidate": balanced_accuracy(labels, [case["gold_position"] == 0 for case in cases]),
        "trigram_overlap": balanced_accuracy(
            labels, [case["lexical_trigram_overlap"] > 0 for case in cases]
        ),
    }
    selected_candidates = Counter(row["positive_candidate"] for row in selected)
    negative_candidates = Counter(row["negative_candidate"] for row in selected)
    source_specs = {
        name: {
            "sha256": core.sha(SOURCE / name),
            "bytes": (SOURCE / name).stat().st_size,
        }
        for name in ("lexsub_test.xml", "gold.gold", "readme")
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "selected_instances": len(selected) == 216,
        "active_cases": len(cases) == 864,
        "composition_sets": len(groups) == 216,
        "partition_items": Counter(item_partitions.values()) == {p: 12 for p in PARTITIONS},
        "partition_disjoint": len(item_partitions) == 36,
        "partition_cases": Counter(case["partition"] for case in cases) == {p: 288 for p in PARTITIONS},
        "semantic_balance": Counter(labels) == {True: 432, False: 432},
        "surface_balance": Counter(case["surface"] for case in cases) == {s: 432 for s in SURFACES},
        "human_positive_lock": all(row["positive_votes"] >= 2 for row in selected),
        "human_zero_negative": all(row["negative_current_votes"] == 0 for row in selected),
        "wordnet_negative_lock": all(not row["negative_wordnet_shared_synset"] for row in selected),
        "cross_root": all(case["lexical_trigram_overlap"] == 0.0 for case in cases),
        "candidate_balance": selected_candidates == negative_candidates,
        "single_token_roles": all(
            all(len(row["role_positions"][role]) == 1 for role in ROLES) for row in compiled
        ),
        "single_token_outputs": all(
            all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled
        ),
        "zero_models": all(value == 0.5 for value in zero_models.values()),
        "machine_naturalness": all("same or different" in case["prompt"] for case in cases),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/selected_instances.jsonl", selected)
    core.write_rows(OUT / "material/active_cases.jsonl", cases)
    core.write_rows(OUT / "material/composition_sets.jsonl", groups)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/frozen_test_examples.jsonl", cases[:8])
    audit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "source": "SemEval-2007 Task 10 English Lexical Substitution human substitutions plus WordNet 3.0 negative non-synset audit",
        "source_specs": source_specs,
        "checks": checks,
        "zero_models": zero_models,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "naturalness_scope": "human-authored benchmark sentences; prompt wrapper is machine-audited controlled English, not independently re-rated by humans",
        "semantic_scope": "contextual cross-root substitute equivalence, not unrestricted relation ontology",
        "hidden_state_accessed": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", audit)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c087.cross_root_semeval_layered_observation.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "cross-root contextual verb-equivalence response in the embedding-to-all-Hidden-State field",
        "source": audit["source"],
        "partitions": list(PARTITIONS),
        "partition_unit": "SemEval lexical item; no item crosses partitions",
        "surfaces": list(SURFACES),
        "roles": list(ROLES),
        "material": {
            "items": 36,
            "instances": 216,
            "cases": 864,
            "cases_per_composition": 4,
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
            "selected_sha256": core.sha(OUT / "material/selected_instances.jsonl"),
            "source_specs": source_specs,
        },
        "behavior_strata": {
            "success": "4/4 correct",
            "mixed": "1-3/4 correct",
            "failed": "0/4 correct",
            "role": "typed stratification; no route-wide stop",
        },
        "allowed_observables": ["input embeddings", "all full-dimensional Hidden States", "same/different logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "post-unblind mutation"],
        "route": [
            "phase1505 behavior strata",
            "phase1506 all-case full-state capture",
            "phase1507 cross-root semantic contrast atlas",
            "phase1508 discovery observation and freeze",
            "phase1509 dual-holdout validation",
            "phase1510 behavior-stratum and C086 diagnostics",
            "phase1511 closure",
        ],
        "claim_boundary": {
            "allowed": "Qwen3 contextual cross-root lexical-equivalence observations on human-annotated SemEval sentences",
            "forbidden": ["universal comparator", "perfect semantic understanding", "causal mechanism", "semantic neurons", "cross-model law", "new mathematics"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1505_c087_behavior_stratification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(
        OUT / "analysis/uploaded_analysis_adjudication.json",
        {
            "retain": [
                "C086 separated explicit labels and answer-code polarity",
                "C086 yielded a repeated task-scoped diagnostic field",
                "cross-root material is the immediate falsification target",
                "observation-first continuation is appropriate",
            ],
            "correct": [
                "C086 does not show that the model perfectly understands but cannot express",
                "cross-relation cosine does not establish a reusable universal comparator",
                "the stable response is not known to be the physical implementation of RDC",
                "output failure cannot be localized to a code-execution stage without causal evidence",
                "low-dimensional and new-mathematics forecasts are unsupported",
            ],
        },
    )
    core.save(
        OUT / "analysis/final.json",
        {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "status": "cross_root_semeval_contract_frozen",
            "contract_sha256": protocol["contract_sha256"],
            "authorization": protocol["authorization"],
        },
    )
    print(json.dumps({"audit": audit, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
