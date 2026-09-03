#!/usr/bin/env python3
"""Phase1512: freeze C088 cross-root semantic-by-answer-code factorial."""
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
PARENT = RESULT / "phase1511_c087_major_stage_closure"
C087 = RESULT / "phase1504_c087_cross_root_semeval_contract"
SOURCE = RESULT / "phase1126_semeval_lexsub_natural_cloze/protocol/source"
OUT = RESULT / "phase1512_c088_cross_root_semantic_code_factorial_contract"
sys.path.insert(0, str(TESTS))

import phase1113_wordnet_semantic_quadrant_protocol as wordnet_source
import phase1126_semeval_lexsub_natural_cloze_protocol as semeval
import phase1331_relational_measurement_core as core
import phase1435_c072_permutation_spectrum_contract as spans
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1512, "C088"
SEED = "C088-cross-root-semantic-code-v1"
PAIRED_PARTITIONS = ("response_discovery", "confirmation", "lockbox")
ALL_PARTITIONS = PAIRED_PARTITIONS + ("fresh_external",)
SURFACES = {
    "a_code": (
        'Source sentence: "{sentence}"\nSource verb: {lemma}\nCandidate verb: {candidate}\n'
        "Response code: {codebook}. Do the two verbs express the same action meaning here? "
        "Answer exactly yes or no using the response code."
    ),
    "b_code": (
        'Consider this sentence: "{sentence}"\nInterpret {lemma} in this context and compare it with {candidate}.\n'
        "Use the {codebook} response code. Are the action meanings the same here? "
        "Reply with exactly yes or no."
    ),
}
SYSTEM = (
    "Judge meaning in the supplied sentence. Standard code means same -> yes and different -> no. "
    "Reversed code means same -> no and different -> yes. Follow the named response code and answer one word."
)
CODEBOOKS = {
    "standard": {"sign": 1, "same": "yes", "different": "no"},
    "reversed": {"sign": -1, "same": "no", "different": "yes"},
}
ROLES = ("source_relation", "candidate_relation", "code_rule", "boundary")


def digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def char_trigrams(value: str) -> set[str]:
    value = re.sub(r"[^a-z]", "", value.lower())
    if len(value) < 3:
        return {value} if value else set()
    return {value[i:i + 3] for i in range(len(value) - 2)}


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
            count = int(fields[3], 16)
        except (IndexError, ValueError):
            continue
        for i in range(count):
            result[fields[4 + 2 * i].replace("_", " ").lower()].add(fields[0])
    return dict(result)


def fresh_rows(tok, used_items):
    instances = semeval.parse_xml(SOURCE / "lexsub_test.xml")
    gold = semeval.parse_gold(SOURCE / "gold.gold")
    eligible = {}
    for item, source_rows in sorted(instances.items()):
        lemma, pos = item.rsplit(".", 1)
        if item in used_items or pos != "v" or len(tok.encode(" " + lemma, add_special_tokens=False)) != 1:
            continue
        rows = []
        for source_row in source_rows:
            candidates = [
                (candidate, votes)
                for candidate, votes in gold.get(source_row["instance_id"], {}).items()
                if votes >= 2
                and candidate != lemma
                and re.fullmatch(r"[a-z]+", candidate)
                and len(tok.encode(" " + candidate, add_special_tokens=False)) == 1
                and trigram_overlap(lemma, candidate) == 0.0
            ]
            if candidates:
                candidate, votes = sorted(candidates, key=lambda x: (-x[1], x[0]))[0]
                rows.append({
                    "item": item,
                    "lemma": lemma,
                    "source_instance_id": source_row["instance_id"],
                    "source_head": source_row["head"],
                    "sentence": " ".join(source_row["sentence"].split()),
                    "positive_candidate": candidate,
                    "positive_votes": votes,
                })
        rows.sort(key=lambda row: (-row["positive_votes"], row["source_instance_id"]))
        if len(rows) >= 4:
            eligible[item] = rows[:4]
    selected_items = sorted(eligible, key=digest)
    rows = [{**row, "partition": "fresh_external", "material_source": "fresh_external"} for item in selected_items for row in eligible[item]]
    synsets = verb_synsets()
    shift = None
    for candidate_shift in range(1, len(rows)):
        if all(
            rows[(i + candidate_shift) % len(rows)]["item"] != row["item"]
            and gold.get(row["source_instance_id"], {}).get(rows[(i + candidate_shift) % len(rows)]["positive_candidate"], 0) == 0
            and trigram_overlap(row["lemma"], rows[(i + candidate_shift) % len(rows)]["positive_candidate"]) == 0.0
            and not (synsets.get(row["lemma"], set()) & synsets.get(rows[(i + candidate_shift) % len(rows)]["positive_candidate"], set()))
            for i, row in enumerate(rows)
        ):
            shift = candidate_shift
            break
    if shift is None:
        raise RuntimeError("no fresh derangement")
    selected = []
    for i, row in enumerate(rows):
        donor = rows[(i + shift) % len(rows)]
        selected.append({
            **row,
            "negative_candidate": donor["positive_candidate"],
            "negative_source_item": donor["item"],
            "negative_source_instance_id": donor["source_instance_id"],
            "negative_current_votes": gold.get(row["source_instance_id"], {}).get(donor["positive_candidate"], 0),
            "negative_wordnet_shared_synset": bool(synsets.get(row["lemma"], set()) & synsets.get(donor["positive_candidate"], set())),
            "derangement_shift": shift,
        })
    return selected, {item: len(eligible[item]) for item in selected_items}


def build_cases(selected):
    cases, groups = [], []
    for group_index, row in enumerate(selected):
        group = {
            "set_id": f"c088-compose-{group_index:04d}",
            "partition": row["partition"],
            "material_source": row["material_source"],
            "item": row["item"],
            "lemma": row["lemma"],
            "source_instance_id": row["source_instance_id"],
        }
        output_order = ["yes", "no"] if group_index % 2 == 0 else ["no", "yes"]
        for surface, template in SURFACES.items():
            for codebook, code in CODEBOOKS.items():
                for semantic_match, semantic_label in ((True, "same"), (False, "different")):
                    candidate = row["positive_candidate"] if semantic_match else row["negative_candidate"]
                    gold_label = code[semantic_label]
                    case = {
                        "case_id": f"c088-a-{len(cases):05d}",
                        "set_id": group["set_id"],
                        "partition": row["partition"],
                        "material_source": row["material_source"],
                        "item": row["item"],
                        "lemma": row["lemma"],
                        "source_instance_id": row["source_instance_id"],
                        "sentence": row["sentence"],
                        "surface": surface,
                        "codebook": codebook,
                        "code_sign": code["sign"],
                        "semantic_match": semantic_match,
                        "semantic_label": semantic_label,
                        "semantic_sign": 1 if semantic_match else -1,
                        "candidate": candidate,
                        "candidate_origin": "human_positive_here" if semantic_match else "human_positive_elsewhere_zero_here",
                        "human_votes_here": row["positive_votes"] if semantic_match else 0,
                        "lexical_trigram_overlap": trigram_overlap(row["lemma"], candidate),
                        "gold_label": gold_label,
                        "candidates": output_order,
                        "gold_position": output_order.index(gold_label),
                    }
                    case["prompt"] = template.format(**case)
                    cases.append(case)
                    group[f"{surface}_{codebook}_{semantic_label}"] = case["case_id"]
        groups.append(group)
    return cases, groups


def compile_cases(tok, cases):
    compiled = []
    for case in cases:
        ids = core.chat_ids(tok, SYSTEM, case["prompt"])
        source = spans.all_spans(tok, ids, case["lemma"])
        candidate = spans.all_spans(tok, ids, case["candidate"])
        code = spans.all_spans(tok, ids, case["codebook"])
        if not source or not candidate or not code:
            raise RuntimeError((case["case_id"], source, candidate, code))
        role_positions = {
            "source_relation": source[-1],
            "candidate_relation": candidate[-1],
            "code_rule": code[-1],
            "boundary": [len(ids) - 1],
        }
        if any(len(role_positions[role]) != 1 for role in ROLES):
            raise RuntimeError((case["case_id"], role_positions))
        compiled.append({
            **case,
            "prompt_ids": ids,
            "role_positions": role_positions,
            "candidate_ids": [list(map(int, tok.encode(" " + value, add_special_tokens=False))) for value in case["candidates"]],
        })
    return compiled


def balanced_accuracy(labels, predictions):
    return sum(
        sum(prediction == truth for label, prediction in zip(labels, predictions) if label == truth) / sum(label == truth for label in labels)
        for truth in ("yes", "no")
    ) / 2.0


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1512 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c088_cross_root_semantic_by_answer_code_factorial" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1511 authorization missing")
    tok = tokenizer()
    paired = [{**row, "material_source": "c087_paired"} for row in core.rows(C087 / "material/selected_instances.jsonl")]
    used_items = {row["item"] for row in paired}
    fresh, fresh_inventory = fresh_rows(tok, used_items)
    selected = paired + fresh
    cases, groups = build_cases(selected)
    compiled = compile_cases(tok, cases)
    labels = [case["gold_label"] for case in cases]
    candidate_majority = {}
    for candidate in {case["candidate"] for case in cases}:
        rows = [case for case in cases if case["candidate"] == candidate]
        candidate_majority[candidate] = Counter(row["gold_label"] for row in rows).most_common(1)[0][0]
    zero_models = {
        "always_yes": balanced_accuracy(labels, ["yes"] * len(cases)),
        "always_no": balanced_accuracy(labels, ["no"] * len(cases)),
        "assume_standard": balanced_accuracy(labels, ["yes" if row["semantic_match"] else "no" for row in cases]),
        "assume_reversed": balanced_accuracy(labels, ["no" if row["semantic_match"] else "yes" for row in cases]),
        "code_only": balanced_accuracy(labels, ["yes" if row["codebook"] == "standard" else "no" for row in cases]),
        "candidate_identity": balanced_accuracy(labels, [candidate_majority[row["candidate"]] for row in cases]),
        "first_candidate": balanced_accuracy(labels, [row["candidates"][0] for row in cases]),
        "surface_only": balanced_accuracy(labels, ["yes" if row["surface"] == "a_code" else "no" for row in cases]),
        "trigram_overlap": balanced_accuracy(labels, ["yes" if row["lexical_trigram_overlap"] > 0 else "no" for row in cases]),
    }
    partition_candidate_balance = {
        partition: Counter(row["positive_candidate"] for row in selected if row["partition"] == partition) == Counter(row["negative_candidate"] for row in selected if row["partition"] == partition)
        for partition in ALL_PARTITIONS
    }
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "paired_groups": len(paired) == 216,
        "fresh_inventory": len(fresh_inventory) == 8 and set(fresh_inventory.values()) == {4},
        "fresh_groups": len(fresh) == 32,
        "groups": len(groups) == 248,
        "cases": len(cases) == 1984,
        "partition_counts": Counter(case["partition"] for case in cases) == {"response_discovery": 576, "confirmation": 576, "lockbox": 576, "fresh_external": 256},
        "factor_balance": Counter((case["semantic_sign"], case["code_sign"]) for case in cases) == {(1, 1): 496, (-1, 1): 496, (1, -1): 496, (-1, -1): 496},
        "output_balance": Counter(labels) == {"yes": 992, "no": 992},
        "surface_balance": Counter(case["surface"] for case in cases) == {"a_code": 992, "b_code": 992},
        "candidate_balance": all(partition_candidate_balance.values()),
        "fresh_disjoint": not (used_items & set(fresh_inventory)),
        "human_positive": all(row["positive_votes"] >= 2 for row in selected),
        "negative_zero": all(row["negative_current_votes"] == 0 for row in selected),
        "negative_wordnet": all(not row["negative_wordnet_shared_synset"] for row in selected),
        "cross_root": all(case["lexical_trigram_overlap"] == 0.0 for case in cases),
        "single_roles": all(all(len(row["role_positions"][role]) == 1 for role in ROLES) for row in compiled),
        "single_outputs": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "zero_models": all(value == 0.5 for value in zero_models.values()),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/selected_instances.jsonl", selected)
    core.write_rows(OUT / "material/active_cases.jsonl", cases)
    core.write_rows(OUT / "material/composition_sets.jsonl", groups)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/frozen_test_examples.jsonl", cases[:12])
    resource = {
        "strict_fresh_target_items": 36,
        "strict_fresh_available_items": 8,
        "strict_fresh_target_met": False,
        "fresh_item_instance_counts": fresh_inventory,
        "paired_identification_role": "reuse C087 semantic panels only to identify the newly introduced answer-code factor",
        "fresh_external_role": "directional external check, not a 36-item independent replication",
    }
    core.save(OUT / "audit/fresh_resource_audit.json", resource)
    audit = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "checks": checks,
        "zero_models": zero_models,
        "partition_candidate_balance": partition_candidate_balance,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "naturalness_scope": "human-authored source sentences; symbolic code wrapper is controlled English without new independent human rating",
        "hidden_state_accessed": False,
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", audit)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c088.cross_root_semantic_by_answer_code_factorial.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization",
        "research_object": "identify code-invariant semantic-match and semantic-by-answer-code response terms in one cross-root natural-context field",
        "partitions": list(ALL_PARTITIONS),
        "paired_partitions": list(PAIRED_PARTITIONS),
        "surfaces": list(SURFACES),
        "codebooks": CODEBOOKS,
        "roles": list(ROLES),
        "factors": ["semantic", "code"],
        "material": {
            "paired_groups": 216,
            "fresh_external_groups": 32,
            "cases": 1984,
            "cases_per_group": 8,
            "selected_sha256": core.sha(OUT / "material/selected_instances.jsonl"),
            "active_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
            "fresh_resource": resource,
        },
        "authoritative_forward": {
            "single_pass_behavior_and_hidden_capture": True,
            "batch_size": 12,
            "output_hidden_states": True,
            "repeat_first_batch": True,
        },
        "behavior_strata": {"success": "8/8", "mixed": "1-7/8", "failed": "0/8", "role": "typed observation; no campaign stop"},
        "observation": {
            "effects": ["semantic", "code", "semantic_code"],
            "discovery_partition": "response_discovery",
            "validation_partitions": ["confirmation", "lockbox"],
            "external_partition": "fresh_external",
            "structure_presence_and_effect_equivalence_are_separate": True,
        },
        "allowed_observables": ["input embeddings", "all full-dimensional Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "post-unblind mutation"],
        "route": ["phase1513 unified forward", "phase1514 factorial atlas", "phase1515 discovery freeze", "phase1516 paired and fresh reveal", "phase1517 diagnostics", "phase1518 closure"],
        "claim_boundary": {
            "allowed": "Qwen3 task-scoped cross-root semantic-by-answer-code observations",
            "forbidden": ["pure semantic vector before factorial separation", "universal comparator", "causal mechanism", "semantic neurons", "cross-model law", "new mathematics"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1513_c088_unified_forward_capture"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/uploaded_analysis_adjudication.json", {
        "retain": [
            "C087 removed the same-root shortcut and found a repeated late boundary response",
            "structure presence and effect-size equality must be adjudicated separately",
            "C088 must orthogonalize semantic truth and output-code polarity",
            "a single authoritative forward should replace behavior/capture execution mismatch",
        ],
        "correct": [
            "Phase1506 execution identity failure is a hard evidence limitation, not a minor blemish",
            "C087 did not establish a cross-root semantic mechanism or universal relation field",
            "state35 partition cosine near one may reflect a shared answer-decision variable",
            "shared-plus-specific and new-mathematics proposals remain hypotheses, not results",
            "future C089/C090 stages are plans and must not be narrated as completed evidence",
        ],
    })
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "status": "semantic_code_factorial_contract_frozen", "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"audit": audit, "resource": resource, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
