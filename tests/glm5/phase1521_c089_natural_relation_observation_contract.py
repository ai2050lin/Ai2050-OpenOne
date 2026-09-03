#!/usr/bin/env python3
"""Phase1521: freeze C089 natural noun-relation full-state observation contract."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1520_c088_full_input_code_semantics_correction"
OUT = RESULT / "phase1521_c089_natural_relation_observation_contract"
DATA = ROOT / "tests/gpt5/result/phase602_three_track_semantics/source/WordNet-3.0/dict/data.noun"
INDEX = ROOT / "tests/gpt5/result/phase602_three_track_semantics/source/WordNet-3.0/dict/index.noun"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1521, "C089"
SEED = "C089-natural-noun-relations-v1"
FAMILIES = ("synonym", "kind_of", "part_of")
PARTITIONS = ("response_discovery", "confirmation", "lockbox")
ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
SYSTEM = (
    'Judge the natural English noun relation stated in the user question. "First" and "second" '
    "refer to the nouns in their written order. Answer exactly yes or no. Do not infer a relation merely from spelling."
)
SURFACES = {
    "synonym": {
        "a_question": ('Consider the nouns "{source}" and "{target}". Is the first noun a synonym of the second?', "synonym"),
        "b_question": ('For "{source}" versus "{target}", does the first have the same noun meaning as the second?', "same noun meaning"),
    },
    "kind_of": {
        "a_question": ('Consider the nouns "{source}" and "{target}". Is the first noun a kind of the second?', "kind of"),
        "b_question": ('For "{source}" versus "{target}", does the first name a type of the second?', "type of"),
    },
    "part_of": {
        "a_question": ('Consider the nouns "{source}" and "{target}". Is the first noun a part of the second?', "part of"),
        "b_question": ('For "{source}" versus "{target}", does the first name one of the parts of the second?', "parts of"),
    },
}


def digest(value: object) -> str:
    return hashlib.sha256(f"{SEED}:{core.canonical(value)}".encode()).hexdigest()


def trigrams(value: str) -> set[str]:
    value = re.sub(r"[^a-z]", "", value.lower())
    return {value[i:i + 3] for i in range(max(0, len(value) - 2))}


def trigram_overlap(left: str, right: str) -> float:
    a, b = trigrams(left), trigrams(right)
    return len(a & b) / max(1, min(len(a), len(b)))


def parse_wordnet() -> tuple[dict, dict[str, set[str]], dict[str, int], dict[str, int]]:
    synsets = {}
    word_offsets: dict[str, set[str]] = defaultdict(set)
    with DATA.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(line) < 10 or not line[:8].isdigit():
                continue
            body, _, gloss = line.partition("|")
            fields = body.split()
            offset = fields[0]
            count = int(fields[3], 16)
            cursor = 4
            words = []
            for _ in range(count):
                word = fields[cursor]
                words.append(word)
                word_offsets[word].add(offset)
                cursor += 2
            pointer_count = int(fields[cursor])
            cursor += 1
            pointers = []
            for _ in range(pointer_count):
                symbol, target, target_pos, source_target = fields[cursor:cursor + 4]
                pointers.append((symbol, target, target_pos, source_target))
                cursor += 4
            synsets[offset] = {"offset": offset, "words": words, "pointers": pointers, "gloss": gloss.strip()}
    tag_counts = {}
    with INDEX.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0].isspace():
                continue
            fields = line.split()
            pointer_count = int(fields[3])
            count = int(fields[5 + pointer_count])
            if count:
                tag_counts[fields[0]] = count
    senses = {word: len(offsets) for word, offsets in word_offsets.items()}
    return synsets, dict(word_offsets), senses, tag_counts


def valid_word(word: str, senses: dict[str, int], tags: dict[str, int], tok) -> bool:
    return (
        word.isalpha() and word.islower() and 3 <= len(word) <= 18
        and senses.get(word) == 1
        and len(tok.encode(" " + word, add_special_tokens=False)) <= 3
    )


def closure(synsets: dict, start: str, symbols: set[str]) -> set[str]:
    seen, queue = set(), deque([start])
    while queue:
        current = queue.popleft()
        for symbol, target, pos, _ in synsets[current]["pointers"]:
            if symbol in symbols and pos == "n" and target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def related(family: str, source_offset: str, target_offset: str, synsets: dict) -> bool:
    if family == "synonym":
        return source_offset == target_offset
    symbols = {"@", "@i"} if family == "kind_of" else {"#p"}
    return target_offset in closure(synsets, source_offset, symbols)


def candidate_rows(tok, synsets, word_offsets, senses, tags) -> dict[str, list[dict]]:
    eligible = {word for word in word_offsets if valid_word(word, senses, tags, tok)}
    rows: dict[str, list[dict]] = {family: [] for family in FAMILIES}
    for offset, synset in synsets.items():
        words = sorted(word for word in synset["words"] if word in eligible)
        for left, right in combinations(words, 2):
            if max(tags.get(left, 0), tags.get(right, 0)) > 0 and trigram_overlap(left, right) == 0:
                rows["synonym"].append({"source": left, "target": right, "source_offset": offset, "target_offset": offset})
        for symbol, target_offset, pos, _ in synset["pointers"]:
            family = "kind_of" if symbol in {"@", "@i"} else "part_of" if symbol == "#p" else None
            if family is None or pos != "n" or target_offset not in synsets:
                continue
            targets = sorted(word for word in synsets[target_offset]["words"] if word in eligible)
            for source in words:
                for target in targets:
                    if source != target and max(tags.get(source, 0), tags.get(target, 0)) > 0 and trigram_overlap(source, target) == 0:
                        rows[family].append({"source": source, "target": target, "source_offset": offset, "target_offset": target_offset})
    for family in FAMILIES:
        unique = {core.canonical(row): row for row in rows[family]}
        rows[family] = sorted(
            unique.values(),
            key=lambda row: (-min(tags.get(row["source"], 0), tags.get(row["target"], 0)), -max(tags.get(row["source"], 0), tags.get(row["target"], 0)), digest(row)),
        )
    return rows


def pair_compatible(family: str, left: dict, right: dict, synsets: dict) -> bool:
    words = (left["source"], left["target"], right["source"], right["target"])
    if len(set(words)) != 4:
        return False
    if any(trigram_overlap(source, target) != 0 for source in (left["source"], right["source"]) for target in (left["target"], right["target"])):
        return False
    cross = ((left["source_offset"], right["target_offset"]), (right["source_offset"], left["target_offset"]))
    return all(
        not related(family, source, target, synsets) and not related(family, target, source, synsets)
        for source, target in cross
    )


def select_groups(candidates: dict[str, list[dict]], synsets: dict) -> list[dict]:
    used_global: set[str] = set()
    selected = []
    for family in ("part_of", "synonym", "kind_of"):
        available = [row for row in candidates[family] if row["source"] not in used_global and row["target"] not in used_global]
        groups = []
        while available and len(groups) < 15:
            left = available.pop(0)
            partner_index = next((i for i, right in enumerate(available) if pair_compatible(family, left, right, synsets)), None)
            if partner_index is None:
                continue
            right = available.pop(partner_index)
            groups.append((left, right))
            used_global.update((left["source"], left["target"], right["source"], right["target"]))
            available = [row for row in available if row["source"] not in used_global and row["target"] not in used_global]
        if len(groups) != 15:
            raise RuntimeError((family, "insufficient compatible groups", len(groups)))
        groups.sort(key=digest)
        for index, (left, right) in enumerate(groups):
            partition = PARTITIONS[index // 5]
            selected.append({
                "set_id": f"c089-{family}-{index:02d}", "family": family, "partition": partition,
                "pair_a": left, "pair_b": right,
            })
    return sorted(selected, key=lambda row: (PARTITIONS.index(row["partition"]), FAMILIES.index(row["family"]), row["set_id"]))


def build_cases(groups: list[dict]) -> list[dict]:
    cases = []
    for group in groups:
        a, b = group["pair_a"], group["pair_b"]
        cells = {
            "aa": (a["source"], a["target"], True),
            "ab": (a["source"], b["target"], False),
            "ba": (b["source"], a["target"], False),
            "bb": (b["source"], b["target"], True),
        }
        for surface, (template, anchor) in SURFACES[group["family"]].items():
            for cell, (source, target, truth) in cells.items():
                case = {
                    "case_id": f"c089-a-{len(cases):04d}", "set_id": group["set_id"],
                    "family": group["family"], "partition": group["partition"], "surface": surface,
                    "cell": cell, "source": source, "target": target, "truth": truth,
                    "truth_sign": 1 if truth else -1, "gold_label": "yes" if truth else "no",
                    "relation_anchor": anchor,
                }
                case["prompt"] = template.format(**case) + " Reply with exactly yes or no."
                case["candidates"] = ["yes", "no"] if len(cases) % 2 == 0 else ["no", "yes"]
                case["gold_position"] = case["candidates"].index(case["gold_label"])
                cases.append(case)
    return cases


def all_spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [list(map(int, tok.encode(form, add_special_tokens=False))) for form in (value, " " + value)]
    found = set()
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                found.add(tuple(range(start, start + len(needle))))
    return [list(span) for span in sorted(found)]


def compile_cases(tok, cases: list[dict]) -> list[dict]:
    compiled = []
    for case in cases:
        ids = core.chat_ids(tok, SYSTEM, case["prompt"])
        positions = {}
        for role, value in (("source_word", case["source"]), ("target_word", case["target"]), ("relation_anchor", case["relation_anchor"])):
            spans = all_spans(tok, ids, value)
            if len(spans) != 1:
                raise RuntimeError((case["case_id"], role, value, spans))
            positions[role] = spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({
            **case, "prompt_ids": ids, "role_positions": positions,
            "candidate_ids": [[int(token) for token in tok.encode(value, add_special_tokens=False)] for value in case["candidates"]],
        })
    return compiled


def balanced_accuracy(labels: list[str], predictions: list[str]) -> float:
    return sum(sum(p == y for p, y in zip(predictions, labels) if y == label) / labels.count(label) for label in ("yes", "no")) / 2


def majority_predictions(cases: list[dict], key: str) -> list[str]:
    lookup = {}
    for value in {case[key] for case in cases}:
        cell = [case["gold_label"] for case in cases if case[key] == value]
        lookup[value] = Counter(cell).most_common(1)[0][0]
    return [lookup[case[key]] for case in cases]


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1521 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "preregister_c089_natural_relation_full_state_observation_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1520 authorization missing")
    tok = tokenizer()
    synsets, word_offsets, senses, tags = parse_wordnet()
    candidates = candidate_rows(tok, synsets, word_offsets, senses, tags)
    groups = select_groups(candidates, synsets)
    cases = build_cases(groups)
    compiled = compile_cases(tok, cases)
    labels = [case["gold_label"] for case in cases]
    zero_models = {
        "always_yes": balanced_accuracy(labels, ["yes"] * len(cases)),
        "always_no": balanced_accuracy(labels, ["no"] * len(cases)),
        "source_identity": balanced_accuracy(labels, majority_predictions(cases, "source")),
        "target_identity": balanced_accuracy(labels, majority_predictions(cases, "target")),
        "family_only": balanced_accuracy(labels, majority_predictions(cases, "family")),
        "surface_only": balanced_accuracy(labels, majority_predictions(cases, "surface")),
        "first_candidate": balanced_accuracy(labels, [case["candidates"][0] for case in cases]),
        "trigram_overlap": balanced_accuracy(labels, ["yes" if trigram_overlap(case["source"], case["target"]) > 0 else "no" for case in cases]),
    }
    selected_words = [word for group in groups for pair in (group["pair_a"], group["pair_b"]) for word in (pair["source"], pair["target"])]
    checks = {
        "parent": parent_audit["all_checks_passed"],
        "wordnet_source": DATA.exists() and INDEX.exists(),
        "groups": len(groups) == 45,
        "cases": len(cases) == 360,
        "family_balance": Counter(case["family"] for case in cases) == {family: 120 for family in FAMILIES},
        "partition_balance": Counter(case["partition"] for case in cases) == {partition: 120 for partition in PARTITIONS},
        "surface_balance": all(Counter(case["surface"] for case in cases if case["family"] == family) == {"a_question": 60, "b_question": 60} for family in FAMILIES),
        "truth_balance": Counter(labels) == {"yes": 180, "no": 180},
        "lexical_disjoint": len(selected_words) == len(set(selected_words)) == 180,
        "monosemous": all(senses[word] == 1 for word in selected_words),
        "wordnet_registered": all(word in word_offsets for word in selected_words),
        "pair_attested_anchor": all(max(tags.get(pair["source"], 0), tags.get(pair["target"], 0)) > 0 for group in groups for pair in (group["pair_a"], group["pair_b"])),
        "cross_root": all(trigram_overlap(case["source"], case["target"]) == 0 for case in cases),
        "positive_truth": all(related(group["family"], pair["source_offset"], pair["target_offset"], synsets) for group in groups for pair in (group["pair_a"], group["pair_b"])),
        "negative_truth": all(not case["truth"] for case in cases if case["cell"] in {"ab", "ba"}),
        "role_spans": all(all(1 <= len(row["role_positions"][role]) <= 4 for role in ROLES) for row in compiled),
        "single_token_outputs": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "zero_models": all(value == 0.5 for value in zero_models.values()),
        "hidden_not_accessed": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/relation_composition_sets.jsonl", groups)
    core.write_rows(OUT / "material/active_cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    core.write_rows(OUT / "material/frozen_test_examples.jsonl", [cases[i] for i in (0, 1, 4, 5, 120, 121, 240, 241)])
    audit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "zero_models": zero_models,
        "candidate_inventory": {family: len(candidates[family]) for family in FAMILIES},
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "semantic_uniqueness": "both nouns are lowercase monosemous WordNet entries and each positive pair has at least one tagged-sense anchor; positives are direct registered relations; cross-swapped negatives are graph-audited absent, including transitive kind/part closure",
        "naturalness_scope": "ordinary English noun-relation questions over WordNet corpus-attested lemmas; machine-audited templates only, without independent human naturalness ratings",
        "complete_input_scope": "system plus user prompt plus generated assistant boundary compiled through the Qwen chat template",
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", audit)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c089.natural_noun_relation_full_state_observation.v1",
        "model": "qwen3-bfloat16-cuda-no-quantization", "system": SYSTEM,
        "research_object": "observe full-dimensional Hidden-State truth-response fields for three explicitly natural noun relations without artificial relation labels or answer codebooks",
        "families": list(FAMILIES), "partitions": list(PARTITIONS), "surfaces": {family: list(SURFACES[family]) for family in FAMILIES},
        "roles": list(ROLES), "role_pooling": "mean over every token in the registered role span",
        "material": {
            "source": str(DATA.relative_to(ROOT)), "groups": 45, "positive_pairs": 90, "cases": 360,
            "groups_per_family_partition": 5, "lexical_items": 180,
            "groups_sha256": core.sha(OUT / "material/relation_composition_sets.jsonl"),
            "cases_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
        },
        "authoritative_forward": {"single_pass_behavior_and_hidden_capture": True, "batch_size": 12, "output_hidden_states": True, "repeat_first_batch": True},
        "behavior_qualification": {
            "discovery_family_balanced_accuracy": 0.70,
            "discovery_each_surface_accuracy": 0.65,
            "scope": "only behavior-qualified families receive semantic Hidden-State interpretation; all cases remain in the descriptive observation archive",
            "route_policy": "failure retires a family interpretation, not the C089 observation campaign",
        },
        "observation_order": ["all-state atlas", "discovery observation", "prediction freeze", "confirmation reveal", "lockbox reveal", "diagnostics", "closure"],
        "validation_components": {
            "discovery_centroid_cosine": 0.40, "confirmation_lockbox_cosine": 0.50,
            "within_partition_surface_cosine": 0.40, "top26_jaccard": 0.15,
            "components_are_reported_separately": True,
        },
        "allowed_observables": ["input embeddings", "all 37 full-dimensional Hidden States", "yes/no logits"],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "post-unblind mutation"],
        "claim_boundary": {
            "allowed": "Qwen3 task-scoped descriptive formation, role-transport, coordinate, shared and family-residual observations",
            "forbidden": ["universal relation vector", "semantic neurons", "necessary or sufficient circuit", "cross-model law", "new mathematics"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1522_c089_unified_forward_capture"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/uploaded_analysis_adjudication.json", {
        "retain": [
            "C088 fixed execution identity and obtained reproducible full-dimensional factorial observations",
            "late semantic-associated effects require separation from labels, instructions and interaction terms",
            "the next route should use natural relation questions and scan all states before freezing a hypothesis",
        ],
        "correct": [
            "Phase1519 omitted the system message and is superseded by Phase1520",
            "C088 code mappings were defined, although the code main effect is not a pure rule-execution variable",
            "late state21-24 or state35 patterns are task-scoped observations, not a universal formation clock",
            "factor components are additive contrasts, not proven orthogonal direct-sum mechanisms",
            "current mathematics is expressive enough; claims that new mathematics is already required are speculative",
        ],
    })
    core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "status": "natural_relation_observation_contract_frozen", "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]})
    print(json.dumps({"audit": audit, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
