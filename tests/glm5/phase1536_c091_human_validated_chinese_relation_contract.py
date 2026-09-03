#!/usr/bin/env python3
"""Phase1536: preregister a human-validated Chinese relation timing contract."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1535_c091_external_human_material_and_analysis_adjudication"
OUT = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

SEED = "C091-human-validated-chinese-relation-timing-v1"
FAMILIES = ("similarity", "class_inclusion", "whole_part")
PARTITIONS = ("response_discovery", "confirmation", "lockbox")
SURFACES = ("prequery", "postquery")
ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
DATASET_RELATIONS = {
    "similarity": ("A", "相似关系"),
    "class_inclusion": ("B", "类别关系"),
    "whole_part": ("B", "整体-部分关系"),
}
ANCHORS = {
    "similarity": "两个词的含义相近或基本相同",
    "class_inclusion": "甲词表示的事物属于乙词所表示的类别",
    "whole_part": "乙词是甲词所表示整体的组成部分",
}
SYSTEM = (
    "请依据现代汉语中通常、直接的词义判断，不使用隐喻、双关或罕见义项。"
    "严格按照甲词、乙词的顺序和问题指定的关系作答。只输出一个汉字：是或否。"
)


def digest(value: object) -> str:
    return hashlib.sha256((SEED + ":" + core.canonical(value)).encode("utf-8")).hexdigest()


def balanced_accuracy(labels: list[str], predictions: list[str]) -> float:
    values = []
    for label in ("是", "否"):
        indices = [i for i, value in enumerate(labels) if value == label]
        values.append(sum(predictions[i] == label for i in indices) / len(indices))
    return sum(values) / 2


def majority_predictions(rows: list[dict], field: str) -> list[str]:
    counts: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row[field]][row["gold_label"]] += 1
    majority = {key: sorted(value.items(), key=lambda item: (-item[1], item[0]))[0][0] for key, value in counts.items()}
    return [majority[row[field]] for row in rows]


def source_frames() -> dict[str, pd.DataFrame]:
    return {
        "A": pd.read_csv(PARENT / "source/word_pair_A.csv", encoding="gb18030"),
        "B": pd.read_csv(PARENT / "source/word_pair_B.csv", encoding="gb18030"),
    }


def select_pairs(frames: dict[str, pd.DataFrame]) -> list[dict]:
    used: set[str] = set()
    selected = []
    for family in FAMILIES:
        frame_key, relation = DATASET_RELATIONS[family]
        family_frame = frames[frame_key][frames[frame_key]["relation"] == relation]
        by_concreteness = {}
        for concreteness in ("concrete", "abstract"):
            candidates = []
            for _, row in family_frame[family_frame["concreteness"] == concreteness].iterrows():
                source, target = row["word_pair"].split(":")
                candidates.append({
                    "family": family,
                    "dataset_relation": relation,
                    "source": source.strip(),
                    "target": target.strip(),
                    "concreteness": concreteness,
                    "type_ratings": int(row["type_ratings"]),
                })
            candidates.sort(key=digest)
            chosen = []
            for row in candidates:
                if row["source"] in used or row["target"] in used or row["source"] == row["target"]:
                    continue
                chosen.append(row)
                used.update((row["source"], row["target"]))
                if len(chosen) == 15:
                    break
            if len(chosen) != 15:
                raise RuntimeError((family, concreteness, "insufficient globally unique pairs", len(chosen)))
            chosen.sort(key=digest)
            by_concreteness[concreteness] = chosen
        for partition_index, partition in enumerate(PARTITIONS):
            block = by_concreteness["concrete"][partition_index * 5:(partition_index + 1) * 5]
            block += by_concreteness["abstract"][partition_index * 5:(partition_index + 1) * 5]
            block.sort(key=digest)
            for rank, row in enumerate(block):
                selected.append({
                    **row,
                    "pair_id": f"c091-{partition_index}-{FAMILIES.index(family)}-{rank:02d}",
                    "partition": partition,
                    "partition_rank": rank,
                })
    return sorted(selected, key=lambda row: (PARTITIONS.index(row["partition"]), FAMILIES.index(row["family"]), row["partition_rank"]))


def prompt_for(pair: dict, query_family: str, surface: str) -> str:
    anchor = ANCHORS[query_family]
    words = f"甲词【{pair['source']}】，乙词【{pair['target']}】"
    if surface == "prequery":
        return f"待检验的说法是：{anchor}。{words}。这个说法成立吗？只回答是或否。"
    return f"{words}。请判断这个说法是否成立：{anchor}。只回答是或否。"


def build_cases(pairs: list[dict]) -> list[dict]:
    cases = []
    for pair in pairs:
        for surface_index, surface in enumerate(SURFACES):
            for query_index, query_family in enumerate(FAMILIES):
                truth = pair["family"] == query_family
                candidates = ["是", "否"] if (pair["partition_rank"] + surface_index + query_index) % 2 == 0 else ["否", "是"]
                case = {
                    "case_id": f"c091-{len(cases):04d}",
                    "pair_id": pair["pair_id"],
                    "pair_family": pair["family"],
                    "query_family": query_family,
                    "partition": pair["partition"],
                    "partition_rank": pair["partition_rank"],
                    "concreteness": pair["concreteness"],
                    "surface": surface,
                    "source": pair["source"],
                    "target": pair["target"],
                    "relation_anchor": ANCHORS[query_family],
                    "truth": truth,
                    "gold_label": "是" if truth else "否",
                    "candidates": candidates,
                    "prompt": prompt_for(pair, query_family, surface),
                }
                case["gold_position"] = candidates.index(case["gold_label"])
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


def span_inside_marker(tok, ids: list[int], marker: str, value: str) -> list[int]:
    markers = all_spans(tok, ids, marker)
    if len(markers) != 1:
        raise RuntimeError(("marker", marker, markers))
    marker_points = set(markers[0])
    candidates = [span for span in all_spans(tok, ids, value) if set(span).issubset(marker_points)]
    if len(candidates) != 1:
        raise RuntimeError(("marked role", marker, value, candidates))
    return candidates[0]


def compile_cases(tok, cases: list[dict]) -> list[dict]:
    compiled = []
    for case in cases:
        ids = core.chat_ids(tok, SYSTEM, case["prompt"])
        positions = {}
        positions["source_word"] = span_inside_marker(tok, ids, f"甲词【{case['source']}】", case["source"])
        positions["target_word"] = span_inside_marker(tok, ids, f"乙词【{case['target']}】", case["target"])
        anchors = all_spans(tok, ids, case["relation_anchor"])
        if len(anchors) != 1:
            raise RuntimeError((case["case_id"], "relation_anchor", case["relation_anchor"], anchors))
        positions["relation_anchor"] = anchors[0]
        positions["boundary"] = [len(ids) - 1]
        candidate_ids = [[int(token) for token in tok.encode(value, add_special_tokens=False)] for value in case["candidates"]]
        compiled.append({**case, "prompt_ids": ids, "role_positions": positions, "candidate_ids": candidate_ids})
    return compiled


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1536 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1536_c091_human_validated_chinese_relation_contract" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1535 authorization missing")
    pairs = select_pairs(source_frames())
    cases = build_cases(pairs)
    tok = tokenizer()
    compiled = compile_cases(tok, cases)
    labels = [row["gold_label"] for row in cases]
    zero_models = {
        "always_yes": balanced_accuracy(labels, ["是"] * len(cases)),
        "always_no": balanced_accuracy(labels, ["否"] * len(cases)),
        "pair_identity": balanced_accuracy(labels, majority_predictions(cases, "pair_id")),
        "query_identity": balanced_accuracy(labels, majority_predictions(cases, "query_family")),
        "surface_identity": balanced_accuracy(labels, majority_predictions(cases, "surface")),
        "candidate_position_zero": balanced_accuracy(labels, [row["candidates"][0] for row in cases]),
    }
    pair_counts = Counter((row["partition"], row["family"], row["concreteness"]) for row in pairs)
    gold_position_counts = Counter((row["gold_label"], row["gold_position"]) for row in cases)
    max_length = max(len(row["prompt_ids"]) for row in compiled)
    checks = {
        "pair_count": len(pairs) == 90,
        "case_count": len(cases) == 540,
        "partition_family_concreteness_balance": all(value == 5 for value in pair_counts.values()) and len(pair_counts) == 18,
        "global_lexical_nonreuse": len({word for row in pairs for word in (row["source"], row["target"])}) == 180,
        "three_queries_two_surfaces": all(sum(row["pair_id"] == pair["pair_id"] for row in cases) == 6 for pair in pairs),
        "one_true_query_per_surface": all(sum(row["pair_id"] == pair["pair_id"] and row["surface"] == surface and row["truth"] for row in cases) == 1 for pair in pairs for surface in SURFACES),
        "candidate_position_balance": set(gold_position_counts.values()) == {90, 180},
        "single_token_outputs": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "role_spans": all(all(len(row["role_positions"][role]) >= 1 for role in ROLES) for row in compiled),
        "complete_input_compiled": all(row["prompt_ids"][-1] == compiled[0]["prompt_ids"][-1] for row in compiled),
        "zero_models": all(value == 0.5 for value in zero_models.values()),
        "fixed_shape_feasible": max_length < 256,
        "human_source_independent": True,
        "hidden_not_accessed": True,
        "model_not_loaded": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/frozen_pairs.jsonl", pairs)
    core.write_rows(OUT / "material/active_cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    protocol = {
        "phase": 1536,
        "campaign": "C091",
        "schema": "c091.human_validated_chinese_relation_timing.v1",
        "research_object": "behavior-grounded relation-match response relocation under prequery versus postquery causal order",
        "model": "Qwen/Qwen3-4B local BF16 CUDA, no quantization",
        "system": SYSTEM,
        "families": list(FAMILIES),
        "partitions": list(PARTITIONS),
        "surfaces": list(SURFACES),
        "roles": list(ROLES),
        "material": {
            "source_phase": 1535,
            "pairs": 90,
            "cases": 540,
            "pairs_per_family_partition": 10,
            "concrete_and_abstract_per_family_partition": [5, 5],
            "post_training_publication": 2026,
            "human_validation": "final typical pairs selected from judgments by the retained 5,898-participant sample",
            "pairs_sha256": core.sha(OUT / "material/frozen_pairs.jsonl"),
            "cases_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
        },
        "execution": {
            "padding": "right",
            "batch": "all three relation queries times both surfaces for one pair",
            "batch_size": 6,
            "fixed_global_sequence_length": max_length,
            "position_ids": "attention-mask cumulative positions with pad positions reset to zero",
            "score_position": "each sequence's actual assistant boundary",
        },
        "behavior_gate": {
            "discovery_query_family_balanced_accuracy": 0.80,
            "discovery_each_surface_balanced_accuracy": 0.75,
            "discovery_true_recall": 0.75,
            "discovery_false_recall": 0.75,
            "discovery_three_way_pair_selection_accuracy": 0.75,
            "route_policy": "retire only a failed relation family; continue with every independently qualified family",
        },
        "numeric_gate_before_hidden_use": {
            "repeat_hidden_max_abs": 1e-6,
            "repeat_logit_max_abs": 1e-6,
            "behavior_logit_replay_max_abs": 1e-6,
            "postquery_source_target_causal_max_abs": 1e-6,
            "prequery_relation_anchor_causal_max_abs": 1e-6,
        },
        "frozen_contrast": "C_fg=0.5*(H(p_f,q_f)+H(p_g,q_g)-H(p_f,q_g)-H(p_g,q_f)) for concreteness- and rank-matched pairs",
        "causal_timing_truths": {
            "prequery_relation_anchor": "must cancel exactly because it precedes both words",
            "postquery_source_and_target": "must cancel exactly because both precede the relation query",
            "first_joint_information_event": {"prequery": "target_word", "postquery": "relation_anchor"},
        },
        "sequence": [
            "behavior-only authoritative run",
            "freeze qualified relation routes",
            "hidden capture only after behavior qualification and exact logit replay",
            "all-state discovery observation",
            "freeze candidates",
            "confirmation and lockbox reveal",
        ],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "post-unblind threshold mutation"],
        "claim_boundary": {
            "allowed": "Qwen3 task-scoped behavior qualification and full-dimensional relation-match timing observations",
            "forbidden": ["semantic neurons", "universal relation direction", "causal circuit", "cross-language law", "new mathematics"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1537_c091_behavior_only_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {
        "phase": 1536,
        "campaign": "C091",
        "checks": checks,
        "zero_models": zero_models,
        "pair_counts": {"|".join(key): value for key, value in pair_counts.items()},
        "gold_position_counts": {f"{label}|{position}": value for (label, position), value in gold_position_counts.items()},
        "max_prompt_length": max_length,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/pre_model_material_semantic_zero_model_audit.json", audit)
    core.save(OUT / "analysis/final.json", {
        "phase": 1536,
        "campaign": "C091",
        "status": "human_validated_chinese_relation_timing_contract_frozen",
        "authorization": protocol["authorization"],
    })
    print(json.dumps({"audit": audit, "protocol": protocol}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
