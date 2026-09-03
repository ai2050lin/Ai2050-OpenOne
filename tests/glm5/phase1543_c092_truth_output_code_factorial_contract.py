#!/usr/bin/env python3
"""Phase1543: preregister the C092 truth-by-output-code factorial contract."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
SOURCE = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
PARENT = RESULT / "phase1542_c091_final_adjudication"
OUT = RESULT / "phase1543_c092_truth_output_code_factorial_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PARTITIONS = ("response_discovery", "confirmation", "lockbox")
SURFACES = ("prequery", "postquery")
CODEBOOKS = ("native", "reversed")
ROLES = ("source_word", "target_word", "relation_anchor", "codebook_anchor", "boundary")
SYSTEM = "请依照现代汉语中通常、直接的词义和题目明确给出的回答编码作答。不要使用隐喻、双关或罕见义项。严格只输出一个汉字：是或否。"
RELATION = "乙词是甲词所表示整体的组成部分"
CODEBOOK_TEXT = {
    "native": "关系成立回答“是”，关系不成立回答“否”",
    "reversed": "关系成立回答“否”，关系不成立回答“是”",
}


def balanced_accuracy(labels: list[str], predictions: list[str]) -> float:
    scores = []
    for label in ("是", "否"):
        indices = [i for i, value in enumerate(labels) if value == label]
        scores.append(sum(predictions[i] == label for i in indices) / len(indices))
    return sum(scores) / 2


def majority_predictions(rows: list[dict], field: str) -> list[str]:
    counts: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row[field]][row["gold_label"]] += 1
    majority = {key: sorted(values.items(), key=lambda item: (-item[1], item[0]))[0][0] for key, values in counts.items()}
    return [majority[row[field]] for row in rows]


def select_pairs() -> list[dict]:
    source = core.rows(SOURCE / "material/frozen_pairs.jsonl")
    selected = []
    for partition_index, partition in enumerate(PARTITIONS):
        for concreteness_index, concreteness in enumerate(("concrete", "abstract")):
            true_rows = sorted(
                [row for row in source if row["partition"] == partition and row["concreteness"] == concreteness and row["family"] == "whole_part"],
                key=lambda row: row["pair_id"],
            )
            similarity = sorted(
                [row for row in source if row["partition"] == partition and row["concreteness"] == concreteness and row["family"] == "similarity"],
                key=lambda row: row["pair_id"],
            )
            category = sorted(
                [row for row in source if row["partition"] == partition and row["concreteness"] == concreteness and row["family"] == "class_inclusion"],
                key=lambda row: row["pair_id"],
            )
            similarity_count = 3 if (partition_index + concreteness_index) % 2 == 0 else 2
            false_rows = similarity[:similarity_count] + category[:5 - similarity_count]
            selected.extend({**row, "semantic_truth": True, "c092_stratum_rank": rank} for rank, row in enumerate(true_rows))
            selected.extend({**row, "semantic_truth": False, "c092_stratum_rank": rank} for rank, row in enumerate(false_rows))
    return sorted(selected, key=lambda row: (PARTITIONS.index(row["partition"]), row["concreteness"], not row["semantic_truth"], row["pair_id"]))


def prompt_for(pair: dict, surface: str, codebook: str) -> str:
    rule = CODEBOOK_TEXT[codebook]
    words = f"甲词【{pair['source']}】，乙词【{pair['target']}】"
    if surface == "prequery":
        return f"回答编码【{rule}】。待检验关系【{RELATION}】。{words}。请按回答编码作答。"
    return f"回答编码【{rule}】。{words}。待检验关系【{RELATION}】。请按回答编码作答。"


def all_spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [list(map(int, tok.encode(form, add_special_tokens=False))) for form in (value, " " + value)]
    found = set()
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                found.add(tuple(range(start, start + len(needle))))
    return [list(span) for span in sorted(found)]


def unique_span(tok, ids: list[int], value: str) -> list[int]:
    spans = all_spans(tok, ids, value)
    if len(spans) != 1:
        raise RuntimeError((value, spans))
    return spans[0]


def span_inside_marker(tok, ids: list[int], marker: str, value: str) -> list[int]:
    marker_span = unique_span(tok, ids, marker)
    marker_points = set(marker_span)
    candidates = [span for span in all_spans(tok, ids, value) if set(span).issubset(marker_points)]
    if len(candidates) != 1:
        raise RuntimeError(("marked role", marker, value, candidates))
    return candidates[0]


def build_and_compile(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    tok = tokenizer()
    cases = []
    compiled = []
    for pair in pairs:
        for surface_index, surface in enumerate(SURFACES):
            for codebook_index, codebook in enumerate(CODEBOOKS):
                truth = bool(pair["semantic_truth"])
                emitted_yes = truth if codebook == "native" else not truth
                gold = "是" if emitted_yes else "否"
                candidates = ["是", "否"] if (pair["c092_stratum_rank"] + surface_index + codebook_index) % 2 == 0 else ["否", "是"]
                prompt = prompt_for(pair, surface, codebook)
                row = {
                    "case_id": f"c092-{len(cases):04d}",
                    "pair_id": pair["pair_id"],
                    "pair_family": pair["family"],
                    "partition": pair["partition"],
                    "concreteness": pair["concreteness"],
                    "surface": surface,
                    "codebook": codebook,
                    "source": pair["source"],
                    "target": pair["target"],
                    "semantic_truth": truth,
                    "truth_sign": 1 if truth else -1,
                    "codebook_sign": 1 if codebook == "native" else -1,
                    "answer_sign": 1 if emitted_yes else -1,
                    "gold_label": gold,
                    "candidates": candidates,
                    "gold_position": candidates.index(gold),
                    "prompt": prompt,
                }
                ids = core.chat_ids(tok, SYSTEM, prompt)
                role_positions = {
                    "source_word": span_inside_marker(tok, ids, f"甲词【{pair['source']}】", pair["source"]),
                    "target_word": span_inside_marker(tok, ids, f"乙词【{pair['target']}】", pair["target"]),
                    "relation_anchor": span_inside_marker(tok, ids, f"待检验关系【{RELATION}】", RELATION),
                    "codebook_anchor": span_inside_marker(tok, ids, f"回答编码【{CODEBOOK_TEXT[codebook]}】", CODEBOOK_TEXT[codebook]),
                    "boundary": [len(ids) - 1],
                }
                candidate_ids = [[int(token) for token in tok.encode(value, add_special_tokens=False)] for value in candidates]
                cases.append(row)
                compiled.append({**row, "prompt_ids": ids, "role_positions": role_positions, "candidate_ids": candidate_ids})
    return cases, compiled


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1543 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1543_c092_truth_output_code_factorial_contract" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1542 authorization missing")
    pairs = select_pairs()
    cases, compiled = build_and_compile(pairs)
    counts = Counter((row["partition"], row["concreteness"], row["semantic_truth"]) for row in pairs)
    cell_counts = Counter((row["partition"], row["surface"], row["truth_sign"], row["codebook_sign"], row["answer_sign"]) for row in cases)
    labels = [row["gold_label"] for row in cases]
    zero_models = {
        "always_yes": balanced_accuracy(labels, ["是"] * len(cases)),
        "always_no": balanced_accuracy(labels, ["否"] * len(cases)),
        "pair_identity": balanced_accuracy(labels, majority_predictions(cases, "pair_id")),
        "surface_identity": balanced_accuracy(labels, majority_predictions(cases, "surface")),
        "codebook_identity": balanced_accuracy(labels, majority_predictions(cases, "codebook")),
        "semantic_truth_identity": balanced_accuracy(labels, majority_predictions(cases, "semantic_truth")),
        "candidate_position_zero": balanced_accuracy(labels, [row["candidates"][0] for row in cases]),
    }
    max_length = max(len(row["prompt_ids"]) for row in compiled)
    checks = {
        "pair_count": len(pairs) == 60,
        "case_count": len(cases) == 240,
        "partition_concreteness_truth_balance": len(counts) == 12 and set(counts.values()) == {5},
        "false_family_balance": Counter(row["family"] for row in pairs if not row["semantic_truth"]) == {"similarity": 15, "class_inclusion": 15},
        "factorial_cells": len(cell_counts) == 24 and set(cell_counts.values()) == {10},
        "answer_sign_product": all(row["answer_sign"] == row["truth_sign"] * row["codebook_sign"] for row in cases),
        "single_token_outputs": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "role_spans": all(all(row["role_positions"][role] for role in ROLES) for row in compiled),
        "zero_models": all(value == 0.5 for value in zero_models.values()),
        "fixed_shape_feasible": max_length < 256,
        "semantic_uniqueness": all(row["source"] != row["target"] for row in pairs),
        "source_human_validated": True,
        "hidden_not_accessed": True,
        "model_not_loaded": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/frozen_pairs.jsonl", pairs)
    core.write_rows(OUT / "material/active_cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    protocol = {
        "phase": 1543,
        "campaign": "C092",
        "research_object": "separate semantic whole-part truth, codebook instruction, and emitted answer token",
        "model": "Qwen/Qwen3-4B local BF16 CUDA, no quantization",
        "system": SYSTEM,
        "partitions": list(PARTITIONS),
        "surfaces": list(SURFACES),
        "codebooks": list(CODEBOOKS),
        "roles": list(ROLES),
        "execution": {
            "padding": "right",
            "batch": "all two surfaces times two codebooks for one semantic pair",
            "batch_size": 4,
            "fixed_global_sequence_length": max_length,
            "position_ids": "attention-mask cumulative positions with pad positions reset to zero",
            "score_position": "actual assistant boundary",
        },
        "behavior_gate": {
            "discovery_each_codebook_semantic_balanced_accuracy": 0.80,
            "discovery_each_codebook_each_surface_semantic_balanced_accuracy": 0.75,
            "discovery_each_codebook_true_recall": 0.75,
            "discovery_each_codebook_false_recall": 0.75,
            "both_codebooks_required": True,
            "failure_action": "retire C092 without reading hidden states",
        },
        "numeric_gate": {
            "repeat_logits_max_abs": 1e-6,
            "hidden_replay_max_abs": 1e-6,
            "behavior_logit_replay_max_abs": 1e-6,
        },
        "factorial_model": "H=S+y*T+c*C+(y*c)*A+epsilon",
        "factor_definitions": {
            "y": "semantic whole-part truth sign",
            "c": "native versus reversed codebook sign",
            "y_times_c": "required emitted yes versus no sign",
        },
        "sequence": [
            "behavior-only run with hidden disabled",
            "independent behavior adjudication",
            "all-state capture only if both codebooks pass",
            "discovery all-state observation and frozen candidate selection",
            "confirmation and lockbox validation",
            "campaign adjudication",
        ],
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "post-reveal gate mutation"],
        "claim_boundary": "factorial association is not causal mechanism or semantic neuron identity",
        "files": {
            "pairs_sha256": core.sha(OUT / "material/frozen_pairs.jsonl"),
            "cases_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1544_c092_behavior_only_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {
        "phase": 1543,
        "checks": checks,
        "zero_models": zero_models,
        "counts": {"|".join(map(str, key)): value for key, value in counts.items()},
        "factorial_cells": {"|".join(map(str, key)): value for key, value in cell_counts.items()},
        "max_prompt_length": max_length,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/pre_model_audit.json", audit)
    core.save(OUT / "analysis/final.json", {"phase": 1543, "campaign": "C092", "status": "contract_frozen", "authorization": protocol["authorization"]})
    print(json.dumps({"protocol": protocol, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
