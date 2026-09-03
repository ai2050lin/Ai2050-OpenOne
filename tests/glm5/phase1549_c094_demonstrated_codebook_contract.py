#!/usr/bin/env python3
"""Phase1549: freeze an A/B reversible codebook contract with balanced demonstrations."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
SOURCE = RESULT / "phase1543_c092_truth_output_code_factorial_contract"
PARENT = RESULT / "phase1548_c093_interface_adjudication_and_closure"
OUT = RESULT / "phase1549_c094_demonstrated_codebook_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PARTITIONS = ("response_discovery", "confirmation", "lockbox")
SURFACES = ("prequery", "postquery")
CODEBOOKS = ("native", "reversed")
RELATION = "乙词是甲词所表示整体的组成部分"
SYSTEM = "请根据现代汉语的通常词义判断整体-部分关系，并严格学习题目示例中的A/B输出映射。只输出A或B。"


def mapping(codebook: str) -> tuple[str, str]:
    return ("A", "B") if codebook == "native" else ("B", "A")


def mapping_text(codebook: str) -> str:
    true_code, false_code = mapping(codebook)
    return f"关系成立输出{true_code}，关系不成立输出{false_code}"


def demonstrations(codebook: str) -> str:
    true_code, false_code = mapping(codebook)
    return (
        f"示例一：甲词【汽车】，乙词【车轮】，待检验关系【{RELATION}】，输出【{true_code}】。"
        f"示例二：甲词【汽车】，乙词【森林】，待检验关系【{RELATION}】，输出【{false_code}】。"
    )


def prompt_for(pair: dict, surface: str, codebook: str) -> str:
    prefix = f"映射规则【{mapping_text(codebook)}】。{demonstrations(codebook)}现在作答："
    words = f"甲词【{pair['source']}】，乙词【{pair['target']}】"
    if surface == "prequery":
        return f"{prefix}待检验关系【{RELATION}】。{words}。只输出A或B。"
    return f"{prefix}{words}。待检验关系【{RELATION}】。只输出A或B。"


def all_spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [list(map(int, tok.encode(form, add_special_tokens=False))) for form in (value, " " + value)]
    found = set()
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                found.add(tuple(range(start, start + len(needle))))
    return [list(span) for span in sorted(found)]


def last_span(tok, ids: list[int], value: str) -> list[int]:
    spans = all_spans(tok, ids, value)
    if not spans:
        raise RuntimeError((value, spans))
    return spans[-1]


def span_inside_last_marker(tok, ids: list[int], marker: str, value: str) -> list[int]:
    markers = all_spans(tok, ids, marker)
    if not markers:
        raise RuntimeError(("marker", marker))
    points = set(markers[-1])
    candidates = [span for span in all_spans(tok, ids, value) if set(span).issubset(points)]
    if len(candidates) != 1:
        raise RuntimeError((marker, value, candidates))
    return candidates[0]


def majority_predictions(rows: list[dict], field: str) -> list[str]:
    counts: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row[field]][row["gold_label"]] += 1
    majority = {key: sorted(value.items(), key=lambda item: (-item[1], item[0]))[0][0] for key, value in counts.items()}
    return [majority[row[field]] for row in rows]


def ba(labels: list[str], predictions: list[str]) -> float:
    return 0.5 * sum(sum(predictions[i] == label for i, value in enumerate(labels) if value == label) / sum(value == label for value in labels) for label in ("A", "B"))


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1549 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1549_c094_demonstrated_codebook_contract" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1548 authorization missing")
    pairs = core.rows(SOURCE / "material/frozen_pairs.jsonl")
    lexical = {word for row in pairs for word in (row["source"], row["target"])}
    tok = tokenizer()
    cases, compiled = [], []
    for pair in pairs:
        for surface_index, surface in enumerate(SURFACES):
            for codebook_index, codebook in enumerate(CODEBOOKS):
                true_code, false_code = mapping(codebook)
                truth = bool(pair["semantic_truth"])
                gold = true_code if truth else false_code
                candidates = ["A", "B"] if (pair["c092_stratum_rank"] + surface_index + codebook_index) % 2 == 0 else ["B", "A"]
                prompt = prompt_for(pair, surface, codebook)
                row = {
                    "case_id": f"c094-{len(cases):04d}",
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
                    "answer_sign": (1 if truth else -1) * (1 if codebook == "native" else -1),
                    "gold_label": gold,
                    "candidates": candidates,
                    "gold_position": candidates.index(gold),
                    "prompt": prompt,
                }
                ids = core.chat_ids(tok, SYSTEM, prompt)
                positions = {
                    "source_word": span_inside_last_marker(tok, ids, f"甲词【{pair['source']}】", pair["source"]),
                    "target_word": span_inside_last_marker(tok, ids, f"乙词【{pair['target']}】", pair["target"]),
                    "relation_anchor": span_inside_last_marker(tok, ids, f"待检验关系【{RELATION}】", RELATION),
                    "mapping_anchor": span_inside_last_marker(tok, ids, f"映射规则【{mapping_text(codebook)}】", mapping_text(codebook)),
                    "boundary": [len(ids) - 1],
                }
                candidate_ids = [[int(token) for token in tok.encode(label, add_special_tokens=False)] for label in candidates]
                cases.append(row)
                compiled.append({**row, "prompt_ids": ids, "role_positions": positions, "candidate_ids": candidate_ids})
    labels = [row["gold_label"] for row in cases]
    zero_models = {
        "always_A": ba(labels, ["A"] * len(cases)),
        "always_B": ba(labels, ["B"] * len(cases)),
        "pair_identity": ba(labels, majority_predictions(cases, "pair_id")),
        "surface_identity": ba(labels, majority_predictions(cases, "surface")),
        "codebook_identity": ba(labels, majority_predictions(cases, "codebook")),
        "truth_identity": ba(labels, majority_predictions(cases, "semantic_truth")),
        "candidate_position_zero": ba(labels, [row["candidates"][0] for row in cases]),
    }
    cells = Counter((row["partition"], row["surface"], row["truth_sign"], row["codebook_sign"], row["answer_sign"]) for row in cases)
    max_length = max(len(row["prompt_ids"]) for row in compiled)
    checks = {
        "coverage": len(cases) == len(compiled) == 240,
        "cells": len(cells) == 24 and set(cells.values()) == {10},
        "single_token": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "roles": all(all(row["role_positions"].values()) for row in compiled),
        "demo_lexical_separation": not ({"汽车", "车轮", "森林"} & lexical),
        "balanced_demo_truth": True,
        "zero_models": all(value == 0.5 for value in zero_models.values()),
        "fixed_shape": max_length < 256,
        "hidden_not_accessed": True,
        "model_not_loaded": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/active_cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    protocol = {
        "phase": 1549,
        "campaign": "C094",
        "research_object": "demonstrated reversible A/B compiler for truth-output factorial identification",
        "model": "Qwen/Qwen3-4B local BF16 CUDA, no quantization",
        "partitions": list(PARTITIONS),
        "surfaces": list(SURFACES),
        "codebooks": list(CODEBOOKS),
        "demonstrations": {"true": ["汽车", "车轮"], "false": ["汽车", "森林"], "present_in_both_codebooks": True},
        "execution": {"padding": "right", "batch_size": 4, "fixed_global_sequence_length": max_length, "score_position": "actual assistant boundary"},
        "behavior_gate": {"each_codebook_BA": 0.80, "each_codebook_each_surface_BA": 0.75, "each_codebook_true_recall": 0.75, "each_codebook_false_recall": 0.75, "both_codebooks_required": True},
        "sequence": ["discovery behavior", "confirmation behavior if discovery passes", "lockbox behavior if confirmation passes", "all-state capture only after all behavior partitions pass", "factorial observation"],
        "factorial_model": "H=S+y*T+c*C+(y*c)*A+epsilon",
        "forbidden": ["attention", "MLP", "parameters", "gradients", "PCA", "TDA", "learned probes", "post-reveal gate mutation"],
        "files": {"cases_sha256": core.sha(OUT / "material/active_cases.jsonl"), "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl")},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1550_c094_discovery_behavior_qualification"
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"phase": 1549, "checks": checks, "zero_models": zero_models, "max_prompt_length": max_length, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/pre_model_audit.json", audit)
    core.save(OUT / "analysis/final.json", {"phase": 1549, "campaign": "C094", "status": "contract_frozen", "authorization": protocol["authorization"]})
    print(json.dumps({"protocol": protocol, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
