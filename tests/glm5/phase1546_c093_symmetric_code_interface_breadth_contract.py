#!/usr/bin/env python3
"""Phase1546: freeze a breadth screen of four low-semantic-load output alphabets."""
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
PARENT = RESULT / "phase1545_c092_behavior_gate_adjudication_and_closure"
OUT = RESULT / "phase1546_c093_symmetric_code_interface_breadth_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PARTITIONS = ("response_discovery", "confirmation", "lockbox")
SURFACES = ("prequery", "postquery")
CODEBOOKS = ("native", "reversed")
INTERFACES = {
    "han": ("甲", "乙"),
    "latin": ("A", "B"),
    "digits": ("1", "2"),
    "xy": ("X", "Y"),
}
RELATION = "乙词是甲词所表示整体的组成部分"
SYSTEM = "请依照现代汉语中通常、直接的词义和题目给出的代码映射作答。不要使用隐喻、双关或罕见义项。严格只输出指定的一个代码字符。"


def majority_predictions(rows: list[dict], field: str) -> list[str]:
    counts: dict[object, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row[field]][row["gold_label"]] += 1
    majority = {key: sorted(value.items(), key=lambda item: (-item[1], item[0]))[0][0] for key, value in counts.items()}
    return [majority[row[field]] for row in rows]


def balanced_accuracy(labels: list[str], predictions: list[str], alphabet: tuple[str, str]) -> float:
    recalls = []
    for label in alphabet:
        indices = [i for i, value in enumerate(labels) if value == label]
        recalls.append(sum(predictions[i] == label for i in indices) / len(indices))
    return sum(recalls) / 2


def mapping_text(codebook: str, first: str, second: str) -> str:
    if codebook == "native":
        return f"关系成立输出{first}，关系不成立输出{second}"
    return f"关系成立输出{second}，关系不成立输出{first}"


def prompt_for(pair: dict, surface: str, interface: str, codebook: str) -> str:
    first, second = INTERFACES[interface]
    alphabet = f"可用代码为{first}和{second}"
    mapping = mapping_text(codebook, first, second)
    words = f"甲词【{pair['source']}】，乙词【{pair['target']}】"
    if surface == "prequery":
        return f"输出字母【{alphabet}】。映射规则【{mapping}】。待检验关系【{RELATION}】。{words}。请只输出代码。"
    return f"输出字母【{alphabet}】。映射规则【{mapping}】。{words}。待检验关系【{RELATION}】。请只输出代码。"


def all_spans(tok, ids: list[int], value: str) -> list[list[int]]:
    needles = [list(map(int, tok.encode(form, add_special_tokens=False))) for form in (value, " " + value)]
    found = set()
    for needle in needles:
        for start in range(len(ids) - len(needle) + 1):
            if ids[start:start + len(needle)] == needle:
                found.add(tuple(range(start, start + len(needle))))
    return [list(span) for span in sorted(found)]


def span_inside(tok, ids: list[int], marker: str, value: str) -> list[int]:
    markers = all_spans(tok, ids, marker)
    if len(markers) != 1:
        raise RuntimeError(("marker", marker, markers))
    points = set(markers[0])
    candidates = [span for span in all_spans(tok, ids, value) if set(span).issubset(points)]
    if len(candidates) != 1:
        raise RuntimeError(("role", marker, value, candidates))
    return candidates[0]


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1546 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1546_c093_symmetric_code_interface_breadth_contract" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1545 authorization missing")
    pairs = core.rows(SOURCE / "material/frozen_pairs.jsonl")
    tok = tokenizer()
    token_ids = {name: {value: list(map(int, tok.encode(value, add_special_tokens=False))) for value in values} for name, values in INTERFACES.items()}
    cases = []
    compiled = []
    for pair in pairs:
        for surface_index, surface in enumerate(SURFACES):
            for interface_index, (interface, alphabet) in enumerate(INTERFACES.items()):
                first, second = alphabet
                for codebook_index, codebook in enumerate(CODEBOOKS):
                    truth = bool(pair["semantic_truth"])
                    first_is_gold = truth if codebook == "native" else not truth
                    gold = first if first_is_gold else second
                    candidates = [first, second] if (pair["c092_stratum_rank"] + surface_index + interface_index + codebook_index) % 2 == 0 else [second, first]
                    prompt = prompt_for(pair, surface, interface, codebook)
                    alphabet_text = f"可用代码为{first}和{second}"
                    map_text = mapping_text(codebook, first, second)
                    row = {
                        "case_id": f"c093-{len(cases):04d}",
                        "pair_id": pair["pair_id"],
                        "pair_family": pair["family"],
                        "partition": pair["partition"],
                        "concreteness": pair["concreteness"],
                        "surface": surface,
                        "interface": interface,
                        "codebook": codebook,
                        "source": pair["source"],
                        "target": pair["target"],
                        "semantic_truth": truth,
                        "truth_sign": 1 if truth else -1,
                        "codebook_sign": 1 if codebook == "native" else -1,
                        "answer_sign": (1 if truth else -1) * (1 if codebook == "native" else -1),
                        "first_code": first,
                        "second_code": second,
                        "gold_label": gold,
                        "candidates": candidates,
                        "gold_position": candidates.index(gold),
                        "prompt": prompt,
                    }
                    ids = core.chat_ids(tok, SYSTEM, prompt)
                    positions = {
                        "source_word": span_inside(tok, ids, f"甲词【{pair['source']}】", pair["source"]),
                        "target_word": span_inside(tok, ids, f"乙词【{pair['target']}】", pair["target"]),
                        "relation_anchor": span_inside(tok, ids, f"待检验关系【{RELATION}】", RELATION),
                        "mapping_anchor": span_inside(tok, ids, f"映射规则【{map_text}】", map_text),
                        "boundary": [len(ids) - 1],
                    }
                    candidate_ids = [[int(value) for value in tok.encode(label, add_special_tokens=False)] for label in candidates]
                    cases.append(row)
                    compiled.append({**row, "prompt_ids": ids, "role_positions": positions, "candidate_ids": candidate_ids})
    max_length = max(len(row["prompt_ids"]) for row in compiled)
    cell_counts = Counter((row["partition"], row["surface"], row["interface"], row["truth_sign"], row["codebook_sign"]) for row in cases)
    zero_models = {}
    for interface, alphabet in INTERFACES.items():
        rows = [row for row in cases if row["interface"] == interface]
        labels = [row["gold_label"] for row in rows]
        zero_models[interface] = {
            "always_first_code": balanced_accuracy(labels, [alphabet[0]] * len(rows), alphabet),
            "always_second_code": balanced_accuracy(labels, [alphabet[1]] * len(rows), alphabet),
            "pair_identity": balanced_accuracy(labels, majority_predictions(rows, "pair_id"), alphabet),
            "surface_identity": balanced_accuracy(labels, majority_predictions(rows, "surface"), alphabet),
            "codebook_identity": balanced_accuracy(labels, majority_predictions(rows, "codebook"), alphabet),
            "semantic_truth_identity": balanced_accuracy(labels, majority_predictions(rows, "semantic_truth"), alphabet),
            "candidate_position_zero": balanced_accuracy(labels, [row["candidates"][0] for row in rows], alphabet),
        }
    checks = {
        "coverage": len(cases) == len(compiled) == 960,
        "single_token_alphabets": all(len(ids) == 1 for mapping in token_ids.values() for ids in mapping.values()),
        "unique_token_ids": len({ids[0] for mapping in token_ids.values() for ids in mapping.values()}) == 8,
        "factorial_cells": len(cell_counts) == 96 and set(cell_counts.values()) == {10},
        "answer_sign": all(row["answer_sign"] == row["truth_sign"] * row["codebook_sign"] for row in cases),
        "role_spans": all(all(row["role_positions"].values()) for row in compiled),
        "candidate_tokens": all(all(len(ids) == 1 for ids in row["candidate_ids"]) for row in compiled),
        "zero_models": all(value == 0.5 for mapping in zero_models.values() for value in mapping.values()),
        "fixed_shape_feasible": max_length < 256,
        "hidden_not_accessed": True,
        "model_not_loaded": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.write_rows(OUT / "material/active_cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3_active.jsonl", compiled)
    protocol = {
        "phase": 1546,
        "campaign": "C093",
        "research_object": "breadth-screen symmetric low-semantic-load output alphabets for truth-code orthogonalization",
        "model": "Qwen/Qwen3-4B local BF16 CUDA, no quantization",
        "partitions": list(PARTITIONS),
        "surfaces": list(SURFACES),
        "interfaces": {key: list(value) for key, value in INTERFACES.items()},
        "codebooks": list(CODEBOOKS),
        "execution": {
            "padding": "right",
            "batch": "16 prompts per semantic pair: 2 surfaces x 4 interfaces x 2 codebooks",
            "fixed_global_sequence_length": max_length,
            "score_position": "actual assistant boundary",
        },
        "discovery_gate_per_interface": {
            "each_codebook_semantic_balanced_accuracy": 0.80,
            "each_codebook_each_surface_semantic_balanced_accuracy": 0.75,
            "each_codebook_true_recall": 0.75,
            "each_codebook_false_recall": 0.75,
            "both_codebooks_required": True,
        },
        "confirmation_gate_per_discovery_pass": "identical to discovery gate",
        "route_policy": "test all interfaces; retire failures; confirm every discovery pass; hidden access only for interfaces passing both partitions",
        "hidden_scope_if_qualified": "embedding plus all hidden states and candidate logits; no attention, MLP, parameters, gradients, PCA, TDA, or learned probes",
        "sequence": ["discovery behavior screen", "freeze all passes", "confirmation behavior", "freeze qualified interfaces", "lockbox all-state factorial observation"],
        "files": {
            "cases_sha256": core.sha(OUT / "material/active_cases.jsonl"),
            "compiled_sha256": core.sha(OUT / "compiled/qwen3_active.jsonl"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1547_c093_discovery_behavior_breadth_screen"
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"phase": 1546, "checks": checks, "token_ids": token_ids, "zero_models": zero_models, "max_prompt_length": max_length, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/pre_model_audit.json", audit)
    core.save(OUT / "analysis/final.json", {"phase": 1546, "campaign": "C093", "status": "contract_frozen", "authorization": protocol["authorization"]})
    print(json.dumps({"protocol": protocol, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
