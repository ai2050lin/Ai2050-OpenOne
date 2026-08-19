#!/usr/bin/env python3
"""Phase1328: freeze C041 with identity-balanced candidate anchors."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1327_c040_cross_model_noun_relation_contract as base  # noqa: E402

PHASE, CAMPAIGN = 1328, "C041"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1328_c041_balanced_noun_relation_contract_audit.py"
SCAFFOLD = T / "phase1327_c040_cross_model_noun_relation_contract.py"
PARENT = T / "result/phase1327_c040_cross_model_noun_relation_contract"
OUT = T / "result/phase1328_c041_balanced_noun_relation_contract"

WORDS = {
    "discovery": {
        "fruit": ("banana", "pear", "lemon", "peach"),
        "animal": ("rabbit", "dolphin", "tiger", "eagle"),
        "tool": ("hammer", "wrench", "knife", "level"),
        "vehicle": ("train", "bicycle", "boat", "canoe"),
    },
    "confirmation": {
        "fruit": ("orange", "cherry", "mango", "guava"),
        "animal": ("panda", "turtle", "monkey", "zebra"),
        "tool": ("shovel", "pliers", "auger", "lathe"),
        "vehicle": ("scooter", "tractor", "wagon", "ferry"),
    },
    "holdout": {
        "fruit": ("coconut", "plum", "berry", "melon"),
        "animal": ("camel", "otter", "penguin", "koala"),
        "tool": ("chisel", "trowel", "spade", "plier"),
        "vehicle": ("airplane", "motorcycle", "moped", "coupe"),
    },
}
ANCHORS = {
    "fruit": ("apple", "grape"),
    "animal": ("horse", "sheep"),
    "tool": ("clamp", "drill"),
    "vehicle": ("truck", "sedan"),
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def terminal_parent() -> tuple[bool, dict[str, Any]]:
    final = load(PARENT / "analysis/final.json")
    failure = load(PARENT / "audit/independent_failure_audit.json")
    ok = (final.get("authorization") == "stop_c040_before_model"
          and failure.get("all_checks_passed") is True
          and failure.get("authorization") == "close_c040_and_permit_new_independent_contract")
    return ok, {"final_authorization": final.get("authorization"),
                "failure_audit": failure.get("all_checks_passed"),
                "failure_authorization": failure.get("authorization")}


def build_behavior(tokenizers: dict[str, Any], graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    case_index = 0
    for item_index, item in enumerate(graph):
        target, family, partition = item["word"], item["family"], item["partition"]
        wrong_families = [name for name in base.FAMILIES if name != family]
        for wrong_index, wrong_family in enumerate(wrong_families):
            signature_index = (item_index + wrong_index) % 2
            correct = ANCHORS[family][signature_index]
            wrong = ANCHORS[wrong_family][signature_index]
            correct_signature = base.token_signature(tokenizers, correct)
            wrong_signature = base.token_signature(tokenizers, wrong)
            if len(correct) != len(wrong) or len(correct) != 5 or correct_signature != wrong_signature:
                raise RuntimeError(f"unmatched anchors: {correct}/{wrong}")
            set_key = f"{partition}:{target}:{wrong_family}"
            for surface, template in base.BEHAVIOR_SURFACES.items():
                for order in (0, 1):
                    candidates = [correct, wrong] if order == 0 else [wrong, correct]
                    prompt = template.format(target=target, a=candidates[0], b=candidates[1])
                    output.append({
                        "case_id": f"c041-{case_index:04d}", "semantic_set": set_key,
                        "partition": partition, "surface": surface, "target": target,
                        "target_family": family, "target_supergroup": base.SUPERGROUP[family],
                        "wrong_family": wrong_family, "candidates": candidates,
                        "candidate_order": order, "gold_value": correct,
                        "gold_position": candidates.index(correct), "prompt": prompt,
                    })
                    case_index += 1
    return output


def configure() -> None:
    base.PHASE, base.CAMPAIGN = PHASE, CAMPAIGN
    base.SCRIPT, base.AUDITOR = SCRIPT, AUDITOR
    base.PARENT, base.OUT = PARENT, OUT
    base.MATERIAL = OUT / "material/frozen_concept_graph.json"
    base.BEHAVIOR = OUT / "material/frozen_behavior_cases.jsonl"
    base.CONTEXT = OUT / "material/frozen_context_cases.jsonl"
    base.NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
    base.TOKEN_AUDIT = OUT / "audit/tokenizer_zero_model_audit.json"
    base.PROTOCOL = OUT / "protocol/preregistration.json"
    base.FINAL = OUT / "analysis/final.json"
    base.WORDS = WORDS
    base.CANDIDATE_POOLS = {partition: ANCHORS for partition in base.PARTITIONS}
    base.terminal_parent = terminal_parent
    base.build_behavior = build_behavior


def normalize_outputs() -> None:
    protocol = load(base.PROTOCOL)
    protocol["schema"] = "phase1328.c041.identity_balanced_relation_contract.v1"
    protocol["research_object"] = "identity-balanced cross-model within-model common-noun relation kernels"
    protocol["material"]["candidate_anchors"] = ANCHORS
    protocol["material"]["candidate_anchors_disjoint_from_targets"] = True
    protocol["implementation_scaffold_sha256"] = base.sha(SCAFFOLD)
    protocol["repair_scope"] = (
        "New campaign: candidate anchors are disjoint from target concepts and assigned with exact correct/wrong identity symmetry."
    )
    timeless = {key: value for key, value in protocol.items()
                if key not in {"contract_sha256", "script_sha256", "auditor_sha256", "created_at_utc"}}
    protocol["contract_sha256"] = base.digest(timeless)
    protocol["script_sha256"] = base.sha(SCRIPT)
    protocol["auditor_sha256"] = base.sha(AUDITOR)
    protocol["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    base.save(base.PROTOCOL, protocol)
    final = load(base.FINAL)
    final["phase"], final["campaign"] = PHASE, CAMPAIGN
    final["protocol_sha256"] = base.sha(base.PROTOCOL)
    final["candidate_identity_exact_balance"] = (
        final["zero_models"]["candidate_identity_majority"] == 0.5
    )
    final["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    base.save(base.FINAL, final)


def build(force: bool) -> None:
    configure()
    base.build(force)
    normalize_outputs()
    print(json.dumps(load(base.FINAL), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.force)


if __name__ == "__main__":
    main()
