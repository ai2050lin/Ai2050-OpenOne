#!/usr/bin/env python3
"""Independent zero-model audit for Phase1304 C033."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1304_c033_role_typed_causal_graph_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "material/frozen_role_typed_lookup_cases.jsonl"
N = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"
MAIN = T / "phase1304_c033_role_typed_causal_graph_contract.py"
SCRIPT = Path(__file__).resolve()
MODEL_PATH = str(__import__("sys").path and ROOT / "models" / "qwen3")


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    protocol, material, review, machine = load(P), rows(M), load(N), load(MACHINE)
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "sources", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(checks, "phase_campaign", (protocol["phase"], protocol["campaign"]) == (1304, "C033"), [protocol["phase"], protocol["campaign"]])
    add(checks, "material_hash", protocol["material"]["material_sha256"] == sha(M), sha(M))
    add(checks, "naturalness_hash", protocol["material"]["naturalness_sha256"] == sha(N), sha(N))
    add(checks, "case_count", len(material) == 6912, len(material))
    add(checks, "case_unique", len({x["case_id"] for x in material}) == 6912, len({x["case_id"] for x in material}))
    dims = Counter((x["partition"], x["panel"], x["surface"], x["binding_state"], x["candidate_order"]) for x in material)
    add(checks, "factorial_balance", len(dims) == 144 and len(set(dims.values())) == 1, {"cells": len(dims), "counts": sorted(set(dims.values()))})
    add(checks, "attributes", {x["attribute"] for x in material} == {"color", "material", "location", "size", "shape", "status"}, sorted({x["attribute"] for x in material}))
    semantic = all(sum(fields[x["attribute"]] == x["target_value"] for fields in x["assignments"].values()) == 1 for x in material)
    gold = all([entity for entity, fields in x["assignments"].items() if fields[x["attribute"]] == x["target_value"]] == [x["gold_candidate"]] for x in material)
    add(checks, "semantic_unique", semantic, semantic)
    add(checks, "gold_recomputed", gold, gold)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in material:
        grouped[x["group_id"]].append(x)
    add(checks, "pairs", len(grouped) == 3456 and all(len(v) == 2 and {x["binding_state"] for x in v} == {0, 1} for v in grouped.values()), len(grouped))
    add(checks, "active_changes", all(v[0]["gold_candidate"] != v[1]["gold_candidate"] for k, v in grouped.items() if "|active|" in k), "active")
    add(checks, "null_preserves", all(v[0]["gold_candidate"] == v[1]["gold_candidate"] for k, v in grouped.items() if "|matched_null|" in k), "null")
    add(checks, "typed_spans", all(len(x["typed_spans"]["records"]) == 3 and len(x["typed_spans"]["query"]) == 1 and len(x["typed_spans"]["answer_boundary"]) == 1 for x in material), "all")
    surface_ok = all(x["candidate_prompt"].endswith("Answer:") and x["candidate_prompt"].count("?") == 1 and "  " not in x["candidate_prompt"] for x in material)
    add(checks, "surface_form", surface_ok, surface_ok)
    articles = True
    for x in material:
        for article, word in re.findall(r"\b(a|an) ([A-Za-z-]+) (?:color|shape)\b", x["candidate_prompt"]):
            articles &= article == ("an" if word[0].lower() in "aeiou" else "a")
    add(checks, "articles", articles, articles)
    add(checks, "naturalness", review["all_checks_passed"] and review["semantic_uniqueness_recomputed_for_all_cases"], review)
    add(checks, "machine", machine["all_machine_checks_passed"], machine)
    add(checks, "fresh_lexicon", not protocol["material"]["prior_entity_overlap"] and not protocol["material"]["prior_value_overlap"], protocol["material"])
    # Load only the tokenizer through the registry path; never load model weights.
    import sys
    sys.path.insert(0, str(T))
    from model_utils import MODEL_CONFIGS
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True)
    names = sorted({e for x in material for e in x["entities"]})
    add(checks, "single_token_entities", len(names) == 72 and all(len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1 for n in names), len(names))
    add(checks, "single_token_candidates", all(len(tokenizer.encode(" " + c, add_special_tokens=False)) == 1 for x in material for c in x["candidates"]), "all")
    shortcuts = {
        "candidate_first": sum(x["candidates"][0] == x["gold_candidate"] for x in material) / len(material),
        "record_first": sum(x["record_order"][0] == x["gold_candidate"] for x in material) / len(material),
        "entity_first": sum(x["entities"][0] == x["gold_candidate"] for x in material) / len(material),
    }
    add(checks, "shortcut", max(shortcuts.values()) <= 0.70, shortcuts)
    graph = protocol["role_typed_graph"]
    add(checks, "typed_graph", graph["aggregator_candidate"] == {"event": "assistant_answer_boundary", "depth": 26} and graph["block_site"]["depth"] == 25 and graph["rescue_site"]["depth"] == 26, graph)
    add(checks, "no_user_cue", graph["user_cue_role"].startswith("not tested") and all("user" not in json.dumps(protocol[key]) for key in ("hidden", "bidirectional_swap", "cross_surface_block_rescue")), graph["user_cue_role"])
    branches = protocol["branches"]
    add(checks, "branch_chain", branches["phase1305_fail"] == "close_c033_without_hidden" and branches["phase1306_fail"] == "close_c033_without_causal" and branches["phase1307_fail"] == "close_c033_without_rescue" and branches["phase1308_fail"] == "close_c033_at_rescue_boundary", branches)
    add(checks, "weights_not_loaded", protocol["model_weights_loaded"] is False and load(OUT / "protocol/environment_snapshot.json")["model_weights_loaded"] is False, False)
    passed = all(x["passed"] for x in checks)
    authorization = "phase1305_qwen3_behavior_only" if passed else "none"
    result = {
        "phase": 1304,
        "campaign": "C033",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(x["passed"] for x in checks),
        "total_count": len(checks),
        "all_checks_passed": passed,
        "authorization": authorization,
        "protocol_digest": protocol["protocol_digest"],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final = load(FINAL)
    final.update({"verdict": "contract_qualified" if passed else "contract_audit_failed", "authorization": authorization, "audit": f"{result['passed_count']}/{result['total_count']}"})
    FINAL.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"passed": result["passed_count"], "total": result["total_count"], "authorization": authorization}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
