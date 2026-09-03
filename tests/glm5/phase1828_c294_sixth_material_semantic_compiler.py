#!/usr/bin/env python3
"""C294: compile and audit the frozen sixth material without loading model weights."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer
from model_utils import MODEL_CONFIGS

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT, PARENT = common.core, common.OUTS["C294"], common.OUTS["C293"]


def main() -> None:
    if (OUT / "protocol/preregistration.json").exists() and core.load(OUT / "protocol/preregistration.json").get("material_compiler_version") == "direct_base_v2":
        raise RuntimeError(OUT)
    parent = core.load(PARENT / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "pre_model_compile": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    rows = common.material()
    compiled = common.compile_qwen(tokenizer, rows)
    lengths = [len(row["prompt_ids"]) for row in compiled]
    role_checks = {
        "all_roles_present": all(set(row["role_positions"]) == set(common.ROLES) - {"boundary"} | {"boundary"} for row in compiled),
        "all_spans_nonempty": all(all(bool(span) for span in row["role_positions"].values()) for row in compiled),
        "within_width": max(lengths) <= common.WIDTH,
        "candidate_single_token": all(all(len(candidate) == 1 for candidate in row["candidate_ids"]) for row in compiled),
        "unique_case_ids": len({row["case_id"] for row in compiled}) == len(compiled),
        "rows": len(compiled) == 768,
    }
    if not all(role_checks.values()):
        raise RuntimeError(role_checks)
    for sub in ("analysis", "audit", "protocol", "compiled"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": 1828,
        "campaign": "C294",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "semantic_compilation_frozen",
        "material_compiler_version": "direct_base_v2",
        "model_tokenizer": "local Qwen3 tokenizer only; no model weights loaded",
        "roles": list(common.ROLES),
        "position_rule": "last matching span for query, first matching span for other semantic roles, final prompt token for boundary",
        "claim_boundary": "Compiler validity establishes token-role identity and width only; it is not behavior or mechanism evidence.",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.save(OUT / "audit/internal_compiler_audit.json", {"checks": role_checks, "all_checks_passed": all(role_checks.values())})
    report = {
        "phase": 1828,
        "campaign": "C294",
        "status": "closed",
        "checks": {**checks, **role_checks, "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]},
        "all_checks_passed": all(checks.values()) and all(role_checks.values()),
        "headline": {"rows": len(compiled), "min_tokens": min(lengths), "max_tokens": max(lengths), "semantic_roles": list(common.ROLES), "first_unit_identity": compiled[0]["role_values"]["primary"], "model_run": False},
        "next_authorization": "C295_qwen_sixth_material_full_field",
    }
    core.save(OUT / "analysis/final.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
