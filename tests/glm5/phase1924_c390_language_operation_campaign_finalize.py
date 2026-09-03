#!/usr/bin/env python3
"""Transactional C390 synthesis and HiddenState cleanup finalizer.

The frozen producer opened the full-token memmap and attempted to unlink it
before closing the mapping. This supplemental finalizer preserves every
scientific gate, closes all mappings, writes a provisional checksum manifest,
and only then removes bulk arrays.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1903_c369_c390_language_operation_graph_campaign as campaign


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_memmap(value) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def begin() -> Path:
    out = campaign.OUTS["C390"]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    checks = {
        "parent": campaign.final("C389")["all_checks_passed"],
        "all_prior_phases_closed": all(campaign.final(f"C{value}")["all_checks_passed"] for value in range(369, 390)),
        "failed_attempt_preserved": (campaign.RESULT / "c390_superseded_pre_memmap_lifecycle_fix_20260824").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": 1924,
        "campaign": "C390",
        "created_at_utc": utc_now(),
        "producer_sha256": producer_hash(),
        "status": "supplemental_transactional_synthesis_frozen",
        "gates": ["cross-material", "conditional I", "semantic order", "graph depth", "natural external", "bilingual abstract response", "causal", "new mathematics"],
        "visualization": "all 2560 Qwen coordinates for 16 families x A/B/I/K at five checkpoints plus one all-token field",
        "cleanup": "close mappings, checksum every existing bulk array, commit provisional manifest, remove, verify, commit final manifest",
        "known_lifecycle_incident": "C374 role array was removed by the failed frozen producer before its in-memory manifest was committed; expected shape is retained but no checksum is invented",
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def close(out: Path, headline: dict, checks: dict) -> dict:
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    protocol = json.loads((out / "protocol/preregistration.json").read_text(encoding="utf-8"))
    final_checks = {
        "contract": json.loads((out / "audit/internal_contract_audit.json").read_text(encoding="utf-8"))["all_checks_passed"],
        "analysis": all(checks.values()),
        "producer_hash": protocol["producer_sha256"] == producer_hash(),
    }
    value = {
        "phase": 1924,
        "campaign": "C390",
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": headline,
        "next_authorization": "independent_audit_and_next_cross_construction_lockbox",
    }
    save(out / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def main() -> None:
    out = begin()
    if (out / "analysis/final.json").exists():
        print((out / "analysis/final.json").read_text(encoding="utf-8"), flush=True)
        return

    abstractions = {name: campaign.model_response_abstraction(name) for name in ("C387", "C388", "C389")}
    save(out / "analysis/model_response_abstractions.json", abstractions)
    model_pairs = []
    for left, right in (("C387", "C388"), ("C387", "C389"), ("C388", "C389")):
        keys = sorted(set(abstractions[left]["vectors"]) & set(abstractions[right]["vectors"]))
        distances = [campaign.total_variation(abstractions[left]["vectors"][key], abstractions[right]["vectors"][key]) for key in keys]
        model_pairs.append({"left": left, "right": right, "common_states": len(keys), "mean_total_variation": float(np.mean(distances)) if distances else None})
    write_rows(out / "analysis/cross_model_pairs.jsonl", model_pairs)

    bilingual = {}
    for name, abstraction in abstractions.items():
        distances = []
        for family, operation in itertools.product(campaign.BILINGUAL_FAMILIES, ("A", "B", "I")):
            en, zh = f"{family}:en:{operation}", f"{family}:zh:{operation}"
            if en in abstraction["vectors"] and zh in abstraction["vectors"]:
                distances.append(campaign.total_variation(abstraction["vectors"][en], abstraction["vectors"][zh]))
        bilingual[name] = {"states": len(distances), "mean_total_variation": float(np.mean(distances)) if distances else None}
    save(out / "analysis/bilingual_consistency.json", bilingual)

    mean = np.load(campaign.OUTS["C375"] / "analysis/family_operation_mean_response.float16.npy", mmap_mode="r")
    q_indices = (0, 12, 24, 36, 37)
    boundary_i = campaign.ROLES.index("boundary")
    heat_rows = []
    for family_i, family in enumerate(campaign.FAMILIES):
        for operation_i, operation in enumerate(campaign.OPS):
            for checkpoint in q_indices:
                heat_rows.append({
                    "id": f"{family}:{operation}:q{checkpoint}:boundary",
                    "family": family,
                    "operation": operation,
                    "checkpoint": checkpoint,
                    "role": "boundary",
                    "values": np.asarray(mean[family_i, operation_i, checkpoint, boundary_i], np.float32).round(6).tolist(),
                })
    close_memmap(mean)
    del mean

    full_path = campaign.OUTS["C374"] / "raw/full_fields_holdout.float16.npy"
    full = np.load(full_path, mmap_mode="r")
    field_map = campaign.load(campaign.OUTS["C374"] / "raw/full_field_row_map.json")
    hidden_index = campaign.read_rows(campaign.OUTS["C374"] / "raw/hidden_index.jsonl")
    source_i = field_map["source_indices"][0]
    length = hidden_index[source_i]["length"]
    token_rows = []
    for checkpoint in (0, 24, 37):
        for token in range(length):
            token_rows.append({
                "id": f"token:{token}:q{checkpoint}",
                "token": token,
                "checkpoint": checkpoint,
                "values": np.asarray(full[0, checkpoint, token], np.float32).round(6).tolist(),
            })
    close_memmap(full)
    del full
    gc.collect()

    visual = {
        "schema": "c390.language_operation_full_coordinate.v1",
        "phase": 1924,
        "campaign": "C390",
        "model": "Qwen3-4B",
        "dimensions": list(range(2560)),
        "family_operation_rows": heat_rows,
        "all_token_rows": token_rows,
        "checkpoints": list(q_indices),
        "roles": ["boundary"],
        "claim_boundary": "Mean signed response rows and one complete token field are parameter-level observations, not causal semantic coordinates.",
    }
    visual_path = ROOT / "frontend/public/vis_data/research_kernel/c390_language_operation_full_coordinate.json"
    save(visual_path, visual)

    gates = {
        "cross_material_any": len(campaign.final("C376")["headline"]["families_with_any_qualified_cross_cell"]) > 0,
        "conditional_i_breadth": campaign.final("C377")["headline"]["passed_count"] >= 8,
        "semantic_order": campaign.final("C378")["headline"]["semantic_scope_result"]["gain"] > 0,
        "graph_depth": campaign.final("C382")["headline"].get("recursive_depth_candidate", False),
        "natural_external": campaign.final("C383")["headline"]["hidden_state_eligible"],
        "bilingual_all_models": all(campaign.final(name)["headline"]["abstract_response_eligible"] for name in ("C387", "C388", "C389")),
        "causal": campaign.final("C386")["headline"]["causal_claim"],
    }
    gates["new_math"] = gates["conditional_i_breadth"] and gates["causal"] and gates["bilingual_all_models"]

    cleanup_paths = [
        campaign.OUTS["C374"] / "raw/role_states.float16.npy",
        campaign.OUTS["C374"] / "raw/full_fields_holdout.float16.npy",
        campaign.OUTS["C381"] / "raw/role_states.float16.npy",
        campaign.OUTS["C383"] / "raw/role_states.float16.npy",
        campaign.OUTS["C387"] / "raw/role_states.float16.npy",
        campaign.OUTS["C388"] / "raw/role_states.float16.npy",
        campaign.OUTS["C389"] / "raw/role_states.float16.npy",
    ]
    manifest = []
    for path in cleanup_paths:
        if path.exists():
            array = np.load(path, mmap_mode="r")
            shape = list(array.shape)
            close_memmap(array)
            del array
            manifest.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
                "shape": shape,
                "status": "checksum_committed_pending_removal",
            })
        elif path == cleanup_paths[0]:
            manifest.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": None,
                "sha256": None,
                "shape": campaign.final("C374")["headline"]["role_shape"],
                "status": "removed_by_failed_frozen_cleanup_before_manifest_commit",
            })
    save(out / "audit/hidden_state_cleanup_manifest.provisional.json", manifest)
    gc.collect()
    for item in manifest:
        path = ROOT / item["path"]
        if path.exists():
            path.unlink()
            item["status"] = "checksum_committed_and_removed"
        item["removed_after_analysis"] = not path.exists()
    save(out / "audit/hidden_state_cleanup_manifest.json", manifest)

    removed_bytes = sum(item["bytes"] or 0 for item in manifest)
    headline = {
        "status": "language_operation_campaign_closed",
        "gates": gates,
        "cross_model_pairs": model_pairs,
        "bilingual_consistency": bilingual,
        "visual_rows": {"family_operation": len(heat_rows), "all_token": len(token_rows), "coordinates": 2560},
        "cleanup": {"files": len(manifest), "known_precommit_loss_files": 1, "verified_bytes_removed": removed_bytes},
        "new_math_gate_passed": gates["new_math"],
        "strict_interpretation": "Typed language candidates map to full-coordinate responses. Positive transfer is local; no universal operator, causal language algebra, cross-model coordinate identity, or new mathematics is claimed.",
    }
    checks = {
        "phases": all(campaign.final(f"C{value}")["all_checks_passed"] for value in range(369, 390)),
        "visual_coordinates": len(visual["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in heat_rows + token_rows),
        "cleanup": all(not (ROOT / item["path"]).exists() for item in manifest),
        "manifest_before_removal": (out / "audit/hidden_state_cleanup_manifest.provisional.json").exists(),
        "finite": campaign.finite(headline),
    }
    close(out, headline, checks)


if __name__ == "__main__":
    main()
