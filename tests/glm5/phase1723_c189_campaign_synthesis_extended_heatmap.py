#!/usr/bin/env python3
"""C189: synthesize C172-C188 and export new-material parameter-coordinate heatmaps."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1723_c189_campaign_synthesis_extended_heatmap"
C183 = RESULT / "phase1717_c183_response_ecology_synthesis_heatmap"
C184 = RESULT / "phase1718_c184_response_ecology_invariant_discovery"
C185 = RESULT / "phase1719_c185_family_conditioned_routing_grammar"
C186 = RESULT / "phase1720_c186_new_material_response_ecology_prediction"
C187 = RESULT / "phase1721_c187_vocabulary_paraphrase_failure_decomposition"
C188 = RESULT / "phase1722_c188_new_material_generic_scaffold_prediction"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c189_new_material_response_scaffold_heatmap.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1723, "C189"


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = {"c183": C183, "c184": C184, "c185": C185, "c186": C186, "c187": C187, "c188": C188}
    checks = {name: core.load(path / "audit/independent_final_audit.json")["all_checks_passed"] for name, path in parents.items()}
    checks["authorization"] = "C189" in core.load(C188 / "audit/independent_final_audit.json")["authorization"]
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "campaign_synthesis_contract_frozen", "sources": list(parents), "asset": "27 observed family x vocabulary x paraphrase cells, full 2560 target-coordinate energy and signed-mean response rows, plus four aggregate cells", "claim_boundary": "parameter-level activation response display; not weights, semantic neurons, or unique causal circuit", "forbidden": ["attention", "MLP", "weights", "PCA", "editing parent results"], "producer_sha256": core.sha(Path(__file__)), "authorization": "build_synthesis_and_heatmap"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def load_cells():
    cells = {}
    existing = np.load(C186 / "raw/new_relation_role_response.float16.npy", mmap_mode="r")
    for row in core.rows(C186 / "raw/response_anchor_index.jsonl"):
        unit = 0 if row["partition"] == "new_confirmation" else 3
        cells[(row["family"], unit, row["phrase_variant"])] = np.asarray(existing[row["anchor_index"]], dtype=np.float32)
    cross = np.load(C187 / "raw/cross_cell_relation_response.float16.npy", mmap_mode="r")
    for row in core.rows(C187 / "raw/response_index.jsonl"):
        cells[(row["family"], row["unit"], row["phrase_variant"])] = np.asarray(cross[row["anchor_index"]], dtype=np.float32)
    return cells


def target_profile(values):
    energy = np.square(values, dtype=np.float64).sum(axis=(0, 1))
    return (energy / max(energy.sum(), 1e-30)).astype(np.float32)


def signed_mean(values):
    return np.mean(values, axis=(0, 1), dtype=np.float64).astype(np.float32)


def build():
    families = core.load(C186 / "protocol/preregistration.json")["families"]
    cells = load_cells()
    rows = []
    all_profiles = []
    for unit in (0, 3):
        for phrase_variant in (0, 1):
            aggregate = []
            for family in families:
                key = (family, unit, phrase_variant)
                if key not in cells:
                    continue
                values = cells[key]
                profile = target_profile(values)
                all_profiles.append(profile)
                aggregate.append(values)
                base = {"family": family, "unit": unit, "phrase_variant": phrase_variant, "cell": f"unit{unit}_phrase{phrase_variant}"}
                rows.append({**base, "kind": "target_energy_profile", "label": f"{family} / unit{unit} / phrase{phrase_variant} / target energy", "values": profile.tolist()})
                rows.append({**base, "kind": "signed_mean_response", "label": f"{family} / unit{unit} / phrase{phrase_variant} / signed mean", "values": signed_mean(values).tolist()})
            stacked = np.stack(aggregate)
            rows.append({"family": "aggregate", "unit": unit, "phrase_variant": phrase_variant, "cell": f"unit{unit}_phrase{phrase_variant}", "kind": "aggregate_target_energy_profile", "label": f"aggregate / unit{unit} / phrase{phrase_variant} / target energy", "observed_family_count": len(aggregate), "values": target_profile(stacked.reshape(-1, 6, 2560)).tolist()})
    variation = np.var(np.stack(all_profiles), axis=0)
    default_coordinates = np.argsort(-variation)[:64].astype(int).tolist()
    c184 = core.load(C184 / "analysis/final.json"); c185 = core.load(C185 / "analysis/final.json"); c186 = core.load(C186 / "analysis/final.json"); c187 = core.load(C187 / "analysis/final.json"); c188 = core.load(C188 / "analysis/final.json")
    synthesis = {"phase": PHASE, "campaign": CAMPAIGN, "status": "campaign_synthesized", "strongest_positive": "generic relation-source local response scaffold replicated across vocabulary and paraphrase cells", "fine_grained_boundary": "relation-family target profile is contract/phrase-conditioned and failed prospective paraphrase invariance", "cross_model_boundary": "no common behavioral family across two models in C181; internal cross-model topology untested", "mechanism_candidate": "stable role-conditioned propagation scaffold plus reconfigurable phrase/context-conditioned signed field", "mathematics": "existing local dynamical systems and conditional response maps suffice descriptively; new middle-level algebra is not yet forced", "next_campaign": "C190 multiple-paraphrase response-equivalence classes across additional language pattern families"}
    core.save(OUT / "analysis/campaign_synthesis.json", synthesis)
    payload = {"schema": "c189_new_material_response_scaffold_heatmap.v1", "result_type": "new_material_response_scaffold_heatmap", "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B", "title": "C184-C189 Generic Scaffold and Phrase-Conditioned Target Fields", "dimensions": list(range(2560)), "default_coordinates": default_coordinates, "rows": rows, "synthesis": synthesis, "evidence": {"c184": c184["headline"], "c185": c185["headline"], "c186": c186["headline"]["summary"], "c187": c187["headline"], "c188": c188["headline"]}, "coordinate_semantics": "Each column is a q25 physical activation coordinate. Energy rows sum squared relation-source responses over 64 q24 source coordinates and six target roles; signed rows retain mean response direction.", "claim_boundary": "The generic scaffold transfers across new vocabulary and paraphrases, while fine-grained family identity does not. These are activation responses, not weights, semantic neurons, or a complete language mechanism."}
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    asset = {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "sha256": core.sha(PUBLIC), "bytes": PUBLIC.stat().st_size, "rows": len(rows), "schema": payload["schema"]}
    core.save(OUT / "analysis/public_asset.json", asset)
    checks = {"observed_cells": len(cells) == 27, "rows": len(rows) == 58, "all_2560": all(len(row["values"]) == 2560 for row in rows), "four_aggregates": sum(row["kind"] == "aggregate_target_energy_profile" for row in rows) == 4, "finite": bool(np.isfinite(variation).all())}
    core.save(OUT / "audit/internal_build_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"synthesis": synthesis, "asset": asset, "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); asset = core.load(OUT / "analysis/public_asset.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "build": core.load(OUT / "audit/internal_build_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"], "asset_hash": core.sha(PUBLIC) == asset["sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": core.load(OUT / "analysis/campaign_synthesis.json"), "asset": asset, "next_authorization": "freeze_C190_multi_paraphrase_multi_pattern_campaign_before_new_runs"}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "build", "close")); args = parser.parse_args(); {"contract": contract, "build": build, "close": close}[args.command]()


if __name__ == "__main__": main()
