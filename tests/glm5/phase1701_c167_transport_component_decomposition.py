#!/usr/bin/env python3
"""C167: observational decomposition of the C161 full-coordinate response graph."""
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
PARENT = RESULT / "phase1695_c161_full_coordinate_local_transmission"
OUT = RESULT / "phase1701_c167_transport_component_decomposition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1701, "C167"
DIM, ROLES, BLOCK = 2560, ("primary", "secondary", "relation", "context", "query", "boundary"), 16
PANELS = ("natural_lexical", "isomorphic_nonce")
RELATIONS = ("is_a", "part_of", "located_in", "precedes")


def now():
    return datetime.now(timezone.utc).isoformat()


def cos_rows(a, b, source_ids=None, remove_identity=False):
    af = a.reshape(len(a), -1).astype(np.float64, copy=False)
    bf = b.reshape(len(b), -1).astype(np.float64, copy=False)
    dot = np.sum(af * bf, axis=1)
    an2 = np.sum(af * af, axis=1)
    bn2 = np.sum(bf * bf, axis=1)
    if remove_identity:
        role = ROLES.index("relation")
        local = np.arange(len(a))
        av = a[local, role, source_ids].astype(np.float64)
        bv = b[local, role, source_ids].astype(np.float64)
        dot -= av * bv
        an2 -= av * av
        bn2 -= bv * bv
    return dot / np.maximum(np.sqrt(np.maximum(an2, 0) * np.maximum(bn2, 0)), 1e-12)


def decompose(x):
    """Balanced two-factor decomposition: panel x relation."""
    shared = x.mean(axis=(0, 1))
    panel = x.mean(axis=1) - shared[None]
    relation = x.mean(axis=0) - shared[None]
    interaction = x - shared[None, None] - panel[:, None] - relation[None]
    return shared, panel, relation, interaction


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    anchors = core.rows(PARENT / "material/anchors.jsonl")
    raw = PARENT / "raw/q24_relation_to_q25_six_role_response.float16.npy"
    checks = {
        "parent_audit": parent_audit["all_checks_passed"],
        "anchors": len(anchors) == 16,
        "balanced": len({(r["partition"], r["panel"], r["relation_family"]) for r in anchors}) == 16,
        "raw": raw.exists(),
        "shape": list(np.load(raw, mmap_mode="r").shape) == [16, DIM, 6, DIM],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "observational_decomposition_frozen",
        "source": "C161 already observed full-coordinate finite-response tensor",
        "epistemic_status": "retrospective observation; thresholds are descriptive labels, not prospective confirmation",
        "decomposition": "R(partition,panel,relation)=shared+panel+relation+panel_relation_interaction",
        "comparisons": [
            "discovery-confirmation cosine for every component",
            "matched relation versus wrong relation before and after removing same-coordinate edge",
            "balanced energy fractions",
        ],
        "descriptive_thresholds": {
            "component_replication_cosine": 0.30,
            "relation_margin": 0.05,
            "identity_removed_relation_margin": 0.03,
        },
        "forbidden": ["attention", "MLP", "weights", "PCA", "prospective language for reused data"],
        "source_hash": core.sha(raw),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_existing_tensor",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps({"checks": checks, "protocol": protocol}, indent=2))


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    anchors = core.rows(PARENT / "material/anchors.jsonl")
    raw = np.load(PARENT / "raw/q24_relation_to_q25_six_role_response.float16.npy", mmap_mode="r")
    index = {(r["partition"], r["panel"], r["relation_family"]): r["anchor_index"] for r in anchors}
    component_cosines = {k: [] for k in ("shared", "panel", "relation", "interaction")}
    relation_margins, conditional_margins = [], []
    relation_margins_no_identity, conditional_margins_no_identity = [], []
    energy = {part: {k: 0.0 for k in ("total", "shared", "panel", "relation", "interaction")} for part in ("discovery", "confirmation")}
    relation_strength = np.zeros(DIM, np.float64)

    for start in range(0, DIM, BLOCK):
        stop = min(start + BLOCK, DIM)
        source_ids = np.arange(start, stop)
        components = {}
        for part in ("discovery", "confirmation"):
            x = np.stack([
                np.stack([np.asarray(raw[index[(part, panel, relation)], start:stop], np.float32) for relation in RELATIONS])
                for panel in PANELS
            ])
            shared, panel, relation, interaction = decompose(x)
            components[part] = (x, shared, panel, relation, interaction)
            energy[part]["total"] += float(np.sum(x.astype(np.float64) ** 2))
            energy[part]["shared"] += 8.0 * float(np.sum(shared.astype(np.float64) ** 2))
            energy[part]["panel"] += 4.0 * float(np.sum(panel.astype(np.float64) ** 2))
            energy[part]["relation"] += 2.0 * float(np.sum(relation.astype(np.float64) ** 2))
            energy[part]["interaction"] += float(np.sum(interaction.astype(np.float64) ** 2))
            if part == "discovery":
                relation_strength[start:stop] = np.mean(np.sum(relation.astype(np.float64) ** 2, axis=(2, 3)), axis=0)

        xd, sd, pd, rd, id_ = components["discovery"]
        xc, sc, pc, rc, ic = components["confirmation"]
        component_cosines["shared"].extend(cos_rows(sd, sc).tolist())
        for pi in range(len(PANELS)):
            component_cosines["panel"].extend(cos_rows(pd[pi], pc[pi]).tolist())
        for ri in range(len(RELATIONS)):
            component_cosines["relation"].extend(cos_rows(rd[ri], rc[ri]).tolist())
            matched = cos_rows(rd[ri], rc[ri])
            wrong = np.median(np.stack([cos_rows(rd[ri], rc[wj]) for wj in range(len(RELATIONS)) if wj != ri]), axis=0)
            matched_no = cos_rows(rd[ri], rc[ri], source_ids, True)
            wrong_no = np.median(np.stack([cos_rows(rd[ri], rc[wj], source_ids, True) for wj in range(len(RELATIONS)) if wj != ri]), axis=0)
            relation_margins.extend((matched - wrong).tolist())
            relation_margins_no_identity.extend((matched_no - wrong_no).tolist())
        for pi in range(len(PANELS)):
            for ri in range(len(RELATIONS)):
                component_cosines["interaction"].extend(cos_rows(id_[pi, ri], ic[pi, ri]).tolist())
                dcond = rd[ri] + id_[pi, ri]
                matched = cos_rows(dcond, rc[ri] + ic[pi, ri])
                wrong = np.median(np.stack([cos_rows(dcond, rc[wj] + ic[pi, wj]) for wj in range(len(RELATIONS)) if wj != ri]), axis=0)
                matched_no = cos_rows(dcond, rc[ri] + ic[pi, ri], source_ids, True)
                wrong_no = np.median(np.stack([cos_rows(dcond, rc[wj] + ic[pi, wj], source_ids, True) for wj in range(len(RELATIONS)) if wj != ri]), axis=0)
                conditional_margins.extend((matched - wrong).tolist())
                conditional_margins_no_identity.extend((matched_no - wrong_no).tolist())

    energy_fractions = {
        part: {k: energy[part][k] / max(energy[part]["total"], 1e-12) for k in ("shared", "panel", "relation", "interaction")}
        for part in energy
    }
    component_summary = {
        key: {
            "count": len(values),
            "median_cosine": float(np.median(values)),
            "positive_rate": float(np.mean(np.asarray(values) > 0)),
            "above_descriptive_threshold_rate": float(np.mean(np.asarray(values) >= protocol["descriptive_thresholds"]["component_replication_cosine"])),
        }
        for key, values in component_cosines.items()
    }
    top = np.argsort(relation_strength)[-64:][::-1]
    top16 = top[:16]
    selected = np.stack([
        np.stack([
            decompose(np.stack([
                np.stack([np.asarray(raw[index[(part, panel, relation)], top16], np.float32) for relation in RELATIONS])
                for panel in PANELS
            ]))[2]
            for part in ("discovery", "confirmation")
        ])
    ])[0]
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/top_relation_component_fields.float16.npy", selected.astype(np.float16))
    core.save(OUT / "analysis/top_relation_source_coordinates.json", {"coordinates": top.tolist(), "top16_tensor_shape": list(selected.shape)})
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "retrospective_transport_decomposition_complete",
        "component_replication": component_summary,
        "energy_fractions": energy_fractions,
        "relation_identity_margin": {
            "median": float(np.median(relation_margins)),
            "positive_rate": float(np.mean(np.asarray(relation_margins) > 0)),
            "identity_removed_median": float(np.median(relation_margins_no_identity)),
            "identity_removed_positive_rate": float(np.mean(np.asarray(relation_margins_no_identity) > 0)),
        },
        "panel_conditioned_relation_margin": {
            "median": float(np.median(conditional_margins)),
            "positive_rate": float(np.mean(np.asarray(conditional_margins) > 0)),
            "identity_removed_median": float(np.median(conditional_margins_no_identity)),
            "identity_removed_positive_rate": float(np.mean(np.asarray(conditional_margins_no_identity) > 0)),
        },
        "descriptive_labels": {
            "shared_replication": component_summary["shared"]["median_cosine"] >= protocol["descriptive_thresholds"]["component_replication_cosine"],
            "relation_component_replication": component_summary["relation"]["median_cosine"] >= protocol["descriptive_thresholds"]["component_replication_cosine"],
            "relation_separation": float(np.median(relation_margins)) >= protocol["descriptive_thresholds"]["relation_margin"],
            "identity_removed_relation_separation": float(np.median(relation_margins_no_identity)) >= protocol["descriptive_thresholds"]["identity_removed_relation_margin"],
        },
        "top_relation_source_coordinates": top.tolist(),
        "claim_boundary": "Balanced retrospective decomposition of C161 finite responses; not prospective confirmation, unique circuit recovery, or natural-use evidence.",
        "next_authorization": "independent audit and synthesis; new data required for prospective relation-residual confirmation",
    }
    core.save(OUT / "analysis/decomposition.json", report)
    checks = {
        "component_counts": all(v["count"] > 0 for v in component_summary.values()),
        "energy_partition": all(abs(sum(v.values()) - 1.0) < 2e-4 for v in energy_fractions.values()),
        "margins": len(relation_margins) == DIM * 4 and len(conditional_margins) == DIM * 8,
        "coordinates": len(set(top.tolist())) == 64,
        "tensor": list(selected.shape) == [2, 4, 16, 6, DIM],
        "finite": bool(np.isfinite(selected).all() and all(np.isfinite(list(v.values())).all() for v in energy_fractions.values())),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "report": report}, indent=2))


def close():
    report = core.load(OUT / "analysis/decomposition.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
    }
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "headline": {
            "component_replication": report["component_replication"],
            "energy_fractions": report["energy_fractions"],
            "relation_identity_margin": report["relation_identity_margin"],
            "panel_conditioned_relation_margin": report["panel_conditioned_relation_margin"],
            "descriptive_labels": report["descriptive_labels"],
        },
        "epistemic_status": "retrospective observation",
        "next_authorization": report["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
