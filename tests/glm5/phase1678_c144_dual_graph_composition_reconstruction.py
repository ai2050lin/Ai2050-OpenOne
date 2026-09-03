#!/usr/bin/env python3
"""C144: language/activation dual graph and held-out composition reconstruction."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1678_c144_dual_graph_composition_reconstruction"
C142 = RESULT / "phase1676_c142_mobius_output_code_separation"
C143 = RESULT / "phase1677_c143_transition_model_competition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1676_c142_mobius_output_code_separation as c142

PHASE, CAMPAIGN = 1678, "C144"
ARMS, ROLES, CHECKPOINTS = c141.ARMS, c141.ROLES, c127.CHECKPOINTS
SUBSETS, NAMES = c142.SUBSETS, c142.SUBSET_NAMES
ORDERS = np.asarray([len(s) for s in SUBSETS])
EPS = np.asarray([[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)], np.float32)
SIGNS = np.asarray([[np.prod(e[list(s)]) for s in SUBSETS] for e in EPS], np.float32)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(train_coeff: np.ndarray, target_coeff: np.ndarray, order: int, *, permute: bool = False) -> dict:
    """Compare centered factorial fields without allocating the full cell tensor."""
    keep = np.flatnonzero(ORDERS <= order)
    dot = pred2 = target2 = sqerr = 0.0
    role_dot = np.zeros((6, 38), np.float64)
    role_p2 = np.zeros((6, 38), np.float64)
    role_t2 = np.zeros((6, 38), np.float64)
    role_err = np.zeros((6, 38), np.float64)
    if permute:
        source = keep[::-1]
    else:
        source = keep
    for ui in range(target_coeff.shape[0]):
        for ei in range(8):
            target = np.tensordot(SIGNS[ei], target_coeff[ui], axes=(0, 0))
            pred = np.tensordot(SIGNS[ei, keep], train_coeff[source], axes=(0, 0))
            p = pred.astype(np.float64, copy=False)
            t = target.astype(np.float64, copy=False)
            e = p - t
            dot += float(np.sum(p * t))
            pred2 += float(np.sum(p * p))
            target2 += float(np.sum(t * t))
            sqerr += float(np.sum(e * e))
            role_dot += np.sum(p * t, axis=-1)
            role_p2 += np.sum(p * p, axis=-1)
            role_t2 += np.sum(t * t, axis=-1)
            role_err += np.sum(e * e, axis=-1)
    cosine = dot / max(np.sqrt(pred2 * target2), 1e-30)
    relative_error = np.sqrt(sqerr / max(target2, 1e-30))
    rows = []
    for ri, role in enumerate(ROLES):
        for q, checkpoint in enumerate(CHECKPOINTS):
            rows.append({
                "role": role,
                "role_index": ri,
                "checkpoint": checkpoint,
                "checkpoint_index": q,
                "cosine": float(role_dot[ri, q] / max(np.sqrt(role_p2[ri, q] * role_t2[ri, q]), 1e-30)),
                "relative_error": float(np.sqrt(role_err[ri, q] / max(role_t2[ri, q], 1e-30))),
            })
    return {"cosine": float(cosine), "relative_error": float(relative_error), "role_checkpoint_rows": rows}


def discover() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C143 / "audit/independent_closure_audit.json")
    if not parent["all_checks_passed"] or parent["authorization"] != "start_C144":
        raise RuntimeError(parent)
    OUT.mkdir(parents=True)
    (OUT / "protocol").mkdir(); (OUT / "analysis").mkdir(); (OUT / "audit").mkdir()
    discovery = np.load(C142 / "analysis/discovery_mobius.float32.npy", mmap_mode="r")
    rows = {}
    score = {1: [], 2: [], 3: []}
    for ai, arm in enumerate(ARMS):
        rows[arm] = {}
        train = np.asarray(discovery[ai, :2]).mean(0)
        target = np.asarray(discovery[ai, 2:])
        for order in (1, 2, 3):
            result = metrics(train, target, order)
            rows[arm][f"order_{order}"] = {k: v for k, v in result.items() if k != "role_checkpoint_rows"}
            score[order].append(result["relative_error"])
    summaries = {str(order): {"median_relative_error": float(np.median(score[order])), "mean_relative_error": float(np.mean(score[order]))} for order in (1, 2, 3)}
    winner = min((1, 2, 3), key=lambda order: summaries[str(order)]["median_relative_error"])
    freeze = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "composition_order_frozen",
        "target": "output-code-averaged centered factorial HiddenState response",
        "orders": [1, 2, 3],
        "frozen_order": winner,
        "discovery_rows": rows,
        "discovery_summary": summaries,
        "confirmation_gate": {
            "relative_error_max": 0.95,
            "cosine_min": 0.30,
            "wrong_language_edge_error_margin_min": 0.05,
            "interaction_gain_over_order1_min_if_selected": 0.02,
        },
        "confirmation_asset_preexisting_but_unread_by_this_analysis": True,
        "source_hashes": {
            "discovery": core.sha(C142 / "analysis/discovery_mobius.float32.npy"),
            "confirmation_sealed": core.sha(C142 / "analysis/confirmation_mobius.float32.npy"),
        },
        "claim_boundary": "centered response reconstruction and typed graph edges, not a unique neural circuit",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "validate_C144",
    }
    core.save(OUT / "protocol/frozen_composition_model.json", freeze)
    checks = {
        "shape": list(discovery.shape) == [5, 4, 7, 6, 38, 2560],
        "orders": len(summaries) == 3,
        "finite": all(np.isfinite(v["median_relative_error"]) for v in summaries.values()),
        "sealed_hash": len(freeze["source_hashes"]["confirmation_sealed"]) == 64,
    }
    core.save(OUT / "audit/internal_discovery_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": freeze["authorization"]})
    print(json.dumps({"checks": checks, "summary": summaries, "winner": winner}, indent=2))


def validate() -> None:
    freeze = core.load(OUT / "protocol/frozen_composition_model.json")
    discovery = np.load(C142 / "analysis/discovery_mobius.float32.npy", mmap_mode="r")
    confirmation = np.load(C142 / "analysis/confirmation_mobius.float32.npy", mmap_mode="r")
    if core.sha(C142 / "analysis/confirmation_mobius.float32.npy") != freeze["source_hashes"]["confirmation_sealed"]:
        raise RuntimeError("confirmation hash drift")
    results, graph = {}, {"language_nodes": list(NAMES), "activation_nodes": [], "edges": []}
    all_scores = {1: [], 2: [], 3: []}
    wrong_scores = []
    for ai, arm in enumerate(ARMS):
        train = np.asarray(discovery[ai]).mean(0)
        target = np.asarray(confirmation[ai])
        results[arm] = {}
        for order in (1, 2, 3):
            result = metrics(train, target, order)
            results[arm][f"order_{order}"] = result
            all_scores[order].append(result["relative_error"])
        wrong = metrics(train, target, freeze["frozen_order"], permute=True)
        results[arm]["wrong_language_edge"] = {k: v for k, v in wrong.items() if k != "role_checkpoint_rows"}
        wrong_scores.append(wrong["relative_error"])
        frozen_nominees = core.load(C142 / "protocol/frozen_nominees.json")["nominees"][arm]
        for name in NAMES:
            node = f"{arm}:{frozen_nominees[name]['role']}@{frozen_nominees[name]['checkpoint']}"
            if node not in graph["activation_nodes"]:
                graph["activation_nodes"].append(node)
            replicated = core.load(C142 / "analysis/confirmation.json")["semantic_results"][arm][name]
            graph["edges"].append({
                "arm": arm,
                "language_effect": name,
                "activation_node": node,
                "confirmation_cosine": replicated["cosine"],
                "top256_overlap": replicated["top256_overlap"],
                "replicated": replicated["passed"],
            })
    summary = {str(order): {"median_relative_error": float(np.median(all_scores[order])), "mean_relative_error": float(np.mean(all_scores[order]))} for order in (1, 2, 3)}
    order = int(freeze["frozen_order"])
    target_error = summary[str(order)]["median_relative_error"]
    target_cos = float(np.median([results[arm][f"order_{order}"]["cosine"] for arm in ARMS]))
    wrong_error = float(np.median(wrong_scores))
    interaction_gain = summary["1"]["median_relative_error"] - target_error
    gate = freeze["confirmation_gate"]
    gates = {
        "relative_error": target_error <= gate["relative_error_max"],
        "cosine": target_cos >= gate["cosine_min"],
        "wrong_language_edge": wrong_error - target_error >= gate["wrong_language_edge_error_margin_min"],
        "interaction_gain": order == 1 or interaction_gain >= gate["interaction_gain_over_order1_min_if_selected"],
    }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "composition_reconstruction_adjudicated",
        "frozen_order": order,
        "summary": summary,
        "arm_results": results,
        "wrong_language_edge_median_relative_error": wrong_error,
        "interaction_gain_over_order1": interaction_gain,
        "frozen_order_median_cosine": target_cos,
        "gates": gates,
        "composition_gate_passed": all(gates.values()),
        "claim_boundary": freeze["claim_boundary"],
        "authorization": "close_C144_continue_C145",
    }
    core.save(OUT / "analysis/dual_graph.json", graph)
    core.save(OUT / "analysis/confirmation_reconstruction.json", report)
    checks = {
        "arms": len(results) == 5,
        "orders": all(all(f"order_{o}" in results[a] for o in (1, 2, 3)) for a in ARMS),
        "graph_edges": len(graph["edges"]) == 35,
        "finite": all(np.isfinite(v["median_relative_error"]) for v in summary.values()),
        "freeze": order == freeze["frozen_order"],
    }
    core.save(OUT / "audit/internal_confirmation_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_composition_gate_passed": all(gates.values()), "authorization": report["authorization"]})
    print(json.dumps({"frozen_order": order, "summary": summary, "wrong_error": wrong_error, "interaction_gain": interaction_gain, "cosine": target_cos, "gates": gates}, indent=2))


def close() -> None:
    result = core.load(OUT / "analysis/confirmation_reconstruction.json")
    checks = {
        "discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"],
        "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"],
        "typed_graph": len(core.load(OUT / "analysis/dual_graph.json")["edges"]) == 35,
    }
    closure = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "dual_graph_composition_closed",
        "headline": {"frozen_order": result["frozen_order"], "gate_passed": result["composition_gate_passed"], "summary": result["summary"]},
        "theory_update": "tests whether low-order typed language effects reconstruct held-out centered activation fields",
        "claim_boundary": result["claim_boundary"],
        "next_authorization": "C145 correct/error trajectory atlas regardless of composition result",
    }
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_final_then_C145"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"discover": discover, "validate": validate, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("discover|validate|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
