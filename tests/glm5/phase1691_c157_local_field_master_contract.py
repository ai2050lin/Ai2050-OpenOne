#!/usr/bin/env python3
"""C157: adjudicate C140-C156 and freeze the local-field batch campaign."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
OUT = RESULT / "phase1691_c157_local_field_master_contract"
ATTACHMENTS = (
    Path(r"C:/Users/Admin/.codex/attachments/d5b63e8b-93c1-4d85-84d4-ab6b9c131302/pasted-text.txt"),
    Path(r"C:/Users/Admin/.codex/attachments/d5265287-12bc-4986-b2ac-69823948046b/pasted-text.txt"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite frozen C157: {OUT}")

    evidence_paths = {
        "C153": RESULT / "phase1687_c153_type_graph_conditional_pool_confirmation/analysis/confirmation.json",
        "C154": RESULT / "phase1688_c154_type_graph_hiddenstate_causal_adjudication/analysis/causal.json",
        "C155": RESULT / "phase1689_c155_checkpoint_transfer_curve/analysis/transfer_curve.json",
        "C156": RESULT / "phase1690_c156_multiroute_campaign_final_audit/analysis/final.json",
    }
    missing = [str(path) for path in (*ATTACHMENTS, *evidence_paths.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    c153 = load(evidence_paths["C153"])
    c154 = load(evidence_paths["C154"])
    c155 = load(evidence_paths["C155"])
    c156 = load(evidence_paths["C156"])

    adjudication = {
        "phase": 1691,
        "campaign": "C157",
        "created_at_utc": now(),
        "accepted": [
            "C153 prospectively predicts paired type-graph half-difference transitions on fresh controlled units.",
            "C154 demonstrates strong donor-conditioned control of the registered answer contrast.",
            "C155 demonstrates a broad fixed-amplitude callable checkpoint window from q24 through q34.",
            "A failed causal identity gate does not invalidate the predictive field; it limits the causal claim.",
        ],
        "corrections": [
            "C153 is local prospective predictive closure, not a complete language-mechanism closure.",
            "C154 uses the observed donor-recipient difference X_q inside its patch; it is not recipient-only inference.",
            "C154 does not isolate observed state, predicted increment, and their sum; C158 must separate them.",
            "C155 uses one unnormalised vector across checkpoints. A stronger q24 effect does not identify a natural formation layer.",
            "C155 measures finite intervention response, not a derivative unless an amplitude limit is established.",
            "C142 event effects do not prove that syntax primarily lives at answer preparation.",
            "Controlled pseudowords do not establish the mechanism of natural apple knowledge.",
            "The cross-model result is only the absence of a common qualified interface among four tested single-token interfaces.",
            "Activation-coordinate resolution is not parameter/weight-level resolution.",
        ],
        "typed_claims": {
            "measured_pass": ["C153 local prediction", "C154 control", "C155 callable window"],
            "measured_fail": ["C154 local checkpoint identity", "C154 population-mean abstraction"],
            "not_tested": [
                "recipient-only counterfactual direction",
                "natural type-graph transfer",
                "unique coordinate transmission circuit",
                "natural formation layer",
                "cross-model HiddenState topology",
            ],
        },
    }

    routes = [
        {
            "phase": 1692,
            "campaign": "C158",
            "name": "increment source decomposition",
            "object": "2X_q, 2Yhat_q, 2(X_q+Yhat_q), and exact 2X_{q+1}",
            "continues_on_failure": True,
            "forbidden_claim": "formation layer or unique mechanism",
        },
        {
            "phase": 1693,
            "campaign": "C159",
            "name": "natural/isomorphic dual graph atlas",
            "object": "full embedding plus all HiddenState checkpoints at aligned roles and representative all-token fields",
            "continues_on_failure": True,
            "naturalness": "hand-curated and machine-audited; independent human blind rating remains missing",
        },
        {
            "phase": 1694,
            "campaign": "C160",
            "name": "recipient-only counterfactual prediction",
            "object": "predict missing relation field from recipient state and registered prompt variables without a test donor",
            "continues_on_failure": True,
        },
        {
            "phase": 1695,
            "campaign": "C161",
            "name": "black-box activation-coordinate transmission",
            "object": "finite symmetric HiddenState perturbations from selected source coordinates to all target coordinates",
            "continues_on_failure": True,
            "forbidden": ["attention", "MLP", "weights", "PCA", "unique circuit language"],
        },
        {
            "phase": 1696,
            "campaign": "C162",
            "name": "broad linguistic-program field",
            "object": "experiencer, embedded agent, attitude, action, patient, coreference, negation scope, voice and surface",
            "continues_on_failure": True,
        },
        {
            "phase": 1697,
            "campaign": "C163",
            "name": "natural graph checkpoint call domain",
            "object": "correct/wrong relation, role and checkpoint interventions plus free-generation and side-effect readouts",
            "dependency": "uses recipient-only direction only if C160 passes; otherwise donor-conditioned direction is explicitly labelled",
            "continues_on_failure": True,
        },
        {
            "phase": 1698,
            "campaign": "C164",
            "name": "three-model free-interface qualification",
            "object": "Qwen3, GLM4 and DeepSeek-7B loaded sequentially on CUDA/offload as registered",
            "continues_on_failure": True,
        },
        {
            "phase": 1699,
            "campaign": "C165",
            "name": "cross-model relative topology",
            "dependency": "at least two qualified models in C164",
            "not_tested_if_missing": True,
        },
        {
            "phase": 1700,
            "campaign": "C166",
            "name": "major-stage synthesis and coordinate heatmap",
            "object": "typed ledger, full coordinate artifacts, visualization and independent audit",
        },
    ]

    protocol = {
        "phase": 1691,
        "campaign": "C157",
        "created_at_utc": now(),
        "status": "local_field_batch_campaign_frozen",
        "research_priority": "observe broad language families, discover repeated structure, then test closure",
        "route_policy": "route-level elimination; no single route failure stops the campaign",
        "evidence_policy": "measured-pass, measured-fail and not-tested are separate ledgers",
        "model_order": ["Qwen3-4B", "GLM4-9B", "DeepSeek-7B"],
        "model_memory_policy": "one model at a time; explicit release and CUDA cache clearing",
        "representation_scope": ["token embedding", "HiddenState checkpoints", "activation coordinates"],
        "excluded_scope": ["attention internals", "MLP internals", "weights", "PCA or other lossy projection"],
        "material_policy": {
            "semantic_uniqueness": "machine checked before model execution",
            "surface_uniqueness": "exact prompt duplicate check",
            "balance": "deterministic factorial quotas",
            "naturalness": "machine lexical/grammar audit; human blind rating reported as missing unless actually supplied",
        },
        "unblinding_policy": "objects and thresholds frozen per route; later routes may use only the registered branch from prior outcomes",
        "routes": routes,
        "source_hashes": {name: sha(path) for name, path in evidence_paths.items()},
        "attachment_hashes": [sha(path) for path in ATTACHMENTS],
        "producer_sha256": sha(Path(__file__)),
        "authorization": "run_C158_through_C166_in_registered_order",
    }

    checks = {
        "c153_gate": bool(c153.get("confirmation_gate_passed")),
        "c154_identity_failed": not bool(c154.get("causal_gate_passed", False)),
        "c155_window": len(c155.get("checkpoint_rows", c155.get("rows", []))) == 11,
        "c156_audit": all(c156.get("checks", {}).values()),
        "route_count": len(routes) == 9,
        "continuous_phases": [route["phase"] for route in routes] == list(range(1692, 1701)),
        "all_sources_present": not missing,
        "no_global_stop": all(route.get("continues_on_failure", True) for route in routes[:-1]),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    save(OUT / "protocol/preregistration.json", protocol)
    save(OUT / "analysis/evidence_adjudication.json", adjudication)
    save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    final = {
        "phase": 1691,
        "campaign": "C157",
        "status": "closed",
        "headline": "C153 is a local predictive closure; C154-C155 show donor-conditioned broad-window callability, not a recipient-only or coordinate-circuit closure.",
        "routes_frozen": len(routes),
        "checks": checks,
        "next_authorization": "C158 increment source decomposition",
    }
    save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
