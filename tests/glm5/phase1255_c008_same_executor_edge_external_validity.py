#!/usr/bin/env python3
"""Phase1255: same-executor free-Transformer edge external validity.

Phase1254 compared fused SDPA training/evaluation against an explicit QK/OV
intervention executor. Here every training, natural-evaluation and intervention
forward uses the same explicit executor. All scientific thresholds and the
discovery/selection/confirmation procedure remain unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1251_c004_causal_slice_competition as task_module
import phase1254_c007_free_transformer_edge_external_validity as base
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer


PHASE = 1255
CONTRACT_ID = "EXP-C008-WP01-001"
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1255_c008_same_executor_edge_external_validity_audit.py"
PHASE1254_DEPENDENCY = ROOT / "tests/glm5/phase1254_c007_free_transformer_edge_external_validity.py"
MODEL_DEPENDENCY = ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"
TASK_DEPENDENCY = ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"
OUT = ROOT / "tests/glm5/result/phase1255_c008_same_executor_edge_external_validity"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_counterfactuals.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
MODELS = OUT / "raw/model_results.jsonl"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/same_executor_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

ARCHITECTURES = base.ARCHITECTURES
REPLICATES = base.REPLICATES
MODEL_SEEDS = {
    "shallow4_r0": 1_255_401_001,
    "shallow4_r1": 1_255_401_101,
    "middle6_r0": 1_255_601_001,
    "middle6_r1": 1_255_601_101,
    "deep8_r0": 1_255_801_001,
    "deep8_r1": 1_255_801_101,
}
WORLD_SEED = 1_255_900_001
WORLD_COUNTS = base.WORLD_COUNTS
PREFIX_SIZES = base.PREFIX_SIZES
THRESHOLDS = base.THRESHOLDS


class SameExecutorTinyCausalTransformer(TinyCausalTransformer):
    """Use the intervention executor as the model's native forward path."""

    def forward(
        self,
        input_ids: torch.Tensor,
        return_states: bool = False,
    ) -> torch.Tensor:
        if return_states:
            raise RuntimeError("same-executor contract does not expose residual-state shortcuts")
        logits, _ = base.explicit_forward(self, input_ids, capture=False)
        return logits


# The imported training function resolves this symbol from its defining module.
task_module.TinyCausalTransformer = SameExecutorTinyCausalTransformer

_original_protocol_payload = base.protocol_payload
_original_make_worlds = base.make_worlds


def make_worlds(seed: int = WORLD_SEED, counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    return _original_make_worlds(seed=seed, counts=counts or WORLD_COUNTS)


def protocol_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = _original_protocol_payload(rows)
    timeless = {key: value for key, value in original.items() if key not in {"created_at_utc", "protocol_digest"}}
    timeless.update({
        "phase": PHASE,
        "schema_version": "phase1255.c008.same_executor_edge.protocol.v1",
        "contract_id": CONTRACT_ID,
        "claim_type": "same_executor_free_transformer_component_edge_external_validity",
        "question": "After eliminating fused-versus-explicit execution mismatch, does the frozen typed-edge camera identify a sparse, identity-specific, bidirectionally causal sufficient coalition across free Transformers?",
        "model_seeds": MODEL_SEEDS,
        "execution_invariant": {
            "training": "explicit QK softmax, per-head OV payload and MLP residual writes",
            "natural_evaluation": "the identical explicit program",
            "intervention_evaluation": "the identical explicit program with registered tensor replacement",
            "native_explicit_logit_gap": "algebraically zero up to repeated-call determinism",
        },
        "source_hashes": {
            "main": base.file_sha256(SCRIPT),
            "auditor": base.file_sha256(AUDITOR),
            "phase1254_dependency": base.file_sha256(PHASE1254_DEPENDENCY),
            "model_dependency": base.file_sha256(MODEL_DEPENDENCY),
            "task_dependency": base.file_sha256(TASK_DEPENDENCY),
        },
        "hard_stops": [
            "This phase repairs an executor-identity confound; no Phase1254 threshold is relaxed.",
            "All training, natural and intervention passes use one explicit arithmetic program.",
            "Free-network task truth and endpoint truth still do not provide internal mechanism truth.",
            "Confirmation cannot select components, prefix size, thresholds or models.",
            "Behavior-unqualified seeds remain in the denominator and are not replaced.",
            "Correct rescue, wrong identity, matched null and reverse blocking remain conjunctive.",
            "A pass establishes a sparse sufficient coalition candidate, not a unique minimal algorithm or semantic circuit.",
            "Failure blocks pretrained-model escalation; pass authorizes only one separately frozen Qwen3 contract.",
        ],
    })
    return {**timeless, "created_at_utc": base.utc_now(), "protocol_digest": base.digest(timeless)}


def configure_base() -> None:
    assignments = {
        "PHASE": PHASE,
        "CONTRACT_ID": CONTRACT_ID,
        "SCRIPT": SCRIPT,
        "AUDITOR": AUDITOR,
        "OUT": OUT,
        "PROTOCOL": PROTOCOL,
        "MATERIAL": MATERIAL,
        "ENVIRONMENT": ENVIRONMENT,
        "PREAUDIT": PREAUDIT,
        "RAW": RAW,
        "MODELS": MODELS,
        "COMPLETE": COMPLETE,
        "ANALYSIS": ANALYSIS,
        "FINAL": FINAL,
        "FINAL_AUDIT": FINAL_AUDIT,
        "MODEL_SEEDS": MODEL_SEEDS,
        "WORLD_SEED": WORLD_SEED,
        "make_worlds": make_worlds,
        "protocol_payload": protocol_payload,
    }
    for name, value in assignments.items():
        setattr(base, name, value)


configure_base()

# Re-export audited primitives for the independent auditor.
canonical_json = base.canonical_json
digest = base.digest
file_sha256 = base.file_sha256
atomic_json = base.atomic_json
read_json = base.read_json
read_jsonl = base.read_jsonl
component_ids = base.component_ids
summarize = base.summarize
verify_protocol = base.verify_protocol


def environment_snapshot() -> dict[str, Any]:
    value = base.environment_snapshot()
    value["precision"] = "fp32_same_explicit_executor_for_training_natural_and_intervention"
    value["executor"] = "manual causal QK softmax, per-head OV and MLP residual write"
    return value


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError("protocol already exists")
    rows = make_worlds()
    base.write_jsonl(MATERIAL, rows)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(rows))
    print(canonical_json({"status": "preregistered", "rows": len(rows)}))


def probe() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(layers=3, width=64, heads=4, mlp_width=128, max_length=23, vocab_size=22)
    rows = make_worlds(seed=1_255_300_001, counts={"discovery": 16, "selection": 16, "confirmation": 32})
    base.set_seed(1_255_301_001)
    result = base.run_model("development3", 0, config, 1_255_301_001, rows, device)
    atomic_json(ROOT / "tests/glm5_temp/phase1255_same_executor_probe.json", result)
    print(canonical_json({
        "behavior_qualified": result["behavior_qualified"],
        "native_explicit_logit_gap": result["native_explicit_logit_gap"],
        "selected_size": result.get("selected_size"),
        "passed": result["passed"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "preregister", "run", "analyze"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "probe":
        probe()
    elif args.command == "preregister":
        preregister(args.force)
    elif args.command == "run":
        base.run(args.device)
    else:
        base.analyze()


if __name__ == "__main__":
    main()
