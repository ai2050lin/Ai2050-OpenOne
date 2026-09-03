#!/usr/bin/env python3
"""Independent artifact audit for C591-C598."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
MAIN = ROOT / "tests/glm5/phase2125_c591_c598_fresh_scope_lockbox_campaign.py"
WORKER = ROOT / "tests/glm5/phase2131_c597_qwen14_fresh_scope_worker.py"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c598_fresh_scope_lockbox_atlas.json"
OUT = RESULT / "phase2133_c599_fresh_scope_lockbox_independent_audit"
PHASES = {
    c: (2125 + c - 591, slug)
    for c, slug in (
        (591, "recovery_amendment_and_fresh_lockbox_contract"),
        (592, "fresh_lexical_construction_and_query_switch_material"),
        (593, "fresh_qwen_behavior_qualification"),
        (594, "fresh_qwen_all_token_all_coordinate_capture"),
        (595, "frozen_passport_composition_and_dynamics_transfer"),
        (596, "output_changing_query_switch_causal_specificity"),
        (597, "qwen14_scale_response_topology"),
        (598, "parameter_atlas_cleanup_and_campaign_synthesis"),
    )
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_out(c: int) -> Path:
    phase, slug = PHASES[c]
    return RESULT / f"phase{phase}_c{c}_{slug}"


def main() -> None:
    checks = {}
    finals = {}
    main_hash = sha(MAIN)
    for c in PHASES:
        base = phase_out(c)
        final_path = base / "analysis/final.json"
        protocol_path = base / "protocol/preregistration.json"
        pre_path = base / "audit/internal_checks.json"
        post_path = base / "audit/internal_checks_post.json"
        checks[f"c{c}_final"] = final_path.exists()
        checks[f"c{c}_protocol"] = protocol_path.exists()
        checks[f"c{c}_pre"] = pre_path.exists()
        checks[f"c{c}_post"] = post_path.exists()
        if final_path.exists():
            value = load(final_path)
            finals[c] = value
            checks[f"c{c}_closed"] = value.get("status") == "closed" and value.get("all_checks_passed") is True
            checks[f"c{c}_phase"] = value.get("phase") == PHASES[c][0]
        if protocol_path.exists():
            checks[f"c{c}_producer_hash"] = load(protocol_path).get("producer_sha256") == main_hash

    checks["phase_continuity"] = [PHASES[c][0] for c in PHASES] == list(range(2125, 2133))
    checks["worker_exists"] = WORKER.exists()
    behavior_rows = [json.loads(line) for line in (phase_out(593) / "behavior/qwen3_behavior.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    behavior = finals[593]["headline"]
    checks["behavior_rows"] = len(behavior_rows) == behavior["rows"] == 2664
    checks["behavior_accuracy"] = abs(sum(row["correct"] for row in behavior_rows) / len(behavior_rows) - behavior["behavior_accuracy"]) < 1e-12
    checks["all_behavior_slices_qualified"] = behavior["qualified_slices"] == behavior["total_slices"] == 45

    transfer = finals[595]["headline"]
    checks["atomic_gate_arithmetic"] = all(value["passed"] == sum(v for key, v in transfer["atomic_gates"].items() if key.startswith(family + "|")) for family, value in transfer["family_summary"].items())
    checks["dynamic_gate_arithmetic"] = transfer["dynamic_summary"]["passed"] == sum(value["gate"] for value in transfer["dynamic_metrics"].values())
    checks["composition_gate_arithmetic"] = transfer["composition_summary"]["passed"] == sum(value["gate"] for value in transfer["composition_metrics"].values())
    causal = finals[596]["headline"]
    for direction in ("false_to_true", "true_to_false"):
        summary = causal["direction_summary"][direction]
        checks[f"causal_{direction}_arithmetic"] = summary["passed"] == sum(v for key, v in causal["gates"].items() if key.endswith(direction)) and summary["total"] == sum(key.endswith(direction) for key in causal["gates"])

    q14 = finals[597]["headline"]["qwen14"]
    checks["qwen14_behavior"] = q14.get("status") == "closed" and q14.get("behavior_accuracy") == 1.0 and q14.get("rows") == 96
    checks["qwen14_full_coordinates"] = q14.get("checkpoints") == 42 and q14.get("coordinates") == 5120
    checks["qwen14_raw_cleanup_recorded"] = q14.get("raw_path") and not (ROOT / q14["raw_path"]).exists()

    checks["visual_exists"] = VISUAL.exists() and VISUAL.stat().st_size > 0
    atlas = load(VISUAL)
    checks["visual_schema"] = atlas.get("schema") == "ai2050.fresh_scope_lockbox_atlas.v1"
    shape = atlas["qwen3_4b"]["representative"]["shape"]
    checks["visual_exact_shape"] = shape[0] == 38 and shape[2] == 2560 and shape[1] == len(atlas["qwen3_4b"]["representative"]["tokens"])
    checks["visual_embedding_and_hidden"] = atlas["qwen3_4b"]["checkpoints"][0] == "embedding" and atlas["qwen3_4b"]["checkpoints"][-1] == "final_norm"
    checks["role_mean_retained"] = (phase_out(594) / "raw/qwen3_role_mean_states.float16.npy").exists()
    checks["role_last_retained"] = (phase_out(594) / "raw/qwen3_role_last_states.float16.npy").exists()
    checks["fresh_bulk_cleaned"] = not (phase_out(594) / "raw/qwen3_full_token_shards").exists()

    route = {
        "same_exact_goal": False,
        "completed_object": "fresh-word and fresh-wrapper lockbox of frozen scope-response passports, composition/dynamics transfer, output-changing query guidance and Qwen3-14B scale panel",
        "surviving_atomic_families": transfer["fresh_atomic_candidates"],
        "dynamic_passed": transfer["dynamic_summary"],
        "composition_passed": transfer["composition_summary"],
        "output_changing_guidance": causal["output_changing_sufficiency_candidate"],
        "directional_boundary": causal["direction_summary"],
        "qwen14_model_internal_candidate": q14.get("functional_candidate", False),
        "next_object": "separate readout-boundary and directionality campaign with natural open-vocabulary answers, plus broader language-family constructions",
        "why_not_automatic_same_goal": "The frozen fresh-lockbox objective is complete. The surviving uncertainty is a different object: asymmetric output readout and natural-language breadth, not another repeat of the same binary scope passport.",
        "foundational_math_authorized": False,
        "strict_boundary": "Strong response transfer is compatible with repeatable prompt-transformation physics. It is not yet a semantic algebra, a necessary circuit, or evidence that existing mathematics is insufficient.",
    }
    all_passed = all(checks.values())
    value = {"phase": 2133, "campaign": "C599", "status": "closed", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": all_passed, "headline": {"status": "fresh_scope_lockbox_independent_audit_closed", "checks_passed": sum(checks.values()), "checks_total": len(checks), "route": route}, "next_authorization": "new_readout_directionality_campaign_freeze"}
    save(OUT / "protocol/preregistration.json", {"phase": 2133, "campaign": "C599", "object": "independent artifact recomputation and next-object adjudication", "main_sha256": main_hash, "worker_sha256": sha(WORKER)})
    save(OUT / "audit/checks.json", checks)
    save(OUT / "analysis/route_adjudication.json", route)
    save(OUT / "analysis/final.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
