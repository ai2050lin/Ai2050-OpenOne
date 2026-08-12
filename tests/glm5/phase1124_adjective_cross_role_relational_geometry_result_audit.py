from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PHASE = 1124
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1124_adjective_cross_role_relational_geometry"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    protocol_body = dict(prereg)
    protocol_digest = protocol_body.pop("protocol_digest")
    final = read_json(OUT_ROOT / "analysis" / "final_summary.json")
    final_body = dict(final)
    final_digest = final_body.pop("final_digest")

    expected_models = set(prereg["models"])
    layer_files = {
        model: OUT_ROOT / "analysis" / f"layer_metrics.{model}.json" for model in prereg["models"]
    }
    layer_details = {model: read_json(path) for model, path in layer_files.items() if path.is_file()}
    panel_names = {"discovery", "independent_confirmation", "heldout"}
    confirmation_names = {"independent_confirmation", "heldout"}
    expected_cells = 4

    checks = {
        "protocol_digest_valid": canonical_digest(protocol_body) == protocol_digest,
        "final_digest_valid": canonical_digest(final_body) == final_digest,
        "source_digest_preserved": final["source_phase1123_final_digest"]
        == prereg["source_phase1123_final_digest"],
        "all_models_present": set(final["model_results"]) == expected_models,
        "all_layer_files_present": set(layer_details) == expected_models,
        "selected_layers_were_eligible": all(
            final["model_results"][model]["selected_layer"] in prereg["eligible_hidden_state_indices"][model]
            for model in expected_models
        ),
        "all_panels_present": all(
            set(final["model_results"][model]["panels"]) == panel_names for model in expected_models
        ),
        "all_four_cells_present": all(
            len(panel["selected"]["cells"]) == expected_cells
            for model in final["model_results"].values()
            for panel in model["panels"].values()
        ),
        "confirmation_exact_nulls_present": all(
            cell["exact_permutation_percentile"] is not None
            for model in final["model_results"].values()
            for panel_name, panel in model["panels"].items()
            if panel_name in confirmation_names
            for cell in panel["selected"]["cells"].values()
        ),
        "qualification_consistent": set(final["qualified_models"])
        == {model for model, result in final["model_results"].items() if result["qualified"]},
        "component_work_denied": final["component_or_causal_work_authorized"] is False,
        "auto_continue_is_frozen_stop": final["auto_continue"]["value"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1124 result audit failed: {checks}")

    audit = {
        "schema_version": "phase1124_adjective_cross_role_relational_geometry_result_audit.v1",
        "phase": PHASE,
        "protocol_digest": protocol_digest,
        "final_digest": final_digest,
        "checks": checks,
        "passed": True,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
    }
    audit["audit_digest"] = canonical_digest(audit)
    write_json(OUT_ROOT / "audit" / "result_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
