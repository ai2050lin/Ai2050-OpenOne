from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PHASE = 1125
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1125_pythia_controlled_bridge_calibration"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def valid_digest(value: dict[str, Any], key: str) -> bool:
    body = dict(value)
    expected = body.pop(key)
    return canonical_digest(body) == expected


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol" / "audit.json")
    base = read_json(OUT_ROOT / "evaluation" / "base.json")
    behavior = read_json(OUT_ROOT / "training" / "behavior_only" / "summary.json")
    forced = read_json(OUT_ROOT / "training" / "bridge_forced" / "summary.json")
    run = read_json(OUT_ROOT / "training" / "run_summary.json")
    final = read_json(OUT_ROOT / "analysis" / "final_summary.json")
    checks = {
        "protocol_digest_valid": valid_digest(prereg, "protocol_digest"),
        "protocol_audit_passed": all(protocol_audit["checks"].values()),
        "base_linked": base["protocol_digest"] == prereg["protocol_digest"],
        "behavior_digest_valid": valid_digest(behavior, "result_digest"),
        "forced_digest_valid": valid_digest(forced, "result_digest"),
        "run_digest_valid": valid_digest(run, "run_digest"),
        "final_digest_valid": valid_digest(final, "final_digest"),
        "all_protocol_links_match": all(
            value["protocol_digest"] == prereg["protocol_digest"]
            for value in (base, behavior, forced, run, final)
        ),
        "both_adapters_exist": all((ROOT / result["adapter_path"]).is_file() for result in (behavior, forced)),
        "both_adapter_hashes_match": all(
            hashlib.sha256((ROOT / result["adapter_path"]).read_bytes()).hexdigest() == result["adapter_sha256"]
            for result in (behavior, forced)
        ),
        "all_partitions_present": all(
            set(condition) == {"train", "calibration", "transfer"}
            for condition in final["condition_summaries"].values()
        ),
        "instrument_calibration_prediction_consistent": final["instrument_calibration_passed"]
        == (
            final["predictions"]["P1_engineering_and_frozen_base"] == "pass"
            and final["predictions"]["P3_forced_bridge_instrument_visibility"] == "pass"
        ),
        "natural_mechanism_denied": final["natural_mechanism_authorized"] is False,
        "component_work_denied": final["component_or_causal_work_authorized"] is False,
        "auto_continue_frozen": final["auto_continue"]["value"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1125 result audit failed: {checks}")
    audit = {
        "schema_version": "phase1125_pythia_controlled_bridge_result_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
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
