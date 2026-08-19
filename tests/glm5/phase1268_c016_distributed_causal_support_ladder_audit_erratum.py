"""Post-registration type erratum for the Phase1268 final audit.

The frozen auditor expected explicit empty selection fields on an unqualified
model.  The frozen producer correctly emitted no event object at all.  This
erratum verifies that this missing-vs-empty mismatch is the only audit failure,
then reruns event mathematics on behavior-qualified models only.  It does not
alter the formal failure verdict or any scientific measurement.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1268_c016_distributed_causal_support_ladder as main
import phase1268_c016_distributed_causal_support_ladder_audit as frozen


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main_cli() -> int:
    protocol = read(main.PROTOCOL)
    audit = read(main.FINAL_AUDIT)
    final = read(main.FINAL)
    results = main.read_jsonl(main.MODELS)
    unqualified = [row for row in results if not row["behavior_qualified"]]
    qualified = [row for row in results if row["behavior_qualified"]]
    non_event_checks = {key: value for key, value in audit["checks"].items() if key != "event_and_selection_math"}
    checks = {
        "frozen_auditor_hash_preserved": sha(Path(frozen.__file__).resolve()) == protocol["source_hashes"]["auditor"],
        "original_audit_exactly_12_of_13": audit["passed_checks"] == 12 and audit["total_checks"] == 13 and audit["all_checks_passed"] is False,
        "only_original_failure_is_event_math": all(non_event_checks.values()) and audit["checks"]["event_and_selection_math"] is False,
        "one_unqualified_model": len(unqualified) == 1 and unqualified[0]["model_key"] == "shallow4_r0",
        "unqualified_has_no_event_object": unqualified[0].get("event_ledger") == [] and "selected_events" not in unqualified[0] and "confirmation_targets" not in unqualified[0],
        "qualified_event_math_passes": frozen.event_math(qualified),
        "formal_verdict_remains_failure": final["passed"] is False and final["gates"]["G-BEHAVIOR"] is False,
        "no_authorization": final["authorization"]["distributed_donor_contract_design"] is False and final["authorization"]["automatic_pretrained_run"] is False,
    }
    payload = {
        "status": "post_registration_audit_type_erratum",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "passed": all(checks.values()),
        "scientific_verdict_changed": False,
    }
    target = main.OUT / "audit/independent_final_audit_erratum.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": f"{payload['passed_checks']}/{payload['total_checks']}", "passed": payload["passed"]}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
