"""Reconcile canonical JSON cell labels with in-memory tuple labels."""

from copy import deepcopy
from pathlib import Path

from kv_route_contrast import configure, source_time
import kv_route_run as R


def finish(version):
    configure(version)
    plan = R.check()
    assert all(R.sha(R.OUT/name)==v for name,v in plan["pilot_artifacts"].items())
    audit = R.read("audit.json")
    stages = {s["suite"]:s for s in R.read("confirmation.json")["ledger"]}
    suites = []
    for known in audit["holdout"]:
        sid = known["suite"]
        if known["status"]!="complete":
            continue
        known = deepcopy(known)
        for statistic in known["measures"].values():
            for cell in statistic["cells"]:
                cell["cell"] = tuple(cell["cell"])
        suites.append(source_time(sid,stages[sid],R.read(f"materials-{sid}.json"),known))
    R.write("matched-contrast.json",dict(suites=suites,plan_sha=R.sha(R.OUT/"plan.json"),
                                         audit_sha=R.sha(R.OUT/"audit.json"),source_sha=R.sha(Path(__file__))))


if __name__=="__main__":
    import sys
    for version in sys.argv[1:]:
        finish(version)
