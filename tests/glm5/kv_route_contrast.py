"""Audit the preregistered one-head and three-head 2x2 read interventions."""

from collections import Counter
from pathlib import Path
import numpy as np

import kv_route_revised as Revised
import kv_route_materials as Original
import kv_route_run as R
from kv_route_audit import margin, test


def source_time(suite, stage, rows, known):
    by = {r["id"]: r for r in rows}
    job = [R.read(s["name"]) for s in stage["shards"]]
    assert all(R.sha(R.OUT / s["name"]) == s["sha"] for s in stage["shards"])
    assert [key for b in job for key in b["ids"]] == list(by)

    def margins(arm):
        return np.array([margin(scores, by[key]) for b in job
                         for key, scores in zip(b["ids"], b["paths"][arm]["scores"], strict=True)])

    groups = {}
    for single, suffix in ((True, "0"), (False, "012")):
        field = {("state", "answer"): margins(f"swap-{suffix}") - margins(f"null-{suffix}")}
        for source, clock in (("state", "early"), ("note", "answer"), ("note", "early")):
            pref = "single-" if single else ""
            field[(source, clock)] = (margins(f"{pref}swap-{source}-{clock}")
                                      - margins(f"{pref}null-{source}-{clock}"))
        a, b, c, d = [field[k] for k in (("state", "answer"), ("state", "early"), ("note", "answer"), ("note", "early"))]
        outcome = {k:test(rows, contrast, floor) for k, contrast, floor in (
            ("source", a-c, .25), ("clock", a-b, .25), ("content", a, .50))}
        if not single:
            assert all(outcome[k] == known["measures"][k] for k in outcome)
        groups["L21H19" if single else "trio"] = dict(
            tests=outcome, raw_means={f"{k[0]}-{k[1]}":float(v.mean()) for k,v in field.items()},
            interaction=float((a-b-c+d).mean()))
    background = margins("natural")
    effect = margins("remaining") - background
    free_crossing = {}
    for arm in ("natural", "swap-012", "remaining", "whole", "whole_null"):
        counts = Counter()
        for b in job:
            if arm not in b["free"]:
                continue
            for key, scores, free in zip(b["ids"],b["paths"][arm]["scores"],b["free"][arm],strict=True):
                row = by[key]
                flip = free["eos"] and free["code"]==row["alt_code"] and free["state"]==row["state"] and free["notes"]==row["notes"]
                value = margin(scores,row)
                if flip != (value>0):
                    counts["tie_flip" if value==0 and flip else "other_disagreement"] += 1
                counts["n"] += 1
        free_crossing[arm] = {x:counts[x] for x in ("n","tie_flip","other_disagreement")}
    return dict(suite=suite, groups=groups, alternative_crossings={
        a:dict(n=len(rows), positive=int((margins(a)>0).sum())) for a in
        ("natural", "swap-0", "swap-012", "remaining", "whole", "whole_null")},
        broad_complement_paired_gain=dict(mean=float(effect.mean()),
                                         positive=int((effect>0).sum()), negative=int((effect<0).sum())),
        free_vs_strict_crossing=free_crossing,
        offsets={k:dict(Counter(x for b in job for x in b["clock"][k])) for k in ("early", "answer")})


def configure(version="v2"):
    R.M = Revised if version == "v2" else Original
    R.OUT = R.Base.ROOT / "tests/glm5/result" / f"kv_route_20261115_{version}"
    R.TEMP = R.Base.ROOT / "tests/glm5_temp" / f"kv_route_20261115_{version}"
    R.RUN = f"KV-ROUTE-20261115-{version}"
    R.setup()


def main():
    configure()
    plan = R.check()
    shas = plan["pilot_artifacts"]
    assert all(R.sha(R.OUT / name) == value for name, value in shas.items())
    audit = R.read("audit.json")
    lookup = {x["suite"]: x for x in audit["holdout"]}
    supplement = [source_time(s["suite"], s, R.read(f"materials-{s['suite']}.json"), lookup[s["suite"]])
                  for s in R.read("confirmation.json")["ledger"]
                  if s["suite"] != "discovery" and s["status"] == "complete"]
    R.write("matched-contrast.json", dict(suites=supplement, plan_sha=R.sha(R.OUT/"plan.json"),
                                         audit_sha=R.sha(R.OUT/"audit.json"), source_sha=R.sha(Path(__file__))))


if __name__ == "__main__":
    main()
