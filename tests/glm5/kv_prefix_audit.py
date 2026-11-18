"""Independent CPU recomputation of the Phase 2949 frozen endpoints and examples."""

from collections import Counter

import numpy as np

import kv_prefix_run as C
from kv_early_audit import native_role_pairs
from kv_origin_audit import summarize, fam, leave_lexeme
from kv_prefix_study import code_margin, state_margin, configurations, read_np, physical, sign


A, B = C.A, C.B
SWITCH = ("state-switch", "state-route-switch")


def natural_score(roll):
    out = {}
    for arm, rows in roll.items():
        observations = rows["observations"]
        assert len(observations) == rows["tested"]
        verified = [r for r in observations.values() if r["verified"]]
        pq = [r for r in verified if r["code"] in (" P", " Q")]
        out[arm] = dict(total=rows["total"], tested=rows["tested"], reached_code=len(verified), actual_pq=len(pq),
                        kept_original=sum(r["code"].strip() == r["expected"] for r in pq),
                        reversed_code=sum(r["code"].strip() != r["expected"] for r in pq),
                        recognized_tested=sum(r["recognized"] for r in observations.values()),
                        recognized_pq=sum(r["recognized"] for r in pq),
                        recognized_and_reversed=sum(r["recognized"] and r["code"].strip() != r["expected"] for r in pq),
                        stops=dict(Counter(r["stop"] for r in observations.values())),
                        shared_with=rows.get("shared_with"))
    return out


def contrast_audit(rows, z, config, p):
    names = list(config)
    cm = code_margin(z, rows)
    sm = state_margin(z)
    state_ids = np.array([r["state_ids"] for r in rows])
    state = z["state_top"]
    switches = (state != state_ids[:, :1]).astype(np.float64)
    localscore = z["local_total"] * sign(rows)[:, None]

    def key(x, name):
        return x[:, names.index(name)].astype(np.float64)

    def random(x, prefix):
        return (key(x, prefix + "1") + key(x, prefix + "2")) / 2

    q = lambda name: key(cm, name)
    e = lambda name: key(sm, name)
    base = q("sham")
    sets = {
        "early-state": random(sm, "early-rand") - e("early"),
        "state-route": random(sm, "state-rand") - e("state-route"),
        "early-code": random(cm, "early-rand") - q("early"),
        "early-local": key(localscore, "early") - random(localscore, "early-rand"),
        "kv-block": random(cm, "kv-rand-block") - q("kv-block"),
        "kv-rescue": q("kv-rescue") - q("early"),
        "kv-rescue-specific": q("kv-rescue") - random(cm, "kv-rand-rescue"),
        "native-full-rescue": q("native-full-rescue") - q("early"),
        "reverse": base - q("reverse"),
        "state-switch": key(switches, "early") - random(switches, "early-rand"),
        "state-route-switch": key(switches, "state-route") - random(switches, "state-rand"),
    }
    assert list(sets) == list(p["gate"]["labels"])
    primary = {}
    for k, values in sets.items():
        family = fam(values)
        record = summarize(family, p["gate"]["confidence"])
        threshold = p["gate"]["minimum_switch"] if k in SWITCH else p["gate"]["minimum_logit"]
        primary[k] = dict(**record, minimum=threshold, pass_gate=bool(record["n"] >= p["gate"]["min_family"]
                              and record["ci"][0] > 0 and record["mean"] >= threshold),
                          leave_one_lexeme=leave_lexeme(family, rows))
    ep = {"early-state": e("sham") - e("early"), "state-route": e("sham") - e("state-route"),
          "early-code": base - q("early"), "state-route-code": base - q("state-route"),
          "early-local": key(localscore, "early"), "kv-block": base - q("kv-block"),
          "native-v-block": base - q("native-v-block"), "kv-rescue-k": q("kv-rescue-k") - q("early"),
          "kv-rescue-v": q("kv-rescue-v") - q("early"), "kv-rescue": sets["kv-rescue"],
          "native-full-rescue": sets["native-full-rescue"], "reverse": sets["reverse"]}
    absolute = {k: summarize(fam(x)) for k, x in ep.items()}
    gold = np.array([r["token"]["letters"][r["code"] == "Q"] for r in rows])
    alternative = np.array([r["token"]["letters"][r["code"] == "P"] for r in rows])
    all_arms = {}
    for j, k in enumerate(names):
        valid = z["first_mismatch"][:, j] == -1
        exact = state[:, j] == state_ids[:, 1]
        other = (state[:, j] != state_ids[:, 0]) & ~exact
        top = z["code_top"][:, j]
        all_arms[k] = dict(state_switch=int(exact.sum() + other.sum()), recognized=int(exact.sum()), other=int(other.sum()),
                           prefix_valid=int(valid.sum()), same_state_then_diverged=int(((state[:, j] == state_ids[:, 0]) & ~valid).sum()),
                           conditional_code_reversed=int((top == alternative).sum()), conditional_code_other=int(((top != alternative) & (top != gold)).sum()),
                           native_code_reversed=int((valid & (top == alternative)).sum()),
                           native_code_other=int((valid & (top != alternative) & (top != gold)).sum()),
                           margin_wrong=int((cm[:, j] <= 0).sum()))
    both_valid = np.logical_and.reduce([z["first_mismatch"][:, names.index(k)] == -1
        for k in ("sham", "early", "early-rand1", "early-rand2", "kv-rescue", "native-full-rescue")])
    same = np.repeat(both_valid.reshape(-1, 8).all(1), 8)
    sensitivity = {k: summarize(fam(sets[k][same])) if same.any() else None
                   for k in ("early-code", "kv-rescue", "native-full-rescue", "kv-rescue-specific")}
    strata = {}
    for side in (0, 1):
        for context in (0, 1):
            take = np.array([r["side"] == side and r["context"] == context for r in rows])
            strata[f"role{side}-rule{context}"] = {k: summarize(x[take].reshape(-1, 4).mean(1)) if take.any() else None for k, x in sets.items()}
    d = z["head_output_directed"] * sign(rows)[:, None]
    comparisons = {}
    for a, b in (("kv-rescue", "early"), ("native-full-rescue", "early"), ("kv-block", "sham"), ("reverse", "sham")):
        estimate = key(d, a) - key(d, b)
        actual = q(a) - q(b)
        nz = (estimate != 0) & (actual != 0)
        both = float(np.corrcoef(fam(estimate), fam(actual))[0, 1]) if fam(estimate).std() and fam(actual).std() else None
        comparisons[a] = dict(tangent_mean=float(estimate.mean()), measured_mean=float(actual.mean()), pearson_family=both,
                              nonzero_both=int(nz.sum()), sign_agreement=float((np.sign(estimate[nz]) == np.sign(actual[nz])).mean()) if nz.any() else None)
    physical_stats = {}
    i = names.index("early")
    for name in ("early", "early-rand1", "early-rand2", "state-route", "kv-block", "kv-rescue", "native-full-rescue", "reverse"):
        j = names.index(name)
        source = i if name in ("kv-rescue", "native-full-rescue") else 0
        physical_stats[name] = dict(
            k_relative_median=float(np.median(np.linalg.norm(z["k_applied"][:, j] - z["k_native"][:, source], axis=-1)
                                               / np.maximum(np.linalg.norm(z["k_native"][:, source], axis=-1), 1e-4))),
            v_relative_median=float(np.median(np.linalg.norm(z["v_applied"][:, j] - z["v_native"][:, source], axis=-1)
                                               / np.maximum(np.linalg.norm(z["v_native"][:, source], axis=-1), 1e-4))),
            alpha_median=float(np.median(z["a_applied"][:, j])),
            head_l2_from_sham=float(np.median(np.linalg.norm(z["head"][:, j] - z["head"][:, 0], axis=-1))))

    def weaker(scores, mask):
        return sorted(np.flatnonzero(mask), key=lambda i: scores[i])[:6]

    examples = {
        "early_state_changed": list(np.flatnonzero(state[:, i] != state_ids[:, 0])[:8]),
        "state_route_changed": list(np.flatnonzero(state[:, names.index("state-route")] != state_ids[:, 0])[:8]),
        "directed_local_without_code_effect": weaker(-key(localscore, "early"), abs(ep["early-code"]) <= .125),
        "native_coordinate_rescue_harms": weaker(sets["kv-rescue"], sets["kv-rescue"] < 0),
        "selected_coordinate_loses_to_random": weaker(sets["kv-rescue-specific"], sets["kv-rescue-specific"] < 0),
        "reverse_improves_original_margin": weaker(sets["reverse"], sets["reverse"] < 0),
        "native_code_changes_under_block": list(np.flatnonzero((z["first_mismatch"][:, names.index("kv-block")] == -1)
                                                                    & (z["code_top"][:, names.index("kv-block")] != gold))[:8]),
    }
    counters = {}
    for kind, indices in examples.items():
        counters[kind] = [dict(row=rows[i], state_top=state[i].tolist(), state=z["state"][i].tolist(), pq=z["pq"][i].tolist(),
                               code_top=z["code_top"][i].tolist(), tangent=d[i].tolist(), alpha=z["a_applied"][i].tolist(),
                               first_mismatch=z["first_mismatch"][i].tolist(), first_emitted=z["first_emitted"][i].tolist()) for i in indices]
    bridge = all(primary[k]["pass_gate"] for k in ("early-code", "early-local", "kv-block", "kv-rescue",
                                               "kv-rescue-specific", "native-full-rescue", "reverse"))
    return dict(n=len(rows)//8, primary=primary, absolute=absolute, arms=all_arms, roles=strata, same_word=sensitivity,
                same_word_family=int(same.sum()//8), tangent_calibration=comparisons, physical=physical_stats,
                examples=counters, bridge=bool(bridge))


def audit(p):
    f = A.load("freeze.json")
    ledger = A.load("confirm.json")
    b = A.load("confirm-behavior.json")
    rows = A.load(p["materials"]["confirm"]["file"])
    assert ledger["freeze"] == b["binding"] == B.sha("freeze.json") and f["plan"] == B.sha("plan.json")
    assert all(B.sha(k) == v for k, v in f["discovery"].items())
    assert (C.OUT / "freeze.json").stat().st_mtime_ns < (C.OUT / "confirm-behavior.json").stat().st_mtime_ns
    assert ledger["behavior"] == B.sha("confirm-behavior.json")
    assert C.N.behavior(rows, b["rows"])["cells"] == b["cells"]
    all_rows = C.O.resolved(rows, dict(rows=b["rows"], eligible=list({r["family"] for r in rows})))
    natural = native_role_pairs(all_rows, b["rows"])
    failures = [dict(row=r, observation=o) for r, o in zip(rows, b["rows"], strict=True) if not o["code_correct"]]
    output = dict(state=ledger["state"], natural=natural, failures=failures, behavior_cells=b["cells"],
                  provenance={k: B.sha(k) for k in ("plan.json", "freeze.json", "confirm.json", "confirm-behavior.json")})
    if ledger["state"] != "finished":
        A.save("audit.json", output)
        return
    assert ledger["rollout"] == B.sha("rollout.json")
    assert all(B.sha(k) == v for k, v in ledger["files"].items())
    cfg = configurations(f)
    use = C.O.resolved(rows, b)
    lookup = {r["id"]: r for r in use}
    fit, failed = [], []
    for name in sorted(ledger["checks"]):
        z = read_np(name)
        recorded = physical(z, cfg)
        assert all(ledger["checks"][name][k] == v for k, v in recorded.items())
        for k, ok in enumerate(z["technical"]):
            if not ok:
                failed.append(dict(id=z["ids"][k], conditional_code=int(z["code_top"][k, 0]), pq=z["pq"][k, 0].tolist(),
                                   top_state=int(z["state_top"][k, 0]), first_mismatch=int(z["first_mismatch"][k, 0]),
                                   first_emitted=int(z["first_emitted"][k, 0]), raw_pq_drift=float(z["numeric_diff"][k])))
        selected = np.repeat(z["technical"].reshape(-1, 8).all(1), 8)
        fit.append({k: v[selected] for k, v in z.items() if k not in ("coord", "dose", "head_model", "local_closure")})
    merged = {k: np.concatenate([x[k] for x in fit]) for k in fit[0]}
    assert merged["ids"].tolist() == ledger["accepted"]
    output.update(eligible=ledger["eligible"], technical_failures=failed, technical=ledger["checks"], arms_order=list(cfg),
                  rollout=natural_score(A.load("rollout.json")))
    if len(merged["ids"]):
        final = [lookup[k] for k in merged["ids"]]
        output.update(contrast_audit(final, merged, cfg, p))
    else:
        output["state"] = "technical-gate-failed"
    A.save("audit.json", output)
    print("Audit accepted", output.get("n"), "bridge", output.get("bridge"),
          {k: (v["mean"], v["pass_gate"]) for k, v in output.get("primary", {}).items()}, flush=True)
