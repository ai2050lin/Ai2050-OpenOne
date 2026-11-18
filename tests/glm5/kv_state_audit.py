"""Phase 2950 frozen readouts, paired natural outcomes, and counterexample reduction."""

from collections import Counter

import numpy as np

import kv_state_run as C
from kv_early_audit import native_role_pairs
from kv_origin_audit import fam, leave_lexeme, summarize
from kv_state_study import configs, load, physical, code_margin, state_margin, sign


A, B = C.A, C.B
RATES = {"e-switch", "e-switch-pq"}


def rollout_counts(data, rows, z, cfg, selected):
    names = list(cfg)
    ix = {r["id"]: j for j, r in enumerate(rows)}
    score, totals, examples = {}, {}, {}
    for arm, run in data.items():
        top = z["state_top"][:, names.index(arm)]
        opposite = [r["state_ids"][1] for r in rows]
        switched = top != np.array([r["state_ids"][0] for r in rows])
        assert run["total"] == switched.sum() and run["tested"] == len(run["observations"]) == len(run["order"])
        assert set(run["order"]) == set(run["observations"])
        assert all(not switched[ix[k]] or k in run["observations"] for k in selected)
        byid = run["observations"]
        for key, s in byid.items():
            i = ix[key]
            assert switched[i] and top[i] == s["target"]
            assert s["recognized"] == (s["target"] == opposite[i])
            assert s["expected"] == rows[i]["code"]
            if s["stop"] == "code":
                assert s["emitted"][0] == s["target"] and s["emitted"][-1] == s["code_token"]
        score[arm] = np.asarray([match_alt(byid.get(k)) for k in selected], dtype=np.float64)
        pq = [s for s in byid.values() if real_pq(s)]
        totals[arm] = dict(switched=int(run["total"]), tested=run["tested"], stops=dict(Counter(s["stop"] for s in byid.values())),
                           reaches_code=sum(s["stop"] == "code" for s in byid.values()), real_pq=len(pq),
                           recognized_tested=sum(s["recognized"] for s in byid.values()), recognized_pq=sum(s["recognized"] for s in pq),
                           original_pq=sum(s["code"].strip() == s["expected"] for s in pq),
                           recognized_original_pq=sum(s["recognized"] and s["code"].strip() == s["expected"] for s in pq),
                           recognized_alternate_pq=sum(match_alt(s) for s in pq), locked_joint=int(score[arm].sum()))
        examples[arm] = {key: next((s for s in byid.values() if check(s)), None) for key, check in {
            "recognized-and-alternate-code": match_alt,
            "recognized-and-original-code": lambda s: real_pq(s) and s["recognized"] and s["code"].strip() == s["expected"],
            "incomplete": lambda s: s["stop"] != "code",
            "unexpected-token": lambda s: not s["recognized"],
        }.items()}
    return score, totals, examples


def real_pq(s):
    return s is not None and s["stop"] == "code" and s["code"] in (" P", " Q")


def match_alt(s):
    return bool(real_pq(s) and s["recognized"] and s["code"].strip() != s["expected"])


def estimate(rows, z, cfg, gate, rollout, eval_ids):
    names = list(cfg)
    codes = code_margin(z, rows)
    states = state_margin(z)
    sl = z["local_total"] * sign(rows)[:, None]
    want = np.array([r["state_ids"] for r in rows])
    recognized = (z["state_top"] == want[:, 1:2]).astype(np.float64)
    switch_data, rolls, rollcases = rollout_counts(rollout, rows, z, cfg, eval_ids)

    def view(a, k):
        return a[:, names.index(k)].astype(np.float64)

    def rand(a, k):
        return (view(a, f"{k}-rand1") + view(a, f"{k}-rand2")) / 2

    def loss(k):
        return view(codes, "sham") - view(codes, k)

    sets = {f"g-{k}-direction": view(sl, k) - rand(sl, k) for k in ("fact", "book", "both")}
    sets.update({"g-both-pq": rand(codes, "both") - view(codes, "both"),
                 "g-kv-rescue": view(codes, "g-kv") - view(codes, "both"), "g-kv-reverse": loss("g-reverse"),
                 "e-switch": view(recognized, "e") - rand(recognized, "e"),
                 "e-native-pq": rand(codes, "e") - view(codes, "e"),
                 "e-switch-pq": switch_data["e"] - (switch_data["e-rand1"] + switch_data["e-rand2"]) / 2})
    assert list(sets) == list(gate["labels"])
    primary = {}
    for k, candidate in sets.items():
        locked = k == "e-switch-pq"
        score = candidate if locked else fam(candidate)
        r = summarize(score, gate["confidence"])
        floor = gate["switch"] if k in RATES else gate["logit"]
        passed = r["n"] >= gate["natural_n" if locked else "n"] and r["mean"] >= floor and r["ci"][0] > 0
        primary[k] = dict(**r, passed=bool(passed), floor=floor,
                          leave_one_lexeme=None if locked else leave_lexeme(score, rows))
    absolute = {f"{k}-pq-loss": loss(k) for k in ("fact", "book", "both", "both-rand1", "both-rand2", "cache-v-zero",
                "g-reverse", "e", "e-rand1", "e-rand2", "e-zero", "e-full")}
    absolute.update({f"{k}-restore": view(codes, k) - view(codes, "both") for k in ("g-k", "g-v", "g-kv")})
    absolute.update({f"{k}-local": view(sl, k) for k in ("fact", "book", "both")})
    absolute.update({f"{k}-state-loss": view(states, "sham") - view(states, k) for k in ("e", "e-rand1", "e-rand2", "e-zero", "e-full")})
    gold = np.array([r["token"]["letters"][r["code"] == "Q"] for r in rows])
    opposite = np.array([r["token"]["letters"][r["code"] == "P"] for r in rows])
    arms, geometry = {}, {}
    for j, k in enumerate(names):
        correct_prefix = z["first_mismatch"][:, j] == -1
        pq = z["code_top"][:, j]
        state = z["state_top"][:, j]
        arms[k] = dict(state_switched=int((state != want[:, 0]).sum()), recognized=int((state == want[:, 1]).sum()),
                       natural_prefix=int(correct_prefix.sum()), natural_switch_code=int((correct_prefix & (pq == opposite)).sum()),
                       conditional_switch_code=int((pq == opposite).sum()), conditional_non_pq=int(((pq != gold) & (pq != opposite)).sum()),
                       median_e_l2=float(np.median(z["e_dose"][:, j])))
        if k in ("sham", "fact", "book", "both", "both-rand1", "both-rand2", "g-kv", "g-reverse"):
            row = dict(alpha=float(np.median(z["a_applied"][:, j])))
            for type_ in ("k", "v"):
                base = z[f"{type_}_native"][:, 0]
                row[type_] = float(np.median(np.linalg.norm(z[f"{type_}_applied"][:, j] - base, axis=-1) /
                                              np.maximum(np.linalg.norm(base, axis=-1), 1e-4)))
            geometry[k] = row
    group_keys = ("fact", "book", "both", "both-rand1", "both-rand2", "g-kv")
    same = np.logical_and.reduce([z["first_mismatch"][:, names.index(k)] == -1 for k in group_keys])
    same = np.repeat(same.reshape(-1, 8).all(1), 8)
    eval_set = {k: summarize(fam(v[same])) for k, v in sets.items() if k.startswith("g-") and same.any()}
    groups = {f"role{a}-rule{b}": np.array([r["side"] == a and r["context"] == b for r in rows]) for a in (0, 1) for b in (0, 1)}
    by_role = {name: {k: summarize(v[sel].reshape(-1, 4).mean(1)) for k, v in sets.items() if k != "e-switch-pq"}
               for name, sel in groups.items() if sel.any()}
    e_key = names.index("e")
    cuts = {
        "recognized-failed-to-switch": np.flatnonzero(z["state_top"][:, e_key] != want[:, 1])[:8],
        "other-token": np.flatnonzero((z["state_top"][:, e_key] != want[:, 0]) & (z["state_top"][:, e_key] != want[:, 1]))[:8],
        "g-kv-rescue-harms": np.argsort(sets["g-kv-rescue"], kind="stable")[:8],
        "g-direction-but-buffered": [i for i in np.argsort(-view(sl, "both"), kind="stable") if abs(loss("both")[i]) <= .125][:8],
        "natural-same-word-code-switch": np.flatnonzero((z["first_mismatch"][:, names.index("both")] == -1) &
                                                         (z["code_top"][:, names.index("both")] == opposite))[:8],
    }
    examples = {k: [dict(row=rows[i], state=z["state"][i].tolist(), top=z["state_top"][i].tolist(), pq=z["pq"][i].tolist(),
                        code_top=z["code_top"][i].tolist(), alpha=z["a_applied"][i].tolist(),
                        local=z["local_total"][i].tolist(), first_mismatch=z["first_mismatch"][i].tolist(),
                        first_emitted=z["first_emitted"][i].tolist()) for i in ix] for k, ix in cuts.items()}
    return dict(n=len(rows)//8, primary=primary, bridge=all(primary[k]["passed"] for k in ("g-both-direction", "g-both-pq", "g-kv-rescue", "g-kv-reverse")),
                absolute={k: summarize(fam(v)) for k, v in absolute.items()}, arms=arms, geometry=geometry, g_common=int(same.sum()//8),
                g_common_effects=eval_set, by_role=by_role, rollout=rolls, rollout_examples=rollcases, examples=examples,
                evaluation=eval_ids)


def audit(p):
    f, ledger, b = (A.load(k) for k in ("freeze.json", "confirm.json", "confirm-behavior.json"))
    assert ledger["freeze"] == b["binding"] == B.sha("freeze.json") and f["plan"] == B.sha("plan.json")
    assert f["behavior"] == B.sha("discover-behavior.json") and ledger["behavior"] == B.sha("confirm-behavior.json")
    assert all(B.sha(k) == v for k, v in f["discovery"].items())
    assert (C.OUT / "freeze.json").stat().st_mtime_ns < (C.OUT / "confirm-behavior.json").stat().st_mtime_ns
    rows = A.load(p["materials"]["confirm"]["file"])
    full = C.O.resolved(rows, dict(rows=b["rows"], eligible=list({r["family"] for r in rows})))
    assert C.N.behavior(rows, b["rows"])["cells"] == b["cells"]
    output = dict(state=ledger["state"], natural=native_role_pairs(full, b["rows"]), cells=b["cells"],
                  failures=[dict(row=r, observation=o) for r, o in zip(rows, b["rows"], strict=True) if not o["code_correct"]],
                  provenance={k: B.sha(k) for k in ("plan.json", "freeze.json", "confirm.json", "confirm-behavior.json")})
    if ledger["state"] == "finished":
        cfg = configs(f)
        assert all(B.sha(k) == v for k, v in ledger["files"].items()) and ledger["rollout"] == B.sha("rollout.json")
        use = {r["id"]: r for r in C.O.resolved(rows, b)}
        names, active, negative = list(cfg), [], []
        for key, ck in sorted(ledger["checks"].items()):
            z = load(key)
            assert all(ck[n] == v for n, v in physical(z, cfg).items())
            for j in np.flatnonzero(~z["technical"]):
                negative.append(dict(id=z["ids"][j], pq=z["pq"][j, 0].tolist(), drift=float(z["numeric_diff"][j]),
                                     first_mismatch=int(z["first_mismatch"][j, 0]), actual=int(z["first_emitted"][j, 0])))
            take = np.repeat(z["technical"].reshape(-1, 8).all(1), 8)
            fields = ("ids", "state", "state_top", "pq", "code_top", "local_total", "first_mismatch", "first_emitted", "e_dose",
                      "k_native", "v_native", "k_applied", "v_applied", "a_applied")
            active.append({k: z[k][take] for k in fields})
        data = {k: np.concatenate([z[k] for z in active]) for k in active[0]}
        accepted = data["ids"].tolist()
        assert accepted == ledger["accepted"]
        assert ledger["evaluation"] == [accepted[i*8+i%8] for i in range(min(C.E_CASES, len(accepted)//8))]
        output.update(technical=ledger["checks"], technical_counterexamples=negative, eligible=ledger["eligible"], arm_names=names)
        if accepted:
            output.update(estimate([use[k] for k in accepted], data, cfg, p["gate"], A.load("rollout.json"), ledger["evaluation"]))
        else:
            output["state"] = "technical-gate-failed"
    A.save("audit.json", output)
    print("Finished", output.get("n"), "G bridge", output.get("bridge"),
          {k:(v["mean"], v["passed"]) for k,v in output.get("primary",{}).items()}, flush=True)
