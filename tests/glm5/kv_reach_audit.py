"""Recompute physical reachability and evaluate two frozen paired comparisons."""

from collections import defaultdict
import sys

import numpy as np
from scipy.stats import binom, binomtest

import kv_bf_audit as PriorAudit
import kv_reach_run as C
from kv_reach_q14 import CAP, METHODS


A, B, P = C.A, C.B, C.P


def geometry(d):
    a = d["natural_alpha"].astype(np.float64)
    lo, hi = a.min(-1), a.max(-1)
    size = np.linalg.norm((d["value_donor"]-d["value_state"]).astype(np.float64), axis=-1)
    mass = lo+hi
    original = np.minimum(hi, 2*lo)*size
    return dict(alpha=a, lo=lo, hi=hi, d=size, ratio=hi/np.maximum(lo, 1e-30), mass=mass,
                original=original, max_original_address=CAP*lo*size, max_same_two_slot_mass=(2/3)*mass*size)


def stats(g, keep, limits):
    floor = limits["norm_floor"]
    bad = g["original"] < floor
    hard = g["max_original_address"] < floor
    if not keep.any():
        return dict(n=0)
    q = lambda v: np.quantile(v[keep], [.10, .5, .90], axis=0).tolist()
    return dict(n=int(keep.sum()),
                quantiles={k:q(v) for k, v in g.items()},
                below_target=bad[keep].sum(0).tolist(), cap_unreachable=hard[keep].sum(0).tolist(),
                even_optimizing_slot_ratio_below_target=(bad & (g["max_same_two_slot_mass"] < floor))[keep].sum(0).tolist(),
                high_to_low_ratio_gt2_below=(bad & (g["ratio"] > 2))[keep].sum(0).tolist(),
                floor_after_typical_value=(np.minimum(g["hi"], 2*g["lo"])*np.median(g["d"][keep], axis=0) < floor)[keep].sum(0).tolist(),
                floor_after_typical_address=(g["d"]*np.median(np.minimum(g["hi"], 2*g["lo"])[keep], axis=0) < floor)[keep].sum(0).tolist())


def recompute(d, truth, lim):
    g = geometry(d)
    keep = d["good"]
    wanted = d["native-want"][:, :, :2]
    expected = np.broadcast_to(g["original"][:, None, :], wanted.shape)
    assert np.allclose(wanted[keep], expected[keep], atol=1e-9, rtol=1e-6)
    flags = (wanted < lim["norm_floor"]) & keep[:, None, None]
    assert np.array_equal(flags, d["native-flags"][:, :, :2, 0])
    cells = {}
    for c in (0, 1):
        for order in (0, 1):
            cell = keep & np.array([truth[x]["context"] == c and truth[x]["order"] == order for x in d["ids"]])
            cells[f"context{c}-order{order}"] = stats(g, cell, lim)
    fail = keep[:, None] & (g["original"] < lim["norm_floor"])
    examples = []
    for h in range(2):
        indices = sorted(np.flatnonzero(fail[:, h]), key=lambda z: (g["original"][z, h], d["ids"][z]))
        # Include cases near the gate as well as the hardest cases.
        indices = list(dict.fromkeys(indices[:3]+indices[-3:]))
        for j in indices:
            r = truth[d["ids"][j]]
            examples.append(dict(row=r, receiver=h+1, **{k: g[k][j, h].tolist() for k in g}))
    return dict(pooled=stats(g, keep, lim), cells=cells, counterexamples=examples)


def historical():
    old = C.prior("q14-confirm.json")
    material = C.prior("material-confirm.json")
    p = C.prior("plan.json")
    current = C.C.OUT
    try:
        C.C.OUT = C.OLD
        d = B.load_data(old)
    finally:
        C.C.OUT = current
    result = recompute(d, {r["id"]: r for r in material}, p["dose"]["limits"])
    result.update(binding={f: A.digest(C.OLD/f) for f in ("plan.json", "q14-confirm.json", "material-confirm.json", "q14-selection.json")},
                  route=C.prior("q14-selection.json")["receiver"], note="Fixed weights. Cap ceiling uses actual unchanged per-slot alpha. The same-mass optimum is a mathematical bound obtained by redistributing mass between State and Note; no response is imputed.")
    A.save("historical-dose-factorization.json", result)


def pool(d, route, truth, lim):
    flags = d["late-flags"]
    residual = d["good"] & ~B.view_pass(d, "late") & ~flags[..., 0].any((1, 2))
    cases = [dict(row=truth[d["ids"][j]], natural_alpha=d["natural_alpha"][j].tolist(),
                  flags=flags[j].tolist(), pair=d["late-pair"][j].tolist(),
                  required=d["late-want"][j].tolist(), actual=d["late-effect-norm"][j].tolist())
             for j in np.flatnonzero(residual)[:12]]
    return dict(arms={m: PriorAudit.classify(d, m, truth) for m in METHODS}, dose=recompute(d, truth, lim),
                above_floor_late_failures=dict(n=int(residual.sum()), rows=cases),
                rounding=PriorAudit.diagnose(d, route["receiver_layer"], lim, truth), addressing=PriorAudit.addressing(d, truth))


def qualified(d):
    fam = defaultdict(list)
    for k, a in zip(PriorAudit.family_ids(d), d["good"], strict=True):
        fam[k].append(a)
    assert all(len(v) == 8 for v in fam.values())
    return {k for k, v in fam.items() if all(v)}


def passes(d, method):
    ok = B.view_pass(d, method)
    fam = defaultdict(list)
    for k, a in zip(PriorAudit.family_ids(d), ok, strict=True):
        fam[k].append(a)
    assert all(len(v) == 8 for v in fam.values())
    return {k: all(v) for k, v in fam.items()}


def paired(d, a, e, b, truth, keep):
    left, right = passes(d, a), passes(e, b)
    ids = sorted(keep)
    diff = np.array([int(left[k])-int(right[k]) for k in ids])
    if not len(diff):
        return dict(n=0)
    gains, losses = int((diff > 0).sum()), int((diff < 0).sum())
    rng = np.random.default_rng(C.SEED)
    means = diff[rng.integers(0, len(diff), size=(12000, len(diff)))].mean(-1)
    interval = np.quantile(means, [.0125, .9875]).tolist()
    p = binomtest(gains, gains+losses, alternative="two-sided").pvalue if gains+losses else 1.
    by_word = defaultdict(set)
    for k in ids:
        r = truth[k+"-000"]
        for name in ("state", "note", "alt"):
            by_word[r[name]].add(k)
    without_word = {word: float(np.mean([v for k, v in zip(ids, diff, strict=True) if k not in excluded]))
                    for word, excluded in by_word.items() if len(excluded) < len(ids)}
    return dict(n=len(diff), gains=gains, losses=losses, unchanged=len(diff)-gains-losses, effect=float(diff.mean()),
                ci97_5=interval, pair_exact_p=float(p), passes=bool(interval[0] > 0 and p < .025),
                gained=[k for k, n in zip(ids, diff, strict=True) if n > 0],
                lost=[k for k, n in zip(ids, diff, strict=True) if n < 0], leave_one_word_out=without_word)


def analyze():
    p = C.verify()
    freeze = A.load("q14-freeze.json")
    methods = {"q4": "q4-confirm.json", "q14-fixed": "q14-confirm-fixed.json", "q14-selected": "q14-confirm-selected.json"}
    bind = {"q4": B.sha("q4-behavior-confirm.json"), "q14": B.sha("q14-behavior-confirm.json")}
    truth = {r["id"]: r for r in A.load("material-confirm.json")}
    lim = p["dose"]["limits"]
    q4b, q14b = (A.load(f"{m}-behavior-confirm.json") for m in ("q4", "q14"))
    assert not (C.OUT/"audit.json").exists() and q4b["passed"] and q14b["passed"]
    data, result, routes = {}, {}, {"q4": dict(source=[20, 5], receiver=[19, 25], receiver_layer=21),
                                     **{f"q14-{m}": freeze[m] for m in ("fixed", "selected")}}
    for name, f in methods.items():
        z = A.load(f)
        assert z["status"] == "finished" and z["behavior"] == bind[name.split("-")[0]]
        if name != "q4":
            assert z["freeze"] == B.sha("q14-freeze.json")
        d = B.load_data(z)
        assert len(d["ids"]) == len(set(d["ids"]))
        assert set(d["ids"]).issubset(truth)
        data[name] = d
        result[name] = pool(d, routes[name], truth, lim)
    new, old = data["q14-selected"], data["q14-fixed"]
    assert np.array_equal(new["ids"], old["ids"]) and len(new["ids"]) == len(truth)
    admitted = qualified(new)
    assert admitted == set(q14b["eligible"]) == qualified(old)
    both = admitted & qualified(data["q4"])
    exact = both & set(q14b["exact_families"])
    primary = dict(route=paired(new, "late", old, "late", truth, admitted),
                   precision=paired(new, "late", new, "native", truth, admitted))
    for m in data:
        for key, subset in (("common", both), ("common_exact", exact)):
            d = PriorAudit.mask_family(data[m], subset)
            result[m][key] = {method: PriorAudit.classify(d, method, truth) for method in METHODS} if subset else {}
    A.save("audit.json", dict(plan=B.sha("plan.json"), binding=bind, freeze=B.sha("q14-freeze.json"),
           families=dict(q4=len(qualified(data["q4"])), q14=len(admitted), common=len(both), common_exact=len(exact)),
           common_families=sorted(both), common_exact_families=sorted(exact), primary=primary, routes=routes, results=result))


if __name__ == "__main__":
    C.setup()
    cmd = sys.argv[1]
    if cmd == "historical":
        historical()
    elif cmd == "freeze-analysis":
        assert not (C.OUT/"q4-behavior-confirm.json").exists()
        n = C.PANEL["confirm"]
        exact_power = 0.
        for discordant in range(1, n+1):
            wins = np.arange(discordant+1)
            reject = 2*binom.cdf(np.minimum(wins, discordant-wins), discordant, .5) < .025
            exact_power += binom.pmf(discordant, n, .30)*binom.pmf(wins[reject], discordant, .25/.30).sum()
        A.save("analysis-binding.json", dict(plan=B.sha("plan.json"), source=A.digest(__file__), stamp=C.stamp(),
               power=dict(example=dict(gain=.25, loss=.05, families=n, alpha=.025), exact_two_sided=float(exact_power),
                          correction="The plan's check_phase call passed the loss probability where the helper expects total discordance. Use this exact mixture over Binomial(n, .30) discordance instead. Applies to the exact test alone with independent pairs, before coverage and lexical dependence.")))
    elif cmd == "analyze":
        a = A.load("analysis-binding.json")
        assert a["plan"] == B.sha("plan.json") and a["source"] == A.digest(__file__)
        analyze()
    else:
        raise ValueError(cmd)
