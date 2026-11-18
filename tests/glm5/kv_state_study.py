"""Pilot, discovery, candidate freeze, confirmation, and actual greedy continuation."""

from itertools import batched
import re
import time

import numpy as np

import kv_state_native as Native
import kv_state_run as C
from kv_prefix_study import directions, code_margin, state_margin, sign
from kv_prefix_native import adjoint


A, B = C.A, C.B
ALL = list(range(128))


def load(name):
    with np.load(C.OUT / name) as z:
        return {k: z[k] for k in z.files}


def persist(name, rows, action):
    if not (C.OUT / name).exists():
        B.write_np(name, action())
    z = load(name)
    assert z["ids"].tolist() == [r["id"] for r in rows], name
    return z


def behavior(m, rows, stage, bind):
    name = f"{stage}-behavior.json"
    if not (C.OUT / name).exists():
        obs, _ = m.observe(rows, f"{stage}-natural")
        A.save(name, dict(**C.N.behavior(rows, obs), binding=bind))
    data = A.load(name)
    assert data["binding"] == bind
    return data


def e_spec(layer, coordinates=(), kind="swap"):
    return dict(layer=layer, coordinates=[int(i) for i in coordinates], kind=kind)


def configs(f):
    fact, book = f["g"]
    both = [fact, book]
    e = f["e"]
    a = Native.arm
    cfg = {"sham": a(), "fact": a([fact]), "book": a([book]), "both": a(both),
           "fact-self": a([fact], own=True), "book-self": a([book], own=True), "both-self": a(both, own=True),
           "g-k": a(both, (0, ALL, [])), "g-v": a(both, (0, [], ALL)), "g-kv": a(both, (0, ALL, ALL)),
           "g-reverse": a(cache=(3, ALL, ALL)), "cache-v-zero": a(cache=(-1, [], ALL)),
           "e": a(e=e), "e-self": a(e=e_spec(e["layer"], e["coordinates"], "self")),
           "e-zero": a(e=e_spec(e["layer"], e["coordinates"], "zero")), "e-full": a(e=e_spec(39, kind="swap-all"))}
    for k, ((fr, br), coords) in enumerate(zip(f["g_random"], f["e_random"], strict=True), 1):
        cfg[f"fact-rand{k}"] = a([fr]); cfg[f"book-rand{k}"] = a([br]); cfg[f"both-rand{k}"] = a([fr, br])
        cfg[f"e-rand{k}"] = a(e=e_spec(e["layer"], coords))
    return cfg


def physical(z, cfg):
    names = list(cfg)
    for key in ("fact-self", "book-self", "both-self", "e-self"):
        for metric in ("state", "state_top", "pq", "code_top", "first_mismatch", "head"):
            assert np.array_equal(z[metric][:, names.index(key)], z[metric][:, 0]), (key, metric)
    for j, (k, spec) in enumerate(cfg.items()):
        if spec["e"] is None:
            assert np.array_equal(z["state"][:, j], z["state"][:, 0]), k
            assert np.array_equal(z["state_top"][:, j], z["state_top"][:, 0]), k
        else:
            for kind in ("k", "v"):
                assert np.array_equal(z[f"{kind}_native"][:, j], z[f"{kind}_native"][:, 0]), (k, kind)
        for i, kind in enumerate(("k", "v"), 1):
            patch = spec["patch"]
            expect = z[f"{kind}_native"][:, j].copy()
            if patch is not None and patch[i]:
                expect[:, patch[i]] = z[f"{kind}_native"][:, patch[0], patch[i]] if patch[0] >= 0 else 0
            assert np.array_equal(z[f"{kind}_applied"][:, j], expect), (k, kind)
    assert np.isfinite(z["local_error"]).all() and z["local_error"].max() < .05
    assert z["local_closure"].max() < .001
    return dict(raw_pq_drift=float(z["numeric_diff"].max()), physical=float(z["local_error"].max()),
                closure=float(z["local_closure"].max()), bad=z["ids"][~z["technical"]].tolist())


def pilot(m, _):
    name = "material-pilot.json"
    if not (C.OUT / name).exists():
        A.save(name, C.material("pilot", m.tok, C.excluded()))
    rows = A.load(name)
    seen = behavior(m, rows, "pilot", B.sha(name))
    assert seen["passed"], seen["cells"]
    use = C.O.resolved(rows, seen)[:8]
    known = {x["id"]: x for x in seen["rows"]}
    d = persist("pilot-adjoint.npz", use, lambda: adjoint(m, use))
    ed = persist("pilot-e-adjoint.npz", use, lambda: Native.e_adjoint(m, use))
    es = np.argsort(-ed["product"].mean(0)[-1], kind="stable")[:768].tolist()
    cfg = configs(dict(g=[(12, 20, "fact"), (23, 5, "book")], e=e_spec(39, es),
                       g_random=[[(12, 21, "fact"), (23, 6, "book")], [(12, 22, "fact"), (23, 7, "book")]],
                       e_random=[list(range(768)), list(range(768, 1536))]))
    z = Native.panel(m, use, cfg, "run", known, directions(d))
    checks = physical(z, cfg)
    assert z["technical"].all() and ed["error"].max() < .05, checks
    top = z["state_top"][:, list(cfg).index("e-full")]
    assert np.array_equal(top, [r["state_ids"][1] for r in use])
    fmap = {39: dict(zip(z["ids"], z["donor_e39"], strict=True))}
    test = replay(m, [(r, int(token)) for r, token in zip(use, top, strict=True)], cfg["e-full"]["e"], fmap)
    assert len(test["observations"]) == 8 and all(x["emitted"] for x in test["observations"].values()), test
    B.write_np("pilot-native.npz", z)
    A.save("pilot-replay.json", test)
    A.save("pilot-engine.json", dict(passed=True, behavior=seen["cells"], chosen_view_count=len(use), waveform=checks,
                                     e_error=float(ed["error"].max()), c_error=float(d["error"].max()),
                                     e_full_recognized=int(len(top)), replay_reached_code=sum(r["stop"] == "code" for r in test["observations"].values())))
    print("Pilot passed", checks, flush=True)


def shortlist(scores):
    out = []
    for j, key in enumerate(C.G_SOURCES):
        best = []
        for metric in (0, 1):
            work = scores[:, j, :, metric].copy()
            for l, h, _ in best:
                work[l] = -np.inf
            for _ in range(4):
                l, h = np.unravel_index(np.argmax(work), work.shape)
                best.append([int(l), int(h), key])
                work[l] = -np.inf
        out.extend(best)
    return out


def match_heads(screen, g, rng):
    result = [[], []]
    log = []
    for l, h, key in g:
        dose = screen[l, C.G_SOURCES.index(key), :, 1]
        pool = [int(i) for i in np.argsort(abs(np.log1p(dose) - np.log1p(dose[h])), kind="stable") if i != h][:10]
        picks = rng.choice(pool, 2, replace=False).tolist()
        for j, i in enumerate(picks):
            result[j].append([l, i, key])
        log.append(dict(pool=pool, original=float(dose[h]), control=dose[picks].tolist()))
    return result, log


def match_e(score, chosen, rng):
    order = np.argsort(score, kind="stable").tolist()
    rank = {i: j for j, i in enumerate(order)}
    out = []
    for _ in range(2):
        blocked = set(chosen)
        random = []
        for i in chosen:
            here = rank[i]
            pool = [j for j in order[max(0, here-96):here+97] if j not in blocked]
            if not pool:
                pool = [j for j in order if j not in blocked]
            j = int(rng.choice(pool))
            random.append(j)
            blocked.add(j)
        out.append(random)
    return out


def choose_e(predict, scores, use, config):
    top = scores["state_top"]
    targets = np.array([r["state_ids"] for r in use])
    assert np.array_equal(top[:, 0], targets[:, 0]) and predict["error"].max() < .05
    margin = state_margin(scores)
    stats = []
    for j, (name, spec) in enumerate(config.items()):
        if name in ("sham", "full"):
            continue
        count = int((top[:, j] == targets[:, 1]).sum())
        stats.append(dict(name=name, recognized=count, any_switch=int((top[:, j] != targets[:, 0]).sum()),
                          state_margin_loss=float((margin[:, 0] - margin[:, j]).mean()),
                          layer=spec["e"]["layer"], budget=len(spec["e"]["coordinates"])))
    eligible = [x for x in stats if x["recognized"] >= .5 * len(use)]
    if eligible:
        winner = min(eligible, key=lambda x: (x["budget"], -x["recognized"], -x["state_margin_loss"], x["layer"]))
    else:
        winner = min(stats, key=lambda x: (-x["recognized"], -x["state_margin_loss"], x["budget"], x["layer"]))
    full = int((top[:, list(config).index("full")] == targets[:, 1]).sum())
    assert full == len(use)
    return config[winner["name"]]["e"], stats, full


def discover(m, p):
    assert not (C.OUT / "freeze.json").exists() and not (C.OUT / "confirm-behavior.json").exists()
    rows = A.load(p["materials"]["discover"]["file"])
    b = behavior(m, rows, "discover", B.sha("plan.json"))
    assert b["passed"], b["cells"]
    use = C.O.resolved(rows, b)
    known = {x["id"]: x for x in b["rows"]}
    rng = np.random.default_rng(C.SEED)
    ed = persist("discover-e-adjoint.npz", use, lambda: Native.e_adjoint(m, use))
    rankings = np.argsort(-ed["product"].mean(0), axis=1, kind="stable")
    a = Native.arm
    e_cfg = {"sham": a(), "full": a(e=e_spec(39, kind="swap-all"))}
    for i, rank in zip(C.E_LAYERS, rankings, strict=True):
        for n in C.E_BUDGETS:
            e_cfg[f"{i}-{n}"] = a(e=e_spec(i, rank[:n]))
    ez = persist("discover-e-search.npz", use, lambda: Native.panel(m, use, e_cfg, "e-search"))
    e, e_stats, full = choose_e(ed, ez, use, e_cfg)
    mean_dose = ed["displacement"].mean(0)[C.E_LAYERS.index(e["layer"])]
    e_random = match_e(mean_dose, e["coordinates"], rng)
    d = persist("discover-c-adjoint.npz", use, lambda: adjoint(m, use))
    sd = directions(d)
    screen = persist("discover-g-screen.npz", use, lambda: Native.panel(m, use, {"sham": a()}, "screen"))["screen"].mean(0)
    offers = shortlist(screen)
    g_cfg = {"sham": a(), **{f"g{i}": a([h]) for i, h in enumerate(offers)}}
    gz = persist("discover-g-single.npz", use, lambda: Native.panel(m, use, g_cfg, "run", known, sd))
    keep = gz["technical"]
    assert keep.mean() >= .95 and np.array_equal(gz["state"], np.broadcast_to(gz["state"][:, :1], gz["state"].shape))
    local = (gz["local_total"][keep, 1:] * sign(use)[keep, None]).mean(0)
    code = code_margin(gz, use)
    best = [int(np.argmax(local[:8]))]
    best.append(max((i for i in range(8, 16) if offers[i][:2] != offers[best[0]][:2]), key=lambda i: local[i]))
    g = [offers[i] for i in best]
    g_random, head_dose = match_heads(screen, g, rng)
    f = dict(plan=B.sha("plan.json"), behavior=B.sha("discover-behavior.json"), time=C.O.stamp(),
             g=g, g_random=g_random, g_match=head_dose, e=e, e_random=e_random, e_search=e_stats,
             e_dose=dict(chosen=float(mean_dose[e["coordinates"]].mean()), random=[float(mean_dose[c].mean()) for c in e_random]),
             g_search=[dict(site=o, direction=float(local[i]), pq_loss=float((code[keep, 0]-code[keep, i+1]).mean()))
                       for i, o in enumerate(offers)], discovered=len(use)//8, positive_control=full)
    cfg = configs(f)
    f["arms"] = list(cfg)
    verify_z = persist("discover-g-and-e.npz", use[:32], lambda: Native.panel(m, use[:32], cfg, "run", known, sd))
    f["physics"] = physical(verify_z, cfg)
    names = ("discover-e-adjoint.npz", "discover-e-search.npz", "discover-c-adjoint.npz", "discover-g-screen.npz",
             "discover-g-single.npz", "discover-g-and-e.npz")
    f["discovery"] = {k: B.sha(k) for k in names}
    A.save("freeze.json", f)
    print("Frozen heads", g, "E", e["layer"], len(e["coordinates"]), "full", full, flush=True)


def sample_order(cases):
    by_family = {}
    for row, token in cases:
        by_family.setdefault(row["family"], []).append((row, token))
    for j in range(8):
        for i, family in enumerate(by_family.values()):
            if j < len(family):
                yield family[(i + j) % len(family)]


def replay(m, cases, spec, donors, preferred=()):
    by_id = {row["id"]: (row, token) for row, token in cases}
    locked = [by_id[k] for k in preferred if k in by_id]
    others = [(r, t) for r, t in cases if r["id"] not in set(preferred)]
    selected = (locked + list(sample_order(others)))[:C.E_CASES]
    active = []
    for row, token in selected:
        at = row["token"]["slots"][0]
        active.append(dict(row=row, top=token, draft=[token, *row["token"]["ids"][at + 1:]][:C.E_TOKENS],
                           prompt=row["token"]["ids"][:at], verified=[], trace=[]))
    eos = m.model.generation_config.eos_token_id
    stopping = {eos} if isinstance(eos, int) else set(eos or [])
    finished = {}

    def render(x, tokens):
        return m.tok.decode([*x["prompt"][x["row"]["token"]["start"]:], *tokens])

    def save(x, stop, code=None):
        row, toks = x["row"], x["verified"]
        code_token = None if code is None else m.tok.decode([code])
        finished[row["id"]] = dict(id=row["id"], stop=stop, emitted=toks, target=x["top"],
                                  text=render(x, toks), first=m.tok.decode([x["top"]]),
                                  recognized=x["top"] == row["state_ids"][1], code_token=code, code=code_token,
                                  expected=row["code"], trace=x["trace"])

    for it in range(C.E_PASSES):
        if not active:
            break
        rows = [dict(x["row"], token=dict(x["row"]["token"], ids=x["prompt"]+x["draft"])) for x in active]
        outputs = Native.panel(m, rows, {"patched": Native.arm(e=spec)}, "rollout", e_donor=donors)
        waiting = []
        for x, pred in zip(active, outputs, strict=True):
            verified = x["verified"]
            if pred[0] != x["top"] or pred[:len(verified)] != verified:
                x["trace"].append(dict(pass_id=it, mismatch="prefix", actual=pred[0]))
                save(x, "replay-conflict")
                continue
            for j in range(len(verified), len(pred)):
                token = pred[j]
                reached = re.search(r"(?:^|\n)Code:[ \t]*$", render(x, verified)) is not None
                verified.append(token)
                if reached or token in stopping or len(verified) >= C.E_TOKENS:
                    x["trace"].append(dict(pass_id=it, verified=len(verified), actual=token))
                    save(x, "code" if reached else "eos" if token in stopping else "token-budget", token if reached else None)
                    break
                if j >= len(x["draft"]) or x["draft"][j] != token:
                    x["draft"] = verified.copy() + x["draft"][j+1:]
                    x["trace"].append(dict(pass_id=it, changed_draft_at=j, actual=token))
                    waiting.append(x)
                    break
        active = waiting
    for x in active:
        save(x, "pass-budget")
    return dict(total=len(cases), tested=len(selected), order=[r["id"] for r, _ in selected], observations=finished)


def confirm(m, p):
    f = A.load("freeze.json")
    assert f["plan"] == B.sha("plan.json") and f["behavior"] == B.sha("discover-behavior.json")
    assert all(B.sha(k) == v for k, v in f["discovery"].items())
    rows = A.load(p["materials"]["confirm"]["file"])
    b = behavior(m, rows, "confirm", B.sha("freeze.json"))
    status = dict(freeze=B.sha("freeze.json"), behavior=B.sha("confirm-behavior.json"))
    if not b["passed"]:
        A.save("confirm.json", dict(state="behavior-gate-failed", **status))
        return
    use = C.O.resolved(rows, b)
    known = {x["id"]: x for x in b["rows"]}
    cfg = configs(f)
    assert list(cfg) == f["arms"]
    trials = {k: [] for k in ("e", "e-rand1", "e-rand2", "e-zero", "e-full")}
    donors = {l: {} for l in {39, f["e"]["layer"]}}
    groups = list(batched(use, C.SLAB))
    start = time.monotonic()
    C.N.progress("state-heldout", 0, len(groups), start)
    files, checks, okids = {}, {}, []
    for j, r in enumerate(groups, 1):
        dn, zn = f"confirm-adjoint-{j:03d}.npz", f"confirm-native-{j:03d}.npz"
        d = persist(dn, r, lambda: adjoint(m, r))
        z = persist(zn, r, lambda: Native.panel(m, r, cfg, "run", known, directions(d)))
        checks[zn] = dict(**physical(z, cfg), adjoint_error=float(d["error"].max()))
        keep = np.repeat(z["technical"].reshape(-1, 8).all(-1), 8)
        okids.extend(z["ids"][keep].tolist())
        for l, bank in donors.items():
            bank.update(zip(z["ids"], z[f"donor_e{l}"], strict=True))
        for key, trial in trials.items():
            top = z["state_top"][:, list(cfg).index(key)]
            trial.extend((row, int(t)) for row, t, ok in zip(r, top, keep, strict=True) if ok and t != row["state_ids"][0])
        files.update({k: B.sha(k) for k in (dn, zn)})
        C.N.progress("state-heldout", j, len(groups), start)
    evaluation = [okids[i * 8 + i % 8] for i in range(min(C.E_CASES, len(okids) // 8))]
    rolls = {}
    for k, trial in trials.items():
        name = f"rollout-{k}.json"
        if not (C.OUT / name).exists():
            A.save(name, replay(m, trial, cfg[k]["e"], donors, evaluation))
        rolls[k] = A.load(name)
        files[name] = B.sha(name)
    A.save("rollout.json", rolls)
    A.save("confirm.json", dict(state="finished", files=files, checks=checks, accepted=okids, eligible=len(use)//8, evaluation=evaluation,
                                rollout=B.sha("rollout.json"), **status))
