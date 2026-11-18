"""Pilot, discovery, freeze, and held-out original-coordinate interventions."""

from itertools import batched
import re
import time

import numpy as np

import kv_prefix_native as Native
import kv_prefix_run as C


A, B = C.A, C.B
ALL = list(range(128))


def read_np(name):
    with np.load(C.OUT / name) as src:
        return {k: src[k] for k in src.files}


def persist(name, rows, compute):
    if not (C.OUT / name).exists():
        B.write_np(name, compute())
    data = read_np(name)
    assert list(data["ids"]) == [r["id"] for r in rows], name
    return data


def behavior(m, rows, stage, binding):
    name = f"{stage}-behavior.json"
    if not (C.OUT / name).exists():
        observations, _ = m.observe(rows, f"{stage}-natural")
        A.save(name, dict(**C.N.behavior(rows, observations), binding=binding))
    out = A.load(name)
    assert out["binding"] == binding
    return out


def directions(data):
    assert np.isfinite(data["d"]).all() and (np.linalg.norm(data["d"], axis=-1) > 0).all()
    assert data["error"].max() <= .05, float(data["error"].max())
    return dict(zip(data["ids"], data["d"], strict=True))


def sign(rows):
    return np.array([1 if r["code"] == "Q" else -1 for r in rows], dtype=np.int8)


def code_margin(z, rows):
    return (z["pq"][..., 1] - z["pq"][..., 0]) * sign(rows)[:, None]


def state_margin(z):
    return z["state"][..., 0] - z["state"][..., 1]


def configurations(f):
    e, s = f["early"], f["state_route"]
    k, v = f["coordinates"]
    arm = Native.arm
    out = {"sham": arm(), "early": arm(e), "early-self": arm(e, self_restore=True),
           "state-route": arm(s), "state-self": arm(s, self_restore=True),
           "kv-block": arm(patch=(-1, k, v)), "kv-self": arm(patch=(-1, k, v), self_restore=True),
           "kv-rescue-k": arm(e, (0, k, [])), "kv-rescue-v": arm(e, (0, [], v)),
           "kv-rescue": arm(e, (0, k, v)), "native-full-rescue": arm(e, (0, ALL, ALL)),
           "reverse": arm(patch=(1, k, v)), "native-v-block": arm(patch=(-1, [], ALL))}
    for i, ((rk, rv), re, rs) in enumerate(zip(f["random_coordinates"], f["early_controls"], f["state_controls"], strict=True), 1):
        out[f"early-rand{i}"] = arm(re)
        out[f"state-rand{i}"] = arm(rs)
        out[f"kv-rand-block{i}"] = arm(patch=(-1, rk, rv))
        out[f"kv-rand-rescue{i}"] = arm(e, (0, rk, rv))
    return out


def physical(z, config):
    arms = {k: i for i, k in enumerate(config)}
    for key in ("early-self", "state-self", "kv-self"):
        for metric in ("pq", "code_top", "state", "first_mismatch", "head"):
            assert np.array_equal(z[metric][:, arms[key]], z[metric][:, 0]), (key, metric)
    unchanged = ["kv-rescue-k", "kv-rescue-v", "kv-rescue", "native-full-rescue", "kv-rand-rescue1", "kv-rand-rescue2"]
    for key in unchanged:
        assert np.array_equal(z["state"][:, arms[key]], z["state"][:, arms["early"]]), key
    for key in ("kv-block", "kv-self", "reverse", "native-v-block", "kv-rand-block1", "kv-rand-block2"):
        assert np.array_equal(z["state"][:, arms[key]], z["state"][:, 0]), key
    for key, spec in config.items():
        p = spec["patch"]
        j = arms[key]
        for i, kind in enumerate(("k", "v"), 1):
            expected = z[f"{kind}_native"][:, j].copy()
            if p is not None and not spec["self_restore"] and p[i]:
                expected[:, p[i]] = 0 if p[0] < 0 else z[f"{kind}_native"][:, p[0], p[i]]
            assert np.array_equal(z[f"{kind}_applied"][:, j], expected), (key, kind)
    assert np.isfinite(z["local_error"]).all() and float(z["local_error"].max()) < .05
    assert np.isfinite(z["local_closure"]).all() and float(z["local_closure"].max()) < .001
    return dict(sham_max=float(z["numeric_diff"].max()), physical_max=float(z["local_error"].max()),
                local_identity_max=float(z["local_closure"].max()), exact_native_equality=True)


def pilot(m, _):
    if not (C.OUT / "material-pilot.json").exists():
        A.save("material-pilot.json", C.material("pilot", m.tok, C.excluded()))
    rows = A.load("material-pilot.json")
    seen = behavior(m, rows, "pilot", B.sha("material-pilot.json"))
    assert seen["passed"], seen["cells"]
    eligible = C.O.resolved(rows, seen)
    assert len(eligible) >= 16
    use = [eligible[0], *eligible[9:16]]
    known = {r["id"]: r for r in seen["rows"]}
    d = persist("pilot-adjoint.npz", use, lambda: Native.adjoint(m, use))
    z0 = persist("pilot-screen.npz", use, lambda: Native.panel(m, use, {"sham": Native.arm()}, "screen"))
    assert z0["screen"].shape == (8, 24, 6, 40, 3)
    f = dict(early=[(6, 3, "accepted"), (12, 20, "role-text"), (23, 0, "book-lines")],
             state_route=[(9, 8, "rejected"), (14, 6, "rule"), (21, 18, "format")],
             coordinates=[list(range(16)), list(range(16, 32))],
             random_coordinates=[[list(range(32, 48)), list(range(48, 64))], [list(range(64, 80)), list(range(80, 96))]],
             early_controls=[[(6, 4, "accepted")], [(6, 5, "accepted")]],
             state_controls=[[(9, 7, "rejected")], [(9, 6, "rejected")]])
    cfg = configurations(f)
    z = Native.panel(m, use, cfg, "pilot-native", known, directions(d))
    checks = physical(z, cfg)
    assert z["technical"].all(), (checks, z["numeric_diff"].tolist(), z["first_mismatch"][:, 0].tolist())
    B.write_np("pilot-native.npz", z)
    A.save("pilot-engine.json", dict(passed=True, views=len(use), checks=checks, adjoint_error=float(d["error"].max()),
                                     direction_norm=np.linalg.norm(d["d"], axis=1).tolist()))
    print("Pilot behavior", seen["cells"], "Native checks", checks, flush=True)


def shortlist(screen):
    out = []
    for i, source in enumerate(C.SOURCES):
        state, dose = screen[:, i, :, 0], screen[:, i, :, 1].copy()
        l, h = np.unravel_index(np.argmax(state), state.shape)
        out.append([int(l), int(h), source])
        dose[l] = -np.inf
        l, h = np.unravel_index(np.argmax(dose), dose.shape)
        out.append([int(l), int(h), source])
    return out


def coalition(order, specs):
    selected = []
    for j in order:
        x = specs[j]
        if all(y[0] != x[0] and y[2] != x[2] for y in selected):
            selected.append(x)
        if len(selected) == 3:
            break
    assert len(selected) == 3
    return sorted(selected)


def head_control(screen, specs, rng):
    controls = [[], []]
    details = []
    for l, h, source in specs:
        dose = screen[l, C.SOURCES.index(source), :, 1]
        pool = [j for j in np.argsort(abs(np.log1p(dose) - np.log1p(dose[h])), kind="stable") if j != h][:10]
        hs = rng.choice(pool, 2, replace=False).tolist()
        for c, j in zip(controls, hs, strict=True):
            c.append([l, j, source])
        details.append(dict(spec=[l, h, source], pool=[int(x) for x in pool], target=float(dose[h]),
                            selected=[float(dose[j]) for j in hs]))
    return controls, details


def coord_control(dose, selected, rng):
    groups, details = [], []
    scaled = np.log1p(dose / np.maximum(np.median(dose, axis=1, keepdims=True), 1e-6)).T
    for _ in range(2):
        used = set(selected)
        chosen, pools = [], []
        for i in selected:
            dist = np.linalg.norm(scaled - scaled[i], axis=-1)
            pool = [int(j) for j in np.argsort(dist, kind="stable") if j not in used][:28]
            j = int(rng.choice(pool))
            chosen.append(j)
            pools.append(pool)
            used.add(j)
        groups.append(chosen)
        details.append(dict(pools=pools, doses=dose[:, chosen].tolist()))
    return groups, details


def discover(m, p):
    assert not (C.OUT / "confirm-behavior.json").exists() and not (C.OUT / "freeze.json").exists()
    rows = A.load(p["materials"]["discover"]["file"])
    b = behavior(m, rows, "discover", B.sha("plan.json"))
    assert b["passed"], b["cells"]
    use = C.O.resolved(rows, b)
    known = {r["id"]: r for r in b["rows"]}
    adj = persist("discover-adjoint.npz", use, lambda: Native.adjoint(m, use))
    d = directions(adj)
    screen = persist("discover-screen.npz", use, lambda: Native.panel(m, use, {"sham": Native.arm()}, "screen"))["screen"].mean(0)
    specs = shortlist(screen)
    single_cfg = {"sham": Native.arm(), **{f"single-{j}": Native.arm([r]) for j, r in enumerate(specs)}}
    singles = persist("discover-singles.npz", use, lambda: Native.panel(m, use, single_cfg, "discover-singles", known, d))
    keep = singles["technical"]
    assert keep.mean() >= .95
    local = (singles["local_total"][keep] * sign(use)[keep, None]).mean(0)[1:]
    sm = state_margin(singles)
    emit = (sm[keep, :1] - sm[keep, 1:]).mean(0)
    loc_order, state_order = (np.argsort(-x, kind="stable") for x in (local, emit))
    mixed = np.argsort(np.argsort(loc_order) + np.argsort(state_order), kind="stable")
    candidates = [coalition(o, specs) for o in (loc_order, state_order, mixed)]
    route_cfg = {"sham": Native.arm(), **{f"coalition-{j}": Native.arm(r) for j, r in enumerate(candidates)}}
    routes = persist("discover-coalitions.npz", use, lambda: Native.panel(m, use, route_cfg, "discover-coalitions", known, d))
    keep = routes["technical"]
    assert keep.mean() >= .95
    loc = (routes["local_total"][keep] * sign(use)[keep, None]).mean(0)[1:]
    e = int(np.argmax(loc))
    state_ids = np.array([r["state_ids"] for r in use])
    sw = (routes["state_top"][keep, 1:] == state_ids[keep, 1:]).sum(0)
    sm = state_margin(routes)
    st = (sm[keep, :1] - sm[keep, 1:]).mean(0)
    s = max(range(len(candidates)), key=lambda i: (sw[i], st[i], -i))
    early, state = candidates[e], candidates[s]
    rng = np.random.default_rng(C.SEED)
    ec, em = head_control(screen, early, rng)
    sc, smeta = head_control(screen, state, rng) if early != state else (ec, em)
    strength = (routes["coord"][keep, e + 1] * sign(use)[keep, None, None]).mean(0)
    indices = [np.argsort(-x, kind="stable")[:C.COORDS].tolist() for x in strength]
    dose = routes["dose"][keep, e + 1].mean(0)
    kctrl, km = coord_control(dose[:2], indices[0], rng)
    vctrl, vm = coord_control(dose[2:], indices[1], rng)
    f = dict(early=early, state_route=state, early_controls=ec, state_controls=sc, early_match=em, state_match=smeta,
             coordinates=indices, random_coordinates=[list(tup) for tup in zip(kctrl, vctrl, strict=True)],
             coord_strength=strength.tolist(), coord_dose=dose.tolist(), coord_matches=[km, vm],
             selection=dict(early=e, state=s, shortlist=specs, local=local.tolist(), state_loss=emit.tolist(),
                            coalitions=candidates, coalition_local=loc.tolist(), coalition_state=st.tolist(),
                            coalition_alternatives=sw.tolist(), discovery_views=len(use), adjoint_error=float(adj["error"].max())),
             plan=B.sha("plan.json"), behavior=B.sha("discover-behavior.json"))
    config = configurations(f)
    reviewed = persist("discover-interventions.npz", use, lambda: Native.panel(m, use, config, "discover-interventions", known, d))
    f["engineering"] = physical(reviewed, config)
    f["arms"] = list(config)
    check = greedy(m, [], [(r, r["state_ids"][0]) for r in use[:8]], 8)
    A.save("discovery-rollout-check.json", check)
    assert sum(x["verified"] and x["code"].strip() == x["expected"] for x in check["observations"].values()) >= 7
    files = ("discover-adjoint.npz", "discover-screen.npz", "discover-singles.npz", "discover-coalitions.npz",
             "discover-interventions.npz", "discovery-rollout-check.json")
    f["discovery"] = {k: B.sha(k) for k in files}
    f["time"] = C.O.stamp()
    A.save("freeze.json", f)
    print("Frozen", early, "State route", state, "coordinates", indices, "local", loc, "State alternatives", sw, flush=True)


def greedy(m, source, selected, capped):
    if not selected:
        return dict(total=0, tested=0, observations={})
    arm = {"early": Native.arm(source)}
    eos = m.model.generation_config.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos or [])
    active = []
    for row, token in selected[:capped]:
        first = row["token"]["slots"][0]
        prompt = row["token"]["ids"][:first]
        draft = [token, *row["token"]["ids"][first + 1:]][:C.ROLLOUT_TOKENS]
        active.append(dict(row=row, choice=token, prompt=prompt, draft=draft, accepted=[], attempts=[]))
    results = {}

    def rendered(sample, tokens):
        start = sample["row"]["token"]["start"]
        return m.tok.decode([*sample["prompt"][start:], *tokens])

    def save(sample, reason, answer=None):
        row, accepted = sample["row"], sample["accepted"]
        glyph = None if answer is None else m.tok.decode([answer])
        results[row["id"]] = dict(id=row["id"], stop=reason, verified=reason == "code", code_token=answer,
            code=glyph, state_token=sample["choice"], state=m.tok.decode([sample["choice"]]),
            recognized=sample["choice"] == row["state_ids"][1], expected=row["code"],
            emitted_tokens=accepted, emitted_text=rendered(sample, accepted), attempts=sample["attempts"])

    for iteration in range(C.ROLLOUT_PASSES):
        if not active:
            break
        payload = [dict(x["row"], token=dict(x["row"]["token"], ids=x["prompt"] + x["draft"])) for x in active]
        predictions = Native.panel(m, payload, arm, "rollout")
        pending = []
        for x, predicted in zip(active, predictions, strict=True):
            prior = len(x["accepted"])
            conflict = next((j for j, (u, v) in enumerate(zip(x["accepted"], predicted)) if u != v), -1)
            if conflict >= 0 or predicted[0] != x["choice"]:
                x["attempts"].append(dict(pass_id=iteration, replay_conflict=max(0, conflict), actual=predicted[max(0, conflict)]))
                save(x, "replay-conflict")
                continue
            for j in range(prior, len(predicted)):
                token = predicted[j]
                code_ready = bool(re.search(r"(?:^|\n)Code:[ \t]*$", rendered(x, x["accepted"])))
                x["accepted"].append(token)
                if code_ready or token in stops or len(x["accepted"]) >= C.ROLLOUT_TOKENS:
                    x["attempts"].append(dict(pass_id=iteration, accepted=len(x["accepted"]), first_change=j))
                    save(x, "code" if code_ready else "eos" if token in stops else "token-budget", token if code_ready else None)
                    break
                if j >= len(x["draft"]) or token != x["draft"][j]:
                    x["draft"] = x["accepted"].copy() + x["draft"][j + 1:]
                    x["attempts"].append(dict(pass_id=iteration, accepted=j + 1, first_change=j, actual=token))
                    pending.append(x)
                    break
        active = pending
    for x in active:
        save(x, "pass-budget")
    return dict(total=len(selected), tested=min(capped, len(selected)), observations=results)


def confirm(m, p):
    f = A.load("freeze.json")
    assert f["plan"] == B.sha("plan.json") and f["behavior"] == B.sha("discover-behavior.json")
    assert all(B.sha(k) == v for k, v in f["discovery"].items())
    rows = A.load(p["materials"]["confirm"]["file"])
    bind = B.sha("freeze.json")
    b = behavior(m, rows, "confirm", bind)
    identity = dict(behavior=B.sha("confirm-behavior.json"), freeze=bind)
    if not b["passed"]:
        A.save("confirm.json", dict(state="behavior-gate-failed", **identity))
        return
    use = C.O.resolved(rows, b)
    known = {r["id"]: r for r in b["rows"]}
    config = configurations(f)
    assert list(config) == f["arms"]
    nsteps = (len(use) + C.SLAB - 1) // C.SLAB
    start = time.monotonic()
    C.N.progress("holdout", 0, nsteps, start)
    files, checks, accepted = {}, {}, []
    switched = {k: [] for k in ("early", "state-route", "early-rand1", "early-rand2", "state-rand1", "state-rand2")}
    for j, part in enumerate(batched(use, C.SLAB), 1):
        ad = f"confirm-adjoint-{j:03d}.npz"
        adj = persist(ad, part, lambda: Native.adjoint(m, part))
        d = directions(adj)
        name = f"confirm-intervention-{j:03d}.npz"
        z = persist(name, part, lambda: Native.panel(m, part, config, "confirm-native", known, d))
        checks[name] = dict(**physical(z, config), adjoint_error=float(adj["error"].max()),
                            invalid_views=z["ids"][~z["technical"]].tolist())
        good = np.repeat(z["technical"].reshape(-1, 8).all(-1), 8)
        accepted.extend(z["ids"][good].tolist())
        for arm, cases in switched.items():
            hit = z["state_top"][:, f["arms"].index(arm)]
            cases.extend((r, int(token)) for r, token, ok in zip(part, hit, good, strict=True) if ok and token != r["state_ids"][0])
        files.update({k: B.sha(k) for k in (ad, name)})
        C.N.progress("holdout", j, nsteps, start)
    roll = {}
    identities = {}
    for a, trials in switched.items():
        patch_id = tuple(tuple(x) for x in config[a]["early"])
        if patch_id in identities:
            other = identities[patch_id]
            assert [(r["id"], t) for r, t in trials] == [(r["id"], t) for r, t in switched[other]]
            roll[a] = {**roll[other], "shared_with": other}
        else:
            identities[patch_id] = a
            filename = f"rollout-{a}.json"
            if not (C.OUT / filename).exists():
                A.save(filename, greedy(m, config[a]["early"], trials, C.ROLLOUT_CASES))
            roll[a] = A.load(filename)
            files[filename] = B.sha(filename)
    A.save("rollout.json", roll)
    A.save("confirm.json", dict(state="finished", files=files, checks=checks, eligible=len(use) // 8,
                                accepted=accepted, rollout=B.sha("rollout.json"), **identity))
