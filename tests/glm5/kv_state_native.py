"""Direct G-source editing and finite native residual transplants at E."""

from itertools import batched
import time

import numpy as np

import kv_state_run as C
import kv_prefix_native as Base


F = Base.F


def arm(g=(), cache=None, own=False, e=None):
    return dict(g=list(g), patch=cache, self_restore=own, e=e)


def indices(m, p):
    masks = Base.site_mask(m, p)
    return {"fact": masks[:, :3].any(1), "book": masks[:, 3]}


def swaps(p):
    keys = {(r["family"], r["side"], r["distance"], r["order"]): i for i, r in enumerate(p["rows"])}
    return [keys[r["family"], 1 - r["side"], r["distance"], r["order"]] for r in p["rows"]]


def shortened(rows):
    return [dict(r, token=dict(r["token"], ids=r["token"]["ids"][:r["token"]["slots"][0]])) for r in rows]


def g_screen(m, layer, p, x, q, k, v, mask):
    t = m.t
    r = t.arange(len(x), device="cuda")
    g = p["slots"][:, 0].cuda()
    shown = mask[r, 0, g][:, None, None]
    a = m.read(q[r, :, g][:, :, None], k, shown)
    selects = t.stack(list(indices(m, p).values()), 1)
    alpha = a[:, None] * selects[:, :, None]
    terms = t.einsum("brhs,bhsd->brhd", alpha, v.repeat_interleave(m.groups, dim=1).float())
    residual = m.projections(layer, tuple(range(m.heads)), terms)
    w = m.model.lm_head.weight.detach()[m.letters].float()
    d = (w[1] - w[0]) * m.decoder.norm.weight.detach().float()
    sign = t.as_tensor([1 if z["code"] == "Q" else -1 for z in p["rows"]], device="cuda")
    scale = (x[r, g].float().square().mean(-1) + 1e-6).sqrt()
    effect = t.einsum("brhd,d->brh", residual, d) * (sign / scale)[:, None, None]
    p.setdefault("screen", []).append(F(t.stack((effect, residual.norm(dim=-1), alpha.sum(-1)), -1)))


def e_adjoint(m, rows):
    t = m.t
    out = []
    for j, slab in enumerate(batched(shortened(rows), C.SLAB), 1):
        packets = [m.packet(r) for r in batched(slab, 8)]
        clock = time.monotonic()
        steps = 40 + 39 - min(C.E_LAYERS)
        C.N.progress("state-adjoint", 0, steps, clock, part=j)
        for i in range(40):
            with m.block(i) as layer:
                for p in packets:
                    x = p["x"].cuda()
                    angles = tuple(z.cuda() for z in p["cs"])
                    vis = p["mask"].cuda()
                    q, k, v = m.qkv(layer, x, angles)
                    o = m.attention(layer, q, k, v, vis)
                    p["x"] = m.residual(layer, x, o).cpu()
                    if i in C.E_LAYERS:
                        p.setdefault("e_native", {})[i] = p["x"][:, -1].clone()
                    if i > min(C.E_LAYERS):
                        p.setdefault("tape", {})[i] = (x[:, -1].cpu(), k.cpu(), v.cpu(), p["x"][:, -1].clone())
            C.N.progress("state-adjoint", i + 1, steps, clock, part=j)
        with t.enable_grad():
            for p in packets:
                x = p["x"][:, -1].cuda().requires_grad_(True)
                rows_ = t.arange(len(x), device="cuda")
                choices = t.as_tensor([r["state_ids"] for r in p["rows"]], device="cuda")
                logits = m.logits(x)[rows_[:, None], choices].float()
                p["gradient"] = t.autograd.grad((logits[:, 1] - logits[:, 0]).sum(), x)[0].cpu()
                p["d"] = {39: p["gradient"]}
                p["err"] = np.zeros(len(x), dtype=np.float32)
        for i in range(39, min(C.E_LAYERS), -1):
            with m.block(i) as layer, t.enable_grad():
                for p in packets:
                    x, k, v, native = (z.cuda() for z in p["tape"].pop(i))
                    x.requires_grad_(True)
                    angles = tuple(z[:, -1:].cuda() for z in p["cs"])
                    vis = p["mask"][:, :, -1:].cuda()
                    q, kk, vv = m.branch_qkv(layer, x, k, v, angles)
                    o = m.attention(layer, q, kk, vv, vis)[:, 0]
                    y = m.residual(layer, x, o)
                    p["gradient"] = t.autograd.grad(y, x, p["gradient"].cuda())[0].cpu()
                    err = (y.float() - native.float()).norm(dim=-1) / native.float().norm(dim=-1).clamp_min(1e-4)
                    p["err"] = np.maximum(p["err"], F(err))
                    if i - 1 in C.E_LAYERS:
                        p["d"][i - 1] = p["gradient"]
            C.N.progress("state-adjoint", 40 + 40 - i, steps, clock, part=j)
        for p in packets:
            pair = swaps(p)
            deltas = [F(p["e_native"][i][pair]) - F(p["e_native"][i]) for i in C.E_LAYERS]
            grads = [F(p["d"][i]) for i in C.E_LAYERS]
            out.append(dict(ids=np.array([r["id"] for r in p["rows"]]), error=p["err"],
                            product=np.stack([d*g for d, g in zip(deltas, grads, strict=True)], 1),
                            displacement=np.abs(np.stack(deltas, 1)), delta_l2=np.linalg.norm(np.stack(deltas, 1), axis=-1)))
    return {k: np.concatenate([r[k] for r in out]) for k in out[0]}


def edit_e(m, p, i, output, native, group, donor):
    t = m.t
    r = t.arange(len(native), device="cuda")
    e = p["e"].cuda()
    loc = e - p["pivot"]
    actual = native[r, e]
    if donor is not None:
        alternate = t.as_tensor(np.stack([donor[i][row["id"]] for row in p["rows"]]), device="cuda", dtype=actual.dtype)
    else:
        alternate = actual[swaps(p)]
    if "e_saved" in p:
        p["e_saved"][f"donor_e{i}"] = F(alternate)
    for j, spec in group:
        op = spec["kind"]
        pick = spec["coordinates"]
        before = output[j, r, loc].clone()
        new = before.clone()
        if op == "swap-all":
            new = alternate.clone()
        elif op == "self":
            new[:, pick] = 0
            new[:, pick] = before[:, pick]
            assert t.equal(new, before)
        else:
            new[:, pick] = alternate[:, pick] if op == "swap" else 0
        output[j, r, loc] = new
        assert t.equal(output[j, r, loc], new)
        p["e_dose"][:, j] = F((new.float() - before.float()).norm(dim=-1))


def step(m, layer, i, p, arms, mode, direction, e_donor):
    t = m.t
    if i == 0:
        p["e"] = p["slots"][:, 0] - 1
        p["pivot"] = int(p["e"].min())
        p["early"] = min([25 if mode == "run" else 39,
                          *[l for a in arms.values() for l, h, key in a["g"]],
                          *[a["e"]["layer"] for a in arms.values() if a["e"] is not None]])
        p["local_error"] = np.zeros(len(arms), dtype=np.float32)
        p["e_dose"] = np.zeros((len(p["rows"]), len(arms)), dtype=np.float32)
        if mode == "run":
            p["e_saved"] = {}
    x = p["x"].cuda()
    cs = tuple(z.cuda() for z in p["cs"])
    mask = p["mask"].cuda()
    q0, k0, v0 = m.qkv(layer, x, cs)
    o0 = m.attention(layer, q0, k0, v0, mask)
    if mode == "screen":
        g_screen(m, layer, p, x, q0, k0, v0, mask)
    native = m.residual(layer, x, o0)
    p["x"] = native.cpu()
    if mode == "screen" or i < p["early"]:
        return
    a, b, pivot = len(arms), len(x), p["pivot"]
    vis = mask[:, :, pivot:]
    live = p["branches"].cuda() if "branches" in p else x[None, :, pivot:].expand(a, -1, -1, -1)
    angles = tuple(z[:, pivot:].repeat(a, 1, 1) for z in cs)
    q, k, v = Base.branch_qkv(m, layer, live, k0, v0, angles, pivot)
    raw = m.attention(layer, q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), vis.repeat(a, 1, 1, 1))
    z = raw.reshape(a, b, live.shape[2], m.heads, 128)
    g_edits = [(j, h, source, a0["self_restore"]) for j, a0 in enumerate(arms.values())
               for l, h, source in a0["g"] if l == i]
    if g_edits:
        rows = t.arange(b, device="cuda")
        g = p["slots"][:, 0].cuda() - pivot
        shown = vis[rows, 0, g][:, None, None]
        masks = indices(m, p)
        for j, h, key, restore in g_edits:
            edit, error = Base.early_edit(m, layer, h, q[j, rows, :, g][:, :, None], k[j], v[j], shown,
                                          masks[key], z[j, rows, g, h], restore)
            z[j, rows, g, h] = edit
            p["local_error"][j] = max(p["local_error"][j], error)
    if mode == "run" and i == 25:
        d = t.as_tensor(np.stack([direction[r["id"]] for r in p["rows"]]), device="cuda")
        p["local"] = Base.coordinate_read(m, layer, p, q, k, v, vis, d)
        Base.coordinate_edit(m, layer, p, q, k, v, vis, z, arms, d)
        # These intermediate scoring arrays were used in Phase 2949 to select coordinates.
        del p["local"]["dose"], p["local"]["head_model"]
    out = m.residual(layer, live, z)
    e_edits = [(j, a0["e"]) for j, a0 in enumerate(arms.values()) if a0["e"] is not None and a0["e"]["layer"] == i]
    if e_edits:
        edit_e(m, p, i, out, native, e_edits, e_donor)
    p["branches"] = out.cpu()


def finish(m, p, arms, mode, known):
    ids = np.array([r["id"] for r in p["rows"]])
    if mode == "screen":
        return dict(ids=ids, screen=np.stack(p["screen"], axis=1))
    if mode == "rollout":
        return Base.finish(m, p, arms, "rollout", None)
    if mode == "e-search":
        t = m.t
        options = t.as_tensor([r["state_ids"] for r in p["rows"]], device="cuda")
        row = t.arange(len(ids), device="cuda")
        logits = t.stack([m.logits(x[:, -1].cuda()) for x in p["branches"]])
        states = F(logits[:, row[:, None], options]).swapaxes(0, 1)
        top = logits.argmax(-1).cpu().numpy().T
        return dict(ids=ids, state=states, state_top=top, e_dose=p["e_dose"])
    return dict(**Base.finish(m, p, arms, "run", known), e_dose=p["e_dose"], **p["e_saved"])


def panel(m, rows, arms, mode, known=None, direction=None, e_donor=None):
    result = []
    nlayer = 25 if mode == "screen" else 40
    stage = {"screen": "g-screen", "e-search": "state-search", "run": "direct-state", "rollout": "true-state-rollout"}[mode]
    if mode == "e-search":
        rows = shortened(rows)
    for j, slab in enumerate(batched(rows, C.SLAB), 1):
        clock = time.monotonic()
        packets = [m.packet(r) for r in batched(slab, 8)]
        nslabs = (len(rows) + C.SLAB - 1) // C.SLAB
        C.N.progress(stage, 0, nlayer, clock, part=j, parts=nslabs)
        for i in range(nlayer):
            with m.block(i) as layer:
                for p in packets:
                    step(m, layer, i, p, arms, mode, direction, e_donor)
            C.N.progress(stage, i + 1, nlayer, clock, part=j, parts=nslabs)
        result.extend(finish(m, p, arms, mode, known) for p in packets)
    if mode == "rollout":
        return [x for packet in result for x in packet]
    return {k: np.concatenate([z[k] for z in result]) for k in result[0]}
