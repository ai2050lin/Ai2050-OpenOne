"""Phase 2949 native attention edits, streamed Code adjoint, and coordinate allocation."""

from itertools import batched
import time

import numpy as np

import kv_prefix_run as C
from kv_temporal_native import branch_qkv, F


def arm(early=(), patch=None, self_restore=False):
    return dict(early=list(early), patch=patch, self_restore=self_restore)


def site_mask(m, p):
    out = np.zeros((len(p["rows"]), len(C.SOURCES), p["ids"].shape[1]), dtype=bool)
    for j, (r, pad) in enumerate(zip(p["rows"], p["padding"], strict=True)):
        for k, label in enumerate(C.SOURCES):
            locations = r["sites"][label]
            assert max(locations) < p["slots"][j, 0] - pad
            out[j, k, [i + pad for i in locations]] = True
    return m.t.as_tensor(out, device="cuda")


def prepare(m, p, arms):
    p["e"] = p["slots"][:, 0] - 1
    p["pivot"] = int(p["e"].min())
    p["earliest"] = min([25, *[s[0] for a in arms.values() for s in a["early"]]])
    p["local_error"] = np.zeros(len(arms), dtype=np.float32)


def screen_step(m, l, i, p, x, q, k, v, vis):
    t = m.t
    at = t.arange(len(p["rows"]), device="cuda")
    query = p["slots"][:, 0].cuda() - 1
    q_e = q[at, :, query][:, :, None]
    mask = vis[at, 0, query][:, None, None]
    a = m.read(q_e, k, mask)
    weighted = a[:, None] * site_mask(m, p)[:, :, None]
    terms = t.einsum("brhs,bhsd->brhd", weighted, v.repeat_interleave(m.groups, dim=1).float())
    writes = m.projections(l, tuple(range(m.heads)), terms)
    states = t.tensor([r["state_ids"] for r in p["rows"]], device="cuda")
    bank = m.model.lm_head.weight.detach()
    d = (bank[states[:, 0]].float() - bank[states[:, 1]].float()) * m.decoder.norm.weight.detach().float()
    rms = (x[at, query].float().square().mean(-1) + 1e-6).sqrt()
    score = t.einsum("brhd,bd->brh", writes, d) / rms[:, None, None]
    p.setdefault("screen", []).append(F(t.stack((score, writes.norm(dim=-1), weighted.sum(-1)), -1)))


def early_edit(m, l, head, q, k, v, mask, source, before, restore):
    t = m.t
    group = head // m.groups
    vv = v.clone()
    vv[:, group].masked_fill_(source[..., None], 0)
    if restore:
        vv[:, group] = t.where(source[..., None], v[:, group], vv[:, group])
        assert t.equal(vv, v)
    else:
        assert not bool(vv[:, group][source].any())
    z0 = m.foreground(l, head, q, k, v, mask)
    z1 = m.foreground(l, head, q, k, vv, mask)
    actual = (before.float() + (z1.float() - z0.float())).to(before.dtype)
    if restore:
        assert t.equal(before, actual)
    alpha = m.read(q, k, mask)[:, head]
    expected = (alpha[..., None] * (vv[:, group].float() - v[:, group].float())).sum(-2)
    err = (actual.float() - before.float() - expected).norm(dim=-1)
    # Absolute error grows with the native write; this scale is recorded for every arm.
    return actual, float((err / (1 + expected.norm(dim=-1))).max())


def adjoint(m, rows):
    """Differentiate Code only. Stream each block again with its untouched causal prefix."""
    t = m.t
    outputs = []
    for j, part in enumerate(batched(rows, C.SLAB), 1):
        packets = [m.packet(batch) for batch in batched(part, 8)]
        clock = time.monotonic()
        C.N.progress("code-adjoint", 0, 55, clock, part=j, parts=(len(rows) + C.SLAB - 1) // C.SLAB)
        for i in range(40):
            with m.block(i) as l:
                for p in packets:
                    x = p["x"].cuda()
                    angles = tuple(u.cuda() for u in p["cs"])
                    mask = p["mask"].cuda()
                    q, k, v = m.qkv(l, x, angles)
                    o = m.attention(l, q, k, v, mask)
                    p["x"] = m.residual(l, x, o).cpu()
                    if i == 25:
                        p["anchor"] = (x[:, -1].cpu(), o[:, -1].cpu())
                    elif i > 25:
                        p.setdefault("tape", {})[i] = (x[:, -1].cpu(), k.cpu(), v.cpu(), p["x"][:, -1].clone())
            C.N.progress("code-adjoint", i + 1, 55, clock, part=j)
        with t.enable_grad():
            for p in packets:
                x = p["x"][:, -1].cuda().requires_grad_(True)
                w = m.model.lm_head.weight.detach()[m.letters]
                lm = t.nn.functional.linear(m.decoder.norm(x), w).float()
                p["grad"] = t.autograd.grad((lm[:, 1] - lm[:, 0]).sum(), x)[0].cpu()
                p["adj_error"] = np.zeros(len(x), dtype=np.float32)
        for i in range(39, 24, -1):
            with m.block(i) as l, t.enable_grad():
                for p in packets:
                    d = p["grad"].cuda()
                    if i == 25:
                        x, o = (s.cuda() for s in p.pop("anchor"))
                        o.requires_grad_(True)
                        y = m.residual(l, x, o)
                        p["direction"] = F(t.autograd.grad(y, o, d)[0][:, 18])
                    else:
                        x, k, v, actual = (s.cuda() for s in p["tape"].pop(i))
                        x.requires_grad_(True)
                        angles = tuple(z[:, -1:].cuda() for z in p["cs"])
                        vis = p["mask"][:, :, -1:].cuda()
                        q, kk, vv = m.branch_qkv(l, x, k, v, angles)
                        o = m.attention(l, q, kk, vv, vis)[:, 0]
                        y = m.residual(l, x, o)
                        p["grad"] = t.autograd.grad(y, x, d)[0].cpu()
                        err = (y.float() - actual.float()).norm(dim=-1) / actual.float().norm(dim=-1).clamp_min(1e-4)
                        p["adj_error"] = np.maximum(p["adj_error"], F(err))
            C.N.progress("code-adjoint", 40 + 40 - i, 55, clock, part=j)
        outputs.extend(dict(ids=np.array([r["id"] for r in p["rows"]]), d=p["direction"],
                            error=p["adj_error"]) for p in packets)
    return {k: np.concatenate([p[k] for p in outputs]) for k in outputs[0]}


def coordinate_read(m, l, p, q, k, v, mask, direction):
    """Exact head-output decomposition with fixed live Q and non-G scores/values."""
    t = m.t
    group = 18 // m.groups
    r = t.arange(len(p["rows"]), device="cuda")
    slot = p["slots"][:, 0].cuda()
    key, value = k[:, r, group, slot], v[:, r, group, slot]
    qq = q[:, :, 18, -1].float()
    bank = v[:, :, group].float()
    score = t.einsum("abd,absd->abs", qq, k[:, :, group].float()) * 128**-.5
    rest = mask[None, :, 0, -1].expand_as(score).clone()
    rest[:, r, slot] = False
    other = score.masked_fill(~rest, -t.inf)
    anchor = other.logsumexp(-1)
    u = t.einsum("abs,absd->abd", other.softmax(-1), bank)
    dk, dv = key[:1].float() - key.float(), value[:1].float() - value.float()
    pieces = qq * dk * 128**-.5
    ds = pieces.sum(-1)
    sk = (qq * key.float()).sum(-1) * 128**-.5
    before = (sk - anchor).sigmoid()
    after = (sk + ds - anchor).sigmoid()
    secant = t.where(ds.abs() > 1e-5, (after - before) / ds, before * (1 - before))
    midpoint = (value[:1].float() + value.float()) / 2
    effect_k = pieces * secant[..., None] * ((midpoint - u) * direction[None]).sum(-1)[..., None]
    effect_v = (before + after)[..., None] / 2 * dv * direction[None]
    o0 = u + before[..., None] * (value.float() - u)
    o1 = u + after[..., None] * (value[:1].float() - u)
    total = ((o1 - o0) * direction[None]).sum(-1)
    residual = (effect_k.sum(-1) + effect_v.sum(-1) - total).abs()
    assert bool((residual < 2e-4 * (1 + total.abs())).all())
    wo = l.self_attn.o_proj.weight.reshape(m.width, m.heads, 128)[:, 18].float().norm(dim=0)
    dose = t.stack((pieces.abs(), (qq * key[:1].float()).abs() * 128**-.5, dv.abs() * wo,
                    value[:1].float().abs().expand_as(dv) * wo), dim=-2)
    return dict(coord=F(t.stack((effect_k, effect_v), -2)).swapaxes(0, 1),
                dose=F(dose).swapaxes(0, 1), local_total=F(total).T, local_closure=F(residual).T,
                k_native=F(key).swapaxes(0, 1), v_native=F(value).swapaxes(0, 1), q_native=F(qq).swapaxes(0, 1),
                a_native=F(before).T, head_model=F(o0).swapaxes(0, 1))


def coordinate_edit(m, l, p, q, k, v, vis, z, arms, direction):
    t = m.t
    r = t.arange(len(p["rows"]), device="cuda")
    gs = p["slots"][:, 0].cuda()
    group = 18 // m.groups
    q = q[:, :, :, -1:]
    mask = vis[:, :, -1:]
    edits_k, edits_v, edited_head, masses = [], [], [], []
    for j, spec in enumerate(arms.values()):
        kk, vv = k[j].clone(), v[j].clone()
        op = spec["patch"]
        if op is not None:
            donor, kidx, vidx = op
            for array, native, indices in ((kk, k, kidx), (vv, v, vidx)):
                if not indices:
                    continue
                dest = array[r, group, gs].clone()
                expected = dest.clone()
                expected[:, indices] = 0 if donor < 0 else native[donor, r, group, gs][:, indices]
                if spec["self_restore"]:
                    expected[:, indices] = dest[:, indices]
                array[r, group, gs] = expected
                assert t.equal(array[r, group, gs], expected)
            baseline = m.foreground(l, 18, q[j], k[j], v[j], mask)
            change = m.foreground(l, 18, q[j], kk, vv, mask).float() - baseline.float()
            old = z[j, :, -1, 18].clone()
            z[j, :, -1, 18] = (old.float() + change).to(z.dtype)
            if spec["self_restore"]:
                assert t.equal(z[j, :, -1, 18], old)
            def analytic(key, value):
                a = m.read(q[j], key, mask)[:, 18]
                return t.einsum("bs,bsd->bd", a, value[:, group].float())
            exact = analytic(kk, vv) - analytic(k[j], v[j])
            error = (z[j, :, -1, 18].float() - old.float() - exact).norm(dim=-1) / (1 + exact.norm(dim=-1))
            p["local_error"][j] = max(p["local_error"][j], float(error.max()))
        edits_k.append(kk[r, group, gs])
        edits_v.append(vv[r, group, gs])
        score = (q[j, :, 18].float() * kk[:, group].float()).sum(-1) * 128**-.5
        masses.append(score.masked_fill(~mask[:, 0, 0], -t.inf).softmax(-1)[r, gs])
        edited_head.append(z[j, :, -1, 18])
    p["local"].update(k_applied=F(t.stack(edits_k, 1)), v_applied=F(t.stack(edits_v, 1)),
                      head=F(t.stack(edited_head, 1)), a_applied=F(t.stack(masses, 1)))
    if direction is not None:
        out = t.stack(edited_head, 1).float()
        p["local"]["head_output_directed"] = F(((out - out[:, :1]) * direction[:, None]).sum(-1))


def step(m, l, i, p, arms, mode, directions):
    t = m.t
    if i == 0:
        prepare(m, p, arms)
    x = p["x"].cuda()
    cs = tuple(z.cuda() for z in p["cs"])
    vis = p["mask"].cuda()
    q0, k0, v0 = m.qkv(l, x, cs)
    o0 = m.attention(l, q0, k0, v0, vis)
    if mode == "screen":
        screen_step(m, l, i, p, x, q0, k0, v0, vis)
    p["x"] = m.residual(l, x, o0).cpu()
    if mode == "screen" or i < p["earliest"]:
        return
    a, b, pivot = len(arms), len(x), p["pivot"]
    live = p["branches"].cuda() if "branches" in p else x[None, :, pivot:].expand(a, -1, -1, -1)
    vis = vis[:, :, pivot:]
    angles = tuple(z[:, pivot:].repeat(a, 1, 1) for z in cs)
    q, k, v = branch_qkv(m, l, live, k0, v0, angles, pivot)
    raw = m.attention(l, q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), vis.repeat(a, 1, 1, 1))
    z = raw.reshape(a, b, live.shape[2], m.heads, 128)
    here = [(j, h, source, spec["self_restore"]) for j, spec in enumerate(arms.values())
            for layer, h, source in spec["early"] if layer == i]
    if here:
        rows = t.arange(b, device="cuda")
        e = p["e"].cuda() - pivot
        masks = site_mask(m, p)
        shown = vis[rows, 0, e][:, None, None]
        for j, h, source, restore in here:
            work, err = early_edit(m, l, h, q[j, rows, :, e][:, :, None], k[j], v[j], shown,
                                   masks[:, C.SOURCES.index(source)], z[j, rows, e, h], restore)
            z[j, rows, e, h] = work
            p["local_error"][j] = max(p["local_error"][j], err)
    if i == 25 and mode != "rollout":
        d = t.as_tensor(np.stack([directions[r["id"]] for r in p["rows"]]), device="cuda")
        p["local"] = coordinate_read(m, l, p, q, k, v, vis, d)
        coordinate_edit(m, l, p, q, k, v, vis, z, arms, d)
    p["branches"] = m.residual(l, live, z).cpu()


def finish(m, p, arms, mode, known):
    rows = p["rows"]
    out = dict(ids=np.array([r["id"] for r in rows]))
    if mode == "screen":
        out["screen"] = np.stack(p["screen"], axis=1)
        return out
    start = (p["e"] - p["pivot"]).numpy()
    if mode == "rollout":
        best = m.logits(p["branches"][0].cuda()).argmax(-1).cpu().numpy()
        return [best[r, at:].tolist() for r, at in enumerate(start)]
    t = m.t
    data = p["ids"].numpy()
    at = t.arange(len(rows), device="cuda")
    e = t.as_tensor(start, device="cuda")
    options = t.tensor([r["state_ids"] for r in rows], device="cuda")
    n, a = len(rows), len(arms)
    pq, state = np.empty((n, a, 2), np.float32), np.empty((n, a, 2), np.float32)
    code_top, state_top, mismatch, emitted = (np.empty((n, a), dtype=np.int32) for _ in range(4))
    for j, act in enumerate(p["branches"]):
        logits = m.logits(act.cuda())
        best = logits.argmax(-1).cpu().numpy()
        pq[:, j] = F(logits[:, -1, m.letters])
        state[:, j] = F(logits[at[:, None], e[:, None], options])
        code_top[:, j] = best[:, -1]
        state_top[:, j] = best[np.arange(n), start]
        for ii, pos in enumerate(start):
            bad = np.flatnonzero(best[ii, pos:-1] != data[ii, p["pivot"] + pos + 1:])
            mismatch[ii, j] = int(bad[0]) if len(bad) else -1
            emitted[ii, j] = int(best[ii, pos + bad[0]]) if len(bad) else -1
    native = np.array([known[r["id"]]["pq"] for r in rows])
    numeric_diff = np.abs(native - pq[:, 0]).max(1)
    check = (mismatch[:, 0] == -1) & (code_top[:, 0] == [known[r["id"]]["top_token"] for r in rows])
    return dict(**out, pq=pq, code_top=code_top, state=state, state_top=state_top, first_mismatch=mismatch,
                first_emitted=emitted, technical=check & (numeric_diff <= C.N.SHAM_TOL), numeric_diff=numeric_diff,
                local_error=np.broadcast_to(p["local_error"], (n, a)).copy(), **p["local"])


def panel(m, rows, arms, mode, known=None, directions=None):
    results = []
    nlayer = 24 if mode == "screen" else 40
    for j, slab in enumerate(batched(rows, C.SLAB), 1):
        clock = time.monotonic()
        packets = [m.packet(group) for group in batched(slab, 8)]
        steps = (len(rows) + C.SLAB - 1) // C.SLAB
        C.N.progress(mode, 0, nlayer, clock, part=j, parts=steps)
        for i in range(nlayer):
            with m.block(i) as l:
                for p in packets:
                    step(m, l, i, p, arms, mode, directions)
            C.N.progress(mode, i + 1, nlayer, clock, part=j, parts=steps)
        results.extend(finish(m, p, arms, mode, known) for p in packets)
    if mode == "rollout":
        return [x for part in results for x in part]
    return {k: np.concatenate([p[k] for p in results]) for k in results[0]}
