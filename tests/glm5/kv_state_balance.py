"""Prospectively recorded equal-write-dose companion for the direct G ablations."""

from itertools import batched
from pathlib import Path
import time

import numpy as np

import kv_prefix_native as Base
import kv_state_native as Native
import kv_state_run as C
from kv_origin_audit import fam, summarize
from kv_state_study import code_margin, load, persist, sign


A, B = C.A, C.B
FAMILIES = 88


def one_change(m, l, head, q, k, v, mask, selected):
    vv = v.clone()
    vv[:, head // m.groups].masked_fill_(selected[..., None], 0)
    after = m.foreground(l, head, q, k, vv, mask).float()
    before = m.foreground(l, head, q, k, v, mask).float()
    return after - before


def dose(m, layer, head, delta):
    return m.projections(layer, (head,), delta[:, None]).norm(dim=-1)[:, 0]


def exact_pair(m, p, layer, mask, q, k, v, z, start, heads, source, key):
    t = m.t
    row = t.arange(len(p["rows"]), device="cuda")
    g = p["slots"][:, 0].cuda() - p["pivot"]
    selected = Native.indices(m, p)[source]
    q0 = q[0, row, :, g][:, :, None]
    at_g = mask[row, 0, g][:, None, None]
    vectors = [one_change(m, layer, h, q0, k[0], v[0], at_g, selected) for h in heads]
    original = [dose(m, layer, h, w) for h, w in zip(heads, vectors, strict=True)]
    common = t.minimum(*original)
    applied = []
    for branch, h, w, d in zip(range(start, start+2), heads, vectors, original, strict=True):
        gain = t.where(d > 0, common/d, t.zeros_like(d))
        before = z[branch, row, g, h].clone()
        assert t.equal(before, z[0, row, g, h])
        after = (before.float() + gain[:, None] * w).to(before.dtype)
        z[branch, row, g, h] = after
        applied.append(dose(m, layer, h, after.float() - before.float()))
    p["dose"][key] = Base.F(t.stack((*original, common, *applied), -1))


def feature_pairs(f):
    return [(f"{j}-{k}", g, r, 1+4*j+2*k) for j, g in enumerate(f["g"])
            for k, r in enumerate(x[j] for x in f["g_random"])]


def compute(m, rows, known, directions, f):
    t = m.t
    clock = time.monotonic()
    packets = [m.packet(u) for u in batched(rows, 8)]
    pairs = feature_pairs(f)
    start = min(z[1][0] for z in pairs)
    arms = dict.fromkeys(range(9))
    C.N.progress("dose-balance", 0, 40, clock)
    for p in packets:
        p["e"] = p["slots"][:, 0] - 1
        p["pivot"] = int(p["e"].min())
        p["local_error"] = np.zeros(len(arms), dtype=np.float32)
        p["dose"] = {}
    for i in range(40):
        with m.block(i) as layer:
            for p in packets:
                x = p["x"].cuda()
                mask = p["mask"].cuda()
                cs = tuple(w.cuda() for w in p["cs"])
                q0, k0, v0 = m.qkv(layer, x, cs)
                o0 = m.attention(layer, q0, k0, v0, mask)
                p["x"] = m.residual(layer, x, o0).cpu()
                if i < start:
                    continue
                a, b, pivot = len(arms), len(x), p["pivot"]
                live = p["branches"].cuda() if "branches" in p else x[None, :, pivot:].expand(a, -1, -1, -1)
                vis = mask[:, :, pivot:]
                angles = tuple(z[:, pivot:].repeat(a, 1, 1) for z in cs)
                q, k, v = Base.branch_qkv(m, layer, live, k0, v0, angles, pivot)
                raw = m.attention(layer, q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), vis.repeat(a, 1, 1, 1))
                z = raw.reshape(a, b, live.shape[2], m.heads, 128)
                for key, (l, h, source), (l2, hc, source2), first in pairs:
                    assert (l, source) == (l2, source2)
                    if i == l:
                        exact_pair(m, p, layer, vis, q, k, v, z, first, (h, hc), source, key)
                if i == 25:
                    d = t.as_tensor(np.stack([directions[r["id"]] for r in p["rows"]]), device="cuda")
                    read = Base.coordinate_read(m, layer, p, q, k, v, vis, d)
                    p["local"] = dict(local_total=read["local_total"], local_closure=read["local_closure"])
                p["branches"] = m.residual(layer, live, z).cpu()
        C.N.progress("dose-balance", i + 1, 40, clock)
    out = []
    for p in packets:
        z = Base.finish(m, p, arms, "run", known)
        assert np.array_equal(z["state_top"], np.broadcast_to(z["state_top"][:, :1], z["state_top"].shape))
        assert np.array_equal(z["state"], np.broadcast_to(z["state"][:, :1], z["state"].shape))
        z.update(dose=np.stack([p["dose"][k] for k, *_ in pairs], 1))
        out.append(z)
    return {k: np.concatenate([p[k] for p in out]) for k in out[0]}


def aggregate(rows, z, spec):
    m = code_margin(z, rows).astype(np.float64)
    local = z["local_total"].astype(np.float64) * sign(rows)[:, None]
    takes = np.repeat(z["technical"].reshape(-1, 8).all(-1), 8)
    if not takes.any():
        return dict(n=0, state="technical-gate-failed", invalid=z["ids"][~z["technical"]].tolist())
    scopes = {"complete": takes, "same_word_and_suffix": takes & np.repeat((z["first_mismatch"] == -1).all(-1).reshape(-1, 8).all(-1), 8)}
    quantities = {}
    for key, take in scopes.items():
        if not take.any():
            continue
        estimates = {}
        for j, kind in enumerate(C.G_SOURCES):
            lo = 1 + j * 4
            c, h = m[take, lo:lo+4], local[take, lo:lo+4]
            for dst, measure in ((f"{kind}-code", ((c[:, 1] - c[:, 0]) + (c[:, 3] - c[:, 2])) / 2),
                                 (f"{kind}-directed", ((h[:, 0] - h[:, 1]) + (h[:, 2] - h[:, 3])) / 2)):
                r = summarize(fam(measure), spec["confidence"])
                estimates[dst] = dict(**r, passed=bool(r["n"] >= spec["min_families"] and r["mean"] >= spec["floor_logit"] and r["ci"][0] > 0))
        quantities[key] = estimates
    res = z["dose"][takes].astype(np.float64)
    medians = np.median(res, axis=0)
    err = abs(res[..., 3] - res[..., 4]) / np.maximum(res[..., 2], .02)
    delivery = {f"{src}-r{j+1}": dict(raw_target=medians[i,0], raw_random=medians[i,1], aim=medians[i,2],
                    target=medians[i,3], random=medians[i,4], relative_error_median=float(np.median(err[:,i])),
                    relative_error_p95=float(np.quantile(err[:,i], .95))) for i, (src, j) in enumerate((
                        ("fact",0), ("fact",1), ("book",0), ("book",1)))}
    return dict(n=int(takes.sum())//8, state="finished", endpoints=quantities, delivery=delivery,
                invalid=z["ids"][~z["technical"]].tolist(), decomposition=float(z["local_closure"].max()),
                max_pq_drift=float(z["numeric_diff"].max()))


def main():
    import torch

    C.BUDGET = 12 * 2**20
    C.setup()
    plan, f, spec, led, behavior = (A.load(k) for k in ("plan.json", "freeze.json", "dose-balance-prereg.json", "confirm.json", "confirm-behavior.json"))
    C.verify()
    assert spec["freeze_sha"] == B.sha("freeze.json") == led["freeze"] == behavior["binding"]
    assert (C.OUT / "dose-balance-prereg.json").stat().st_mtime_ns < (C.OUT / "confirm-behavior.json").stat().st_mtime_ns
    if led["state"] != "finished":
        A.save("dose-balance.json", dict(state=led["state"], prereg=B.sha("dose-balance-prereg.json")))
        return
    assert all(B.sha(k) == v for k, v in led["files"].items() if k.startswith("confirm-adjoint"))
    rows = C.O.resolved(A.load(plan["materials"]["confirm"]["file"]), behavior)[:8*FAMILIES]
    source = Path(__file__).relative_to(C.ROOT).as_posix()
    require = dict(source=source, source_hash=A.digest(C.ROOT / source), prereg=B.sha("dose-balance-prereg.json"),
                   freeze=led["freeze"], behavior=led["behavior"], selected=[r["id"] for r in rows])
    if (C.OUT / "dose-balance-binding.json").exists():
        assert require == A.load("dose-balance-binding.json")
    else:
        A.save("dose-balance-binding.json", require)
    known = {r["id"]: r for r in behavior["rows"]}
    directions = {}
    for name in led["files"]:
        if name.startswith("confirm-adjoint"):
            ds = load(name)
            directions.update(zip(ds["ids"], ds["d"], strict=True))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    assert C.model_identity() == plan["snapshot"]
    shards, outputs = {}, []
    with C.P.GpuLease(), torch.no_grad():
        m = C.N.Engine()
        clock = time.monotonic()
        chunks = list(batched(rows, C.SLAB))
        C.N.progress("dose-balance-shards", 0, len(chunks), clock)
        try:
            for j, r in enumerate(chunks, 1):
                name = f"dose-balance-{j:03d}.npz"
                outputs.append(persist(name, r, lambda: compute(m, r, known, directions, f)))
                shards[name] = B.sha(name)
                C.N.progress("dose-balance-shards", j, len(chunks), clock)
        finally:
            m.close()
    merged = {k: np.concatenate([z[k] for z in outputs]) for k in outputs[0]}
    result = aggregate(rows, merged, spec)
    A.save("dose-balance.json", dict(**result, prereg=require["prereg"], binding=B.sha("dose-balance-binding.json"), shards=shards))
    print("Equal-dose confirmation", result["state"], result["n"], result.get("endpoints", {}).get("complete"), flush=True)


if __name__ == "__main__":
    main()
