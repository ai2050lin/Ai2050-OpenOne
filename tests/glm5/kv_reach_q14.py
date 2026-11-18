"""Select model-local routes by attainable natural write dose and replay frozen routes."""

from collections import defaultdict
from itertools import batched
import sys
import time

import numpy as np

import kv_bf_q14 as Native
import kv_reach_run as C


A, B, P = C.A, C.B, C.P
METHODS = (*B.Q.METHODS, "dynamic")
CAP = 2.05
SOURCE_LAYERS = tuple(range(19, 28))


class Engine(Native.Engine):
    def pipeline_step(self, l, i, packet, mode, design, variants):
        if mode != "source":
            return super().pipeline_step(l, i, packet, mode, design, variants)
        t = self.t
        s, n = packet["s"], packet["s"]["n"]
        x = packet["x"].cuda()
        q, k, v = self.qkv(l, x, tuple(j.cuda() for j in s["cs"]))
        if 19 <= i <= 28:
            slots = s["slots"].cuda()
            weights = self.slots(self.read(q[:n, :, -1:], k[:n], s["last"].cuda()), slots)
            values = self.slot_vectors(v, slots.repeat(2, 1)).float()
            d = (values[n:, :, 0] - values[:n, :, 0]).repeat_interleave(self.groups, dim=1)
            dnorm = t.linalg.vector_norm(d, dim=-1)
            target = t.minimum(weights.amax(-1), 2*weights.amin(-1))
            w = l.self_attn.o_proj.weight.reshape(self.width, self.heads, 128)
            projected = t.linalg.vector_norm(t.einsum("bhd,mhd->bhm", d, w.float()), dim=-1)
            packet["scans"][i] = dict(alpha=Native.f32(weights), dnorm=Native.f32(dnorm),
                                         projected=Native.f32(target*projected))
        ans = self.residual(l, x, self.attention(l, q, k, v, s["mask"].cuda()))
        packet["x"] = ans.cpu()

    def support(self, packs, progress):
        original = Native.SOURCE_LAYERS
        try:
            Native.SOURCE_LAYERS = (*SOURCE_LAYERS, 28)
            return self.pipeline(packs, "source", progress)
        finally:
            Native.SOURCE_LAYERS = original


def packs(stage):
    rows = A.load(f"material-{stage}.json")
    b = A.load(f"q4-behavior-{stage}.json")
    assert b["passed"] and all(r["id"] == o["id"] for r, o in zip(rows, b["rows"], strict=True))
    return [tuple(list(x) for x in zip(*g, strict=True))
            for g in batched(zip(rows, b["rows"], strict=True), Native.BATCH)]


def behavior(m, batches, stage, progress, binding):
    seen = [x for slab in m.pipeline(batches, "behavior", progress) for x in slab]
    condition, fam = defaultdict(list), defaultdict(list)
    for r, o in zip((r for batch, _ in batches for r in batch), seen, strict=True):
        assert r["id"] == o["id"]
        condition[r["id"].split("-")[1]].append(o)
        fam[r["family"]].append(o)
    cells = {k: dict(n=len(v), correct=sum(x["conditional"] for x in v),
                     exact_prefix=sum(x["greedy_prefix_and_code"] for x in v),
                     passed=sum(x["conditional"] for x in v)/len(v) >= .90) for k, v in condition.items()}
    result = dict(rows=seen, cells=cells, passed=all(x["passed"] for x in cells.values()), binding=binding,
        eligible=[k for k, v in fam.items() if len(v) == 8 and all(x["conditional"] for x in v)],
        exact_families=[k for k, v in fam.items() if len(v) == 8 and all(x["greedy_prefix_and_code"] for x in v)])
    A.save(f"q14-behavior-{stage}.json", result)
    return result


def rank_routes(d, floor, eligible, groups):
    """One best route per source layer, lexicographic in joint reach and source strength."""
    a, dnorm, projection = (d[k][:, :, eligible] for k in ("alpha", "dnorm", "projected"))
    lo, hi = a.min(-1), a.max(-1)
    target = np.minimum(hi, 2*lo)
    norm = target*dnorm
    possible = (norm >= floor) & (target <= (CAP+2e-6)*np.maximum(lo, 1e-12))
    n = int(eligible.sum())
    assert n and n % 8 == 0
    # Shape: head, family. All eight views must pass independently.
    full = possible.reshape(len(a), a.shape[1], n//8, 8).all(-1)
    rankings = []
    for i, layer in enumerate(SOURCE_LAYERS):
        # Keep the stronger natural receiver first. The precision change is on the second receiver only.
        robust = np.quantile(norm[i+1], .10, axis=1)
        horder = sorted(range(len(robust)), key=lambda h: (-robust[h], h))
        pairs = [(u, v) for j, u in enumerate(horder) for v in horder[j+1:] if u//groups != v//groups]
        j0, j1 = np.asarray(pairs).T
        family = full[i+1, j0] & full[i+1, j1]
        views = possible[i+1, j0] & possible[i+1, j1]
        cover = (full[i, :, None] & family[None]).sum(-1)
        count = (possible[i, :, None] & views[None]).sum(-1)
        src = np.mean(projection[i], axis=-1)
        # Prefer a stronger native upstream write when target reach is tied.
        indices = max(np.ndindex(cover.shape), key=lambda h: (cover[h], count[h], src[h[0]],
                                                               robust[j0[h[1]]], robust[j1[h[1]]]))
        h, j = indices
        rankings.append(dict(source=[layer, h], receiver=[int(j0[j]), int(j1[j])], receiver_layer=layer+1,
                             predicted_families=int(cover[indices]), predicted_views=int(count[indices]),
                             projected_source_mean=float(src[h]), rank10_receiver=[float(robust[j0[j]]), float(robust[j1[j]])]))
    return sorted(rankings, key=lambda x: (-x["predicted_families"], -x["predicted_views"], -x["projected_source_mean"], x["source"]))


def material_precision(m, batches, a, progress, key, design, allowed, p):
    written = {}
    for part, batch_group in enumerate(batched(batches, 16), 1):
        names = [f"q14-{key}-{part:02d}-{j:02d}.npz" for j in range(1, len(batch_group)+1)]
        if not all((C.OUT/f).exists() for f in names):
            pack = m.pipeline(batch_group, "precision-"+key, progress, design=design, variants=a, check_prefix=False)
            for f, ((rows, _), (rec, res)) in zip(names, zip(batch_group, pack, strict=True), strict=True):
                rec["good"] = np.array([r["family"] in allowed for r in rows])
                B.write_np(f, B.arms_summary(rec, res, p["dose"]["limits"]))
        written.update({f: B.sha(f) for f in names})
    return written


def discover(m, p, progress):
    assert not (C.OUT/"q4-behavior-confirm.json").exists()
    assert not (C.OUT/"q14-freeze.json").exists()
    batches = packs("discover")
    if not (C.OUT/"q14-calibration.json").exists():
        r, o = batches[0]
        start = time.monotonic()
        progress("explicit-causal-native-check", 0, 1, start)
        A.save("q14-calibration.json", m.native_check(r[:2], o[:2]))
        progress("explicit-causal-native-check", 1, 1, start)
    bind = B.sha("plan.json")
    b = A.load("q14-behavior-discover.json") if (C.OUT/"q14-behavior-discover.json").exists() else behavior(m, batches, "discover", progress, bind)
    assert b["binding"] == bind and b["passed"]
    ok = set(b["eligible"])
    ids = np.array([r["id"] for rows, _ in batches for r in rows])
    eligible = np.array([i.split("-")[0] in ok for i in ids])
    name = "q14-discover-all-head-support.npz"
    if not (C.OUT/name).exists():
        slabs = m.support(batches, progress)
        scan = {key: np.stack([np.concatenate([s[i][key] for s in slabs]).swapaxes(0, 1)
                               for i in (*SOURCE_LAYERS, 28)]) for key in ("alpha", "dnorm", "projected")}
        B.write_np(name, dict(ids=ids, eligible=eligible, **scan))
    with np.load(C.OUT/name) as scan:
        assert np.array_equal(scan["ids"], ids) and np.array_equal(scan["eligible"], eligible)
        ranking = rank_routes(scan, p["dose"]["limits"]["norm_floor"], eligible, m.groups)
    A.save("q14-natural-ranking.json", dict(all_layers=ranking, input=B.sha(name), plan=bind, heads=m.heads, group_size=m.groups))
    designs = ranking[:p["search"]["max_shortlist"]]
    comparison = []
    for j, design in enumerate([*designs, p["prior_fixed_q14"]]):
        key = f"discover-route-{j}" if j < len(designs) else "discover-fixed"
        shards = material_precision(m, batches, ("native", "late"), progress, key, design, ok, p)
        data = B.load_data(dict(shards=shards))
        assert np.array_equal(data["ids"], ids)
        counts = {}
        for method in ("native", "late"):
            yes = B.view_pass(data, method)
            counts[method] = dict(complete=int(yes.reshape(-1, 8).all(1).sum()), views=int(yes.sum()))
        comparison.append(dict(route=design, shards=shards, counts=counts, key=key))
    selected = max(comparison[:-1], key=lambda v: (v["counts"]["late"]["complete"], v["counts"]["late"]["views"]))
    A.save("q14-freeze.json", dict(stamp=C.stamp(), plan=bind, natural=B.sha("q14-natural-ranking.json"),
           calibration=B.sha("q14-calibration.json"), discovery=B.sha("q14-behavior-discover.json"),
           selected=selected["route"], fixed=p["prior_fixed_q14"], all_discovery=comparison, precision=METHODS))


def confirm(m, p, progress):
    freeze = A.load("q14-freeze.json")
    assert freeze["plan"] == B.sha("plan.json")
    for x in freeze["all_discovery"]:
        assert all(B.sha(k) == h for k, h in x["shards"].items())
    assert B.sha("q14-natural-ranking.json") == freeze["natural"]
    bs = packs("confirm")
    bind = B.sha("q14-freeze.json")
    key = "q14-behavior-confirm.json"
    b = A.load(key) if (C.OUT/key).exists() else behavior(m, bs, "confirm", progress, bind)
    assert b["binding"] == bind and b["passed"]
    for route in ("selected", "fixed"):
        shards = material_precision(m, bs, METHODS, progress, f"confirm-{route}", freeze[route], set(b["eligible"]), p)
        A.save(f"q14-confirm-{route}.json", dict(status="finished", shards=shards, behavior=B.sha(key), freeze=bind))


def run(stage):
    import torch

    p = C.verify()
    assert B.model_identity("q14") == p["identity"]["q14"]
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)

    def progress(name, done, total, start, **details):
        P.stage_log("q14", name, done, total, start, **details)

    with P.GpuLease(), torch.inference_mode():
        start = time.monotonic()
        progress("load", 0, 1, start)
        m = Engine()
        progress("load", 1, 1, start, resident=m.bytes)
        try:
            {"discover": discover, "confirm": confirm}[stage](m, p, progress)
        finally:
            m.close()


if __name__ == "__main__":
    C.setup()
    run(sys.argv[1])
