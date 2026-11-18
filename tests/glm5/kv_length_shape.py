"""Post-freeze sensitivity check using identical right padding for paired prefixes."""

from itertools import batched
import time

import numpy as np
import torch

import kv_length_run as C
from kv_length_audit import stat
from kv_length_study import active


def packet(model, rows, width):
    sequences = [r["token"]["ids"] for r in rows]
    lengths = torch.tensor([len(s) for s in sequences], device="cuda")
    ids = torch.tensor([s + [model.tok.pad_token_id] * (width-len(s)) for s in sequences], device="cuda")
    positions = torch.arange(width, device="cuda")
    valid = positions[None, :] < lengths[:, None]
    visible = valid[:, None, None, :] & (positions[None, :] <= positions[:, None])[None, None]
    x = model.decoder.embed_tokens(ids)
    angles = model.rope(x, positions.expand(len(rows), -1).masked_fill(~valid, 1))
    source = torch.tensor([r["token"]["slots"][0] for r in rows])
    return dict(rows=rows, x=x.cpu(), mask=visible.cpu(), angles=[y.cpu() for y in angles], g=source, c=lengths.cpu()-1)


def step(model, layer, index, panel):
    x, mask = panel["x"].cuda(), panel["mask"].cuda()
    q, k, v = model.qkv(layer, x, [y.cuda() for y in panel["angles"]])
    panel["x"] = model.residual(layer, x, model.attention(layer, q, k, v, mask)).cpu()
    if index == 25:
        batch = torch.arange(len(x), device="cuda")
        g, c = panel["g"].cuda(), panel["c"].cuda()
        group = 18 // model.groups
        query = q[batch, :, c].unsqueeze(2)
        key = k[batch, group, g].float()
        val = v[batch, group, g].float()
        alpha = model.read(query, k, mask[batch, :, c].unsqueeze(2))[batch, 18, g]
        score = (query[:, 18, 0].float() * key).sum(-1) * 128**-.5
        panel.update(key=key.cpu().numpy(), value=val.cpu().numpy(), alpha=alpha.cpu().numpy(), score=score.cpu().numpy())


def responses(model, rows, width):
    measured = []
    # Use each complete 16-view family. Every short row shares a packet with its paired long row.
    pairs = [[row for pair in zip(f[:8], f[8:], strict=True) for row in pair] for f in batched(rows, 16)]
    groups = list(batched(pairs, 8))
    for part, families in enumerate(groups, 1):
        clock = time.monotonic()
        panels = [packet(model, batch, width) for f in families for batch in batched(f, 8)]
        C.N.progress("length-common-shape", 0, len(model.layers), clock, part=part, parts=len(groups))
        for i in range(len(model.layers)):
            with model.block(i) as layer:
                for p in panels:
                    step(model, layer, i, p)
            C.N.progress("length-common-shape", i+1, len(model.layers), clock, part=part, parts=len(groups))
        for p in panels:
            size = len(p["rows"])
            y = model.logits(p["x"][torch.arange(size), p["c"]].cuda())
            measured.append(dict(ids=np.array([r["id"] for r in p["rows"]]), top=y.argmax(-1).cpu().numpy(),
                                 pq=y[:, model.letters].float().cpu().numpy(),
                                 **{k:p[k] for k in ("alpha", "score", "key", "value")}))
    return {k:np.concatenate([part[k] for part in measured]) for k in measured[0]}


def summarize(rows, known, z, gate):
    lookup = {r["id"]:r for r in rows}
    pairs = [(lookup[a], lookup[b]) for a, b in batched(z["ids"], 2)]
    for a, b in pairs:
        assert a["id"].split("-")[0]==b["id"].split("-")[0] and a["id"][-3:]==b["id"][-3:]
        row_a, row_b = a["token"], b["token"]
        ga, gb = row_a["slots"][0], row_b["slots"][0]
        assert ga==gb and row_a["ids"][:ga+1]==row_b["ids"][:gb+1]
    original = [known[r] for r in z["ids"]]
    expected_top = np.array([r["top"] for r in original])
    expected_pq = np.array([r["pq"] for r in original])
    stable = z["top"]==expected_top
    pair_differences = {}
    for k in ("key", "value"):
        short, long = z[k][::2], z[k][1::2]
        rel = np.linalg.norm(long-short, axis=-1) / np.maximum(np.linalg.norm(short, axis=-1), 1e-3)
        pair_differences[k] = dict(exact=int((rel==0).sum()), max_relative=float(rel.max()))
    for k in ("alpha", "score"):
        short, long = z[k][::2], z[k][1::2]
        pair_differences[k] = dict(short=float(np.median(short)), long=float(np.median(long)), paired_change=float(np.median(long-short)))
    conflict = np.array([a["state"]!=a["fact"] for a, _ in pairs]).reshape(-1, 8)
    sign = np.array([1 if a["code_s"]=="Q" else -1 for a, _ in pairs]).reshape(-1, 8)
    margin = (z["pq"][:, 1]-z["pq"][:, 0]).reshape(-1, 8, 2)
    signed = (margin*sign[...,None])[conflict].reshape(-1, 4, 2)
    families = (signed[...,1]-signed[...,0]).mean(1)
    return dict(n=len(pairs), exact_top=int(stable.sum()), mismatches=z["ids"][~stable].tolist(),
                max_pq_drift=float(np.abs(z["pq"]-expected_pq).max()), pairs=pair_differences,
                short_margin=float(signed[...,0].mean()), long_margin=float(signed[...,1].mean()), distance=stat(families,gate))


def main():
    C.setup()
    plan = C.verify()
    assert C.identity()==plan["snapshot"]
    raw = C.A.load("confirm-natural.json")
    gate = C.A.load("confirm-gate.json")
    panel_name = plan["panels"]["confirm"]["file"]
    rows = active(C.A.load(panel_name), raw["rows"], gate)
    assert len(rows)==95*16
    width = max(len(r["token"]["ids"]) for r in rows)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    with C.P.GpuLease(), torch.no_grad():
        clock = time.monotonic()
        C.N.progress("length-common-shape-load", 0, 1, clock)
        model = C.N.Engine()
        C.N.progress("length-common-shape-load", 1, 1, clock)
        try:
            z = responses(model, rows, width)
        finally:
            model.close()
    C.B.write_np("common-shape-native.npz", z)
    known = {r["id"]:r for r in raw["rows"]}
    result = summarize(rows, known, z, plan["gate"])
    binding = (panel_name, "plan.json", "freeze.json", "confirm-natural.json", "confirm-gate.json", "audit.json", "common-shape-native.npz")
    C.A.save("common-shape.json", dict(width=width, batch=8, result=result, binding={k:C.B.sha(k) for k in binding}))
    print(result, flush=True)


if __name__=="__main__":
    main()
