"""Reconstruct the Phase 2949 head projection and freeze from saved arrays on CPU."""

import numpy as np

import kv_prefix_run as C
from kv_prefix_study import read_np


A, B = C.A, C.B


def verify_coordinates(z, direction):
    q = z["q_native"].astype(np.float64)
    k = z["k_native"].astype(np.float64)
    v = z["v_native"].astype(np.float64)
    a = z["a_native"].astype(np.float64)
    head = z["head_model"].astype(np.float64)
    d = direction[:, None, :].astype(np.float64)
    v_delta = v[:, :1] - v
    advance = (q * (k[:, :1] - k)).sum(-1) / np.sqrt(q.shape[-1])
    # Recover the non-G mixture from the saved unedited head and original softmax mass.
    valid = (a > 1e-7) & (a < 1 - 1e-7)
    angle = np.log(a[valid] / (1 - a[valid])) + advance[valid]
    new_mass = 1 / (1 + np.exp(-angle))
    other = (head[valid] - a[valid, None] * v[valid]) / (1 - a[valid, None])
    changed = other + new_mass[:, None] * (v[valid] + v_delta[valid] - other)
    wanted = ((changed - head[valid]) * np.broadcast_to(d, v.shape)[valid]).sum(-1)
    observed = z["local_total"][valid]
    bound = np.abs(wanted - observed) / (1 + np.abs(wanted))
    max_projection_error = float(bound.max(initial=0))
    assert max_projection_error < 2e-4
    delta = z["head"].astype(np.float64) - z["head"][:, :1]
    actual = (delta * d).sum(-1)
    max_edit_error = float(np.max(np.abs(actual - z["head_output_directed"]) / (1 + abs(actual))))
    assert max_edit_error < 2e-6
    return dict(rows=len(z["ids"]), reconstructed_queries=int(valid.sum()), softmax_boundary=int((~valid).sum()),
                max_relative_projection_error=max_projection_error, max_relative_applied_error=max_edit_error)


def verify_material(p):
    seen = set()
    result = {}
    for role, name in (("pilot", "material-pilot.json"), *[(k, v["file"]) for k, v in p["materials"].items()]):
        rows = A.load(name)
        words = {w for r in rows for w in (*r["values"].values(), r["rejected"])}
        assert not seen & words
        seen |= words
        ix = {r["id"]: r for r in rows}
        for row in rows:
            sites = [i for group in row["sites"].values() for i in group]
            assert len(sites) == len(set(sites)) and max(sites) < row["token"]["start"] < row["token"]["slots"][0]
            assert row["token"]["ids"][row["token"]["slots"][0]] == row["state_ids"][0]
            if row["side"]:
                continue
            other = ix[f"{row['family']}-1{row['distance']}{row['order']}"]
            left, right = (r["prompt"].split("\nCodebook:\n", 1)[1].split("\n\n", 1)[0] for r in (row, other))
            assert left == right and row["code"] != other["code"]
            assert row["state_ids"] == other["state_ids"][::-1]
        result[role] = dict(rows=len(rows), words=len(words), role_pairs=len(rows) // 2, sha=B.sha(name))
    return result


def main():
    C.setup()
    p, f, hold = C.verify(), A.load("freeze.json"), A.load("confirm.json")
    assert f["plan"] == B.sha("plan.json") and hold["freeze"] == B.sha("freeze.json")
    locks = f["discovery"] | hold.get("files", {})
    assert all(B.sha(k) == v for k, v in locks.items())
    by_stage = verify_material(p)
    data, adj = read_np("discover-coalitions.npz"), read_np("discover-adjoint.npz")
    assert np.array_equal(data["ids"], adj["ids"])
    discovery = verify_coordinates(data, adj["d"])
    words = {r["id"]: r["code"] for r in A.load(p["materials"]["discover"]["file"])}
    take = data["technical"]
    sign = np.array([1 if words[k] == "Q" else -1 for k in data["ids"][take]], dtype=np.int8)
    scores = (data["coord"][take, f["selection"]["early"] + 1] * sign[:, None, None]).mean(axis=0)
    assert np.array_equal(scores, np.asarray(f["coord_strength"]))
    assert [sorted(range(len(x)), key=lambda j: -x[j])[:C.COORDS] for x in scores] == f["coordinates"]
    confirm = {}
    state_margins = []
    for name in sorted(hold.get("checks", {})):
        data, adj = read_np(name), read_np(name.replace("intervention", "adjoint"))
        assert np.array_equal(data["ids"], adj["ids"])
        confirm[name] = verify_coordinates(data, adj["d"])
        keep = np.repeat(data["technical"].reshape(-1, 8).all(1), 8)
        state_margin = data["state"][keep, :, 0] - data["state"][keep, :, 1]
        state_margins.append(state_margin[:, [f["arms"].index(k) for k in ("sham", "early", "state-route")]])
    summaries = {}
    if state_margins and len(margins := np.concatenate(state_margins)):
        summaries = dict(zip(("sham", "early", "state-route"), np.percentile(margins, [0, 10, 50, 90, 100], axis=0).T.tolist(), strict=True))
    A.save("native-recheck.json", dict(material=by_stage, discovery=discovery, confirm=confirm,
                                      frozen_rank_exact=True, state_margin_quantiles=summaries,
                                      binding={k: B.sha(k) for k in ("plan.json", "freeze.json", "confirm.json")}))
    print("Recomputed frozen coordinate ranking and local head projection in", len(confirm), "confirmation shards", flush=True)


if __name__ == "__main__":
    main()
