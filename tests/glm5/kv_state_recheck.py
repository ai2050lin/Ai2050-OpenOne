"""Reconstruct the Phase 2950 frozen rank, coordinate sum, and applied head projection."""

import re

import numpy as np

import kv_state_run as C
from kv_state_study import code_margin, configs, load, sign


A, B = C.A, C.B


def fact_pair_material(p):
    used, results = set(), {}
    files = {"pilot": "material-pilot.json", **{k: v["file"] for k, v in p["materials"].items()}}
    for kind, name in files.items():
        rows = A.load(name)
        words = C.N.words(rows)
        assert not used & words
        used |= words
        for a in range(0, len(rows), 8):
            for first, second in zip(rows[a:a+4], rows[a+4:a+8], strict=True):
                assert first["family"] == second["family"] and first["state_ids"] == second["state_ids"][::-1]
                books = [r["prompt"].split("\nCodebook:\n", 1)[1].split("\n\n")[0] for r in (first, second)]
                assert books[0] == books[1] and first["code"] != second["code"]
                for row in (first, second):
                    flat = [i for x in row["sites"].values() for i in x]
                    assert len(flat) == len(set(flat)) and max(flat) < row["token"]["start"] < row["token"]["slots"][0]
        results[kind] = dict(rows=len(rows), role_pairs=len(rows)//2, words=len(words), hash=B.sha(name))
    return results


def complete_state_fields(ledger, rows):
    result = {}
    if ledger["state"] != "finished":
        return result
    data = A.load("rollout.json")
    for arm, report in data.items():
        total = dict(reached=0, exact_alternative=0, original_word=0, other_field=0, field_missing_or_repeated=0,
                     alternate_word_and_new_code=0, alternate_word_and_old_code=0, exceptions=[])
        for rid, observation in report["observations"].items():
            if observation["stop"] != "code":
                continue
            total["reached"] += 1
            fields = re.findall(r"^State:[ \t]*([^\r\n]*)", observation["text"], re.M)
            full = [f.strip() for f in fields]
            row = rows[rid]
            if full == [row["rejected"]]:
                total["exact_alternative"] += 1
                if observation["code"] in (" P", " Q"):
                    k = "alternate_word_and_old_code" if observation["code"].strip() == row["code"] else "alternate_word_and_new_code"
                    total[k] += 1
            else:
                status = "field_missing_or_repeated" if len(full) != 1 else "original_word" if full == [row["values"]["State"]] else "other_field"
                total[status] += 1
                total["exceptions"].append(dict(id=rid, state_fields=full, classification=status, first=observation["first"], code=observation["code"]))
        result[arm] = total
    return result


def main():
    C.setup()
    p = C.verify()
    f, ledger = A.load("freeze.json"), A.load("confirm.json")
    assert ledger["freeze"] == B.sha("freeze.json") and f["plan"] == B.sha("plan.json")
    locks = f["discovery"] | ledger.get("files", {})
    assert all(B.sha(k) == v for k, v in locks.items())
    m = fact_pair_material(p)
    es = load("discover-e-adjoint.npz")
    coord = f["e"]["coordinates"]
    values = es["product"].mean(0)[C.E_LAYERS.index(f["e"]["layer"])]
    assert sorted(range(len(values)), key=lambda i: -values[i])[:len(coord)] == coord
    assert not (set(coord) & set(f["e_random"][0]) or set(coord) & set(f["e_random"][1]))
    b = A.load("confirm-behavior.json")
    use = {r["id"]: r for r in C.O.resolved(A.load(p["materials"]["confirm"]["file"]), b)}
    selected = set(ledger.get("accepted", []))
    contrasts = (("g-k", "both"), ("g-v", "both"), ("g-kv", "both"), ("g-reverse", "sham"), ("cache-v-zero", "sham"))
    cfg = configs(f)
    names = list(cfg)
    terminal = [names.index(k) for k, spec in cfg.items() if spec["e"] is not None and spec["e"]["layer"] == 39]
    discovery_full = load("discover-g-and-e.npz")
    lookup_d = {r["id"]: r for r in A.load(p["materials"]["discover"]["file"])}
    discovery_ok = np.repeat(discovery_full["technical"].reshape(-1, 8).all(1), 8)
    discovery_e = discovery_full["state_top"][discovery_ok, names.index("e")]
    target = [lookup_d[k]["state_ids"][1] for k in discovery_full["ids"][discovery_ok]]
    discovery_full_prefix = dict(views=len(target), recognized=int((discovery_e == target).sum()))
    evidence = {a: [] for a, _ in contrasts}
    rechecks = {}
    for name in sorted(ledger.get("checks", {})):
        z, d = load(name), load(name.replace("native", "adjoint"))
        assert np.array_equal(z["ids"], d["ids"])
        native = z["head"].astype(np.float64)
        delta = ((native - native[:, :1]) * d["d"][:, None].astype(np.float64)).sum(-1)
        direct_error = float(np.max(abs(delta - z["head_output_directed"]) / (1 + abs(delta))))
        decomposition = z["coord"].sum((-2, -1), dtype=np.float64)
        residual = float(np.max(abs(decomposition - z["local_total"]) / (1 + abs(decomposition))))
        assert direct_error < 2e-6 and residual < 2e-4
        if terminal:
            fixed = z["pq"][:, terminal]
            assert np.array_equal(fixed, np.broadcast_to(z["pq"][:, :1], fixed.shape))
        rows = [use[k] for k in z["ids"]]
        keep = np.array([k in selected for k in z["ids"]], dtype=bool)
        y = code_margin(z, rows).astype(np.float64)
        directed = delta * sign(rows)[:, None]
        for a, b_ in contrasts:
            i, j = names.index(a), names.index(b_)
            evidence[a].append(np.stack((directed[keep, i] - directed[keep, j], y[keep, i] - y[keep, j]), axis=-1))
        rechecks[name] = dict(rows=len(rows), heads=int(delta.size), coordinate_error=residual, applied_error=direct_error,
                              terminal_fixed_code_cases=len(rows)*len(terminal))
    calibration = {}
    for a, parts in evidence.items():
        if not parts:
            continue
        paired = np.concatenate(parts)
        if not len(paired):
            continue
        families = paired.reshape(-1, 8, 2).mean(axis=1)
        nonzero = (paired != 0).all(1)
        calibration[a] = dict(predicted=float(paired[:, 0].mean()), measured=float(paired[:, 1].mean()),
                              pearson_family=float(np.corrcoef(families.T)[0, 1]) if families.std(0).min() else None,
                              signs=float((np.sign(paired[nonzero, 0]) == np.sign(paired[nonzero, 1])).mean()) if nonzero.any() else None,
                              sign_cases=int(nonzero.sum()))
    A.save("native-recheck.json", dict(coordinate_rank_exact=True, rechecks=rechecks, calibration=calibration, materials=m,
                                      complete_state_fields=complete_state_fields(ledger, use),
                                      discovery_full_prefix=discovery_full_prefix,
                                      binding={k: B.sha(k) for k in ("plan.json", "freeze.json", "confirm.json")}))
    print("Checked original State rank, paired materials and", len(rechecks), "G/C shards", flush=True)


if __name__ == "__main__":
    main()
