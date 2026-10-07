"""Audit all six puzzle groups; CPU-only reanalysis, no model execution."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/rdc_six_puzzles_audit_20260923"
HIST = ROOT / "tests/glm5/result/rdc_query_construction_20260913"
SOURCES = []


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def track(p):
    SOURCES.append({"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)})
    return p


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")


def read_result(n):
    files = list((HIST / f"phase{n}").glob("*/result.json"))
    assert len(files) == 1, files
    p = track(files[0])
    (OUT / f"source_result_{n}.json").write_bytes(p.read_bytes())
    return json.loads(p.read_text(encoding="utf-8"))


def auc_columns(y, scores, ties=True):
    x = np.asarray(scores)
    if x.ndim == 1:
        x = x[:, None]
    if ties:
        ranks = rankdata(x, method="average", axis=0)
    else:
        order = np.argsort(x, axis=0, kind="mergesort")
        ranks = np.empty(x.shape, dtype=np.float64)
        np.put_along_axis(ranks, order, np.arange(1, len(x)+1)[:, None], axis=0)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    assert n1 and n0
    return (ranks[y == 1].sum(axis=0) - n1*(n1+1)/2) / (n1*n0)


def summary(a):
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a)), "frac_gt_0.7": float(np.mean(a > .7)),
            "frac_gt_0.9": float(np.mean(a > .9))}


def select_field(p):
    # NPZ does not support mmap. Stream one record at a time to avoid a >1GB allocation.
    with zipfile.ZipFile(p) as zf, zf.open("X.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        assert len(shape) == 4 and shape[1:] == (6, 9, 2560) and not fortran
        per = int(np.prod(shape[1:])) * dtype.itemsize
        X = np.empty((shape[0], shape[-1]), dtype=np.float32)
        for i in range(shape[0]):
            data = stream.read(per)
            assert len(data) == per
            X[i] = np.frombuffer(data, dtype=dtype).reshape(shape[1:])[5, 8, :]
            if (i+1) % 512 == 0:
                print(f"Read native field slice: {i+1}/{shape[0]}", flush=True)
    return X, {"stored_shape": shape, "stored_dtype": str(dtype),
               "selection": "X[:, last(position index 5), layer-slot 8, all 2560 coordinates]"}


def strings(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x])


def main():
    assert not OUT.exists(), "Do not replace a completed audit."
    OUT.mkdir(parents=True)
    gpt = track(ROOT / "research/gpt5/docs/AGI_GPT5_MEMO.md")
    text = gpt.read_text(encoding="utf-8")
    (OUT / "gpt_memo_snapshot.md").write_text(text, encoding="utf-8")
    hs = list(re.finditer(r"^## Phase (\d+).*", text, re.M))
    wanted = set(range(3103, 3124)) | {3035,3036,3039,3040,3041}
    index, excerpts = {}, []
    for j, h in enumerate(hs):
        n = int(h.group(1))
        if n in wanted:
            index[n] = {"line": text.count("\n", 0, h.start())+1, "title": h.group(0)}
            excerpts.append(text[h.start(): hs[j+1].start() if j+1<len(hs) else len(text)])
    write("history_index.json", index)
    (OUT / "selected_history.md").write_text("\n".join(excerpts), encoding="utf-8")
    attachment = track(Path(r"C:\Users\Admin\.codex\attachments\ce8dc325-de55-4f97-a8b6-4da1e8c6586b\已粘贴的文本.txt"))
    (OUT / "six_puzzles_source.txt").write_bytes(attachment.read_bytes())
    results = {n: read_result(n) for n in range(3104, 3124)}
    for n in (3107,3110,3111,3112,3116,3117,3120,3122,3123):
        scripts = list((ROOT / "tests/glm5").glob(f"phase{n}_omega*.py"))
        assert len(scripts) == 1, scripts
        track(scripts[0])

    cap = list((HIST / "phase3105").glob("*/capture.npz"))
    assert len(cap) == 1
    track(cap[0])
    X, field = select_field(cap[0])
    with np.load(cap[0], allow_pickle=False) as z:
        keep = strings(z["tag"]) == "main"
        y = z["truth"].astype(int)[keep]
        split = strings(z["split"])[keep]
    X = X[keep]
    tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X-mu)/sd
    raw_old = auc_columns(y[te], Z[te], False)
    raw_new = auc_columns(y[te], Z[te], True)
    train_auc = auc_columns(y[tr], Z[tr], True)
    old_free, new_free = np.maximum(raw_old,1-raw_old), np.maximum(raw_new,1-raw_new)
    train_oriented = np.where(train_auc >= .5, raw_new, 1-raw_new)
    repeats = 1 - np.array([len(np.unique(Z[te,j]))/len(te) for j in range(2560)])
    k5 = []
    for r in range(10):
        sel = np.sort(np.random.RandomState(31100+5+97*r).choice(2560,5,replace=False))
        F = Z[tr][:,sel]
        w = np.linalg.solve((F.T@F)/len(tr)+.01*np.eye(5,dtype=np.float32),
                            (F.T@(y[tr]*2.-1.).astype(np.float32))/len(tr)).astype(np.float32)
        pred = Z[te][:,sel]@w
        k5.append({"seed_index": r, "coordinates": sel.tolist(),
                   "old_auc": float(auc_columns(y[te], pred,False)[0]),
                   "tie_aware_auc": float(auc_columns(y[te], pred,True)[0])})
    k5_saved = results[3110]["gates"]["ksweep_truth"]["K5"]["aucs"]
    audit_auc = {
        "status": "Existing capture reanalysis, not independent model replication",
        "field": field, "n_main": len(y), "n_train": len(tr), "n_test": len(te),
        "test_positive": int(y[te].sum()), "test_negative": int(len(te)-y[te].sum()),
        "historical_saved": results[3111]["gates"]["DC_per_coordinate"],
        "historical_function_reproduced": summary(old_free),
        "tie_aware_test_oriented_descriptive": summary(new_free),
        "tie_aware_training_oriented_test": summary(train_oriented),
        "median_repeat_fraction": float(np.median(repeats)),
        "max_abs_tie_correction": float(np.max(np.abs(new_free-old_free))),
        "changed_orientation_count": int(np.sum((raw_new>=.5) != (train_auc>=.5))),
        "K5": k5, "K5_tie_aware_median": float(np.median([a["tie_aware_auc"] for a in k5])),
        "K5_max_abs_vs_saved_rounded": float(max(abs(a["old_auc"]-b) for a,b in zip(k5,k5_saved))),
        "limitation": "One existing material/position/layer. Free sign chosen on test is descriptive; training-oriented values remove that use. No new confidence interval or independent generalization claim.",
    }
    write("auc_reanalysis.json", audit_auc)
    np.savez_compressed(OUT / "coordinate_auc_reanalysis.npz", old_test_free=old_free,
                        tie_aware_test_free=new_free, train_oriented_test=train_oriented,
                        train_auc=train_auc, repeated_fraction=repeats)
    print("AUC reanalysis complete: "+json.dumps(audit_auc["tie_aware_training_oriented_test"]),flush=True)

    ledger_file = track(HIST / "phase3103/omega_p101_formula_audit/proposition_ledger.json")
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    rows = []
    for value in ledger.values():
        if isinstance(value,list) and value and isinstance(value[0],dict) and "grade" in value[0]:
            rows += value
    counts = Counter(g for row in rows for g in row["grade"].split("+"))
    only = Counter(row["grade"] for row in rows)
    cv = results[3116]["coverage"]
    num = {"ledger": {"unique_rows":len(rows), "saved":ledger["grade_distribution"],
                      "recounted_grade_memberships":dict(counts), "exact_grade_combinations":dict(only)},
           "ablation": {"sum_single":cv["sum_single"],"d_all":cv["d_all"],
                        "ratio":cv["sum_single"]/cv["d_all"],"relative_discrepancy":cv["cov"]},
           "sentence_results": {k:v for k,v in results[3122].items() if k in ("part_b","part_a")},
           "syntax_layer_results": results[3123].get("part_c"),
           "pair_ablation": results[3117],
           "linear_mean_regression": results[3120].get("part_c")}
    # The saved curve used by phase3123 directly contradicts its text's >=0.975 premise.
    p18 = list((HIST / "phase3118").glob("*/traj_readout.npz"))
    assert len(p18)==1
    with np.load(track(p18[0]), allow_pickle=False) as z:
        num["real_auc_curve"] = z["auc_curve"].astype(float).tolist()
    num["simulated_auc_curve"] = results[3123]["part_a"]["sim"]["auc_sim_dir"]
    write("numerical_checks.json", num)
    write("evidence_manifest.json", {"phase":2759,"created_local":datetime.now().astimezone().isoformat(),
                                    "model_runs":0,"sources":SOURCES,
                                    "script_sha256":digest(Path(__file__))})
    print(json.dumps({"auc":audit_auc["tie_aware_training_oriented_test"],
                      "K5_median":audit_auc["K5_tie_aware_median"],
                      "ledger":num["ledger"],"output":str(OUT)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
