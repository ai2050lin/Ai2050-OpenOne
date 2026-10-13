"""Tier A cleanup: generate FULL manifest before any deletion. Read-only."""
import os, json, hashlib, collections, time, re

TESTS = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT   = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tierA_manifest.json"

JUNC = set(); stack = [TESTS]
while stack:
    cur = stack.pop()
    try:
        with os.scandir(cur) as it:
            for e in it:
                try:
                    if not e.is_dir(follow_symlinks=False): continue
                    st = e.stat(follow_symlinks=False)
                    if getattr(st, 'st_file_attributes', 0) & 0x400:
                        JUNC.add(os.path.normcase(os.path.abspath(e.path)))
                    else:
                        stack.append(e.path)
                except OSError: pass
    except OSError: pass

manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": {}}

# ---------- A1: __pycache__ directories ----------
pycache_dirs = []
for dp, dn, fns in os.walk(TESTS):
    dn[:] = [d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp, d))) not in JUNC]
    for d in list(dn):
        if d == "__pycache__":
            pycache_dirs.append(os.path.join(dp, d)); dn.remove(d)
files = []; sz = 0
for base in pycache_dirs:
    for a, b, cc in os.walk(base):
        for f in cc:
            fp = os.path.join(a, f)
            try: sz += os.path.getsize(fp); files.append(fp)
            except OSError: pass
manifest["groups"]["A1_pycache"] = {
    "desc": "Python bytecode cache, auto-rebuilt on import",
    "dirs": [os.path.relpath(p, TESTS) for p in pycache_dirs],
    "files": files, "bytes": sz, "count": len(files),
    "method": "direct delete (fully regenerable)",
}

# ---------- A2: third-party packages ----------
third = []
for rel in [r"glm5_temp\rdc_update_browser_packages", r"gpt5_temp\tf457"]:
    p = os.path.join(TESTS, rel)
    if not os.path.isdir(p): continue
    fl = []; s = 0
    for a, b, cc in os.walk(p):
        for f in cc:
            fp = os.path.join(a, f)
            try: s += os.path.getsize(fp); fl.append(fp)
            except OSError: pass
    third.append({"rel": rel, "files": fl, "bytes": s, "count": len(fl)})
manifest["groups"]["A2_thirdparty"] = {
    "desc": "pip-installed 3rd-party copies (playwright/greenlet/pyee, HF transformers)",
    "dirs": third,
    "bytes": sum(d["bytes"] for d in third),
    "count": sum(d["count"] for d in third),
    "method": "direct delete (pip reinstallable)",
}

# ---------- A3: *.log / *.bak ----------
logfiles = []; lsz = 0
for dp, dn, fns in os.walk(TESTS):
    dn[:] = [d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp, d))) not in JUNC]
    for fn in fns:
        if fn.endswith((".log", ".bak")):
            fp = os.path.join(dp, fn)
            try: lsz += os.path.getsize(fp); logfiles.append(fp)
            except OSError: pass
manifest["groups"]["A3_log_bak"] = {
    "desc": "run logs and editor backups; conclusions already in MEMO",
    "files": logfiles, "bytes": lsz, "count": len(logfiles),
    "method": "recycle bin",
}

# ---------- A4: exact duplicate groups (keep one per group) ----------
MIN = 1024 * 1024
by_size = collections.defaultdict(list)
for dp, dn, fns in os.walk(TESTS):
    dn[:] = [d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp, d))) not in JUNC]
    for fn in fns:
        fp = os.path.join(dp, fn)
        try: s = os.path.getsize(fp)
        except OSError: continue
        if s >= MIN: by_size[s].append(fp)

def h32(fp):
    x = hashlib.blake2b(digest_size=16)
    with open(fp, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b: break
            x.update(b)
    return x.hexdigest()

def h128(fp):
    x = hashlib.sha256()
    with open(fp, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b: break
            x.update(b)
    return x.hexdigest()

groups = collections.defaultdict(list); t0 = time.time(); done = 0
cand = {s: v for s, v in by_size.items() if len(v) > 1}
tot = sum(len(v) for v in cand.values())
for s, fl in cand.items():
    for fp in fl:
        try: groups[(s, h32(fp))].append(fp)
        except OSError: pass
    done += len(fl)
    if done % 1000 < len(fl):
        print("  hashed %d/%d (%.0fs)" % (done, tot, time.time() - t0), flush=True)

BAD = re.compile(r"(smoke|pilot|_bak|failed|\.tmp\.)", re.I)
dupgroups = []; keep_all = []; drop_all = []; kbytes = 0; dbytes = 0
for (s, hh), v in groups.items():
    if len(v) < 2: continue
    # keeper preference: no smoke/pilot/bak/failed in path, then shortest path
    v_sorted = sorted(v, key=lambda p: (bool(BAD.search(p)), len(p), p))
    keeper = v_sorted[0]
    drops = v_sorted[1:]
    # belt-and-braces: verify keeper vs each drop with sha256 before marking deletable
    kh = h128(keeper); ok = True
    for dpath in drops:
        try:
            if h128(dpath) != kh: ok = False; break
        except OSError: ok = False; break
    if not ok:
        continue
    dupgroups.append({"sha256": kh, "size": s, "keeper": keeper,
                      "keepers_rel": os.path.relpath(keeper, TESTS),
                      "drop": drops, "drop_rel": [os.path.relpath(p, TESTS) for p in drops]})
    keep_all.append(keeper); drop_all.extend(drops)
    kbytes += s; dbytes += s * len(drops)

manifest["groups"]["A4_dup_groups"] = {
    "desc": "byte-identical duplicate files; keep one, delete rest (sha256-verified)",
    "groups": dupgroups, "group_count": len(dupgroups),
    "files": drop_all, "bytes": dbytes, "count": len(drop_all),
    "kept_files": keep_all, "kept_bytes": kbytes,
    "method": "recycle bin",
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

tot_b = 0; tot_c = 0
print("\n=== TIER A MANIFEST ===")
for k, g in manifest["groups"].items():
    print("%-18s %10.2f MB  %7d files" % (k, g["bytes"]/1024**2, g["count"]))
    tot_b += g["bytes"]; tot_c += g["count"]
print("%-18s %10.2f MB  %7d files" % ("TOTAL", tot_b/1024**2, tot_c))
print("hash time %.0fs" % (time.time()-t0))
print("manifest ->", OUT)
