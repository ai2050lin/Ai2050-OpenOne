# -*- coding: utf-8 -*-
# READ-ONLY. Precise reference + artifact-class analysis of tests/glm5/result.
import os, time, shutil, datetime, collections

ROOT   = r"D:\AI2050\Ai2050-OpenOne"
TARGET = os.path.join(ROOT, "tests", "glm5", "result")
REPORT = os.path.join(ROOT, "tests", "gpt5", "result", "glm5_result_refs_20261109.txt")
NOW = time.time(); DAY = 86400.0; T0 = time.time()
L = []
def w(s=""): L.append(s)
def human(n):
    n = float(n)
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024.0 or u == "TB":
            return ("%d B" % int(n)) if u == "B" else ("%.2f %s" % (n,u))
        n /= 1024.0
    return "%.2f TB" % n
def ts(t):
    try: return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    except Exception: return "?"
def dstr(t):
    try: return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d")
    except Exception: return "?"

os.makedirs(os.path.dirname(REPORT), exist_ok=True)
w("GLM5 RESULT PRECISE ANALYSIS (READ-ONLY)  %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
for d in ["C:\\","D:\\"]:
    u = shutil.disk_usage(d)
    w("[DISK] %s free=%s / %s" % (d, human(u.free), human(u.total)))
for rb in [r"D:\$RECYCLE.BIN", r"C:\$Recycle.Bin"]:
    s = 0; n = 0
    for dp, dn, fn in os.walk(rb, onerror=lambda e: None):
        for f in fn:
            try: s += os.path.getsize(os.path.join(dp,f)); n += 1
            except OSError: pass
    w("[BIN]  %s size=%s files=%d" % (rb, human(s), n))

# ---------- pass 1: per-experiment artifact classes ----------
agg = collections.defaultdict(lambda: {"sz":0,"n":0,"npy":0,"npz":0,"js":0,"other":0,
                                       "mint":NOW,"maxt":0,"r24":0,"r7":0,
                                       "days":collections.Counter(),"sha":0,"rj":0,"ej":0,"beh":0})
day_all = collections.Counter(); sha_files = []
for dp, dn, fn in os.walk(TARGET, topdown=True, onerror=lambda e: None):
    kept = []
    for x in dn:
        q = os.path.join(dp,x)
        try:
            if getattr(os.stat(q, follow_symlinks=False), "st_file_attributes", 0) & 0x400: continue
        except OSError: continue
        kept.append(x)
    dn[:] = kept
    for f in fn:
        p = os.path.join(dp,f)
        try: st = os.stat(p)
        except OSError: continue
        sz = st.st_size; mt = st.st_mtime
        rel = os.path.relpath(p, TARGET); parts = rel.split(os.sep)
        top = parts[0] if len(parts)>1 else "(root files)"
        ext = os.path.splitext(f)[1].lower()
        a = agg[top]; a["sz"] += sz; a["n"] += 1
        a["mint"] = min(a["mint"], mt); a["maxt"] = max(a["maxt"], mt)
        if mt > NOW-DAY: a["r24"] += sz
        if mt > NOW-7*DAY: a["r7"] += sz
        if ext == ".npy": a["npy"] += sz
        elif ext == ".npz": a["npz"] += sz
        elif ext == ".json": a["js"] += sz
        else: a["other"] += sz
        dk = dstr(mt); a["days"][dk] += sz; day_all[dk] += sz
        if f == "result.json": a["rj"] += 1
        if f == "execution.json": a["ej"] += 1
        if f == "behavior.json": a["beh"] += 1
        if ext == ".sha256": a["sha"] += 1; sha_files.append((sz, p, mt))

w("\n[1] TOTAL %s in %d experiment dirs, %d files" %
  (human(sum(a["sz"] for a in agg.values())), len(agg), sum(a["n"] for a in agg.values())))
w("\n[2] BY MTIME DAY (bytes written)")
for d, s in sorted(day_all.items()):
    w("  %10s  %s" % (human(s), d))

# ---------- clean corpus: code + docs ONLY (no data json) ----------
SKIP = {".git","node_modules","__pycache__",".venv","venv","binaries","dist","build",
        ".cache","models","distributed_data","release","video","frontend","site-packages",
        "_trash_stage","ms-playwright",".pytest_cache",".mypy_cache","result","data",
        "desktop-1.0.8","desktop-1.0.9","desktop-1.0.10","desktop-1.0.11",
        "desktop-1.0.12","desktop-1.0.13"}
CODE_EXT = {".py",".md",".yaml",".yml",".toml",".sh",".ps1",".cfg",".ini",".txt",".ipynb"}
docs = []
for dp, dn, fn in os.walk(ROOT, topdown=True, onerror=lambda e: None):
    kept = []
    for x in dn:
        if x in SKIP: continue
        q = os.path.join(dp,x)
        try:
            if getattr(os.stat(q, follow_symlinks=False), "st_file_attributes", 0) & 0x400: continue
        except OSError: continue
        kept.append(x)
    dn[:] = kept
    for f in fn:
        if os.path.splitext(f)[1].lower() not in CODE_EXT: continue
        p = os.path.join(dp,f)
        try:
            if os.stat(p).st_size > 2*1024*1024: continue
            with open(p,"r",encoding="utf-8",errors="ignore") as fh: text = fh.read()
        except OSError: continue
        docs.append((os.path.relpath(p, ROOT), text))
w("\n[3] CLEAN CORPUS: %d code/doc files, %s text" %
  (len(docs), human(sum(len(t) for _,t in docs))))

names = list(agg.keys())
refhits = {}
for name in names:
    if len(name) < 6 or name == "(root files)":
        refhits[name] = ([], -1); continue
    hits = [p for p,t in docs if name in t]
    refhits[name] = (hits[:8], len(hits))

rows = sorted(((a["sz"], name, a) for name,a in agg.items()), reverse=True, key=lambda x: x[0])
w("\n[4] TOP 45 EXPERIMENTS (artifact-class breakdown)")
w("  cols: total | npy | npz | json | other | files | first | last | 24h | code_hits | rj/ej/beh/sha | name")
for sz, name, a in rows[:45]:
    hits, nh = refhits[name]
    w("  %10s|%9s|%9s|%8s|%8s|%7d|%s|%s|%9s| c=%-4d | %d/%d/%d/%d | %s" %
      (human(sz), human(a["npy"]), human(a["npz"]), human(a["js"]), human(a["other"]),
       a["n"], ts(a["mint"]), ts(a["maxt"]), human(a["r24"]), nh,
       a["rj"], a["ej"], a["beh"], a["sha"], name))

w("\n[5] TOP 20: exact code/doc files mentioning the experiment name")
for sz, name, a in rows[:20]:
    hits, nh = refhits[name]
    w("  --- %s  (%s, code_hits=%d) ---" % (name, human(sz), nh))
    for h in hits: w("      %s" % h)
    if not hits: w("      (none)")

w("\n[6] ZERO CODE/DOC REFERENCE dirs (size desc, top 35)")
zero = [(sz,name,a) for sz,name,a in rows if refhits[name][1] == 0 and name != "(root files)"]
w("  count=%d  total=%s" % (len(zero), human(sum(x[0] for x in zero))))
for sz, name, a in zero[:35]:
    w("    %10s  %7d files  last=%s  %s" % (human(sz), a["n"], ts(a["maxt"]), name))

w("\n[7] .sha256 REGISTRATION FILES")
w("  count=%d" % len(sha_files))
for sz, p, mt in sorted(sha_files, reverse=True)[:25]:
    w("    %10s  %s  %s" % (human(sz), ts(mt), os.path.relpath(p,TARGET)))

w("\n[8] ACTIVE (written in last 24h) dirs")
for sz, name, a in rows:
    if a["r24"] > 0:
        w("    %10s / %10s  %s" % (human(a["r24"]), human(sz), name))

w("\n[9] CONCLUSION-LEVEL JSON HEAVY dirs (json >= 100MB, keep evidence chain)")
for sz, name, a in rows:
    if a["js"] >= 100*1024*1024:
        w("    json=%10s  total=%10s  rj=%4d ej=%4d  %s" %
          (human(a["js"]), human(sz), a["rj"], a["ej"], name))

w("\n===== END (scan_sec=%.1f) =====" % (time.time()-T0))
with open(REPORT,"w",encoding="utf-8") as f: f.write("\n".join(L))
print("DONE_GLM5_REFS")
