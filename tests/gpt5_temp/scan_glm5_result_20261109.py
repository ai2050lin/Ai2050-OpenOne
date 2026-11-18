# -*- coding: utf-8 -*-
# READ-ONLY analysis of tests/glm5/result. No file is deleted, moved or renamed.
import os, re, time, shutil, datetime, hashlib, collections

ROOT   = r"D:\AI2050\Ai2050-OpenOne"
TARGET = os.path.join(ROOT, "tests", "glm5", "result")
REPORT = os.path.join(ROOT, "tests", "gpt5", "result", "glm5_result_cleanup_analysis_20261109.txt")
NOW = time.time()
DAY = 86400.0
T0 = time.time()
L = []
def w(s=""):
    L.append(s)
def human(n):
    n = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0 or u == "TB":
            return ("%d B" % int(n)) if u == "B" else ("%.2f %s" % (n, u))
        n /= 1024.0
    return "%.2f TB" % n
def ts(t):
    try:
        return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"

os.makedirs(os.path.dirname(REPORT), exist_ok=True)
w("GLM5 RESULT CLEANUP ANALYSIS (READ-ONLY)  %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("TARGET = %s" % TARGET)
for d in ["C:\\", "D:\\"]:
    try:
        u = shutil.disk_usage(d)
        w("[DISK] %s total=%s used=%s free=%s (%.1f%% used)" %
          (d, human(u.total), human(u.used), human(u.free), 100.0*u.used/u.total))
    except Exception as e:
        w("[DISK] %s ERROR %s" % (d, e))

# ---------- 1. collect every file ----------
files = []
ndirs = [0]
for dp, dn, fn in os.walk(TARGET, topdown=True, onerror=lambda e: None):
    ndirs[0] += 1
    kept = []
    for x in dn:
        q = os.path.join(dp, x)
        try:
            st = os.stat(q, follow_symlinks=False)
            if getattr(st, "st_file_attributes", 0) & 0x400:
                continue
        except OSError:
            continue
        kept.append(x)
    dn[:] = kept
    for f in fn:
        p = os.path.join(dp, f)
        try:
            st = os.stat(p)
        except OSError:
            continue
        r = os.path.relpath(p, TARGET)
        parts = r.split(os.sep)
        top = parts[0] if len(parts) > 1 else "(root files)"
        sec = os.sep.join(parts[:2]) if len(parts) > 2 else top
        ext = os.path.splitext(f)[1].lower() or "(noext)"
        files.append((p, r, f, st.st_size, st.st_mtime, ext, top, sec))
w("\n[1] OVERVIEW")
w("  files=%d  dirs=%d  total=%s  scan_sec=%.1f" %
  (len(files), ndirs[0], human(sum(x[3] for x in files)), time.time()-T0))

ext_agg = collections.defaultdict(lambda: [0, 0])
month_agg = collections.defaultdict(lambda: [0, 0])
top_agg = collections.defaultdict(lambda: {"sz":0,"n":0,"mint":NOW,"maxt":0,
                                           "r24":0,"r7":0,"r30":0,
                                           "resultjson":0,"execjson":0,"behavior":0,
                                           "ext":collections.Counter(),"subs":set()})
sec_agg = collections.defaultdict(lambda: [0, 0])
datapack_files = []
tmp_big = []
recent_files = []
for p, r, b, sz, mt, ext, top, sec in files:
    ext_agg[ext][0] += sz; ext_agg[ext][1] += 1
    mk = datetime.datetime.fromtimestamp(mt).strftime("%Y-%m")
    month_agg[mk][0] += sz; month_agg[mk][1] += 1
    a = top_agg[top]
    a["sz"] += sz; a["n"] += 1
    a["mint"] = min(a["mint"], mt); a["maxt"] = max(a["maxt"], mt)
    if mt > NOW-DAY: a["r24"] += sz
    if mt > NOW-7*DAY: a["r7"] += sz
    if mt > NOW-30*DAY: a["r30"] += sz
    a["ext"][ext] += sz
    a["subs"].add(sec)
    if b == "result.json": a["resultjson"] += 1
    if b == "execution.json": a["execjson"] += 1
    if b == "behavior.json": a["behavior"] += 1
    sec_agg[sec][0] += sz; sec_agg[sec][1] += 1
    low = r.lower()
    if "datapack" in low or "research-library" in b.lower():
        datapack_files.append((sz, p, mt))
    if ext == ".tmp" and sz >= 50*1024*1024:
        tmp_big.append((sz, p, mt))
    if mt > NOW-DAY and sz >= 1024*1024:
        recent_files.append((sz, p, mt))

w("\n[2] BY EXTENSION (size desc)")
for e, (s, n) in sorted(ext_agg.items(), key=lambda kv: -kv[1][0]):
    w("  %10s  %8d files  %s" % (human(s), n, e))
w("\n[3] BY MTIME MONTH")
for m, (s, n) in sorted(month_agg.items()):
    w("  %10s  %8d files  %s" % (human(s), n, m))

# ---------- 2. corpus for reference analysis ----------
SKIP = {".git","node_modules","__pycache__",".venv","venv","binaries","dist","build",
        ".cache","models","distributed_data","release","video","frontend","site-packages",
        "_trash_stage","ms-playwright",".pytest_cache",".mypy_cache","result",
        "desktop-1.0.8","desktop-1.0.9","desktop-1.0.10","desktop-1.0.11",
        "desktop-1.0.12","desktop-1.0.13"}
CODE_EXT = {".py",".yaml",".yml",".toml",".json",".sh",".ps1",".cfg",".ini",".txt",".ipynb"}
DOC_EXT = {".md"}
TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]{5,}")
code_df = collections.Counter(); doc_df = collections.Counter()
code_docs = [0]; doc_docs = [0]
for dp, dn, fn in os.walk(ROOT, topdown=True, onerror=lambda e: None):
    kept = []
    for x in dn:
        if x in SKIP:
            continue
        q = os.path.join(dp, x)
        try:
            st = os.stat(q, follow_symlinks=False)
            if getattr(st, "st_file_attributes", 0) & 0x400:
                continue
        except OSError:
            continue
        kept.append(x)
    dn[:] = kept
    for f in fn:
        ext = os.path.splitext(f)[1].lower()
        if ext not in CODE_EXT and ext not in DOC_EXT:
            continue
        p = os.path.join(dp, f)
        try:
            if os.stat(p).st_size > 12*1024*1024:
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        toks = set(TOKEN.findall(text))
        if ext in DOC_EXT:
            doc_docs[0] += 1
            for t in toks: doc_df[t] += 1
        else:
            code_docs[0] += 1
            for t in toks: code_df[t] += 1
w("\n[4] REFERENCE CORPUS")
w("  code/config files=%d  doc files=%d  distinct tokens=%d(code)/%d(doc)" %
  (code_docs[0], doc_docs[0], len(code_df), len(doc_df)))
w("  (code_df = how many code/config files mention the experiment name; doc_df = how many .md files)")

def ref(name):
    if len(name) < 6:
        return (-1, -1)
    return (code_df.get(name, 0), doc_df.get(name, 0))

# ---------- 3. per-experiment table ----------
def tier_of(name, a, ch, dh):
    if a["r7"] > 0 or a["behavior"] > 0 or ch > 0:
        return "C-KEEP"
    if ch == 0 and a["r30"] == 0:
        return "A-CANDIDATE"
    return "B-CHECK"

tier_sum = collections.defaultdict(lambda: [0, 0, 0])
rows = []
for name, a in top_agg.items():
    ch, dh = ref(name)
    tier = tier_of(name, a, ch, dh)
    tier_sum[tier][0] += a["sz"]; tier_sum[tier][1] += a["n"]; tier_sum[tier][2] += 1
    rows.append((a["sz"], name, a, ch, dh, tier))
rows.sort(reverse=True, key=lambda x: x[0])

w("\n[5] TIER SUMMARY (heuristic, NOT a deletion authorization)")
for t in ["A-CANDIDATE", "B-CHECK", "C-KEEP"]:
    s, n, d = tier_sum[t]
    w("  %-12s  %10s  %8d files  %4d experiment dirs" % (t, human(s), n, d))

w("\n[6] ALL LEVEL-1 EXPERIMENT DIRS (size desc)")
w("  cols: size | files | last_mtime | code_ref | doc_ref | behavior.json | result.json/execution.json | wrote_24h | tier | name")
for sz, name, a, ch, dh, tier in rows:
    w("  %10s | %7d | %s | c=%s d=%s | beh=%3d | rj=%3d ej=%3d | 24h=%9s | %-12s | %s" %
      (human(sz), a["n"], ts(a["maxt"]), ch, dh, a["behavior"],
       a["resultjson"], a["execjson"], human(a["r24"]), tier, name))

w("\n[7] LEVEL-2 BIG DIRS TOP40")
for k, (s, n) in sorted(sec_agg.items(), key=lambda kv: -kv[1][0])[:40]:
    w("  %10s  %8d files  %s" % (human(s), n, k))

# ---------- 4. duplicate candidates ----------
groups = collections.defaultdict(list)
for p, r, b, sz, mt, ext, top, sec in files:
    if sz >= 10*1024*1024:
        groups[(b, sz)].append(p)
dup = {k: v for k, v in groups.items() if len(v) > 1}
def fp(p, sz):
    h = hashlib.md5()
    try:
        with open(p, "rb") as f:
            h.update(f.read(262144))
            if sz > 524288:
                f.seek(-262144, 2)
                h.update(f.read(262144))
    except OSError:
        return None
    return h.hexdigest()
verified = []
for (b, sz), paths in sorted(dup.items(), key=lambda kv: -kv[0][1]*(len(kv[1])-1))[:300]:
    fps = [fp(p, sz) for p in paths[:6]]
    base = next((x for x in fps if x), None)
    if base and all(x == base for x in fps if x):
        verified.append((sz, len(paths), b, paths[:6]))
w("\n[8] DUPLICATE CANDIDATES (>=10MB, same basename+size, head/tail md5 sampled)")
w("  raw groups=%d  fingerprint-verified groups (<=300 checked)=%d" % (len(dup), len(verified)))
w("  reclaimable if keeping 1 copy (verified only) = %s" %
  human(sum(sz*(cnt-1) for sz, cnt, _, _ in verified)))
for sz, cnt, b, paths in verified[:25]:
    w("  %10s x%d extra=%10s  %s" % (human(sz), cnt, human(sz*(cnt-1)), b))
    for p in paths[:3]:
        w("        %s" % p)

# ---------- 5. special / suspicious ----------
w("\n[9] DATAPACK / research-library RESIDUE")
w("  files=%d  total=%s" % (len(datapack_files), human(sum(x[0] for x in datapack_files))))
for sz, p, mt in sorted(datapack_files, reverse=True)[:20]:
    w("    %10s  %s  %s" % (human(sz), ts(mt), p))
w("\n[10] BIG .tmp FILES (>=50MB)")
w("  files=%d  total=%s" % (len(tmp_big), human(sum(x[0] for x in tmp_big))))
for sz, p, mt in sorted(tmp_big, reverse=True)[:20]:
    w("    %10s  %s  %s" % (human(sz), ts(mt), p))

SUSP = re.compile(r"(^|[_.\-])(temp|tmp|backup|bak|old|copy|checks|weights|trash|deprecated|archive|staging|stage)([_.\-]|$)", re.I)
susp_agg = collections.defaultdict(lambda: [0, 0])
for p, r, b, sz, mt, ext, top, sec in files:
    parts = r.split(os.sep)[:-1]
    hit = None
    for i, part in enumerate(parts):
        if SUSP.search(part):
            hit = os.sep.join(parts[:i+1])
            break
    if hit:
        susp_agg[hit][0] += sz; susp_agg[hit][1] += 1
w("\n[11] SUSPICIOUS DIR NAMES (temp/tmp/backup/old/_checks/_weights/...)")
for k, (s, n) in sorted(susp_agg.items(), key=lambda kv: -kv[1][0])[:35]:
    w("    %10s  %8d files  %s" % (human(s), n, k))

w("\n[12] WRITTEN IN LAST 24H (ACTIVE - must not touch)")
w("  files(>=1MB)=%d  bytes=%s" % (len(recent_files), human(sum(x[0] for x in recent_files))))
for sz, p, mt in sorted(recent_files, reverse=True)[:20]:
    w("    %10s  %s  %s" % (human(sz), ts(mt), os.path.relpath(p, TARGET)))

w("\n[13] PROTECTED: dirs containing behavior.json (hard dependency)")
for sz, name, a, ch, dh, tier in rows:
    if a["behavior"] > 0:
        w("    %10s  beh=%3d  %s" % (human(sz), a["behavior"], name))

w("\n[14] NOTES / RISKS")
w("  - code_df/doc_df use token extraction from project code+docs; a 0 score is necessary but NOT sufficient for safety.")
w("  - Project discipline: result artifacts are immutable and SHA256-registered; check ledger/manifest before any deletion.")
w("  - Historical: 2026-11-01 batch 3 moved ~305GB of old glm5 activations to Recycle Bin (restorable);")
w("    rdc_query_construction_20260913 40.9GB was permanently deleted due to bin quota; 209 behavior.json were")
w("    relocated to D:/ai2050_native/_behavior/_legacy_agent/tasks/ and scripts patched (BEHAVIOR_ROOT fallback).")
w("  - safe-delete hook: deletion goes to Recycle Bin, space is NOT freed until bin entries are purged.")
w("  - du-style apparent sizes can exceed real usage (sparse / preallocated files); trust disk_usage free delta.")
w("\n===== END (scan_sec=%.1f) =====" % (time.time()-T0))
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("DONE_GLM5_SCAN")
