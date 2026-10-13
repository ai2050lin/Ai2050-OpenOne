import os, collections, time
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_trash_scan.txt"
lines=[]
def p(s): lines.append(str(s))

# A) cache / junk categories across whole tests tree
cats = collections.defaultdict(lambda:[0,0])
CAT_RULES = [
    ("__pycache__", lambda fp: "__pycache__" in fp.split(os.sep)),
    (".pyc", lambda fp: fp.endswith(".pyc")),
    ("tmp_* script", lambda fp: os.path.basename(fp).startswith("tmp_")),
    ("*.log", lambda fp: fp.endswith(".log")),
    ("*.bak", lambda fp: fp.endswith(".bak")),
    ("_temp dir", lambda fp: any(x.endswith("_temp") for x in fp.split(os.sep))),
    ("smoke", lambda fp: "smoke" in fp.lower()),
    ("*.tmp/*~", lambda fp: fp.endswith((".tmp","~"))),
]
for dp,dn,fns in os.walk(ROOT):
    for fn in fns:
        fp=os.path.join(dp,fn)
        try: sz=os.path.getsize(fp)
        except OSError: continue
        for name,rule in CAT_RULES:
            try:
                if rule(fp):
                    d=cats[name]; d[0]+=sz; d[1]+=1
                    break
            except Exception: pass
p("=== JUNK CATEGORY (exclusive-first-match) ===")
for k,(sz,c) in sorted(cats.items(), key=lambda x:-x[1][0]):
    p("%10.1f MB %8d  %s" % (sz/1024**2,c,k))

# B) *_temp directory sizes
p("\n=== *_temp DIRECTORIES ===")
for dp,dn,fns in os.walk(ROOT):
    for d in list(dn):
        if d.endswith("_temp"):
            base=os.path.join(dp,d)
            sz=0;cnt=0
            for a,b,cc in os.walk(base):
                for f in cc:
                    try: sz+=os.path.getsize(os.path.join(a,f)); cnt+=1
                    except OSError: pass
            p("%10.1f MB %7d  %s" % (sz/1024**2,cnt,os.path.relpath(base,ROOT)))

# C) .npy/.npz by "raw"/"derived"/"cache"-like dir names
p("\n=== npz/npy BY NAME HINT ===")
hints=collections.defaultdict(lambda:[0,0])
for dp,dn,fns in os.walk(ROOT):
    for fn in fns:
        if not fn.endswith((".npz",".npy")): continue
        try: sz=os.path.getsize(os.path.join(dp,fn))
        except OSError: continue
        low=os.path.relpath(os.path.join(dp,fn),ROOT).lower()
        tag="other"
        for k in ("cache","smoke","tmp","derived","raw","backup","field_store","scratch"):
            if k in low: tag=k; break
        d=hints[tag]; d[0]+=sz; d[1]+=1
for k,(sz,c) in sorted(hints.items(), key=lambda x:-x[1][0]):
    p("%12.1f MB %8d  %s" % (sz/1024**2,c,k))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(lines))
print("ok")
