import os, collections, re
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_tier_scan.txt"
res=[]

JUNC=set(); stack=[ROOT]
while stack:
    cur=stack.pop()
    try:
        with os.scandir(cur) as it:
            for e in it:
                try:
                    if not e.is_dir(follow_symlinks=False): continue
                    st=e.stat(follow_symlinks=False)
                    if getattr(st,'st_file_attributes',0) & 0x400:
                        JUNC.add(os.path.normcase(os.path.abspath(e.path)))
                    else: stack.append(e.path)
                except OSError: pass
    except OSError: pass

# find all dirs matching tier patterns, roll up non-overlapping (outermost wins)
PATTERNS = {
  "__pycache__": re.compile(r"^__pycache__$", re.I),
  "smoke":       re.compile(r"smoke", re.I),
  "pilot":       re.compile(r"pilot", re.I),
  "failed":      re.compile(r"failed", re.I),
  "tmp/scratch": re.compile(r"(^tmp|_tmp$|scratch|smoke_)", re.I),
  "_temp dir":   re.compile(r"_temp$", re.I),
}
hits=collections.defaultdict(list)
for dp,dn,fns in os.walk(ROOT):
    if os.path.normcase(os.path.abspath(dp)) in JUNC: 
        dn[:]=[]; continue
    dn[:]=[d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp,d))) not in JUNC]
    for d in dn:
        for name,rx in PATTERNS.items():
            if rx.search(d): hits[name].append(os.path.join(dp,d))

def tsize(base):
    s=0;c=0
    for a,b,cc in os.walk(base):
        for f in cc:
            try: s+=os.path.getsize(os.path.join(a,f)); c+=1
            except OSError: pass
    return s,c

def dedup(paths):
    """keep outermost only (a path whose ancestor is also in list is dropped)"""
    ps=sorted(set(os.path.normcase(os.path.abspath(p)) for p in paths), key=len)
    keep=[]
    for p in ps:
        if any(k in p for k in keep): continue
        keep.append(p)
    return keep

res.append("=== TIER CANDIDATE DIRECTORIES (outermost, junction-excluded) ===")
for name,paths in hits.items():
    k=dedup(paths)
    tot=0;cnt=0
    rows=[]
    for p in k:
        s,c=tsize(p); tot+=s; cnt+=c
        rows.append((s,c,os.path.relpath(p,ROOT)))
    rows.sort(key=lambda x:-x[0])
    res.append("\n--- %s : %d dirs, %.2f GB, %d files ---" % (name,len(k),tot/1024**3,cnt))
    for s,c,r in rows[:25]:
        res.append("   %9.1f MB %6d  %s" % (s/1024**2,c,r))
    if len(rows)>25: res.append("   ... +%d more dirs" % (len(rows)-25))

# client_visualization_assets vs result mirror overlap
res.append("\n\n=== MIRROR CHECK: glm5/result/client_visualization_assets vs tests/result ===")
A=r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\client_visualization_assets"
B=r"D:\AI2050\Ai2050-OpenOne\tests\result"
def wm(base):
    m={}
    for dp,dn,fns in os.walk(base):
        for f in fns:
            fp=os.path.join(dp,f)
            try: m[os.path.relpath(fp,base)]=os.path.getsize(fp)
            except OSError: pass
    return m
ma=wm(A); mb=wm(B)
common=set(ma)&set(mb)
res.append("A: %d files %.2f GB | B: %d files %.2f GB | same-relpath common: %d (%.2f GB)" % (
    len(ma),sum(ma.values())/1024**3,len(mb),sum(mb.values())/1024**3,len(common),sum(ma[r] for r in common)/1024**3))
aonly=[r for r in ma if r not in mb]
res.append("A-only: %d files %.2f GB" % (len(aonly), sum(ma[r] for r in aonly)/1024**3))
for r in sorted(aonly, key=lambda r:-ma[r])[:15]:
    res.append("   A-only %9.1f MB  %s" % (ma[r]/1024**2, r))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok")
