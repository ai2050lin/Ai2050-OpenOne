import os, re, collections
GLM = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
MEMO= r"D:\AI2050\Ai2050-OpenOne\research\glm5\docs\AGI_GLM5_MEMO.md"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_orphan_check.txt"
res=[]

# memo text for provenance lookup
try: memo=open(MEMO,encoding="utf-8",errors="ignore").read()
except OSError: memo=""

def dsize(base):
    s=0;c=0
    for a,b,cc in os.walk(base):
        for f in cc:
            try: s+=os.path.getsize(os.path.join(a,f)); c+=1
            except OSError: pass
    return s,c

# enumerate all top-level result dirs
tops=[]
for name in os.listdir(GLM):
    p=os.path.join(GLM,name)
    if os.path.isdir(p):
        s,c=dsize(p); tops.append((s,c,name,p))
tops.sort(key=lambda x:-x[0])

SNAP=re.compile(r"(source_snapshots|code_snapshots|sources)$", re.I)
res.append("=== PER-DIR: has source snapshot? mentioned in MEMO? ===")
res.append("%-52s %9s %7s %-6s %-8s" % ("dir","GB","files","snap","memo"))
no_snap=[]; no_snap_nomemo=[]
for s,c,name,p in tops[:120]:
    has_snap=False
    for dp,dn,fns in os.walk(p):
        for d in dn:
            if SNAP.search(d): has_snap=True; break
        if has_snap: break
    mentioned = name in memo
    res.append("%-52s %9.2f %7d %-6s %-8s" % (name[:52], s/1024**3, c, "YES" if has_snap else "-", "yes" if mentioned else "-"))
    if not has_snap: no_snap.append((s,c,name,mentioned))
    if not has_snap and not mentioned: no_snap_nomemo.append((s,c,name))

res.append("\n=== NO SOURCE SNAPSHOT ===")
res.append("count=%d  total=%.2f GB" % (len(no_snap), sum(x[0] for x in no_snap)/1024**3))
for s,c,n,m in sorted(no_snap,key=lambda x:-x[0])[:40]:
    res.append("%9.2f GB %7d  %-52s memo=%s" % (s/1024**3,c,n[:52],"yes" if m else "NO"))

res.append("\n=== NO SNAPSHOT *AND* NOT IN MEMO (strongest delete candidate) ===")
res.append("count=%d  total=%.2f GB" % (len(no_snap_nomemo), sum(x[0] for x in no_snap_nomemo)/1024**3))
for s,c,n in sorted(no_snap_nomemo,key=lambda x:-x[0])[:50]:
    res.append("%9.2f GB %7d  %s" % (s/1024**3,c,n))

# how big ARE the snapshots overall?
res.append("\n=== SOURCE SNAPSHOT / SOURCES DIRS TOTAL ===")
snap_tot=0;snap_cnt=0
for dp,dn,fns in os.walk(GLM):
    for d in list(dn):
        if SNAP.search(d):
            s,c=dsize(os.path.join(dp,d))
            snap_tot+=s; snap_cnt+=1
res.append("%d dirs, %.2f MB total" % (snap_cnt, snap_tot/1024**2))

# MEMO coverage overall
res.append("\n=== MEMO coverage of top-40 result dirs ===")
cov=0
for s,c,name,p in tops[:40]:
    if name in memo: cov+=s
res.append("top-40 dirs total %.2f GB, covered by MEMO mention %.2f GB" % (sum(x[0] for x in tops[:40])/1024**3, cov/1024**3))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok")
