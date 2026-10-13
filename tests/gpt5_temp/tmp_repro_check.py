import os, collections, re
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_repro_check.txt"
res=[]

# A) ALL .py outside result dirs
py=collections.defaultdict(lambda:[0,0])
for dp,dn,fns in os.walk(ROOT):
    rel=os.path.relpath(dp,ROOT).split(os.sep)
    key="/".join(rel[:2]) if len(rel)>1 else "(root)"
    for fn in fns:
        if fn.endswith(".py"):
            try: sz=os.path.getsize(os.path.join(dp,fn))
            except OSError: continue
            d=py[key]; d[0]+=sz; d[1]+=1
res.append("=== .py COUNT BY DIR (all, incl. pycache-inside) ===")
tot=0
for k,(s,c) in sorted(py.items(), key=lambda x:-x[1][1])[:30]:
    res.append("%9.1f MB %6d  %s"%(s/1024**2,c,k)); tot+=c
res.append("TOTAL .py files: %d" % sum(v[1] for v in py.values()))

# B) .py NOT under any 'result' dir  == true driver scripts
res.append("\n=== DRIVER SCRIPTS (outside result/) ===")
drivers=[]
for dp,dn,fns in os.walk(ROOT):
    rel=os.path.relpath(dp,ROOT).split(os.sep)
    if "result" in rel or "results" in rel: continue
    for fn in fns:
        if fn.endswith((".py",".sh")):
            fp=os.path.join(dp,fn)
            try: sz=os.path.getsize(fp)
            except OSError: continue
            drivers.append((sz,os.path.relpath(fp,ROOT)))
res.append("count: %d  total %.1f MB" % (len(drivers), sum(s for s,_ in drivers)/1024**2))
for sz,r in sorted(drivers, key=lambda x:-x[0])[:60]:
    res.append("%9.1f KB  %s"%(sz/1024,r))

# C) .py INSIDE result dirs
res.append("\n=== .py INSIDE result/ dirs (co-located scripts) ===")
inner=collections.defaultdict(lambda:[0,0])
for dp,dn,fns in os.walk(ROOT):
    rel=os.path.relpath(dp,ROOT).split(os.sep)
    if "result" not in rel: continue
    for fn in fns:
        if fn.endswith((".py",".sh")):
            try: sz=os.path.getsize(os.path.join(dp,fn))
            except OSError: continue
            key="/".join(rel[:4]) if len(rel)>1 else "(root)"
            d=inner[key]; d[0]+=sz; d[1]+=1
res.append("total .py under result/: %d files" % sum(v[1] for v in inner.values()))
for k,(s,c) in sorted(inner.items(), key=lambda x:-x[1][1])[:30]:
    res.append("%9.1f MB %6d  %s"%(s/1024**2,c,k))

# D) do result dirs contain a runnable entry (main.py / run_*.py / driver)?
res.append("\n=== ENTRYPOINT-LIKE FILES inside result/ ===")
ent=collections.Counter()
for dp,dn,fns in os.walk(ROOT):
    rel=os.path.relpath(dp,ROOT).split(os.sep)
    if "result" not in rel: continue
    for fn in fns:
        if re.search(r"^(run|main|driver|collect|generate|compute|execute)[_-]", fn, re.I) and fn.endswith(".py"):
            ent["entry:"+fn.split('_')[0]]+=1
        elif re.search(r"^(analysis|analysis_)", fn, re.I) and fn.endswith(".py"):
            ent["analysis"]+=1
for k,v in ent.most_common(20): res.append("%6d  %s"%(v,k))
if not ent: res.append("(none found)")

# E) repo-wide: where do real analysis scripts live?
res.append("\n=== scripts/ + research/ py counts (repo driver location) ===")
for d in (r"D:\AI2050\Ai2050-OpenOne\scripts", r"D:\AI2050\Ai2050-OpenOne\research",
          r"D:\AI2050\Ai2050-OpenOne\tests\glm5", r"D:\AI2050\Ai2050-OpenOne\tests\gpt5"):
    n=0;s=0
    for dp,dn,fns in os.walk(d):
        dn[:]=[x for x in dn if x!="__pycache__"]
        for fn in fns:
            if fn.endswith(".py"):
                n+=1
                try: s+=os.path.getsize(os.path.join(dp,fn))
                except OSError: pass
    res.append("%-45s %6d .py  %8.1f MB" % (d.replace("D:\\AI2050\\Ai2050-OpenOne",""), n, s/1024**2))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok")
