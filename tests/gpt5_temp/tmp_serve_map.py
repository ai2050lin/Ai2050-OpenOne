import os, re, collections, subprocess
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
SRV  = r"D:\AI2050\Ai2050-OpenOne\server"
OUT  = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_serve_map.txt"
res=[]

# 1) which result dirs are referenced by server code?
refs=collections.defaultdict(set)
rx=re.compile(r"tests[\s/\\]+(glm5|result|gpt5|codex|deepseek)[\s/\\]+result[\s/\\]+([A-Za-z0-9_.\-]+)")
rx2=re.compile(r"tests[\s/\\]+(glm5|result)[\s/\\]+([A-Za-z0-9_.\-]+)")
for dp,dn,fns in os.walk(SRV):
    dn[:]=[d for d in dn if d!="__pycache__"]
    for fn in fns:
        if not fn.endswith((".py",".js",".jsx",".ts",".tsx")): continue
        fp=os.path.join(dp,fn)
        try: txt=open(fp,encoding="utf-8",errors="ignore").read()
        except OSError: continue
        for m in rx.finditer(txt): refs[m.group(2)].add(os.path.relpath(fp,SRV))
        for m in rx2.finditer(txt): refs[m.group(2)].add(os.path.relpath(fp,SRV))

# 2) size of each result subdir + does it contain its own script?
GLM=os.path.join(ROOT,"glm5","result")
sizes={}
for dp,dn,fns in os.walk(GLM):
    s=0;c=0
    for fn in fns:
        try: s+=os.path.getsize(os.path.join(dp,fn)); c+=1
        except OSError: pass
    sizes[dp]=(s,c)

tops=collections.defaultdict(lambda:[0,0])
for dp,(s,c) in sizes.items():
    rel=os.path.relpath(dp,GLM)
    k=rel.split(os.sep)[0] if rel!="." else "(root)"
    d=tops[k]; d[0]+=s; d[1]+=c

res.append("=== result-subdir: SERVED-BY-SERVER vs ORPHAN ===")
res.append("(served = referenced by server/ code path constant)")
res.append("")
served_sz=0; orphan_sz=0
rows=[]
for k,(s,c) in tops.items():
    hit = k in refs
    if hit: served_sz+=s
    else: orphan_sz+=s
    rows.append((s,c,k,hit,sorted(refs.get(k,[]))))
rows.sort(key=lambda x:-x[0])
for s,c,k,hit,who in rows[:70]:
    res.append("%10.2f GB %7d  %-46s %s" % (s/1024**3,c,k,"SERVED" if hit else "orphan"))
    if hit:
        for w in who[:4]: res.append("        <- %s" % w)
res.append("")
res.append("SERVED total: %.2f GB   ORPHAN total: %.2f GB" % (served_sz/1024**3, orphan_sz/1024**3))

# 3) top-level tests/result and other lines: served?
res.append("\n=== other lines ===")
for line in ("result","gpt5","codex","deepseek","gemini","nfb_data"):
    base=os.path.join(ROOT,line)
    if not os.path.isdir(base): continue
    s=0;c=0;hit=line in refs
    for dp,dn,fns in os.walk(base):
        for fn in fns:
            try: s+=os.path.getsize(os.path.join(dp,fn)); c+=1
            except OSError: pass
    res.append("%-12s %9.2f GB %7d  %s" % (line, s/1024**3, c, "SERVED" if hit else "orphan"))

# 4) does each big campaign have a driver script inside tests/?
res.append("\n=== DRIVER SCRIPT PRESENCE per big result dir ===")
drivers=collections.defaultdict(int)
for dp,dn,fns in os.walk(ROOT):
    for fn in fns:
        if fn.endswith((".py",".sh",".pyc")) and ("phase" in fn.lower() or "rdc" in fn.lower()):
            rel=os.path.relpath(dp,ROOT).split(os.sep)
            key="/".join(rel[:2]) if len(rel)>1 else "(root)"
            drivers[key]+=1
for k,(s,c) in sorted(tops.items(), key=lambda x:-x[1][0])[:25]:
    res.append("%10.2f GB  %-46s  driver-like scripts found: %d" % (s/1024**3,k,drivers.get("glm5/"+k,0)+drivers.get("result/"+k,0)))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok")
