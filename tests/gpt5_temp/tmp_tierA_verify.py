import os, collections, json, time
TESTS=r"D:\AI2050\Ai2050-OpenOne\tests"
OUT=r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tierA_verify.txt"
res=[]

JUNC=set(); stack=[TESTS]
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

total=0;n=0
lvl1=collections.defaultdict(lambda:[0,0])
for dp,dn,fns in os.walk(TESTS):
    dn[:]=[d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp,d))) not in JUNC]
    for fn in fns:
        fp=os.path.join(dp,fn)
        try: sz=os.path.getsize(fp)
        except OSError: continue
        total+=sz;n+=1
        k=os.path.relpath(dp,TESTS).split(os.sep)[0]
        d=lvl1[k]; d[0]+=sz; d[1]+=1

res.append("=== POST-TIER-A TRUE SIZE ===")
res.append("files: %d   size: %.2f GB" % (n, total/1024**3))
res.append("baseline was 403.96 GB / 346,548 files")
res.append("delta: %.2f GB reclaimed" % (403.96 - total/1024**3))
res.append("")
res.append("=== level-1 ===")
for k,(s,c) in sorted(lvl1.items(), key=lambda x:-x[1][0])[:12]:
    res.append("%12.1f MB %8d  %s" % (s/1024**2,c,k))

# critical junction still intact?
res.append("\n=== JUNCTION INTEGRITY (must be intact) ===")
base=os.path.join(TESTS,"glm5","result","rdc_query_construction_20260913")
for name in ("phase2746","phase2747"):
    fs=os.path.join(base,name,"field_store")
    fld=os.path.join(base,name+"_fields")
    ok_junc = os.path.isdir(fs)
    real = os.path.realpath(fs)
    n=0;s=0
    for a,b,cc in os.walk(fld):
        for f in cc:
            n+=1
            try: s+=os.path.getsize(os.path.join(a,f))
            except OSError: pass
    res.append("%s: field_store exists=%s -> resolves=%s | %s files %.2f GB" % (
        name, ok_junc, os.path.basename(real), n, s/1024**3))

# key served dirs still non-empty
res.append("\n=== SERVED DIRS SPOT CHECK ===")
serve=json.load(open(r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tierA_manifest.json",encoding="utf-8"))
for rel in ("glm5/result/rdc_query_construction_20260913",
            "glm5/result/client_visualization_assets/research_kernel",
            "glm5/result/rdc_feature_campaign_20260909",
            "glm5/result/rdc_conditional_campaign_20260910",
            "result/pattern_family_atlas"):
    p=os.path.join(TESTS,rel)
    n=0;s=0
    for a,b,cc in os.walk(p):
        for f in cc:
            n+=1
            try: s+=os.path.getsize(os.path.join(a,f))
            except OSError: pass
    res.append("%-58s %8.2f GB  %7d files" % (rel, s/1024**3, n))

# residual junk
res.append("\n=== RESIDUAL JUNK ===")
pc=0;pcn=0
for dp,dn,fns in os.walk(TESTS):
    if "__pycache__" in dn:
        for a,b,cc in os.walk(os.path.join(dp,"__pycache__")):
            for f in cc:
                pcn+=1
                try: pc+=os.path.getsize(os.path.join(a,f))
                except OSError: pass
res.append("__pycache__ remaining: %d files %.2f MB" % (pcn, pc/1024**2))
lg=0;lgn=0
for dp,dn,fns in os.walk(TESTS):
    for fn in fns:
        if fn.endswith((".log",".bak")):
            lgn+=1
            try: lg+=os.path.getsize(os.path.join(dp,fn))
            except OSError: pass
res.append("*.log/*.bak remaining: %d files %.2f MB" % (lgn, lg/1024**2))
for rel in (r"glm5_temp\rdc_update_browser_packages", r"gpt5_temp\tf457"):
    res.append("%-46s exists=%s" % (rel, os.path.exists(os.path.join(TESTS,rel))))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("\n".join(res))
