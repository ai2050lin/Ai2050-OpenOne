import os, collections, re
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_layer_scan.txt"
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

# Layer classification by path semantics
LAYERS = [
 ("SCRIPT",    lambda r,fn: fn.endswith((".py",".pyc",".mjs",".js",".sh",".ps1",".bat"))),
 ("LOG",       lambda r,fn: fn.endswith((".log",".txt")) or "log" in fn.lower().split('.')[0]),
 ("CAPTURE",   lambda r,fn: re.search(r"[\\/](capture|captures|raw|collect)[\\/]", r) is not None),
 ("DERIVED",   lambda r,fn: re.search(r"[\\/](derived|field_store|_fields|features|points|gradients|adjoint)[\\/]", r) is not None),
 ("ANALYSIS",  lambda r,fn: re.search(r"[\\/](analysis|fit|figures|verification|report|commits)[\\/]", r) is not None),
 ("CLIENT",    lambda r,fn: "client" in r.lower() or "visualization" in r.lower()),
]
layer=collections.defaultdict(lambda:[0,0,collections.Counter()])
total=0
for dp,dn,fns in os.walk(ROOT):
    dn[:]=[d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp,d))) not in JUNC]
    for fn in fns:
        fp=os.path.join(dp,fn)
        try: sz=os.path.getsize(fp)
        except OSError: continue
        r=os.path.relpath(fp,ROOT)
        total+=sz
        tag="RESULT_OTHER"
        for name,rule in LAYERS:
            if rule(r,fn): tag=name; break
        d=layer[tag]; d[0]+=sz; d[1]+=1
        d[2][os.path.splitext(fn)[1].lower() or "(noext)"]+=1

res.append("=== BY SEMANTIC LAYER (true size, junction-excluded) ===")
res.append("TOTAL %.2f GB" % (total/1024**3))
for k,(sz,c,ext) in sorted(layer.items(), key=lambda x:-x[1][0]):
    top=" ".join("%s:%d"%(a,b) for a,b in ext.most_common(4))
    res.append("%10.2f GB %8d files  %-14s  [%s]" % (sz/1024**3,c,k,top))

# scripts inventory per top dir
res.append("\n=== SCRIPT INVENTORY (keep these) ===")
sc=collections.defaultdict(lambda:[0,0])
for dp,dn,fns in os.walk(ROOT):
    dn[:]=[d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp,d))) not in JUNC]
    for fn in fns:
        if not fn.endswith((".py",".pyc",".mjs")): continue
        try: sz=os.path.getsize(os.path.join(dp,fn))
        except OSError: continue
        rel=os.path.relpath(os.path.join(dp,fn),ROOT).split(os.sep)
        key="/".join(rel[:2]) if len(rel)>1 else "(root)"
        d=sc[key]; d[0]+=sz; d[1]+=1
for k,(sz,c) in sorted(sc.items(), key=lambda x:-x[1][0])[:25]:
    res.append("%9.1f MB %6d  %s" % (sz/1024**2,c,k))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok")
