import os, time, collections
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_true_size.txt"

JUNC = set()
stack=[ROOT]
t0=time.time()
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
                    else:
                        stack.append(e.path)
                except OSError: pass
    except OSError: pass

res=[]
res.append("JUNCTION/REPARSE dirs: %d (%.0fs)" % (len(JUNC), time.time()-t0))
for j in sorted(JUNC):
    res.append("   "+os.path.relpath(j,ROOT))

def tree_size(base):
    s=0;c=0
    for a,b,cc in os.walk(base):
        for f in cc:
            try: s+=os.path.getsize(os.path.join(a,f)); c+=1
            except OSError: pass
    return s,c

junc_extra=0; junc_cnt=0
for j in sorted(JUNC):
    s,c=tree_size(j)
    junc_extra+=s; junc_cnt+=c
    res.append("  junction %s -> %.2f GB / %d files" % (os.path.relpath(j,ROOT), s/1024**3, c))

lvl1=collections.defaultdict(lambda:[0,0])
real_total=0; real_files=0
for dp,dn,fns in os.walk(ROOT):
    keep=[]
    for d in dn:
        ap=os.path.normcase(os.path.abspath(os.path.join(dp,d)))
        if ap in JUNC: continue
        keep.append(d)
    dn[:]=keep
    for fn in fns:
        try: sz=os.path.getsize(os.path.join(dp,fn))
        except OSError: continue
        real_total+=sz; real_files+=1
        rel=os.path.relpath(dp,ROOT).split(os.sep)[0]
        d=lvl1[rel]; d[0]+=sz; d[1]+=1

res.append("")
res.append("=== TRUE SIZE ===")
res.append("real files: %d  = %.2f GB" % (real_files, real_total/1024**3))
res.append("junction alias (NOT extra disk): %.2f GB / %d files" % (junc_extra/1024**3, junc_cnt))
res.append("naive os.walk overcount: %.2f GB" % ((real_total+junc_extra)/1024**3))
res.append("")
res.append("=== TRUE LEVEL-1 ===")
for k,(sz,c) in sorted(lvl1.items(), key=lambda x:-x[1][0]):
    res.append("%12.1f MB %8d  %s" % (sz/1024**2,c,k))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok %.0fs" % (time.time()-t0))
