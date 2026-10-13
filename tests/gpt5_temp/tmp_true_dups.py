import os, hashlib, time, collections, json
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_true_dups.txt"

# 1) find junctions
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

def walk_real(base):
    for dp,dn,fns in os.walk(base):
        dn[:]=[d for d in dn if os.path.normcase(os.path.abspath(os.path.join(dp,d))) not in JUNC]
        for fn in fns:
            fp=os.path.join(dp,fn)
            try: yield fp, os.path.getsize(fp)
            except OSError: pass

MIN=1024*1024
t0=time.time()
by_size=collections.defaultdict(list)
total=0;n=0
for fp,sz in walk_real(ROOT):
    total+=sz;n+=1
    if sz>=MIN: by_size[sz].append(fp)
cand={s:v for s,v in by_size.items() if len(v)>1}
cand_files=sum(len(v) for v in cand.values())
res=[]
res.append("TRUE tree: %d files %.2f GB; junctions=%d" % (n, total/1024**3, len(JUNC)))
res.append("same-size groups>=1MB: %d groups / %d files (%.1f%% of tree)" % (
    len(cand), cand_files, 100.0*cand_files/max(n,1)))
res.append("upper-bound dup space: %.2f GB" % (sum(s*len(v) for s,v in cand.items())/1024**3))

def h32(fp):
    x=hashlib.blake2b(digest_size=16)
    with open(fp,"rb") as f:
        while True:
            b=f.read(8*1024*1024)
            if not b: break
            x.update(b)
    return x.hexdigest()

groups=collections.defaultdict(list); done=0
for s,files in cand.items():
    for fp in files:
        try: groups[(s,h32(fp))].append(fp)
        except OSError: pass
    done+=len(files)
    if done%1000 < len(files):
        res.append("  hashed %d/%d (%.0fs)"%(done,cand_files,time.time()-t0))
        with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))

dg={k:v for k,v in groups.items() if len(v)>1}
reclaim=sum(k[0]*(len(v)-1) for k,v in dg.items())
res.append("")
res.append("=== TRUE DUP SUMMARY (junction-excluded) ===")
res.append("dup groups: %d   reclaimable: %.2f GB   hash time %.0fs" % (len(dg), reclaim/1024**3, time.time()-t0))

res.append("")
res.append("=== TOP 25 TRUE DUP GROUPS ===")
for (sz,hh),v in sorted(dg.items(), key=lambda x:-x[0][0]*(len(x[1])-1))[:25]:
    res.append("\n[%.2f MB x%d]"%(sz/1024**2,len(v)))
    for fp in v[:6]: res.append("   "+os.path.relpath(fp,ROOT))
    if len(v)>6: res.append("   ...+%d"%(len(v)-6))

# group dups by "which side is redundant" heuristics
res.append("")
res.append("=== DUP PAIRS: same basename different dir (top 30) ===")
byname=collections.defaultdict(list)
for (sz,hh),v in dg.items():
    for fp in v: byname[os.path.basename(fp)].append((sz,fp))
cnt=0
for bn,items in sorted(byname.items(), key=lambda x:-sum(i[0] for i in x[1])):
    if len(items)<2: continue
    szs=set(i[0] for i in items)
    res.append("\n%s  (%d copies, %.2f MB each)"%(bn,len(items),items[0][0]/1024**2))
    for sz,fp in items: res.append("   "+os.path.relpath(fp,ROOT))
    cnt+=1
    if cnt>=30: break

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok %.0fs"%(time.time()-t0))
