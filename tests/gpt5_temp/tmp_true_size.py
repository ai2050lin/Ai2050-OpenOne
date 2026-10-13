import os, json, time, collections
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_true_size.txt"
lines=[]
def p(s): lines.append(str(s))

# --- collect junction/reparse dirs (do NOT recurse into them) ---
JUNC = set()
def scan_reparse(base):
    stack=[base]
    while stack:
        cur=stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False) and e.is_symlink():
                            JUNC.add(os.path.normcase(os.path.abspath(e.path)))
                        elif e.is_dir(follow_symlinks=False):
                            st=e.stat(follow_symlinks=False)
                            # Windows junction: FILE_ATTRIBUTE_REPARSE_POINT = 0x400
                            if getattr(st,'st_file_attributes',0) & 0x400:
                                JUNC.add(os.path.normcase(os.path.abspath(e.path)))
                            else:
                                stack.append(e.path)
                    except OSError: pass
        except OSError: pass

t0=time.time()
scan_reparse(ROOT)
p("JUNCTION/REPARSE dirs found: %d (%.0fs)" % (len(JUNC), time.time()-t0))
for j in sorted(JUNC): p("   "+os.path.relpath(j,ROOT))

# --- walk WITHOUT following junctions ---
real_total=0; real_files=0
lvl1=collections.defaultdict(lambda:[0,0])
junc_pretend=0; junc_files=0
stack=[(ROOT, ROOT)]
seen=set()
while stack:
    cur, key = stack.pop()
    for dp,dn,fns in os.walk(cur):
        # prune junction children
        pruned=[]
        for d in dn:
            ap=os.path.normcase(os.path.abspath(os.path.join(dp,d)))
            if ap in JUNC:
                # measure but do not descend
                s=0;c=0
                for a,b,cc in os.walk(os.path.join(dp,d)):
                    for f in cc:
                        try: s+=os.path.getsize(os.path.join(a,f)); c+=1
                        except OSError: pass
                junc_pretend+=s; junc_files+=c
                continue
            pruned.append(d)
        dn[:] = pruned
        for fn in fns:
            fp=os.path.join(dp,fn)
            try: sz=os.path.getsize(fp)
            except OSError: continue
            real_total+=sz; real_files+=1
            d=lvl1[key]; d[0]+=sz; d[1]+=1
        # descend with proper key
        for d in dn:
            nk = os.path.relpath(os.path.join(dp,d), ROOT).split(os.sep)[0]
            stack.append((os.path.join(dp,d), nk))

p("\n=== TRUE SIZE (junctions NOT double counted) ===")
p("real files: %d  real bytes: %.2f GB" % (real_files, real_total/1024**3))
p("junction-reachable extra (already counted once): %.2f GB / %d files" % (junc_pretend/1024**3, junc_files))
p("naive os.walk total would be: %.2f GB" % ((real_total+junc_pretend)/1024**3))
p("\n=== TRUE LEVEL-1 ===")
for k,(sz,c) in sorted(lvl1.items(), key=lambda x:-x[1][0]):
    p("%12.1f MB %8d  %s" % (sz/1024**2,c,k))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(lines))
print("ok")
