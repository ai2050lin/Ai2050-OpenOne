import os, hashlib, time
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_mirror_verify.txt"
lines=[]
def p(s): lines.append(str(s))

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
PAIRS = [("phase2746", os.path.join(ROOT,"phase2746","field_store"), os.path.join(ROOT,"phase2746_fields")),
         ("phase2747", os.path.join(ROOT,"phase2747","field_store"), os.path.join(ROOT,"phase2747_fields"))]

def h(fp):
    x=hashlib.blake2b(digest_size=16)
    with open(fp,"rb") as f:
        while True:
            b=f.read(8*1024*1024)
            if not b: break
            x.update(b)
    return x.hexdigest()

def walkmap(base):
    m={}
    for dp,dn,fns in os.walk(base):
        for fn in fns:
            fp=os.path.join(dp,fn)
            m[os.path.relpath(fp,base)]=fp
    return m

t0=time.time()
total_identical=0; total_bytes=0; mismatch=[]
for name, a, b in PAIRS:
    ma=walkmap(a); mb=walkmap(b)
    p("\n=== %s  A=%s (%d)  B=%s (%d) ===" % (name, os.path.relpath(a,ROOT), len(ma), os.path.relpath(b,ROOT), len(mb)))
    onlyA=set(ma)-set(mb); onlyB=set(mb)-set(ma); common=sorted(set(ma)&set(mb))
    p("onlyA=%d onlyB=%d common=%d" % (len(onlyA),len(onlyB),len(common)))
    ident=0; ibytes=0; n=0
    for rel in common:
        sa=os.path.getsize(ma[rel]); sb=os.path.getsize(mb[rel])
        if sa!=sb:
            mismatch.append((name,rel,"size %d vs %d"%(sa,sb))); continue
        if h(ma[rel])==h(mb[rel]):
            ident+=1; ibytes+=sa
        else:
            mismatch.append((name,rel,"content differs"))
        n+=1
        if n%800==0: p("  ...%d/%d hashed (%.0fs)"%(n,len(common),time.time()-t0))
    p("  IDENTICAL: %d/%d files, %.2f GB  (of common %.2f GB)" % (
        ident,len(common),ibytes/1024**3, sum(os.path.getsize(ma[r]) for r in common)/1024**3))
    total_identical+=ident; total_bytes+=ibytes
    for r in sorted(onlyA)[:20]: p("  ONLY-A: "+r)
    for r in sorted(onlyB)[:20]: p("  ONLY-B: "+r)

p("\n=== VERDICT ===")
p("fully identical files: %d ; reclaimable if B side removed: %.2f GB ; elapsed %.0fs" % (
    total_identical, total_bytes/1024**3, time.time()-t0))
p("mismatches: %d" % len(mismatch))
for m in mismatch[:60]: p("  MISMATCH %s | %s | %s" % m)

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(lines))
print("ok")
