import os, collections
ROOT = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_rqc_scan.txt"
lines = []
def p(s): lines.append(str(s))

# level 1 inside rdc_query_construction
agg = collections.defaultdict(lambda:[0,0])
for dp, dn, fns in os.walk(ROOT):
    rel = os.path.relpath(dp, ROOT)
    key = rel.split(os.sep)[0] if rel != "." else "(root)"
    for fn in fns:
        try: sz = os.path.getsize(os.path.join(dp,fn))
        except OSError: continue
        d = agg[key]; d[0]+=sz; d[1]+=1
p("=== rdc_query_construction_20260913 LEVEL-1 ===")
tot=0
for k,(sz,c) in sorted(agg.items(), key=lambda x:-x[1][0]):
    p("%10.1f MB %7d  %s" % (sz/1024**2,c,k)); tot+=sz
p("TOTAL %.1f GB" % (tot/1024**3))

# phase2746 vs phase2746_fields : compare sizes per relative path
import hashlib
def walkmap(base):
    m={}
    for dp,dn,fns in os.walk(base):
        for fn in fns:
            fp=os.path.join(dp,fn)
            rel=os.path.relpath(fp,base)
            try: m[rel]=os.path.getsize(fp)
            except OSError: pass
    return m

pairs = [("phase2746","phase2746\\field_store","phase2746_fields"),
         ("phase2747","phase2747\\field_store","phase2747_fields")]
for name, a, b in pairs:
    pa=os.path.join(ROOT,a); pb=os.path.join(ROOT,b)
    if not (os.path.isdir(pa) and os.path.isdir(pb)): 
        p("\n%s: missing" % name); continue
    ma=walkmap(pa); mb=walkmap(pb)
    p("\n=== %s : %s (%d files, %.2f GB)  vs  %s (%d files, %.2f GB) ===" % (
        name,a,len(ma),sum(ma.values())/1024**3,b,len(mb),sum(mb.values())/1024**3))
    onlya = set(ma)-set(mb); onlyb = set(mb)-set(ma)
    both = set(ma)&set(mb)
    szdiff = [r for r in both if ma[r]!=mb[r]]
    p("only_in_%s: %d (%.2f GB) | only_in_%s: %d (%.2f GB) | common: %d | size-diff: %d (%.2f GB)" % (
        a,len(onlya),sum(ma[r] for r in onlya)/1024**3,
        b,len(onlyb),sum(mb[r] for r in onlyb)/1024**3,
        len(both),len(szdiff),sum(ma[r] for r in szdiff)/1024**3))
    for r in sorted(onlya)[:10]: p("   onlyA: "+r)
    for r in sorted(onlyb)[:10]: p("   onlyB: "+r)
    for r in sorted(szdiff)[:10]: p("   szdiff: %s  %d vs %d" % (r,ma[r],mb[r]))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(lines))
print("ok")
