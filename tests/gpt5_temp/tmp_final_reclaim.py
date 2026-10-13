import os, re, collections
GLM = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
TESTS= r"D:\AI2050\Ai2050-OpenOne\tests"
REPO = r"D:\AI2050\Ai2050-OpenOne"
OUT  = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_final_reclaim.txt"
res=[]

# collect ALL referenced result-dir names from whole repo code (server, frontend, scripts, research)
refs=collections.defaultdict(set)
rx=re.compile(r"(?:result/|result\\\\|RESULT/)['\"]?([A-Za-z0-9_.\-]{6,})")
rx2=re.compile(r"tests/glm5/result/([A-Za-z0-9_.\-]+)")
SCAN=[os.path.join(REPO,"server"), os.path.join(REPO,"scripts"), os.path.join(REPO,"research"),
      os.path.join(REPO,"frontend","src"), os.path.join(REPO,"shared")]
nfile=0
for base in SCAN:
    if not os.path.isdir(base): continue
    for dp,dn,fns in os.walk(base):
        dn[:]=[d for d in dn if d not in ("__pycache__","node_modules",".git","dist")]
        for fn in fns:
            if not fn.endswith((".py",".js",".jsx",".ts",".tsx")): continue
            fp=os.path.join(dp,fn)
            try: txt=open(fp,encoding="utf-8",errors="ignore").read()
            except OSError: continue
            nfile+=1
            for m in rx.finditer(txt): refs[m.group(1)].add(os.path.relpath(fp,REPO))
            for m in rx2.finditer(txt): refs[m.group(1)].add(os.path.relpath(fp,REPO))
res.append("scanned %d code files, found %d referenced result-dir names" % (nfile, len(refs)))

def dsize(base):
    s=0;c=0
    for a,b,cc in os.walk(base):
        for f in cc:
            try: s+=os.path.getsize(os.path.join(a,f)); c+=1
            except OSError: pass
    return s,c

tops=[]
for name in os.listdir(GLM):
    p=os.path.join(GLM,name)
    if os.path.isdir(p):
        s,c=dsize(p); tops.append((s,c,name))
tops.sort(key=lambda x:-x[0])

res.append("\n=== FINAL: referenced vs truly-unreferenced ===")
res.append("%10s %8s  %-54s %s" % ("GB","files","dir","status"))
unref=[]
for s,c,name in tops:
    hit = name in refs
    if not hit: unref.append((s,c,name))
    res.append("%10.2f %8d  %-54s %s" % (s/1024**3,c,name[:54], "referenced" if hit else "UNREFERENCED"))
res.append("")
res.append("UNREFERENCED: %d dirs, %.2f GB" % (len(unref), sum(x[0] for x in unref)/1024**3))
res.append("REFERENCED  : %d dirs, %.2f GB" % (len(tops)-len(unref), sum(x[0] for x in tops[:len(tops)-len(unref)])/1024**3))

# also: tests/result and other lines
res.append("\n=== tests/ other top dirs ===")
for name in sorted(os.listdir(TESTS)):
    p=os.path.join(TESTS,name)
    if not os.path.isdir(p) or name=="glm5": continue
    s,c=dsize(p)
    res.append("%10.2f GB %8d  %-20s %s" % (s/1024**3,c,name,"referenced" if name in refs else "UNREFERENCED"))

# glm5 non-result
res.append("\n=== tests/glm5 non-result ===")
for name in sorted(os.listdir(os.path.join(TESTS,"glm5"))):
    p=os.path.join(TESTS,"glm5",name)
    if not os.path.isdir(p): continue
    s,c=dsize(p)
    res.append("%10.2f GB %8d  %s" % (s/1024**2,c,name))

# deletable: pycache, playwright, tf457, tmp scripts, logs
res.append("\n=== TIER A concrete ===")
pyc=0;pycn=0
for dp,dn,fns in os.walk(TESTS):
    for d in list(dn):
        if d=="__pycache__":
            for a,b,cc in os.walk(os.path.join(dp,d)):
                for f in cc:
                    try: pyc+=os.path.getsize(os.path.join(a,f)); pycn+=1
                    except OSError: pass
            dn.remove(d)
res.append("__pycache__        : %8.2f MB / %d files" % (pyc/1024**2,pycn))
for nm in ("rdc_update_browser_packages",):
    p=os.path.join(TESTS,"glm5_temp",nm)
    if os.path.isdir(p):
        s,c=dsize(p); res.append("glm5_temp/%-16s: %8.2f MB / %d files"%(nm,s/1024**2,c))
p=os.path.join(TESTS,"gpt5_temp","tf457")
if os.path.isdir(p):
    s,c=dsize(p); res.append("gpt5_temp/tf457          : %8.2f MB / %d files (HF transformers 全量副本)"%(s/1024**2,c))
log=0;logn=0
for dp,dn,fns in os.walk(TESTS):
    for fn in fns:
        if fn.endswith((".log",".bak")):
            try: log+=os.path.getsize(os.path.join(dp,fn)); logn+=1
            except OSError: pass
res.append("*.log + *.bak     : %8.2f MB / %d files" % (log/1024**2,logn))

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(res))
print("ok")
