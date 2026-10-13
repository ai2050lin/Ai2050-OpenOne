import os, ctypes, subprocess, json
OUT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_junction_check.txt"
lines=[]
def p(s): lines.append(str(s))

BASE = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913"
paths = [
 BASE+r"\phase2746\field_store",
 BASE+r"\phase2746_fields",
 BASE+r"\phase2747\field_store",
 BASE+r"\phase2747_fields",
 BASE+r"\phase2748",
 BASE+r"\capture",
 r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_conditional_campaign_20260910\i_factorial\features\state.npz",
]
ps = ";".join("'%s'" % x for x in paths)
script = (
 "$ErrorActionPreference='SilentlyContinue';"
 "foreach($p in @(" + ps + ")){"
 "$i=Get-Item -LiteralPath $p -Force;"
 "if($i){"
 "$a=($i.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0;"
 "$t=if($a){$i.Target}else{''};"
 "Write-Output ($p+'|ATTRIB='+$i.Attributes+'|REPARSE='+$a+'|LINKTYPE='+$i.LinkType+'|TARGET='+($t -join ';')+'|PS_IS='+$i.PSIsContainer);"
 "}else{Write-Output ($p+'|MISSING')}"
 "}"
)
r = subprocess.run(["powershell","-NoProfile","-Command",script], capture_output=True, text=True)
p("=== JUNCTION / LINK CHECK (via PowerShell) ===")
p(r.stdout)
p("STDERR: "+r.stderr[:500])

p("\n=== os.path checks ===")
for x in paths:
    try:
        p("%s | islink=%s | isdir=%s | exists=%s | resolve=%s" % (
            os.path.basename(x), os.path.islink(x), os.path.isdir(x), os.path.exists(x), os.path.realpath(x)))
    except Exception as e:
        p("%s ERR %s" % (x,e))

p("\n=== C:/AI2050-RDC-Archive exists? ===")
arch = r"C:\AI2050-RDC-Archive"
p("exists=%s" % os.path.exists(arch))
if os.path.exists(arch):
    for dp,dn,fns in os.walk(arch):
        rel=os.path.relpath(dp,arch)
        d=[x for x in dn]
        p("  DIR %s  (subdirs=%d)" % (rel, len(d)))
        if rel.count(os.sep)>=1: break

with open(OUT,"w",encoding="utf-8",newline="") as f: f.write("\n".join(lines))
print("ok")
