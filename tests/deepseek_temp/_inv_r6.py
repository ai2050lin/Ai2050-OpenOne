# -*- coding: utf-8 -*-
import os, hashlib, json
ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, r"tests\deepseek_temp\_inv_r6.txt")
r = []
def add(s): r.append(s)
def sha8(p):
    try: return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]
    except Exception as e: return "ERR"
def size(p):
    try: return os.path.getsize(p)
    except Exception: return -1

def summ(root, label):
    add("== %s : %s ==" % (label, root))
    if not os.path.isdir(root):
        add("   (NOT EXIST)"); add(""); return
    for e in sorted(os.listdir(root)):
        fp = os.path.join(root, e)
        if os.path.isdir(fp):
            n = 0; b = 0
            for dp, dn, fns in os.walk(fp):
                for fn in fns:
                    n += 1
                    try: b += os.path.getsize(os.path.join(dp, fn))
                    except Exception: pass
            add("   DIR  %-34s files=%4d bytes=%d" % (e + "/", n, b))
        else:
            add("   FILE %-34s %9d B  %s" % (e, size(fp), sha8(fp)))
    add("")

summ(os.path.join(ROOT, r"research\deepseek"), "research/deepseek")
summ(os.path.join(ROOT, r"research\deepseek\docs"), "research/deepseek/docs")
summ(os.path.join(ROOT, r"research\deepseek\atlas"), "research/deepseek/atlas")
summ(os.path.join(ROOT, r"research\gpt5\docs"), "research/gpt5/docs")
summ(os.path.join(ROOT, r"research\gpt5\atlas"), "research/gpt5/atlas")
summ(os.path.join(ROOT, r"tests\deepseek"), "tests/deepseek")
summ(os.path.join(ROOT, r"tests\deepseek_temp"), "tests/deepseek_temp")

P = os.path.join(ROOT, r"research\deepseek\docs\AGI_DEEPSEEK_MEMO.md")
add("== AGI_DEEPSEEK_MEMO.md ==")
if os.path.exists(P):
    raw = open(P, "rb").read()
    txt = raw.decode("utf-8-sig")
    add("   bytes=%d chars=%d lines=%d sha8=%s bom=%s" % (
        len(raw), len(txt), txt.count("\n") + 1, hashlib.sha256(raw).hexdigest()[:8],
        raw.startswith(b"\xef\xbb\xbf")))
    add("   --- headings ---")
    for i, l in enumerate(txt.split("\n"), 1):
        if l.startswith("## ") or l.startswith("# "):
            add("   %6d|%s" % (i, l[:120]))
else:
    add("   MISSING")
add("")

txt = "\n".join(r)
open(OUT, "w", encoding="utf-8").write(txt)
print("WROTE %s chars=%d" % (OUT, len(txt)))
