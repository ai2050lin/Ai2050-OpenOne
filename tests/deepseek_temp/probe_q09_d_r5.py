# -*- coding: utf-8 -*-
"""R5 探针 D：从 3151/3152 result 现场取 K1 逐模型数字 + MDE。只读。"""
import os, re, json, glob, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
OUT = os.path.join(ROOT, "tests", "deepseek_temp", "_review", "probe_q09_d_r5.txt")
R = []
def A(s=""):
    R.append(s)
def sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]

BASE = os.path.join(ROOT, "tests", "glm5", "result", "rdc_query_construction_20260913")

A("=== 3152 目录树 ===")
d3152 = os.path.join(BASE, "phase3152")
for dirpath, dirnames, filenames in os.walk(d3152):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        A("  %-72s %9d B %s" % (os.path.relpath(fp, BASE), os.path.getsize(fp), sha8(fp)))

A("")
A("=== 3151 目录树 ===")
d3151 = os.path.join(BASE, "phase3151")
for dirpath, dirnames, filenames in os.walk(d3151):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        A("  %-72s %9d B %s" % (os.path.relpath(fp, BASE), os.path.getsize(fp), sha8(fp)))

def walk_find_keys(obj, want, path="$", out=None, depth=0):
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = str(k)
            if any(w.lower() in kk.lower() for w in want):
                out.append(("%s.%s" % (path, kk), v))
            walk_find_keys(v, want, "%s.%s" % (path, kk), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            walk_find_keys(v, want, "%s[%d]" % (path, i), out, depth + 1)
    return out

A("")
A("=== 3152 result 中与 K1 相关的键（mde / readout / k_star / margin / above / m1） ===")
for arm in ["qwen3-4b", "qwen3-14b", "glm4k1", "summary"]:
    d = os.path.join(d3152, "g1p2_tri_model_k1", arm)
    if not os.path.isdir(d):
        A("  [%s] MISSING" % arm); continue
    for f in sorted(os.listdir(d)):
        if not f.startswith("result") or not f.endswith(".json"):
            continue
        fp = os.path.join(d, f)
        try:
            obj = json.loads(open(fp, "rb").read().decode("utf-8-sig"))
        except Exception as e:
            A("  [%s/%s] parse err %s" % (arm, f, e)); continue
        A("")
        A("--- %s/%s (sha8 %s) ---" % (arm, f, sha8(fp)))
        hits = walk_find_keys(obj, ["mde", "readout", "k_star", "kstar", "margin", "above", "m1", "b4"])
        seen = set()
        for p, v in hits:
            key = (p.split(".")[-1], str(type(v)))
            if isinstance(v, (int, float, bool)):
                A("   %-64s = %s" % (p[:64], v))
            elif isinstance(v, list) and len(v) <= 12 and all(isinstance(x, (int, float)) for x in v):
                A("   %-64s = %s" % (p[:64], v))
            elif key not in seen:
                seen.add(key)
                A("   %-64s : %s" % (p[:64], str(v)[:100]))

A("")
A("=== 3151 result 相关键 ===")
for f in sorted(glob.glob(os.path.join(d3151, "**", "result*.json"), recursive=True)):
    try:
        obj = json.loads(open(f, "rb").read().decode("utf-8-sig"))
    except Exception as e:
        A("  %s parse err %s" % (f, e)); continue
    A("--- %s (sha8 %s) ---" % (os.path.relpath(f, BASE), sha8(f)))
    hits = walk_find_keys(obj, ["mde", "readout", "margin", "above", "b4", "m1"])
    for p, v in hits:
        if isinstance(v, (int, float, bool)):
            A("   %-64s = %s" % (p[:64], v))
        elif isinstance(v, list) and len(v) <= 12 and all(isinstance(x, (int, float)) for x in v):
            A("   %-64s = %s" % (p[:64], v))

open(OUT, "w", encoding="utf-8").write("\n".join(R))
print("WROTE", OUT, len(R), "lines")
