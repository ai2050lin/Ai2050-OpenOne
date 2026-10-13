# -*- coding: utf-8 -*-
"""Phase 3179 独立复核（独立实现，只读封存产物；FAIL 只修 verify 不动封存产物）"""
import json, io, hashlib, re, sys

BASE = r"D:/AI2050/Ai2050-OpenOne"
OUT = BASE + r"/tests/glm5/result/rdc_query_construction_20260913/phase3179/g5b4_ftr23_registration"
V15 = BASE + r"/tests/glm5/result/rdc_query_construction_20260913/phase3178/g5b3_ftr14_a2_attribution/atlas_registry_v1_5.json"

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

fails = []
def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond:
        fails.append(name)

# 1. 文件 sha 断言（磁盘回读口径）
r16 = io.open(OUT + "/atlas_registry_v1_6.json", "rb").read()
html = io.open(OUT + "/atlas_v1_7.html", "rb").read()
res = json.loads(io.open(OUT + "/result.json", "rb").read().decode("utf-8"))
chk("registry_v1_6 sha8 == bebdf38d", sha8(r16) == "bebdf38d", sha8(r16))
chk("atlas_v1_7 sha8 == b2bcec66", sha8(html) == "b2bcec66", sha8(html))
chk("result seal == e1b5e3a2", res["seal_sha8"] == "e1b5e3a2")

# 2. registry v1.6 结构：23 特征、FTR-23 就位、其余字段与 v1.5 字节保全
v15 = json.loads(io.open(V15, "rb").read().decode("utf-8"))
v16 = json.loads(r16.decode("utf-8"))
f15 = {f["id"]: json.dumps(f, sort_keys=True, ensure_ascii=False) for f in v15["features"]}
f16 = {f["id"]: json.dumps(f, sort_keys=True, ensure_ascii=False) for f in v16["features"]}
chk("features 22->23", len(f15) == 22 and len(f16) == 23, str(len(f15)) + "->" + str(len(f16)))
chk("FTR-23 present", "FTR-23" in f16 and "FTR-23" not in f15)
same = all(f16[k] == f15[k] for k in f15)
chk("22 legacy features byte-preserved", same)
# failures / upgrade_log / gaps / appendix 不动
for key in ("failures", "upgrade_log", "gaps", "appendix"):
    if key in v15 and key in v16:
        chk("untouched: " + key,
            json.dumps(v15[key], sort_keys=True, ensure_ascii=False) ==
            json.dumps(v16[key], sort_keys=True, ensure_ascii=False))
chk("lineage version bumped", str(v16.get("version", "")) >= "1.6", str(v16.get("version")))

# 3. FTR-23 内容关键数值断言（对 3178 定判值）
f23 = json.loads(f16["FTR-23"])
st = f23["statement"]
for tok in ("0.0918", "0.0951", "0.1289"):
    chk("FTR-23 statement contains " + tok, tok in st)
chk("FTR-23 statement non-trivial", len(st) >= 200, "len=%d" % len(st))
chk("FTR-23 evidence_level", f23.get("evidence_level") == "E1_repeatable")
chk("FTR-23 family", f23.get("family") == "context")

# 4. HTML 逐字段校验（独立 flatten 重算）
# 独立 flatten（点路径，与渲染器 data-k 语义对齐）
def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, prefix + "." + str(k) if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, prefix + "." + str(i) if prefix else str(i)))
    else:
        out[prefix] = obj
    return out

flat = flatten(v16)
metas = re.findall(r'data-k="([^"]+)"', html.decode("utf-8"))
n_found = len(metas)
dups = n_found - len(set(metas))
missing = [k for k in set(metas) if k not in flat]
extra = [k for k in flat if k not in set(metas)]
chk("html fields 698/698", n_found == 698, str(n_found))
chk("html no dups/missing", dups == 0 and len(missing) == 0,
    "dups=%d missing=%d" % (dups, len(missing)))
# 值抽查：data-k 值与 flatten 值一致（抽 12 个字段）
mismatch = []
html_text = html.decode("utf-8")
for k in sorted(set(metas))[:12]:
    m = re.search(r'data-k="%s"[^>]*>([^<]*)<' % re.escape(k), html_text)
    if m is None or str(flat.get(k, "")) not in m.group(1):
        mismatch.append(k)
chk("spot value match 12 fields", not mismatch, str(mismatch))

# 5. result.json counts
c = res["counts"]
chk("counts 23/16/5/4/5", (c["features"], c["failures"], c["upgrades"], c["gaps"], c["appendix"]) == (23, 16, 5, 4, 5))

print("=== INDEPENDENT VERIFY:", "ALL PASS" if not fails else "FAIL: " + ",".join(fails), "===")
sys.exit(1 if fails else 0)
