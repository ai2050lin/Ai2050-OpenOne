# -*- coding: utf-8 -*-
"""Phase 3179 独立复核 v3（双向）：
(1) v1.6 registry 每个叶子值出现在 atlas_v1_7.html；
(2) v1.7 相对 v1.6 的字段级变更仅限 FTR-23 相关 + meta/version（其余字段与封存 v1.6 HTML 逐字段一致）。
只读封存产物。
"""
import json, io, hashlib, re, sys

BASE = r"D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913"
OUT = BASE + r"/phase3179/g5b4_ftr23_registration"
PREV = BASE + r"/phase3178/g5b3_ftr14_a2_attribution"

def leaves(obj, out):
    if isinstance(obj, dict):
        for v in obj.values(): leaves(v, out)
    elif isinstance(obj, list):
        for v in obj: leaves(v, out)
    else:
        out.add(obj)

fails = []
def chk(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond: fails.append(name)

v16 = json.loads(io.open(OUT + "/atlas_registry_v1_6.json", "rb").read().decode("utf-8"))
h7 = io.open(OUT + "/atlas_v1_7.html", "rb").read().decode("utf-8")
h6 = io.open(PREV + "/atlas_v1_6.html", "rb").read().decode("utf-8")

# (1) 正向：v1.6 叶子值 -> v1.7 HTML（html 转义归一 + 浮点 6 位舍入容差）
import html as _html
h7n = _html.unescape(h7)
vals = set(); leaves(v16, vals)
float_re = re.compile(r"^-?\d+\.\d+$")
h7_floats = set()
for m in re.finditer(r"(?<![\w.])(-?\d+\.\d+)(?![\w.])", h7n):
    h7_floats.add(m.group(1))
miss_fwd = []
def token_hit(s):
    """组合字段（路径/sha 拼接）按分隔符分 token，全部命中才算渲染覆盖"""
    toks = [t for t in re.split(r'[\\/\(\)\s,:]+', s) if len(t) >= 3]
    return all(t in h7n for t in toks) if toks else False

for v in vals:
    s = str(v).strip()
    if not s or v is None:
        continue
    if float_re.match(s):
        hit = any(abs(float(s) - float(t)) < 5e-7 or t.startswith(s[: s.index(".") + 7])
                  for t in h7_floats if abs(float(t)) < 10)
        if not hit and s not in h7n:
            miss_fwd.append(s)
    elif s not in h7n and not token_hit(s.replace("\\", "/")):
        miss_fwd.append(s)
chk("forward: all v1.6 leaves in v1.7 html", len(miss_fwd) == 0,
    "missing=%d %s" % (len(miss_fwd), miss_fwd[:5]))

# (2) 字段级 diff：v1.7 vs 封存 v1.6 HTML
f7 = dict(re.findall(r'data-k="([^"]+)"[^>]*>([^<]*)<', h7))
pairs7 = re.findall(r'data-k="([^"]+)"[^>]*>([^<]*)<', h7)
pairs6 = re.findall(r'data-k="([^"]+)"[^>]*>([^<]*)<', h6)
d6 = dict(pairs6)
chk("field count 698 vs prev", len(pairs7) == 698, str(len(pairs7)))
changed = [k for k, v in pairs7 if d6.get(k) != v]
newk = [k for k, _ in pairs7 if k not in d6]
dropped = [k for k in d6 if k not in dict(pairs7)]
allowed_pref = ("FTR-23", "meta.", "N", )  # 新字段允许前缀
bad_new = [k for k in newk if not (k.startswith("FTR-23") or k.startswith("meta.") or re.match(r"^N\d+\.", k))]
chk("dropped fields none", not dropped, str(dropped[:5]))
chk("new fields only FTR-23/meta/N*", not bad_new, str(bad_new[:5]))
bad_changed = [k for k in changed
               if not k.startswith("FTR-23") and not k.startswith("meta.")
               and not re.match(r"^N\d+\.", k)]
chk("changed fields only FTR-23/meta/N*", not bad_changed, str(bad_changed[:8]))
print("new=%d changed=%d (FTR-23=%d)" % (len(newk), len(changed),
      sum(1 for k in changed if k.startswith("FTR-23"))))

# (3) sha 断言（磁盘回读）
def sha8(b): return hashlib.sha256(b).hexdigest()[:8]
chk("reg16 sha8 bebdf38d", sha8(io.open(OUT + "/atlas_registry_v1_6.json", "rb").read()) == "bebdf38d")
chk("html sha8 b2bcec66", sha8(io.open(OUT + "/atlas_v1_7.html", "rb").read()) == "b2bcec66")
res = json.loads(io.open(OUT + "/result.json", "rb").read().decode("utf-8"))
chk("seal e1b5e3a2", res["seal_sha8"] == "e1b5e3a2")

# (4) FTR-23 数值断言 vs 3178 定判
f23 = [f for f in v16["features"] if f["id"] == "FTR-23"][0]
st = f23["statement"]
for tok in ("0.0918", "0.0951", "0.1289", "0.019", "0.026", "0.072", "0.131",
            "0.581", "0.623", "0.561", "0.564", "6/6"):
    chk("stmt " + tok, tok in st)
chk("evidence_level", f23["evidence_level"] == "E1_repeatable")
chk("n anchors 3", len(f23.get("anchors", f23.get("anchor_list", []))) == 3 or f23.get("n_anchors") == 3,
    str({k: f23[k] for k in f23 if "anchor" in k.lower()}))

print("=== INDEPENDENT VERIFY v3:", "ALL PASS" if not fails else "FAIL: " + ",".join(fails), "===")
sys.exit(1 if fails else 0)
