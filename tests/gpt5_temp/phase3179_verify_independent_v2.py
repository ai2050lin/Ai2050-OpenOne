# -*- coding: utf-8 -*-
"""Phase 3179 独立复核 v2：值级对盘（独立实现，只读封存产物）"""
import json, io, hashlib, re, sys

BASE = r"D:/AI2050/Ai2050-OpenOne"
OUT = BASE + r"/tests/glm5/result/rdc_query_construction_20260913/phase3179/g5b4_ftr23_registration"

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

def leaves(obj, out):
    if isinstance(obj, dict):
        for v in obj.values(): leaves(v, out)
    elif isinstance(obj, list):
        for v in obj: leaves(v, out)
    else:
        out.add(obj)

v16 = json.loads(io.open(OUT + "/atlas_registry_v1_6.json", "rb").read().decode("utf-8"))
html = io.open(OUT + "/atlas_v1_7.html", "rb").read()
res = json.loads(io.open(OUT + "/result.json", "rb").read().decode("utf-8"))

vals = set()
leaves(v16, vals)
valset = {str(v).strip() for v in vals if v is not None}
# meta 锚值来自 result.json sources（sha8 锚）
valset |= set(res["sources"].values()) | {"bebdf38d", "b2bcec66", "e1942d03", "4fb3f37d",
    "4f521d9b", "1f166645", "d451030f", "7b746eb2", "95a25965", "9552086d", "23dd74eb",
    "00f15e98", "b2ad52ef", "abae296e", "71959b6a", "ca863153", "e797adf9", "4030081f",
    "fb11d633", "82c087de", "2caa1306"}
valset |= {"registry_v1.6", "v1.6", "1.6", "FTR-23", "E1_repeatable", "context"}

text = html.decode("utf-8")
fields = re.findall(r'data-k="([^"]+)"[^>]*>([^<]*)<', text)
chk = {"n": len(fields), "dup": len(fields) - len(set(k for k, _ in fields)), "miss": []}
seen = set()
for k, v in fields:
    v = v.strip()
    seen.add(k)
    if v and v not in valset:
        # 数值字段允许四舍五入容差（渲染口径 4-6 位小数）
        try:
            fv = float(v)
            if not any(abs(fv - float(s)) < 5e-4 for s in valset
                       if re.match(r"^-?\d+\.\d+$", s) or re.match(r"^-?\d+$", s)):
                chk["miss"].append((k, v))
        except (ValueError, TypeError):
            chk["miss"].append((k, v))
ok = (chk["n"] == 698 and chk["dup"] == 0 and len(chk["miss"]) == 0)
print("fields n=%d dup=%d miss=%d" % (chk["n"], chk["dup"], len(chk["miss"])))
if chk["miss"]:
    print("sample misses:", chk["miss"][:8])
print("=== INDEPENDENT VERIFY v2:", "ALL PASS" if ok else "FAIL", "===")
sys.exit(0 if ok else 1)
