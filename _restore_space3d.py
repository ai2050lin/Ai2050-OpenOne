# -*- coding: utf-8 -*-
"""restore_space3d.py — 恢复 derive_atlas_tree.py 的 space3d 段（内容与回退前一致，断言防护）"""
from pathlib import Path

p = Path("server/derive_atlas_tree.py")
s = p.read_bytes().decode("utf-8")

BLOCK = '''# ---- space3d（design/atlas_space3d_scheme_v1.md §4）：确定性派生坐标，零随机 ----
GOLDEN = 2.399963229728653   # 黄金角（rad）：族环布点与族内偏移共用，保证分布均匀且确定


def _round3(v):
    return [round(x, 4) for x in v]


def build_space3d(domains, cross, mecs, idx):
    di = {d["id"]: i for i, d in enumerate(domains)}
    # 族质心：黄金角环（xz 平面），y 按域序抬升
    fam_cent, k = {}, 0
    for d in domains:
        for fam in d["families"]:
            ang = k * GOLDEN
            fam_cent[fam["id"]] = (round(9.0 * math.cos(ang), 4),
                                   round(di[d["id"]] * 7.0, 4),
                                   round(9.0 * math.sin(ang), 4))
            k += 1
    dom_cent = {}
    for d in domains:
        cs = [fam_cent[f["id"]] for f in d["families"]]
        dom_cent[d["id"]] = tuple(round(sum(c[i] for c in cs) / len(cs), 4) for i in range(3))

    def ftr_point(fid, origin, xyz):
        m = idx[fid]
        return {"ftr_id": fid, "family": m["family"], "e_level": m["evidence_level"],
                "statement": m["statement"],
                "n_anchors": m["n_anchors"], "models": m["model_scope"],
                "anchor_phases": m["anchor_phases"],
                "pos": {"origin": origin, "xyz": _round3(xyz)}}

    points = []
    for d in domains:
        for fam in d["families"]:
            c = fam_cent[fam["id"]]
            for j, f in enumerate(fam["features"]):
                a2 = j * GOLDEN
                off = (2.4 * math.cos(a2), ((j % 3) - 1) * 1.4, 2.4 * math.sin(a2))
                points.append(ftr_point(f["id"], "family_centroid",
                                        [c[i] + off[i] for i in range(3)]))
    for x in cross:
        ds = x["domains"]
        c = tuple((dom_cent[ds[0]][i] + dom_cent[ds[1]][i]) / 2 for i in range(3))
        for j, f in enumerate(x["detail"]):
            a2 = (j + 1) * GOLDEN
            off = (2.0 * math.cos(a2), 0.0, 2.0 * math.sin(a2))
            p = ftr_point(f["id"], "cross_midpoint", [c[i] + off[i] for i in range(3)])
            p["cross"] = x["edge"]
            points.append(p)
    # 机制层独占特征（不入域/交叉）：外圈机制环（半径 13），归 MEC 边表相连
    in_space = {p["ftr_id"] for p in points}
    mi = 0
    for m in mecs:
        for f in m["features"]:
            if f["id"] in in_space:
                continue
            ang = mi * GOLDEN
            xyz = (13.0 * math.cos(ang), 3.5, 13.0 * math.sin(ang))
            p = ftr_point(f["id"], "mech_ring", list(xyz))
            p["mech"] = m["id"]
            points.append(p)
            mi += 1

    families = []
    for d in domains:
        for fam in d["families"]:
            lv = [level_bucket(idx[f["id"]]["evidence_level"]) for f in fam["features"]]
            families.append({"family": fam["id"], "domain": d["id"],
                             "centroid": _round3(fam_cent[fam["id"]]),
                             "n_features": len(fam["features"]),
                             "e2plus": lv.count("e2") + lv.count("e3"), "e1": lv.count("e1")})

    return {
        "schema": "space3d.v1",
        "design": "design/atlas_space3d_scheme_v1.md",
        "points": {"ftr": points, "families": families},
        "edges": {"mech": [{"id": m["id"], "ftr_ids": [f["id"] for f in m["features"]]} for m in mecs],
                  "cross": [{"edge": x["edge"], "ftr_ids": [f["id"] for f in x["detail"]],
                             "domains": x["domains"]} for x in cross]},
        "cloud": {"source": "derived", "n_points": 1024, "layer": "L6",
                  "note": "FTR 点位与背景云均为派生坐标（确定性公式），非模型真实几何；"
                          "npz 真实坐标管线接入后替换（scheme D 步），界面须保留口径标注"},
    }
'''

subs = [
    ("  横切：MEC-xx 机制层（多对多 encoded_by）+ cross_link 域间交叉边（SxK/RxS/RxK 的家）\n",
     "  横切：MEC-xx 机制层（多对多 encoded_by）+ cross_link 域间交叉边（SxK/RxS/RxK 的家）\n"
     "  space3d 段（design/atlas_space3d_scheme_v1.md）：FTR 点位（确定性派生坐标）+ 机制/交叉边表 +\n"
     "  族质心（族晕用）+ 云口径注记——真实 npz 坐标接入前界面必须标注「派生坐标，非模型真实几何」。\n", 1),
    ("  - coverage 只由本派生器计算，人手禁写；",
     "  - coverage / space3d 点位与边表只由本派生器计算，人手禁写；", 1),
    ("import json\nimport re", "import json\nimport math\nimport re", 1),
    ('    return "e1"\n\n\ndef build():',
     '    return "e1"\n\n\n' + BLOCK + "\ndef build():", 1),
    ('                              "note": "机制层独占特征不计入域覆盖度"}\n\n    tree = {',
     '                              "note": "机制层独占特征不计入域覆盖度"}\n\n'
     '    space3d = build_space3d(domains, cross, mecs, idx)\n'
     '    assert len(space3d["points"]["ftr"]) == len(idx), "space3d must cover all features exactly once"\n\n'
     '    tree = {', 1),
    ('        "coverage": coverage,\n    }',
     '        "coverage": coverage,\n        "space3d": space3d,\n    }', 1),
]
for old, new, cnt in subs:
    c = s.count(old)
    assert c == cnt, "ASSERT FAIL: %r count=%d want=%d" % (old[:60], c, cnt)
    s = s.replace(old, new)
p.write_bytes(s.encode("utf-8"))
print("RESTORED server/derive_atlas_tree.py")
