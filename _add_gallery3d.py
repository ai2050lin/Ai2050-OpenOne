# -*- coding: utf-8 -*-
"""add_gallery3d.py — derive_atlas_tree.py 增补 gallery3d 段（立体馆，确定性布局，断言防护）"""
from pathlib import Path

p = Path("server/derive_atlas_tree.py")
s = p.read_bytes().decode("utf-8")

BLOCK = '''# ---- gallery3d（design/atlas_gallery3d_scheme_v1.md §5）：立体馆确定性布局，零随机 ----
GAL_TOWER_R = 14.0     # 三塔圆环布位半径
GAL_TOWER_BODY = 6.0   # 塔体半径
GAL_FLOOR_H = 6.0      # 楼层层距（装饰性：层高=族序，无信息语义，界面须注记）
GAL_SHAFT_RING = 5.5   # 中庭竖井环半径


def build_gallery3d(domains, cross, mecs, idx):
    towers, t_pos = [], {}
    for k, d in enumerate(domains):
        ang = k * 2.0 * math.pi / 3.0
        pos = (round(GAL_TOWER_R * math.cos(ang), 4), round(GAL_TOWER_R * math.sin(ang), 4))
        t_pos[d["id"]] = pos
        towers.append({"domain": d["id"], "angle": round(ang, 4), "pos": list(pos),
                       "n_floors": len(d["families"]),
                       "n_features": sum(len(f["features"]) for f in d["families"])})

    floors, fl_center = [], {}
    for d in domains:
        for j, fam in enumerate(d["families"]):
            r = round(3.5 + 1.6 * math.sqrt(len(fam["features"])), 4)
            cx, cz = t_pos[d["id"]]
            c = [cx, round(3.0 + j * GAL_FLOOR_H, 4), cz]
            fl_center[fam["id"]] = (c, r)
            floors.append({"family": fam["id"], "tower": d["id"], "level": j,
                           "radius": r, "center": c, "n_features": len(fam["features"])})

    in_fam = {}
    for d in domains:
        for fam in d["families"]:
            for fid in fam["features"]:
                in_fam[fid] = fam["id"]
    cross_fids = {fid for x in cross for fid in x["features"]}
    shaft_of = {}
    for m in mecs:
        for f in m["features"]:
            shaft_of.setdefault(f["id"], []).append(m["id"])

    def node(fid, home, family, origin, xyz):
        m = idx[fid]
        return {"ftr_id": fid, "home": home, "family": family,
                "e_level": m["evidence_level"], "statement": m["statement"],
                "n_anchors": m["n_anchors"], "models": m["model_scope"],
                "anchor_phases": m["anchor_phases"], "shafts": shaft_of.get(fid, []),
                "pos": {"origin": origin, "xyz": [_round3(v) for v in xyz]}}

    nodes = []
    # 1) 楼层节点：族楼层盘面黄金角环布点
    for d in domains:
        for fam in d["families"]:
            c, r = fl_center[fam["id"]]
            for j, fid in enumerate(fam["features"]):
                ang = j * GOLDEN
                rr = r * 0.62
                nodes.append(node(fid, "floor", fam["id"], "floor_disc",
                                  [c[0] + rr * math.cos(ang), c[1] + 1.2, c[2] + rr * math.sin(ang)]))
    # 2) 天桥节点：交叉特征悬于两塔之间桥面（y=12 固定桥高）
    bridges = []
    for x in cross:
        da, db = x["domains"]
        ax, az = t_pos[da]
        bx, bz = t_pos[db]
        mid = [(ax + bx) / 2, 12.0, (az + bz) / 2]
        for j, fid in enumerate(x["features"]):
            ang = (j + 1) * GOLDEN
            nodes.append(node(fid, "bridge", None, "bridge_mid",
                              [mid[0] + 1.2 * math.cos(ang), mid[1], mid[2] + 1.2 * math.sin(ang)]))
        bridges.append({"edge": x["edge"], "type": x["type"], "from": da, "to": db,
                        "ftr_ids": [f["id"] for f in x["detail"]], "mid": _round3(mid)})
    # 3) 中庭节点：机制层独占特征立于中央中庭（各 MEC 竖井柱脚环）
    atrium_used = {}
    for mi, m in enumerate(mecs):
        exc = [f["id"] for f in m["features"]
               if f["id"] not in in_fam and f["id"] not in cross_fids]
        ang0 = mi * GOLDEN
        bx, bz = GAL_SHAFT_RING * math.cos(ang0), GAL_SHAFT_RING * math.sin(ang0)
        for j, fid in enumerate(exc):
            ang = ang0 + (j + 1) * 0.7
            nodes.append(node(fid, "atrium", None, "atrium_ring",
                              [bx + 1.8 * math.cos(ang), 1.2, bz + 1.8 * math.sin(ang)]))
        atrium_used[m["id"]] = (bx, bz)

    # 覆盖断言：26 特征恰好一次；井/桥端点必须命中已登记节点
    ids = [n["ftr_id"] for n in nodes]
    assert len(ids) == len(set(ids)) == len(idx), "gallery3d nodes must cover features exactly once"
    assert set(ids) == set(idx), "gallery3d node set mismatch"
    node_ids = set(ids)

    # 竖井：柱脚=中庭环位，柱顶=该机制特征最高节点 +1；横杆连每个特征节点
    shafts = []
    for m in mecs:
        base = atrium_used.get(m["id"])
        if base is None:
            ang = GOLDEN * (len(atrium_used) + 0.5)
            base = (round(GAL_SHAFT_RING * math.cos(ang), 4), round(GAL_SHAFT_RING * math.sin(ang), 4))
        fids = [f["id"] for f in m["features"]]
        ys = [n["pos"]["xyz"][1] for n in nodes if n["ftr_id"] in fids]
        shafts.append({"id": m["id"], "ftr_ids": fids,
                       "base": [round(base[0], 4), 0.0, round(base[1], 4)],
                       "top": round(max(ys) + 1.0, 4)})

    return {
        "schema": "gallery3d.v1",
        "design": "design/atlas_gallery3d_scheme_v1.md",
        "towers": towers,
        "floors": floors,
        "nodes": nodes,
        "shafts": shafts,
        "bridges": bridges,
        "note": "立体馆为结构示意（确定性派生布局）：楼层高度=族序（装饰性）、塔高不编码信息；"
                "npz 真实坐标接入后仅替换节点 pos，馆体骨架不变",
    }


'''

subs = [
    ("  族质心（族晕用）+ 云口径注记——真实 npz 坐标接入前界面必须标注「派生坐标，非模型真实几何」。\n",
     "  族质心（族晕用）+ 云口径注记——真实 npz 坐标接入前界面必须标注「派生坐标，非模型真实几何」。\n"
     "  gallery3d 段（design/atlas_gallery3d_scheme_v1.md）：三域塔/族楼层/证据节点/机制竖井/交叉天桥立体馆。\n", 1),
    ("  - coverage / space3d 点位与边表只由本派生器计算，人手禁写；",
     "  - coverage / space3d / gallery3d 点位与边表只由本派生器计算，人手禁写；", 1),
    ("\n\ndef build():", "\n\n" + BLOCK + "def build():", 1),
    ('    assert len(space3d["points"]["ftr"]) == len(idx), "space3d must cover all features exactly once"\n\n    tree = {',
     '    assert len(space3d["points"]["ftr"]) == len(idx), "space3d must cover all features exactly once"\n\n'
     '    gallery3d = build_gallery3d(domains, cross, mecs, idx)\n\n    tree = {', 1),
    ('        "space3d": space3d,\n    }',
     '        "space3d": space3d,\n        "gallery3d": gallery3d,\n    }', 1),
]
for old, new, cnt in subs:
    c = s.count(old)
    assert c == cnt, "ASSERT FAIL: %r count=%d want=%d" % (old[:60], c, cnt)
    s = s.replace(old, new)
p.write_bytes(s.encode("utf-8"))
print("ADDED gallery3d section")
