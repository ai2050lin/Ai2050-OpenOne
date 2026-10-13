# -*- coding: utf-8 -*-
"""derive_atlas_tree.py — 语言图谱三域树派生器（design/atlas_domain_tree_scheme_v1.md）。

从 registry 最新版（tests/glm5/result/rdc_query_construction_20260913/**/atlas_registry_v1_N.json
取最大 N）+ atlas_ledger.json 派生三域树档 atlas_tree_v1.json：
  L1 域 KN/AR/GR → L2 语言族 → L3 特征 FTR-xx（registry 字段引用）→ L4 样本 SAM-xx（Phase C 槽位）
  横切：MEC-xx 机制层（多对多 encoded_by）+ cross_link 域间交叉边（SxK/RxS/RxK 的家）
  space3d 段（design/atlas_space3d_scheme_v1.md）：FTR 点位（确定性派生坐标）+ 机制/交叉边表 +
  族质心（族晕用）+ 云口径注记——真实 npz 坐标接入前界面必须标注「派生坐标，非模型真实几何」。
  gallery3d 段（design/atlas_gallery3d_scheme_v1.md）：三域塔/族楼层/证据节点/机制竖井/交叉天桥立体馆。

纪律（design §5）：
  - coverage / space3d / gallery3d 点位与边表只由本派生器计算，人手禁写；
  - 域/族/机制映射表 = 研究决策（design §2 归位表），改动须走预注册；
  - 本档是派生视图，不改动 registry / ledger 任何字节（immutable / append-only 不受影响）；
  - 输出确定性（无时间戳、无随机），tree_sha256_8 为内容 sha（排除自身字段），--check 可幂等校验。

v2 段（design/atlas_module_view_scheme_v1.md §4/§9，2026-10-13 用户确认落地）：
  追加输出 atlas_tree_v2.json = 三模块视图数据（语言模板 × 编码机制关联）：
  - TEMPLATES / TEMPLATE_MODULE / TEMPLATE_LINKS 为研究决策表（scheme §4），改动须走预注册；
  - 模板 3D 布点 = 模块聚簇 + 黄金角（确定性，非模型真实几何，界面须标注）；
  - 关联 kind：a=实测锚定 c=采集中 x=跨模块桥；缺口（无关联模板/无模板特征）只由派生器计算。

用法：python server/derive_atlas_tree.py [--check]
"""
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULT_DIR = ROOT / "tests" / "glm5" / "result" / "rdc_query_construction_20260913"
ATLAS_DIR = ROOT / "research" / "gpt5" / "atlas"
LEDGER_PATH = ATLAS_DIR / "atlas_ledger.json"
OUT_PATH = ATLAS_DIR / "atlas_tree_v1.json"

# ---- 三域映射（design §2 归位表 = 唯一事实源；26 特征全覆盖断言见 main） ----
DOMAINS = [
    {"id": "KN", "name": "知识网络",
     "desc": "实体/类/属性/关系的表征、读出与泛化边界",
     "families": [
         {"id": "KN.readout", "name": "读出几何", "features": ["FTR-01", "FTR-02", "FTR-17", "FTR-18"]},
         {"id": "KN.class_axis", "name": "类轴方向", "features": ["FTR-04"]},
         {"id": "KN.oov_limit", "name": "谱外边界", "features": ["FTR-20", "FTR-21", "FTR-22", "FTR-25", "FTR-26"]},
     ]},
    {"id": "AR", "name": "分析推理",
     "desc": "逻辑/真值/推理方向的表征与可分性",
     "families": [
         {"id": "AR.logic", "name": "逻辑方向", "features": ["FTR-03"]},
     ]},
    {"id": "GR", "name": "语法系统",
     "desc": "句式/位置/上下文的编码几何",
     "families": [
         {"id": "GR.subspace", "name": "属性·语法子空间", "features": ["FTR-06"]},
         {"id": "GR.context", "name": "上下文与位置", "features": ["FTR-12", "FTR-13", "FTR-14", "FTR-23", "FTR-24"]},
     ]},
]
CROSS_LINKS = [
    {"edge": "X1", "type": "SxK", "features": ["FTR-05"], "domains": ["GR", "KN"]},
    {"edge": "X2", "type": "RxS", "features": ["FTR-07"], "domains": ["AR", "GR"]},
    {"edge": "X3", "type": "RxK", "features": ["FTR-08"], "domains": ["AR", "KN"]},
]
MECHANISMS = [
    {"id": "MEC-01", "name": "attention 再分配执行消耗", "features": ["FTR-09", "FTR-10", "FTR-11"]},
    {"id": "MEC-02", "name": "上下文二元门控", "features": ["FTR-12"]},
    {"id": "MEC-03", "name": "RoPE 纯相对编码（输出级）", "features": ["FTR-13"]},
    {"id": "MEC-04", "name": "残差流全流分布式读出", "features": ["FTR-01"]},
    {"id": "MEC-05", "name": "读出-控制解耦（C_steer=0）", "features": ["FTR-16"]},
    {"id": "MEC-06", "name": "k*/读出双轨判定", "features": ["FTR-19"]},
    {"id": "MEC-07", "name": "局部薄邻域等价边界", "features": ["FTR-15"]},
]
LEVEL_ORDER = ["E1_repeatable", "E2_predictive", "E2_held_out", "E3_causal_scoped"]

# ================= 模块视图 v2（scheme §4 决策表；改动须走预注册） =================
OUT_PATH_V2 = ATLAS_DIR / "atlas_tree_v2.json"

# 模块元信息（色值为 UI 词表，与前端 token 对齐）
MOD_META = {
    "KN": {"name": "知识网络", "en": "KNOWLEDGE NETWORK", "color": "#0ea5e9",
           "desc": "实体/类/属性/关系的表征、读出与泛化边界"},
    "AR": {"name": "推理功能", "en": "ANALYTICAL REASONING", "color": "#d85a30",
           "desc": "逻辑/真值/推理方向的表征、可分性与行为杠杆"},
    "GR": {"name": "语法机制", "en": "GRAMMAR MECHANISM", "color": "#059669",
           "desc": "句式/位置/上下文的编码几何与门控机制"},
}
# 模板 3D 聚簇中心（派生坐标，非真实几何）
MOD_CENTERS = {"KN": (-2.6, 0.0, -1.15), "AR": (2.6, 0.0, -1.15), "GR": (0.0, 0.15, 2.45)}

# 语言模板表（scheme §4.1）：st ∈ measured/collecting/demo/pending/done
TEMPLATES = [
    {"id": "is_a", "mod": "KN", "name": "is-a 上位/下位", "pat": "{X} 是一种 {Y}",
     "tm": "TM-07", "st": "measured", "metric": "E_read=0.3316", "family": "关系族",
     "ex": "苹果 是 一种 水果 ，富含 维生素", "prod": {"nodes": 5, "cells": 312, "agg": "collecting"}},
    {"id": "attr", "mod": "KN", "name": "属性关系", "pat": "{X} 是 {ADJ} 的",
     "tm": "TM-08", "st": "demo", "metric": "", "family": "关系族",
     "ex": "石头 很 坚硬 ，海绵 却 柔软", "prod": {"nodes": 4, "cells": 388, "agg": "collecting"}},
    {"id": "part_of", "mod": "KN", "name": "部分-整体", "pat": "{X} 的 {Y}",
     "tm": "", "st": "demo", "metric": "", "family": "关系族",
     "ex": "车轮 是 汽车 的 部件", "gap": "meronymy 读出未立项——无任何特征锚定",
     "prod": {"nodes": 0, "cells": 0, "agg": "pending"}},
    {"id": "tm06", "mod": "KN", "name": "数字与计数", "pat": "{N} 个 {X}",
     "tm": "TM-06", "st": "pending", "metric": "", "family": "内容探针",
     "ex": "三个 苹果 和 五 颗 葡萄", "gap": "数词/计数机制未立项",
     "prod": {"nodes": 3, "cells": 214, "agg": "pending"}},
    {"id": "tm05", "mod": "KN", "name": "实体×位置 等变网格", "pat": "内容寻址解引用",
     "tm": "TM-05", "st": "collecting", "metric": "", "family": "结构探针",
     "ex": "把 苹果 放到 桌子 上", "prod": {"nodes": 1, "cells": 6, "agg": "collecting"}},
    {"id": "tm04", "mod": "KN", "name": "句式×知识 因子网格", "pat": "反词嵌入指纹",
     "tm": "TM-04", "st": "collecting", "metric": "", "family": "结构探针",
     "ex": "同义改写 × 实体替换 网格", "prod": {"nodes": 1, "cells": 16, "agg": "collecting"}},
    {"id": "tm03", "mod": "AR", "name": "逻辑连接词", "pat": "因为 / 但是 / 而且",
     "tm": "TM-03", "st": "collecting", "metric": "", "family": "逻辑族",
     "ex": "因为 下雨 ，所以 比赛 取消", "prod": {"nodes": 4, "cells": 1204, "agg": "collecting"}},
    {"id": "tm12", "mod": "AR", "name": "否定与转折", "pat": "不是 / 然而 / 反而",
     "tm": "TM-12", "st": "collecting", "metric": "", "family": "逻辑族",
     "ex": "这 不是 苹果 ，而是 梨", "prod": {"nodes": 3, "cells": 305, "agg": "collecting"}},
    {"id": "logic_tf", "mod": "AR", "name": "真/假命题判别", "pat": "{P} 是对的 / 错的",
     "tm": "3154/3166 实测", "st": "measured", "metric": "held-out 全过 ×3 模型", "family": "逻辑族",
     "ex": "「香蕉是水果」是对的", "prod": {"nodes": 0, "cells": 0, "agg": "done"}},
    {"id": "syntax", "mod": "GR", "name": "句法角色", "pat": "{X} {V} {Y}（语序消融）",
     "tm": "TM-09", "st": "demo", "metric": "", "family": "句法族",
     "ex": "猫 追 狗 / 狗 被 猫 追", "prod": {"nodes": 4, "cells": 96, "agg": "pending"}},
    {"id": "tm10", "mod": "GR", "name": "重复填充句", "pat": "纯位置响应 p_ℓ(pos)",
     "tm": "TM-10", "st": "done", "metric": "738 cells done", "family": "位置探针",
     "ex": "的的的的…（位置探针）", "prod": {"nodes": 2, "cells": 738, "agg": "done"}},
    {"id": "tm01", "mod": "GR", "name": "语言最小对（中↔英）", "pat": "同一语义 × 两种语言",
     "tm": "TM-01", "st": "collecting", "metric": "812 cells", "family": "最小对",
     "ex": "苹果 是 水果 / Apple is a fruit", "prod": {"nodes": 6, "cells": 812, "agg": "collecting"}},
    {"id": "tm11", "mod": "GR", "name": "多语言混合句", "pat": "中英混排 · 语码转换",
     "tm": "TM-11", "st": "collecting", "metric": "261 cells", "family": "最小对",
     "ex": "我 like 吃 apple", "prod": {"nodes": 3, "cells": 261, "agg": "collecting"}},
    {"id": "tm02", "mod": "GR", "name": "风格最小对（论文↔聊天）", "pat": "同一内容 × 两种语体",
     "tm": "TM-02", "st": "collecting", "metric": "533 cells", "family": "最小对",
     "ex": "本文论证… / 说白了…", "prod": {"nodes": 5, "cells": 533, "agg": "collecting"}},
]

# 模板 → 特征关联表（scheme §4.2）：kind a=实测锚定 c=采集中 x=跨模块桥
TEMPLATE_LINKS = [
    ("is_a", "FTR-01", "a"), ("is_a", "FTR-02", "a"), ("is_a", "FTR-17", "a"),
    ("is_a", "FTR-18", "a"), ("is_a", "FTR-04", "a"), ("is_a", "FTR-20", "a"),
    ("is_a", "FTR-21", "a"), ("is_a", "FTR-22", "a"), ("is_a", "FTR-25", "a"),
    ("is_a", "FTR-26", "a"),
    ("attr", "FTR-06", "c"),
    ("tm05", "FTR-12", "c"), ("tm04", "FTR-01", "c"),
    ("tm03", "FTR-03", "c"),
    ("tm12", "FTR-03", "c"), ("tm12", "FTR-08", "x"),
    ("logic_tf", "FTR-03", "a"), ("logic_tf", "FTR-07", "x"),
    ("syntax", "FTR-06", "c"), ("syntax", "FTR-23", "c"),
    ("tm10", "FTR-13", "a"),
    ("tm01", "FTR-14", "c"), ("tm11", "FTR-24", "c"), ("tm02", "FTR-12", "c"),
]


def sha8_bytes(b):
    return hashlib.sha256(b).hexdigest()[:8]


def latest_registry():
    cands = glob.glob(str(RESULT_DIR / "**" / "atlas_registry_v1_*.json"), recursive=True)
    if not cands:
        raise SystemExit("atlas_registry_v1_*.json not found under %s" % RESULT_DIR)
    def vkey(p):
        return int(re.search(r"v1_(\d+)\.json$", p).group(1))
    path = max(cands, key=vkey)
    ver = re.search(r"v1_(\d+)\.json$", path).group(1)
    return Path(path), ver


def feature_index(reg):
    idx = {}
    for f in reg.get("features", []):
        phases = sorted({int(a["phase"]) for a in f.get("anchors", [])
                         if isinstance(a.get("phase"), int)})
        idx[f["id"]] = {"family": f.get("family"), "evidence_level": f.get("evidence_level"),
                        "statement": f.get("statement"), "model_scope": f.get("model_scope", []),
                        "n_anchors": len(f.get("anchors", [])), "anchor_phases": phases}
    return idx


def level_bucket(lv):
    if lv == "E3_causal_scoped":
        return "e3"
    if lv in ("E2_predictive", "E2_held_out"):
        return "e2"
    return "e1"


# ---- space3d（design/atlas_space3d_scheme_v1.md §4）：确定性派生坐标，零随机 ----
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

# ---- gallery3d（design/atlas_gallery3d_scheme_v1.md §5）：立体馆确定性布局，零随机 ----
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
            for f in fam["features"]:
                in_fam[f["id"]] = fam["id"]
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
                "pos": {"origin": origin, "xyz": [round(v, 4) for v in xyz]}}

    nodes = []
    # 1) 楼层节点：族楼层盘面黄金角环布点
    for d in domains:
        for fam in d["families"]:
            c, r = fl_center[fam["id"]]
            for j, f in enumerate(fam["features"]):
                fid = f["id"]
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


def build_v2(domains, cross, mecs, coverage, idx, derived_from):
    """模块视图 v2 派生（scheme §4/§9）：模板布点 + 关联 + 缺口，全部确定性计算。"""
    tpl_ids = [t["id"] for t in TEMPLATES]
    assert len(tpl_ids) == len(set(tpl_ids)), "duplicate template id"
    for t in TEMPLATES:
        assert t["mod"] in MOD_META, "template %s bad module" % t["id"]
    for tid, fid, kind in TEMPLATE_LINKS:
        assert tid in tpl_ids, "link references unknown template %s" % tid
        assert fid in idx, "link references unknown feature %s" % fid
        assert kind in ("a", "c", "x"), "bad link kind %s" % kind

    # 模板 3D 布点：模块聚簇 + 黄金角（与 mockup 同公式，确定性）
    by_mod = {}
    for t in TEMPLATES:
        by_mod.setdefault(t["mod"], []).append(t)
    templates_out = []
    for m, arr in by_mod.items():
        c = MOD_CENTERS[m]
        for i, t in enumerate(arr):
            r = 0.62 + 0.30 * i
            a = i * GOLDEN
            pos = [round(c[0] + r * math.cos(a), 4),
                   round(c[1] + 0.30 * math.sin(i * 1.7) + 0.15 * i / len(arr), 4),
                   round(c[2] + r * math.sin(a), 4)]
            templates_out.append({**t, "pos": pos})

    links_out = [{"tpl": tid, "ftr": fid, "kind": k} for tid, fid, k in TEMPLATE_LINKS]

    # 缺口：无关联模板 / 无模板覆盖特征（域族 + 交叉特征口径；机制层独占特征不计）
    linked_tpls = {t for t, _, _ in TEMPLATE_LINKS}
    linked_ftrs = {f for _, f, _ in TEMPLATE_LINKS}
    tpl_gap = [t["id"] for t in TEMPLATES if t["id"] not in linked_tpls]
    dom_ftrs = [f["id"] for d in domains for fam in d["families"] for f in fam["features"]]
    cross_ftrs = [fid for x in cross for fid in x["features"]]
    ftr_gap = [f for f in dom_ftrs + cross_ftrs if f not in linked_ftrs]

    modules = []
    for d in domains:
        mm = MOD_META[d["id"]]
        modules.append({"id": d["id"], "name": mm["name"], "en": mm["en"],
                        "color": mm["color"], "desc": mm["desc"],
                        "coverage": coverage[d["id"]],
                        "center": list(MOD_CENTERS[d["id"]]),
                        "families": d["families"],
                        "templates": [t for t in templates_out if t["mod"] == d["id"]]})

    return {
        "schema": "atlas_tree_v2",
        "design": "design/atlas_module_view_scheme_v1.md",
        "derived_from": derived_from,
        "modules": modules,
        "links": links_out,
        "gaps": {"templates_without_feature": tpl_gap,
                 "features_without_template": ftr_gap,
                 "note": "特征口径=域族+交叉；机制层独占特征不计"},
        "cross_links": cross,
        "mechanisms": mecs,
        "space3d_note": "模板 pos 为模块聚簇+黄金角派生坐标，非模型真实几何（scheme §9）",
    }


def build():
    reg_path, reg_ver = latest_registry()
    reg_raw = reg_path.read_bytes()
    reg = json.loads(reg_raw.decode("utf-8"))
    ledger_raw = LEDGER_PATH.read_bytes()
    ledger = json.loads(ledger_raw.decode("utf-8"))
    idx = feature_index(reg)

    placed = set()

    def fsum(fid):
        if fid not in idx:
            raise SystemExit("mapping references unknown feature %s" % fid)
        placed.add(fid)
        return {"id": fid, **idx[fid]}

    domains = []
    for d in DOMAINS:
        fams = []
        for fam in d["families"]:
            fams.append({"id": fam["id"], "name": fam["name"],
                         "features": [fsum(fid) for fid in fam["features"]]})
        domains.append({"id": d["id"], "name": d["name"], "desc": d["desc"], "families": fams})

    cross = []
    for x in CROSS_LINKS:
        cross.append({**x, "detail": [fsum(fid) for fid in x["features"]]})

    mecs = []
    for m in MECHANISMS:
        fs = [fsum(fid) for fid in m["features"]]
        lv = [f["evidence_level"] for f in fs]
        floor = min(lv, key=lambda x: LEVEL_ORDER.index(x)) if lv else None
        mecs.append({"id": m["id"], "name": m["name"], "features": fs,
                     "evidence_level_floor": floor})

    missing = sorted(set(idx) - placed)
    if missing:
        raise SystemExit("features not placed in tree mapping: %s" % missing)

    # coverage：只由派生器计算。域成员 = 族特征 ∪ 指向该域的 cross_link 特征；
    # 机制层独占特征（不入域）不计入域覆盖度，单列 mechanisms 覆盖。
    coverage = {}
    for d in domains:
        ids = [f["id"] for fam in d["families"] for f in fam["features"]]
        ids += [fid for x in cross for fid in x["features"] if d["id"] in x["domains"]]
        buckets = {"e3": 0, "e2": 0, "e1": 0}
        for fid in ids:
            buckets[level_bucket(idx[fid]["evidence_level"])] += 1
        buckets["n"] = len(ids)
        buckets["e2plus"] = buckets["e2"] + buckets["e3"]
        coverage[d["id"]] = buckets
    coverage["mechanisms"] = {"n": len(mecs),
                              "e3": sum(1 for m in mecs if m["evidence_level_floor"] == "E3_causal_scoped"),
                              "note": "机制层独占特征不计入域覆盖度"}

    space3d = build_space3d(domains, cross, mecs, idx)
    assert len(space3d["points"]["ftr"]) == len(idx), "space3d must cover all features exactly once"

    gallery3d = build_gallery3d(domains, cross, mecs, idx)

    tree = {
        "schema": "atlas_tree_v1",
        "design": "design/atlas_domain_tree_scheme_v1.md",
        "derived_from": {
            "registry_version": reg_ver,
            "registry_file": reg_path.name,
            "registry_file_sha8": sha8_bytes(reg_raw),
            "ledger_measurements": len(ledger.get("measurements", [])),
            "ledger_sha256_8": ledger.get("ledger_sha256_8"),
            "ledger_file_sha8": sha8_bytes(ledger_raw),
        },
        "domains": domains,
        "cross_links": cross,
        "mechanisms": mecs,
        "evidence_samples": [],   # Phase C：从封存 npz 锚定 token 级样本后填充
        "coverage": coverage,
        "space3d": space3d,
        "gallery3d": gallery3d,
    }
    payload = json.dumps(tree, ensure_ascii=False, indent=1, sort_keys=False).encode("utf-8")
    tree["tree_sha256_8"] = sha8_bytes(payload)

    tree2 = build_v2(domains, cross, mecs, coverage, idx, tree["derived_from"])
    payload2 = json.dumps(tree2, ensure_ascii=False, indent=1, sort_keys=False).encode("utf-8")
    tree2["tree_sha256_8"] = sha8_bytes(payload2)
    return tree, tree2


def main():
    check = "--check" in sys.argv
    tree, tree2 = build()
    payload = json.dumps(tree, ensure_ascii=False, indent=1).encode("utf-8")
    payload2 = json.dumps(tree2, ensure_ascii=False, indent=1).encode("utf-8")
    if check:
        ok = True
        for path, want, tag in ((OUT_PATH, payload, "v1"), (OUT_PATH_V2, payload2, "v2")):
            if not path.is_file():
                print("CHECK FAIL(%s): %s missing" % (tag, path))
                ok = False
            elif path.read_bytes() != want:
                print("CHECK FAIL(%s): on-disk tree differs (disk=%s)" % (tag, sha8_bytes(path.read_bytes())))
                ok = False
        if ok:
            print("CHECK OK v1=%s v2=%s" % (tree["tree_sha256_8"], tree2["tree_sha256_8"]))
        else:
            sys.exit(1)
        return
    OUT_PATH.write_bytes(payload)
    OUT_PATH_V2.write_bytes(payload2)
    cov = tree["coverage"]
    print("WROTE %s (%d bytes) tree_sha256_8=%s registry=v%s ledger=n%d" % (
        OUT_PATH, len(payload), tree["tree_sha256_8"],
        tree["derived_from"]["registry_version"], tree["derived_from"]["ledger_measurements"]))
    print("WROTE %s (%d bytes) tree_sha256_8=%s templates=%d links=%d tpl_gap=%d ftr_gap=%d" % (
        OUT_PATH_V2, len(payload2), tree2["tree_sha256_8"],
        len(TEMPLATES), len(TEMPLATE_LINKS),
        len(tree2["gaps"]["templates_without_feature"]),
        len(tree2["gaps"]["features_without_template"])))
    print("coverage: " + " ".join("%s E2+%d/E1 %d(n=%d)" % (k, v["e2plus"], v["e1"], v["n"])
                                  for k, v in cov.items() if isinstance(v, dict) and "e2plus" in v))


if __name__ == "__main__":
    main()
