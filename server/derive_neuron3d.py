# -*- coding: utf-8 -*-
"""derive_neuron3d.py —— 神经元 3D 派生器（design/atlas_neuron3d_scheme_v1.md §4）

从 registry 最新版（glob 自动发现，与 server/derive_atlas_tree.py latest_registry 同口径）
+ 封存 collect npz（phase3169 g5a6_oov_panel，三模型）+ phase3188 readout 层位，
派生 atlas_neuron3d_v1.json：

  schema neuron3d.v1：
  - grid      每模型 (n_layers, d_model)，来源 = 封存 npz H 形状实测（非记忆值）
  - points    registered 白名单：
                · feature 标记（26 FTR，特征级，无原生坐标，origin=feature_level_no_native_coord）
                · readout_layer 标记（phase3188 kout 实测层位）
  - npz       数据清单（相对路径 / file_sha256_8 / shape），供 /atlas/neuron 端点按需加载
  - labels    policy=registry_only（前端零标签生成；未入册点只显示数字证据）

纪律（延续 derive_atlas_tree 红线）：
  1. 白名单 / 边表 / grid 全部由本派生器计算，--check 幂等，人手禁写；
  2. 坐标含义诚实标注：feature 标记不是神经元坐标；网格点 = 原生坐标索引，非模型真实几何；
  3. 只读输入：registry / npz / phase 结果一律不修改。

用法：
  python server/derive_neuron3d.py           # 派生并写 atlas_neuron3d_v1.json
  python server/derive_neuron3d.py --check   # 幂等校验（重算比对，不改文件）
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATLAS_DIR = ROOT / "research" / "gpt5" / "atlas"
RESULT_DIR = ROOT / "tests" / "glm5" / "result" / "rdc_query_construction_20260913"
OUT_PATH = ATLAS_DIR / "atlas_neuron3d_v1.json"
DESIGN = "design/atlas_neuron3d_scheme_v1.md"

MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]
NPZ_REL = "tests/glm5/result/rdc_query_construction_20260913/phase3169/g5a6_oov_panel/collect_{m}.npz"
P3188 = RESULT_DIR / "phase3188" / "g5c1_gap4_dv_attribution" / "final_result.json"


def sha8_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


def latest_registry():
    """glob 递归发现 atlas_registry_v1_N.json，取最大 N（与 derive_atlas_tree 同口径）。"""
    import glob
    import re
    cands = []
    for s in glob.glob(str(RESULT_DIR / "**" / "atlas_registry_v1_*.json"), recursive=True):
        m = re.search(r"v1_(\d+)\.json$", s)
        if m:
            cands.append((int(m.group(1)), Path(s)))
    cands.sort(key=lambda x: x[0])
    return cands[-1] if cands else (None, None)


def npz_shape(path: Path):
    """只读 npz 头拿 H 形状：zip 目录 + header 解析，不展开 300-700MB 数据体。"""
    import struct
    import zipfile
    with zipfile.ZipFile(path) as zf:
        # numpy npz: entry "<name>.npy"；读取 H.npy 头部前 128 字节足够拿到 header
        name = [n for n in zf.namelist() if n.endswith("H.npy")][0]
        with zf.open(name) as fh:
            head = fh.read(128)
    # npy header: magic \x93NUMPY, ver, header_len(u16/u32), dict
    assert head[:6] == b"\x93NUMPY", "not a npy entry"
    major = head[6]
    if major == 1:
        hlen = struct.unpack("<H", head[8:10])[0]
        dict_start = 10
    else:
        hlen = struct.unpack("<I", head[8:12])[0]
        dict_start = 12
    header = head[dict_start:dict_start + hlen].decode("latin1")
    # 从 header dict 提取 shape 与 descr（避免 eval）
    import ast
    d = ast.literal_eval(header.strip())
    shape = tuple(int(x) for x in d["shape"])
    dtype = str(d["descr"])
    return shape, dtype


def build():
    n_reg, reg_path = latest_registry()
    if not reg_path:
        raise SystemExit("atlas registry not found")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    feats = reg.get("features", [])
    reg_version = reg.get("version")

    # readout 层位（phase3188 实测；缺失容忍 → 标记省略，不造数）
    kout = {}
    if P3188.is_file():
        try:
            p3188 = json.loads(P3188.read_text(encoding="utf-8"))
            pm = (p3188.get("arm") or {}).get("per_model") or {}
            for m, v in pm.items():
                if isinstance(v, dict) and v.get("kout") is not None:
                    kout[m] = int(v["kout"])
        except Exception:
            kout = {}

    npz_manifest, grids, errors = {}, {}, []
    for m in MODELS:
        rel = NPZ_REL.format(m=m)
        p = ROOT / rel
        if not p.is_file():
            errors.append("npz missing: %s" % rel)
            continue
        shape, dtype = npz_shape(p)
        if len(shape) != 4:
            errors.append("npz unexpected H rank: %s %s" % (rel, shape))
            continue
        n_templates, n_rows, n_layers, d_model = shape
        grids[m] = {"n_layers": n_layers, "d_model": d_model,
                    "n_templates": n_templates, "n_rows": n_rows,
                    "axis_semantics": "H[template,row,layer,dim]；第 0 维=句式模板(3)，非 seed",
                    "origin": "sealed_collect_npz_H_shape"}
        npz_manifest[m] = {"rel_path": rel.replace("\\", "/"),
                           "file_sha256_8": sha8_bytes(p.read_bytes()) if p.stat().st_size < 2**31 else None,
                           "shape": list(shape), "dtype": dtype,
                           "bytes": p.stat().st_size,
                           "source_phase": 3169, "source_name": "g5a6_oov_panel",
                           "note": "封存全场激活 H[template,row,layer,dim]（float16 原生精度）；"
                                   "row = 73 实体 × 10 类别 的完全交叉面板，行语义见 server/neuron_panel.py；"
                                   "sha8 仅对 <2GB 文件计算"}

    # registered 白名单：feature 标记（特征级，不占神经元坐标）+ readout 层标记
    points = []
    for f in feats:
        points.append({
            "kind": "feature", "id": f.get("id"), "family": f.get("family"),
            "e_level": f.get("evidence_level"),
            "model_scope": f.get("model_scope", []),
            "n_anchors": len(f.get("anchors", [])),
            "origin": "feature_level_no_native_coord",
            "note": "特征级标记：registry 未登记原生 (layer,dim)（v1.33 内在维度普查为挂账 3238-A）；"
                    "在场景中显示为族属标记点，不声称占据任何神经元坐标",
        })
    for m, k in sorted(kout.items()):
        if m in grids and 0 <= k < grids[m]["n_layers"]:
            points.append({
                "kind": "readout_layer", "model": m, "layer": k,
                "origin": "phase3188_kout_measured",
                "note": "读出层标记（logit 读出位口径，phase3188 per_model.kout 实测）",
            })

    out = {
        "schema": "neuron3d.v1",
        "design": DESIGN,
        "created_by": "server/derive_neuron3d.py",
        "registry_version": reg_version,
        "registry_source": reg_path.name,
        "models_mainline": MODELS,
        "grid": grids,
        "points": points,
        "npz": npz_manifest,
        "npz_errors": errors,
        "labels": {"policy": "registry_only",
                   "forbidden": "frontend_generated",
                   "note": "前端零标签生成：未入册点只显示数字证据；特征标签回链 registry 条目（phase+sha8）"},
        "coords_meaning": "X/Z=原生坐标索引（不重排），Y=层号；位置=坐标索引，非模型真实几何",
    }

    blob = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8")
    return blob, errors


def main():
    check = "--check" in sys.argv
    blob, errors = build()
    if check:
        if OUT_PATH.is_file() and OUT_PATH.read_bytes() == blob:
            print("check OK (idempotent): %s" % OUT_PATH.name)
        else:
            print("check FAIL: %s differs from recomputation" % OUT_PATH)
            sys.exit(1)
    else:
        ATLAS_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_bytes(blob)
        print("written %s (%d bytes) sha8=%s" % (OUT_PATH, len(blob), sha8_bytes(blob)))
    if errors:
        print("warnings:", *errors, sep="\n  ")
    out = json.loads(blob)
    print("models=%d grid=%s points=%d" % (
        len(out["grid"]),
        {m: (g["n_layers"], g["d_model"]) for m, g in out["grid"].items()},
        len(out["points"])))


if __name__ == "__main__":
    main()
