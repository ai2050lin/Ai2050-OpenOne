# -*- coding: utf-8 -*-
"""revert_space3d.py — 回退 3D 轮（space3d/AtlasSpace）改动：派生器剥离 + CSS 孤儿规则清理"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def patch(path, subs):
    p = Path(path)
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    s = raw.decode("utf-8")
    for old, new, cnt in subs:
        c = s.count(old)
        assert c == cnt, "ASSERT FAIL %s: %r count=%d want=%d" % (path, old[:60], c, cnt)
        s = s.replace(old, new)
    p.write_bytes(s.encode("utf-8"))
    print("PATCHED", path)

# ---- 1) derive_atlas_tree.py：剥离 space3d 段 ----
lines = io.StringIO(
    (ROOT / "server" / "derive_atlas_tree.py").read_bytes().decode("utf-8")
).readlines()
# 段落边界按行号断言（1-based）：105 行起 space3d 注释，188 行 return 结束，190 行空行
assert lines[104].startswith("# ---- space3d"), lines[104]
assert lines[187].rstrip() == "    }", lines[187]
assert lines[189].strip() == "", lines[189]
del lines[104:190]   # 删 105-190 行（space3d 段 + 尾随空行）
src = "".join(lines)

patch_subs_tree = [
    # 文档头
    ("  横切：MEC-xx 机制层（多对多 encoded_by）+ cross_link 域间交叉边（SxK/RxS/RxK 的家）\n"
     "  space3d 段（design/atlas_space3d_scheme_v1.md）：FTR 点位（确定性派生坐标）+ 机制/交叉边表 +\n"
     "  族质心（族晕用）+ 云口径注记——真实 npz 坐标接入前界面必须标注「派生坐标，非模型真实几何」。\n",
     "  横切：MEC-xx 机制层（多对多 encoded_by）+ cross_link 域间交叉边（SxK/RxS/RxK 的家）\n", 1),
    ("  - coverage / space3d 点位与边表只由本派生器计算，人手禁写；",
     "  - coverage 只由本派生器计算，人手禁写；", 1),
    # import math 仅 space3d 使用
    ("import json\nimport math\nimport re", "import json\nimport re", 1),
    # build() 内调用与输出字段
    ("    space3d = build_space3d(domains, cross, mecs, idx)\n"
     "    assert len(space3d[\"points\"][\"ftr\"]) == len(idx), \"space3d must cover all features exactly once\"\n\n",
     "", 1),
    ("        \"coverage\": coverage,\n        \"space3d\": space3d,\n",
     "        \"coverage\": coverage,\n", 1),
]
for old, new, cnt in patch_subs_tree:
    c = src.count(old)
    assert c == cnt, "ASSERT FAIL tree: %r count=%d want=%d" % (old[:60], c, cnt)
    src = src.replace(old, new)
(ROOT / "server" / "derive_atlas_tree.py").write_bytes(src.encode("utf-8"))
print("PATCHED server/derive_atlas_tree.py")

# ---- 2) rdc_fusion.css：删 .at-space-* 孤儿规则（两段，保留夹在中间的行业进展 v5 规则） ----
css_path = ROOT / "frontend" / "src" / "components" / "app" / "rdc_fusion" / "rdc_fusion.css"
css = css_path.read_bytes().decode("utf-8")

block1_head = "/* ---- 图谱 3D 空间（AtlasSpace，space3d.v1；design/atlas_space3d_scheme_v1.md） ---- */\n"
assert css.count(block1_head) == 1
i1 = css.index(block1_head)
# 段落结束锚：famclear 行 + 空行 + 行业进展 v5 注释
anchor1 = "\n/* ===== 行业进展 v5：阶段目标 / 关键成果 改为上下两行 ===== */\n"
assert css.count(anchor1) == 1
i2 = css.index(anchor1)
seg1 = css[i1:i2]
n_rules1 = seg1.count(".at-space")
seg1_keep = [l for l in seg1.splitlines(True)
             if l.strip() and not l.lstrip().startswith(".at-space") and not l.startswith("/* ---- 图谱 3D")]
css = css[:i1] + "".join(seg1_keep) + css[i2:]

tip_lines = [l for l in css.splitlines(True) if l.lstrip().startswith(".at-space-tip")]
assert len(tip_lines) == 3, "expect 3 .at-space-tip lines, got %d" % len(tip_lines)
for l in tip_lines:
    css = css.replace(l, "", 1)

assert ".at-space" not in css, "residual .at-space rules remain"
css_path.write_bytes(css.encode("utf-8"))
print("PATCHED rdc_fusion.css (removed %d block1 rules + 3 tip rules)" % n_rules1)
print("ALL OK")
