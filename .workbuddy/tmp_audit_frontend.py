import os, re, json, io

SRC = r"D:\AI2050\Ai2050-OpenOne\frontend\src"
OUT_MD = r"D:\AI2050\Ai2050-OpenOne\ai2050_research_os\docs\CLIENT_ASSET_AUDIT_2026-10-03.md"
OUT_JSON = r"D:\AI2050\Ai2050-OpenOne\ai2050_research_os\docs\CLIENT_ASSET_AUDIT_2026-10-03.json"

rows = []
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in ("node_modules",)]
    for f in sorted(files):
        if not f.endswith((".jsx", ".js")):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, SRC)
        try:
            t = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception as e:
            rows.append({"file": rel, "error": str(e)})
            continue
        lines = t.count("\n") + 1
        imports = re.findall(r"from\s+['\"]([^'\"]+)['\"]", t)
        deps = set()
        for m in imports:
            if m.startswith("three") or m.startswith("@react-three"):
                deps.add("3d")
            if "plotly" in m:
                deps.add("plotly")
            if "recharts" in m:
                deps.add("recharts")
        uses_snapshot = ("useResearchSnapshot" in t) or ("research_snapshot" in t) or ("research_data" in t)
        uses_api = ("fetch(" in t) or ("axios" in t) or ("/api/" in t)
        demo_arrays = re.findall(r"const\s+([A-Z_]{4,}(?:DATA|POINTS|NODES|EDGES|VALUES|MATRIX)?)\s*=\s*\[", t)
        rows.append({
            "file": rel.replace("\\", "/"),
            "lines": lines,
            "deps": sorted(deps),
            "uses_snapshot": uses_snapshot,
            "uses_api": uses_api,
            "demo_consts": demo_arrays[:6],
            "n_demo_consts": len(demo_consts) if (demo_consts := demo_arrays) else 0,
        })

def classify(r):
    if "error" in r:
        return "error"
    if r["uses_snapshot"]:
        return "A_snapshot_anchored"
    if r["uses_api"]:
        return "B_service_driven"
    if r["deps"] or r["n_demo_consts"]:
        return "C_concept_demo"
    return "D_non_visual_or_unclear"

for r in rows:
    if "error" not in r:
        r["grade"] = classify(r)

mj = os.path.join(SRC, "main.jsx")
entry = None
if os.path.exists(mj):
    tt = io.open(mj, encoding="utf-8", errors="replace").read()
    m = re.findall(r"import\s+(\w+)\s+from\s+['\"]\./(\w+\.jsx)['\"]", tt)
    entry = {"main_imports": m, "renders": re.findall(r"createRoot\([^)]*\)\.render\(<([\w.]+)", tt) or re.findall(r"render\(\s*<([\w.]+)", tt)}

summary = {}
for r in rows:
    if "error" in r:
        continue
    summary[r["grade"]] = summary.get(r["grade"], 0) + 1

payload = {"generated": "2026-10-03", "entry": entry, "grade_summary": summary, "files": rows}
io.open(OUT_JSON, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=1))

def zh_grade(g):
    return {
        "A_snapshot_anchored": "A 已接 Canonical Snapshot",
        "B_service_driven": "B 服务端 API 驱动",
        "C_concept_demo": "C 概念示意/演示数据",
        "D_non_visual_or_unclear": "D 非可视化或待人工复核",
        "error": "读取失败",
    }.get(g, g)

md = []
md.append("# 客户端可视化资产审计 [2026-10-03]")
md.append("")
md.append("范围：`frontend/src` 全部 .jsx/.js（M1c）。分档判据：组件是否有可回查数据来源。")
md.append("本文件是工程审计记录，不是研究事实源，不进入 Registry。")
md.append("")
md.append("## 总览")
md.append("")
md.append("| 分档 | 数量 | 含义 |")
md.append("| --- | --- | --- |")
for g in ["A_snapshot_anchored", "B_service_driven", "C_concept_demo", "D_non_visual_or_unclear", "error"]:
    if g in summary:
        md.append("| %s | %d | %s |" % (zh_grade(g), summary[g], ""))
md.append("")
if entry:
    md.append("## 入口")
    md.append("")
    md.append("- main.jsx 导入：" + json.dumps(entry.get("main_imports", []), ensure_ascii=False))
    md.append("")
md.append("## 明细")
md.append("")
md.append("| 文件 | 行数 | 依赖 | Snapshot | API | 演示常量 | 分档 |")
md.append("| --- | --- | --- | --- | --- | --- | --- |")
for r in rows:
    if "error" in r:
        md.append("| %s | - | - | - | - | - | 读取失败 |" % r["file"])
        continue
    md.append("| %s | %d | %s | %s | %s | %d | %s |" % (
        r["file"], r["lines"], ",".join(r["deps"]) or "-",
        "Y" if r["uses_snapshot"] else "-", "Y" if r["uses_api"] else "-",
        r["n_demo_consts"], zh_grade(r["grade"])))
md.append("")
io.open(OUT_MD, "w", encoding="utf-8", newline="\r\n").write("\n".join(md))
print("OK files=%d summary=%s" % (len(rows), json.dumps(summary)))
