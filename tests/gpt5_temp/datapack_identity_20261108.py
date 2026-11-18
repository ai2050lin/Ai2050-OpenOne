# -*- coding: utf-8 -*-
# 只读: 识别 datapack 解包目录的来源应用身份
import os, json, time, datetime

OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/datapack_identity_20261108.txt"
L = []
T = "C:/Users/Admin/AppData/Local/Temp/tmp4wmiv7ed"

L.append("检查对象: %s" % T)
L.append("生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def listdirmax(path, n=50):
    try:
        rows = []
        for e in os.scandir(path):
            try:
                st = e.stat(follow_symlinks=False)
                rows.append((st.st_size, e.name, e.is_dir(follow_symlinks=False)))
            except Exception:
                pass
        rows.sort(key=lambda r: -r[0])
        return rows[:n]
    except Exception as ex:
        L.append("  读取失败 %r" % (ex,))
        return []

L.append("")
L.append("== tmp 目录顶层 ==")
for sz, name, isd in listdirmax(T):
    L.append("  %-10s %-40s %s" % (human(sz) if False else "%.1f KB" % (sz / 1024) if sz < 1024 * 1024 else "%.1f MB" % (sz / 1024 / 1024), name, "[DIR]" if isd else ""))

dp = os.path.join(T, "datapack")
L.append("")
L.append("== datapack/ 顶层 ==")
for sz, name, isd in listdirmax(dp):
    L.append("  %-40s %s (%s)" % (name, "[DIR]" if isd else "%.1f MB" % (sz / 1024 / 1024), "%.1f KB" % (sz / 1024) if sz < 1024 * 1024 else "%.1f MB" % (sz / 1024 / 1024)))

# 找小的 json 元数据
L.append("")
L.append("== 元数据文件搜索 ==")
found_meta = []
for root, dirs, files in os.walk(dp):
    for fn in files:
        if fn.lower() in ("manifest.json", "metadata.json", "info.json", "pack.json", "version.json", "datapack.json") or fn.lower().endswith(".meta"):
            p = os.path.join(root, fn)
            try:
                found_meta.append((p, os.path.getsize(p)))
            except Exception:
                pass
    if len(found_meta) > 10:
        break
for p, sz in found_meta[:10]:
    L.append("  %s (%d B)" % (p, sz))

for p, sz in found_meta[:3]:
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(2000)
        L.append("")
        L.append("-- %s 前2000字符 --" % p)
        L.append(content)
    except Exception as ex:
        L.append("  读取失败 %r" % (ex,))

# tmp 顶层小 json
L.append("")
L.append("== tmp 顶层小 json 内容 ==")
for sz, name, isd in listdirmax(T, 60):
    if not isd and name.lower().endswith(".json") and sz < 512 * 1024:
        p = os.path.join(T, name)
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                L.append("-- %s (%d B) --" % (name, sz))
                L.append(f.read(1500))
        except Exception as ex:
            L.append("  读取失败 %r" % (ex,))
        break

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("IDENTITY WRITTEN")
