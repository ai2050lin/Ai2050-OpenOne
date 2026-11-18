# -*- coding: utf-8 -*-
# 只读扫描：Temp / .codex / .cache 顶层子项明细。不做任何修改。
import os, time, datetime

OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/admin_cleanup_detail_20261107.txt"
REPARSE = 0x400

def human(n):
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or u == "TB":
            return ("%d B" % int(x)) if u == "B" else ("%.2f %s" % (x, u))
        x /= 1024.0

def is_reparse(e):
    try:
        st = e.stat(follow_symlinks=False)
        return bool(getattr(st, "st_file_attributes", 0) & REPARSE)
    except Exception:
        return False

def dir_size(root, deadline=None):
    total = 0; nf = 0; trunc = False
    stack = [root]
    while stack:
        if deadline and time.time() > deadline:
            trunc = True
            break
        d = stack.pop()
        try:
            it = os.scandir(d)
        except Exception:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if is_reparse(e):
                            continue
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                        nf += 1
                except Exception:
                    pass
    return total, nf, trunc

def top_items(root, limit=30, deadline=None):
    rows = []
    try:
        ents = list(os.scandir(root))
    except Exception:
        return rows
    for e in ents:
        if is_reparse(e):
            continue
        try:
            if e.is_dir(follow_symlinks=False):
                sz, nf, tr = dir_size(e.path, deadline)
                rows.append((sz, nf, e.name + "/", tr))
            else:
                sz = e.stat(follow_symlinks=False).st_size
                rows.append((sz, 1, e.name, False))
        except Exception:
            pass
    rows.sort(key=lambda r: -r[0])
    return rows[:limit]

L = []
L.append("=" * 78)
L.append("Temp / .codex / .cache 顶层明细（只读扫描）")
L.append("生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
L.append("=" * 78)

for title, path, limit in [
    ("一、AppData/Local/Temp 顶层 TOP40", "C:/Users/Admin/AppData/Local/Temp", 40),
    ("二、.codex 顶层 TOP20", "C:/Users/Admin/.codex", 20),
    ("三、.cache 顶层 TOP20", "C:/Users/Admin/.cache", 20),
]:
    L.append("")
    L.append("【%s】" % title)
    rows = top_items(path, limit)
    if not rows:
        L.append("  (不存在或无法读取)")
        continue
    tot = sum(r[0] for r in rows)
    for sz, nf, name, tr in rows:
        mark = " partial" if tr else ""
        L.append("%12s  %8d  %-50s%s" % (human(sz), nf, name, mark))
    L.append("  -- TOP合计: %s" % human(tot))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("DETAIL WRITTEN")
