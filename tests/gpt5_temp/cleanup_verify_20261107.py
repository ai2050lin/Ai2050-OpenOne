# -*- coding: utf-8 -*-
# 清理后核实: C盘空间 / 回收站占用 / DXCache剩余 / Temp剩余 / pip uv剩余
import os, shutil, datetime

OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/cleanup_verify_20261107.txt"
L = []

def human(n):
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or u == "TB":
            return ("%d B" % int(x)) if u == "B" else ("%.2f %s" % (x, u))
        x /= 1024.0

def dir_size(root):
    total = 0; nf = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except Exception:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                        nf += 1
                except Exception:
                    pass
    return total, nf

L.append("核实时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
t, u, f = shutil.disk_usage("C:/")
L.append("C: 可用 %s / 总 %s (起始可用 9.66 GB)" % (human(f), human(t)))

s, n = dir_size("C:/$Recycle.Bin")
L.append("C:/$Recycle.Bin 当前占用: %s (%d 项) [清理前 2.88MB]" % (human(s), n))

for name, p in [
    ("DXCache 剩余", "C:/Users/Admin/AppData/Local/NVIDIA/DXCache"),
    ("Temp 剩余(claude等)", "C:/Users/Admin/AppData/Local/Temp"),
    ("pip cache 剩余", "C:/Users/Admin/AppData/Local/pip/cache"),
    ("uv cache 剩余", "C:/Users/Admin/AppData/Local/uv/cache"),
    ("Chrome Cache 剩余", "C:/Users/Admin/AppData/Local/Google/Chrome/User Data/Default/Cache"),
    ("electron Cache 剩余", "C:/Users/Admin/AppData/Local/electron/Cache"),
]:
    if os.path.exists(p):
        s2, n2 = dir_size(p)
        L.append("%s: %s (%d files)" % (name, human(s2), n2))
    else:
        L.append("%s: 不存在" % name)

with open(OUT, "w", encoding="utf-8") as fo:
    fo.write("\n".join(L))
print("VERIFY WRITTEN")
