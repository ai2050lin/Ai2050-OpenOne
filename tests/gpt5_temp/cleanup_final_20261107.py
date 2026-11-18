# -*- coding: utf-8 -*-
# 清空 C 盘回收站(用户已确认) + 重试 DXCache 残余 + 最终核实
import os, shutil, ctypes, datetime

OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/cleanup_final_20261107.txt"
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

def free_c():
    t, u, f = shutil.disk_usage("C:/")
    return f

L.append("时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
f0 = free_c()
L.append("清空前 C: 可用 %s" % human(f0))

s0, n0 = dir_size("C:/$Recycle.Bin")
L.append("清空前回收站: %s (%d 项)" % (human(s0), n0))

SHERB = 0x1 | 0x2 | 0x4  # NOCONFIRMATION | NOPROGRESSUI | NOSOUND
res = ctypes.windll.shell32.SHEmptyRecycleBinW(None, "C:\\", SHERB)
L.append("SHEmptyRecycleBinW(C:) 返回: 0x%X (0=成功)" % (res & 0xFFFFFFFF))

s1, n1 = dir_size("C:/$Recycle.Bin")
L.append("清空后回收站: %s (%d 项)" % (human(s1), n1))

# 重试 DXCache 残余 15 个小文件
DX = "C:/Users/Admin/AppData/Local/NVIDIA/DXCache"
retry_ok = 0; retry_fail = 0
if os.path.isdir(DX):
    for e in os.scandir(DX):
        if e.is_file(follow_symlinks=False):
            try:
                os.remove(e.path)
                retry_ok += 1
            except Exception:
                retry_fail += 1
L.append("DXCache 残余重试: 删除 %d, 失败 %d" % (retry_ok, retry_fail))

f1 = free_c()
L.append("清空后 C: 可用 %s (本次回收站释放 %s)" % (human(f1), human(f1 - f0)))
t, u, f2 = shutil.disk_usage("C:/")
L.append("")
L.append("=== 最终战报 ===")
L.append("C: 可用空间: 9.66 GB -> %s (总释放 %.2f GB)" % (human(f2), (f2 - 9.66 * 1024 ** 3) / 1024 ** 3))
s3, n3 = dir_size("C:/Users/Admin/AppData/Local/Temp")
L.append("Temp 目录当前: %s (%d files)" % (human(s3), n3))
s4, n4 = dir_size("C:/Users/Admin/AppData/Local/NVIDIA/DXCache")
L.append("DXCache 当前: %s (%d files)" % (human(s4), n4))

with open(OUT, "w", encoding="utf-8") as fo:
    fo.write("\n".join(L))
print("FINAL WRITTEN")
