# -*- coding: utf-8 -*-
# 修正探测: tmp 目录活跃度(11字符) + 卷GUID映射盘符 + 最终待删清单
import os, ctypes, datetime

OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/cleanup_probe2_20261107.txt"
L = []

def human(n):
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or u == "TB":
            return ("%d B" % int(x)) if u == "B" else ("%.2f %s" % (x, u))
        x /= 1024.0

L.append("生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 卷 GUID -> 盘符
L.append("== 卷映射 ==")
dirs = ctypes.create_unicode_buffer(512)
for guid in ("{0d52e1e4-ad5c-41e8-a6aa-4d7059693e16}", "{362b7f2a-0000-0000-0000-100000000000}", "{cf48ac52-3ed3-4272-9a3d-db8bb7b66d84}"):
    p2 = ctypes.create_unicode_buffer("\\\\?\\Volume" + guid + "\\")
    ok = ctypes.windll.kernel32.GetVolumePathNamesForVolumeNameW(p2.value, dirs, 512, None)
    L.append("%s -> %s (ok=%d)" % (guid, dirs.value if ok else "?", ok))

# tmp 目录活跃度(修正: tmp+8=11)
L.append("")
L.append("== Temp 下 tmp* 大目录(>=100MB) 活跃度 ==")
TEMP = "C:/Users/Admin/AppData/Local/Temp"
now = datetime.datetime.now().timestamp()
rows = []
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

for e in os.scandir(TEMP):
    if e.is_dir(follow_symlinks=False) and e.name.startswith("tmp") and len(e.name) == 11:
        try:
            st = e.stat(follow_symlinks=False)
            age_h = (now - st.st_mtime) / 3600.0
            rows.append((e.name, age_h, st.st_mtime))
        except Exception:
            pass
for name, age, _ in sorted(rows, key=lambda x: x[1]):
    flag = "  <-- 24h 内有改动" if age < 24 else ""
    L.append("%s  mtime %.1f 小时前%s" % (name, age, flag))
L.append("tmp* 11字符目录总数: %d" % len(rows))

# 大 tmp 目录精确大小(只测>=100MB mtime candidates: 全测, 已有OS缓存很快)
L.append("")
L.append("== 各 tmp 目录大小(用于最终清单) ==")
tot = 0
for name, age, _ in sorted(rows, key=lambda x: x[1]):
    s, n = dir_size(os.path.join(TEMP, name))
    tot += s
    L.append("%s  %s  (%d files, mtime %.1f小时前)" % (name, human(s), n, age))
L.append("合计: %s" % human(tot))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("PROBE2 WRITTEN")
