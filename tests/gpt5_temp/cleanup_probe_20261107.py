# -*- coding: utf-8 -*-
# 只读探测：磁盘空间 / 回收站容量 / uv npm 路径 / tmp 目录活跃度
import os, shutil, winreg, datetime

OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/cleanup_probe_20261107.txt"
L = []

def human(n):
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or u == "TB":
            return ("%d B" % int(x)) if u == "B" else ("%.2f %s" % (x, u))
        x /= 1024.0

L.append("生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

for d in ("C:/", "D:/"):
    try:
        t, u, f = shutil.disk_usage(d)
        L.append("磁盘 %s 总 %s 已用 %s 可用 %s" % (d, human(t), human(u), human(f)))
    except Exception as e:
        L.append("磁盘 %s 读取失败 %r" % (d, e))

# 回收站容量 (MaxCapacity 单位 MB, 每卷)
L.append("")
L.append("== 回收站容量注册表 ==")
for hive, name in [(winreg.HKEY_CURRENT_USER, "HKCU"), (winreg.HKEY_LOCAL_MACHINE, "HKLM")]:
    base = r"Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume"
    try:
        k = winreg.OpenKey(hive, base)
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(k, i)
                try:
                    sk = winreg.OpenKey(k, sub)
                    val, typ = winreg.QueryValueEx(sk, "MaxCapacity")
                    L.append("%s %s MaxCapacity=%s MB" % (name, sub, val))
                except FileNotFoundError:
                    L.append("%s %s (无 MaxCapacity 键)" % (name, sub))
                i += 1
            except OSError:
                break
    except FileNotFoundError:
        L.append("%s 无 BitBucket\\Volume 键" % name)

# 当前回收站占用 (C盘)
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

try:
    s, n = dir_size("C:/$Recycle.Bin")
    L.append("C:/$Recycle.Bin 当前占用 %s (%d 项, 含系统开销)" % (human(s), n))
except Exception as e:
    L.append("回收站占用读取失败 %r" % (e,))

L.append("")
L.append("== 工具路径 ==")
for exe in ("uv.exe", "npm.cmd", "pip.exe"):
    found = []
    for c in (r"C:/Users/Admin/.local/bin", r"C:/Program Files/nodejs", r"C:/Users/Admin/AppData/Roaming/npm",
              r"D:/AI2050/Ai2050-OpenOne/.venv/Scripts", r"C:/Users/Admin/AppData/Roaming/uv"):
        p = os.path.join(c, exe)
        if os.path.exists(p):
            found.append(p)
    L.append("%s: %s" % (exe, found or "未找到"))

L.append("")
L.append("== 18 个 tmp 解包目录活跃度 ==")
TEMP = "C:/Users/Admin/AppData/Local/Temp"
now = datetime.datetime.now().timestamp()
tmp_dirs = []
for e in os.scandir(TEMP):
    if e.is_dir(follow_symlinks=False) and e.name.startswith("tmp") and len(e.name) == 10:
        try:
            st = e.stat(follow_symlinks=False)
            age_h = (now - st.st_mtime) / 3600.0
            tmp_dirs.append((e.name, age_h))
        except Exception:
            pass
tmp_dirs.sort(key=lambda x: x[1])
for name, age in tmp_dirs:
    flag = "  <-- 24h 内有改动,可能在使用" if age < 24 else ""
    L.append("%s  最近改动 %.1f 小时前%s" % (name, age, flag))
L.append("tmp 目录总数: %d" % len(tmp_dirs))

L.append("")
L.append("== claude Temp 目录活跃度 ==")
cl = os.path.join(TEMP, "claude")
if os.path.isdir(cl):
    st = os.stat(cl)
    L.append("claude/ 最近改动 %.1f 小时前" % ((now - st.st_mtime) / 3600.0))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("PROBE WRITTEN")
