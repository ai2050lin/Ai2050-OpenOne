# -*- coding: utf-8 -*-
# 只读复查: C盘空间回落原因定位。扫描 C:\Users\Admin 并与 2026-11-07 报告对比, 附加关键嫌疑点核查。
import os, shutil, time, datetime

PROFILE = "C:/Users/Admin"
OUT = "D:/AI2050/Ai2050-OpenOne/tempdata/c_regrowth_check_20261108.txt"
PROG = "D:/AI2050/Ai2050-OpenOne/tempdata/c_regrowth_progress_20261108.txt"
START = time.time()
DEADLINE = START + 460.0
REPARSE = 0x400
LARGE = 200 * 1024 * 1024

large = {}
err = 0

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

def dir_size(root, deadline):
    global err
    total = 0; nf = 0; nd = 0; trunc = False
    stack = [root]
    while stack:
        if time.time() > deadline:
            trunc = True
            break
        d = stack.pop()
        try:
            it = os.scandir(d)
        except Exception:
            err += 1
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if is_reparse(e):
                            continue
                        nd += 1
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        st = e.stat(follow_symlinks=False)
                        total += st.st_size
                        nf += 1
                        if st.st_size >= LARGE:
                            large[e.path] = (st.st_size, st.st_mtime)
                except Exception:
                    err += 1
    return {"size": total, "files": nf, "dirs": nd, "trunc": trunc}

progf = open(PROG, "a", encoding="utf-8")
def prog(msg):
    progf.write("[%s] %s\n" % (datetime.datetime.now().strftime("%H:%M:%S"), msg))
    progf.flush()

L = []
L.append("=" * 78)
L.append("C 盘空间回落定位报告（只读扫描）")
L.append("生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
L.append("基线: 2026-11-07 06:04 清理完成, C: 可用 83.77 GB; 昨日顶层扫描数据见 admin_cleanup_scan_20261107.txt")
L.append("=" * 78)

def free_c():
    t, u, f = shutil.disk_usage("C:/")
    return f

# Phase A: 总量
f_now = free_c()
L.append("")
L.append("【0】总量")
L.append("C: 当前可用: %s  (昨天清理完成时 83.77 GB, 变化 %+.2f GB)" % (human(f_now), (f_now - 83.77 * 1024 ** 3) / 1024 ** 3))
_rbin = dir_size("C:/$Recycle.Bin", min(DEADLINE, time.time() + 30))
s_bin, n_bin = _rbin["size"], _rbin["files"]
L.append("回收站占用: %s (%d 项)" % (human(s_bin), n_bin))
prog("Phase A done")

# C 盘根目录大文件 (pagefile/hiberfil 等)
L.append("")
L.append("【0b】C:/ 根目录文件")
try:
    for e in os.scandir("C:/"):
        try:
            if e.is_file(follow_symlinks=False):
                st = e.stat(follow_symlinks=False)
                if st.st_size >= 500 * 1024 * 1024:
                    L.append("  %s  %s" % (human(st.st_size), e.name))
        except Exception:
            pass
except Exception:
    pass

# Phase B: 顶层对比
YDAY_TOP = {
    "AppData": 101.85, ".workbuddy": 13.03, ".cache": 8.40, ".codex": 2.46,
    ".vscode": 2.11, ".claude": 0.2196, "Documents": 0.1960, ".codebuddy": 0.1074,
    ".local": 0.0731, "WorkBuddy": 0.0222,
}
top = []
try:
    entries = list(os.scandir(PROFILE))
except Exception:
    entries = []
dirs_first = [e for e in entries if e.is_dir(follow_symlinks=False)]
pending = []
for e in sorted(dirs_first, key=lambda x: x.name.lower()):
    if is_reparse(e):
        continue
    budget = 120.0 if e.name == "AppData" else 25.0
    r = dir_size(e.path, min(DEADLINE, time.time() + budget))
    r["name"] = e.name
    top.append(r)
    if r["trunc"]:
        pending.append(e.path)
if pending and time.time() < DEADLINE:
    budget = max(30.0, (DEADLINE - time.time()) / max(1, len(pending)))
    for p in pending:
        if time.time() > DEADLINE:
            break
        r2 = dir_size(p, min(DEADLINE, time.time() + budget))
        for row in top:
            if row["name"] == os.path.basename(p):
                row["size"] = r2["size"]; row["files"] = r2["files"]; row["trunc"] = r2["trunc"]
prog("Phase B done")

L.append("")
L.append("【一】C:\\Users\\Admin 顶层对比（昨天 04:44 扫描 -> 现在, 单位 GB, 负=变小）")
L.append("%-24s%14s%14s%10s  %s" % ("目录", "昨天", "现在", "变化", "文件数(今天)"))
now_map = {}
for row in sorted(top, key=lambda r: -r["size"]):
    now_map[row["name"]] = row["size"] / 1024 ** 3
    yd = YDAY_TOP.get(row["name"])
    if yd is None:
        yd_s = "  (昨天无此目录)"
        delta_s = "+NEW"
    else:
        yd_s = "%.2f" % yd
        delta_s = "%+.2f" % (row["size"] / 1024 ** 3 - yd)
    mark = " partial" if row["trunc"] else ""
    L.append("%-24s%14s%14.2f%10s  %d%s" % (row["name"], yd_s, row["size"] / 1024 ** 3, delta_s, row["files"], mark))
prog("Phase B report done")

# Phase C: AppData/Local 与 Roaming 子目录
def scan_children(base, limit=45):
    rows = []
    try:
        ents = [e for e in os.scandir(base) if e.is_dir(follow_symlinks=False) and not is_reparse(e)]
    except Exception:
        return rows
    for e in sorted(ents, key=lambda x: x.name.lower()):
        if time.time() > DEADLINE:
            r = {"size": 0, "files": 0, "trunc": True}; r["name"] = e.name; rows.append(r); continue
        r = dir_size(e.path, min(DEADLINE, time.time() + 45.0))
        r["name"] = e.name
        rows.append(r)
    rows.sort(key=lambda r: -r["size"])
    return rows[:limit]

local_rows = scan_children("C:/Users/Admin/AppData/Local")
prog("Phase C local done")
roaming_rows = scan_children("C:/Users/Admin/AppData/Roaming")
prog("Phase C roaming done")

L.append("")
L.append("【二】AppData/Local 子目录 TOP45（当前）")
for row in local_rows:
    mark = " partial" if row["trunc"] else ""
    L.append("%12.2f GB  %8d  %-40s%s" % (row["size"] / 1024 ** 3, row["files"], row["name"], mark))
L.append("")
L.append("【三】AppData/Roaming 子目录 TOP25（当前）")
for row in roaming_rows[:25]:
    mark = " partial" if row["trunc"] else ""
    L.append("%12.2f GB  %8d  %-40s%s" % (row["size"] / 1024 ** 3, row["files"], row["name"], mark))

# Phase D: Temp 明细 + mtime
TEMP = "C:/Users/Admin/AppData/Local/Temp"
now_ts = time.time()
L.append("")
L.append("【四】Temp 子目录 TOP30（当前, 含最近改动时间）")
temp_rows = []
try:
    for e in os.scandir(TEMP):
        if e.is_dir(follow_symlinks=False) and not is_reparse(e):
            r = dir_size(e.path, min(DEADLINE, time.time() + 30.0))
            try:
                mt = e.stat(follow_symlinks=False).st_mtime
            except Exception:
                mt = 0
            r["name"] = e.name; r["mtime"] = mt
            temp_rows.append(r)
        elif e.is_file(follow_symlinks=False):
            try:
                sz = e.stat(follow_symlinks=False).st_size
                if sz >= 100 * 1024 * 1024:
                    temp_rows.append({"name": e.name, "size": sz, "files": 1, "mtime": e.stat(follow_symlinks=False).st_mtime, "isfile": True})
            except Exception:
                pass
except Exception:
    pass
temp_rows.sort(key=lambda r: -r["size"])
for row in temp_rows[:30]:
    age = (now_ts - row.get("mtime", 0)) / 3600.0
    L.append("%12s  %8d  %-40s (改动于 %.1f 小时前)" % (human(row["size"]), row["files"], row["name"], age))
prog("Phase D done")

# Phase E: 关键嫌疑点精确核对
L.append("")
L.append("【五】清理目标回潮核查（昨天 06:04 清理完成时的值 -> 现在）")
POST = {
    "AppData/Local/Temp": 0.183,
    "AppData/Local/NVIDIA/DXCache": 0.0185,
    "AppData/Local/pip/cache": 0.0,
    "AppData/Local/uv/cache": 0.0,
    "AppData/Local/electron/Cache": 0.0,
    "AppData/Local/Google/Chrome/User Data/Default/Cache": 0.2411,
}
for rel, base_gb in POST.items():
    p = "C:/Users/Admin/" + rel
    if os.path.exists(p):
        r = dir_size(p, min(DEADLINE, time.time() + 60.0))
        L.append("%-52s  %.3f GB -> %.2f GB  (%+.2f)" % (rel, base_gb, r["size"] / 1024 ** 3, r["size"] / 1024 ** 3 - base_gb))
    else:
        L.append("%-52s  清理后已不存在(现为 0)" % rel)

# Claude vhdx / HF / codex
L.append("")
L.append("【六】其他大块头现状")
for label, p in [
    ("Claude VM rootfs.vhdx", "C:/Users/Admin/AppData/Local/Claude-Data/claude_pzs8sxrjxfjjc/vm_bundles/claudevm.bundle/rootfs.vhdx"),
    (".codex/sessions", "C:/Users/Admin/.codex/sessions"),
    (".cache/huggingface", "C:/Users/Admin/.cache/huggingface"),
    (".cache/codex-runtimes", "C:/Users/Admin/.cache/codex-runtimes"),
]:
    if os.path.exists(p):
        if os.path.isfile(p):
            L.append("%-24s %s" % (label, human(os.path.getsize(p))))
        else:
            r = dir_size(p, min(DEADLINE, time.time() + 45.0))
            L.append("%-24s %s (%d files)" % (label, human(r["size"]), r["files"]))
    else:
        L.append("%-24s 不存在" % label)

# Phase F: C:\Windows\Temp 与 ProgramData（轻量）
L.append("")
L.append("【七】用户目录之外的常见增长点（排除用）")
for label, p, budget in [
    ("C:/Windows/Temp", "C:/Windows/Temp", 30.0),
    ("C:/ProgramData", "C:/ProgramData", 45.0),
]:
    if time.time() > DEADLINE:
        L.append("%-18s 超时未测" % label)
        continue
    r = dir_size(p, min(DEADLINE, time.time() + budget))
    L.append("%-18s %s (%d files%s)" % (label, human(r["size"]), r["files"], ", partial" if r["trunc"] else ""))
prog("Phase F done")

# 大文件清单
L.append("")
L.append("【八】当前大文件 TOP40 (>=200MB, 含改动时间)")
for p, (sz, mt) in sorted(large.items(), key=lambda kv: -kv[1][0])[:40]:
    age = (now_ts - mt) / 3600.0
    L.append("%10s  改动于%7.1f小时前  %s" % (human(sz), age, p))

L.append("")
L.append("不可读条目: %d ; 总耗时: %.0fs" % (err, time.time() - START))

with open(OUT, "w", encoding="utf-8") as fo:
    fo.write("\n".join(L))
progf.close()
print("REGROWTH CHECK WRITTEN")
