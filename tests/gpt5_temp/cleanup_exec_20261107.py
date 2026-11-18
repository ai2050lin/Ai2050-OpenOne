# -*- coding: utf-8 -*-
# 执行第一档清理(用户二次确认: 永久删除)。分批<=10目标, 逐批验证, 锁定文件跳过, 全程日志。
import os, shutil, subprocess, datetime, sys

TEMP = "C:/Users/Admin/AppData/Local/Temp"
LOG = "D:/AI2050/Ai2050-OpenOne/tempdata/cleanup_exec_log_20261107.txt"
UV = "C:/Users/Admin/.local/bin/uv.exe"

BIG18 = ["tmpdmhbx230", "tmp8gv3ml8_", "tmpt14k6b7m", "tmpvj29zgct", "tmpt2o3orxw",
         "tmppl4yvwfo", "tmplnl51nxa", "tmpqepbpugf", "tmpnx5qbnno", "tmple1hwn87",
         "tmpgt6b19qg", "tmp5ulkh_1y", "tmp81e9sn_m", "tmphivke_ay", "tmpacjdm9uy",
         "tmp48c3iou4", "tmp__exe9bq", "tmppso29gnc"]

logf = open(LOG, "a", encoding="utf-8")
skipped = []
freed_bytes = [0]

def log(msg):
    logf.write("[%s] %s\n" % (datetime.datetime.now().strftime("%H:%M:%S"), msg))
    logf.flush()

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

def rm_onexc(func, path, exc_info):
    skipped.append(path)
    log("    SKIP(locked/error): %s" % path)

def rmtree_dir(path):
    """删整个目录, 返回 (ok, size_before, n_skip_before)"""
    if not os.path.exists(path):
        return "missing", 0, 0
    s, n = dir_size(path)
    before = len(skipped)
    try:
        shutil.rmtree(path, onexc=rm_onexc)
    except Exception as e:
        log("    FAIL %r" % (e,))
    now_skip = len(skipped) - before
    gone = not os.path.exists(path)
    return ("gone" if gone else "partial"), s, now_skip

def clear_children(path):
    """只清空目录内容, 保留目录本身"""
    if not os.path.isdir(path):
        return "missing", 0, 0
    total_s = 0; total_skip = 0
    try:
        entries = list(os.scandir(path))
    except Exception:
        return "missing", 0, 0
    for e in entries:
        s, n = 0, 0
        try:
            if e.is_dir(follow_symlinks=False):
                s, n = dir_size(e.path)
                st, s2, sk = rmtree_dir(e.path)
                total_s += s2; total_skip += sk
            elif e.is_file(follow_symlinks=False):
                s = e.stat(follow_symlinks=False).st_size
                try:
                    os.remove(e.path)
                    total_s += s
                except Exception:
                    skipped.append(e.path); total_skip += 1
        except Exception:
            pass
    return "done", total_s, total_skip

def free_c():
    t, u, f = shutil.disk_usage("C:/")
    return f

log("=" * 60)
log("EXEC START (用户已二次确认永久删除)")
free_before = free_c()
log("C: 可用空间(开始): %s" % human(free_before))

# ---- G1: 18 个大 tmp 目录, 分 2 批 ----
log("== G1: 18 个大 tmp 解包目录 ==")
g1_freed = 0; g1_skip = 0; g1_ok = 0
for bi in range(0, len(BIG18), 9):
    batch = BIG18[bi:bi + 9]
    log("  batch %d/%d: %s" % (bi // 9 + 1, (len(BIG18) + 8) // 9, ",".join(batch)))
    for name in batch:
        st, s, sk = rmtree_dir(os.path.join(TEMP, name))
        if st == "gone":
            g1_freed += s; g1_ok += 1
        elif st == "partial":
            g1_freed += s; g1_skip += sk
        g1_skip += sk if st == "gone" else 0
        log("  %s -> %s (size=%s, skip=%d)" % (name, st, human(s), sk))
    log("  批次验证: 剩余不存在=%d/%d" % (sum(1 for n2 in batch if not os.path.exists(os.path.join(TEMP, n2))), len(batch)))
log("G1 结果: 成功删除 %d/18 个目录, 释放 %s, 跳过锁定 %d 个" % (g1_ok, human(g1_freed), g1_skip))
freed_bytes[0] += g1_freed

# ---- G2: Temp 其他小垃圾 ----
log("== G2: Temp 其他 tmp* 小目录 / node-compile-cache / playwright profile ==")
g2_freed = 0; g2_skip = 0; g2_cnt = 0
small = []
try:
    for e in os.scandir(TEMP):
        if e.is_dir(follow_symlinks=False):
            nm = e.name
            if (nm.startswith("tmp") and len(nm) == 11 and nm not in BIG18) or \
               nm == "node-compile-cache" or nm.startswith("playwright_chromiumdev_profile"):
                small.append(e.path)
except Exception:
    pass
for bi in range(0, len(small), 10):
    batch = small[bi:bi + 10]
    for p in batch:
        st, s, sk = rmtree_dir(p)
        if st in ("gone", "partial"):
            g2_freed += s; g2_cnt += 1; g2_skip += sk
    log("  batch %d: 完成 %d 个目录, 累计释放 %s, 跳过 %d" % (bi // 10 + 1, g2_cnt, human(g2_freed), g2_skip))
log("G2 结果: 删除 %d 个目录, 释放 %s, 跳过 %d" % (g2_cnt, human(g2_freed), g2_skip))
freed_bytes[0] += g2_freed

# ---- G3: DXCache 文件, 10 个/批 ----
log("== G3: NVIDIA DXCache ==")
DX = "C:/Users/Admin/AppData/Local/NVIDIA/DXCache"
g3_freed = 0; g3_skip = 0; g3_cnt = 0
files = []
try:
    for e in os.scandir(DX):
        if e.is_file(follow_symlinks=False):
            files.append((e.path, e.stat(follow_symlinks=False).st_size))
except Exception:
    pass
for bi in range(0, len(files), 10):
    chunk = files[bi:bi + 10]
    ok_n = 0
    for p, s in chunk:
        try:
            os.remove(p)
            g3_freed += s; g3_cnt += 1; ok_n += 1
        except Exception:
            skipped.append(p); g3_skip += 1
    gone_n = sum(1 for p, _ in chunk if not os.path.exists(p))
    log("  chunk %d: 删除 %d/10, 验证不存在 %d/10" % (bi // 10 + 1, ok_n, gone_n))
log("G3 结果: 删除 %d/%d 个文件, 释放 %s, 跳过 %d" % (g3_cnt, len(files), human(g3_freed), g3_skip))
freed_bytes[0] += g3_freed

# ---- G4: pip cache 内容 ----
log("== G4: pip cache 内容 ==")
st, s, sk = clear_children("C:/Users/Admin/AppData/Local/pip/cache")
log("G4 结果: %s 释放 %s 跳过 %d" % (st, human(s), sk))
freed_bytes[0] += s

# ---- G5: Chrome 缓存 ----
log("== G5: Chrome Cache/CodeCache ==")
g5 = 0
for sub in (r"AppData/Local/Google/Chrome/User Data/Default/Cache",
            r"AppData/Local/Google/Chrome/User Data/Default/Code Cache"):
    st, s, sk = clear_children("C:/Users/Admin/" + sub)
    g5 += s
    log("  %s -> %s 释放 %s 跳过 %d" % (sub.split("/")[-1], st, human(s), sk))
freed_bytes[0] += g5

# ---- G6: electron / VS Code / D3DSCache / GLCache / 缩略图 ----
log("== G6: 其他可再生缓存 ==")
g6 = 0
for sub in (r"AppData/Local/electron/Cache",
            r"AppData/Roaming/Code/Cache",
            r"AppData/Roaming/Code/CachedData",
            r"AppData/Roaming/Code/Code Cache",
            r"AppData/Roaming/Code/GPUCache",
            r"AppData/Roaming/Code/logs",
            r"AppData/Local/D3DSCache",
            r"AppData/Local/NVIDIA/GLCache"):
    st, s, sk = clear_children("C:/Users/Admin/" + sub)
    g6 += s
    log("  %s -> %s 释放 %s 跳过 %d" % (sub.split("/")[-1], st, human(s), sk))
exp_dir = "C:/Users/Admin/AppData/Local/Microsoft/Windows/Explorer"
if os.path.isdir(exp_dir):
    try:
        for e in os.scandir(exp_dir):
            if e.is_file(follow_symlinks=False) and (e.name.startswith("thumbcache") or e.name.startswith("iconcache")):
                s = e.stat(follow_symlinks=False).st_size
                try:
                    os.remove(e.path); g6 += s
                except Exception:
                    skipped.append(e.path)
    except Exception:
        pass
log("G6 结果: 释放 %s" % human(g6))
freed_bytes[0] += g6

# ---- G7: uv cache clean (官方 CLI) ----
log("== G7: uv cache clean ==")
try:
    r = subprocess.run([UV, "cache", "clean"], capture_output=True, text=True, timeout=300)
    log("  uv returncode=%d stdout=%s stderr=%s" % (r.returncode, (r.stdout or "")[:200], (r.stderr or "")[:200]))
except Exception as e:
    log("  uv FAIL %r" % (e,))
try:
    s, n = dir_size("C:/Users/Admin/AppData/Local/uv/cache")
    log("  uv/cache 剩余: %s (%d files)" % (human(s), n))
    freed_bytes[0] += max(0, 4.43 * 1024 ** 3 - s)
except Exception:
    pass

# ---- 收尾 ----
free_after = free_c()
log("C: 可用空间(结束): %s" % human(free_after))
log("实际释放(按磁盘可用空间): %s" % human(free_after - free_before))
log("按目标尺寸累计: %s ; 锁定/错误跳过: %d 个" % (human(freed_bytes[0]), len(skipped)))
if skipped:
    log("跳过清单(前30): " + "; ".join(skipped[:30]))
log("EXEC END")
logf.close()
print("CLEANUP DONE")
