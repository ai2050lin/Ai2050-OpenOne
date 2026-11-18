# -*- coding: utf-8 -*-
# 清理 10 个新 tmp 解包目录(用户 2026-11-08 已确认)。分批<=10, 逐个验证, 全程日志。
import os, shutil, datetime, time

TEMP = "C:/Users/Admin/AppData/Local/Temp"
LOG = "D:/AI2050/Ai2050-OpenOne/tempdata/cleanup_exec_log_20261108.txt"

TARGETS = ["tmpe859_y8c", "tmpovbasn2_", "tmp4wmiv7ed", "tmpnk9n8_kt", "tmpyzbwdl4w",
           "tmpgfokdoxb", "tmph5qwz2an", "tmpaofdyj_x", "tmprmpqkjiw", "tmp0deoy09c"]

logf = open(LOG, "a", encoding="utf-8")

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

skipped = []
def rm_onexc(func, path, exc_info):
    skipped.append(path)
    log("    SKIP: %s" % path)

def rmtree_dir(path):
    if not os.path.exists(path):
        return "missing", 0
    s, n = dir_size(path)
    try:
        shutil.rmtree(path, onexc=rm_onexc)
    except Exception as e:
        log("    FAIL %r" % (e,))
    gone = not os.path.exists(path)
    return ("gone" if gone else "partial"), s

def free_c():
    t, u, f = shutil.disk_usage("C:/")
    return f

log("=" * 60)
log("EXEC START 2026-11-08 (用户已确认删除 10 个 datapack 解包目录)")
f0 = free_c()
log("C: 可用(开始): %s" % human(f0))

freed = 0; ok_n = 0
for name in TARGETS:
    st, s = rmtree_dir(os.path.join(TEMP, name))
    if st == "gone":
        freed += s; ok_n += 1
    log("  %s -> %s (size=%s)" % (name, st, human(s)))
log("批次验证: 已消失 %d/%d" % (sum(1 for n2 in TARGETS if not os.path.exists(os.path.join(TEMP, n2))), len(TARGETS)))

s_left, n_left = dir_size(TEMP)
f1 = free_c()
log("Temp 剩余: %s (%d files)" % (human(s_left), n_left))
log("C: 可用(结束): %s (本次释放 %s)" % (human(f1), human(f1 - f0)))
log("结果: 删除 %d/10 个目录, 释放 %s, 跳过锁定 %d 个" % (ok_n, human(freed), len(skipped)))
log("EXEC END")
logf.close()
print("DONE 20261108")
