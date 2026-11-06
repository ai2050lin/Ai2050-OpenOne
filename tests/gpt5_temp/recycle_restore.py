"""从回收站按原始路径还原被误删的文件（同卷 rename，不占额外空间）。
只还原指定前缀下的条目；目标已存在则跳过。用法：python recycle_restore.py <前缀> [--apply]
"""
import os
import struct
import sys
from collections import defaultdict

RB = r"D:\\$RECYCLE.BIN"
APPLY = "--apply" in sys.argv
prefix = next((a for a in sys.argv[1:] if not a.startswith("--")), r"D:\AI2050\Ai2050-OpenOne\models\hf")
prefix = prefix.replace("/", "\\")  # Git Bash 会把反斜杠参数转成斜杠

sid = [os.path.join(RB, d) for d in os.listdir(RB) if d.startswith("S-1-5-21")][0]
LP = "\\\\?\\"

plan = []
skipped_exist = 0
for f in os.listdir(sid):
    if not f.startswith("$I"):
        continue
    data = os.path.join(sid, f)
    try:
        blob = open(data, "rb").read()
    except OSError:
        continue
    if len(blob) < 28:
        continue
    size = struct.unpack("<Q", blob[8:16])[0]
    plen = struct.unpack("<I", blob[24:28])[0]
    orig = blob[28:28 + plen * 2].decode("utf-16-le", errors="replace").rstrip("\x00")
    if not orig.startswith(prefix):
        continue
    src = os.path.join(sid, "$R" + f[2:])
    if not os.path.exists(src):
        continue
    if os.path.exists(orig):
        skipped_exist += 1
        continue
    plan.append((src, orig, size))

tot = sum(p[2] for p in plan)
print(f"前缀: {prefix}")
print(f"可还原 {len(plan)} 项, 合计 {tot/2**30:.2f} GB；目标已存在跳过 {skipped_exist} 项")
if not APPLY:
    print("[dry-run] 加 --apply 执行")
    sys.exit(0)

done = 0
freed = 0
for src, orig, size in plan:
    try:
        d = os.path.dirname(orig)
        if not os.path.isdir(LP + d):
            os.makedirs(LP + d, exist_ok=True)
        os.replace(LP + src, LP + orig)
        done += 1
        freed += size
    except OSError as e:
        print("FAIL", orig, e)
    # 同步清理 $I 元数据
    meta = os.path.join(sid, "$I" + os.path.basename(src)[2:])
    if os.path.exists(meta) and not os.path.exists(src):
        try:
            os.remove(meta)
        except OSError:
            pass

print(f"已还原 {done} 项, {freed/2**30:.2f} GB")
