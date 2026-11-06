"""从回收站定向抹除指定来源的条目，验证是否能真正释放空间。

用法:
  python purge_recycle_subset.py --match wtest.bin          # 预演
  python purge_recycle_subset.py --match wtest.bin --apply  # 真删

按 $I 元数据里的原始路径做子串匹配，只处理命中项；删 $R 实体再删 $I 索引。
"""
import os
import shutil
import struct
import sys

RB = r"D:\$RECYCLE.BIN"
APPLY = "--apply" in sys.argv
MATCH = []
for i, a in enumerate(sys.argv):
    if a == "--match" and i + 1 < len(sys.argv):
        MATCH.append(sys.argv[i + 1])

def free_gb():
    return shutil.disk_usage("D:\\").free / 2**30

def parse(sid):
    rows = []
    for f in os.listdir(sid):
        if not f.startswith("$I"):
            continue
        p = os.path.join(sid, f)
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        if len(data) < 28:
            continue
        size = struct.unpack("<Q", data[8:16])[0]
        plen = struct.unpack("<I", data[24:28])[0]
        path = data[28:28 + plen * 2].decode("utf-16-le", "replace").rstrip("\x00")
        rows.append({"size": size, "path": path,
                     "I": f, "R": "$R" + f[2:], "dirR": "$R" + f[2:]})
    return rows

sid = [os.path.join(RB, d) for d in os.listdir(RB) if d.startswith("S-1-5-21")][0]
rows = parse(sid)

def hit(path):
    if not MATCH:
        return False
    return any(m.lower() in path.lower() for m in MATCH)

sel = [r for r in rows if hit(r["path"])]
tot = sum(r["size"] for r in sel)
print(f"匹配 {len(sel)} 项, 表观 {tot/2**30:.2f} GB")
print(f"free before = {free_gb():.2f} GB\n")
for r in sorted(sel, key=lambda x: -x["size"])[:10]:
    print(f"  {r['size']/2**30:7.2f} GB  {r['path'][:110]}")

if not APPLY:
    print("\n[dry-run] 加 --apply 才真删")
    sys.exit(0)

ok = err = 0
for r in sel:
    ri = os.path.join(sid, r["I"])
    rr = os.path.join(sid, r["R"])
    try:
        if os.path.isdir(rr) and not os.path.islink(rr):
            shutil.rmtree(rr)
        elif os.path.exists(rr):
            os.unlink(rr)
        os.unlink(ri)
        ok += 1
    except OSError as e:
        err += 1
        print(f"  FAIL {r['path'][:80]}: {e}")

print(f"\npurged={ok} failed={err}")
print(f"free after  = {free_gb():.2f} GB")

still = parse(sid)
print(f"回收站剩余 {len(still)} 项, {sum(x['size'] for x in still)/2**30:.2f} GB")
