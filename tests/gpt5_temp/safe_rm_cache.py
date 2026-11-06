"""安全删除 HF 下载残留：只删指定目录内的普通文件，绝不跟随 symlink/junction。"""
import os
import sys
import shutil

ROOT = r"D:\AI2050\Ai2050-OpenOne"
TARGET = os.path.join(ROOT, "models", "hf", "Qwen3-14B", ".cache", "huggingface", "download")

GUARD = os.path.join(ROOT, "models", "hf", "Qwen3-14B", ".cache")
APPLY = "--apply" in sys.argv

def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"

def main():
    t = os.path.abspath(TARGET)
    g = os.path.abspath(GUARD)
    if not t.startswith(g + os.sep):
        print(f"ABORT: target not under guard\n  target={t}\n  guard={g}")
        return 2
    if not os.path.isdir(t):
        print(f"ABORT: not a dir: {t}")
        return 2

    files, dirs, links = [], [], []
    for entry in os.scandir(t):
        if entry.is_symlink():
            links.append(entry.path)
        elif entry.is_dir(follow_symlinks=False):
            dirs.append(entry.path)
        elif entry.is_file(follow_symlinks=False):
            files.append((entry.path, entry.stat(follow_symlinks=False).st_size))

    total = sum(s for _, s in files)
    print(f"target = {t}")
    print(f"files={len(files)}  subdirs={len(dirs)}  symlinks={len(links)}")
    print(f"total  = {human(total)}")
    if links:
        print("SYMLINK FOUND (will NOT touch):")
        for l in links:
            print("  ", l)

    if not APPLY:
        print("\n[dry-run] 加 --apply 才真删")
        for p, s in sorted(files, key=lambda x: -x[1])[:5]:
            print(f"  {human(s):>10}  {os.path.basename(p)[:70]}")
        return 0

    ok = err = 0
    for p, s in files:
        try:
            os.chmod(p, 0o666)
        except OSError:
            pass
        try:
            os.unlink(p)
            ok += 1
        except OSError as e:
            err += 1
            print(f"  FAIL {p}: {e}")

    for d in sorted(dirs, key=len, reverse=True):
        try:
            shutil.rmtree(d, ignore_errors=False)
        except OSError as e:
            print(f"  rmtree FAIL {d}: {e}")

    print(f"\nunlinked={ok}  failed={err}  freed~{human(total)}")
    print(f"target still exists: {os.path.exists(t)}")

    # 删除后断言：模型目录与 shard 仍在
    mdir = os.path.join(ROOT, "models", "hf", "Qwen3-14B")
    shards = [f for f in os.listdir(mdir) if f.endswith(".safetensors")]
    print(f"POST-CHECK Qwen3-14B shards = {len(shards)}")
    hf = os.path.join(ROOT, "models", "hf")
    print(f"POST-CHECK models/hf entries = {len(os.listdir(hf))}")
    return 0 if err == 0 and len(shards) == 8 else 1

sys.exit(main())
