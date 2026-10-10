# -*- coding: utf-8 -*-
"""Launch vite dev server as a job-breakaway detached process (survives session cleanup)."""
import subprocess, sys, time, os

NODE = r"C:\Users\Admin\.workbuddy\binaries\node\versions\22.21.0\node.exe"
VITE = r"D:\AI2050\Ai2050-OpenOne\frontend\node_modules\vite\bin\vite.js"
CWD = r"D:\AI2050\Ai2050-OpenOne\frontend"
LOG = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\vite_5173.log"
PIDF = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\vite_5173.pid"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

out = []
# close stale pid file record first
try:
    old = open(PIDF).read().strip()
    if old:
        out.append("stale_pid=%s" % old)
except Exception:
    pass

cmd = [NODE, VITE, "--port", "5173", "--strictPort"]
logf = open(LOG, "ab")

for tag, flags in (
    ("breakaway", DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB),
    ("detached_only", DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP),
    ("newpgm", CREATE_NEW_PROCESS_GROUP),
):
    try:
        p = subprocess.Popen(cmd, cwd=CWD, stdout=logf, stderr=logf,
                             stdin=subprocess.DEVNULL, creationflags=flags)
        out.append("launched mode=%s pid=%d" % (tag, p.pid))
        break
    except Exception as e:
        out.append("mode=%s failed: %r" % (tag, e))
else:
    out.append("ALL-MODES-FAILED")
    open(LOG + ".launch.txt", "w", encoding="utf-8").write("\n".join(out))
    sys.exit(1)

with open(PIDF, "w") as f:
    f.write(str(p.pid))

# wait up to 20s for port 5173
import urllib.request
ok = False
for _ in range(20):
    time.sleep(1)
    try:
        r = urllib.request.urlopen("http://127.0.0.1:5173/", timeout=2)
        ok = (r.status == 200)
        if ok:
            break
    except Exception:
        pass

out.append("port5173=%s" % ("200" if ok else "DOWN"))
out.append("poll_objects=%s" % (p.poll()))
open(LOG + ".launch.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
