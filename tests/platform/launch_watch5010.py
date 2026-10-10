# -*- coding: utf-8 -*-
"""Launch the :5010 watchdog detached (job-breakaway). Single-shot, then exit."""
import subprocess, sys, os

PY = r"D:\AI2050\Ai2050-OpenOne\.venv\Scripts\python.exe"
SCRIPT = r"D:\AI2050\Ai2050-OpenOne\tests\platform\watch_dist5010.py"
LOG = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\watch_dist5010.log"
PIDF = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\watch_dist5010.pid"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\watch_launch.txt"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

out = []
p = None
for tag, flags in (
    ("breakaway", DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB),
    ("detached_only", DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP),
    ("newpgm", CREATE_NEW_PROCESS_GROUP),
):
    try:
        logf = open(LOG, "ab")
        p = subprocess.Popen([PY, "-u", SCRIPT], cwd=r"D:\AI2050\Ai2050-OpenOne",
                             stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
                             creationflags=flags, env=os.environ.copy())
        out.append("launched mode=%s pid=%d" % (tag, p.pid))
        break
    except Exception as e:
        out.append("mode=%s failed: %r" % (tag, e))
if p is None:
    out.append("ALL-MODES-FAILED")
    open(REPORT, "w", encoding="utf-8").write("\n".join(out))
    sys.exit(1)
with open(PIDF, "w") as f:
    f.write(str(p.pid))
import time
time.sleep(3)
alive = p.poll() is None
out.append("watchdog_alive=%s" % alive)
open(REPORT, "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
