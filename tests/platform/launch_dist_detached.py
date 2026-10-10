# -*- coding: utf-8 -*-
"""Launch :5010 with the BASE uv interpreter directly (no venv trampoline).

v3 rationale: venv python.exe is a trampoline with a kill-on-close job (kills the
real interpreter with it). The survivors (vite node.exe) were trampoline-free.
PYTHONPATH points at the venv site-packages so fastapi/uvicorn resolve.
"""
import subprocess, sys, os, time, urllib.request

BASE_PY = r"C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.14-windows-x86_64-none\python.exe"
VENV_SP = r"D:\AI2050\Ai2050-OpenOne\.venv\Lib\site-packages"
CWD = r"D:\AI2050\Ai2050-OpenOne\deploy"
SVC_LOG = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\dist_5010.log"
PIDF = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\dist_5010.pid"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\dist_5010_launch.txt"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB

def http_up(timeout=2):
    try:
        return urllib.request.urlopen("http://127.0.0.1:5010/", timeout=timeout).status == 200
    except Exception:
        return False

out = []
env = os.environ.copy()
env["AI2050_DIST_DIR"] = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data"
env["PYTHONPATH"] = VENV_SP

logf = open(SVC_LOG, "ab")
p = subprocess.Popen([BASE_PY, "-m", "distributed_service"], cwd=CWD, env=env,
                     stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
                     creationflags=FLAGS)
out.append("direct_pid=%d" % p.pid)
with open(PIDF, "w") as f:
    f.write(str(p.pid))

ok = False
for _ in range(40):
    time.sleep(1)
    if http_up():
        ok = True
        break
    if p.poll() is not None:
        out.append("died early rc=%s" % p.poll())
        break
out.append("port5010=%s" % (200 if ok else "DOWN"))
txt = "\n".join(out)
open(REPORT, "w", encoding="utf-8").write(txt)
print(txt)
sys.exit(0 if ok else 2)
