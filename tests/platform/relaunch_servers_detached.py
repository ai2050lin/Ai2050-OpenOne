# -*- coding: utf-8 -*-
"""Relaunch :5001 server and :5010 distributed service as job-breakaway detached processes."""
import subprocess, sys, time, os, signal

PY = r"D:\AI2050\Ai2050-OpenOne\.venv\Scripts\python.exe"
ROOT = r"D:\AI2050\Ai2050-OpenOne"
DEPLOY = os.path.join(ROOT, "deploy")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB

JOBS = [
    # (name, cwd, cmd, env, port, log, pidfile)
    ("server5001", ROOT, [PY, "-m", "server.server"],
     {"AI2050_SKIP_MODEL_LOAD": "1"}, 5001,
     os.path.join(ROOT, r"tests\dist_test_data\server_5001.log"),
     os.path.join(ROOT, r"tests\dist_test_data\server_5001.pid")),
    ("dist5010", DEPLOY, [PY, "-m", "distributed_service"],
     {"AI2050_DIST_DIR": os.path.join(ROOT, "tests", "dist_test_data")}, 5010,
     os.path.join(ROOT, r"tests\dist_test_data\dist_5010.log"),
     os.path.join(ROOT, r"tests\dist_test_data\dist_5010.pid")),
]

out = []

# 1) kill existing listeners on both ports (they are session job objects, will be recycled anyway)
for port in (5001, 5010):
    r = subprocess.run(["netstat", "-ano"], capture_output=True)
    pids = set()
    for line in r.stdout.decode("gbk", errors="ignore").splitlines():
        if (":%d " % port) in line and "LISTENING" in line.upper():
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.add(int(parts[-1]))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            out.append("killed old %d pid=%d" % (port, pid))
        except Exception as e:
            out.append("kill %d pid=%d err=%r" % (port, pid, e))
time.sleep(2)

# 2) relaunch breakaway
import urllib.request
for name, cwd, cmd, env, port, log, pidf in JOBS:
    e = os.environ.copy()
    e.update(env)
    logf = open(log, "ab")
    p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=logf, stderr=logf,
                         stdin=subprocess.DEVNULL, creationflags=FLAGS)
    with open(pidf, "w") as f:
        f.write(str(p.pid))
    out.append("launched %s pid=%d" % (name, p.pid))

# 3) verify ports
ok = {"5001": False, "5010": False}
for _ in range(25):
    time.sleep(1)
    allok = True
    for port in ok:
        if ok[port]:
            continue
        try:
            r = urllib.request.urlopen("http://127.0.0.1:%d/" % int(port), timeout=2)
            if r.status == 200:
                ok[port] = True
        except Exception:
            pass
        # /health for 5001
        if not ok["5001"]:
            try:
                r = urllib.request.urlopen("http://127.0.0.1:5001/health", timeout=2)
                if r.status == 200:
                    ok["5001"] = True
            except Exception:
                pass
        if not ok["5010"]:
            try:
                r = urllib.request.urlopen("http://127.0.0.1:5010/", timeout=2)
                if r.status == 200:
                    ok["5010"] = True
            except Exception:
                pass
    if ok["5001"] and ok["5010"]:
        break

out.append("p5001=%s p5010=%s" % (200 if ok["5001"] else "DOWN", 200 if ok["5010"] else "DOWN"))
txt = "\n".join(out)
open(os.path.join(ROOT, r"tests\dist_test_data\relaunch_servers_report.txt"), "w", encoding="utf-8").write(txt)
print(txt)
