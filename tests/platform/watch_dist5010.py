# -*- coding: utf-8 -*-
"""Watchdog for :5010 distributed service.

Runs detached (job-breakaway). Every 10s: probe http://127.0.0.1:5010/;
if down and port bindable, spawn the service detached and log the action.
Log capped: rewritten once past 800 lines.
"""
import subprocess, time, socket, os, sys, datetime, urllib.request

PY = r"D:\AI2050\Ai2050-OpenOne\.venv\Scripts\python.exe"
CWD = r"D:\AI2050\Ai2050-OpenOne\deploy"
SVC_LOG = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\dist_5010.log"
WD_LOG = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\dist_5010_watchdog.log"
PIDF = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data\dist_5010.pid"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB

def log(msg, _n=[0]):
    line = "%s %s" % (datetime.datetime.now().strftime("%m-%d %H:%M:%S"), msg)
    _n[0] += 1
    try:
        if _n[0] > 800:
            with open(WD_LOG, "w", encoding="utf-8") as f:
                f.write("log-reset\n")
            _n[0] = 1
        with open(WD_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def up():
    try:
        r = urllib.request.urlopen("http://127.0.0.1:5010/", timeout=2)
        return r.status == 200
    except Exception:
        return False

def port_free():
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", 5010))
        return True
    except OSError:
        return False
    finally:
        s.close()

log("watchdog start pid=%d" % os.getpid())
env = os.environ.copy()
env["AI2050_DIST_DIR"] = r"D:\AI2050\Ai2050-OpenOne\tests\dist_test_data"

while True:
    try:
        if not up():
            if port_free():
                logf = open(SVC_LOG, "ab")
                p = subprocess.Popen([PY, "-m", "distributed_service"], cwd=CWD, env=env,
                                     stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
                                     creationflags=FLAGS)
                with open(PIDF, "w") as f:
                    f.write(str(p.pid))
                log("spawned pid=%d" % p.pid)
                time.sleep(8)   # give it time to boot before next probe
            else:
                log("down but port held (zombie?) - not spawning")
    except Exception as e:
        try:
            log("loop err %r" % (e,))
        except Exception:
            pass
    time.sleep(10)
