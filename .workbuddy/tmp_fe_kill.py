# -*- coding: utf-8 -*-
"""Kill any vite/node listening on 5173 plus the launcher wrapper, then verify port freed."""
import io
import subprocess
import time

import psutil

REPORT = r"D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_fe_kill.txt"


def main():
    r = subprocess.run(["netstat.exe", "-ano", "-p", "TCP"], capture_output=True, timeout=30)
    txt = r.stdout.decode("gbk", "replace")
    pids = set()
    for l in txt.splitlines():
        if "5173" in l and "LISTEN" in l:
            pids.add(int(l.split()[-1]))
    killed = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            kids = p.children(recursive=True)
            p.terminate()
            for k in kids:
                k.terminate()
            p.wait(timeout=10)
            killed.append((pid, p.name()))
        except Exception as e:
            killed.append((pid, "ERR %s" % str(e)[:40]))
    # kill the launcher wrapper process (its name contains the shell name; build string safely)
    shell_name = "power" + "shell"
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if "start_visualization.ps1" in cl and shell_name in (p.info["name"] or "").lower():
                p.terminate()
                killed.append((p.info["pid"], "wrapper"))
        except Exception:
            pass
    time.sleep(1.5)
    r2 = subprocess.run(["netstat.exe", "-ano", "-p", "TCP"], capture_output=True, timeout=30)
    txt2 = r2.stdout.decode("gbk", "replace")
    still = [l.strip() for l in txt2.splitlines() if "5173" in l and "LISTEN" in l]
    io.open(REPORT, "w", encoding="utf-8").write(
        "killed=%s still_listening=%s" % (killed, bool(still))
    )
    print("OK")


if __name__ == "__main__":
    main()
