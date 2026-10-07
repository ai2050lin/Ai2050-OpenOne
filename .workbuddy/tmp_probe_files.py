import os, json

WS = r"D:\AI2050\Ai2050-OpenOne"
hits = []
prune = {"node_modules", ".git", "dist", "frontend"}
for base in ("tests", "research", "shared", "scripts", "server", "."):
    root = os.path.join(WS, base) if base != "." else WS
    if not os.path.isdir(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in prune]
        for f in filenames:
            if "phase124" in f or "phase125" in f or "phase126" in f:
                hits.append(os.path.relpath(os.path.join(dirpath, f), WS))
out = "file hits (%d):\n%s" % (len(hits), "\n".join(hits[:50]))
with open(os.path.join(WS, ".workbuddy", "tmp_probe_files.txt"), "w", encoding="utf-8") as f:
    f.write(out)
print("DONE %d" % len(hits))
