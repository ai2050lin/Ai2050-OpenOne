# -*- coding: utf-8 -*-
"""Scan for NTFS junctions/reparse points under tests root (read-only)."""
import os
import ctypes

ROOT = r"D:\AI2050\Ai2050-OpenOne\tests"
OUT = r"D:\AI2050\Ai2050-OpenOne\gpt5_temp\junction_scan.txt"

FILE_ATTRIBUTE_REPARSE_POINT = 0x400

junctions = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    for d in list(dirnames):
        full = os.path.join(dirpath, d)
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(full)
        except Exception:
            continue
        if attrs != -1 and attrs & FILE_ATTRIBUTE_REPARSE_POINT:
            try:
                target = os.readlink(full).replace("\\\\?\\", "")
            except OSError:
                target = "(readlink failed)"
            junctions.append((full, target))

lines = ["junction/reparse points under tests root: %d" % len(junctions)]
for f, t in junctions:
    lines.append("%s -> %s" % (f, t))
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("junctions found: %d" % len(junctions))
