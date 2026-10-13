# -*- coding: utf-8 -*-
"""Diagnose which heading in 3188-3226 does not match/transform."""
import re

PATH = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\fix_ts_diag.txt"

HEAD_RE = re.compile(r"^## Phase (3(1(88|9\d)|2[0-2]\d)) .*$", re.M)

with open(PATH, "r", encoding="utf-8", newline="") as f:
    text = f.read()

nums = [int(m.group(1)) for m in HEAD_RE.finditer(text)]
missing = sorted(set(range(3188, 3227)) - set(nums))

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("matched=%d\nmissing=%r\n\n" % (len(nums), missing))
    for m in HEAD_RE.finditer(text):
        f.write(repr(m.group(0)) + "\n")
print("OK")
