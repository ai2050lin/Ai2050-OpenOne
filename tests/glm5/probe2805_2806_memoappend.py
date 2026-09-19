# -*- coding: utf-8 -*-
"""Append Phase 2805+2806 sections to AGI_GPT5_MEMO.md (idempotency guard by marker)."""
import io, os

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
SRC = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\memo_append_2805_2806.md"
MARK = "Phase 2805（自动续研，LPF-18"

with io.open(MEMO, "r", encoding="utf-8") as f:
    cur = f.read()

if MARK in cur:
    with io.open(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2805_2806_memocheck.txt", "w", encoding="utf-8") as f:
        f.write("SKIP: marker already present\n")
else:
    with io.open(SRC, "r", encoding="utf-8") as f:
        add = f.read()
    with io.open(MEMO, "a", encoding="utf-8") as f:
        f.write(add)
    # verify on real disk
    with io.open(MEMO, "r", encoding="utf-8") as f:
        cur2 = f.read()
    ok18 = "Phase 2805（自动续研，LPF-18" in cur2
    ok19 = "Phase 2806（自动续研，LPF-19" in cur2
    nlines = cur2.count("\n")
    with io.open(r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2805_2806_memocheck.txt", "w", encoding="utf-8") as f:
        f.write("APPENDED ok2805=%s ok2806=%s total_lines=%d size=%d\n" % (ok18, ok19, nlines, len(cur2)))
print("done")
