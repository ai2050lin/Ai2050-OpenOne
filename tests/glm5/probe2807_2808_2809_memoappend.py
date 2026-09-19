# -*- coding: utf-8 -*-
"""Append Phase 2807+2808+2809 sections to AGI_GPT5_MEMO.md (guarded)."""
import io

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
SRC = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\memo_append_2807_2808_2809.md"
CHK = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\probe2807_2808_2809_memocheck.txt"
MARK = "Phase 2807（自动续研，LPF-20"

with io.open(MEMO, "r", encoding="utf-8") as f:
    cur = f.read()

if MARK in cur:
    msg = "SKIP: marker already present\n"
else:
    with io.open(SRC, "r", encoding="utf-8") as f:
        add = f.read()
    with io.open(MEMO, "a", encoding="utf-8") as f:
        f.write(add)
    with io.open(MEMO, "r", encoding="utf-8") as f:
        cur2 = f.read()
    ok7 = "Phase 2807（自动续研，LPF-20" in cur2
    ok8 = "Phase 2808（用户指令，LPF-21" in cur2
    ok9 = "Phase 2809（自动续研，LPF-22" in cur2
    msg = "APPENDED ok2807=%s ok2808=%s ok2809=%s total_lines=%d size=%d\n" \
        % (ok7, ok8, ok9, cur2.count("\n"), len(cur2))

with io.open(CHK, "w", encoding="utf-8") as f:
    f.write(msg)
print("done")
