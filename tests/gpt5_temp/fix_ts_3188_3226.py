# -*- coding: utf-8 -*-
"""Fix Phase 3188-3226 heading timestamps to [2026-10-10 13:15] format."""
import re

PATH = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\fix_ts_3188_3226_report.txt"
TS = "[2026-10-10 13:15]"

# headings: ## Phase 3188..3226 (exclude '### 预注册')
HEAD_RE = re.compile(r"^## Phase (3(18[89]|19\d|2[0-2]\d)) .*$", re.M)

with open(PATH, "r", encoding="utf-8", newline="") as f:
    text = f.read()

orig_text = text
changed = []

def fix(m):
    line = m.group(0)
    new = re.sub(r"\s*【2026-10-\d{2}】\s*$", "", line)          # trailing 【date】
    new = re.sub(r"，\s*2026-10-\d{2}\s*）\s*$", "）", new)       # date as last item inside （）
    if TS not in new:
        new = new.rstrip() + " " + TS
    if new != line:
        changed.append((line, new))
    return new

text = HEAD_RE.sub(fix, text)

assert len(changed) == 39, f"expected 39 headings, got {len(changed)}"

with open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(text)

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("changed=%d\n\n" % len(changed))
    for old, new in changed:
        f.write("OLD: %s\nNEW: %s\n\n" % (old, new))
print("OK changed=%d" % len(changed))
