# -*- coding: utf-8 -*-
import re

text = open(r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md", "r", encoding="utf-8", newline="").read()
heads = [m.group(0) for m in re.finditer(r"^## Phase (318[89]|319\d|32[0-2]\d) .*$", text, re.M)]
ok = [h for h in heads if h.rstrip().endswith("[2026-10-10 13:15]")]
leftover = [h for h in heads if re.search(r"2026-10-1[01][）】]", h)]
nums = sorted(int(re.match(r"^## Phase (\d+)", h).group(1)) for h in heads)
missing = sorted(set(range(3188, 3227)) - set(nums))
out = "total=%d ok=%d leftover=%r missing=%r\n" % (len(heads), len(ok), leftover, missing)
open(r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\fix_ts_verify.txt", "w", encoding="utf-8").write(out)
