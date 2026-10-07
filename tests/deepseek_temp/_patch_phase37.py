# -*- coding: utf-8 -*-
"""为 append_memo_q03_phase37.py 插入占位符替换（避免反斜杠匹配）。"""
fp = r"D:\AI2050\Ai2050-OpenOne\tests\deepseek\append_memo_q03_phase37.py"
src = open(fp, encoding="utf-8").read()
lines = src.split("\n")
idx = [i for i, l in enumerate(lines) if l.startswith("entry_lf = ")]
assert len(idx) == 1, "found %d" % len(idx)
if ".replace('__TS__'" in src:
    print("ALREADY_PATCHED")
else:
    ins = [
        "entry_lf = (entry_lf.replace('__TS__', TS)",
        "            .replace('__DS__', ex['design_sha'][:8])",
        "            .replace('__RS__', q['res_sha8']))",
        "assert '__TS__' not in entry_lf and '__DS__' not in entry_lf and '__RS__' not in entry_lf, 'placeholder residue'",
    ]
    lines[idx[0] + 1:idx[0] + 1] = ins
    src = "\n".join(lines)
    open(fp, "w", encoding="utf-8", newline="\n").write(src)
chk = open(fp, encoding="utf-8").read()
print("PATCH_OK has_replace =", ".replace('__TS__', TS)" in chk)
print("has_assert =", "placeholder residue" in chk)
print("line_count =", chk.count("\n") + 1)
