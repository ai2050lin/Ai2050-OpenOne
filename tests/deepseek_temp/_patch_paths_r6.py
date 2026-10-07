# -*- coding: utf-8 -*-
"""R6 补丁：tests/deepseek/ 下耐用脚本内嵌的 `_review` 输出目录已不存在（本索引已归位），
统一改指 tests/deepseek/result/。逐文件断言 + 复核。"""
import os, hashlib

ROOT = r"D:\AI2050\Ai2050-OpenOne"
DUR = os.path.join(ROOT, r"tests\deepseek")
REPORT = os.path.join(ROOT, r"tests\deepseek_temp\_patch_paths_r6_report.txt")
log = []
def A(s): log.append(s)

RULES = [
    ('"tests", "deepseek_temp", "_review"', '"tests", "deepseek", "result"'),
    ('"tests/deepseek_temp/_review"', '"tests/deepseek/result"'),
    ('tests/deepseek_temp/_review', 'tests/deepseek/result'),
    ('tests\\deepseek_temp\\_review', 'tests\\deepseek\\result'),
    ('tests/deepseek/_review/', 'tests/deepseek/'),
    ("'tests', 'deepseek_temp', 'memo_review_20261001'", "'tests', 'deepseek', 'result', 'memo_review_20261001'"),
]

files = sorted(f for f in os.listdir(DUR) if f.endswith(".py") and os.path.isfile(os.path.join(DUR, f)))
A("durable scripts = %d" % len(files))
total = 0
for fn in files:
    fp = os.path.join(DUR, fn)
    raw = open(fp, "rb").read()
    src = raw.decode("utf-8")
    orig = src
    hits = {}
    for old, new in RULES:
        c = src.count(old)
        if c:
            src = src.replace(old, new)
            hits[old] = c
    if src != orig:
        open(fp, "w", encoding="utf-8", newline="").write(src)
        back = open(fp, "rb").read().decode("utf-8")
        assert back == src, "readback mismatch %s" % fn
        n = sum(hits.values())
        total += n
        A("  PAT %-34s %2d 处  sha8 %s -> %s" % (fn, n, hashlib.sha256(raw).hexdigest()[:8], hashlib.sha256(back.encode()).hexdigest()[:8]))
        for k, v in hits.items():
            A("        %dx  %s" % (v, k))
A("total edits = %d" % total)

# verify no dead path remains
bad = []
for fn in files:
    t = open(os.path.join(DUR, fn), "rb").read().decode("utf-8")
    if "deepseek_temp/_review" in t or "deepseek_temp\\_review" in t or "deepseek/_review/" in t:
        bad.append(fn)
A("残留死路径文件 = %s" % bad)
assert not bad, bad
# 反向确认未误伤 propositions_review
n_prop = sum(open(os.path.join(DUR, f), "rb").read().decode("utf-8").count("propositions_review") for f in files)
A("propositions_review 出现次数（应保持不变，>0） = %d" % n_prop)

txt = "\n".join(log)
open(REPORT, "w", encoding="utf-8").write(txt)
print(txt)
print("PATCH_PATHS_OK")
