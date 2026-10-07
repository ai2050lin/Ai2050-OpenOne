# -*- coding: utf-8 -*-
"""Round 2: locate I=5.15 interaction index source; find K_d verdict in N3 reports."""
import os, re

TMP = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_review_20261001\verify_round2.txt'

lines = []

def grep_dir(patterns, prefixes, exts=('.txt',)):
    for fn in sorted(os.listdir(TMP)):
        if not fn.endswith(exts):
            continue
        if not any(fn.startswith(p) for p in prefixes):
            continue
        p = os.path.join(TMP, fn)
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            txt = f.read()
        for pat in patterns:
            for ln in txt.splitlines():
                if re.search(pat, ln):
                    lines.append('%s | /%s/ | %s' % (fn, pat, ln.strip()[:180]))

lines.append('=== A. interaction index 5.15 / 0.007 source ===')
grep_dir([r'5\.15', r'0\.007', r'inter', r'I_?all|Iall|交互'], ['n2'])

lines.append('')
lines.append('=== B. K_d / polarity verdict in N3 reports ===')
grep_dir([r'K_d', r'极性', r'neg', r'不是一种', r'cross_ctx', r'crossctx'], ['n3'])

lines.append('')
lines.append('=== C. N1b L0 scores (tied +18.87 / untied ~0) ===')
grep_dir([r'L0.*18\.|18\.87|24\.42|24\.10|\+0\.068|\+0\.40'], ['n1b_report'])

lines.append('')
lines.append('=== D. N1 A-arm dS windows (significant windows) ===')
grep_dir([r'显著窗|sig_win|dS'], ['n1v2_report_qwen3-4b'])

lines.append('')
lines.append('=== E. N2h1 zeroing necessity (random-5dim definition) ===')
grep_dir([r'随机 5 维|random.*5|zero.*rand|置零'], ['n2h1_report_qwen3-4b', 'n2h1b_report'])

with open(OUT, 'w', encoding='utf-8') as w:
    w.write('\n'.join(lines))
print('OK ->', OUT)
