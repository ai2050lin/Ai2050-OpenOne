# -*- coding: utf-8 -*-
"""基线 sections 键异常的三假设检验（L2305/L3151/L3429）。

H1  基线时刻标题为「短时间戳」[HH:MM]，现盘为 [YYYY-MM-DD HH:MM]。
H2  基线时刻标题本身在文件里被截断为恰好 44 字符。
H3  基线时刻标题在 44 字符后被截断但保留了短时间戳（H1∩其它）。
输出 probe_heading_forms.txt
"""
import io
import os
import re
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'probe_heading_forms.txt')

L = []
def w(s=''):
    L.append(str(s))
    print(s)

raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8-sig')
lines = txt.split('\r\n')
base = json.load(io.open(BASE, encoding='utf-8'))
sec = base['sections']

w('=' * 78)
w('三假设检验：基线键 vs 现盘标题')
w('=' * 78)

TARGETS = [2305, 3151, 3429]
pat_full = re.compile(r'\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})\]$')

tot_h1 = 0
for ln in TARGETS:
    cur = lines[ln - 1]
    key = None
    for k, v in sec.items():
        if v == ln:
            key = k
            break
    w('')
    w('--- L%d ---' % ln)
    w('  基线键 (len=%d) : %r' % (len(key), key))
    w('  现盘标题(len=%d): %r' % (len(cur), cur))
    w('  现盘 [:44]      : %r' % cur[:44])
    w('  H2 成立? %s' % (cur[:44] == key))
    m = pat_full.search(cur)
    short = cur[:m.start()] + '[' + m.group(2) + ']' if m else cur
    w('  H1 短形式      : %r' % short)
    w('  H1 短形式[:44] : %r' % short[:44])
    ok = (short[:44] == key)
    w('  H1 成立? %s' % ok)
    if ok:
        tot_h1 += len(cur.encode('utf-8')) - len(short.encode('utf-8'))
    # 并列候选：仅 HH:MM（含 '[' 共 7 字符）→ [+11 B]；含秒 → 更多
    w('  长度: 现盘 %d B ; 短形式 %d B ; Δ(H1) = %+d B'
      % (len(cur.encode('utf-8')), len(short.encode('utf-8')),
         len(cur.encode('utf-8')) - len(short.encode('utf-8'))))

w('')
w('=' * 78)
w('合计：H1 下三行字节增 %+d B ；实盘-基线 = %+d B'
  % (tot_h1, len(raw) - base['bytes']))
w('=' * 78)
w('尚未解释的残余 = %+d B' % ((len(raw) - base['bytes']) - tot_h1))

# 进一步：若三个标题以外的任何行有变化，只能靠"行数不变 + 字节变化"推断。
# 给出"若仅有这 3 行变化且形式为 H1，则基线应为 N B"的推算
w('')
w('若三行均为 H1 形式且其余行逐字节相同，则基线应为 %d B（实测基线 %d B，差 %+d B）'
  % (base['bytes'] + 0, base['bytes'], 0))
est = len(raw) - tot_h1
w('  即：现盘 - H1三行Δ = %d B  ←应是基线字节' % est)
w('  基线实测          = %d B' % base['bytes'])
w('  残差              = %+d B' % (est - base['bytes']))

w('')
w('=' * 78)
w('检查：这三个标题是否在同一 Phase 内被"改判/纠错"过（append-only 语境下的补写）')
w('=' * 78)
for ln in TARGETS:
    lo = max(0, ln - 1)
    hi = min(len(lines), ln + 40)
    hits = [i + 1 for i in range(lo, hi) if '标题' in lines[i] or 'heading' in lines[i].lower()]
    w('  L%-5d 后续 40 行内含「标题/heading」的行: %s' % (ln, hits or '无'))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('\nWROTE', OUT)
