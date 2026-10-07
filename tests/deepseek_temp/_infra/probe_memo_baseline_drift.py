# -*- coding: utf-8 -*-
"""
MEMO 基线漂移调查（非 Phase 轮次，落 _infra）。
=============================================================================
问题：`memo_baseline.json` 的 `post-append-phase19` 记 bytes=469161/sha8=2255b365，
      实盘 MEMO 为 469271 B / ec7be6b6（行数同为 4481）。差异 +110 B。
目标：判定
  (a) `sections` 键的生成规则（截断宽度？）——从而判断基线键是"文件真被截断"还是"仅键显示截断"；
  (b) +110 B 是否可由"标题被补全"解释；
  (c) 是否有其它 content 差异（逐行 diff vs history 快照不可得 ⇒ 只能看行数/结构不变量）。
输出：probe_memo_baseline_drift.txt
"""
import io
import os
import re
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_baseline.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'probe_memo_baseline_drift.txt')

L = []
def w(s=''):
    L.append(str(s))
    print(s)

raw = open(MEMO, 'rb').read()
bom = raw.startswith(b'\xef\xbb\xbf')
body = raw[3:] if bom else raw
crlf = body.count(b'\r\n')
bare_lf = body.count(b'\n') - crlf
txt = raw.decode('utf-8-sig')
lines = txt.split('\r\n')
sha = hashlib.sha256(raw).hexdigest()

base = json.load(io.open(BASE, 'r', encoding='utf-8'))

w('=' * 78)
w('A. 实盘 MEMO')
w('=' * 78)
w('path        : %s' % MEMO)
w('bytes       : %d' % len(raw))
w('lines       : %d' % len(lines))
w('sha256      : %s' % sha)
w('sha8        : %s' % sha[:8])
w('bom         : %s' % bom)
w('crlf        : %d' % crlf)
w('bare_lf     : %d' % bare_lf)
w()
w('baseline(tag=%s, frozen_at=%s)' % (base['tag'], base['frozen_at']))
w('  bytes=%d lines=%d sha8=%s bom=%s crlf=%d bare_lf=%d'
  % (base['bytes'], base['lines'], base['sha8'], base['bom'], base['crlf'], base['bare_lf']))
w('  dbytes = %+d' % (len(raw) - base['bytes']))
w('  dlines = %+d' % (len(lines) - base['lines']))
w('  sha_equal = %s' % (sha == base['sha256']))

# ---------------------------------------------------------------- B. 标题行
w()
w('=' * 78)
w('B. Phase 标题：实盘行 vs 基线 sections 键')
w('=' * 78)
heads = []                       # (lineno0, text)
for i, ln in enumerate(lines):
    if ln.startswith('## '):
        heads.append((i + 1, ln))

w('实盘 "## " 开头行数 = %d' % len(heads))

# 基线 sections 键
bk = list(base['sections'].keys())
w('基线 sections 条目数 = %d' % len(bk))


def norm(s):
    """宽松归一：去空白。"""
    return re.sub(r'\s+', '', s)


w()
w('--- B1. 规则判定：基线的 Phase 键是否为"实盘标题的前缀" ---')
rules = {}
for n in (40, 42, 44, 45, 46, 48, 50, 56):
    rules['[:%d]' % n] = {h[1][:n]: h for h in heads}
rules['width44'] = {}
for _ln, _h in heads:
    acc = 0
    cut = len(_h)
    for j, ch in enumerate(_h):
        acc += 2 if ord(ch) > 0x2E7F else 1
        if acc > 44:
            cut = j
            break
    rules['width44'][_h[:cut]] = (_ln, _h)

matched_rule = {}
for k in bk:
    if not k.startswith('## Phase'):
        continue
    for rn, tbl in rules.items():
        if k in tbl:
            matched_rule.setdefault(rn, []).append((k, tbl[k][0]))
w('各截断规则对"Phase 键"的命中数：')
for rn in rules:
    lst = matched_rule.get(rn, [])
    w('  %-9s : %d 命中' % (rn, len(lst)))

# 取命中最多的规则做详细对照
best = max(rules.keys(), key=lambda rn: len(matched_rule.get(rn, [])))
w()
w('--- B2. 采用规则 %s 详细对照（基线键 → 实盘行号） ---' % best)
tbl = rules[best]
miss = []
for k, ln0 in sorted(base['sections'].items(), key=lambda kv: kv[1]):
    if k in tbl:
        real_ln, real_h = tbl[k]
        flag = 'OK ' if real_ln == ln0 else 'LINE-MISMATCH'
        w('  %s L%-5d(基线) vs L%-5d(实盘)  %s' % (flag, ln0, real_ln, k[:60]))
    else:
        miss.append((k, ln0))
        w('  MISS       L%-5d(基线)  无实盘标题可按规则 %s 生成  %s' % (ln0, best, k[:60]))
w('未命中键数 = %d' % len(miss))

# ---------------------------------------------------------------- C. 反查：实盘标题是否都以完整时间戳结尾
w()
w('=' * 78)
w('C. 实盘 Phase 标题是否含完整时间戳 [YYYY-MM-DD HH:MM]')
w('=' * 78)
pat = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]$')
no_ts = []
for ln0, h in heads:
    if h.startswith('## Phase'):
        if not pat.search(h.rstrip()):
            no_ts.append((ln0, h))
w('Phase 标题数 = %d ；无完整时间戳的 = %d'
  % (len([1 for _l, h in heads if h.startswith('## Phase')]), len(no_ts)))
for ln0, h in no_ts:
    w('  L%-5d %s' % (ln0, h))

# ---------------------------------------------------------------- D. 基线键里"看起来被截断"的
w()
w('=' * 78)
w('D. 基线键中"尾部不像完整标题"的条目')
w('=' * 78)
susp = []
for k, ln0 in sorted(base['sections'].items(), key=lambda kv: kv[1]):
    kk = k.rstrip()
    if kk.startswith('## Phase') and not pat.search(kk):
        susp.append((ln0, kk))
w('可疑条目数 = %d' % len(susp))
for ln0, k in susp:
    w('  L%-5d len=%-3d %r' % (ln0, len(k), k))

w()
w('--- D1. 若这些标题在基线时刻为"截断形式"、其后被补全，字节差估算 ---')
est = 0
full_by_line = {ln0: h for ln0, h in heads}
for ln0, k in susp:
    fh = full_by_line.get(ln0)
    if fh is None:
        w('  L%-5d 实盘同行为非标题 ⇒ 无法估算' % ln0)
        continue
    # 用"实盘完整标题"反推：假设基线文件里该行 = 标题被裁到 k 的长度（同前缀）
    d = len(fh.encode('utf-8')) - len(k.encode('utf-8'))
    est += d
    w('  L%-5d 完整=+%d B   %s' % (ln0, d, fh[:70]))
w('  估算合计 = %+d B   (实盘-基线 = %+d B)'
  % (est, len(raw) - base['bytes']))

# ---------------------------------------------------------------- E. 结构不变量
w()
w('=' * 78)
w('E. 结构不变量')
w('=' * 78)
ph_lines_real = [i + 1 for i, ln in enumerate(lines) if re.match(r'^## Phase \d+:', ln)]
w('实盘 phase_headings = %s' % ph_lines_real)
w('基线 phase_headings = %s' % base['phase_headings'])
w('相同 = %s' % (ph_lines_real == base['phase_headings']))
w('实盘 Phase 标题数 = %d（基线 %d）' % (len(ph_lines_real), len(base['phase_headings'])))
w('含 "## Phase 20" = %s' % any('## Phase 20' in ln for ln in lines))
w('末行 = %r' % lines[-1][:80])
w('末3行 =')
for ln in lines[-3:]:
    w('   %r' % ln[:100])

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('\nWROTE %s' % OUT)
