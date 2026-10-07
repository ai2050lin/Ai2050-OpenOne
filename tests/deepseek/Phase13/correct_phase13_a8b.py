# -*- coding: utf-8 -*-
"""Phase 13 勘误第二步：修独立复核脚本（3 处）+ 回滚 Ledger / MEMO / wlog。"""
import io
import os
import json
import hashlib
import shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
S13 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase13')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
LEDGER_BK = os.path.join(T13, 'atlas_ledger_backup_pre_phase13.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(S13, 'correct_phase13_a8b.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def patch(path, pairs):
    t = io.open(path, encoding='utf-8').read()
    for name, old, new, cnt in pairs:
        c = t.count(old)
        assert c == cnt, '[%s] %r count=%d (期望 %d)' % (os.path.basename(path), name, c, cnt)
        t = t.replace(old, new)
        w('  OK %-34s count=%d' % (name, c))
    io.open(path, 'w', encoding='utf-8').write(t)
    return t


w('=== [1] 修独立复核脚本 ===')
DV = os.path.join(S13, 'disk_verify_phase13.py')
patch(DV, [
    ('verifier perm 2 draws/iter',
     'for b in range(BP):\n    perm_x[b] = spearman(rng.permutation(XV), site_arr)\n',
     'for b in range(BP):\n    perm_x[b] = spearman(rng.permutation(XV), site_arr)\n'
     '    _ = rng.permutation(RV)   # 与 Phase 12/13 一致：每轮消费 2 次 permutation\n',
     1),
    ('verifier tighten median',
     "('R13 tighten_median == 0.5475', abs(R['A5_counts']['tighten_median'] - 0.5475) < 1e-12),",
     "('R13 tighten_median == 0.5475', round(R['A5_counts']['tighten_median'], 4) == 0.5475),",
     1),
    ('verifier A8 margin',
     "    ('A8 α=0.4 剔除时 XH_RANGE < 0.10',\n     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) < 0.10),",
     "    ('A8 α=0.4 剔除时 XH_RANGE 仍 > 0.10（不跌破）',\n"
     "     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) > 0.10),\n"
     "    ('A8 裕度压缩后 < 1%（0.1007/0.10 - 1）',\n"
     "     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) / 0.10 - 1.0 < 0.01),",
     1),
])

w('')
w('=== [2] 回滚 ===')
led_now = sha(LEDGER)
n_led = len(json.load(io.open(LEDGER, encoding='utf-8'))['measurements'])
assert n_led == 296, 'Ledger 不是 296（实际 %d）' % n_led
shutil.copy2(LEDGER_BK, LEDGER)
L2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('  Ledger %d -> %d ; sha8 %s -> %s ; backup sha8 %s' %
  (n_led, len(L2['measurements']), led_now[:8], sha(LEDGER)[:8], sha(LEDGER_BK)[:8]))
assert len(L2['measurements']) == 295 and L2['measurements'][-1]['phase'] == 12

BASE = os.path.join(T13, 'memo_baseline_preappend_phase13.json')
B = json.load(io.open(BASE, encoding='utf-8'))
cur = open(MEMO, 'rb').read()
pre = cur[:B['bytes']]
assert hashlib.sha256(pre).hexdigest() == B['sha256'], 'MEMO 前缀与基线 sha 不符（不可回滚）'
assert cur[B['bytes']:B['bytes'] + 4].decode('utf-8') == '\r\n\r\n', '截断点不是节边界'
open(MEMO, 'wb').write(pre)
w('  MEMO %d -> %d B ; sha8 %s -> %s (baseline %s)' %
  (len(cur), len(pre), hashlib.sha256(cur).hexdigest()[:8],
   hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8], B['sha8']))

curw = open(WLOG, 'rb').read()
PREW = 9249
assert len(curw) > PREW, 'wlog 未增长'
tail = curw[PREW:PREW + 40].decode('utf-8')
assert tail.startswith('\n\n## Phase 13 / N2h1-α-6'), 'wlog 截断点不对: %r' % tail
open(WLOG, 'wb').write(curw[:PREW])
w('  wlog %d -> %d B ; 截断点后原为 %r' % (len(curw), PREW, tail[:24]))

w('')
w('ROLLBACK DONE -> 可重跑 closeout_phase13 / do_append_phase13 / closeout_docs_phase13')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
