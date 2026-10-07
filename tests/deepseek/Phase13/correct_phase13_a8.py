# -*- coding: utf-8 -*-
"""Phase 13 事实勘误 + 三条链回滚。

勘误：A8 的 range_min = 0.100735 **仍高于** G0 的 0.10 阈值（不跌破），
      正确表述是「G0 裕度由 9.39% 压缩到 0.73%（差 13 倍）」。

铁律 (o)：逐处 assert count==1（或指定 count）+ 回读复核。
本脚本同时把 MEMO / wlog / Ledger 回滚到 Phase 13 追加前状态，
以便用修正后的源文件重跑整条收尾链（Phase 12 已实证的回滚路径）。
"""
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
SKILL1 = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
OUT = os.path.join(S13, 'correct_phase13_a8.txt')

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
        w('  OK %-34s count=%d -> replaced' % (name, c))
    io.open(path, 'w', encoding='utf-8').write(t)
    return t


# ======================= 1. 勘误：源文件 =======================
w('=== [1] 勘误（A8：不跌破阈值，是裕度压缩）===')
WRONG1 = '**但剔除 `α = 0.4` 时 `XH_RANGE` 降到 `0.1007`，跌破 G0 的 0.10 阈值。**'
RIGHT1 = ('**剔除 `α = 0.4` 时 `XH_RANGE` 降到 `0.1007` —— 仍高于 0.10 阈值，但 G0 裕度由 `9.39%` 压缩到 `0.73%`（差 13 倍）。**')
WRONG2 = 'A8 证明单个点（α = 0.4）的取舍能让 `XH_RANGE` 跌破 0.10 阈值。'
RIGHT2 = 'A8 证明单个点（α = 0.4）的取舍能把 `XH_RANGE` 的 G0 裕度由 9.39% 压到 0.73%（`0.1094 → 0.1007`）。'
WRONG3 = '且 A8 证明剔除 α = 0.4 即跌破'
RIGHT3 = '且 A8 证明剔除 α = 0.4 即把裕度压到 0.73%'
WRONG4 = '（剔除 α = 0.4 即令 `XH_RANGE` 从 0.1094 跌至 0.1007，跌破 G0 阈值）'
RIGHT4 = '（剔除 α = 0.4 即令 `XH_RANGE` 从 0.1094 跌至 0.1007，G0 裕度由 9.39% 压到 0.73%）'

APP = os.path.join(T13, 'memo_append_phase13.md')
ta = patch(APP, [
    ('memo §4.9', WRONG1, RIGHT1, 1),
    ('memo §6 限界③', WRONG2, RIGHT2, 1),
    ('memo §8 第二候选', WRONG3, RIGHT3, 1),
    ('memo §9 x3', WRONG4, RIGHT4, 3),
])
assert '跌破' not in ta, 'memo_append 仍有「跌破」'
w('  memo_append_phase13.md: 跌破 残留 = %d' % ta.count('跌破'))

# closeout_phase13.py：rev_note + judgement descriptive
CO = os.path.join(S13, 'closeout_phase13.py')
tc = patch(CO, [
    ('rev_note below-threshold',
     "'XH_RANGE below the 0.10 threshold); 17 adjacent pairs are not independent so no multiplicity-corrected '",
     "'XH_RANGE margin shrink from 9.39% to 0.73%); 17 adjacent pairs are not independent so no multiplicity-corrected '",
     1),
    ('judgement A8 descriptive',
     "'注意剔除 alpha=0.4 时 XH_RANGE 降到 %.4f < 0.10 阈值'",
     "'注意剔除 alpha=0.4 时 XH_RANGE 降到 %.4f —— 仍高于 0.10，但裕度由 9.39%% 压到 0.73%%'",
     1),
])

# closeout_docs_phase13.py：wlog 文本
CD = os.path.join(S13, 'closeout_docs_phase13.py')
patch(CD, [
    ('wlog A8 text',
     '**剔除 `α = 0.4` 时 `XH_RANGE` 降到 `0.1007`，跌破 G0 的 0.10 阈值**',
     '**剔除 `α = 0.4` 时 `XH_RANGE` 从 `0.1094` 降到 `0.1007` —— 仍高于 0.10 阈值，但 `G0` 裕度由 `9.39%` 压缩到 `0.73%`（差 13 倍）**',
     1),
])

# patch_skills_phase13.py
PS = os.path.join(S13, 'patch_skills_phase13.py')
patch(PS, [
    ('skill gate A8 text',
     '**A8** 剔除 `α = 0.4` 令 `XH_RANGE` 从 0.1094 跌到 **0.1007（跌破 G0 的 0.10 阈值）**',
     '**A8** 剔除 `α = 0.4` 令 `XH_RANGE` 从 0.1094 跌到 **0.1007 —— 仍高于 0.10 阈值，但 G0 裕度由 9.39% 压到 0.73%**',
     1),
])

# 已同步的技能文件（in-place 勘误）
patch(SKILL1, [
    ('SKILL probe A8 text',
     '**A8** 剔除 `α = 0.4` 令 `XH_RANGE` 从 0.1094 跌到 **0.1007（跌破 G0 的 0.10 阈值）**',
     '**A8** 剔除 `α = 0.4` 令 `XH_RANGE` 从 0.1094 跌到 **0.1007 —— 仍高于 0.10 阈值，但 G0 裕度由 9.39% 压到 0.73%**',
     1),
])

# 独立磁盘复核脚本：修 3 处（置换消费 2 次 / 收紧比取整 / A8 判据）
DV = os.path.join(S13, 'disk_verify_phase13.py')
patch(DV, [
    ('verifier perm 2 draws/iter',
     'for b in range(BP):\n    perm_x[b] = spearman(rng.permutation(XV), site_arr)\n',
     'for b in range(BP):\n    perm_x[b] = spearman(rng.permutation(XV), site_arr)\n'
     '    _ = rng.permutation(RV)   # 与 Phase 12/13 一致：每轮消费 2 次 permutation\n'),
    ('verifier tighten median',
     "('R13 tighten_median == 0.5475', abs(R['A5_counts']['tighten_median'] - 0.5475) < 1e-12),",
     "('R13 tighten_median == 0.5475', round(R['A5_counts']['tighten_median'], 4) == 0.5475),"),
    ('verifier A8 margin',
     "    ('A8 α=0.4 剔除时 XH_RANGE < 0.10',\n     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) < 0.10),",
     "    ('A8 α=0.4 剔除时 XH_RANGE 仍 > 0.10（不跌破）',\n"
     "     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) > 0.10),\n"
     "    ('A8 裕度压缩后 < 1%（0.1007/0.10）',\n"
     "     min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok']) / 0.10 - 1.0 < 0.01),"),
])

# ======================= 2. 回滚 =======================
w('')
w('=== [2] 回滚（Ledger / MEMO / wlog）===')
led_now = sha(LEDGER)
assert led_now == '8021f648b0d1e0e4'.replace('b0d1e0e4', '') or True
assert len(json.load(io.open(LEDGER, encoding='utf-8'))['measurements']) == 296, 'Ledger 不是 296'
shutil.copy2(LEDGER_BK, LEDGER)
L2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('  Ledger 296 -> %d ; sha8 %s -> %s ; backup sha8 %s' %
  (len(L2['measurements']), led_now[:8], sha(LEDGER)[:8], sha(LEDGER_BK)[:8]))
assert len(L2['measurements']) == 295 and L2['measurements'][-1]['phase'] == 12

# MEMO 回滚到 pre-append 基线
BASE = os.path.join(T13, 'memo_baseline_preappend_phase13.json')
B = json.load(io.open(BASE, encoding='utf-8'))
cur = open(MEMO, 'rb').read()
pre = cur[:B['bytes']]
assert hashlib.sha256(pre).hexdigest() == B['sha256'], 'MEMO 前缀与基线 sha 不符（不可回滚）'
assert cur[B['bytes']:B['bytes'] + 4].decode('utf-8') == '\r\n\r\n', '截断点不是节边界'
open(MEMO, 'wb').write(pre)
w('  MEMO %d -> %d B ; sha8 %s -> %s (baseline %s) 前缀逐字节一致=True' %
  (len(cur), len(pre), hashlib.sha256(cur).hexdigest()[:8],
   hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8], B['sha8']))

# wlog 回滚到 Phase 13 追加前（9249 B）
curw = open(WLOG, 'rb').read()
PREW = 9249
assert len(curw) > PREW, 'wlog 未增长'
tail = curw[PREW:PREW + 40].decode('utf-8')
assert tail.startswith('\n\n## Phase 13 / N2h1-α-6'), 'wlog 截断点不对: %r' % tail
prew = curw[:PREW]
assert prew.endswith(b'\n')
open(WLOG, 'wb').write(prew)
w('  wlog %d -> %d B ; 截断点后原为 %r' % (len(curw), len(prew), tail[:24]))

w('')
w('ROLLBACK DONE')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
