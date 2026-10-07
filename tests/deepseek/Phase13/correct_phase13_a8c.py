# -*- coding: utf-8 -*-
"""Phase 13 回滚第三步：MEMO + wlog（修正边界断言）。Ledger 已在上一步回滚完毕。"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
S13 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase13')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(S13, 'correct_phase13_a8c.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


BASE = os.path.join(T13, 'memo_baseline_preappend_phase13.json')
B = json.load(io.open(BASE, encoding='utf-8'))
cur = open(MEMO, 'rb').read()
pre = cur[:B['bytes']]
assert hashlib.sha256(pre).hexdigest() == B['sha256'], 'MEMO 前缀与基线 sha 不符（不可回滚）'
# 追加格式 = rstrip(pre_text) + CRLF CRLF + body + CRLF ⇒ 前 2 字节分隔符已被前缀吃掉
nxt = cur[B['bytes']:B['bytes'] + 14].decode('utf-8')
assert nxt.startswith('\r\n## Phase 13'), '截断点不在 Phase 13 节首: %r' % nxt
open(MEMO, 'wb').write(pre)
w('  MEMO %d -> %d B ; sha8 %s -> %s (baseline sha8 %s, 逐字节一致=%s)' %
  (len(cur), len(pre), hashlib.sha256(cur).hexdigest()[:8],
   hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8], B['sha8'],
   hashlib.sha256(open(MEMO, 'rb').read()).hexdigest() == B['sha256']))
w('    BOM=%s bare_lf=%d Phase 标题=%d' %
  (pre[:3] == b'\xef\xbb\xbf', pre.count(b'\n') - pre.count(b'\r\n'),
   pre.decode('utf-8-sig').count('\n## Phase ')))

curw = open(WLOG, 'rb').read()
PREW = 9249
assert len(curw) > PREW, 'wlog 未增长'
tail = curw[PREW:PREW + 40].decode('utf-8')
assert tail.startswith('\n\n## Phase 13 / N2h1-α-6'), 'wlog 截断点不对: %r' % tail
open(WLOG, 'wb').write(curw[:PREW])
w('  wlog %d -> %d B ; 截断点后原为 %r' % (len(curw), PREW, tail[:24]))
w('    wlog 末尾 40 字符: %r' % open(WLOG, 'rb').read()[-40:].decode('utf-8'))

L = json.load(io.open(os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json'), encoding='utf-8'))
w('  Ledger 复检: n=%d ; tail phase=%s' % (len(L['measurements']), L['measurements'][-1]['phase']))
assert len(L['measurements']) == 295

w('')
w('ROLLBACK DONE -> 重跑 closeout_phase13 / do_append_phase13 / closeout_docs_phase13')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
