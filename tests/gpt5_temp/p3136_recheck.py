# -*- coding: utf-8 -*-
import io, json, os
ROOT = r'D:\AI2050\Ai2050-OpenOne'
out = []
# 1 ledger real readback
led = json.load(io.open(ROOT + r'\research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
e36 = [e for e in led['measurements'] if e.get('phase') == 3136]
out.append('ledger n=%d 3136_entries=%d sha8=%s' % (len(led['measurements']), len(e36), led['ledger_sha256_8']))
if e36:
    out.append('ledger verdict in entry: %s' % e36[0]['verdict'][:60])
    out.append('ledger result sha: %s' % e36[0]['hashes']['result_sha256_8'])
# 2 wlog real readback
wl = ROOT + r'\.workbuddy\memory\2026-09-28.md'
ex = os.path.exists(wl)
out.append('wlog exists=%s' % ex)
if ex:
    t = io.open(wl, encoding='utf-8').read()
    out.append('wlog len=%d has3136=%s' % (len(t), '3136' in t))
    out.append('wlog tail: %s' % t[-260:].replace('\n', ' | '))
# 3 memo real readback
m = io.open(ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
i = m.rfind('## Phase 3136')
sec = m[i:]
out.append('memo 3136 section len=%d' % len(sec))
out.append('memo has sha 3903af46: %s' % ('3903af46' in sec))
out.append('memo has key numbers: 0.4219=%s 0.0781=%s 0.2422=%s' % ('0.4219' in sec, '0.0781' in sec, '0.2422' in sec))
out.append('memo has 3137 prereg: %s' % ('3137 预注册' in sec))
out.append('memo title: %s' % sec[:60].split('\n')[0])
# 4 wmem
mm = io.open(ROOT + r'\.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
out.append('wmem: max=3136 next 3137 = %s' % ('max=3136，下一 3137' in mm))
out.append('wmem: has 3136 line = %s' % ('3136（T4）' in mm))
out.append('wmem: len=%d' % len(mm))
io.open(ROOT + r'\tests\gpt5_temp\p3136_recheck.txt','w',encoding='utf-8').write('\n'.join(out))
print('OK')
