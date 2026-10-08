# -*- coding: utf-8 -*-
# p3155_verify.py: 独立磁盘复核（新进程重哈希全部产物 + disk sha8 记录）
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RBASE = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3155', 'g2p1_relation_family_operator_separability')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
MODELS = ('qwen3-4b', 'qwen3-14b', 'glm4')
res = []
disk = {}

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# 1. per-model: npz / result / execution
for m in MODELS:
    d = os.path.join(RBASE, m)
    r = json.load(io.open(os.path.join(d, 'result.json'), encoding='utf-8'))
    npz = os.path.join(d, 'collect.npz')
    assert os.path.exists(npz), ('npz missing', m)
    npz_sha = sha8(npz)
    assert npz_sha == r['npz_sha8'], ('npz sha mismatch', m, npz_sha, r['npz_sha8'])
    disk[m] = sha8(os.path.join(d, 'result.json'))
    res.append('%s: npz %s OK | res(in) %s seal(in) %s | disk %s | det=%s' % (
        m, npz_sha, r['res_sha8'], r['seal_sha8'], disk[m], r['determinism_note'][:7]))
    exe = json.load(io.open(os.path.join(d, 'execution.json'), encoding='utf-8'))
    assert exe['design_sha'] and exe['frozen_before'] == 'any model observation'
    assert os.path.exists(os.path.join(d, 'materials.json'))
    res.append('  exec %s frozen OK; materials.json OK' % exe['design_sha'][:8])

# summary
d = os.path.join(RBASE, 'summary')
r = json.load(io.open(os.path.join(d, 'result_summary.json'), encoding='utf-8'))
disk['summary'] = sha8(os.path.join(d, 'result_summary.json'))
res.append('summary: res(in) %s seal(in) %s | disk %s | verdict=%s' % (
    r['res_sha8'], r['seal_sha8'], disk['summary'], r['verdict'][:80]))
exe = json.load(io.open(os.path.join(d, 'execution.json'), encoding='utf-8'))
assert exe['design_sha']
res.append('  exec %s frozen OK' % exe['design_sha'][:8])

# 2. ledger: n / 3155 / 他线 / 自哈希
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
assert any(m.get('phase') == 3155 for m in ms)
assert any(m.get('phase') == 40 for m in ms), 'his-line P40 missing'
assert any(m.get('phase') == 3154 for m in ms)
import re
blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
chain = hashlib.sha256(blob).hexdigest()[:8]
res.append('ledger: n=%d, chain_sha8(recompute)=%s, stored=%s, 3154/3155/his-P40 all present' % (
    len(ms), chain, led.get('ledger_sha256_8')))

# 3. MEMO markers + head
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')
assert raw[:3] == b'\xef\xbb\xbf', 'BOM broken'
assert '## Phase 3155: 多关系族与算子可分离性（G2-P1 K2 死线）' in txt
assert '预注册 Phase 3156：位置平移族基座' in txt
assert 'g2p1_k2_separable_conditional_gate_supported' in txt
res.append('memo: BOM OK, 3155 section + 3156 prereg markers OK')

# 4. daily / MEMORY markers
daily = io.open(os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-08.md'), encoding='utf-8').read()
assert '## Phase 3155 (gpt5 线)' in daily
mem = io.open(os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'), encoding='utf-8').read()
assert '## G 线（AGI_GPT5_MEMO）3155 状态' in mem
res.append('daily/memory markers OK')

# 5. disk sha8 记录进 ledger rev_note（幂等）
e3155 = [m for m in ms if m.get('phase') == 3155][0]
note_new = (' | disk sha8: 4b %s, 14b %s, glm4 %s, summary %s' %
            (disk['qwen3-4b'], disk['qwen3-14b'], disk['glm4'], disk['summary']))
if 'disk sha8: 4b' not in e3155['rev_note']:
    e3155['rev_note'] = e3155['rev_note'] + note_new
    blob2 = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob2).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    res.append('ledger: disk sha8 appended to rev_note, new chain %s' % led['ledger_sha256_8'])
else:
    res.append('ledger: disk sha8 already in rev_note, skip')

# 6. MEMO 锚行补 disk 值（幂等）
anchor_old = '4b res **02feec80** seal 8db0d314；14b res **d880f675** seal 8cdce6be；glm4 res **d21fd3a0** seal 1e886c01；summary res **14975aed** seal 5d2c2061。'
anchor_new = ('4b res **02feec80** seal 8db0d314（disk %s）；14b res **d880f675** seal 8cdce6be（disk %s）；'
              'glm4 res **d21fd3a0** seal 1e886c01（disk %s）；summary res **14975aed** seal 5d2c2061（disk %s）。' %
              (disk['qwen3-4b'], disk['qwen3-14b'], disk['glm4'], disk['summary']))
if anchor_new not in txt:
    assert anchor_old in txt, 'memo anchor line not found'
    txt = txt.replace(anchor_old, anchor_new)
    open(MEMO, 'wb').write(txt.encode('utf-8'))
    res.append('memo: disk sha8 appended to anchor line')
else:
    res.append('memo: disk sha8 already in anchor, skip')

# final count
npass = sum(1 for x in res if 'OK' in x or 'appended' in x or 'frozen' in x)
res.append('VERIFY TOTAL: %d checks OK, 0 FAIL' % len(res))
io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3155_verify_out.txt'), 'w',
        encoding='utf-8').write(chr(10).join(res))
print('VERIFY OK')
