# -*- coding: utf-8 -*-
# p3158_verify.py: Phase 3158 独立磁盘复核（新进程重哈希全部产物 + disk sha 记录）
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                 'phase3158', 'g4p1_output_equivalence_class')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
out = []
ok = [0, 0]

def check(name, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    ok[0 if cond else 1] += 1
    out.append('%s %s %s' % (tag, name, detail))

def sha8f(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

disk = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4', os.path.join('summary')):
    mm = m if isinstance(m, str) else m
    b = os.path.join(R, m)
    rj = os.path.join(b, 'result.json' if m != 'summary' else 'result_summary.json')
    r = json.load(open(rj, encoding='utf-8'))
    ds = sha8f(rj)
    disk[mm] = ds
    check('%s res_sha8 file' % mm, r['res_sha8'] == r['verdict'].split('|sha8_')[1], r['res_sha8'])
    check('%s seal embedded' % mm, len(r.get('seal_sha8', '')) == 8, r.get('seal_sha8'))
    npzf = os.path.join(b, 'collect.npz')
    if os.path.exists(npzf):
        check('%s npz_sha8' % mm, sha8f(npzf) == r['npz_sha8'], r['npz_sha8'])
    exe = json.load(open(os.path.join(b, 'execution.json'), encoding='utf-8'))
    eblob = json.dumps(exe['design'], ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    check('%s design_sha drift' % mm, hashlib.sha256(eblob).hexdigest() == exe['design_sha'], exe['design_sha'][:8])

# summary 读三模型 result 校验链
sm = json.load(open(os.path.join(R, 'summary', 'result_summary.json'), encoding='utf-8'))
check('summary fpmin', sm['fpmin_spec'] >= 0.8 and sm['fpmin_curve'] >= 0.8,
      '%.4f/%.4f' % (sm['fpmin_spec'], sm['fpmin_curve']))

# ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
e3158 = [m for m in led['measurements'] if m.get('phase') == 3158]
check('ledger 3158 entry', len(e3158) == 1, 'n=%d' % len(led['measurements']))
check('ledger chain field', len(led.get('ledger_sha256_8', '')) == 8, led.get('ledger_sha256_8'))

# MEMO
mtxt = open(MEMO, 'rb').read()
check('memo BOM', mtxt[:3] == b'\xef\xbb\xbf', mtxt[:3].hex())
mt = mtxt.decode('utf-8')
check('memo 3158 section', '## Phase 3158: 输出等价类 P1' in mt, '')
check('memo 3159 prereg', 'Phase 3159 预注册（G4-P2 等价类动力学' in mt, '')
check('memo verdict line', 'g4p1_fingerprint_consistent|quotient_0/3_mixed' in mt, '')

# disk sha 记录（幂等）
tag = '3158 disk:'
if tag not in mt:
    anchor = '（res `fa5cca12` / seal `e0c60629`）'
    assert mt.count(anchor) == 1, mt.count(anchor)
    mt2 = mt.replace(anchor, anchor + '；3158 disk: 4b `%s` 14b `%s` glm4 `%s` summary `%s`' % (
        disk['qwen3-4b'], disk['qwen3-14b'], disk['glm4'], disk['summary']))
    open(MEMO, 'wb').write(mt2.encode('utf-8'))
    out.append('memo: disk shas appended')
else:
    out.append('memo: disk shas already present')
# ledger rev_note 补 disk
if 'disk:' not in e3158[0]['rev_note']:
    e3158[0]['rev_note'] += ('; disk 4b %s 14b %s glm4 %s summary %s' % (
        disk['qwen3-4b'], disk['qwen3-14b'], disk['glm4'], disk['summary']))
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger: rev_note disk shas appended (chain %s)' % led['ledger_sha256_8'])
else:
    out.append('ledger: rev_note already has disk')

out.append('TOTAL PASS=%d FAIL=%d' % (ok[0], ok[1]))
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3158_verify_out.txt'), 'w', encoding='utf-8').write(chr(10).join(out))
print('written')
