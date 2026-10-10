# -*- coding: utf-8 -*-
# p3157_verify.py: independent disk verification + disk-sha recording (fresh process)
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RBASE = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3157', 'g2p2_transform_algebra_commutator')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-08.md')
MEMORY = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

res = []
ok = fail = 0
def check(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        res.append('PASS %s %s' % (name, detail))
    else:
        fail += 1
        res.append('FAIL %s %s' % (name, detail))

EXPECT = {
    'qwen3-4b': ('c0db3455', '41916ed8', '95a25965'),
    'qwen3-14b': ('1270c921', '98c06ab0', '9552086d'),
    'glm4': ('a1d9a98a', 'e8be1026', '23dd74eb'),
}
disk_shas = {}
for m, (re8, se8, np8) in EXPECT.items():
    rp = os.path.join(RBASE, m, 'result.json')
    rj = json.load(io.open(rp, encoding='utf-8'))
    disk = sha8(rp)
    disk_shas[m] = disk
    check('%s res' % m, rj.get('res_sha8') == re8 and rj.get('seal_sha8') == se8,
          'disk=%s' % disk)
    check('%s npz' % m, sha8(os.path.join(RBASE, m, 'collect.npz')) == np8)
    check('%s verdict' % m, rj['verdict'].startswith('g2p2_commutative_partial'))
sp = os.path.join(RBASE, 'summary', 'result_summary.json')
sj = json.load(io.open(sp, encoding='utf-8'))
check('summary', sj.get('res_sha8') == '0fe043bf' and sj.get('fpmin_kout', 0) >= 0.8,
      'disk=%s fpmin=%.4f' % (sha8(sp), sj.get('fpmin_kout', -1)))
check('summary_fp3', all(v['pearson_kout'] >= 0.8 for v in sj['fp_pairs'].values()))

mtxt = open(MEMO, 'rb').read().decode('utf-8')
check('memo_bom', open(MEMO, 'rb').read()[:3] == b'\xef\xbb\xbf')
check('memo_3157', '## Phase 3157: 变换代数 P1——算子对易子（G2-P2）' in mtxt)
check('memo_3158', 'G4-P1 输出等价类' in mtxt and '零空间扰动' in mtxt)

led = json.load(io.open(LEDGER, encoding='utf-8'))
e = [m for m in led['measurements'] if m.get('phase') == 3157]
check('ledger_3157', len(e) == 1 and led['measurements'][-1]['phase'] == 3157,
      'n=%d' % len(led['measurements']))
check('daily_marker', '3157 G2-P2 变换代数对易子闭环' in open(DAILY, 'rb').read().decode('utf-8'))
check('memory_marker', '3157 变换代数' in open(MEMORY, 'rb').read().decode('utf-8'))

# record disk shas into ledger rev_note (idempotent)
e2 = [m for m in led['measurements'] if m.get('phase') == 3157][0]
if 'disk:' not in e2['rev_note']:
    e2['rev_note'] = e2['rev_note'] + ' | disk: 4b=%s 14b=%s glm4=%s summary=%s' % (
        disk_shas['qwen3-4b'], disk_shas['qwen3-14b'], disk_shas['glm4'], sha8(sp))
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    res.append('ledger: disk shas recorded (chain=%s)' % led['ledger_sha256_8'])
else:
    res.append('ledger: disk shas already present')
anchor = '**主判决：`g2p2_commutative_partial|fpmin_0.986|exch_mean_1.273`（res `0fe043bf` / seal `cfa5c3ed`）**'
if anchor in mtxt and '3157 disk:' not in mtxt:
    mtxt2 = mtxt.replace(anchor, anchor + '；3157 disk: 4b `%s` 14b `%s` glm4 `%s` summary `%s`' % (
        disk_shas['qwen3-4b'], disk_shas['qwen3-14b'], disk_shas['glm4'], sha8(sp)))
    open(MEMO, 'wb').write(mtxt2.encode('utf-8'))
    res.append('memo: disk shas appended to verdict line')
else:
    res.append('memo: disk line present or anchor missing-check')

res.append('VERIFY TOTAL: PASS=%d FAIL=%d' % (ok, fail))
with io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3157_verify_out.txt'), 'w', encoding='utf-8') as f:
    f.write(chr(10).join(res))
print('written')
