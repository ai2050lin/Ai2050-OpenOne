# -*- coding: utf-8 -*-
# 3159 verify: independent re-hash of all artifacts + record disk sha8 into MEMO anchor line & ledger rev_note
import json, os, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3159', 'g4p2_equivalence_dynamics')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
EXPECT_RES = {'qwen3-4b': 'dccd4753', 'qwen3-14b': '7c0c8cc6', 'glm4': '6db4b360'}
EXPECT_SEAL = {'qwen3-4b': '0fbb67c0', 'qwen3-14b': 'a6320757', 'glm4': 'd18055c4'}
EXPECT_NPZ = {'qwen3-4b': '8d282e5b', 'qwen3-14b': '8394c72e', 'glm4': '5267689b'}
out = []
checks = []

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def check(name, cond, detail=''):
    checks.append((name, bool(cond), detail))

disk_sha = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    rp = os.path.join(RDIR, m, 'result.json')
    r = json.load(open(rp, encoding='utf-8'))
    check('%s res_sha8' % m, r['res_sha8'] == EXPECT_RES[m], r['res_sha8'])
    check('%s seal' % m, r['seal_sha8'] == EXPECT_SEAL[m], r['seal_sha8'])
    check('%s verdict_tag' % m, r['verdict'].endswith(r['res_sha8']))
    check('%s npz' % m, sha8(os.path.join(RDIR, m, 'collect.npz')) == EXPECT_NPZ[m])
    check('%s exec_frozen' % m, len(json.load(open(os.path.join(RDIR, m, 'execution.json'),
                                                   encoding='utf-8'))['design_sha']) == 64)
    disk_sha[m] = sha8(rp)
sp = os.path.join(RDIR, 'summary', 'result_summary.json')
s = json.load(open(sp, encoding='utf-8'))
check('summary res', s['res_sha8'] == 'f9c1fe35', s['res_sha8'])
check('summary seal', s['seal_sha8'] == 'db48b8a9')
check('summary verdict', s['verdict'].startswith('g4p2_fingerprint_consistent'))
disk_sha['summary'] = sha8(sp)
check('addendum', sha8(os.path.join(RDIR, 'qwen3-4b', 'result_addendum.json')) == 'a3ba83c9')

# ledger
led = json.loads(open(LEDGER, 'rb').read().decode('utf-8-sig'))
e3159 = [m for m in led['measurements'] if m.get('phase') == 3159]
check('ledger_3159', len(e3159) == 1)
blob_probe = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
check('ledger_chain', led.get('ledger_sha256_8') is not None, led.get('ledger_sha256_8'))
check('ledger_n', len(led['measurements']) >= 311, str(len(led['measurements'])))

# MEMO
mtxt = open(MEMO, 'rb').read()
check('memo_bom', mtxt[:3] == b'\xef\xbb\xbf')
mt = mtxt.decode('utf-8')
check('memo_sec', '## Phase 3159: 等价类动力学（G4-P2）' in mt)
check('memo_prereg3160', '### 预注册 Phase 3160' in mt)

# 记录 disk sha 到 MEMO 锚行（幂等）
if '3159 disk:' in mt:
    out.append('memo disk-sha already present')
else:
    seg = 'summary res **f9c1fe35** seal db48b8a9。'
    assert mt.count(seg) == 1, mt.count(seg)
    mt2 = mt.replace(seg, seg + '3159 disk: 4b `%s` 14b `%s` glm4 `%s` summary `%s`；' % (
        disk_sha['qwen3-4b'], disk_sha['qwen3-14b'], disk_sha['glm4'], disk_sha['summary']))
    open(MEMO, 'wb').write(mt2.encode('utf-8'))
    out.append('memo disk-sha appended')

# ledger rev_note（幂等）
if '3159 disk' in str(e3159[0].get('rev_note', '')):
    out.append('ledger rev_note already present')
else:
    e3159[0]['rev_note'] = ('disk sha8: 4b %s 14b %s glm4 %s summary %s; addendum a3ba83c9; '
                            'chain %s' % (disk_sha['qwen3-4b'], disk_sha['qwen3-14b'],
                                          disk_sha['glm4'], disk_sha['summary'],
                                          led['ledger_sha256_8']))
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger rev_note updated (chain %s)' % led['ledger_sha256_8'])

npass = sum(1 for _, v, _ in checks if v)
nfail = sum(1 for _, v, _ in checks if not v)
for n, v, d in checks:
    if not v:
        out.append('FAIL %s (%s)' % (n, d))
out.append('VERIFY: %d PASS / %d FAIL' % (npass, nfail))
out.append('disk_sha8: %s' % {k: v for k, v in disk_sha.items()})
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3159_verify_out.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('VERIFY DONE %d/%d' % (npass, npass + nfail))
