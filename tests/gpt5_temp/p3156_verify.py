# -*- coding: utf-8 -*-
# p3156_verify.py: independent disk verification + disk-sha recording (fresh process)
# 1) re-hash artifacts from real disk; 2) cross-check markers; 3) record disk sha8
#    into MEMO anchor line + ledger rev_note (dual-value convention from 3153/3154)
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
B4 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                  'phase3156', 'g3p1_position_shift_family', 'qwen3-4b')
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

# ---- artifact hashes (fresh read) ----
rp = os.path.join(B4, 'result.json')
rj = json.load(io.open(rp, encoding='utf-8'))
disk_res = sha8(rp)
check('result_exists', True, 'res_sha8=%s seal=%s disk=%s' % (rj.get('res_sha8'), rj.get('seal_sha8'), disk_res))
check('verdict_prefix', rj['verdict'].startswith('g3p1_rope_'), rj['verdict'][:70])
check('res_sha8_embedded', rj.get('res_sha8') == 'c8c66d9f')
ap = os.path.join(B4, 'result_addendum.json')
aj = json.load(io.open(ap, encoding='utf-8'))
check('addendum', aj.get('addendum_sha8') == 'a1b773b1' and sha8(ap) not in ('',), 'disk=%s' % sha8(ap))
npz_sha = sha8(os.path.join(B4, 'collect.npz'))
check('npz', npz_sha == '84f7db0b', npz_sha)
exe = json.load(io.open(os.path.join(B4, 'execution.json'), encoding='utf-8'))
check('exec_frozen', exe['design_sha'].startswith('bf474ef3'), exe['design_sha'][:16])
smk = os.path.join(B4, 'smoke', 'result.json')
check('smoke_sealed', os.path.exists(smk) and json.load(io.open(smk, encoding='utf-8')).get('verdict', '').startswith('SMOKE_g3p1_rope_supported'))

# ---- MEMO ----
mtxt = open(MEMO, 'rb').read().decode('utf-8')
check('memo_bom', open(MEMO, 'rb').read()[:3] == b'\xef\xbb\xbf')
check('memo_3156_section', '## Phase 3156: 位置平移族基座（G3-P1）' in mtxt)
check('memo_3157_prereg', 'G2-P2 变换代数' in mtxt and 'exch_R' in mtxt)
check('memo_anchor_res', 'c8c66d9f' in mtxt and 'b54418f2' in mtxt)

# ---- ledger ----
led = json.load(io.open(LEDGER, encoding='utf-8'))
e = [m for m in led['measurements'] if m.get('phase') == 3156]
check('ledger_3156', len(e) == 1, 'n=%d' % len(led['measurements']))
check('ledger_last', led['measurements'][-1]['phase'] == 3156)
check('chain_field', isinstance(led.get('ledger_sha256_8'), str) and len(led['ledger_sha256_8']) == 8)

# ---- daily / memory ----
check('daily_marker', '3156 G3-P1 位置平移族基座闭环' in open(DAILY, 'rb').read().decode('utf-8'))
check('memory_marker', '3156 位置平移族' in open(MEMORY, 'rb').read().decode('utf-8'))

# ---- record disk sha into MEMO anchor line + ledger rev_note (idempotent) ----
disk_note = '3156 disk sha8: result=%s addendum=%s' % (disk_res, sha8(ap))
if disk_note not in mtxt:
    anchor = '（res `c8c66d9f` / seal `b54418f2`）'
    assert mtxt.count(anchor) == 1, 'anchor not unique: %d' % mtxt.count(anchor)
    mtxt2 = mtxt.replace(anchor, anchor + '；disk `%s`' % disk_note)
    open(MEMO, 'wb').write(mtxt2.encode('utf-8'))
    res.append('memo: disk note appended to anchor line')
else:
    res.append('memo: disk note already present')
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
e2 = [m for m in led2['measurements'] if m.get('phase') == 3156][0]
if 'disk result=' not in e2['rev_note']:
    e2['rev_note'] = e2['rev_note'] + ' | disk result=%s addendum=%s' % (disk_res, sha8(ap))
    blob = json.dumps(led2, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led2['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led2, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    res.append('ledger: rev_note disk sha recorded (chain=%s)' % led2['ledger_sha256_8'])
else:
    res.append('ledger: disk sha already present')

res.append('VERIFY TOTAL: PASS=%d FAIL=%d' % (ok, fail))
with io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3156_verify_out.txt'), 'w', encoding='utf-8') as f:
    f.write(chr(10).join(res))
print('written')
