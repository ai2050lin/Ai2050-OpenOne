# -*- coding: utf-8 -*-
import io, json, os, hashlib
import numpy as np
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3138\omega_p136_statebank_'
       r'bicdecomp_probecontrast')
out = []
ok = 0
fail = 0
def chk(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        out.append('PASS %s %s' % (name, detail))
    else:
        fail += 1
        out.append('FAIL %s %s' % (name, detail))

# 1 result + seal + npz + 8 bank shards
for f in ('result.json', 'design_seal.json',
          'run_log.txt', 'p136_readout.npz'):
    chk('file:' + f, os.path.exists(os.path.join(OUT, f)))
for d in ('P', 'A1'):
    for t in range(4):
        f = 'bank_%s_T%d.npz' % (d, t)
        p = os.path.join(OUT, f)
        ex = os.path.exists(p)
        chk('file:' + f, ex)
        if ex:
            z = np.load(p, allow_pickle=False)
            chk('shard:' + f,
                'scale' in z.files
                and z['states'].shape == (672, 40, 4096)
                and bool(np.isfinite(z['states'].astype(np.float32)).all()),
                str(z['states'].shape))
raw = io.open(OUT + r'\result.json', 'rb').read()
chk('result_sha8', hashlib.sha256(raw).hexdigest()[:8] == 'f7ef08be')
r = json.loads(raw.decode('utf-8'))
chk('phase', r['phase'] == 3138)
chk('smoke_false', r['smoke'] is False)
chk('verdict_tags', r['verdict'].count('|') == 7)
z = np.load(OUT + r'\p136_readout.npz', allow_pickle=False)
chk('npz_keys_15', len(z.files) == 15, str(len(z.files)))
chk('retr_L17', abs(float(z['retr'][17]) - 1.0) < 1e-9)
chk('retr_L29', abs(float(z['retr'][29]) - 0.7842) < 1e-3)
chk('auc_raw_x_L17', float(z['auc_raw_x'][17]) == 1.0)

# 2 ledger
led = json.load(io.open(ROOT + r'\research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
e38 = [e for e in led['measurements'] if e.get('phase') == 3138]
chk('ledger_n275', len(led['measurements']) == 275, str(len(led['measurements'])))
chk('ledger_3138_entry', len(e38) == 1)
if e38:
    chk('ledger_sha', e38[0]['hashes']['result_sha256_8'] == 'f7ef08be')
    chk('ledger_verdict', e38[0]['verdict'] == r['verdict'])

# 3 MEMO
m = io.open(ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
i = m.rfind('## Phase 3138:')
chk('memo_section', i > 0)
if i > 0:
    sec = m[i:]
    chk('memo_T4_21', 'T4 第21 Phase' in sec[:80])
    for frag in ('范数占比 ≠ 因果占比', '身份重写窗口', 'Tier-1 Level-1',
                 'f7ef08be', '0.784', '0.9996', '3139 预注册',
                 'L26–32', 'rev-3138a'):
        chk('memo_has:' + frag[:14], frag in sec)
    chk('memo_x3_1', sec.count('范数占比 ≠ 因果占比') >= 3)
    chk('memo_x3_2', sec.count('身份重写窗口定位') >= 3)
    chk('memo_x3_3', sec.count('跨模板泛化 AUC 1.0') >= 3)

# 4 wlog
wl = io.open(ROOT + r'\.workbuddy\memory\2026-09-28.md', encoding='utf-8').read()
chk('wlog_3138_closed', 'Phase 3138 (Ω-P136) 闭环' in wl)

# 5 workspace MEMORY
mm = io.open(ROOT + r'\.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk('wmem_3138_line', '3138（T4）' in mm)
chk('wmem_max3138', 'max=3138，下一 3139' in mm)
chk('wmem_next_updated', '端口消费测试' in mm)
chk('wmem_old_next_gone', '3138 起转 Absolute State Bank' not in mm)

out.append('SUMMARY ok=%d fail=%d' % (ok, fail))
io.open(ROOT + r'\tests\gpt5_temp\p3138_disk_verify.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
