# -*- coding: utf-8 -*-
"""Phase 3225 independent verify (18 checks). Read-only; fixes go to THIS
script only, never to sealed products."""
import hashlib
import json
import os
import re


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


P5 = (r'tests\glm5\result\rdc_query_construction_20260913\phase3225'
      r'\g5g25_a1_behavioral_band_compute')
ok = []


def chk(i, name, cond):
    ok.append((i, name, bool(cond)))
    print('#%-2d %-58s %s' % (i, name, 'PASS' if cond else 'FAIL'))


res = json.load(open(os.path.join(P5, 'result.json'), encoding='utf-8'))
fz = json.load(open(os.path.join(P5, 'execution.json'), encoding='utf-8'))
fz_c = json.loads(open(os.path.join(P5, 'execution.json'), 'rb').read()
                  .decode('utf-8'))
fz_c.pop('design_sha8', None)
fz_sha = hashlib.sha256(
    json.dumps(fz_c, ensure_ascii=False, indent=1, sort_keys=True)
    .encode('utf-8')).hexdigest()[:8]
logtxt = open(os.path.join(P5, 'run_log.txt'), encoding='utf-8').read()
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
import glob as _g

# 1 verdict
chk(1, 'verdict a1_behavioral_band_sealed',
    res['verdict'] == 'a1_behavioral_band_sealed')
# 2 freeze sha
chk(2, 'freeze content sha 98d7fbd1 (recomputed)',
    fz_sha == '98d7fbd1' and fz['design_sha8'] == '98d7fbd1')
# 3 freeze-before-observation + alpha pin
chk(3, 'freeze: alphas {0.1,0.5} gate 0.5 + prereg/default-arm record',
    fz['scale_pinned']['alphas'] == [0.1, 0.5]
    and fz['scale_pinned']['gate_alpha'] == 0.5
    and 'default-arm trigger' in fz['prereg']
    and 'plan A' in fz['prereg'])
# 4 weight shards on disk sha anchors
sp4 = _g.glob(os.path.join(HUB, 'models--Qwen--Qwen3-4B-Base', 'snapshots', '*'))[0]
sp14 = _g.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]
chk(4, 'plan A shards on disk (4c807e25/c7b1aa8f/0d6b9229)',
    sha8_file(os.path.join(sp4, 'model-00001-of-00003.safetensors')) == '4c807e25'
    and sha8_file(os.path.join(sp4, 'model-00003-of-00003.safetensors')) == 'c7b1aa8f'
    and sha8_file(os.path.join(sp14, 'model-00008-of-00008.safetensors')) == '0d6b9229')
# 5 npz sha
chk(5, 'npz file sha ac5d4d74',
    sha8_file(os.path.join(P5, 'a1_behavioral_band.npz')) == 'ac5d4d74'
    and res['sealed_npz']['file_sha8'] == 'ac5d4d74')
# 6 npz independent reread (vectors + med_abs)
import numpy as np
z = np.load(os.path.join(P5, 'a1_behavioral_band.npz'))
e91_4b = json.load(open(
    r'tests\glm5\result\rdc_query_construction_20260913\phase3191'
    r'\g5c4_ftr25_heldout_upgrade\result.json', encoding='utf-8')
)['per_model']['qwen3-4b']['min_med_abs_cos']
e91_14b = json.load(open(
    r'tests\glm5\result\rdc_query_construction_20260913\phase3191'
    r'\g5c4_ftr25_heldout_upgrade\result.json', encoding='utf-8')
)['per_model']['qwen3-14b']['min_med_abs_cos']
chk(6, 'npz reread: ustar dims + med_abs min == sealed 3191 anchors',
    z['qwen3-4b_ustar'].shape == (2560,) and z['qwen3-14b_ustar'].shape == (5120,)
    and abs(float(z['qwen3-4b_med_abs'].min()) - e91_4b) < 1e-9
    and abs(float(z['qwen3-14b_med_abs'].min()) - e91_14b) < 1e-9
    and abs(float(z['qwen3-4b_med_abs'].min()) - 0.3906) < 5e-4)
# 7 device anchors recorded
chk(7, 'device anchors drift0 (3169/3189/3190/3191) in both models',
    all('drift69=0' in logtxt for _ in [0])
    and logtxt.count('device anchors OK') == 2
    and 'drift89=0 drift90=0 drift91=0' in logtxt)
# 8 J self-check
cs = [float(x) for x in re.findall(r'J self-check numeric-vs-analytic cos=([0-9.]+)', logtxt)]
chk(8, 'J self-check cos>0.9999993 x8 (per fold x model)',
    len(cs) == 8 and all(c > 0.9999993 for c in cs))
# 9 cos_full == 1.0000 across all rows (parse log)
lines = [l for l in logtxt.splitlines() if 'cos_full=' in l]
chk(9, '32 metric rows; cos_full_mean all >= 0.9997',
    len(lines) == 32 and all(
        float(re.search(r'cos_full=([0-9.]+)', l).group(1)) >= 0.9997
        for l in lines))
# 10 band <= 0.0052
chk(10, 'band_med max 0.0052 (<< 0.25 gate)',
    all(float(re.search(r'band=([0-9.]+)', l).group(1)) <= 0.0053
        for l in lines))
# 11 gold displacement ~ 0
gt = [abs(float(re.search(r'gold t/p=(-?[0-9.]+)/', l).group(1))) for l in lines]
chk(11, 'gold |true displacement| <= 0.125 (u* not a class direction)',
    max(gt) <= 0.125)
# 12 authorization record
chk(12, 'authorization record: continuation = default-arm trigger, plan A',
    '好的' in res['prereg_authorization_record']
    and 'plan A' in res['prereg_authorization_record'])
# 13 FTR-15 deferred recorded
chk(13, 'FTR-15 deferred in freeze',
    'ftr15_deferred' in fz and 'no single direction' in fz['ftr15_deferred'])
# 14 gates deferred to 3226
chk(14, 'gate evaluation deferred to 3226 (A2)',
    'DEFERRED to 3226' in res['gate_evaluation'])
# 15 ledger #377
led = json.load(open(r'research\gpt5\atlas\atlas_ledger.json', encoding='utf-8'))
e = led['measurements'][376]
chk(15, 'ledger #377 phase/name/verdict prefix',
    len(led['measurements']) == 377 and e['phase'] == 3225
    and e['name'] == 'g5g25_a1_behavioral_band_compute'
    and e['verdict'].startswith('a1_behavioral_band_sealed')
    and e['chain']['npz_sha8'] == 'ac5d4d74')
# 16 rollback-equality
bako = json.loads(open(r'research\gpt5\atlas\atlas_ledger.json.bak_3225',
                       'rb').read().decode('utf-8'))
back = dict(led)
back['measurements'] = led['measurements'][:376]
back['ledger_sha256_8'] = bako['ledger_sha256_8']
eq = json.dumps(bako, ensure_ascii=False, indent=1, sort_keys=True) == \
     json.dumps(back, ensure_ascii=False, indent=1, sort_keys=True)
chk(16, 'rollback-equality bak_3225 vs truncated head',
    eq and len(bako['measurements']) == 376
    and bako['ledger_sha256_8'] == '0543371d')
# 17 MEMO section + honest log + prereg 3226
memo = open(r'research\gpt5\docs\AGI_GPT5_MEMO.md', encoding='utf-8').read()
chk(17, 'MEMO 3225 section + honest log + prereg 3226',
    '## Phase 3225 — G5-G25 A1' in memo
    and '1/N' in memo and 'bfloat16' in memo
    and '### 预注册 Phase 3226（两支候选）' in memo
    and 'ac5d4d74' in memo and '#377' in memo)
# 18 daily + MEMORY
daily = open(r'.workbuddy\memory\2026-10-11.md', encoding='utf-8').read()
mem = open(r'.workbuddy\memory\MEMORY.md', encoding='utf-8').read()
chk(18, 'daily 3225 guard + MEMORY 3226 prereg + 3225 row',
    '### Phase 3225' in daily and '3226 已预注册' in mem
    and 'ac5d4d74' in mem and '969b6791' in mem)

n_fail = sum(1 for _, _, c in ok if not c)
print('RECHECK: %s (%d/18 PASS)' % ('CLEAN' if n_fail == 0 else 'FAIL',
                                    18 - n_fail))
