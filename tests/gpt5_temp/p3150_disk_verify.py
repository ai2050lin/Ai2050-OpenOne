# -*- coding: utf-8 -*-
"""Phase 3150 independent disk verify."""

import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
ok = 0
fail = []


def chk(name, cond):
    global ok
    if cond:
        ok += 1
    else:
        fail.append(name)


# 1) result.json + sha + verdict
p = os.path.join(ROOT, 'tests', 'glm5', 'result',
                 'rdc_query_construction_20260913',
                 'phase3150',
                 'p0_freeze_carrierdecode',
                 'result.json')
chk('result exists', os.path.exists(p))
raw = io.open(p, 'rb').read()
chk('result sha8 a2158ea5',
    hashlib.sha256(raw).hexdigest()[:8]
    == 'a2158ea5')
r = json.loads(raw.decode('utf-8'))
chk('seal 1c6eb2f9',
    r.get('seal_sha8') == '1c6eb2f9')
chk('verdict 5 tags', r['verdict'].count('|') == 6
    and 'carrier_skeleton_class' in r['verdict'])
chk('part_d rows 25', len(r['part_d']['rows']) == 25)
chk('part_e 2 levels',
    len(r['part_e']['contours']) == 2)
chk('part_f alpha* in range',
    0.10 <= r['part_f']['alpha_star'] <= 0.20)
chk('part_g margin', r['part_g']['margin_x'] >= 4.0)
chk('npz exists', os.path.exists(
    os.path.join(os.path.dirname(p),
                 'carrier_tokens.npz')))
chk('run_log GATES', all(
    g in io.open(os.path.join(
        os.path.dirname(p), 'run_log.txt'),
        encoding='utf-8').read()
    for g in ['P-GATE: PASS', 'D-GATE: PASS',
              'E-GATE: PASS', 'F-GATE: PASS',
              'G-GATE: PASS']))
chk('execution frozen', os.path.exists(
    os.path.join(os.path.dirname(p),
                 'execution.json')))

# 2) ledger n=287 + 3150 entry
led = json.load(io.open(os.path.join(
    ROOT, 'research', 'gpt5', 'atlas',
    'atlas_ledger.json'), encoding='utf-8'))
chk('ledger n=287',
    len(led['measurements']) == 287)
m = led['measurements'][-1]
chk('ledger 3150 entry', m['phase'] == 3150
    and m['seal_sha8'] == '1c6eb2f9'
    and m['evidence_level'] == 'statistical')
chk('ledger schema v3',
    led.get('schema_version') == 3)
chk('ledger v3 backfilled',
    all('evidence_level' in mm
        for mm in led['measurements']))

# 3) metric_dict
md = json.load(io.open(os.path.join(
    ROOT, 'research', 'gpt5', 'atlas',
    'metric_dict.json'), encoding='utf-8'))
chk('metric_dict 7 metrics',
    len(md['metrics']) == 7)
chk('metric_dict meta F2+F7',
    'F2_verdict_grading' in md['meta_rules']
    and 'F7_bit_anchor_status' in md['meta_rules'])

# 4) docs on disk
for rel, needles in [
    (r'research\gpt5\docs\PARADIGM_SHIFT_VERDICT_v1.md',
     ['总裁决', 'K-G4', 'K-G5', 'carrier_skeleton_class',
      '逐条判定表']),
    (r'research\gpt5\docs\FIRST_PRINCIPLES_3090_3149.md',
     ['Phase 3090', 'Phase 3102', 'Phase 3149']),
    (r'tests\glm5\gate_precheck.py',
     ['def precheck', 'def mde_binomial']),
    (r'tests\glm5\counterexample_grep.py',
     ['def grep_claim'])]:
    fp = os.path.join(ROOT, rel)
    t = io.open(fp, encoding='utf-8').read()
    for nd in needles:
        chk('%s:%s' % (os.path.basename(rel), nd),
            nd in t)

# 5) MEMO 3150 section
memo = io.open(os.path.join(
    ROOT, 'research', 'gpt5', 'docs',
    'AGI_GPT5_MEMO.md'), encoding='utf-8').read()
chk('MEMO 3150 title',
    '## Phase 3150: P0制度冻结+载体解码' in memo)
chk('MEMO 3150 verdict', 'kx_product_superlinear'
    in memo and 'carrier_skeleton_class' in memo)
chk('MEMO 3151 prereg',
    '## 预注册 Phase 3151' in memo
    or '预注册 Phase 3151' in memo)

# 6) daily + MEMORY
daily = io.open(os.path.join(
    ROOT, '.workbuddy', 'memory', '2026-10-01.md'),
    encoding='utf-8').read()
chk('daily 3150', 'Phase 3150 (gpt5 线)' in daily)
mem = io.open(os.path.join(
    ROOT, '.workbuddy', 'memory', 'MEMORY.md'),
    encoding='utf-8').read()
chk('MEMORY 3150 line', '3150：P0 完成' in mem)
chk('MEMORY size ok', len(mem) < 3400)

print('VERIFY %d/%d PASS' % (ok, ok + len(fail)))
for f in fail:
    print('FAIL:', f)
