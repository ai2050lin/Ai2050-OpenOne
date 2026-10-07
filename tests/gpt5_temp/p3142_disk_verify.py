# -*- coding: utf-8 -*-
"""Phase 3142 independent disk verify."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D42 = os.path.join(
    RDIR, 'phase3142',
    'omega_p140_l19anat_blockmat_'
    'quartile_multitpl')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = ROOT + r'\.workbuddy\memory' \
       r'\2026-09-29.md'
WMEM = ROOT + r'\.workbuddy\memory' \
       r'\MEMORY.md'

P = F = 0


def ck(name, cond):
    global P, F
    if cond:
        P += 1
    else:
        F += 1
    print('%s %s' % ('PASS' if cond
                     else 'FAIL', name))


# artifacts
for f in ('result.json',
          'design_seal.json',
          'p140_readout.npz',
          'run_log.txt'):
    ck('artifact %s' % f,
       os.path.exists(os.path.join(D42, f)))
raw = io.open(os.path.join(D42,
                           'result.json'),
              'rb').read()
ck('result sha8 == 7291cfc5',
   hashlib.sha256(raw).hexdigest()[:8]
   == '7291cfc5')
res = json.loads(raw.decode('utf-8'))
ck('verdict match',
   res['verdict'].startswith('a_3141_ok')
   and 'repro_bit_9' in res['verdict']
   and 'write_structural_absent'
   in res['verdict'])
ck('seal_sha8 == 449f8161',
   str(res['seal_sha8']) == '449f8161')
ck('smoke false', res['smoke'] is False)
ck('xphase P=1.0',
   res['part_c']['xphase_P'] == 1.0)
ck('xphase A1=1.0',
   res['part_c']['xphase_A1'] == 1.0)
ck('dvec19 sha8',
   res['part_c']['dvec19']['sha8']
   == '6a0332a6')
ck('dvec19 n=672',
   int(res['part_c']['dvec19']['n']) == 672)
ck('dvec19 all_d1 0.7890625',
   abs(res['part_c']['dvec19_trials']
       ['all_d1'] - 0.7890625) < 1e-9)
ck('s1 d4 L21 0.3671875',
   abs(res['part_c']['step1_scan']['4.0']
       ['21'] - 0.3671875) < 1e-9)
ck('all d2 L19 0.4921875',
   abs(res['part_c']['allstep_ctrl']
       ['19'] - 0.4921875) < 1e-9)
ck('C bit 2/2',
   all(v['match'] for v in
       res['part_c']
       ['bit_anchors_3141'].values())
   and len(res['part_c']
           ['bit_anchors_3141']) == 2)
ck('crosslayer 0.1953125',
   abs(res['part_d']
       ['crosslayer_joint']
       - 0.1953125) < 1e-9)
ck('rev delta 0',
   res['part_d']['rev_delta'] == 0.0)
ck('BI(2.0,0.5) 0.698',
   abs(res['part_d']['matrix']
       ['2.0_0.5']['BI']
       - 0.6981132075471698) < 1e-9)
ck('D bit 4/4',
   all(v['match'] for v in
       res['part_d']
       ['bit_anchors_3141'].values())
   and len(res['part_d']
           ['bit_anchors_3141']) == 4)
ck('E bit 3/3',
   all(v['match'] for v in
       res['part_e']
       ['bit_anchors_3137'].values())
   and len(res['part_e']
           ['bit_anchors_3137']) == 3)
ck('E d1.0 Q1 0.15625',
   abs(res['part_e']['q_chg']['1.0'][0]
       - 0.15625) < 1e-9)
ck('F f2 == 3141 gen anchor',
   abs(res['part_f']['retr_f2_gen_v1']
       ['17'] - 0.047619047619047616)
   < 1e-9)
ck('F f3 quad all 0.0357',
   all(abs(v - 0.03571428571428571)
       < 1e-9 for v in
       res['part_f']
       ['retr_f3_quad'].values()))
ck('F tag structural_absent',
   res['part_f']['write_tag']
   == 'write_structural_absent')
# ledger
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
ck('ledger n=279',
   len(led['measurements']) == 279)
last = led['measurements'][-1]
ck('ledger last phase 3142',
   last.get('phase') == 3142)
ck('ledger hashes',
   last['hashes']['result_sha256_8']
   == '7291cfc5'
   and last['hashes']['seal_sha256_8']
   == '449f8161')
# MEMO
t = io.open(MEMO, encoding='utf-8').read()
ck('MEMO 3142 heading x1',
   t.count('## Phase 3142: L19 载体解剖')
   == 1)
ck('MEMO 3143 prereg',
   '3143（Ω-P141）预注册' in t)
ck('MEMO dvec19 finding',
   '0.7891' in t)
ck('MEMO anchor sha',
   '7291cfc5' in t)
# wlog
tw = io.open(WLOG, encoding='utf-8').read()
ck('wlog 3142 closeout',
   'Phase 3142 (Omega-P140) closeout'
   in tw)
ck('wlog 3142 exec',
   'Phase 3142 (Omega-P140) 执行中' in tw)
# MEMORY.md
tm = io.open(WMEM, encoding='utf-8').read()
ck('MEMORY max=3142',
   '- max=3142，下一 3143' in tm)
ck('MEMORY no max=3141',
   '- max=3141，下一 3142' not in tm)
ck('MEMORY 3142 findings',
   '写入侵入' in tm or '写入免疫' in tm)
print('TOTAL %d PASS / %d FAIL'
      % (P, F))
with io.open(
        ROOT + r'\tests\gpt5_temp'
        r'\p3142_disk_verify_report.txt',
        'w', encoding='utf-8') as f:
    f.write('3142 disk verify: '
            '%d PASS / %d FAIL\n' % (P, F))
