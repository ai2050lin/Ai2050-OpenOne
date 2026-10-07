# -*- coding: utf-8 -*-
"""
Phase 21 closeout（幂等）：
  1) 读 result_phase21.json，导出三级标签 judgement_phase21.json
  2) 备份 + 补登 Ledger（n 303 -> 304）
  3) 冻结 MEMO 追加前基线 memo_baseline_preappend_phase21.json
  4) 写 closeout_phase21.txt
用法：python tests/deepseek/Phase21/closeout_phase21.py
"""
import io
import os
import json
import time
import shutil
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
EXECP = os.path.join(P21T, 'execution_phase21.json')
SEALP = os.path.join(P21T, 'N2h1a14_design_seal.json')
RESULTP = os.path.join(P21T, 'result_phase21.json')
JUDGEP = os.path.join(P21T, 'judgement_phase21.json')
BASEP = os.path.join(P21T, 'memo_baseline_preappend_phase21.json')
LOGP = os.path.join(P21T, 'closeout_phase21.txt')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return sha(p)[:8]


out = []


def w(s=''):
    out.append(str(s))
    print(s)


EX = json.load(io.open(EXECP, encoding='utf-8'))
assert sha(SEALP) == EX['seal_sha256'], 'DRIFT seal'
R = json.load(io.open(RESULTP, encoding='utf-8'))
P = R['predictions']
pairs = R['quant_pairs']
cal = R['calibration']

labels = []
lab_map = [('P1_calib_A0bf16_reproduces_P8', 'calib_p8_ok'),
           ('P2_share_v_mlp_stable', 'share_v_stable'),
           ('P3_max_head_share_v_stable', 'maxv_stable'),
           ('P4_argmax_head_v_same', 'argmax_v_same'),
           ('P5_G1_core_both_precisions', 'g1_core_both'),
           ('P6_W_stable', 'w_stable'),
           ('P7_spearman_share_v', 'rho_share_v_ok'),
           ('P8_conf_same_band', 'conf_same_band'),
           ('P9_floors', 'floors_ok')]
for k, lab in lab_map:
    if P.get(k) is True:
        labels.append(lab)
verdict = '__'.join(labels) if labels else 'none'

n_rows = 0
n_fw = {}
for a in R['arm_order']:
    rec = R['arms'][a]
    n_fw[a] = int(rec.get('n_fw', 0))
    n_rows += int(rec['M1']['n_pairs']) + int(rec.get('confirmation', {}).get('n', 0) or 0)

JUDGE = dict(
    phase=21, line='N2h1-alpha-14', kind='judgement',
    created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
    seal_sha8=sha8(SEALP), exec_sha8=sha8(EXECP), result_sha8=sha8(RESULTP),
    bit_anchored=dict(
        calib_A0_bf16_vs_P8=cal,
        p8_anchor=R['p8_anchor'],
        determinism=[R['arms'][a]['E0_selfcheck']['determinism_maxdiff'] for a in R['arm_order']],
        F4_ok=[R['arms'][a]['F4_dims_ok'] for a in R['arm_order']],
        F5_ok=[R['arms'][a]['F5_o_proj_ok'] for a in R['arm_order']],
    ),
    statistical=dict(
        quant_pairs=pairs,
        predictions=P,
        n_pass=R['n_pass'], n_total=R['n_total'],
    ),
    descriptive=dict(
        per_arm_M1={a: dict(share_v_mlp=R['arms'][a]['M1']['share_v_mlp'],
                            max_head_share_v=R['arms'][a]['M1']['max_head_share_v'],
                            argmax_head_v=R['arms'][a]['M1']['argmax_head_v'],
                            loo_vec_top1=R['arms'][a]['M1']['loo_vec_top1']) for a in R['arm_order']},
        per_arm_M2={a: dict(max_head_share=R['arms'][a].get('M2', {}).get('max_head_share'),
                            argmax_head=R['arms'][a].get('M2', {}).get('argmax_head'),
                            mlp_share_vs_attn=R['arms'][a].get('M2', {}).get('mlp_share_vs_attn'))
                    for a in R['arm_order']},
        per_arm_G1={a: dict(G1_core=R['arms'][a]['G1_core'], G1_full=R['arms'][a]['G1_full'])
                    for a in R['arm_order']},
    ),
    verdict=verdict,
    n_rows=n_rows, n_forwards_per_arm=n_fw,
)
with io.open(JUDGEP, 'w', encoding='utf-8', newline='\n') as f:
    f.write(json.dumps(JUDGE, ensure_ascii=False, indent=1))
w('judgement -> %s (%d B)' % (JUDGEP, os.path.getsize(JUDGEP)))
w('verdict = %s' % verdict)
w('n_rows=%d n_fw=%s' % (n_rows, json.dumps(n_fw)))

# ---------------- Ledger ----------------
shutil.copy2(LEDGER, os.path.join(P21T, 'atlas_ledger_backup_pre_phase21.json'))
L = json.load(io.open(LEDGER, encoding='utf-8'))
m = L['measurements']
alread = [x for x in m if x.get('phase') == 21]
if alread:
    w('Ledger 已含 phase 21，跳过追加（幂等）')
else:
    entry = dict(
        phase=21,
        name='n2h1a14_component_vector_budget_and_weight_capacity_quant_scheme_robustness_qwen3_4b_glm4_9b',
        seal_sha8=sha8(SEALP), exec_sha8=sha8(EXECP), result_sha8=sha8(RESULTP),
        probe_sha8=None,
        evidence_level='statistical',
        model_scope='qwen3-4b + glm4-9b-chat-hf (nf4 vs bf16); Qwen3-14B excluded',
        n_rows=int(n_rows),
        n_forwards_per_arm=n_fw,
        prereg_id='N2h1a14',
        superseded_by=None,
        verdict=verdict,
        rev_note=('deepseek/N line Phase 21 (N2h1-alpha-14), PRECISION ROBUSTNESS OF THE COMPONENT-LEVEL '
                  'VECTOR BUDGET AND THE WEIGHT-LEVEL CAPACITY: Phase 8 (N2h1-alpha) computed the write-operator '
                  'component budget at L6 under **bf16** (first metric share_v: MLP %.4f, max single head %s %.4f, '
                  'second metric efficiency share %.4f, I_nl %.3f; weight capacity W: max_head_share %.4f at head #%d) '
                  'while Phases 16/17/18 generalised that budget entirely under **nf4**. Phase 19 closed the vector '
                  'side and Phase 20 the behavioural/profile side; the two ORIGINAL quantities of the Phase-8 line '
                  'were never recomputed under the other precision. Phase 21 recomputed them in the SAME apparatus '
                  'over {nf4, bf16} x {qwen3-4b, glm4-9b}. A0_bf16 reproduces the Phase-8 frozen anchors bit-for-bit '
                  '(calibration), and the nf4/bf16 pair per model gives the cross-precision deltas. Verdict flags: %s.'
                  ) % (R['p8_anchor']['share_v_mlp'], R['p8_anchor']['argmax_head_v'],
                       R['p8_anchor']['max_head_share_v'], R['p8_anchor']['max_head_share_eff'],
                       R['p8_anchor']['I_nl'], R['p8_anchor']['W_max_head_share'],
                       R['p8_anchor']['W_argmax_head'], verdict),
        created=time.strftime('%Y-%m-%d'),
        amend1_sha8=None,
        anchor_result_sha8=sha8(os.path.join(ROOT, EX['anchor_p8_result_path'])),
    )
    m.append(entry)
    with io.open(LEDGER, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(L, ensure_ascii=False, indent=1))
    w('Ledger appended phase 21: n %d -> %d' % (len(m) - 1, len(m)))
w('ledger sha8 now %s' % sha8(LEDGER))

# ---------------- MEMO 基线（追加前）----------------
if os.path.exists(BASEP):
    w('baseline 已存在，跳过（幂等）')
else:
    mb = open(MEMO, 'rb').read()
    txt = mb.decode('utf-8-sig')
    heads = {}
    for i, ln in enumerate(txt.split('\r\n'), 1):
        if ln.startswith('## Phase '):
            heads[ln.rstrip()] = i
    base = dict(path=os.path.relpath(MEMO, ROOT).replace('/', '\\'),
                bytes=len(mb), sha256=hashlib.sha256(mb).hexdigest(),
                sha8=hashlib.sha256(mb).hexdigest()[:8],
                lines=txt.count('\r\n') + 1,
                phase_headings=len(heads),
                tag='post-append-phase20',
                sections=heads, sections_key_rule='完整标题行（内射，含碰撞断言）',
                frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                note='Phase 21 追加前基线')
    with io.open(BASEP, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(base, ensure_ascii=False, indent=1))
    w('MEMO baseline frozen: %d B / %s / %d headings' % (base['bytes'], base['sha256'][:8], len(heads)))

with io.open(LOGP, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(out) + '\n')
print('CLOSEOUT DONE')
