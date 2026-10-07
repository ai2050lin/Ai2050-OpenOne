# -*- coding: utf-8 -*-
"""Q06 closeout：phase_queue Q06 -> sealed + atlas_ledger append phase40 + 回读复核。"""
import json, hashlib, io, os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
Q_P = os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'phase_queue_v1.json')
L_P = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
R_P = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q06_result.json')

res = json.load(open(R_P, encoding='utf-8'))
res_sha8 = res['res_sha8']
cs = res['C_steer_main']
sn = res['sensitivity']
co = res['collateral']

# ---------- 1. queue ----------
q = json.load(open(Q_P, encoding='utf-8'))
q06 = [it for it in q['queue'] if it['id'] == 'Q06'][0]
assert q06['status'] == 'contract_frozen', q06['status']
q06['status'] = 'sealed'
q06['sealed_at'] = '2026-10-07 08:05'
q06['seal_record'] = 'tests/deepseek/result/q06_result.json'
q06['res_sha8'] = res_sha8
q06['note'] = ('C_steer 基座测量完成（annex v2，design 7130906b）：C_steer_main=0.0000 (0/376 eligible, '
               '10 配置全 0)；rand 同规则对照 0.0000，spec_diff=0.0；Wilson95 上界 0.0101；'
               '灵敏度 argmax moved 9/4410、maxd<=0.94 logit；collateral 干净（mean 0.008/max 1/frac0 0.933）；'
               'identity 两 prompt 逐位恒等 0.00e+00；独立复核 16 PASS/0 FAIL。'
               'I1：E_read/E_ar 引用未变 + C_steer 新增读数 => Ledger 登记 catalog。')
q['status_updated_at'] = '2026-10-07 08:05'
q['status_updated_by'] = 'tests/deepseek/Phase40/closeout_q06.py'
if 'Q06' not in q['sealed_items']:
    q['sealed_items'].append('Q06')
with open(Q_P, 'w', encoding='utf-8') as f:
    json.dump(q, f, ensure_ascii=False, indent=1)

# ---------- 2. ledger ----------
led = json.load(open(L_P, encoding='utf-8'))
ms = led['measurements']
assert all(m.get('phase') != 40 for m in ms), 'phase40 already present'
entry = {
    'phase': 40,
    'name': 'q06_c_steer_base_readout_substitution_v1_axis',
    'seal_sha8': '7130906b',
    'exec_sha8': '7130906b',
    'result_sha8': res_sha8,
    'evidence_level': 'statistical',
    'model_scope': 'qwen3-4b (bf16)',
    'n_rows': 441,
    'prereg_id': 'Q06',
    'superseded_by': None,
    'verdict': ('q06_c_steer_base_zero|cs_main_0.0000_0/376|wilson_hi_0.0101|spec_diff_0.0|'
                'collat_clean_frac0_0.933|argmax_moved_9_of_4410|maxd_le_0.94logit|'
                'identity_bit_0.00e+00|verify_16_0'),
    'rev_note': ('C_steer first measurement (I7 base): v1 load-bearing axis (L29 WR-main-PC, '
                 'panel interaction-residual PC1 isomorph-ported def; prereg ebf960cf) + readout '
                 'substitution h+(t-s)v, dt=sgn*alpha*sigma alpha in {0.05..0.5} (prereg 5-pt grid); '
                 'targets true_class on E_read heldout panel (376 eligible of 441; S1 seeds 7/8/9). '
                 'ZERO steer success across all 10 configs; rand-direction control identical (0.0); '
                 'collateral clean -> zero is NOT collateral-dirt but missing directional leverage '
                 '(v1 = global form/magnitude axis, not class-identity lever; maxd<=0.94 logit vs '
                 'prior-class head gap). Asymmetry vs M14: clip(removal) breaks 66% generation @a0.5 '
                 'while push/pull substitution moves readout <1 logit -> load-bearing axis carries '
                 'generation stability, not class content. I1 => catalog (no KPI decreased). '
                 'annex v2 revise log: t-rule/target/G2-gate fixed after SMOKE v1 (design 329e0115->6d98d580 smoke, 7130906b formal).'),
}
ms.append(entry)
with open(L_P, 'w', encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False, indent=1)

# ---------- 3. 回读复核 ----------
q2 = json.load(open(Q_P, encoding='utf-8'))
q062 = [it for it in q2['queue'] if it['id'] == 'Q06'][0]
l2 = json.load(open(L_P, encoding='utf-8'))
ok_q = q062['status'] == 'sealed' and 'Q06' in q2['sealed_items'] and q062['res_sha8'] == res_sha8
ok_l = l2['measurements'][-1]['phase'] == 40 and l2['measurements'][-1]['result_sha8'] == res_sha8
print('queue sealed:', ok_q, '| ledger n=%d phase40:' % len(l2['measurements']), ok_l)
print('res_sha8:', res_sha8)
assert ok_q and ok_l
print('CLOSEOUT FILES OK')
