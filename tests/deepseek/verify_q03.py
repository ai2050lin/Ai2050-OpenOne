# -*- coding: utf-8 -*-
"""Q03 独立复核（新进程；不复用复算脚本的任何内存对象）"""
import os, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
B3152 = os.path.join(RDIR, 'phase3152', 'g1p2_tri_model_k1')
B3151 = os.path.join(RDIR, 'phase3151', 'g1p1_combo_additive_vs_interaction')
DKR = os.path.join(ROOT, 'tests', 'deepseek', 'result')
OUT = os.path.join(DKR, 'verify_q03.txt')
rows = []
P = F = 0

def chk(name, ok, detail=''):
    global P, F
    if ok:
        P += 1
    else:
        F += 1
    rows.append('  [%s] %-58s %s' % ('PASS' if ok else 'FAIL', name, detail))

def sha8b(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def J(p):
    return json.loads(open(p, 'rb').read().decode('utf-8-sig'))

rows.append('Q03 独立复核（独立进程，重新读盘）')
rows.append('=' * 78)

# 1) carriers 未变
rows.append('1) held-out carriers（sha8 须与 metric_dict 冻结值一致）')
carr = [('qwen3-4b', os.path.join(B3152, 'qwen3-4b', 'collect.npz'), '4ef190b7'),
        ('qwen3-14b', os.path.join(B3152, 'qwen3-14b', 'collect.npz'), '733182c1'),
        ('glm4-9b', os.path.join(B3151, 'collect.npz'), 'c711946c')]
for n, p, e in carr:
    g = sha8b(p)
    chk('carrier %s sha8' % n, g == e, '%s == %s' % (g, e))

# 2) q03_result 与三方源 result.json 独立比对
rows.append('2) q03_result 的复算值 vs 3151/3152 原始 result.json')
q = J(os.path.join(DKR, 'q03_result.json'))
srcs = {'qwen3-4b': os.path.join(B3152, 'qwen3-4b', 'result.json'),
        'qwen3-14b': os.path.join(B3152, 'qwen3-14b', 'result.json'),
        'glm4-9b': os.path.join(B3152, 'glm4k1', 'result.json')}
for n, sp in srcs.items():
    rp = J(sp)['k1_model_report']
    rc = q['per_model'][n]['b4_rel_readout_per_seed_recompute']
    ra = rp['b4_rel_readout_per_seed']
    chk('%s per-seed == source' % n,
        len(rc) == 3 and max(abs(a - b) for a, b in zip(rc, ra)) < 1e-9,
        'max|d|=%.2e' % max(abs(a - b) for a, b in zip(rc, ra)))
    chk('%s mean == source' % n,
        abs(q['per_model'][n]['b4_rel_readout_mean3seed_recompute'] - rp['b4_rel_readout_mean3seed']) < 1e-9,
        '%.6f' % rp['b4_rel_readout_mean3seed'])
    chk('%s readout layer matches source' % n,
        q['per_model'][n]['readout_layer'] == rp['readout'],
        'layer=%d' % rp['readout'])

# 3) 池化 E_read vs metric_dict 冻结
rows.append('3) E_read 池化 vs metric_dict.json（I1 真源）')
md = J(os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'metric_dict.json'))
E_md = None
er = md.get('global_kpis', {}).get('E_read', {})

def deep_find(o, keys):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in keys:
                return v
            r = deep_find(v, keys)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = deep_find(v, keys)
            if r is not None:
                return r
    return None

E_md = deep_find(er, {'E_read_pooled', 'pooled_mean', 'E_read_mean', 'pooled'})
rows.append('     [INFO] metric_dict 未含显式池化字段（仅定义+gate 在册）；按三份 result 独立重算')
if E_md is None:
    vals = [J(sp)['k1_model_report']['b4_rel_readout_mean3seed'] for sp in srcs.values()]
    E_md = float(np.mean(vals))
    rows.append('     （metric_dict 无显式池化字段，独立从三份 result 重算）')
chk('池化值一致', abs(float(E_md) - q['summary']['pooled_mean']) < 1e-9,
    'md=%.6f vs q03=%.6f' % (float(E_md), q['summary']['pooled_mean']))
chk('三模型 E_read 顺序值一致',
    all(abs(q['summary']['E_read'][n] - J(sp)['k1_model_report']['b4_rel_readout_mean3seed']) < 1e-9
        for n, sp in srcs.items()),
    str({k: round(v, 6) for k, v in q['summary']['E_read'].items()}))

# 4) gate 判定
rows.append('4) 5% 门判定')
chk('全部模型未过门 (0/3)', q['summary']['gate_pass_models'] == 0, 'pass=%d/3' % q['summary']['gate_pass_models'])
chk('最小 E_read = 门的 6.63x', abs(q['summary']['min_E_x'] - 6.63) < 0.01,
    'min=%.6f  %.2fx' % (q['summary']['min_E'], q['summary']['min_E_x']))
chk('阈值 = 0.05', abs(q['summary']['gate_threshold'] - 0.05) < 1e-12, str(q['summary']['gate_threshold']))

# 5) design_sha 自洽（去掉 design_sha 键后重算）
rows.append('5) 预注册 design_sha 自洽性')
ex = J(os.path.join(DKR, 'q03_execution.json'))
d2 = {k: v for k, v in ex.items() if k != 'design_sha'}
blob = json.dumps(d2, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
chk('design_sha 重算一致', hashlib.sha256(blob).hexdigest() == ex['design_sha'],
    ex['design_sha'][:8])

# 6) bootstrap CI 合理性
rows.append('6) bootstrap 95% CI 合理性（mean ∈ CI，宽度>0）')
for n, v in q['per_model'].items():
    for s, b in v['bootstrap_per_seed'].items():
        ok = b['ci_lo'] <= b['mean'] <= b['ci_hi'] and b['ci_hi'] > b['ci_lo']
        chk('%s seed%s CI 覆盖点估计' % (n, s), ok,
            '[%.4f, %.4f] contains %.4f' % (b['ci_lo'], b['ci_hi'], b['mean']))
chk('n_boot=10000', q['bootstrap']['n_boot'] == 10000, str(q['bootstrap']['n_boot']))

# 7) 统一 held-out 指纹
rows.append('7) 统一 held-out 指纹（三模型共用）')
fp = q['fingerprint']
chk('3 seed 各 49 test pairs', all(v['n_test_pairs'] == 49 for v in fp.values()),
    str({k: v['n_test_pairs'] for k, v in fp.items()}))
chk('3 seed 各 197 train pairs', all(v['n_train_pairs'] == 197 for v in fp.values()),
    str({k: v['n_train_pairs'] for k, v in fp.items()}))
chk('test fold sha8 三 seed 互异', len(set(v['test_pairs_sha8'] for v in fp.values())) == 3,
    str([v['test_pairs_sha8'] for v in fp.values()]))

# 8) 受保护文件未变
rows.append('8) 受保护（跨线/冻结）文件指纹未变')
prot = [
    ('AGI_GPT5_MEMO.md', os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md'), '2a84776b'),
    ('atlas_ledger.json', os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json'), 'bbda63df'),
    ('proposition_ledger.json',
     os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                  'phase3103', 'omega_p101_formula_audit', 'proposition_ledger.json'), '9a3c6ff4'),
    ('RDC_TESTPLAN_v1.md', os.path.join(ROOT, 'tests', 'deepseek_temp', '_archive_r6', 'gpt5_docs', 'RDC_TESTPLAN_v1.md'), '71b85673'),
    ('metric_dict.json', os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'metric_dict.json'), '03887e51'),
    ('phase_queue_v1.json', os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'phase_queue_v1.json'), '37da9a8d'),
    ('phase3152 script', os.path.join(ROOT, 'tests', 'glm5', 'phase3152_g1p2_tri_model_k1.py'), 'd795f41c'),
    ('phase3151 script', os.path.join(ROOT, 'tests', 'glm5', 'phase3151_g1p1_combo_additive_vs_interaction.py'), '292ed9f3'),
]
for lbl, p, e in prot:
    if not os.path.exists(p):
        chk('%s sha8' % lbl, False, 'FILE MISSING (post-R6 relocation?) %s' % p)
        continue
    g = sha8b(p)
    chk('%s sha8' % lbl, g == e, '%s == %s' % (g, e))

# 9) 产物指纹
rows.append('8b) 研究日志现状（并发写者存在；仅记录，不判 FAIL）')
_mm = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
_mraw = open(_mm, 'rb').read()
_mt = _mraw.decode('utf-8-sig')
rows.append('     bytes=%d lines=%d sha8=%s' % (len(_mraw), _mt.count(chr(13)+chr(10))+1, sha8b(_mm)))
rows.append('     Phase 35 内容在位 = %s' % ('A 闸门 seal 执行与关闭' in _mt))
rows.append('     Phase 36（并发写者 E4/E4b）在位 = %s' % ('## Phase 36' in _mt))
_pre = _mraw[:685649]
rows.append('     prefix(685649) sha8=%s (Phase36 脚本登记 before=5be5bf12)' % hashlib.sha256(_pre).hexdigest()[:8])

rows.append('9) Q03 产物指纹')
for lbl, p in [('q03_execution.json', os.path.join(DKR, 'q03_execution.json')),
               ('q03_result.json', os.path.join(DKR, 'q03_result.json')),
               ('q03_report.txt', os.path.join(DKR, 'q03_report.txt'))]:
    rows.append('     %-22s %8d B  %s' % (lbl, os.path.getsize(p), sha8b(p)))
chk('q03_result res_sha8 self-consistent',
    q['res_sha8'] == hashlib.sha256(
        json.dumps({k: v for k, v in q.items() if k != 'res_sha8'},
                   ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8],
    q['res_sha8'])

# 10) 3251/3152 脚本 sha8 与 3152 断言锚
rows.append('10) 与 3152 内建锚的交叉核对')
chk('glm4 anchor 0.389835 与 3152 断言一致',
    abs(q['per_model']['glm4-9b']['b4_rel_readout_mean3seed_recompute'] - 0.389835258324941) < 1e-9,
    '%.6f' % q['per_model']['glm4-9b']['b4_rel_readout_mean3seed_recompute'])
chk('qwen3-4b kstar 锚 0.007910657984515032',
    abs(q['per_model']['qwen3-4b']['b4_rel_kstar_mean3seed_recompute'] - 0.007910657984515032) < 1e-12,
    '%.9f' % q['per_model']['qwen3-4b']['b4_rel_kstar_mean3seed_recompute'])

rows.append('')
rows.append('=' * 78)
rows.append('汇总: PASS=%d  FAIL=%d  ->  %s' % (P, F, 'ALL_PASS' if F == 0 else 'HAS_FAILURE'))
txt = '\n'.join(rows)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print(txt)
