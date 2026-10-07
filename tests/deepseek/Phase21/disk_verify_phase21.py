# -*- coding: utf-8 -*-
"""
Phase 21 独立磁盘复核（只读；不复用主脚本函数，按主脚本文义重算并按判据导出标签再比对）。
用法：python tests/deepseek/Phase21/disk_verify_phase21.py
产出：tests/deepseek_temp/Phase21/disk_verify_phase21.txt（末行 N/N 项通过）
"""
import io
import os
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
P8T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
OUT = os.path.join(P21T, 'disk_verify_phase21.txt')

n_ok = 0
n_bad = 0
rows = []


def chk(name, cond, got=None, exp=None):
    global n_ok, n_bad
    ok = bool(cond)
    if ok:
        n_ok += 1
    else:
        n_bad += 1
    rows.append('[%s] %-62s got=%s exp=%s' % ('PASS' if ok else 'FAIL', name, got, exp))
    return ok


def h8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


EXEC = os.path.join(P21T, 'execution_phase21.json')
SEAL = os.path.join(P21T, 'N2h1a14_design_seal.json')
RES = os.path.join(P21T, 'result_phase21.json')
JUD = os.path.join(P21T, 'judgement_phase21.json')
BASE = os.path.join(P21T, 'memo_baseline_preappend_phase21.json')

for p in (EXEC, SEAL, RES, JUD, BASE):
    chk('存在 %s' % os.path.basename(p), os.path.exists(p))

EX = json.load(io.open(EXEC, encoding='utf-8'))
S = json.load(io.open(SEAL, encoding='utf-8'))
R = json.load(io.open(RES, encoding='utf-8'))
J = json.load(io.open(JUD, encoding='utf-8'))
B = json.load(io.open(BASE, encoding='utf-8'))

_seal_full = hashlib.sha256(open(SEAL, 'rb').read()).hexdigest()
chk('seal sha 与 exec.seal_sha256 一致', _seal_full == EX['seal_sha256'], _seal_full[:8], EX['seal_sha256'][:8])
chk('result.seal_sha8 == seal sha8', R['seal_sha8'] == h8(SEAL), R['seal_sha8'], h8(SEAL))
chk('result.exec_sha8 == exec sha8', R['exec_sha8'] == h8(EXEC), R['exec_sha8'], h8(EXEC))
chk('judgement.result_sha8 == result sha8', J['result_sha8'] == h8(RES), J['result_sha8'], h8(RES))

# --- 校准：与 P8 冻结文件独立比对
P8R = os.path.join(ROOT, EX['anchor_p8_result_path'])
chk('P8 result 盘上 sha 与 exec 锚一致', hashlib.sha256(open(P8R, 'rb').read()).hexdigest() == EX['anchor_p8_result_sha256'])
R8 = json.load(io.open(P8R, encoding='utf-8'))
A8 = R8['amend1']
cal = R['calibration']
chk('cal.applies (A0_bf16)', cal.get('applies') is True)
chk('cal share_v(mlp) 复现 P8 (<=1e-4)', abs(cal['share_v_mlp']['got'] - float(A8['share_v']['mlp'])) <= 1e-4,
    cal['share_v_mlp']['got'], A8['share_v']['mlp'])
chk('cal max_head_share_v 复现 P8', abs(cal['max_head_share_v']['got'] - float(A8['max_head_share_v'])) <= 1e-4,
    cal['max_head_share_v']['got'], A8['max_head_share_v'])
chk('cal argmax_head_v 复现 P8', cal['argmax_head_v']['got'] == A8['argmax_head_v'],
    cal['argmax_head_v']['got'], A8['argmax_head_v'])
chk('cal W.max_head_share 复现 P8', abs(cal['W_max_head_share']['got'] - float(R8['W']['max_head_share'])) <= 1e-4,
    cal['W_max_head_share']['got'], R8['W']['max_head_share'])
chk('cal W.argmax_head 复现 P8', int(cal['W_argmax_head']['got']) == int(R8['W']['argmax_head']),
    cal['W_argmax_head']['got'], R8['W']['argmax_head'])
chk('cal I_nl 复现 P8', abs(cal['I_nl']['got'] - float(A8['I_nl'])) <= 1e-4, cal['I_nl']['got'], A8['I_nl'])
chk('cal T[diff6] 复现 P8', abs(cal['T_diff6']['got'] - float(R8['T']['diff6']['dDonor'])) <= 1e-4,
    cal['T_diff6']['got'], R8['T']['diff6']['dDonor'])
chk('cal 总判 ok', cal.get('ok') is True)

# --- 每臂内部一致性
AO = R['arm_order']
FL = S['floors']
for a in AO:
    rec = R['arms'][a]
    m1 = rec['M1']
    sv = m1['share_v']
    tot = sum(sv.values())
    chk('%s share_v 求和 == 1 (1e-9)' % a, abs(tot - 1.0) <= 1e-9, tot, 1.0)
    mx = max(sv[k] for k in sv if k != 'mlp')
    chk('%s max_head_share_v 与 share_v 自洽' % a, abs(mx - m1['max_head_share_v']) <= 1e-12, mx, m1['max_head_share_v'])
    chk('%s argmax_head_v 与 share_v 自洽' % a,
        abs(m1['share_v'][m1['argmax_head_v']] - m1['max_head_share_v']) <= 1e-12)
    chk('%s loo_vec_top1 == 1 - max_head_share_v' % a, abs(m1['loo_vec_top1'] - (1 - m1['max_head_share_v'])) <= 1e-12)
    chk('%s G1_core 与阈值自洽' % a,
        bool(rec['G1_core']) == bool(m1['max_head_share_v'] <= FL['G1_MAXHEAD_V'] and m1['share_v_mlp'] <= FL['G1_MLP_SHARE_V']),
        rec['G1_core'])
    chk('%s F4/F5 通过' % a, bool(rec['F4_dims_ok']) and bool(rec['F5_o_proj_ok']))
    chk('%s determinism == 0' % a, abs(rec['E0_selfcheck']['determinism_maxdiff']) <= 1e-12)

# --- 配对重算（独立）
for p in R['quant_pairs']:
    a_n, a_b = p['a_nf4'], p['a_bf16']
    mn, mb = R['arms'][a_n]['M1'], R['arms'][a_b]['M1']
    chk('pair %s d_share_v_mlp 重算' % p['model'],
        abs(p['d_share_v_mlp'] - (mn['share_v_mlp'] - mb['share_v_mlp'])) <= 1e-12)
    chk('pair %s d_max_head_share_v 重算' % p['model'],
        abs(p['d_max_head_share_v'] - (mn['max_head_share_v'] - mb['max_head_share_v'])) <= 1e-12)
    chk('pair %s argmax 同与两侧一致' % p['model'],
        bool(p['argmax_head_v_same']) == bool(mn['argmax_head_v'] == mb['argmax_head_v']))
    # spearman 独立实现（平均秩）
    def avg_rank(x):
        x = list(x)
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    keys = ['head%d' % h for h in range(32)] + ['mlp']
    va = [mn['share_v'][k] for k in keys]
    vb = [mb['share_v'][k] for k in keys]
    ra, rb = avg_rank(va), avg_rank(vb)
    ma = sum(ra) / len(ra); mb_ = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb_) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb_) ** 2 for y in rb)) ** 0.5
    rho_avg = num / den if den > 0 else None
    d = abs((rho_avg or 0) - (p['spearman_share_v'] or 0))
    chk('pair %s spearman 独立（平均秩）差 <= 0.05' % p['model'], d <= 0.05, rho_avg, p['spearman_share_v'])

# --- 预测与 verdict 自洽
P = R['predictions']
chk('n_pass == True 计数', R['n_pass'] == sum(1 for v in P.values() if v is True), R['n_pass'])
chk('n_total == 非 None 计数', R['n_total'] == sum(1 for v in P.values() if v is not None), R['n_total'])
lab_map = [('P1_calib_A0bf16_reproduces_P8', 'calib_p8_ok'), ('P2_share_v_mlp_stable', 'share_v_stable'),
           ('P3_max_head_share_v_stable', 'maxv_stable'), ('P4_argmax_head_v_same', 'argmax_v_same'),
           ('P5_G1_core_both_precisions', 'g1_core_both'), ('P6_W_stable', 'w_stable'),
           ('P7_spearman_share_v', 'rho_share_v_ok'), ('P8_conf_same_band', 'conf_same_band'),
           ('P9_floors', 'floors_ok')]
exp_labels = [lab for k, lab in lab_map if P.get(k) is True]
chk('judgement.verdict == 由 predictions 导出', J['verdict'] == '__'.join(exp_labels), J['verdict'], '__'.join(exp_labels))
chk('judgement.result_sha8 与 result 一致', J['result_sha8'] == h8(RES))
chk('judgement.n_forwards_per_arm 与 result 一致',
    J['n_forwards_per_arm'] == {a: int(R['arms'][a]['n_fw']) for a in AO})

# --- Ledger
L = json.load(io.open(LEDGER, encoding='utf-8'))
ent = [x for x in L['measurements'] if x.get('phase') == 21]
chk('Ledger 含 1 条 phase 21', len(ent) == 1, len(ent))
if ent:
    e = ent[0]
    chk('Ledger result_sha8 == result sha8', e['result_sha8'] == h8(RES), e['result_sha8'], h8(RES))
    chk('Ledger seal_sha8 == seal sha8', e['seal_sha8'] == h8(SEAL), e['seal_sha8'], h8(SEAL))
    chk('Ledger verdict == judgement.verdict', e['verdict'] == J['verdict'], e['verdict'], J['verdict'])
chk('Ledger 备份存在', os.path.exists(os.path.join(P21T, 'atlas_ledger_backup_pre_phase21.json')))

# --- MEMO
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig')
chk('MEMO BOM', mb[:3] == b'\xef\xbb\xbf')
chk('MEMO bare_lf == 0', mb.count(b'\n') - mb.count(b'\r\n') == 0)
chk('MEMO 含 Phase 21 标题（唯一）', mt.count('## Phase 21:') == 1, mt.count('## Phase 21:'))
chk('MEMO 前缀锚 == 基线 bytes', len(mb) >= int(B['bytes']))
chk('MEMO 前缀 sha8 == 基线 sha8', hashlib.sha256(mb[:int(B['bytes'])]).hexdigest()[:8] == B['sha8'],
    hashlib.sha256(mb[:int(B['bytes'])]).hexdigest()[:8], B['sha8'])

# --- present
PP = os.path.join(P21T, 'present_phase21.html')
if os.path.exists(PP):
    pt = io.open(PP, encoding='utf-8').read()
    chk('present 含 G1 结论与校准表', ('装置校准' in pt) and ('G1' in pt))
else:
    chk('present 存在', False)

rows.append('')
rows.append('=== 汇总：PASS=%d FAIL=%d ===' % (n_ok, n_bad))
rows.append('VERDICT = %s' % ('ALL_PASS' if n_bad == 0 else 'HAS_FAIL'))
rows.append('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(rows) + '\n')
print('\n'.join(rows[-4:]))
print('DISK VERIFY DONE  PASS=%d FAIL=%d -> %s' % (n_ok, n_bad, OUT))
