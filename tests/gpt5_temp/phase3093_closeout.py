# -*- coding: utf-8 -*-
"""Phase 3093 closeout (idempotent, dual-arm):
Ledger(meas3093 + L14) -> MEMO Phase 3093 ->
audit 54 addendum -> wlog -> MEMORY.md ->
closeout_log.
Arm A1: omega_p90_qwen14b_layer_scan (fifth
spectrum point qwen3-14b, layer scan L_CAND).
Arm A2: omega_p91_qwen14b_l{LB}_full_arbitration
(discovered dynamically; exists only when A1
verdict == layer_rescue and the arbitration run
has completed).
All numbers are read from sealed result.json /
npz at closeout time - nothing is hardcoded
before observation.  If A1 == layer_rescue but
A2 is missing, exits A2_NOT_DONE (code 2) with
no writes."""
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PRES = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3093')
A1D = os.path.join(
    PRES, 'omega_p90_qwen14b_layer_scan')
A1NPZ = A1D + (r'\omega_p90_qwen14b_'
               r'layer_scan.npz')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
LOGF = A1D + r'\closeout_log.txt'
o = []


def f4(x):
    return 'n/a' if x is None else '%+.4f' % x


def f5p(x):
    return 'n/a' if x is None else '%.5f' % x


# ---------- load A1 ----------
res1 = json.load(io.open(
    A1D + r'\result.json', encoding='utf-8'))
seal1 = json.load(io.open(
    A1D + r'\seal.json', encoding='utf-8'))
z1 = np.load(A1NPZ, allow_pickle=False)
v1 = res1['verdict']
assert v1 in ('layer_rescue',
              'layer_partial',
              'layer_absent'), v1
assert str(z1['VERDICT']) == v1
assert bool(z1['SMOKE']) is False
assert bool(z1['SETUP_OK'])
assert res1['stats']['setup_ok_all'] is True
FA1 = res1['stats']['fam_anchors']
assert all(
    bool(vv) if kk.endswith('_ok')
    else (abs(float(vv)) <= 1e-9
          if kk.startswith(('b0_', 'b1_',
                            'b6_', 'b7a_'))
          else True)
    for fk in ('A', 'B', 'C')
    for kk, vv in FA1[fk].items())
L_CAND = [int(x) for x in z1['L_CAND']]
FL = res1['stats']['families_layers']
fw1 = int(res1['forwards'])
el1 = float(res1['elapsed'])
o.append('A1 loaded: verdict=%s fw=%d el=%.0fs'
         % (v1, fw1, el1))

# ---------- discover A2 ----------
a2dirs = [nm for nm in sorted(os.listdir(PRES))
          if re.match(
              r'omega_p91_qwen14b_l\d+_'
              r'full_arbitration$', nm)]
LB = None
res2 = seal2 = None
v2 = None
fw2 = el2 = 0
if v1 == 'layer_rescue':
    if not a2dirs:
        io.open(LOGF, 'w',
                encoding='utf-8').write(
            'A2_NOT_DONE (A1 rescue but no '
            'arbitration dir yet)\n')
        print('A2_NOT_DONE')
        sys.exit(2)
    assert len(a2dirs) == 1, a2dirs
    LB = int(res1['stats']['rescue_best'])
    assert a2dirs[0] == (
        'omega_p91_qwen14b_l%d_full_'
        'arbitration' % LB), (a2dirs, LB)
    A2D = os.path.join(PRES, a2dirs[0])
    res2 = json.load(io.open(
        A2D + r'\result.json',
        encoding='utf-8'))
    seal2 = json.load(io.open(
        A2D + r'\seal.json', encoding='utf-8'))
    z2 = np.load(
        A2D + '\\' + a2dirs[0] + '.npz',
        allow_pickle=False)
    v2 = res2['verdict']
    assert v2.startswith('fifth_'), v2
    assert str(z2['VERDICT']) == v2
    assert bool(z2['SMOKE']) is False
    fw2 = int(res2['forwards'])
    el2 = float(res2['elapsed'])
    rp = res2['stats'].get('repro')
    if rp:
        assert all(
            abs(float(dv)) <= 1e-9
            for fd in rp.values()
            for kk, dv in fd.items()
            if 'diff' in kk)
    o.append('A2 loaded: dir=%s verdict=%s '
             'fw=%d el=%.0fs'
             % (a2dirs[0], v2, fw2, el2))
else:
    assert not a2dirs, a2dirs
    assert res1['stats']['rescue_layers'] == []
    assert res1['stats']['rescue_best'] is None

# ---------- A1 grid table ----------
grid = []
for L in L_CAND:
    cells = []
    for fk in ('A', 'B', 'C'):
        d = FL[fk][str(L)]
        cells.append('%d / %s / %s'
                     % (d['n_neg'],
                        f4(d['med_c']),
                        f4(d['r_all'])))
    grid.append('| L%d | %s |'
                % (L, ' | '.join(cells)))
grid_txt = '\n'.join(grid)
if v1 == 'layer_rescue':
    nn_best = min(FL[f][str(LB)]['n_neg']
                  for f in ('A', 'B', 'C'))
    resc_txt = ('rescue_layers=%s, best=L%d '
                '(min-family n_neg=%d)'
                % (res1['stats']['rescue_layers'],
                   LB, nn_best))
else:
    resc_txt = 'rescue_layers=[]'

# ---------- A2 tables (when present) ----------
if v2 is not None:
    fa2 = res2['stats']['families']
    f2rows = '\n'.join(
        '| %s | %s | %s | %d | %s | %s |'
        % (fk, fa2[fk]['domain'],
           f4(fa2[fk]['med_c']),
           int(fa2[fk]['n_neg']),
           f4(fa2[fk]['capture8']),
           f4(fa2[fk]['r_all']))
        for fk in ('A', 'B', 'C'))
    g = res2['gates']
    cross = res2['stats'].get('cross')
    if cross is not None:
        e3rows = '\n'.join(
            '| %s | %s | %s |'
            % (k, f4(d.get('sp')),
               f5p(d.get('p')))
            for k, d
            in sorted(cross['e3'].items()))
        gate_txt = (
            'G0(G_DS)=%s, count=%d/6, n_bonf=%d, '
            'min_sp=%s, stouffer=%s, G1=%s; '
            'f2~T_AB sp=%s p=%s; spec_class=%s'
            % (g.get('G_DS'),
               int(g.get('count_sig_pos', -1)),
               int(g.get('n_bonf', -1)),
               f4(g.get('min_sp')),
               f4(g.get('stouffer_z')),
               g.get('G1'),
               f4(g.get('f2_TAB_sp')),
               f5p(g.get('f2_TAB_p')),
               g.get('spec_class')))
    else:
        e3rows = '(skipped - early exit)'
        gate_txt = ('early exit %s, gates null'
                    % v2)
else:
    f2rows = e3rows = gate_txt = ''

# ---------- narrative ----------
verd = v2 if v2 is not None else v1
g_cnt = (int(g.get('count_sig_pos', -1))
         if v2 is not None else -1)
g_min = (f4(g.get('min_sp'))
         if v2 is not None else 'n/a')
if v2 is None:
    narr = (
        'A1 判决 %s（%s）：qwen3-14b 谱点在预注册'
        '全族判据下未达 layer_rescue，A2 patch 按'
        '设计拒绝执行（assert verdict==layer_rescue'
        '）——按 3087 前例改判后续设计，A2 缺席'
        '如实登记。\n' % (v1, resc_txt))
elif v2 == 'fifth_mixed_absent':
    narr = (
        'A2 在 L%d 给出 fifth_mixed_absent：谱位'
        '混合 + G0 计数门缺席（count=%d/6, '
        'min_sp=%s）——与 3087 GLM4 L37 同名格局。'
        'G_DS 在 Qwen3 家族 3.9x 尺度上仍不迁移，'
        '规模本身不制造 G_DS；谱位定容量、trunk 定'
        '锁定的双格局在第五点复制为缺席侧。'
        % (LB, g_cnt, g_min))
elif 'locked' in v2:
    narr = (
        'A2 在 L%d 给出 %s：G0 门通过且 f2~T_AB '
        '显著——五谱点首次 locked 迁移信号。G_DS '
        '在 qwen3-14b L%d 过合取复制标准，与 '
        '3087 GLM4 缺席形成跨尺度对照：迁移状态'
        '依模型/层位而异，非普适缺席。'
        % (LB, v2, LB))
elif v2.endswith('migrates') or 'partial' in v2:
    narr = (
        'A2 在 L%d 给出 %s：G0 门通过（count=%d/6, '
        'min_sp=%s）但 f2~T_AB 判据分出迁移强度。'
        'G_DS 在 qwen3-14b 过复制标准——与 3087 '
        'GLM4 缺席形成跨尺度对照，迁移状态依模型/'
        '层位而异。'
        % (LB, v2, g_cnt, g_min))
elif 'no_migrate' in v2:
    narr = (
        'A2 在 L%d 给出 %s：谱位非混合（%s）且 '
        'G0 门缺席（count=%d/6, min_sp=%s）——'
        'qwen3-14b 谱点在 trunk/dispersed 形态下'
        'G_DS 缺席，与 3087 混合缺席互补：缺席'
        '不依赖谱位形态。'
        % (LB, v2, g.get('spec_class'),
           g_cnt, g_min))
else:
    narr = (
        'A2 退化出口 %s 如实登记（G0 门未达评价'
        '条件）——负结果一等公民，不放宽判据。'
        % v2)

if v2 is not None and (
        'locked' in v2
        or v2.endswith('migrates')
        or 'partial' in v2):
    concl = (
        'G_DS 迁移信号首次出现在第五谱点'
        '（qwen3-14b L%d）。接续 3094：'
        '复制验证优先（第二 rescue 层 / L38 式'
        '重复仲裁），随后 G_DS 门敏感性复查'
        '（3092 框架扩到第五点）；备选 B 4B 主干'
        '解剖 / C R1 复用拓扑全景。'
        % LB)
elif v2 is not None:
    concl = (
        '第五谱点缺席登记完成：G_DS 在 G0 合取'
        '复制门下于 qwen3-14b L%d 缺席存活，'
        '谱点矩阵更新为 4 架构 × 5 判定点'
        '（跨架构声明维持 4 架构 1 非 Qwen 点，'
        'qwen3-14b 属同族尺度扩展）。接续 3094：'
        'B 4B 主干解剖；C R1 复用拓扑全景'
        '（reuse_inventory 底册已备）。' % LB)
elif v1 == 'layer_partial':
    concl = (
        'qwen3-14b 记为部分谱点（C 族某层恢复、'
        '全族判据未过）。接续 3094：C R1 复用拓扑'
        '全景；备选密集网格（候选层邻域加密）与 '
        'B 4B 主干解剖。')
else:
    concl = (
        '第五谱点直接缺席（无任何候选层位使 C 族 '
        'n_neg>=8）。接续 3094：C R1 复用拓扑全景'
        '；备选 B 4B 主干解剖。')

# ---------- title / prereg / hardbounds ----------
if v2 is not None:
    title = ('Ω-P90/P91 qwen3-14b 第五谱点'
             '（层位扫描+全仲裁）——A1 %s / A2 %s'
             % (v1, v2))
else:
    title = ('Ω-P90 qwen3-14b 第五谱点层位扫描'
             '——判决 %s（A2 未执行）' % v1)
hardb = (
    '- 层网格仅 %d 个等比点（0.78/0.85/0.93/'
    '0.95），rescue 层位可能位于网格间隙；\n'
    '- n_neg>=8 为预注册约定阈值，临界层'
    '（=8/9）对判定敏感；\n'
    '- sysmem fallback（~12GB 跨 PCIe）bf16 '
    '路径：repro 锚证 bit 级确定性，但单 fw '
    '~0.6s 慢于常驻显存；\n'
    '- 继承 frozen spearman tie-order 约定'
    '（3091）；G0 计数门为唯一判决门（3092 '
    '纪律，无符号/符号 Stouffer 仅作对照）；\n'
    '- qwen3-14b untied（40L GQA 8kv）：'
    'logits-only 协议不受影响（run_log 登记）；'
    '同族尺度扩展不增加跨架构声明覆盖。'
    % len(L_CAND))
prereg_txt = (
    'A1/A2 均预注册冻结（execution.json 先于'
    '任何观测；A2 patch 在 A2 观测前生成且'
    '断言 A1 verdict==layer_rescue）')

# ---------- MEMO entry ----------
memo_txt = (
    '\n## Phase 3093: %s [%%s]\n\n'
    '**状态**: 已执行（A1 %%d 前向 / %%.0fs；'
    '%%s qwen3-14b bf16 全载 29.54GB，'
    'sysmem fallback 非 OOM）。%%s。\n\n'
    '### 1. 目标与原理\n'
    '第五谱点 qwen3-14b（40L/5120H/40Q/8kv '
    'GQA/vocab 151936/untied，3.9x 4B 同族'
    '尺度）——R1 底册最大缺口。A1 层位扫描：'
    'L_CAND=%s（40 层等比 0.78/0.85/0.93/'
    '0.95），三族（A everyday-causal 3074 '
    '文本 bit 锚族 / B / C）× 每层（24 ladder '
    '+ 40x24 head-scan + 24 R_ALL）；判据 '
    'n_neg>=8 全三族 -> layer_rescue。A2 '
    '全仲裁：L_INJ=rescue best 层，24 ladder '
    'x 3 族 + E3 六检验 + G0 计数门（3092 '
    '纪律：唯一判决门）。\n\n'
    '### 2. A1 层位网格（n_neg / med_c / '
    'R_ALL）\n'
    '| 层 | 族A | 族B | 族C |\n'
    '|---|---|---|---|\n'
    '%s\n\n'
    'A1 判决：**%s**（%s）。\n\n'
    '%%s'
    '### 5. 分析\n'
    '%%s\n\n'
    '### 6. 硬伤与边界\n'
    '%s\n\n'
    '### 7. 结论与接续\n'
    '%%s\n\n'
    '资源消耗：A1 %%d fw %%.0fs；%%s产物 '
    'sealed（A1 npz8=%%s result8=%%s；%%s）。\n'
    % (title, L_CAND, grid_txt, v1, resc_txt,
       hardb))
memo_txt = memo_txt % (
    datetime.now().strftime('%Y-%m-%d %H:%M'),
    fw1, el1,
    ('A2 全仲裁 %d 前向 / %.0fs；'
     % (fw2, el2)) if v2 is not None else '',
    prereg_txt,
    ('### 3. A2 全仲裁（L%d）\n' % LB
      + '| 族 | domain | med_c | n_neg | '
        'capture8 | R_ALL |\n'
      + '|---|---|---|---|---|---|\n'
      + f2rows + '\n\n'
      + 'E3 六检验（sp / p）：\n'
      + '| 检验 | sp | p |\n'
      + '|---|---|---|\n'
      + e3rows + '\n\n'
      + '门摘要：%s\n\n'
      + 'repro 锚：A2 对 A1 L%d npz（n_neg/'
        'top8 精确相等，med_c/CS1H/R_ALL '
        '<=1e-9）全过。\n\n'
      + '### 4. 门语义\n'
      + 'G0 合取复制标准为唯一判决门；'
        'stouffer_z/G1 仅作对照记录（3092 '
        '纪律）。\n\n') % (gate_txt, LB)
     if v2 is not None
     else ('### 3. A2 未执行\nA2 patch 按设计'
           '拒绝执行（预注册守门），A2 缺席'
           '如实登记。\n\n'),
    narr, concl,
    fw1, el1,
    ('A2 %d fw %.0fs；' % (fw2, el2))
    if v2 is not None else '',
    seal1['npz_sha256_8'],
    seal1['result_sha256_8'],
    ('A2 npz8=%s result8=%s'
     % (seal2['npz_sha256_8'],
        seal2['result_sha256_8']))
    if v2 is not None else 'A2 无产物')

# ---------- Ledger ----------
meas = {
    'meas_id':
        'meas3093_omega_p90_p91_qwen14b'
        '_fifth_point',
    'phase': 3093,
    'claim':
        'Omega-P90/P91 (3093 dual-arm when '
        'A2 ran) - qwen3-14b (40L/40Q/8kv/'
        '5120, vocab 151936, untied, 3.9x 4B '
        'within Qwen3 family) fifth spectrum '
        'point.  A1 layer scan L%s (40L '
        'ratios .78/.85/.93/.95): 3 families '
        'x (24 ladder + 40x24 head-scan + 24 '
        'R_ALL) per layer, %d forwards; '
        'rescue = all-family n_neg>=8 -> %s '
        '(%s).  %s'
        % (L_CAND, fw1, v1, resc_txt,
           ('A2 full arbitration at L%d: '
            '24 ladder x 3 families + E3 six '
            'tests + G0 count gate (sole '
            'decision gate per 3092) -> %s '
            '(G_DS=%s count=%d/6 n_bonf=%d '
            'min_sp=%s stouffer=%s G1=%s).  '
            % (LB, v2, g.get('G_DS'),
               g_cnt, int(g.get('n_bonf', -1)),
               g_min, f4(g.get('stouffer_z')),
               g.get('G1')))
           if v2 is not None
           else 'A2 not run (prereg gate '
                'refused non-rescue verdict).  '),
    'verdict': verd,
    'anchors':
        'A1: fam anchors b0/b1/b3/b6/b7a '
        'bit-0 x3 families + per-layer b4/b8 '
        'all ok; A2 (if run): same anchor set '
        'at L_INJ=%s + repro anchor vs A1 '
        'best-layer npz (n_neg/top8 exact, '
        'med_c/CS1H/R_ALL <=1e-9) all True'
        % (LB,) if v2 is not None else
        'A1: fam anchors b0/b1/b3/b6/b7a '
        'bit-0 x3 families + per-layer b4/b8 '
        'all ok (setup_ok_all=True)',
    'artifacts':
        ({} if v2 is None else {
            'result':
                'phase3093/%s/result.json'
                % a2dirs[0],
            'npz': 'phase3093/%s/%s.npz'
                   % (a2dirs[0], a2dirs[0]),
            'a1_scan':
                'phase3093/omega_p90_qwen14b_'
                'layer_scan/omega_p90_qwen14b_'
                'layer_scan.npz'})
        if v2 is not None else {
            'result':
                'phase3093/omega_p90_qwen14b_'
                'layer_scan/result.json',
            'npz': 'phase3093/omega_p90_qwen14b_'
                   'layer_scan/omega_p90_qwen14b_'
                   'layer_scan.npz'},
    'hashes':
        ({'npz_sha256_8':
            seal2['npz_sha256_8'],
          'result_sha256_8':
            seal2['result_sha256_8'],
          'script_sha256_8':
            seal2['script_sha256_8'],
          'a1_npz_sha256_8':
            seal1['npz_sha256_8']}
         if v2 is not None else
         {'npz_sha256_8':
            seal1['npz_sha256_8'],
          'result_sha256_8':
            seal1['result_sha256_8'],
          'script_sha256_8':
            seal1['script_sha256_8']}),
    'note':
        'prereg frozen in execution.json '
        'before observation; A2 patch '
        'generated before any A2 observation '
        'with verdict==layer_rescue assert',
}
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(isinstance(m, dict)
           and m.get('phase') == 3093
           for m in led['measurements']):
    led['measurements'].append(meas)
    assert len(led['measurements']) == 231
    l14['connects'].append({
        'meas_id':
            'meas3093_omega_p90_p91_qwen14b'
            '_fifth_point',
        'phase': 3093,
        'axis': 'lang',
        'verdict': verd,
        'grade_change':
            'Omega-P90/P91: qwen3-14b fifth '
            'spectrum point (3.9x scale '
            'WITHIN Qwen3 family) - A1 %s '
            '(%s); %s  Next: %s'
            % (v1, resc_txt,
               ('A2 %s (G0 gate G_DS=%s, '
                'count=%d/6, min_sp=%s)'
                % (v2, g.get('G_DS'), g_cnt,
                   g_min))
               if v2 is not None
               else 'A2 not run (prereg gate)',
               ('replicate migration signal '
                '(2nd rescue layer / L38-style '
                'replica), then gate-sensitivity '
                '5-point extension'
                if v2 is not None and (
                    'locked' in v2
                    or v2.endswith('migrates')
                    or 'partial' in v2)
                else '3094 B 4B trunk anatomy; '
                     'C R1 reuse topology '
                     'panorama'))})
    assert len(l14['connects']) == 199
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d '
             'sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already has 3093')

# ---------- MEMO append ----------
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3093:' not in memo:
    i3092 = memo.rindex('## Phase 3092:')
    assert i3092 > memo.rindex(
        '## Phase 3091:')
    memo = memo.rstrip('\n') + '\n' + memo_txt
    with io.open(MEMO, 'w',
                 encoding='utf-8') as f:
        f.write(memo)
    o.append('memo appended ts=%s' % ts)
else:
    o.append('memo already has 3093')

# ---------- audit 54 ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '五十四' not in aud:
    aud = aud.rstrip('\n') + (
        '\n\n---\n\n'
        '## 五十四、3093 增补：qwen3-14b '
        '第五谱点——%s\n'
        '1. **判决**：A1 层位扫描（L31/L34/L37/'
        'L38 等比网格，三族 24 ladder + 40x24 '
        'head-scan + 24 R_ALL）给出 %s（%s）；'
        '%s\n'
        '2. **门语义**：G0 合取复制计数门维持'
        '唯一判决门（3092 纪律）；%s\n'
        '3. **谱点矩阵**：4 架构 × 5 判定点'
        '（qwen3-14b 为 Qwen3 家族内 3.9x 同族'
        '尺度扩展，不新增跨架构覆盖）。\n'
        % (title, v1, resc_txt,
           ('A2 全仲裁 L%d 给出 %s（G_DS=%s, '
            'count=%d/6, min_sp=%s）。'
            % (LB, v2, g.get('G_DS'), g_cnt,
               g_min))
           if v2 is not None
           else 'A2 按预注册守门未执行。',
           ('stouffer/G1 仅对照记录')
           if v2 is not None
           else '（无门对比数据）'))
    with io.open(AUDIT, 'w',
                 encoding='utf-8') as f:
        f.write(aud)
    o.append('audit 54 appended')
else:
    o.append('audit 54 already')

# ---------- wlog ----------
wlf = os.path.join(
    WLOG_DIR,
    datetime.now().strftime('%Y-%m-%d')
    + '.md')
try:
    prev = io.open(wlf, encoding='utf-8').read()
except IOError:
    prev = ''
WLOG_TAG = 'fifth spectrum point: A1'
if WLOG_TAG not in prev:
    line = ('- Phase 3093 Omega-P90/P91 '
            'qwen3-14b fifth spectrum point: '
            'A1 %s (%s); %s  Audit 54; '
            'ledger 231/L14 199.\n'
            % (v1, resc_txt,
               ('A2 %s (G0 G_DS=%s count=%d/6 '
                'min_sp=%s).'
                % (v2, g.get('G_DS'), g_cnt,
                   g_min))
               if v2 is not None
               else 'A2 not run (prereg gate).'))
    with io.open(wlf, 'a',
                 encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended %s'
             % os.path.basename(wlf))
else:
    o.append('wlog already')

# ---------- MEMORY.md ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
mem_cur = io.open(MEMO_W,
                  encoding='utf-8').read()
if 'max=3093' not in mem_cur:
    verd_short = (v1 + (' + A2 ' + v2)
                  if v2 is not None
                  else v1 + '（A2 未执行）')
    mig_sig = (v2 is not None and (
        'locked' in v2
        or v2.endswith('migrates')
        or 'partial' in v2))
    if mig_sig:
        nxt = ('- max=3093（qwen3-14b 第五谱点 '
               '%s，G_DS 迁移信号出现）——下一个 '
               '3094：复制验证（第二 rescue 层/'
               'L38 式重复）→ 门敏感性复查；备选 '
               'B 4B 主干解剖 / C R1 复用拓扑全景。'
               % verd_short)
    else:
        nxt = ('- max=3093（qwen3-14b 第五谱点 '
               '%s，G0 门下缺席/未执行）——下一个 '
               '3094：B 4B 主干解剖；C R1 复用拓扑'
               '全景（reuse_inventory 底册已备）。'
               % verd_short)
    anchor1 = ('- max=3092，下一个 3093（A '
               'qwen3-14b 第五谱点 ~21k 前向 GPU；'
               'B 4B 主干解剖；C R1 复用拓扑全景'
               '——reuse_inventory 底册已备）。')
    assert anchor1 in mem_cur
    mem_cur = mem_cur.replace(anchor1, nxt)
    anchor2 = '；repro 锚 bit 级通过）。'
    assert anchor2 in mem_cur
    mem_cur = mem_cur.replace(
        anchor2,
        '；repro 锚 bit 级通过）、qwen3-14b（'
        '40L 40Q 8kv GQA 5120 vocab 151936 '
        'untied bf16 29.54GB sysmem fallback '
        '非 OOM；3093 第五谱点）。')
    anchor3 = ('3092 G_DS 门敏感性 '
               'gate_sensitive_substantive'
               '（无符号 Stouffer 否决）**。')
    assert anchor3 in mem_cur
    mem_cur = mem_cur.replace(
        anchor3,
        '3092 G_DS 门敏感性 '
        'gate_sensitive_substantive（无符号 '
        'Stouffer 否决）→ 3093 qwen3-14b 第五'
        '谱点（%s）**。' % verd_short)
    assert len(mem_cur) < 3000, len(mem_cur)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_cur)
    o.append('memory updated %d chars'
             % len(mem_cur))
else:
    o.append('memory already max=3093')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
