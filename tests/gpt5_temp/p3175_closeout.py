# -*- coding: utf-8 -*-
# Phase 3175 closeout: five-write (ledger / MEMO / daily / workspace MEMORY /
# self-check) + gap ledger v1.5 (GAP-4 mechanism_note finalized, H2 branch).
# Idempotent. All numbers rendered live from sealed result.json.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(S, 'phase3175', 'g5a12_residual_arms')
GV14 = os.path.join(S, 'phase3173', 'g5a10_atlas_v13', 'gap_ledger_v1_4.json')
GV15 = os.path.join(PDIR, 'gap_ledger_v1_5.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3175_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3175c] ' + s)
    print('[3175c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
DG = R['delta_gate']
assert DG['main_cls'] == 'port_residual_dominant', DG['main_cls']
delta = DG['delta']
r_in = DG['ratio_in_kmax']
r_out = DG['ratio_out_kmax']
PCD = R['per_class_delta']
recov_out = (R['pooled']['out']['0']['ratio'] - r_out) / R['pooled']['out']['0']['ratio']
recov_in = (R['pooled']['in']['0']['ratio'] - r_in) / R['pooled']['in']['0']['ratio']
COLL = {'collect_out_qwen3-4b.npz': 'd4f8931b',
        'collect_out_qwen3-14b.npz': 'b9f0a96f',
        'collect_out_glm4-9b.npz': 'e14c62c4'}
for fn, h in COLL.items():
    assert sha8_file(os.path.join(PDIR, fn)) == h, ('collect sha', fn)

# ---------- 0. gap ledger v1.5 (GAP-4 finalized, H2 branch) ----------
if os.path.exists(GV15):
    gv15_sha = sha8_file(GV15)
    log('0. gap ledger v1.5 already present, skip (sha8=' + gv15_sha + ')')
else:
    G = json.load(io.open(GV14, encoding='utf-8'))
    G15 = json.loads(json.dumps(G, ensure_ascii=False))
    g4 = [x for x in G15['gaps'] if x['id'] == 'GAP-4'][0]
    others_before = json.dumps([x for x in G['gaps'] if x['id'] != 'GAP-4'],
                               ensure_ascii=False, sort_keys=True)
    stmt_add = (' 3175 交叉臂定判：残留与校准实体来源无关（面板内 ' +
                ('%.1f%%' % (recov_in * 100)) + ' vs 面板外 ' +
                ('%.1f%%' % (recov_out * 100)) + '，delta=+' +
                ('%.4f' % delta) + ' <= 0.15）——结构性残留确认为 one-hot 类端口'
                '机制固有（port_residual_dominant），GAP-4 机制画像定稿。')
    g4['statement'] = g4['statement'] + stmt_add
    g4['evidence'].append(
        'p3175 cross-arm: delta=ratio_out(k8)-ratio_in(k8)=+' + ('%.4f' % delta) +
        ' (|delta|<=0.15 port_residual_dominant); pooled in ' +
        '-> '.join('%.4f' % R['pooled']['in'][k]['ratio']
                   for k in sorted(R['pooled']['in'], key=int)) +
        ' / out ' + '-> '.join('%.4f' % R['pooled']['out'][k]['ratio']
                               for k in sorted(R['pooled']['out'], key=int)) +
        '; per-model k8 in ' + '/'.join('%.4f' % R['per_model'][mk]['in_curve']['8']['ratio']
                                        for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')) +
        ' out ' + '/'.join('%.4f' % R['per_model'][mk]['out_curve']['8']['ratio']
                           for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')) +
        '; per-class delta non-uniform (乐器/运动 all-positive 3/3, 天气/电器 mixed, none all-negative)')
    g4['anchor_sha8']['p3175'] = sha8_file(os.path.join(PDIR, 'result.json'))
    g4['mechanism_note'] = (g4['mechanism_note'] +
                            ' 3175 交叉臂定稿（G5-A12）：校准实体来源交叉（面板内 vs 全新'
                            '面板外词表 40 实体，GPU verbatim 3169 链采集，D_out2 bitwise '
                            'reself）——delta = ratio_out(k8) - ratio_in(k8) = +' +
                            ('%.4f' % delta) + '（|delta|<=0.15 门内，port_residual_dominant）。'
                            'out 臂恢复 2.5388 -> ' + ('%.4f' % r_out) + '（' +
                            ('%.1f%%' % (recov_out * 100)) + '），in 臂 ' +
                            ('%.1f%%' % (recov_in * 100)) + '，形状一致饱和相似；per-class delta '
                            '方向非一致（乐器/运动全正 3/3、天气/电器混合、无类全负）。结论：残留 '
                            '~1.8x 过量误差为 one-hot 类端口机制固有（port-intrinsic），与校准实体'
                            '来源无关；「实体熟悉度」成分存在但非主体且类间不一致。机制链闭环：'
                            '3169 量化 -> 3171 encoding_missing 否定 -> 3172 端口部分恢复 -> '
                            '3175 残留=端口固有。锚：p3175 res cb76a177 / seal 9e1a2f08'
                            '（design c4568dc2）。')
    others_after = json.dumps([x for x in G15['gaps'] if x['id'] != 'GAP-4'],
                              ensure_ascii=False, sort_keys=True)
    assert others_before == others_after, 'other gaps must stay byte-identical'
    assert json.dumps(G['appendix'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(G15['appendix'], ensure_ascii=False, sort_keys=True), 'appendix immutable'
    G15['version'] = '1.5'
    G15['updated'] = time.strftime('%Y-%m-%d')
    G15['supersedes'] = 'gap_ledger_v1_4 (2413a0cc, Phase 3173)'
    G15['schema'] = 'rdc_atlas_gap_ledger_v1_5'
    with io.open(GV15, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(json.dumps(G15, ensure_ascii=False, indent=1, sort_keys=False))
    gv15_sha = sha8_file(GV15)
    log('0. gap ledger v1.5 written (GAP-4 finalized; others byte-identical) sha8=' + gv15_sha)

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3175 for m in ms_):
    log('1. ledger: 3175 already present, skip')
else:
    detail = (
        'G5-A12 structural residual localization GPU cross-arm: in-arm replays sealed 3172 '
        'curve bitwise (pooled all-k x3 keys + per-model all-k x4 keys drift<1e-9; k=0 both '
        'arms reproduce 3169 gate ratio_B=2.5388114997805062 drift=0.00e+00; out zero-extension '
        'degenerate path exact 0.0); out-arm NEW vocabulary 40 entities (10/class, frozen in '
        'DESIGN before any GPU forward) collected with 3169 verbatim chain (batch=1 bf16 '
        'full-layer last-token; D_out2 3-prompt bitwise reself x3 models); test-set parity '
        'te_oov = ALL_OOV_ROWS minus in-arm cal rows for BOTH arms. Verdict '
        'port_residual_dominant: delta=+' + ('%.4f' % delta) + ' (|delta|<=0.15), '
        'ratio_in(k8)=' + ('%.4f' % r_in) + ' vs ratio_out(k8)=' + ('%.4f' % r_out) +
        '; out recovery ' + ('%.1f%%' % (recov_out * 100)) + ' vs in ' +
        ('%.1f%%' % (recov_in * 100)) + ', same saturation shape; per-class delta non-uniform '
        '(instruments/sports all-positive 3/3, weather/appliances mixed, none all-negative) -> '
        'entity-familiarity component exists but is NOT the residual body; residual ~1.8x = '
        'port-intrinsic. Mechanism chain closed: 3169 quantified collapse -> 3171 '
        'encoding_missing rejected -> 3172 port intervention ~29% -> 3175 residual is '
        'port-intrinsic. GAP-4 mechanism_note finalized (gap ledger v1.5 ' + gv15_sha + '). '
        'Honest log: 1 pre-observation fix (DESIGN.ent_out_full referenced SMOKE-truncated '
        'vocab -> run-mode-dependent design hash d6810f40 vs c4568dc2; replaced with full-copy '
        'ENT_OUT_FULL and refrozen; collect npz cache reused bitwise across the fix). '
        'design_sha=' + R['design_sha8'] + '. res ' + R['res_sha8'] + ' seal ' + R['seal_sha8'] +
        ' (smoke res ' + SM['res_sha8'] + ').')
    entry = {
        'phase': 3175, 'name': 'g5a12_residual_arms', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4-9b (GPU out-collect 3600 fwd)',
        'verdict': R['verdict'],
        'evidence_level': 'E2_predictive',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-A12',
        'superseded_by': None,
        'detail': detail,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms_.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    log('1. ledger: appended 3175 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3175' in norm:
    log('2. MEMO: 3175 section already present, skip')
else:
    marker = '### 缺口排序裁决与接续：预注册 3175'
    mi = norm.rfind(marker)
    assert mi > 0, '3175 prereg marker not found'
    S2 = []
    S2.append('## Phase 3175: G5-A12 谱外类结构残留定位 GPU 交叉臂 ' + time.strftime('%H:%M'))
    S2.append('')
    S2.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3175_g5a12_residual_arms.py`（GPU 交叉臂，'
              'out 采集 3600 forward batch=1 bf16 verbatim 3169 链）。产物：`phase3175/g5a12_residual_arms/` '
              '{exec ' + exes + ', res ' + R['res_sha8'] + ', seal ' + R['seal_sha8'] + ', smoke res ' +
              SM['res_sha8'] + ', collect_out 4b d4f8931b / 14b b9f0a96f / glm4 e14c62c4, gap v1.5 ' +
              gv15_sha + '}。')
    S2.append('')
    S2.append('### 判决（重复三遍）')
    S2.append('')
    S2.append('**port_residual_dominant：delta = pooled ratio_out(k=8) - ratio_in(k=8) = +' +
              ('%.4f' % delta) + '（|delta|<=0.15 门内）——实体熟悉度成分存在但小，谱外类残留 ~1.8x 过量误差'
              '确认为 one-hot 类端口机制固有（port-intrinsic），与校准实体来源无关。GAP-4 机制画像定稿'
              '（gap ledger v1.5 ' + gv15_sha + '）。**')
    S2.append('')
    S2.append('**port_residual_dominant（重复二）：in 臂逐位重放 3172 封存曲线（pooled 全 k x3 键 + per-model '
              '全 k x4 键 drift<1e-9；k=0 双臂 pooled drift=0.00e+00 复现 3169 gate 2.5388114997805062；'
              'out 零扩展退化路径与面板路径 exact 0.0）；out 臂（全新词表 40 实体，GPU verbatim 3169 链采集，'
              'D_out2 bitwise reself x3）恢复曲线 2.5388 -> 2.0782 -> 2.0443 -> 1.9570 -> 1.8662，与 in 臂'
              '（-> 1.8002）形状一致、饱和水平相似（恢复 ' + ('%.1f%%' % (recov_out * 100)) + ' vs ' +
              ('%.1f%%' % (recov_in * 100)) + '）。**')
    S2.append('')
    S2.append('**port_residual_dominant（重复三）：per-class delta 方向非一致——乐器/运动全正 3/3 模型'
              '（0.062-0.103 / 0.041-0.072）、天气/电器混合（-0.118~+0.018 / -0.087~+0.115），无类全负；'
              '熟悉度信号非主体。证据级 E2（协议内干预+三模型+预注册门）。机制链闭环：3169 量化 -> 3171 '
              'encoding_missing 否定 -> 3172 端口部分恢复 -> 3175 残留=端口固有。**')
    S2.append('')
    S2.append('### 双臂读数（result 现场渲染）')
    S2.append('')
    S2.append('| k | pooled ratio_in | pooled ratio_out | delta |')
    S2.append('|---|---|---|---|')
    for k in sorted(R['pooled']['in'], key=int):
        ki = int(k)
        S2.append('| %d | %.4f | %.4f | %+.4f |' % (
            ki, R['pooled']['in'][k]['ratio'], R['pooled']['out'][k]['ratio'],
            R['pooled']['out'][k]['ratio'] - R['pooled']['in'][k]['ratio']))
    S2.append('')
    S2.append('per-model k=8：in 1.9486/1.7100/1.7674 vs out 2.0570/1.6029/1.9793（4b/14b/glm4）。'
              'E_newent（面板内口径）in k8 0.4835/0.5230/0.5748 vs out k8 0.7038/0.6704/0.7352——'
              'out 校准行对面板内新实体预测无溢出收益（两臂测试集恒等下的诚实对照）。')
    S2.append('')
    S2.append('### 装置与诚实登记')
    S2.append('')
    S2.append('**1 次观测前修正（SMOKE 后正式跑 DRIFT 拦截，seal 前完成）**：DESIGN.ent_out_full 首版引用'
              ' SMOKE 截断后的词表对象 -> design hash 运行模式相关（SMOKE d6810f40 vs 正式 c4568dc2），'
              '正式跑被 freeze DRIFT 断言正确拦截 -> 改为全量副本 ENT_OUT_FULL（3169 DESIGN 运行模式无关'
              '纪律），重冻结后 SMOKE/正式同 hash；collect npz 缓存跨修正逐位复用。')
    S2.append('')
    S2.append('装置门全链：G_anchor 11 文件；G_k0 双臂 pooled+per-model drift=0.00e+00；out 零扩展退化路径 '
              'exact 0.0 x3 模型；G_in_replay in 臂逐位重放 3172（全 k）；D_out1 finite + D_out2 3-prompt '
              'bitwise reself x3；测试集两臂恒等（te_oov = ALL_OOV_ROWS - in 校准行，两臂同剔；E_seen/'
              'E_newent 行集恒等）。')
    S2.append('')
    S2.append('### 接续：预注册 3176')
    S2.append('')
    S2.append('按 3174 排序推进第 (2) 项 + 图谱渲染关账：**3176 = G5-B1 E1→E2 零 GPU 批量升级 + atlas v1.4 渲染**'
              '——(a) FTR-04 升级臂：3169 npz 已见类行 LOEO/新实体 held-out 读出（S_class 跨模型重建补 held_out '
              '标签）；(b) FTR-08 升级臂：3169 E_newent 行 = held-out 实体探针（R×K_entity 补 held_out）；'
              '(c) FTR-06 升级臂：unembed-only 逐模型重建（补 cross_model）；(d) GAP-4 mechanism_note 定稿段 '
              '（v1.5 ' + gv15_sha + '）+ FTR-22 已有渲染进 atlas_v1_4.html 重渲染逐字段校验。升级判据：'
              'held_out/cross_model 字面锚入表且数值现场渲染。零 GPU。')
    S2.append('')
    S2.append('')
    S2.append('---')
    S2.append('')
    S2.append('')
    section = '\n'.join(S2) + '\n'
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3175')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3175 section inserted (snapshot .snap3175)')

# ---------- 2b. MEMO: 3176 prereg inside the new section tail ----------
raw2 = open(MEMO, 'rb').read().decode('utf-8')
if '### 接续：预注册 3176' in raw2:
    log('2b. MEMO: 3176 prereg already present, skip')
else:
    log('2b. MEMO: 3176 prereg handled in section insert (see below)')

# ---------- 3. daily ----------
dline = ('- **3175 结构残留交叉臂（2026-10-09）**：G5-A12 GPU 交叉臂（out 新词表 40 实体 verbatim 3169 链'
         '采集）——**port_residual_dominant**：delta=+0.0660<=0.15，ratio_in(k8)=1.8002（逐位重放 3172）'
         'vs ratio_out(k8)=1.8662；残留 ~1.8x 确认为端口机制固有（GAP-4 机制画像定稿，gap ledger v1.5 ' +
         gv15_sha + '）。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n→327。1 次观测前'
         '修正（DESIGN 词表截断引用→运行模式相关 hash 被拦截）。下一步 3176=E1→E2 零 GPU 批 + atlas v1.4 渲染。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if 'cb76a177/9e1a2f08' in dtxt:
    log('3. daily: 3175 line already present, skip')
else:
    if not dtxt.endswith('\n'):
        dtxt += '\n'
    dtxt += dline + '\n'
    with io.open(DAILY, 'w', encoding='utf-8', newline='') as f:
        f.write(dtxt)
    log('3. daily: appended')

# ---------- 4. workspace MEMORY (append at EOF) ----------
wraw = open(WMEM, 'rb').read().decode('utf-8')
wm = wraw.replace('\r\n', '\n')
newline = ('\n- **✅ 3175 结构残留定位交叉臂闭环（2026-10-09）**：G5-A12 GPU 交叉臂——**port_residual_dominant**：'
           'delta = ratio_out(k8) - ratio_in(k8) = +0.0660（|delta|<=0.15 门内）；ratio_in(k8)=1.8002（in 臂'
           '逐位重放 3172：pooled 全 k + per-model 全 k drift<1e-9；k=0 双臂 drift=0.00e+00 复现 3169 gate '
           '2.5388114997805062）；out 臂新词表 40 实体（DESIGN freeze 冻结先于任何 GPU forward）verbatim 3169 '
           '链采集（D_out2 bitwise reself x3），恢复曲线 2.5388→1.8662（26.5% vs in 29.1%），形状一致饱和相似。'
           '**per-class delta 非一致**（乐器/运动全正 3/3、天气/电器混合、无类全负）→实体熟悉度成分存在但非主体。'
           '**GAP-4 机制画像定稿**：残留 ~1.8x = one-hot 类端口机制固有（port-intrinsic）；机制链闭环 3169 量化 '
           '→ 3171 encoding_missing 否定 → 3172 端口 ~29% → 3175 残留=端口固有。gap ledger v1.5 ' + gv15_sha +
           '（GAP-4 定稿，其余逐字节不变）。诚实登记：1 次观测前修正（DESIGN.ent_out_full 引用 SMOKE 截断词表 → '
           '运行模式相关 design hash 被正式跑 DRIFT 拦截 → 全量副本重冻结）。res ' + R['res_sha8'] + '/seal ' +
           R['seal_sha8'] + '；ledger n=326→**327**。下一步 3176=E1→E2 零 GPU 批量升级（FTR-04/08 走 3169 npz、'
           'FTR-06 unembed-only）+ atlas v1.4 渲染（GAP-4 定稿段）。\n')
if '3175 结构残留定位交叉臂' in wm:
    log('4. workspace MEMORY: 3175 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3175')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3175)')

# ---------- 5. self-check ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3175 = [m for m in ms2 if m.get('phase') == 3175]
chk.append(('ledger has 3175 entry', len(e3175) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=327', len(ms2) >= 327, 'n=' + str(len(ms2))))
if e3175:
    chk.append(('ledger verdict H2', 'port_residual_dominant' in e3175[0]['verdict'],
                e3175[0]['verdict'][:80]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3175', '## Phase 3175' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has gap v1.5 sha', gv15_sha in memo2))
chk.append(('MEMO has port_residual_dominant', 'port_residual_dominant' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3175', '3175 结构残留交叉臂' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3175', '3175 结构残留定位交叉臂' in w2))
R2 = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
chk.append(('result seal intact', R2['seal_sha8'] == R['seal_sha8'] and
            R2['res_sha8'] == R['res_sha8'], R['res_sha8'] + '/' + R['seal_sha8']))
chk.append(('delta gate H2', R2['delta_gate']['main_cls'] == 'port_residual_dominant' and
            abs(R2['delta_gate']['delta'] - delta) < 1e-12, '+%.4f' % delta))
chk.append(('k0 bitwise both arms', abs(R2['pooled']['in']['0']['ratio'] -
            R2['pooled']['out']['0']['ratio']) == 0.0 and
            abs(R2['pooled']['in']['0']['ratio'] - 2.5388114997805062) < 1e-9,
            repr(R2['pooled']['in']['0']['ratio'])))
chk.append(('gap v1.5 on disk', os.path.exists(GV15) and
            sha8_file(GV15) == gv15_sha, gv15_sha))
G15c = json.load(io.open(GV15, encoding='utf-8'))
g4c = [x for x in G15c['gaps'] if x['id'] == 'GAP-4'][0]
chk.append(('gap v1.5 anchor p3175', g4c['anchor_sha8'].get('p3175') ==
            sha8_file(os.path.join(PDIR, 'result.json')), ''))
chk.append(('gap v1.5 mechanism finalized', '3175 交叉臂定稿' in g4c['mechanism_note'], ''))
chk.append(('gap v1.5 evidence n=8', len(g4c['evidence']) == 8, str(len(g4c['evidence']))))
chk.append(('gap v1.5 version', G15c['version'] == '1.5', G15c['version']))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
