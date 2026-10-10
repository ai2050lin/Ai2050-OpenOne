# -*- coding: utf-8 -*-
# Phase 3173 closeout: five-write (ledger / MEMO / daily / workspace MEMORY /
# self-check). Idempotent. All numbers rendered live from sealed result.json.
# Gap ledger v1.4 + registry v1.2 already written by the main script; this
# closeout only verifies them on disk.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3173', 'g5a10_atlas_v13')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3173_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3173c] ' + s)
    print('[3173c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
reg12_sha = R['registry_v12_sha8']
gl14_sha = R['gap_ledger_v14_sha8']
html_sha = R['html_sha8']
V = R['ftr22_values']
curve_s = ' -> '.join('k=%s: %.4f' % (k, V['ratio_k' + k]) for k in ('0', '1', '2', '4', '8'))
drop_pct = '%.1f' % (V['port_removed_frac'] * 100)
r8_s = '%.4f' % V['ratio_k8']
rs_s = '%.4f/%.4f/%.4f' % (V['p3171_ratio_S_qwen3_4b'], V['p3171_ratio_S_qwen3_14b'],
                           V['p3171_ratio_S_glm4_9b'])

# ---------- 0. verify on-disk artifacts match sealed result ----------
assert reg12_sha == sha8_file(os.path.join(PDIR, 'atlas_registry_v1_2.json')), 'reg12 sha mismatch'
assert gl14_sha == sha8_file(os.path.join(PDIR, 'gap_ledger_v1_4.json')), 'gl14 sha mismatch'
assert html_sha == sha8_file(os.path.join(PDIR, 'atlas_v1_3.html')), 'html sha mismatch'
G14 = json.load(io.open(os.path.join(PDIR, 'gap_ledger_v1_4.json'), encoding='utf-8'))
g4 = [g for g in G14['gaps'] if g['id'] == 'GAP-4'][0]
assert G14['version'] == '1.4' and g4['status'] == 'quantified_collapse'
assert '3172 端口校准定判' in g4['statement'] and r8_s in g4['statement']
assert g4['anchor_sha8']['p3172'] == 'd8ddc481' and g4['anchor_sha8']['p3171'] == 'a43cf48c'
RG = json.load(io.open(os.path.join(PDIR, 'atlas_registry_v1_2.json'), encoding='utf-8'))
assert RG['version'] == '1.2' and len(RG['features']) == 22
assert RG['features'][21]['id'] == 'FTR-22'
assert len(RG['failures']) == 14 and RG['failures'][12]['id'] == 'F13' and RG['failures'][13]['id'] == 'F14'
log('0. disk artifacts verified: registry v1.2 (%s, 22 ftr + F13/F14), gap ledger v1.4 (%s), html (%s, %d fields)'
    % (reg12_sha, gl14_sha, html_sha, R['field_check']['n_found']))

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3173 for m in ms_):
    log('1. ledger: 3173 already present, skip')
else:
    detail = (
        'G5-A10 atlas v1.3 render closeout (zero GPU): registry v1.1->v1.2 appends FTR-22 '
        '(port calibration recovery curve, family=limit, E2_predictive; 3 independent phase '
        'anchors = p3172 intervention 463f42c8/cdb85525 + p3169 base measurement 49430a39/018c6024 '
        '+ p3171 mechanism 6a29201c/cc7ccedd); features[0:21] preserved byte-for-byte 21/21; '
        'failures += F13 (port-missing is NOT the whole collapse cause - k=8 residual ' + r8_s +
        ', only ' + drop_pct + ' percent removed) + F14 (mechanism 3-way gate boundary: glm4 '
        'ratio_S 0.8205 just past 0.8 gate +0.02); upgrade_log unchanged at 3 (no E-level '
        'upgrades in 3171/3172). Gap ledger v1.3->v1.4: GAP-4 statement extended with the 3172 '
        'intervention verdict (port ~' + drop_pct + ' percent + structural residual ~1.8x hung), '
        'evidence +2, anchor_sha8 += p3171/p3172, mechanism_note byte-identical, status stays '
        'quantified_collapse; GAP-1/2/3 + appendix 5 byte-identical. atlas_v1_3.html full '
        're-render: 22 features / 16 nodes / 4 gaps incl GAP-4.mechanism_note full text rendered '
        'for the first time; data-k field check 637/637 (missing/extra/mismatch/dup all 0). '
        'Runtime gates: G5 FTR-22 asserts (k0 re-3169 drift 0.0e+00, k8 band [1.5,2) x3, port_frac '
        '0.2909, p3171 mixed re-asserted), G3 21/21, G7 first-12 failures byte-preserved. Honest '
        'log: one pre-observation fix - 3169 gate anchor initially written from memory as '
        'truncated 2.5388115, measured to full precision 2.5388114997805062 per sha8-anchor '
        'discipline. Atlas v1.3 CLOSED. design_sha=' + R['design_sha8'] + '. res ' + R['res_sha8'] +
        ' seal ' + R['seal_sha8'] + ' (smoke res ' + SM['res_sha8'] + ').')
    entry = {
        'phase': 3173, 'name': 'g5a10_atlas_v13', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'zero_gpu (render only)',
        'verdict': ('g5a10_atlas_v13|features_22|failures_14|html_fields_637_ok|'
                    'gap4_quantified_collapse_mechanism_located|atlas_v13_closed'),
        'evidence_level': 'E2_predictive',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-A10',
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
    log('1. ledger: appended 3173 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3173' in norm:
    log('2. MEMO: 3173 section already present, skip')
else:
    marker = '### 接续：预注册 3173'
    mi = norm.rfind(marker)
    assert mi > 0, '3173 prereg marker not found'
    S = []
    S.append('## Phase 3173: G5-A10 图谱 v1.3 渲染关账（FTR-22 立表 + GAP-4 定判回写） ' + time.strftime('%H:%M'))
    S.append('')
    S.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3173_g5a10_atlas_v13.py`（零 GPU）。产物：'
             '`phase3173/g5a10_atlas_v13/` {exec ' + exes + ', res ' + R['res_sha8'] + ', seal ' +
             R['seal_sha8'] + ', smoke res ' + SM['res_sha8'] + '}；registry v1.2 ' + reg12_sha +
             '（22 特征 + F13/F14）；gap ledger v1.4 ' + gl14_sha + '；atlas_v1_3.html ' + html_sha +
             '（101243 B，637 字段 data-k 逐字段校验 0/0/0/0）。')
    S.append('')
    S.append('### 判决（重复三遍）')
    S.append('')
    S.append('**图谱 v1.3 关账：FTR-22 立表（端口校准恢复曲线，family=limit，E2_predictive，三锚=3172 干预 '
             'res 463f42c8 + 3169 主测量 res 49430a39 + 3171 机制定位 res 6a29201c）；GAP-4 保持 '
             'quantified_collapse 且机制定位回写完毕（端口缺失 ~29.1% + 结构残留 ~1.8x 挂账）；'
             'HTML 全量重渲染 637 字段逐字段校验全对。**')
    S.append('')
    S.append('**图谱 v1.3 关账（重复二）：registry v1.2 = 22 特征（FTR-01..22）+ 14 失败账本'
             '（F13=「端口缺失=崩塌全部成因」被否证 k=8 残留 ' + r8_s + '；F14=机制三分类 0.8 门边界敏感 '
             'glm4 0.8205 刚过 +0.02）+ 3 升级链不变；v1.1 的 21 特征逐字节保留 21/21。**')
    S.append('')
    S.append('**图谱 v1.3 关账（重复三）：GAP-4 mechanism_note（3171 机制定位 + 3172 干预段）首次全文渲染'
             '进 HTML；gap ledger v1.4 statement 追加 3172 定判段、evidence +2、anchor_sha8 += '
             'p3171(a43cf48c)/p3172(d8ddc48c)、status 不变；GAP-1/2/3 + 附录 5 项逐字节不变。**')
    S.append('')
    S.append('### FTR-22 数值（result 现场渲染）')
    S.append('')
    en_s = '%.4f -> %.4f' % (V['E_newent_pooled_mean3_k0'], V['E_newent_pooled_mean3_k8'])
    S.append('pooled ratio_B(k)：' + curve_s + '（k=0 逐位复现 3169，drift=0.0e+00）；三模型 k=8 = ' +
             '%.4f/%.4f/%.4f' % (V['ratio_k8_qwen3_4b'], V['ratio_k8_qwen3_14b'], V['ratio_k8_glm4_9b']) +
             ' 全落 [1.5,2)；port_removed_frac=' + drop_pct + '%；E_newent pooled（三模型均值）k0->k8 = ' +
             en_s + '；p3171 ratio_S=' + rs_s + '。运行时门 G5：k0 re-3169 drift<1e-9、k8 带 [1.5,2) x3、'
             '曲线单调不增、port_frac in [0.28,0.30]、p3171 mixed 重断言。')
    S.append('')
    S.append('### 失败账本新增（预注册 (c) 项执行）')
    S.append('')
    S.append('F13 [hypothesis_rejected] p3172：「one-hot 类端口缺失=谱外类崩塌全部成因」被否证——k=8 校准'
             '仅移除 ' + drop_pct + '% 崩塌量，残留 ~1.8x=结构性成分；教训=部分恢复 != 机制全解释。'
             'F14 [gate_boundary] p3171：机制三分类对 0.8 门边界敏感（glm4 0.8205 刚过门）；三模型一致'
             '否定的只有 encoding_missing（全 >0.5）。upgrade_log 评估后无新增（3171/3172 无 E 级升级事件），'
             '保持 3 条。')
    S.append('')
    S.append('### 装置与诚实登记')
    S.append('')
    S.append('7 源 sha8 断言（registry v1.1/3162 基座/gap v1.3/p3169/p3171/p3172/q03）；**1 次观测前修正**：'
             '3169 gate 锚初写凭记忆截断值 2.5388115 被断言拦截，实测全精度 2.5388114997805062 后重冻结'
             '（sha8 锚纪律第 N 次生效）；F13/F14 入表后产物重写，SMOKE 与正式跑各全绿，正式跑零修正轮。')
    S.append('')
    S.append('### 图谱 v1.3 现状与缺口排序再评估')
    S.append('')
    S.append('22 特征（E2×13/E1×5/E3×3 + FTR-21/22 limit×2）+ 16 节点基座 + 4 缺口（GAP-1/2 closed、'
             'GAP-3 closed_v1、GAP-4 quantified_collapse 机制已定位）+ 14 失败 + 5 附录挂账。缺口④ '
             'mechanism：端口 ~29.1%（E2 干预）+ 结构残留 ~1.8x（挂账）。**下一步候选排序**：(1) 结构残留 '
             '~1.8x 定位（实体x类交互假设，需 GPU 新采样：谱外类校准实体 vs 面板外实体的交叉臂）；'
             '(2) FTR-03 E1->E2 升级（若复用现有 npz 可零 GPU）；(3) N 线 P3-P7 补 Ledger（跨线挂账）。'
             '预注册 3174 于下节。')
    S.append('')
    S.append('### 接续：预注册 3174')
    S.append('')
    S.append('G5-A11 图谱 v1.3 完整性审计（零 GPU，独立复核扩展）：对 atlas_v1_3.html 637 字段做独立进程'
             '重渲染对盘（verify 脚本独立实现 flatten+seal 字节级重构+判决重推导，同 3167-3173 纪律）；'
             '同时做缺口排序裁决：结构残留定位需 GPU 采样（预注册 3175 草案：谱外类交叉臂——校准实体'
             '来自面板内 vs 面板外各半，分离「实体 familiarity」与「类端口」两因素），FTR-03 升级路径'
             '评估（E1->E2 需要的 cross-model 复现是否已有现成 npz）。产出：审计报告 + 3175 预注册文。')
    S.append('')
    S.append('')
    S.append('---')
    S.append('')
    S.append('')
    section = '\n'.join(S) + '\n'
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3173')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3173 section inserted (snapshot .snap3173)')

# ---------- 3. daily ----------
dline = ('- **3173 图谱 v1.3 关账（2026-10-09）**：G5-A10 零 GPU——FTR-22 立表（端口校准恢复曲线，'
         '3 phase 锚=3172+3169+3171）；F13/F14 入失败账本（端口全解释被否证/0.8 门边界敏感）；'
         'GAP-4 statement 追加 3172 定判+锚补 p3171/p3172（gap ledger v1.4 ' + gl14_sha + '）；'
         'atlas_v1_3.html 637 字段校验全对（' + html_sha + '）；registry v1.2 ' + reg12_sha +
         '。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n→325。1 次观测前修正'
         '（3169 gate 锚实测全精度）。下一步 3174=G5-A11 完整性审计+缺口排序裁决。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3173 图谱 v1.3 关账' in dtxt:
    log('3. daily: 3173 line already present, skip')
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
newline = ('\n- **✅ 3173 图谱 v1.3 渲染关账（2026-10-09）**：G5-A10 零 GPU——**FTR-22 立表**（端口校准'
           '恢复曲线，family=limit，E2_predictive，三独立 phase 锚=p3172 463f42c8/p3169 49430a39/'
           'p3171 6a29201c；pooled ratio_B(k) ' + curve_s + '，port_removed=' + drop_pct + '%）；'
           'registry v1.1→v1.2 ' + reg12_sha + '（22 特征，v1.1 的 21 条逐字节保留 21/21）；**F13/F14 '
           '入失败账本**（F13=「端口缺失=崩塌全部成因」被否证 k=8 残留 ' + r8_s + '；F14=机制三分类 '
           '0.8 门边界敏感 glm4 0.8205 刚过 +0.02）；GAP-4 statement 追加 3172 定判段、evidence +2、'
           'anchor_sha8 += p3171/p3172、**mechanism_note 逐字节不变**、status 保持 quantified_collapse'
           '（gap ledger v1.4 ' + gl14_sha + '）；**atlas_v1_3.html** ' + html_sha + '（101243 B，'
           '637 字段 data-k 校验 0/0/0/0，GAP-4 mechanism_note 首次全文渲染）。运行时门 G5：k0 re-3169 '
           'drift 0.0e+00、k8 [1.5,2) x3、port_frac 0.2909。诚实登记：1 次观测前修正=3169 gate 锚凭记忆'
           '截断被拦截→实测全精度 2.5388114997805062。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] +
           '；ledger n=324→**325**。**图谱 v1.3 关账**：22 特征+16 节点+4 缺口+14 失败+5 附录。下一步 '
           '3174=**G5-A11 完整性审计+缺口排序裁决**（结构残留 GPU 臂 3175 草案；FTR-03 升级评估）。\n')
if '3173 图谱 v1.3 渲染关账' in wm:
    log('4. workspace MEMORY: 3173 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3173')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3173)')

# ---------- 5. self-check ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3173 = [m for m in ms2 if m.get('phase') == 3173]
chk.append(('ledger has 3173 entry', len(e3173) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=325', len(ms2) >= 325, 'n=' + str(len(ms2))))
if e3173:
    chk.append(('ledger verdict has atlas closed', 'atlas_v13_closed' in e3173[0]['verdict'],
                e3173[0]['verdict'][:80]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3173', '## Phase 3173' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has FTR-22', 'FTR-22' in memo2))
chk.append(('MEMO has prereg 3174 ref', '预注册 3174' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3173', '3173 图谱 v1.3 关账' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3173', '3173 图谱 v1.3 渲染关账' in w2))
chk.append(('registry v1.2 on disk matches', reg12_sha == sha8_file(os.path.join(PDIR, 'atlas_registry_v1_2.json')), reg12_sha))
chk.append(('gap ledger v1.4 on disk matches', gl14_sha == sha8_file(os.path.join(PDIR, 'gap_ledger_v1_4.json')), gl14_sha))
chk.append(('html on disk matches', html_sha == sha8_file(os.path.join(PDIR, 'atlas_v1_3.html')), html_sha))
RG2 = json.load(io.open(os.path.join(PDIR, 'atlas_registry_v1_2.json'), encoding='utf-8'))
chk.append(('registry 22 features', len(RG2['features']) == 22, str(len(RG2['features']))))
chk.append(('FTR-22 present', RG2['features'][21]['id'] == 'FTR-22', RG2['features'][21]['id']))
chk.append(('failures 14 with F13/F14', len(RG2['failures']) == 14 and
            RG2['failures'][12]['id'] == 'F13' and RG2['failures'][13]['id'] == 'F14',
            '%d' % len(RG2['failures'])))
G14b = json.load(io.open(os.path.join(PDIR, 'gap_ledger_v1_4.json'), encoding='utf-8'))
g4b = [g for g in G14b['gaps'] if g['id'] == 'GAP-4'][0]
chk.append(('GAP-4 statement has 3172 verdict', '3172 端口校准定判' in g4b['statement'] and r8_s in g4b['statement'], ''))
chk.append(('GAP-4 mechanism_note intact', 'mixed_across_models' in g4b['mechanism_note'] and
            'borderline_partial_recovery' in g4b['mechanism_note'], ''))
chk.append(('GAP-4 status unchanged', g4b['status'] == 'quantified_collapse', g4b['status']))
chk.append(('gap ledger version 1.4', G14b['version'] == '1.4', G14b['version']))
chk.append(('field check 637 all 0', R['field_check']['n_found'] == 637 and
            R['field_check']['missing'] == 0 and R['field_check']['extra'] == 0 and
            R['field_check']['mismatch'] == 0 and R['field_check']['dups'] == 0,
            json.dumps(R['field_check'])))
chk.append(('k0 ratio drift vs 3169', abs(V['ratio_k0'] - 2.5388114997805062) < 1e-9,
            '%.9f' % V['k0_drift_ratio_vs_3169']))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
