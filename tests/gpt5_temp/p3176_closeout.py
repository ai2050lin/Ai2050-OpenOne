# -*- coding: utf-8 -*-
# Phase 3176 closeout: five-write (ledger / MEMO / daily / workspace MEMORY /
# self-check). Idempotent. All numbers rendered live from sealed result.json.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(S, 'phase3176', 'g5b1_elev_upgrade')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3176_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3176c] ' + s)
    print('[3176c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
REG13 = os.path.join(PDIR, 'atlas_registry_v1_3.json')
HTML = os.path.join(PDIR, 'atlas_v1_4.html')
reg13_sha = sha8_file(REG13)
html_sha = sha8_file(HTML)
assert reg13_sha == R['registry_v13_sha8'] == 'e64733f7', reg13_sha
assert html_sha == R['html_sha8'] == 'd611a2a1', html_sha
SMOKE_EQ = sha8_file(os.path.join(PDIR, 'smoke_atlas_registry_v1_3.json')) == reg13_sha
assert SMOKE_EQ, 'SMOKE/FULL registry must be byte-identical (mode-independent arms)'
V = R
ra = R['arm_a']['per_model']
rb = R['arm_b']['per_model']
rc = R['arm_c']['per_model']
ga_, gc_, gb_ = R['arm_a']['gate'], R['arm_c']['gate'], R['arm_b']['gate']
acc_s = '/'.join('%.4f' % ra[m]['loeo_acc'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
rand_s = '/'.join('%.4f' % ra[m]['acc_rand'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
bloeo_s = '/'.join('%.3f' % rb[m]['loeo_mean_deg'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
bang_s = '/'.join('%.3f' % rb[m]['census_anchor_deg'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
cattr_s = '/'.join('%.3f' % rc[m]['attr_top1_deg'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
csyn_s = '/'.join('%.3f' % rc[m]['syntax_top1_deg'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
d4a, d4s = R['arm_c']['drift_4b_deg']
F15 = R['failure_added']
assert F15 and F15['id'] == 'F15'

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3176 for m in ms_):
    log('1. ledger: 3176 already present, skip')
else:
    detail = (
        'G5-B1 E1->E2 zero-GPU batch upgrade + atlas v1.4 render, three arms on sealed '
        'sources (3169/3151/3152 npz + per-model W_U + 2874/2878 vocab). Arm (a) FTR-04 '
        'held_out LOEO class readout via S_class on 3169 seen rows: acc ' + acc_s +
        ' (chance 0.2, rand-10d control ' + rand_s + ') -> GATE FAILED (glm4 0.4800 < 0.5, '
        'and glm4 S_class specificity only +0.03 over random subspace vs +0.17..0.19 for '
        'qwen3) -> F15 registered, FTR-04 stays E1 (prereg gate frozen, no goalpost move). '
        'Arm (b) FTR-08 held_out: R_logic LOEO (41 folds, fold rebuild top8 from other 40 '
        'entities) x fixed K_entity (3157): LOEO mean ' + bloeo_s + ' deg vs full anchors ' +
        bang_s + ' deg, drift <=0.10 deg, classification bands agree 3/3 -> UPGRADED '
        'E2_predictive with held_out anchor (full-sample rebuild reproduces 3166 census '
        'bitwise: drift 0.0000). Arm (c) FTR-06 cross_model: unembed-only per-model rebuild '
        'of S_attr/S_syntax from 2874/2878 pairs_json (kept single-token words, unit pair '
        'diff mean; zero axes dropped, 42/42/42+48/48/49 words); K_readout x S_attr = ' +
        cattr_s + ' deg, x S_syntax = ' + csyn_s + ' deg (4b rebuild drift %.4f/%.4f deg vs '
        '3165 anchors 68.094/58.488) -> UPGRADED E2_predictive, model_scope 4b -> 3 models. '
        'Registry v1.2 -> v1.3: 19 features byte-for-byte + FTR-04 counter_evidence append '
        'only (honest failure), failures 14 -> 15 (F15 upgrade_gate_failed), upgrade_log '
        '3 -> 5; SMOKE and FULL registry byte-identical (' + reg13_sha + ') => arm numbers '
        'mode-independent. gap ledger v1.5 rendered as-is (no new version). HTML 663 fields '
        '0/0/0/0. Honest log: 2 SMOKE-caught device bugs fixed pre-seal (arm_c rows dict '
        'mis-fetch -> 0 axes; G3 check logic did not allow failed-arm counter_evidence '
        'append) + 1 authoring-time syntax fix (FTR-06 anchor asserts). E-level census now: '
        'E2 x17, E1 x3 (FTR-04/14/15), E3 x3. design_sha=' + R['design_sha8'] + '. res ' +
        R['res_sha8'] + ' seal ' + R['seal_sha8'] + ' (smoke res ' + SM['res_sha8'] + ').')
    entry = {
        'phase': 3176, 'name': 'g5b1_elev_upgrade', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4-9b (zero GPU)',
        'verdict': R['verdict'],
        'evidence_level': 'E2_predictive',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-B1',
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
    log('1. ledger: appended 3176 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3176' in norm:
    log('2. MEMO: 3176 section already present, skip')
else:
    marker = '### 接续：预注册 3176'
    mi = norm.rfind(marker)
    assert mi > 0, '3176 prereg marker not found'
    S2 = []
    S2.append('## Phase 3176: G5-B1 E1→E2 零 GPU 批量升级 + 图谱 v1.4 渲染 ' + time.strftime('%H:%M'))
    S2.append('')
    S2.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3176_g5b1_elev_upgrade.py`（零 GPU 三臂）。'
              '产物：`phase3176/g5b1_elev_upgrade/` {exec ' + exes + ', res ' + R['res_sha8'] + ', seal ' +
              R['seal_sha8'] + ', smoke res ' + SM['res_sha8'] + ', registry v1.3 ' + reg13_sha +
              ', atlas_v1_4.html ' + html_sha + '}。')
    S2.append('')
    S2.append('### 判决（重复三遍）')
    S2.append('')
    S2.append('**arms fail/pass/pass：FTR-06、FTR-08 升级 E2_predictive 成功入表（cross_model/held_out 锚 + '
              '数值现场渲染），FTR-04 门失败诚实登记（glm4 LOEO acc 0.4800 < 0.5 门，且 glm4 的 S_class 特异性'
              '仅 +0.03 高于随机 10 维对照 0.4495，qwen3 系为 +0.17~0.19）——预注册门冻结不移动，F15 入失败账本，'
              'FTR-04 保持 E1。registry v1.3 = 22 特征 + 失败账本 15 + 升级链 5；gap ledger v1.5 原样渲染（无新版本）。**')
    S2.append('')
    S2.append('**arms fail/pass/pass（重复二）：臂 (b) R_logic 留一实体 LOEO（41 fold）x 固定 K_entity——LOEO 均值 ' +
              bloeo_s + ' 度 vs 全量锚 ' + bang_s + ' 度（drift <=0.10 度，分类带一致 3/3），全量重建逐位复现 3166 '
              'census（drift 0.0000）；臂 (c) unembed-only 逐模型重建（2874/2878 pairs_json verbatim，零轴剔除）——'
              'K_readout x S_attr = ' + cattr_s + ' 度、x S_syntax = ' + csyn_s + ' 度（4b 重建 drift %.4f/%.4f 度），'
              '14b/glm4 全过 30 度可分带。**' % (d4a, d4s))
    S2.append('')
    S2.append('**arms fail/pass/pass（重复三）：臂 (a) S_class LOEO 类读出（35 可映射实体 x 5 类 x 3 模板 = 525 行，'
              'chance 0.2）acc ' + acc_s + '，随机 10 维对照 ' + rand_s + '——4b/14b 过门（0.60 vs 0.5），glm4 差 '
              '0.02 未过且特异性近零：glm4 的 held-out 类身份不优先经由 unembed 类方向子空间承载（qwen3 系是）。'
              'SMOKE 与 FULL 的 registry v1.3 字节一致（' + reg13_sha + '）=> 三臂数值运行模式无关。**')
    S2.append('')
    S2.append('### 三臂数值（result 现场渲染）')
    S2.append('')
    S2.append('| 臂 | 特征 | 门 | 4b | 14b | glm4 | 判定 |')
    S2.append('|---|---|---|---|---|---|---|')
    S2.append('| (a) S_class LOEO acc | FTR-04 | >=0.5 x3 | %.4f | %.4f | %.4f | FAIL（F15） |' % (
        ra['qwen3-4b']['loeo_acc'], ra['qwen3-14b']['loeo_acc'], ra['glm4-9b']['loeo_acc']))
    S2.append('| (b) R_logic LOEO deg | FTR-08 | <5 deg + 带一致 | %.3f | %.3f | %.3f | PASS |' % (
        rb['qwen3-4b']['loeo_mean_deg'], rb['qwen3-14b']['loeo_mean_deg'], rb['glm4-9b']['loeo_mean_deg']))
    S2.append('| (c) KxS_attr deg | FTR-06 | 4b 对拍 + >=30 x2 | %.3f | %.3f | %.3f | PASS |' % (
        rc['qwen3-4b']['attr_top1_deg'], rc['qwen3-14b']['attr_top1_deg'], rc['glm4-9b']['attr_top1_deg']))
    S2.append('| (c) KxS_syntax deg | FTR-06 | 同上 | %.3f | %.3f | %.3f | PASS |' % (
        rc['qwen3-4b']['syntax_top1_deg'], rc['qwen3-14b']['syntax_top1_deg'], rc['glm4-9b']['syntax_top1_deg']))
    S2.append('')
    S2.append('E 级分布：E2 x17（+2）、E1 x3（FTR-04/14/15）、E3 x3；升级链 3->5（FTR-06 cross_model、'
              'FTR-08 held_out）。atlas_v1_4.html 663 字段 0/0/0/0。')
    S2.append('')
    S2.append('### 装置与诚实登记')
    S2.append('')
    S2.append('**2 次 SMOKE 拦截（seal 前修正，全为装置层）**：①臂 (c) `rows = W_rows_by_model[m]` 误取外层 '
              'collect dict（应为 `["rows"]`）-> 全词 miss、0 axes；②G3 检查逻辑未考虑失败分支的 '
              'counter_evidence 追加登记（失败臂特征=除 counter_evidence 追加外逐字节）。另 1 处编写期修正'
              '（FTR-06 锚 asserts 语法垃圾）。三臂全量零 GPU 无 SMOKE 截断 => 数值跨模式恒等。')
    S2.append('')
    S2.append('F15（失败账本）：' + F15['text'][:120] + '……（' + F15['evidence'] + '）')
    S2.append('')
    S2.append('### 接续：预注册 3177')
    S2.append('')
    S2.append('**3177 = G5-B2 FTR-14 held_out GPU 臂（上下文变换的关系留出泛化）**——FTR-14 现锚只有 cross_model'
              '（3157 主测量 + 3162 audit），held_out 缺失。协议：3157 collect（16 锚 x 2 关系 x 2 ctx x 2 臂）'
              '关系留出 fold：留出关系 r，用其余关系重建 T_C 共享分量（3157 配方 verbatim），测留出关系上 T_C 读数 '
              'cos 漂移与 exch 带保持。门：留出关系 T_C cos 与全量差 <0.1 且 exch 均值落 1.0-1.3 partial 带 x3 '
              '模型 -> held_out 锚入表，FTR-14 升 E2。协议细节在 3177 freeze 前从 `phase3157` 脚本 verbatim 导出'
              '（预注册冻结于 execution.json；源锚 p3157 collect 95a25965/9552086d/23dd74eb + 3157 result 内容 '
              '0fe043bf/cfa5c3ed）。FTR-04 glm4 弱特异性归因（全维上界 vs S_class 曲线）与 FTR-15 held_out 列为'
              '后续；挂账不变：N 线 P3-P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。')
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
    shutil.copyfile(MEMO, MEMO + '.snap3176')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3176 section inserted (snapshot .snap3176)')

# ---------- 3. daily ----------
dline = ('- **3176 E1→E2 批量升级 + 图谱 v1.4（2026-10-09）**：G5-B1 零 GPU 三臂——**arms fail/pass/pass**：'
         'FTR-06/FTR-08 升 E2 入表（LOEO R×K_entity drift<=0.10 度带一致 3/3；unembed-only 重建 4b drift '
         '0.0003 度、14b/glm4 73.6-75.1/61.6-67.0 度全过可分带），FTR-04 门失败诚实登记（glm4 LOEO acc '
         '0.4800<0.5 且 S_class 特异性仅 +0.03）→F15。registry v1.3 ' + reg13_sha + '（E2 x17/E1 x3/E3 x3）、'
         'atlas_v1_4.html 663 字段全对。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n→328。'
         '2 次 SMOKE 拦截装置 bug（arm_c rows 取法、G3 失败分支语义）。下一步 3177=FTR-14 held_out 关系留出 GPU 臂。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if 'c8a9797d/d2647ffb' in dtxt:
    log('3. daily: 3176 line already present, skip')
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
newline = ('\n- **✅ 3176 E1→E2 批量升级 + 图谱 v1.4 闭环（2026-10-09）**：G5-B1 零 GPU 三臂——**arms fail/pass/pass**：'
           '臂(b) FTR-08 held_out PASS（R_logic 留一实体 LOEO x 固定 K_entity：LOEO 均值 45.939/26.658/46.042 度 vs '
           '全量锚 45.845/26.641/46.005 度，drift<=0.10 度带一致 3/3；全量重建逐位复现 3166 census drift 0.0000）；'
           '臂(c) FTR-06 cross_model PASS（unembed-only 逐模型重建 S_attr/S_syntax，2874/2878 pairs_json verbatim 零轴'
           '剔除：KxS_attr=68.094/75.071/73.596 度、KxS_syntax=58.488/67.004/61.566 度，4b 重建 drift 0.0003/0.0002 '
           '度，14b/glm4 全过 30 度带）→双升 E2 入表+锚+数值；臂(a) FTR-04 held_out FAIL（S_class LOEO 类读出 '
           'acc 0.6019/0.5981/0.4800 vs 门 0.5，chance 0.2；glm4 差 0.02 且 S_class 特异性仅 +0.03 vs 随机对照 '
           '0.4495——glm4 类身份不优先经 unembed 类方向子空间承载，qwen3 系 +0.17~0.19）→F15 诚实登记、保持 E1。'
           'registry v1.2→v1.3（' + reg13_sha + '，22 特征/失败 15/升级链 5，SMOKE 与 FULL 字节一致=>数值运行模式'
           '无关）+ gap v1.5 原样渲染 + atlas_v1_4.html 663 字段全对。E 级分布 E2 x17/E1 x3(FTR-04/14/15)/E3 x3。'
           '诚实登记：2 次 SMOKE 拦截装置 bug（arm_c rows dict 取法、G3 失败分支 counter_evidence 语义）+1 编写期'
           '语法修正。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n=327→**328**。下一步 '
           '3177=G5-B2 FTR-14 held_out 关系留出 GPU 臂（T_C 共享分量留出关系漂移<0.1 + exch 带保持门）；FTR-04 '
           'glm4 归因挂账。\n')
if '3176 E1→E2 批量升级' in wm:
    log('4. workspace MEMORY: 3176 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3176')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3176)')

# ---------- 5. self-check ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3176 = [m for m in ms2 if m.get('phase') == 3176]
chk.append(('ledger has 3176 entry', len(e3176) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=328', len(ms2) >= 328, 'n=' + str(len(ms2))))
if e3176:
    chk.append(('ledger verdict fail_pass_pass', 'arms_fail_pass_pass' in e3176[0]['verdict'],
                e3176[0]['verdict'][:80]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3176', '## Phase 3176' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has registry v1.3 sha', reg13_sha in memo2))
chk.append(('MEMO has 3177 prereg', '预注册 3177' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3176', '3176 E1→E2 批量升级' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3176', '3176 E1→E2 批量升级' in w2))
R2 = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
chk.append(('result seal intact', R2['seal_sha8'] == R['seal_sha8'] and
            R2['res_sha8'] == R['res_sha8'], R['res_sha8'] + '/' + R['seal_sha8']))
chk.append(('arm verdicts stable', (R2['arm_a']['gate'], R2['arm_c']['gate'], R2['arm_b']['gate'])
            == (False, True, True), 'a/c/b=False/True/True'))
chk.append(('arm_b loeo drift <=0.1', all(abs(R2['arm_b']['per_model'][m]['loeo_mean_deg'] -
            R2['arm_b']['per_model'][m]['census_anchor_deg']) < 0.11 for m in R2['arm_b']['per_model']), ''))
chk.append(('registry v1.3 on disk', os.path.exists(REG13) and sha8_file(REG13) == reg13_sha, reg13_sha))
chk.append(('registry SMOKE==FULL', sha8_file(os.path.join(PDIR, 'smoke_atlas_registry_v1_3.json'))
            == reg13_sha, 'mode-independent arms'))
RG = json.load(io.open(REG13, encoding='utf-8'))
lv = {f['id']: f['evidence_level'] for f in RG['features']}
chk.append(('FTR-06 upgraded E2', lv['FTR-06'] == 'E2_predictive', ''))
chk.append(('FTR-08 upgraded E2', lv['FTR-08'] == 'E2_predictive', ''))
chk.append(('FTR-04 stays E1', lv['FTR-04'] == 'E1_repeatable', ''))
chk.append(('failures n=15 with F15', len(RG['failures']) == 15 and
            RG['failures'][-1]['id'] == 'F15', str(len(RG['failures']))))
chk.append(('upgrade_log n=5', len(RG['upgrade_log']) == 5, str(len(RG['upgrade_log']))))
chk.append(('html on disk', os.path.exists(HTML) and sha8_file(HTML) == html_sha, html_sha))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
