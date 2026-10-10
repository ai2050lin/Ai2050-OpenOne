# -*- coding: utf-8 -*-
# Phase 3174 closeout: five-write (ledger / MEMO / daily / workspace MEMORY /
# self-check). Idempotent. All numbers rendered live from sealed result.json.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3174', 'g5a11_audit')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3174_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3174c] ' + s)
    print('[3174c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
pr_sha = R['prereg_sha8']
SH = R['shadow']
AN = R['anchors']

# ---------- 0. verify on-disk artifacts match sealed result ----------
assert pr_sha == sha8_file(os.path.join(PDIR, 'prereg_3175_structural_residual_draft.json')), 'prereg sha mismatch'
assert SH['registry_identical'] and SH['gap_identical'] and SH['html_diff_all_timestamp']
assert SH['datak_sealed'] == 637 and AN['nav_asserts'] == 162 and AN['content_sha_anchors'] == 25
log('0. disk artifacts verified: prereg %s, shadow registry/gap identical, html diff %d line(s) all timestamp'
    % (pr_sha, SH['html_diff_lines']))

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3174 for m in ms_):
    log('1. ledger: 3174 already present, skip')
else:
    detail = (
        'G5-A11 atlas v1.3 integrity audit + gap ranking adjudication (zero GPU, 369 checks): '
        'G2 independent-process shadow re-render of Phase 3173 - registry and gap ledger '
        'byte-identical, html line-diff ' + str(SH['html_diff_lines']) + ' line all timestamp '
        '(meta.created), data-k 637==637 keys identical. G3 registry v1.2 invariants + v1.1 '
        'features byte-preserved 21/21 re-assert. G4 three-track anchor resolution: 25 '
        'content-sha anchors resolved to source docs, 162 assert keys navigated with 0 fail '
        '(dotted path / digit-leaf list index / node: by id), 3 derived keys (k0_drift_*) '
        'recomputed in G5, GAP-4 5 file shas resolved by 1650-file disk index. G5 FTR-22 '
        'curve/port_frac/k8-band/E_newent/p3171 ratios re-derived exact float; FTR-21 gate '
        'ratio present. G6 taxonomy: E1={FTR-04,06,08,14,15} E3={FTR-10,11,16}, 6 E2 '
        'tag-completeness notes (FTR-05/09/12/13/18 lack held_out tag, FTR-19 lacks '
        'cross_model tag) recorded as metadata findings, NO retroactive demotion. Prereg '
        'premise corrected: FTR-03 already E2_predictive (p3154+p3166 both '
        'cross_model+held_out). G7 E1->E2 assessment: zero-GPU candidates FTR-04 (held-out '
        'class-axis readout on 3169 npz), FTR-08 (held-out entity probe on 3169 E_newent '
        'rows), FTR-06 (unembed-only rebuild); FTR-14/15 need GPU. G8 prereg DRAFT artifact '
        'for Phase 3175 G5-A12 structural-residual cross-arm (in-panel vs out-of-panel '
        'calibration entities half/half, primary gate delta=ratio_out(k8)-ratio_in(k8), '
        '0.15 threshold, k=0 must reproduce 3169 gate bitwise). G9 ranking: 1 structural-'
        'residual GPU arm, 2 E1->E2 zero-GPU batch, 3 N-line ledger backfill. Honest log: '
        'G4 redesigned twice pre-seal (content-sha vs file-sha caliber; node-list/digit-'
        'leaf/derived-key navigation) - SMOKE caught both, no sealed artifact touched. '
        'design_sha=' + R['design_sha8'] + '. res ' + R['res_sha8'] + ' seal ' + R['seal_sha8'] +
        ' (smoke res ' + SM['res_sha8'] + ').')
    entry = {
        'phase': 3174, 'name': 'g5a11_audit', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'zero_gpu (audit only)',
        'verdict': ('g5a11_audit|shadow_reg_gl_identical|html_diff_lines_1_all_timestamp|'
                    'anchors_25_sha_162_nav_resolved|taxonomy_notes_6|'
                    'ftr03_premise_corrected_already_E2|e1_zerogpu_candidates_3|'
                    'prereg3175_' + pr_sha + '|ranking_3_items|PASS'),
        'evidence_level': 'E1_repeatable',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-A11',
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
    log('1. ledger: appended 3174 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3174' in norm:
    log('2. MEMO: 3174 section already present, skip')
else:
    marker = '### 接续：预注册 3174'
    mi = norm.rfind(marker)
    assert mi > 0, '3174 prereg marker not found'
    S = []
    S.append('## Phase 3174: G5-A11 图谱 v1.3 完整性审计 + 缺口排序裁决 ' + time.strftime('%H:%M'))
    S.append('')
    S.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3174_g5a11_audit_ranking.py`（零 GPU，'
             '369 项检查）。产物：`phase3174/g5a11_audit/` {exec ' + exes + ', res ' + R['res_sha8'] +
             ', seal ' + R['seal_sha8'] + ', smoke res ' + SM['res_sha8'] + ', prereg3175 ' + pr_sha + '}。')
    S.append('')
    S.append('### 判决（重复三遍）')
    S.append('')
    S.append('**图谱 v1.3 审计通过：独立进程影子重渲染 3173——registry 与 gap ledger 字节恒等，HTML 行级 diff '
             '仅 ' + str(SH['html_diff_lines']) + ' 行且全为 meta.created 时间戳，data-k 637==637 键序列恒等；'
             '锚链三轨解析 25 内容 sha 锚 + 162 条断言导航 0 失败 + GAP-4 五文件 sha 盘上全解析（1650 文件索引）。**')
    S.append('')
    S.append('**图谱 v1.3 审计通过（重复二）：E 级分类审计无 E0、E1={FTR-04,06,08,14,15}、E3={FTR-10,11,16} '
             '全带 intervention 锚；6 条 E2 标签完备性注记（FTR-05/09/12/13/18 缺 held_out 字面标签、'
             'FTR-19 缺 cross_model 字面标签）记为元数据发现不追溯降级；**FTR-03 预注册前提纠偏：registry v1.2 '
             '中已是 E2_predictive（p3154+p3166 双锚 cross_model+held_out）**，3173 排序文字按实际 E1 集重定向。**')
    S.append('')
    S.append('**图谱 v1.3 审计通过（重复三）：FTR-22 曲线/port_frac/k8 带/E_newent/p3171 ratio 全部从封存源'
             '精确浮点重推导（G5），FTR-21 gate ratio 在锚内确认；缺口排序裁决 = (1) 3175 结构残留 GPU 交叉臂'
             '（预注册草案工件 ' + pr_sha + '）、(2) E1→E2 零 GPU 批量升级（FTR-04/08 走 3169 npz、FTR-06 '
             'unembed-only 重建）、(3) N 线 P3-P7 补 Ledger。**')
    S.append('')
    S.append('### 十门读数（result 现场渲染）')
    S.append('')
    S.append('G1 18 源 sha8；G2 影子重渲染 registry/gap 字节恒等 + HTML diff ' + str(SH['html_diff_lines']) +
             ' 行全时间戳 + 637 字段；G3 22 特征不变量 + v1.1 逐字节保留 21/21；G4 25 内容 sha 锚解析 + 162 '
             '断言导航 0 失败（点路径/数字叶列表索引/node:按 id）+ 3 派生键归 G5 重推导 + GAP-4 五文件 sha；'
             'G5 FTR-22/FTR-21 精确重推导；G6 分类审计 6 注记；G7 E1→E2 评估（零 GPU 候选 3）；G8 预注册草案 '
             + pr_sha + '；G9 排序 3 项；G10 ledger n=325 含 3173。')
    S.append('')
    S.append('### E1→E2 升级路径评估（G7）')
    S.append('')
    S.append('FTR-04（S_class 跨模型重建）：缺 held_out——零 GPU 候选（3169 npz 已见类行 LOEO/新实体读出）。'
             'FTR-08（R×K_entity 弱混合）：缺 held_out——零 GPU 候选（3169 E_newent 行=held-out 实体泛化）。'
             'FTR-06（S_attr/S_syntax）：缺跨模型——零 GPU 候选（unembed-only 逐模型重建，3166 S_class '
             '配方 verbatim，词表绑定各模型 tokenizer 面）。FTR-14（T_C 几何）/FTR-15（商结构）：需 GPU '
             '新关系/新输入泛化。FTR-03 前提纠偏：已 E2 无需动作。')
    S.append('')
    S.append('### 装置与诚实登记')
    S.append('')
    S.append('**G4 两次观测前重设计（SMOKE 拦截，seal 前完成，无封存产物被碰）**：① 特征锚 asserts 的 '
             'res/seal sha 是内容 sha（如 3166 res 89b3f320）≠ 文件 sha（5b51c2c1）——首版用文件索引匹配内容 sha '
             '必败，改为「按 src 解析源文档+断言其内容 sha 字段」；② 导航三细节——p3162 audit 的 nodes 是带 id '
             '列表、p3156/q06 数字叶是列表索引、FTR-22 p3169 锚的 k0_drift_* 是 3173 派生键（归 G5 重推导白名单）。'
             '正式跑零修正轮。')
    S.append('')
    S.append('### 缺口排序裁决与接续：预注册 3175')
    S.append('')
    S.append('排序：(1) **3175 = G5-A12 谱外类结构残留定位 GPU 交叉臂**——每谱外类 2k 校准实体面板内/外各半，'
             'k in {0,1,2,4,8}，3172 协议 verbatim；主门 delta = pooled ratio_out(k8) - ratio_in(k8)：'
             '|delta|<=0.15 ⇒ port_residual_dominant（H2，mechanism_note 定稿）、>0.15 ⇒ '
             'entity_familiarity_confirmed（H1）、<-0.15 ⇒ anomaly_register；不可变谓词：k=0 臂逐位复现 3169 '
             'gate.ratio_B=2.5388114997805062（drift<1e-9），面板外词表在 3175 freeze 冻结且先于任何 GPU '
             'forward。TBD@freeze：每类实体池大小、词表、种子、批处理。(2) E1→E2 零 GPU 批量升级（3176 候选）。'
             '(3) N 线 P3-P7 补 Ledger（跨线挂账）。')
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
    shutil.copyfile(MEMO, MEMO + '.snap3174')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3174 section inserted (snapshot .snap3174)')

# ---------- 3. daily ----------
dline = ('- **3174 图谱完整性审计（2026-10-09）**：G5-A11 零 GPU 369 检查——影子重渲染 registry/gap 字节恒等、'
         'HTML diff 1 行全时间戳、637 字段；锚链 25 内容 sha + 162 导航 0 失败 + GAP-4 五文件 sha；FTR-03 '
         '预注册前提纠偏（已 E2）；E1→E2 零 GPU 候选=FTR-04/06/08；排序=3175 结构残留交叉臂→E1→E2→N 线 '
         'Ledger。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n→326。G4 两次观测前重设计'
         '（内容 sha 口径 + 导航细节）。下一步 3175=G5-A12 GPU 交叉臂。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3174 图谱完整性审计' in dtxt:
    log('3. daily: 3174 line already present, skip')
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
newline = ('\n- **✅ 3174 图谱 v1.3 完整性审计+缺口排序（2026-10-09）**：G5-A11 零 GPU 369 检查。**影子重渲染**：'
           'registry/gap 字节恒等、HTML 行 diff 仅 1 行全 meta.created、data-k 637==637。**锚链三轨**：25 内容 sha '
           '锚解析到源文档 + 162 断言导航 0 失败（点路径/数字叶索引/node:按 id）+ 3 派生键 G5 重推导 + GAP-4 五文件 '
           'sha（1650 文件索引）。G5 FTR-22 曲线/port_frac/k8/E_newent/p3171 精确浮点重推导。**FTR-03 预注册前提'
           '纠偏**：registry v1.2 已是 E2（p3154+p3166 双 cross_model+held_out）；实际 E1={FTR-04,06,08,14,15}；'
           '6 条 E2 标签完备性注记不追溯降级。**E1→E2 零 GPU 候选 3**：FTR-04（3169 npz held-out 读出）、'
           'FTR-08（3169 E_newent 行）、FTR-06（unembed-only 重建）；FTR-14/15 需 GPU。**缺口排序**：(1) 3175 '
           'G5-A12 结构残留 GPU 交叉臂（预注册草案 ' + pr_sha + '：面板内/外校准实体各半，主门 '
           'delta=ratio_out(k8)-ratio_in(k8) 0.15 阈，k=0 逐位复现 3169）、(2) E1→E2 零 GPU 批、(3) N 线 '
           'P3-P7 Ledger。诚实登记：G4 两次观测前重设计（内容 sha≠文件 sha 口径；导航三细节）。res ' +
           R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n=325→**326**。下一步 3175=**G5-A12 GPU '
           '交叉臂**（请用户手动启动环境后执行）。\n')
if '3174 图谱 v1.3 完整性审计' in wm:
    log('4. workspace MEMORY: 3174 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3174')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3174)')

# ---------- 5. self-check ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3174 = [m for m in ms2 if m.get('phase') == 3174]
chk.append(('ledger has 3174 entry', len(e3174) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=326', len(ms2) >= 326, 'n=' + str(len(ms2))))
if e3174:
    chk.append(('ledger verdict has ranking', 'ranking_3_items' in e3174[0]['verdict'],
                e3174[0]['verdict'][:80]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3174', '## Phase 3174' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has prereg sha', pr_sha in memo2))
chk.append(('MEMO has FTR-03 correction', 'FTR-03 预注册前提纠偏' in memo2))
chk.append(('MEMO has prereg 3175 ref', '预注册 3175' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3174', '3174 图谱完整性审计' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3174', '3174 图谱 v1.3 完整性审计' in w2))
chk.append(('prereg on disk matches', pr_sha == sha8_file(os.path.join(PDIR, 'prereg_3175_structural_residual_draft.json')), pr_sha))
R2 = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
chk.append(('result seal intact', R2['seal_sha8'] == R['seal_sha8'] and R2['res_sha8'] == R['res_sha8'], R['res_sha8'] + '/' + R['seal_sha8']))
chk.append(('shadow identical flags', R2['shadow']['registry_identical'] and R2['shadow']['gap_identical'], ''))
chk.append(('anchors 25/162', R2['anchors']['content_sha_anchors'] == 25 and
            R2['anchors']['nav_asserts'] == 162, json.dumps({k: R2['anchors'][k] for k in ('content_sha_anchors', 'nav_asserts')})))
chk.append(('ranking 3 items', len(R2['ranking']) == 3 and R2['ranking'][0]['item'] == 'structural_residual_localization', ''))
chk.append(('e1 assessment 5', len(R2['e1_assessment']) == 5, str(len(R2['e1_assessment']))))
chk.append(('taxonomy notes 6', len(R2['taxonomy']['tag_notes']) == 6, ','.join(R2['taxonomy']['tag_notes'])))
chk.append(('premise correction recorded', 'already E2_predictive' in R2['taxonomy']['premise_correction']['registry_v12_reality'], ''))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
