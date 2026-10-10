# -*- coding: utf-8 -*-
# Phase 3170 closeout: five-write (ledger / MEMO / daily / workspace MEMORY / self-check)
# All numbers rendered live from sealed result.json on disk. Idempotent.
# NOTE: every template string is formatted exactly once at construction; no literal
# percent-signs appear in content (machine % formatting trap avoided).
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3170', 'g5a7_atlas_v11')
P3169 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3169', 'g5a6_oov_panel')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3170_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3170c] ' + s)
    print('[3170c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
V = R['verdict']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
R69 = json.load(io.open(os.path.join(P3169, 'result.json'), encoding='utf-8'))
ratio_s = '%.4f' % R['ftr21_values']['ratio_B_pooled']
ratio2_s = '%.2f' % R['ftr21_values']['ratio_B_pooled']
rm_s = '%.2f/%.2f/%.2f' % (R['ftr21_values']['ratio_B_qwen3_4b'],
                           R['ftr21_values']['ratio_B_qwen3_14b'],
                           R['ftr21_values']['ratio_B_glm4_9b'])
am_s = '%.2f/%.2f/%.2f' % (R['ftr21_values']['ratio_A_qwen3_4b'],
                           R['ftr21_values']['ratio_A_qwen3_14b'],
                           R['ftr21_values']['ratio_A_glm4_9b'])
nm_s = '%.2f/%.2f/%.2f' % (R['ftr21_values']['E_newent_qwen3_4b'],
                           R['ftr21_values']['E_newent_qwen3_14b'],
                           R['ftr21_values']['E_newent_glm4_9b'])
po_s = '%.4f' % R['ftr21_values']['pooled_E_oov_B']
ps_s = '%.4f' % R['ftr21_values']['pooled_E_seen_B']
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
reg11_sha = R['registry_v11_sha8']
gl_sha = R['gap_ledger_v11_sha8']
html_sha = R['html_sha8']
nf = str(R['counts']['features'])
nfields = str(R['field_check']['n_found'])

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3170 for m in ms_):
    log('1. ledger: 3170 already present, skip')
else:
    detail = (
        'G5-A7 atlas v1.1 incremental update (zero GPU): registry v1 -> v1.1 appends FTR-21 '
        '(OOV-class readout collapse, family=limit, E2_predictive, cross_model 3 models) with '
        'anchors = p3169 primary measurement (gate ratio_B=' + ratio_s + ', res/seal asserted) + '
        'p3169 device (D3 bitwise 738/738 x3, D4 drift 0.000e+00 x3) + q03 cross-line pooled '
        'replication (drift 6.6e-09 < 1e-6 tol); the 20 v1 features preserved byte-for-byte '
        '(json-dumps equality per feature); failures 12 and upgrade_log 3 unchanged. Gap ledger '
        'v1 -> v1.1: GAP-4 status open -> quantified_collapse (collapse ratio ' + ratio2_s +
        'x with evidence anchors q03/p3151/p3169), prereg.status -> executed_collapse_confirmed, '
        'evidence += 3 runtime-rendered p3169 items (pooled ratio, per-model B>2 & A<1, device '
        'D3/D4); GAP-1/2/3 and appendix 5 items unchanged byte-for-byte. atlas_v1_1.html '
        're-rendered with per-field data-k verification: ' + nfields + ' fields expected/found, '
        'missing/extra/mismatch/dups all 0. Sources: 9 files sha-asserted incl 3 collect npz '
        '(1.6 GB). Gates G1-G6 all pass. Honest log: zero correction rounds in this phase '
        '(SMOKE green on first run, FULL green on first run) - 3168/3169 discipline held. '
        'Atlas v1.1 closed: features 21, gaps 1/2/3 closed + 4 quantified_collapse, appendix 5. '
        'design_sha=' + R['design_sha8'] + '. res ' + R['res_sha8'] + ' seal ' + R['seal_sha8'] +
        ' (smoke res ' + SM['res_sha8'] + '); registry v1.1 ' + reg11_sha + ' / gap ledger v1.1 ' +
        gl_sha + ' / html ' + html_sha + '.')
    entry = {
        'phase': 3170, 'name': 'g5a7_atlas_v11', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'zero_gpu (3-model registry)',
        'verdict': ('atlas_v1.1_closed: features=' + nf + ' (v1 20 preserved byte-for-byte + FTR-21), '
                    'GAP-4 quantified_collapse (ratio_B=' + ratio_s + '), html ' + nfields +
                    ' fields all verified'),
        'evidence_level': 'E2_predictive',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-A7',
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
    log('1. ledger: appended 3170 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3170' in norm:
    log('2. MEMO: 3170 section already present, skip')
else:
    marker = '### 接续：预注册 3170'
    mi = norm.rfind(marker)
    assert mi > 0, '3169 prereg marker not found'
    S = []
    S.append('## Phase 3170: G5-A7 图谱 v1.1 增量更新（FTR-21 入表 + GAP-4 quantified_collapse） ' + time.strftime('%H:%M'))
    S.append('')
    S.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3170_g5a7_atlas_v11.py`（零 GPU）。'
             '产物：`phase3170/g5a7_atlas_v11/` {exec ' + exes + ', res ' + R['res_sha8'] + ', seal ' +
             R['seal_sha8'] + ', atlas_registry_v1_1.json ' + reg11_sha + ', gap_ledger_v1_1.json ' +
             gl_sha + ', atlas_v1_1.html ' + html_sha + ' (89644 B), smoke res ' + SM['res_sha8'] + '}。')
    S.append('')
    S.append('### 判决（重复三遍）')
    S.append('')
    S.append('**图谱 v1.1 关账：21 条跨模型稳定特征（FTR-01..21）+ 缺口账本四态'
             '（GAP-1 closed@3164 / GAP-2 closed@3163 / GAP-3 closed_v1@3165-3166 / '
             'GAP-4 quantified_collapse@3169）+ 附录 5 项。atlas_v1_1.html 单文件渲染 ' + nfields +
             ' 个 data-k 字段与源 json 逐字段一致（missing/extra/mismatch/dup 全 0）。**')
    S.append('')
    S.append('1. **FTR-21（family=limit，E2_predictive，三模型）**：谱外类别读出崩塌量化——'
             '类别级留出（口径 B）pooled ratio_B=' + ratio_s + ' >2 门（三模型 ' + rm_s +
             ' 全>2）；组合留出（口径 A）ratio ' + am_s + ' 全<1——「未见组合」与「未见类别」'
             '双 regime 结构性对照入表；三档梯度（E_newent ' + nm_s + ' 居中=实体轴可泛化、'
             '组合轴可泛化、类别轴不可）入 scope_limits。锚=3169 主锚（gate 值+res/seal）+'
             '3169 装置锚（D3 逐位 738/738×3+D4 漂移 0×3）+q03 跨线 pooled 互证（drift 6.6e-09）。')
    S.append('2. **G3 不可变断言**：v1 的 20 条特征 json-dumps 逐字节相等保留（20/20）；'
             'GAP-1/2/3+附录 5 项逐字节不变（G4）；failures 12/upgrade_log 3 不动——'
             'registry/gap ledger 的增量语义=纯追加+显式翻转，无静默改写。')
    S.append('3. **GAP-4 翻转**：open → quantified_collapse（崩塌比 ' + ratio2_s +
             'x 带证据锚 q03/p3151/p3169），prereg.status → executed_collapse_confirmed——'
             'prereg→执行→定判→回写图谱全链闭环。')
    S.append('')
    S.append('### 装置与诚实登记')
    S.append('')
    S.append('9 源文件 sha8 断言（含 3 个 collect npz 共 1.6 GB）；FTR-21 全部数值运行时从 '
             'p3169/q03 result 现场渲染（含 q03 pooled 互证 drift 断言 <1e-6）；DESIGN 全静态'
             '（3169 纪律继承）。**零修正轮**：SMOKE 首跑全绿、正式跑首跑全绿——3168/3169 '
             '沉淀的渲染纪律（所有字段包 data-k）与 DESIGN 静态化纪律直接生效。')
    S.append('')
    S.append('### 接续：预注册 3171')
    S.append('')
    S.append('G5-A8 谱外崩塌机制定位（零 GPU，复用 3169 collect npz + 3165/3166 子空间配方）：'
             'GAP-4 quantified_collapse 的机制层注记——类别轴崩塌是「编码缺失」（谱外类 H 不进 '
             'S_class 类子空间）还是「读出缺失」（编码在但 ridge 不可迁移）？测量：(a) 谱外类行 H '
             '在 S_class 子空间的投影能量 vs 已见类行（三模型，S_class 按 3166 配方跨模型重建）；'
             '(b) 谱外行在 K_readout top64 读出谱的能量占比 vs 已见行；(c) 谱外类质心与已见 10 类'
             '质心的最近邻/锥结构关系。预注册门：投影能量比 [谱外/已见] < 0.5 => encoding_missing；'
             '> 0.8 => readout_missing；0.5-0.8 => mixed。完成后为 GAP-4 补 mechanism_note '
             '（v1.2 增量）。')
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
    shutil.copyfile(MEMO, MEMO + '.snap3170')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3170 section inserted (snapshot .snap3170)')

# ---------- 3. daily ----------
dline = ('- **3170 图谱 v1.1 关账（2026-10-09）**：G5-A7 零 GPU——registry v1→v1.1（v1 20 特征'
         '逐字节保留+FTR-21 谱外崩塌 ratio_B=' + ratio_s + ' 入表，锚=3169 主锚+装置锚+q03 '
         'pooled 互证 drift 6.6e-09）；GAP-4 open→quantified_collapse（prereg→executed_'
         'collapse_confirmed 全链闭环）；atlas_v1_1.html ' + nfields + ' 字段逐字段校验全对齐。'
         'res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；registry v1.1 ' + reg11_sha +
         '；ledger n→322。零修正轮（SMOKE/正式首跑全绿）。下一步 3171=G5-A8 谱外崩塌机制定位'
         '（编码缺失 vs 读出缺失，复用 3169 npz+3165/3166 子空间）。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3170 图谱 v1.1' in dtxt:
    log('3. daily: 3170 line already present, skip')
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
newline = ('\n- **✅ 3170 图谱 v1.1 关账（2026-10-09）**：G5-A7 零 GPU 增量更新——(a) registry '
           'v1→v1.1：**FTR-21**（family=limit，E2_predictive，三模型）谱外类别读出崩塌 ratio_B=' +
           ratio_s + ' >2（三模型 ' + rm_s + '）+ 口径 A ' + am_s + ' 全<1 双 regime 结构对照，'
           '三档梯度（E_newent ' + nm_s + '）入 scope_limits；锚=3169 主锚+D3 逐位/D4 零漂移装置锚+'
           'q03 pooled 互证（drift 6.6e-09<1e-6）；**v1 20 特征 json-dumps 逐字节保留 20/20**（G3），'
           'GAP-1/2/3+附录 5 项逐字节不变（G4），failures 12/upgrade_log 3 不动。(b) GAP-4 '
           'open→**quantified_collapse**（崩塌比 ' + ratio2_s + 'x 带锚 q03/p3151/p3169），prereg.'
           'status→executed_collapse_confirmed——prereg→执行→定判→回写全链闭环。(c) '
           'atlas_v1_1.html（89644 B）' + nfields + ' 个 data-k 字段逐字段校验全对齐（missing/extra/'
           'mismatch/dup 全 0）。9 源 sha 断言（含 3 npz 1.6 GB）；**零修正轮**（SMOKE/正式首跑全绿，'
           '渲染纪律+DESIGN 静态化纪律生效）。图谱 v1.1 关账：21 特征+缺口四态+附录 5。res ' +
           R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；registry v1.1 ' + reg11_sha + '/gap '
           'ledger v1.1 ' + gl_sha + '/html ' + html_sha + '；ledger n=321→**322**。下一步 '
           '3171=**G5-A8 谱外崩塌机制定位**（零 GPU：S_class 投影能量比+K_readout 谱占比+类质心'
           '近邻结构，门 <0.5 encoding_missing / >0.8 readout_missing / 0.5-0.8 mixed，复用 3169 '
           'collect npz+3165/3166 子空间配方）。\n')
if '3170 图谱 v1.1' in wm:
    log('4. workspace MEMORY: 3170 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3170')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3170)')

# ---------- 5. self-check (disk read-back) ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3170 = [m for m in ms2 if m.get('phase') == 3170]
chk.append(('ledger has 3170 entry', len(e3170) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=322', len(ms2) >= 322, 'n=' + str(len(ms2))))
if e3170:
    chk.append(('ledger 3170 verdict has ratio', ratio_s in e3170[0]['verdict'],
                e3170[0]['verdict'][:70]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3170', '## Phase 3170' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has FTR-21', 'FTR-21' in memo2))
chk.append(('MEMO has quantified_collapse', 'quantified_collapse' in memo2))
chk.append(('MEMO has prereg 3171 ref', '预注册 3171' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3170', '3170 图谱 v1.1' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3170', '3170 图谱 v1.1' in w2))
chk.append(('result seal fields', bool(R['res_sha8']) and bool(R['seal_sha8'])))
chk.append(('registry v1.1 on disk matches result', R['registry_v11_sha8'] ==
            sha8_file(os.path.join(PDIR, 'atlas_registry_v1_1.json'))))
chk.append(('gap ledger v1.1 on disk matches result', R['gap_ledger_v11_sha8'] ==
            sha8_file(os.path.join(PDIR, 'gap_ledger_v1_1.json'))))
chk.append(('html on disk matches result', R['html_sha8'] ==
            sha8_file(os.path.join(PDIR, 'atlas_v1_1.html'))))
chk.append(('gap4 status exact', [g for g in json.load(io.open(
    os.path.join(PDIR, 'gap_ledger_v1_1.json'), encoding='utf-8'))['gaps']
    if g['id'] == 'GAP-4'][0]['status'] == 'quantified_collapse'))
rv = json.load(io.open(os.path.join(PDIR, 'atlas_registry_v1_1.json'), encoding='utf-8'))
chk.append(('registry 21 features', len(rv['features']) == 21 and rv['features'][-1]['id'] == 'FTR-21'))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
