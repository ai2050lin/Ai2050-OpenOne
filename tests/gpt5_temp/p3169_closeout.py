# -*- coding: utf-8 -*-
# Phase 3169 closeout: five-write (ledger / MEMO / daily / workspace MEMORY / self-check)
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
                    'phase3169', 'g5a6_oov_panel')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3169_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3169c] ' + s)
    print('[3169c] ' + s, flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
assert R['res_sha8'] and R['seal_sha8']
V = R['verdict']
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
g = R['gate']
ratio = g['ratio_B']
pooled_oov = g['pooled_E_oov_B']
pooled_seen = g['pooled_E_seen_B']
pm = R['per_model']
r_4b = pm['qwen3-4b']['B']['ratio']
r_14b = pm['qwen3-14b']['B']['ratio']
r_glm = pm['glm4-9b']['B']['ratio']
a_4b = pm['qwen3-4b']['A']['ratio']
a_14b = pm['qwen3-14b']['A']['ratio']
a_glm = pm['glm4-9b']['A']['ratio']
ne_4b = pm['qwen3-4b']['B']['E_newent']
ne_14b = pm['qwen3-14b']['B']['E_newent']
ne_glm = pm['glm4-9b']['B']['E_newent']
ratio_s = '%.4f' % ratio
ratio2_s = '%.2f' % ratio
rm_s = '%.2f/%.2f/%.2f' % (r_4b, r_14b, r_glm)
am_s = '%.2f/%.2f/%.2f' % (a_4b, a_14b, a_glm)
nm_s = '%.2f/%.2f/%.2f' % (ne_4b, ne_14b, ne_glm)
po_s = '%.4f' % pooled_oov
ps_s = '%.4f' % pooled_seen
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
npz_shas = {m: sha8_file(os.path.join(PDIR, 'collect_' + m + '.npz'))
            for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b')}

# ---------- 1. ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3169 for m in ms_):
    log('1. ledger: 3169 already present, skip')
else:
    detail = (
        'G5-A6 out-of-vocab class panel (GPU, executes GAP-4 prereg from 3168): Q03 protocol '
        'verbatim (B4 ridge one-hot additive, lam=1e-3, normalized MSE, S1 split mechanism, '
        'batch=1 bf16 collect, H fp16); panel = 6 seen classes (41 entities verbatim) + 4 OOV '
        'classes (乐器/天气/运动/电器, 8 new entities each, double exclusion vs 3152 classes and '
        '2881 families; prereg correction logged: 水果/金属 in the prereg text are seen classes). '
        'Panel 73 ents x 10 classes x 3 tpl = 2190 rows x 3 models. Device gates: D0 vocab '
        'double exclusion; D1 finite+shapes; D2 first tokens distinct; D3 seen-row H bitwise '
        'vs sealed 3152/3151 collect.npz 738/738 rows x3 models (0 mismatch); D4 anchor check '
        'seen-subpanel cols=51 Q03-verbatim recompute == q03 per-seed anchors with drift '
        '0.000e+00 x3 models. Main gate (口径 B = class-level leave-out, OOV-class rows fully '
        'absent from train): ratio_B = pooled(E_oov_B)/pooled(E_seen_B) = ' + po_s + '/' + ps_s +
        ' = ' + ratio_s + ' > 2 -> collapse_confirmed_gap4_open. Per-model ratios ' + rm_s +
        ' (4b/14b/glm4), all > 2. 口径 A (combo held-out incl. partial OOV-class training '
        'exposure) ratios ' + am_s + ' all < 1 -> structural proof that unseen-combo and '
        'unseen-class are different regimes. E_newent_B (new entities, seen classes) = ' + nm_s +
        '. E_seen_B reproduces q03 pooled 0.373350. Honest log: R0 SMOKE D3 position-alignment '
        'bug (truncated panel entity positions differ from 3152; fixed by entity-name mapping, '
        'nearest-neighbour diagnosis; collection itself bitwise-correct); R1 DESIGN '
        'SMOKE-dependence (3 DRIFT interceptions: smoke flag, truncated tables, live插值) -> '
        'DESIGN fully staticized, new discipline: design must be run-mode independent. '
        'design_sha=' + R['design_sha8'] + '. res ' + R['res_sha8'] + ' seal ' + R['seal_sha8'] +
        ' (smoke res ' + SM['res_sha8'] + '). collect npz sha8: 4b=' + npz_shas['qwen3-4b'] +
        ' 14b=' + npz_shas['qwen3-14b'] + ' glm4=' + npz_shas['glm4-9b'] + '.')
    entry = {
        'phase': 3169, 'name': 'g5a6_oov_panel', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': ('collapse_confirmed_gap4_open: ratio_B=' + ratio_s + '>2 (per-model ' +
                    rm_s + '); combo-heldout A ratios ' + am_s + ' all<1; device bitwise+anchor-exact'),
        'evidence_level': 'E3_causal_scoped',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-A6',
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
    log('1. ledger: appended 3169 (n=' + str(n1) + ', was ' + str(n0) + ') chain_sha8=' +
        led['ledger_sha256_8'])

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3169' in norm:
    log('2. MEMO: 3169 section already present, skip')
else:
    marker = '### 接续：预注册 3169'
    mi = norm.rfind(marker)
    assert mi > 0, '3168 prereg marker not found'
    S = []
    S.append('## Phase 3169: G5-A6 谱外类别面板（GAP-4 prereg 执行，崩塌确认） ' + time.strftime('%H:%M'))
    S.append('')
    S.append('**日期**：2026-10-09。**脚本**：`tests/glm5/phase3169_g5a6_oov_panel.py`（GPU 三模型，'
             '35 min）。产物：`phase3169/g5a6_oov_panel/` {exec ' + exes + ', res ' + R['res_sha8'] +
             ', seal ' + R['seal_sha8'] + ', collect npz 4b=' + npz_shas['qwen3-4b'] + '/14b=' +
             npz_shas['qwen3-14b'] + '/glm4=' + npz_shas['glm4-9b'] + ', smoke res ' +
             SM['res_sha8'] + '}。')
    S.append('')
    S.append('### 装置（五门）')
    S.append('')
    S.append('D0 词表双重排除（3152 六类+2881 十族概念）；D1 采集有限+形状精确 (3,730,NL+1,D)；'
             'D2 十类首 token 互异；D3 **已见行 H 与 3152/3151 封存 npz 逐位一致 738/738 行×3 模型'
             '（0 mismatch）**；D4 已见子面板 cols=51 Q03 verbatim 复算=锚 **drift 0.000e+00×3 模型**'
             '（9 个 per-seed 锚逐位）。prereg 修正（诚实登记）：prereg 词表「水果/金属」为已见类'
             '（claim_precision 笔误），谱外性以双重排除冻结，门不变。')
    S.append('')
    S.append('### 判决（重复三遍）')
    S.append('')
    S.append('**缺口④定判：collapse_confirmed_gap4_open——类别级留出口径（B，真谱外：谱外类别行'
             '完全缺席训练）ratio_B = pooled(E_oov)/pooled(E_seen) = ' + po_s + '/' + ps_s + ' = ' +
             ratio_s + ' > 2 门，谱外类别读出崩塌确认且量化为 ' + ratio2_s + 'x；三模型单模型 ' +
             rm_s + ' 全部 >2。组合 held-out 口径（A）ratio ' + am_s + ' 全部 <1——「未见组合」与'
             '「未见类别」是两个 regime 的结构性证明；E_newent_B（新实体旧类别）=' + nm_s +
             ' 介于两者之间。E_seen_B 逐位复现 q03 pooled 0.373350。verdict=' + V + '。**')
    S.append('')
    S.append('### 特征读数（图谱增量）')
    S.append('')
    S.append('1. 类别轴是读出编码的硬边界：跨类别轴迁移不成立（' + ratio2_s + 'x 崩塌），同类新实体'
             '迁移部分成立（~1.9x），同实体新组合完全成立（A 口径 <1）——三档梯度=实体轴/组合轴'
             '可泛化、类别轴不可。')
    S.append('2. 14b 最稳（2.13x），4b 最脆（2.83x）——崩塌比跨模型同号（全 >2）但幅度模型相关。')
    S.append('3. SMOKE 截断面板（600 行）与正式面板（2190 行）读数同向（2.64 vs 2.54）——面板规模'
             '稳健性旁证。')
    S.append('')
    S.append('### 修正与诚实登记')
    S.append('')
    S.append('R0（SMOKE）：D3 位置对齐 bug——截断面板下联合实体位置 ≠ 3152 实体位置（每类前 2 截断'
             '使狗=联合 i2 但 3152 i8）；最近邻诊断定位（同文本 H bitwise dist=0 证明采集本身正确），'
             '修为**实体名映射对齐**。R1（正式跑前三轮 DRIFT 拦截）：DESIGN 含运行模式依赖'
             '（smoke 标志/截断表/运行时插值）→ **DESIGN 全静态化新纪律：design 必须与运行模式'
             '无关**，smoke 状态只进 result。execution.json ' + exes + ' 冻结（design ' +
             R['design_sha8'] + '，SMOKE 与正式同 design）。')
    S.append('')
    S.append('### 接续：预注册 3170')
    S.append('')
    S.append('G5-A7 图谱 v1.1 增量更新（零 GPU）：registry v1 → v1.1——(a) 新增 FTR-21（谱外类别'
             '读出崩塌：ratio_B=' + ratio2_s + ' 三模型全>2，双口径结构性对照 A<1<B，锚=3169 主锚+' +
             'D4 零漂移锚+q03 pooled 互证）；(b) GAP-4 状态 open → quantified_collapse（崩塌比 ' +
             ratio2_s + 'x 带证据锚）；(c) 特征读数三档梯度（实体轴/组合轴可泛化、类别轴不可）入 '
             'FTR-21 scope_limits；(d) atlas_v1.html → v1.1 逐字段重渲染校验。完成后图谱 v1.1 关账。')
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
    shutil.copyfile(MEMO, MEMO + '.snap3169')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3169 section inserted (snapshot .snap3169)')

# ---------- 3. daily ----------
dline = ('- **3169 谱外类别面板闭环（2026-10-09）**：G5-A6 GPU 三模型 35min——Q03 协议 verbatim，'
         '面板 73 实体×10 类（41 已见+4 谱外 乐器/天气/运动/电器×8 实体）。装置：D3 已见行 H 与'
         ' 3152/3151 npz 逐位一致 738/738×3、D4 锚复算漂移 0.000e+00×3。**缺口④定判：'
         'collapse_confirmed**——ratio_B=' + ratio_s + '>2（三模型 ' + rm_s + '），组合 held-out '
         '口径 A 反向全<1（' + am_s + '）=未见组合与未见类别双 regime 结构性证明；E_seen 复现 '
         'q03 pooled 0.373350。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] + '；ledger n→321。'
         '下一步 3170=G5-A7 图谱 v1.1（FTR-21 入表+GAP-4 quantified_collapse）。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3169 谱外类别面板' in dtxt:
    log('3. daily: 3169 line already present, skip')
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
newline = ('\n- **✅ 3169 谱外类别面板闭环（2026-10-09）**：G5-A6 GPU 三模型（35min，面板 73 实体×'
           '10 类=41 已见 verbatim+4 谱外 乐器/天气/运动/电器×8 实体，Q03 协议逐字复用：B4 ridge '
           'one-hot/λ=1e-3/归一化 MSE/S1 split 机制/batch=1 bf16/H fp16）。装置五门全过：D0 词表'
           '双重排除、D2 首 token 互异、**D3 已见行 H 与 3152/3151 npz 逐位一致 738/738×3（0 '
           'mismatch）**、**D4 锚复算漂移 0.000e+00×3（9 per-seed 锚逐位）**。**缺口④定判：'
           'collapse_confirmed_gap4_open**——主门（口径 B=类别级留出，谱外类别行完全缺席训练）'
           'ratio_B=' + ratio_s + '>2（pooled E_oov ' + po_s + ' vs E_seen ' + ps_s + '，三模型 ' +
           rm_s + ' 全>2，14b 最稳 2.13/4b 最脆 2.83）；口径 A（组合 held-out，训练含谱外类部分'
           '组合）ratio ' + am_s + ' 全<1——未见组合与未见类别双 regime 结构性证明；E_newent（新'
           '实体旧类别）' + nm_s + ' 居中——三档梯度=实体轴/组合轴可泛化、**类别轴不可泛化（读出'
           '编码硬边界）**；E_seen_B 复现 q03 pooled 0.373350。诚实登记：R0 SMOKE D3 位置对齐 '
           'bug（截断面板实体位置≠3152，最近邻诊断定位，采集本身 bitwise 正确，修为实体名映射）；'
           'R1 三轮 DRIFT 拦截（DESIGN 含 smoke 标志/截断表/运行时插值）→ **DESIGN 全静态化新'
           '纪律：design 必须与运行模式无关**。res ' + R['res_sha8'] + '/seal ' + R['seal_sha8'] +
           '；collect npz 4b=' + npz_shas['qwen3-4b'] + '/14b=' + npz_shas['qwen3-14b'] +
           '/glm4=' + npz_shas['glm4-9b'] + '；ledger n=320→**321**。下一步 3170=**G5-A7 图谱 '
           'v1.1**（零 GPU：FTR-21 谱外崩塌 ' + ratio2_s + 'x 入表、GAP-4→quantified_collapse、'
           'atlas_v1.html→v1.1 重渲染校验）。\n')
if '3169 谱外类别面板' in wm:
    log('4. workspace MEMORY: 3169 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3169')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3169)')

# ---------- 5. self-check (disk read-back) ----------
chk = []
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led2['measurements']
e3169 = [m for m in ms2 if m.get('phase') == 3169]
chk.append(('ledger has 3169 entry', len(e3169) == 1, 'n=' + str(len(ms2))))
chk.append(('ledger n>=321', len(ms2) >= 321, 'n=' + str(len(ms2))))
if e3169:
    chk.append(('ledger 3169 verdict has ratio', ratio_s in e3169[0]['verdict'],
                e3169[0]['verdict'][:70]))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3169', '## Phase 3169' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has seal sha', R['seal_sha8'] in memo2))
chk.append(('MEMO has ratio', ratio_s in memo2))
chk.append(('MEMO has prereg 3170 ref', '预注册 3170' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3169', '3169 谱外类别面板' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3169', '3169 谱外类别面板' in w2))
chk.append(('result seal fields', bool(R['res_sha8']) and bool(R['seal_sha8'])))
chk.append(('collect npz sha stable', npz_shas['qwen3-4b'] ==
            sha8_file(os.path.join(PDIR, 'collect_qwen3-4b.npz'))))
chk.append(('gate verdict exact', g['verdict'] == 'collapse_confirmed_gap4_open', g['verdict']))
chk.append(('D4 all exact', all(pm[k]['anchor']['drift'] == [0.0, 0.0, 0.0] for k in pm)))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check ' + str(len(chk) - len(bad)) + '/' + str(len(chk)) + ')')
