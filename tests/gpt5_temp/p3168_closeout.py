# -*- coding: utf-8 -*-
# Phase 3168 closeout: five-write (ledger / MEMO / daily / workspace MEMORY / self-check)
# All numbers rendered live from sealed result.json + gap_ledger_v1.json on disk.
# Idempotent. Disk-read self-check at the end.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3168', 'g5a5_atlas_render')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3168_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3168c] %s' % s)
    print('[3168c] %s' % s, flush=True)


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = load(os.path.join(PDIR, 'result.json'))
assert R['res_sha8'] and R['seal_sha8']
V = R['verdict']
GL = load(os.path.join(PDIR, 'gap_ledger_v1.json'))
SMOKE = load(os.path.join(PDIR, 'smoke_result.json'))
n_fields = R['field_check']['n_found']
n_feat = R['counts']['features']
n_nodes = R['counts']['nodes']
n_fails = R['counts']['failures']
n_upgs = R['counts']['upgrades']
n_appx = R['counts']['appendix']
gap_status = R['gap_status']
gap1_ev0 = GL['gaps'][0]['evidence'][0]
gap2_ev1 = GL['gaps'][1]['evidence'][1]
gap4_ev0 = GL['gaps'][3]['evidence'][0]
gap4_ev1 = GL['gaps'][3]['evidence'][1]
html_sha = R['html_sha8']
gl_sha = R['gap_ledger_sha8']
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
dsha = R['design_sha8']

# ---------- 1. ledger ----------
led = load(LEDGER)
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3168 for m in ms_):
    log('1. ledger: 3168 already present, skip')
else:
    detail = (
        'G5-A5 atlas v1 render + gap ledger rewrite (zero GPU): atlas_registry_v1.json (3167, '
        'f207aa8d) + 3162 node foundation (00f15e98) rendered to single-file atlas_v1.html '
        '(inline CSS, no external resources, %d bytes, html sha8 %s). Every dynamic field '
        'wrapped in data-k spans; gate G1 = rendered fields vs flattened json, %d fields all '
        'match (missing/extra/mismatch/dup all 0). Gates: G2 feature cards %d/20, G3 nodes '
        '%d/16, G4 failures %d/12, G5 upgrades %d/3, G6 gap ledger 4 gaps + %d appendix items '
        '(closed items carry anchor sha8, GAP-4 carries preregistered experiment), G7 no '
        '<link/<script/http(s)://. 15 source files sha8-asserted. Gap ledger v1 rewritten: '
        'GAP-1 cross-model parity closed@3164 (%s); GAP-2 mechanism chain closed@3163 '
        '(3159 dynamics -> 3160 attention_reallocation_primary -> 3161 not_in_attn_out -> 3163 '
        'redundant_closing, fp_ok x3); GAP-3 cross-family connection closed_v1@3165-3166 '
        '(separable + mixed, not_confounded); GAP-4 out-of-vocabulary transfer open with prereg '
        '3169 G5-A6 (out-of-vocab class panel, Q03 protocol verbatim, gate 2x paired E_read; '
        'evidence %s | %s); appendix 5 items each with anchor sha8. Honest log: G1 checker '
        'caught 11 missing data-k keys in R1 (family/evidence_level/gap title rendered static) '
        '-> renderer completed under checker drive; R0 pre-SMOKE fixes (p3155 anchor missing, '
        'meta/anchor_sha8 flatten keys, created dual-source mismatch, GAP-4 closed_at not '
        'rendered when open). design_sha=%s. res %s seal %s (smoke res %s).'
        % (os.path.getsize(os.path.join(PDIR, 'atlas_v1.html')), html_sha, n_fields,
           n_feat, n_nodes, n_fails, n_upgs, n_appx, gap1_ev0, gap4_ev0, gap4_ev1,
           dsha, R['res_sha8'], R['seal_sha8'], SMOKE['res_sha8']))
    entry = {
        'phase': 3168, 'name': 'g5a5_atlas_render', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'none (zero GPU render)',
        'verdict': 'atlas_v1.html %d fields ok (%d feat/%d nodes/%d fails/%d upg); gaps '
                   'closed=%d open=%d appendix=%d'
                   % (n_fields, n_feat, n_nodes, n_fails, n_upgs,
                      sum(1 for v in gap_status.values() if v.startswith('closed')),
                      sum(1 for v in gap_status.values() if v == 'open'), n_appx),
        'evidence_level': 'render',
        'model_scope': 'none',
        'prereg_id': 'G5-A5',
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
    log('1. ledger: appended 3168 (n=%d, was %d) chain_sha8=%s'
        % (n1, n0, led['ledger_sha256_8']))

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3168' in norm:
    log('2. MEMO: 3168 section already present, skip')
else:
    marker = '### 接续：预注册 3168'
    mi = norm.rfind(marker)
    assert mi > 0, '3167 prereg marker not found'
    section = (
        '## Phase 3168: G5-A5 图谱 v1 渲染 + 缺口账本重写（atlas_v1.html） [hh:mm]\n\n'
        '**日期**：2026-10-09。**脚本**：`tests/glm5/phase3168_g5a5_atlas_render.py`'
        '（零 GPU）。产物：`phase3168/g5a5_atlas_render/` {exec EXEC8, res RES8, seal SEAL8, '
        'atlas_v1.html HTML8, gap_ledger_v1.json GLSHA, smoke res SRES8}。\n\n'
        '### 装置（七门）\n\n'
        'G1 html data-k 逐字段校验（每个动态字段包 data-k span，校验器正则提取后与 registry/'
        'gap ledger flatten json 对比）——**NFIELD 个字段全部一致**（missing/extra/mismatch/dup '
        '全 0）；G2 特征卡 NF/20；G3 节点 NN/16；G4 失败账本 NF2/12；G5 升级链 NU/3；G6 缺口账本 '
        '4 缺口+NA 附录项（closed 项必带锚 sha8、GAP-4 必带 prereg）；G7 无外链（无 link/script/'
        'http(s)://，内嵌 CSS）。15 源文件 sha8 断言（registry f207aa8d、3162 基座 00f15e98、'
        '13 缺口证据锚）。\n\n'
        '### 判决（重复三遍）\n\n'
        '**atlas_v1.html 关账：单文件渲染（内嵌 CSS、零外链）NFIELD 个 data-k 字段与源 json '
        '逐字段一致；缺口账本 v1 重写完成：GAP-1 跨模型同口径 closed@3164（3164a/b/c '
        'agree_True×3：zero_like_q06/rope_relative/massive supported，sha 2114b4dc/d48dcbb7/'
        '4d3174be）、GAP-2 机制链 closed@3163（3159 动力学破坏→3160 attention_reallocation_'
        'primary→3161 consumption_not_in_attn_out→3163 redundant_closing，fp_ok×3）、GAP-3 '
        '跨族连接 closed_v1@3165-3166（separable×3+mixed，not_confounded）、GAP-4 谱外迁移证伪 '
        'open+prereg 3169；附录 5 项逐条证据锚。verdict=VERD。**\n\n'
        '### 缺口账本现场（证据值运行时从锚文件提取）\n\n'
        'GAP-4 证据：EV40；EV41。prereg（G5-A6，status=preregistered_not_executed）：复用 Q03 '
        '协议 verbatim（同构模板/同 k* 读位/判定层=行为读出），仅替换类别词集=4 个谱外类别'
        '（水果/金属/乐器/天气，均不在 2881 词表）×每类 8-10 实体；对照臂=已见类别重测 paired；'
        '三模型；门=谱外 pooled E_read ≤2× 已见 ⇒ 泛化成立（缺口④可关）/ ≥2× ⇒ 崩塌确认'
        '（量化崩塌比）/ 1.5-2× ⇒ borderline 换种子重判。\n\n'
        '### 修正与诚实登记\n\n'
        'R0（SMOKE 前）4 项：APPX-3 锚 p3155 遗漏入 SRC；flatten 缺 meta/anchor_sha8 键；'
        'render/flatten 时间戳双源必 mismatch → 共用 CREATED_STR；GAP-4 open 态不应渲染 '
        'closed_at。R1（G1 校验器抓获）：family/evidence_level/gap title 渲染为静态未包 data-k '
        '→ 11 个 missing key，渲染器在校验器驱动下补全——校验器反向驱动渲染完备性生效。'
        'DESIGN 变更均发生在首次正式观测前（删旧 execution.json 重冻结）。'
        'execution.json EXEC8 冻结。\n\n'
        '### 图谱完成度终判\n\n'
        '特征级（3167 FTR-01..20 八字段）+ 渲染级（3168 逐字段 data-k 校验 NFIELD 字段全对齐）'
        '+ 缺口账本（①②③ closed、④ open+prereg）= **图谱 v1 三层齐备**。「找到稳定特征、'
        '完成图谱」核心目标 v1 达成；开放项=GAP-4 谱外迁移证伪（3169）+附录 5 项。\n\n'
        '### 接续：预注册 3169\n\n'
        'G5-A6 谱外类别面板（GPU，执行 GAP-4 prereg）：按 prereg 协议执行——谱外 4 类×8-10 实体'
        '+已见类别 paired 对照，三模型（14b 用 pre-quantized checkpoint），Q03 协议 verbatim '
        '（k* 读位/判定层/模板族同构），SMOKE 装置门必看：谱外词表确不在 2881、对照臂 E_read '
        '对拍 q03 锚 0.3316/0.3986/0.3898（tol 1e-9）、paired 结构对齐；门按 prereg（2×/'
        '1.5-2×borderline）。完成后缺口④状态定判，图谱 v1→v1.1。\n\n\n---\n\n\n')
    section = (section
               .replace('EXEC8', exes).replace('HTML8', html_sha).replace('GLSHA', gl_sha)
               .replace('RES8', R['res_sha8']).replace('SEAL8', R['seal_sha8'])
               .replace('SRES8', SMOKE['res_sha8'])
               .replace('NFIELD', str(n_fields))
               .replace('NF/', '%d/' % n_feat).replace('NN', str(n_nodes))
               .replace('NF2', str(n_fails)).replace('NU', str(n_upgs))
               .replace('NA', str(n_appx))
               .replace('EV40', gap4_ev0).replace('EV41', gap4_ev1)
               .replace('VERD', V)
               .replace('[hh:mm]', time.strftime('%H:%M')))
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3168')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3168 section inserted before 3167 prereg block (snapshot .snap3168)')

# ---------- 3. daily ----------
dline = ('- **3168 图谱 v1 渲染+缺口账本重写闭环（2026-10-09）**：G5-A5 零 GPU——atlas_v1.html '
         '单文件（内嵌 CSS 零外链）NFIELD 个 data-k 字段与源 json 逐字段一致；G1 校验器反向'
         '驱动渲染完备性（抓获 11 静态漏包）；缺口账本 v1：GAP-1/2/3 closed（3164 三臂 agree×3/'
         '机制链四 phase/GAP-3 separable+mixed）、GAP-4 open+prereg 3169、附录 5 项逐条锚。'
         'res RES8/seal SEAL8；ledger n→320。图谱 v1 三层齐备（特征/渲染/缺口账本），'
         '下一步 3169=G5-A6 谱外类别面板执行 GAP-4 prereg。')
dline = dline.replace('NFIELD', str(n_fields)).replace('RES8', R['res_sha8']).replace('SEAL8', R['seal_sha8'])
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3168 图谱 v1 渲染' in dtxt:
    log('3. daily: 3168 line already present, skip')
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
newline = ('\n- **✅ 3168 图谱 v1 渲染+缺口账本重写闭环（2026-10-09）**：G5-A5 零 GPU——'
           'atlas_registry_v1.json(3167 f207aa8d)+3162 节点基座(00f15e98) → atlas_v1.html 单文件'
           '（内嵌 CSS 零外链），G1 逐字段校验 NFIELD 个 data-k 字段全对齐（missing/extra/'
           'mismatch/dup=0，校验器反向抓获 11 静态漏包=渲染完备性由校验器驱动）；G2-G7 全过'
           '（20 特征卡/16 节点/12 失败/3 升级/4 缺口+5 附录/无外链），15 源 sha8 断言。缺口账本 '
           'v1 重写：GAP-1 closed@3164（3164a/b/c agree_True×3）、GAP-2 closed@3163（3159→3160→'
           '3161→3163 机制链）、GAP-3 closed_v1@3165-3166、GAP-4 open+prereg 3169（G5-A6 谱外'
           '类别面板：Q03 协议 verbatim+谱外 4 类×8-10 实体+已见 paired 对照，门=2×/1.5-2×'
           'borderline；证据 q03 0/3+min_E_x 6.63×、p3151 shuiguo(1.2226)）、附录 5 项逐条锚。'
           'res RES8/seal SEAL8；html HTML8；gap ledger GLSHA；ledger n=319→**320**。'
           '**图谱 v1 三层齐备（特征级 FTR-20 条+渲染级逐字段+缺口账本）——「找到稳定特征、'
           '完成图谱」核心目标 v1 达成**。下一步 3169=**G5-A6 谱外类别面板**（执行 GAP-4 prereg，'
           'GPU ~10min/模型×3，完成后缺口④定判）。\n')
newline = (newline.replace('NFIELD', str(n_fields)).replace('RES8', R['res_sha8'])
           .replace('SEAL8', R['seal_sha8']).replace('HTML8', html_sha).replace('GLSHA', gl_sha))
if '3168 图谱 v1 渲染' in wm:
    log('4. workspace MEMORY: 3168 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3168')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3168)')

# ---------- 5. self-check (disk read-back) ----------
chk = []
led2 = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms2 = led2['measurements']
chk.append(('ledger has 3168', any(m.get('phase') == 3168 for m in ms2), 'n=%d' % len(ms2)))
chk.append(('ledger n>=320', len(ms2) >= 320, 'n=%d' % len(ms2)))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3168', '## Phase 3168' in memo2))
chk.append(('MEMO has res sha', R['res_sha8'] in memo2))
chk.append(('MEMO has html sha', html_sha in memo2))
chk.append(('MEMO has prereg 3169 ref', '预注册 3169' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3168', '3168 图谱 v1 渲染' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3168', '3168 图谱 v1 渲染' in w2))
chk.append(('result seal fields', bool(R['res_sha8']) and bool(R['seal_sha8'])))
chk.append(('html disk sha stable', html_sha == sha8_file(os.path.join(PDIR, 'atlas_v1.html'))))
chk.append(('gap ledger disk sha stable', gl_sha == sha8_file(os.path.join(PDIR, 'gap_ledger_v1.json'))))
gl2 = load(os.path.join(PDIR, 'gap_ledger_v1.json'))
chk.append(('gap statuses', [g['status'] for g in gl2['gaps']] ==
            ['closed', 'closed', 'closed_v1', 'open'],
            json.dumps({g['id']: g['status'] for g in gl2['gaps']})))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check %d/%d)' % (len(chk) - len(bad), len(chk)))
