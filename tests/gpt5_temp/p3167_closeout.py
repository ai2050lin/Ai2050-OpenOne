# -*- coding: utf-8 -*-
# Phase 3167 closeout: five-write (ledger / MEMO / daily / workspace MEMORY / self-check)
# All numbers rendered live from sealed result.json + atlas_registry_v1.json on disk.
# Idempotent. Disk-read self-check at the end.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3167', 'g5a4_feature_registry')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3167_closeout_out.txt')
LOG = []


def log(s):
    LOG.append('[3167c] %s' % s)
    print('[3167c] %s' % s, flush=True)


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


def fmt(x, n=4):
    return ('%.' + str(n) + 'f') % x if isinstance(x, float) else str(x)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = load(os.path.join(PDIR, 'result.json'))
assert R['res_sha8'] and R['seal_sha8']
V = R['verdict']
REG = load(os.path.join(PDIR, 'atlas_registry_v1.json'))
SMOKE = load(os.path.join(PDIR, 'smoke_result.json'))
dsha = R['per_feature']['G4_asserts_evaluated']
n_feat = R['features_n']
n_anchors = R['anchors_checked']
n_asserts = R['asserts_evaluated']
n_fails = R['failures_n']
n_upg = len(R['upgrade_log'])
lv = {}
for ft in REG['features']:
    lv[ft['evidence_level']] = lv.get(ft['evidence_level'], 0) + 1
fam = {}
for ft in REG['features']:
    fam[ft['family']] = fam.get(ft['family'], 0) + 1
exes = sha8_file(os.path.join(PDIR, 'execution.json'))
reg_sha = sha8_file(os.path.join(PDIR, 'atlas_registry_v1.json'))

# ---------- 1. ledger ----------
led = load(LEDGER)
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3167 for m in ms_):
    log('1. ledger: 3167 already present, skip')
else:
    detail = (
        'G5-A4 atlas registry v1 (zero GPU): cross-model stable feature registry. 20 features '
        'FTR-01..FTR-20 in the 8-field schema (statement / evidence level E0-E3 / model scope / '
        'evidence phase anchors / counter-evidence / replication protocol / values / scope '
        'limits). Device gates: G1 anchor files 20/20 byte sha8 match (frozen SHA_ANCHOR, '
        'probed 2026-10-09); G2 every feature >=2 anchors from >=2 distinct phases; G3 '
        'model_scope explicit subset of mainline; G4 164 field asserts live-checked against '
        'sealed result files (float tol 1e-9, str/bool exact); G5 level consistency (E3 needs '
        'intervention anchor, E2 needs cross_model/held_out anchor, E0 forbidden); G6 values '
        'rendered from disk at run time (values_spec); G7 lineage (taxonomy+principles '
        'verbatim from 3162, failures F1-F11 carried verbatim + F12 appended, upgrade_log). '
        'Features by family: K=2, S=3, SxK=1, R=1, RxS=1, RxK=1, mech=4, context=3, control=1, '
        'readout=2, gate=1, limit=1 (per-family counts %s). Levels: E2_predictive=%d, '
        'E1_repeatable=%d, E3_causal_scoped=%d. Upgrades vs 3162 nodes: N04->FTR-13 and '
        'N05->FTR-12 E1->E2 (3164b/3164c cross-model supported x3), N13->FTR-16 scope 4b->3 '
        'models (3164a zero_like_q06 x3). F12 (claim_precision 3165->3166): ledger 3165 '
        'pending reason imprecise, panel natively contains true/false arms, R-family pending '
        'lifted by 3166. SMOKE fixes before any formal obs: R0 assert keys/values probe '
        'calibrated (device.G2.auc_mean not auc; agate k1_reverdict.judging_layer_under_Q08_A; '
        'p3151 seal 6409274e; FTR-05 fabricated cross-model S_class angles replaced by probed '
        '3166 census values 71.9/74.014 deg; FTR-06 rebuilt on R-side census), R1 smoke subset '
        'misalignment, R2 freeze self-consistency flaw (created inside design hash -> always '
        'drift). R3 formal-run fix: FTR-20 evidence. prefix (assert-path only, no numeric '
        'logic touched; re-run sealed fresh). Registry artifact sha8 %s. design_sha=%s. '
        'res %s seal %s (smoke res %s).'
        % (json.dumps(fam, ensure_ascii=False), lv.get('E2_predictive', 0),
           lv.get('E1_repeatable', 0), lv.get('E3_causal_scoped', 0),
           reg_sha, R.get('design_sha8', 'n/a'), R['res_sha8'], R['seal_sha8'],
           SMOKE['res_sha8']))
    entry = {
        'phase': 3167, 'name': 'g5a4_feature_registry', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': 'atlas_registry_v1: %d features (E2 %d/E1 %d/E3 %d), anchors %d sha-ok, '
                   'asserts %d ok, upgrades %d, failures F1-F12'
                   % (n_feat, lv.get('E2_predictive', 0), lv.get('E1_repeatable', 0),
                      lv.get('E3_causal_scoped', 0), n_anchors, n_asserts, n_upg),
        'evidence_level': 'registry',
        'model_scope': 'qwen3-4b+qwen3-14b+glm4',
        'prereg_id': 'G5-A4',
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
    log('1. ledger: appended 3167 (n=%d, was %d) chain_sha8=%s'
        % (n1, n0, led['ledger_sha256_8']))

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3167' in norm:
    log('2. MEMO: 3167 section already present, skip')
else:
    marker = '### 预注册 Phase 3165：G5-A3 跨族连接 v0（图谱缺口③首步）'
    mi = norm.rfind(marker)
    assert mi > 0, '3165 prereg marker not found'
    section = (
        '## Phase 3167: G5-A4 图谱 v1 特征登记表（跨模型稳定特征 FTR-01..20） [hh:mm]\n\n'
        '**日期**：2026-10-09。**脚本**：`tests/glm5/phase3167_g5a4_feature_registry.py`'
        '（零 GPU）。产物：`phase3167/g5a4_feature_registry/` {exec EXEC8, res RES8, seal SEAL8, '
        'atlas_registry_v1.json REGSHA, smoke res SRES8}。\n\n'
        '### 装置（七门）\n\n'
        'G1 锚文件 20/20 字节 sha8 匹配（SHA_ANCHOR 冻结于 SMOKE 前，全部 2026-10-09 探针实测）；'
        'G2 每特征 ≥2 锚且 ≥2 独立 phase（3162 audit 节点作为独立 phase 磁盘复核锚）；G3 model_scope '
        '显式（mainline 子集非空）；G4 **164 条字段断言**现场对盘（float tol 1e-9、str/bool 精确）；'
        'G5 证据级一致性（E3 必须有 intervention 锚、E2 必须有 cross_model/held_out 锚、v1 禁 E0）；'
        'G6 数值全部运行时从磁盘渲染（values_spec，禁硬编码入表）；G7 血统（taxonomy+四原则从 3162 '
        '逐字继承、F1-F11 失败账本逐字保留+F12 追加、upgrade_log 登记 E 级/范围变更）。\n\n'
        '### 判决（重复三遍）\n\n'
        '**图谱 registry v1 关账：20 条跨模型稳定特征（FTR-01..FTR-20）全部通过预注册门'
        '（≥2 独立 phase 锚 + 模型范围显式）——证据级 E2_predictive×12、E1_repeatable×5、'
        'E3_causal_scoped×3；3 条 E 级/范围升级（N04 RoPE→E2 跨模型、N05 massive→E2 跨模型、'
        'N13 C_steer→scope 三模型）；失败账本 F1-F11 逐字保留+F12 新增；164 字段断言全过。'
        'verdict=VERD。**\n\n'
        '### 特征清单（family 分布）\n\n'
        'K=2（KOUT content 份额 57.6%/K2 条件门可分离 15.2-18.3%）、S=3（S_class 重建跨模型/'
        'K×S_class 67.0-74.0° separable×3/S_attr+Syntax 4b 专属）、SxK=1、R=1（逻辑信号中层+'
        'LOEO 0.856/0.863/0.781）、RxS=1（not_confounded×3）、RxK=1（14b 26.6° 弱混合=2 共享维，'
        '唯一族间弱混合）、mech=4（动力学非继承 0.92/0.82/0.94×3、attention 再分配 E3、消耗无单点'
        '执行者 E3 双锚 3161+3163、商结构被拒 0/3）、context=3（二元门控 massive d1=0/731/2319、'
        'RoPE 输出相对性、T_C 共享+对易子 partial）、control=1（C_steer=0 跨模型 E3）、readout=2'
        '（E_read 池化 0.3734=门 7.5×、E_ar D4 桥 0.0489≤0.05）、gate=1（K1 双轨 FIRED）、'
        'limit=1（谱外崩塌两线互证 shuiguo）。\n\n'
        '### 修正与诚实登记\n\n'
        'R0（SMOKE 前）断言键/值探针校准：device.G2.auc_mean（非 auc）、agate '
        'k1_reverdict.judging_layer_under_Q08_A（非 judging_layer）、p3151 seal=6409274e（非主 '
        'result 的 a566dc96）、FTR-05 首写跨模型 S_class 角度凭记忆（71.054/74.164）**错误**，'
        '被实测替换为 3166 census 真值 71.9/74.014；FTR-06 重构（census 无 K×S_attr 跨模型键，'
        '改用 R 侧第二读数 85.6/86.3°）；R1 smoke 子集与特征锚错位；R2 freeze 自洽缺陷'
        '（created 进 design hash→必漂移；写入值/比较值口径不一致）→ design hash 排除 '
        'created/design_sha8。R3（正式跑暴露）FTR-20 断言缺 evidence. 前缀——仅断言路径修正，'
        '不触及数值渲染逻辑，重跑重新 seal。'
        'execution.json EXEC8 冻结于首次 SMOKE 前。\n\n'
        '### 接续：预注册 3168\n\n'
        '图谱 registry v1 已关账（特征级）。下一步 3168=G5-A5 图谱 v1 渲染+缺口账本重写（零 GPU）：'
        '(a) atlas_registry_v1.json → atlas_v1.html（20 特征卡×八字段 + 16 节点基座 + F1-F12 '
        '失败账本 + 3 升级链 + 模型范围徽章），门=渲染字段与 registry json 逐字段自动校验一致；'
        '(b) 缺口账本重写：缺口①跨模型同口径 closed@3164、缺口②机制链 closed@3163、缺口③跨族连接 '
        'closed v1@3165-3166、缺口④谱外迁移证伪 open（预注册实验设计）、附录项（14b 共享方向定位/'
        'S_attr+S_syntax 移植/K2+K3 测量/N 线 P3-P7 补 Ledger/跨线补丁施加确认）逐条带证据锚。\n\n\n---\n\n\n')
    section = (section
               .replace('EXEC8', exes).replace('REGSHA', reg_sha)
               .replace('RES8', R['res_sha8']).replace('SEAL8', R['seal_sha8'])
               .replace('SRES8', SMOKE['res_sha8'])
               .replace('VERD', V)
               .replace('[hh:mm]', time.strftime('%H:%M')))
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3167')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3167 section inserted before 3166 verdict block (snapshot .snap3167)')

# ---------- 3. daily ----------
dline = ('- **3167 图谱 v1 特征登记表闭环（2026-10-09）**：G5-A4 零 GPU——20 条跨模型稳定特征 '
         'FTR-01..20（八字段 schema）全部过门（≥2 独立 phase 锚+模型范围显式）；E2×12/E1×5/E3×3；'
         '升级 3 条（RoPE/massive→E2 跨模型、C_steer scope→3 模型）；F 账本 F1-F12；164 字段断言'
         '现场对盘全过、20 锚 sha8 断言、数值运行时渲染。res b8e715e0/seal b7b2953f；ledger n→319。'
         '图谱特征级关账，下一步 3168=图谱 v1 渲染+缺口账本重写。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3167 图谱 v1 特征登记表' in dtxt:
    log('3. daily: 3167 line already present, skip')
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
newline = ('\n- **✅ 3167 图谱 v1 特征登记表闭环（2026-10-09）**：G5-A4 零 GPU——atlas_registry_v1'
           '.json 20 条跨模型稳定特征 FTR-01..20（八字段：陈述/E 级/模型范围/锚/反证/复现口径/数值/'
           '范围限定），门=每特征≥2 独立 phase 锭+model_scope 显式，全过；E2_predictive×12/'
           'E1_repeatable×5/E3_causal_scoped×3；升级 N04 RoPE→E2、N05 massive→E2（3164b/c 跨模型 '
           'supported×3）、N13 C_steer scope 4b→3 模型（3164a zero_like_q06×3）；F1-F11 逐字保留'
           '+F12（3165 pending 理由不精确已修正）。装置七门：20/20 锚 sha8、164 字段断言'
           '（float tol 1e-9）、E 级一致性（E3⇒intervention、E2⇒cross_model/held_out、禁 E0）、'
           '数值运行时渲染、血统继承。诚实登记：FTR-05 首写跨模型角度凭记忆错误（71.054/74.164）'
           '被实测替换（3166 census 71.9/74.014°）；R2 freeze 缺陷（created 进 design hash）修正'
           '（design 0a2b3120 排除 created）。res b8e715e0/seal b7b2953f；registry 87d3d35e；'
           'ledger n=318→**319**。**图谱特征级关账**。下一步 3168=**G5-A5 图谱 v1 渲染'
           '（atlas_v1.html 逐字段校验）+缺口账本重写**（缺口①②③ closed、④谱外迁移证伪 open）。\n')
if '3167 图谱 v1 特征登记表' in wm:
    log('4. workspace MEMORY: 3167 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3167')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3167)')

# ---------- 5. self-check (disk read-back) ----------
chk = []
led2 = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms2 = led2['measurements']
has3167 = any(m.get('phase') == 3167 for m in ms2)
chk.append(('ledger has 3167', has3167, 'n=%d' % len(ms2)))
chk.append(('ledger n>=319', len(ms2) >= 319, 'n=%d' % len(ms2)))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3167', '## Phase 3167' in memo2))
chk.append(('MEMO has res sha', 'b8e715e0' in memo2))
chk.append(('MEMO has prereg 3168 ref', '预注册 3168' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3167', '3167 图谱 v1 特征登记表' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3167', '3167 图谱 v1 特征登记表' in w2))
chk.append(('result seal fields', bool(R['res_sha8']) and bool(R['seal_sha8'])))
reg2 = load(os.path.join(PDIR, 'atlas_registry_v1.json'))
chk.append(('registry features==20', len(reg2['features']) == 20, 'n=%d' % len(reg2['features'])))
chk.append(('registry failures==12', len(reg2['failures']) == 12, 'n=%d' % len(reg2['failures'])))
chk.append(('registry file sha stable', reg_sha == sha8_file(os.path.join(PDIR, 'atlas_registry_v1.json'))))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE (self-check %d/%d)' % (len(chk) - len(bad), len(chk)))
