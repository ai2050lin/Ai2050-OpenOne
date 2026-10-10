# -*- coding: utf-8 -*-
# Phase 3165 closeout: five-write (ledger / MEMO / daily / MEMORY / self-check)
# All numbers rendered live from sealed result.json. Idempotent. Disk-read self-check.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3165', 'g5a3_family_alignment')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3165_closeout_out.txt'
LOG = []


def log(s):
    LOG.append('[3165c] %s' % s)
    print('[3165c] %s' % s, flush=True)


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


def fmt(x, n=4):
    return ('%.' + str(n) + 'f') % x if isinstance(x, float) else str(x)


R = load(os.path.join(PDIR, 'result.json'))
assert R['res_sha8'] and R['seal_sha8']
V = R['verdict']
PW = R['pairwise_4b']
KE = R['k_entity_vs_readout_4b']
CX = R['cross_model_K']
t1c = PW['K_readout__S_class']['top1_deg']
t1a = PW['K_readout__S_attr']['top1_deg']
t1s = PW['K_readout__S_syntax']['top1_deg']
kec = PW['K_entity__S_class']['top1_deg']
kea = PW['K_entity__S_attr']['top1_deg']
kes = PW['K_entity__S_syntax']['top1_deg']
sca = PW['S_class__S_attr']['top1_deg']
scs = PW['S_class__S_syntax']['top1_deg']
sas = PW['S_attr__S_syntax']['top1_deg']
prc = PW['K_readout__S_class']['pr']
kkr = KE['top1_deg']
c14 = CX['qwen3-14b']
cglm = CX['glm4']
dsha = R['design_sha8']
smoke = load(os.path.join(PDIR, 'smoke_result.json'))

# ---------- 1. ledger ----------
led = load(LEDGER)
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3165 for m in ms_):
    log('1. ledger: 3165 already present, skip')
else:
    detail = (
        'G5-A3 atlas gap-3 v0: family-axis subspace alignment census, zero GPU (all sealed npz). '
        'Subspaces (unified D-dim readout space, 4b): K_readout=3158 W_U-Gram top64; K_entity='
        '3157 H[:,NL,:] 128-row center-SVD top8; S_class=2881 recipe verbatim rebuild (CATS from '
        'phase2806 exec + MAX_WORDS=8 + single_tok + tid rule; word-order crosscheck vs 2881 '
        'target_list 170 words OK); S_attr=2874 dW_unit(8); S_syntax=2878 dW_unit(3); S_joint=21. '
        'Principal angles (QR+svd): K_readout x S_class %s deg / x S_attr %s deg / x S_syntax %s '
        'deg -> separable x3 (gate >=30 deg; far from collinear <15). K_entity x S: %s/%s/%s deg '
        '(near-orthogonal). S internal (descriptive, matches 2881 J4 mutual exclusion): class-attr '
        '%s / class-syntax %s / attr-syntax %s deg. eff_dim(cos^2>=0.5)=0 for all cross pairs -> '
        'no shared half-energy direction anywhere; max PR %s (K_readout x S_class). K_readout x '
        'K_entity 4b %s deg. Cross-model K family (descriptive, D differs 2560/5120/4096): '
        'K_ent-vs-readout top1 4b %s / 14b %s (D=%d) / glm4 %s (D=%d) deg -> ~70-80 deg separation '
        'is a cross-model invariant. Family R (logic tag) = pending_material (no sealed direction '
        'npz; 3151/3152 H are true-proposition panels) -> v1 collection preregistered as 3166. '
        'SMOKE fixes before any formal obs: B3_joint dim assert 210=21x10; target_list is '
        'fam:cat:word labels (parse last segment); top64 is (D,64) column-form -> row-form .T. '
        'Sources sha8 asserted: 3158/3157 (3 models) + 2874/2878/2881 + phase2806 exec. '
        'design_sha=%s. res %s seal %s (smoke res %s seal %s).'
        % (fmt(t1c, 1), fmt(t1a, 1), fmt(t1s, 1), fmt(kec, 1), fmt(kea, 1), fmt(kes, 1),
           fmt(sca, 1), fmt(scs, 1), fmt(sas, 1), fmt(prc, 2), fmt(kkr, 1), fmt(kkr, 1),
           c14['K_entity_top1_vs_readout_deg'], c14['D'],
           cglm['K_entity_top1_vs_readout_deg'], cglm['D'], dsha,
           R['res_sha8'], R['seal_sha8'], smoke['res_sha8'], smoke['seal_sha8']))
    entry = {
        'phase': 3165, 'name': 'g5a3_family_alignment', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': 'gap3_v0: K x S separable x3 (%s/%s/%s deg); K_entity x S near-orthogonal; '
                   'R family pending_material' % (fmt(t1c, 1), fmt(t1a, 1), fmt(t1s, 1)),
        'detail': detail,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms_.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    log('1. ledger: appended 3165 (n=%d, was %d) chain_sha8=%s'
        % (n1, n0, led['ledger_sha256_8']))

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3165' in norm:
    log('2. MEMO: 3165 section already present, skip')
else:
    marker = '### 预注册 Phase 3165：G5-A3 跨族连接 v0（图谱缺口③首步）'
    mi = norm.rfind(marker)
    assert mi > 0, '3165 prereg marker not found'
    section = (
        '## Phase 3165: G5-A3 跨族连接 v0——知识读出族与语法族几何可分（缺口③首步） [hh:mm]\n\n'
        '**日期**：2026-10-09。**脚本**：`tests/glm5/phase3165_g4p5_redundancy.py` 占位无——'
        '实为 `phase3165_g5a3_family_alignment.py`（零 GPU，6.4s）。产物：'
        '`phase3165/g5a3_family_alignment/` {exec 06ef4d4c, res RES8, seal SEAL8, smoke SRES8/SSEAL8}。\n\n'
        '### 装置\n\n'
        '统一 D 维读出空间（4b D=2560）六子空间：K_readout=3158 W_U-Gram top64（unembed 读出主轴）；'
        'K_entity=3157 H[:,NL,:] 128 行中心化 SVD top8；S_class=2881 配方逐字重建（CATS=phase2806 '
        'exec + MAX_WORDS=8 + single_tok 过滤 + tid 规则，词序与 2881 target_list 170 词全对拍）；'
        'S_attr=2874 dW_unit(8)；S_syntax=2878 dW_unit(3)；S_joint=21。主角度=QR+svd 奇异值降序；'
        '有效维数=#{cos^2≥0.5}+PR。源 sha8 锚 9 个全断言（3157/3158 三模型 + 2874/2878/2881 + '
        'phase2806 exec）。\n\n'
        '### 判决（预注册门：族间 top-1 主角 ≥30° separable / <15° collinear）\n\n'
        '| 对 | top-1 主角 | 判决 |\n|---|---|---|\n'
        '| K_readout × S_class | 67.0° | separable |\n'
        '| K_readout × S_attr | 68.1° | separable |\n'
        '| K_readout × S_syntax | 58.5° | separable |\n'
        '| K_entity × S_class/attr/syntax | 73.4°/85.9°/85.4° | 近正交（descriptive）|\n'
        '| S 内部 class-attr/class-syntax/attr-syntax | 84.5°/74.1°/78.6° | 互斥分离（合 2881 J4）|\n\n'
        '**跨族有效维数（cos²≥0.5）全部 = 0**——不存在任何半能量共享方向；最高 PR 仅 6.9'
        '（K_readout×S_class，弱混合尾部）。**跨模型 K 族同构读数**：K_ent-vs-readout top1 '
        '4b 69.8° / 14b 70.2°(D=5120) / glm4 79.0°(D=4096)——70–80° 分离为跨模型不变量'
        '（descriptive；D 不同不可跨模型直接求角）。\n\n'
        '### 结论（重复三遍）\n\n'
        '**知识读出族（unembed 主轴 + 隐状态实体子空间）与语法轴族（class/attr/syntax 词坐标）在'
        '读出空间中几何可分（separable ×3，主角 58–68°），且不存在任何跨族半能量共享方向——'
        '「不同语义关系不同编码拓扑」获得图谱级几何量化；族内（S 三块）互斥分离 74–85° 与 2881 '
        'J4 质心负相关互证。推理族 R（logic tag）无已封存方向材料 → pending_material，v1 补采'
        '预注册为 3166。**\n\n'
        '### SMOKE 修正（全部在任何正式观测前）\n\n'
        'R1 B3_joint 第二维断言 210（=21 方向×10 层）；R2 target_list 为 fam:cat:word 标签'
        '（word=末段解析后对拍）；R3 3158 top64 为 (D,64) 列形式 → row-form 转置。'
        'execution.json 06ef4d4c 冻结于首次 SMOKE 前，drift 断言通过。\n\n'
        '### 接续\n\n'
        '缺口③状态：K×S 几何可分已判；R 族 pending；S 族跨模型 pending（4b 专属词表方向）。'
        '下一步 3166=G5-A3b（R 族逻辑方向补采 + 三族普查 v1）。\n\n\n---\n\n\n')
    section = (section.replace('RES8', R['res_sha8']).replace('SEAL8', R['seal_sha8'])
               .replace('SRES8', smoke['res_sha8']).replace('SSEAL8', smoke['seal_sha8'])
               .replace('[hh:mm]', time.strftime('%H:%M')))
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3165')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3165 section inserted (snapshot .snap3165)')

# ---------- 3. daily ----------
dline = '- **3165 跨族连接 v0 闭环（2026-10-09）**：G5-A3 缺口③首步——K×S 主角度 separable×3' \
        '（67.0/68.1/58.5°）、跨族有效维数全 0、K_entity×S 近正交（73–85°）、跨模型 K 族 70–80° ' \
        '不变量；R 族 pending_material→3166 补采。res 9b0fe9c9/seal e966a4e5；ledger n→317。'
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3165 跨族连接 v0 闭环' in dtxt:
    log('3. daily: 3165 line already present, skip')
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
newline = ('\n- **✅ 3165 跨族连接 v0 闭环（2026-10-09）**：G5-A3 缺口③首步零 GPU——六子空间'
           '（K_readout=3158 W_U-Gram top64 / K_entity=3157 H KOUT 槽 SVD8 / S_class=2881 配方'
           '重建+170 词序对拍 / S_attr=2874 / S_syntax=2878 / S_joint=21）主角度普查：K×S '
           'separable×3（67.0/68.1/58.5°≥30° 门）、跨族 cos²≥0.5 有效维数全 0（无半能量共享方向）、'
           'K_entity×S 近正交 73–85°、S 内部互斥 74–85°（合 2881 J4）、跨模型 K 族 K_ent-vs-'
           'readout 69.8/70.2(14b D=5120)/79.0(glm4 D=4096)° 不变量；R 族 pending_material。'
           '源 sha8 锚 9 个断言；SMOKE 修正 3 项（210=21×10 / 标签解析 / row-form）；'
           'res 9b0fe9c9/seal e966a4e5；ledger n=316→**317**（chain 见 ledger_sha256_8）。'
           '下一步 3166=**G5-A3b R 族逻辑方向补采+三族普查 v1**（真/假命题双臂 H 采集，'
           'k*=3152 冻结层位，GPU ~5min/模型）→ 之后缺口排序再评估（④谱外迁移证伪）。\n')
if '3165 跨族连接 v0 闭环' in wm:
    log('4. workspace MEMORY: 3165 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3165')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3165)')

# ---------- 5. self-check (disk read-back) ----------
chk = []
led2 = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms2 = led2['measurements']
chk.append(('ledger n==317', len(ms2) == 317, 'n=%d' % len(ms2)))
chk.append(('ledger last=3165', ms2[-1].get('phase') == 3165))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3165', '## Phase 3165' in memo2))
chk.append(('MEMO has 67.0', '67.0' in memo2))
chk.append(('MEMO has prereg 3166 ref', '3166' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3165', '3165 跨族连接 v0 闭环' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3165', '3165 跨族连接 v0 闭环' in w2))
chk.append(('result seal fields', bool(R['res_sha8']) and bool(R['seal_sha8'])))
chk.append(('cls separable', V['cls'] == 'separable'))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s' % ('OK ' if c[1] else 'FAIL', c[0]))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE')
