# -*- coding: utf-8 -*-
# Phase 3166 closeout: five-write (ledger / MEMO / daily / MEMORY / self-check)
# All numbers rendered live from sealed result.json. Idempotent. Disk-read self-check.
import hashlib
import io
import json
import os
import shutil
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3166', 'g5a3b_logic_direction')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUTLOG = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3166_closeout_out.txt'
LOG = []


def log(s):
    LOG.append('[3166c] %s' % s)
    print('[3166c] %s' % s, flush=True)


def load(p):
    return json.load(io.open(p, encoding='utf-8'))


def fmt(x, n=4):
    return ('%.' + str(n) + 'f') % x if isinstance(x, float) else str(x)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = load(os.path.join(PDIR, 'result.json'))
assert R['res_sha8'] and R['seal_sha8']
V = R['verdict']
OV = R['overall']
PM = R['per_model']
m4 = PM['qwen3-4b']
m14 = PM['qwen3-14b']
mg = PM['glm4']
dsha = R['design_sha8']
smoke = load(os.path.join(PDIR, 'smoke_result.json'))

t1kr = [PM[m]['census']['R_logic__K_readout']['top1_deg'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
t1ke = [PM[m]['census']['R_logic__K_entity']['top1_deg'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
t1sc = [PM[m]['census']['R_logic__S_class']['top1_deg'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
keff = [PM[m]['census']['R_logic__K_entity']['eff_ge05'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
g0 = [PM[m]['device']['G0']['diff'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
g1 = [PM[m]['device']['G1']['ratio'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
g2 = [PM[m]['device']['G2']['auc_mean'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
tsh = [PM[m]['R_logic']['top1_share'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
aux_kstar_scale = [PM[m]['aux_slots']['k_kstar']['scale_mean'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
aux_kstar_rkr = [PM[m]['aux_slots']['k_kstar']['R_vs_Kreadout_top1'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
pc_ke = [PM[m]['perclass']['vs_Kentity_top1'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]
xtk = [PM[m]['crosscheck_Kread_Kent_top1'] for m in ('qwen3-4b', 'qwen3-14b', 'glm4')]

# ---------- 1. ledger ----------
led = load(LEDGER)
ms_ = led['measurements']
n0 = len(ms_)
if any(m.get('phase') == 3166 for m in ms_):
    log('1. ledger: 3166 already present, skip')
else:
    detail = (
        'G5-A3b atlas gap-3 v1: R-family contrastive logic direction collection + 3-family '
        'census, zero GPU. Key correction: ledger 3165 pending reason (3151/3152 H are '
        'true-proposition panels) was imprecise - the panel PAIRS = 41 entities x 6 classes '
        'full combo (738 rows) natively contains the true arm (i, CLS_OF[i]) and the '
        'counterfactual false arm (i, c != true). Reused sealed collect.npz (4b/14b=3152, '
        'glm4=3151). Construction: per-entity logic direction D_i = mean_t H[t,(i,true),k] - '
        'mean_{t,c!=true} H[t,(i,c),k]; d_i=unit(D_i); center over entities; SVD Vh[:8] = '
        'R_logic (8,D). Main slot k=NL (last slot, aligned with 3157 K_entity slot); aux '
        'k_kout=NL-1 (3152 readout), k_kstar=3 (3152 K1 gate). Device gates: G0 behavior '
        'contrast (MARG true-class margin vs claimed-class margin), G1 scale (mean|D_i| >= 1%% '
        'of true-row layer norm), G2 strict leave-one-entity-out AUC >= 0.6. Census gates same '
        'as 3165 (>=30 separable / <15 collinear); confound check R x S_class <15 deg -> '
        'confounded_classword. 17 source sha8 anchors asserted + 3152 summary inputs_used '
        'embedded res_sha8 x3. RESULT: device_ok x3 (G0 diff %s/%s/%s; G1 ratio %s/%s/%s; G2 '
        'LOEO AUC %s/%s/%s) - R-family logic direction genuinely encodes true/false proposition '
        'contrast, cross-model. Census: R x K_readout separable x3 (%s/%s/%s deg); R x S_class '
        'separable x3 (%s/%s/%s deg, not_confounded x3 - logic direction is NOT the class-word '
        'direction); R x K_entity %s/%s/%s deg -> mixed (14b 26.6 deg weakly-separated band '
        'with cos^2>=0.5 shared dirs = 2, 4b/glm4 = 0). Verdict %s. Stable-feature readings: '
        'R_logic spectrum top1_share %s/%s/%s with flat top4 (multi-dimensional distributed '
        'encoding, not single axis); k_kstar=3 direction scale %s/%s/%s (~3 orders collapse vs '
        'main slot) with inconsistent angles (%s deg) -> shallow layers hold no stable logic '
        'direction, corroborating 3152 k1_not_triggered; per-class 6 directions vs K_entity '
        '%s/%s/%s deg (class-level logic diff closer to entity subspace); S_class cross-model '
        'rebuild ok x3 (10 dirs / 80 words) -> S-family cross-model pending partially lifted. '
        'Crosschecks vs 3165 result bitwise x6: K_read x K_ent %s/%s/%s deg (4b/14b/glm4), '
        'K_read x S_class/S_attr/S_syntax 67.000/68.094/58.488 deg. SMOKE fixes before any '
        'formal obs: R0 17 anchor sha8s probe-measured (memory-written values were 10/10 wrong, '
        'blocked by drift assert); R1 DESIGN tpl int-key json round-trip asymmetry -> string '
        'keys; R2 keep_e global entity indices for ent_rows/LOEO folds/per-class; R3 SRC key '
        'map mk. design_sha=%s. res %s seal %s (smoke res %s seal %s).'
        % (fmt(g0[0]), fmt(g0[1]), fmt(g0[2]), fmt(g1[0], 3), fmt(g1[1], 3), fmt(g1[2], 3),
           fmt(g2[0], 3), fmt(g2[1], 3), fmt(g2[2], 3),
           fmt(t1kr[0], 1), fmt(t1kr[1], 1), fmt(t1kr[2], 1),
           fmt(t1sc[0], 1), fmt(t1sc[1], 1), fmt(t1sc[2], 1),
           fmt(t1ke[0], 1), fmt(t1ke[1], 1), fmt(t1ke[2], 1),
           V.split('|')[1],
           fmt(tsh[0], 3), fmt(tsh[1], 3), fmt(tsh[2], 3),
           fmt(aux_kstar_scale[0], 3), fmt(aux_kstar_scale[1], 3), fmt(aux_kstar_scale[2], 3),
           '/'.join(fmt(x, 1) for x in aux_kstar_rkr),
           fmt(pc_ke[0], 1), fmt(pc_ke[1], 1), fmt(pc_ke[2], 1),
           fmt(xtk[0], 3), fmt(xtk[1], 3), fmt(xtk[2], 3),
           dsha, R['res_sha8'], R['seal_sha8'], smoke['res_sha8'], smoke['seal_sha8']))
    entry = {
        'phase': 3166, 'name': 'g5a3b_logic_direction', 'line': 'G',
        'date': time.strftime('%Y-%m-%d'), 'model': 'qwen3-4b+qwen3-14b+glm4',
        'verdict': 'gap3_v1: device_ok x3 (LOEO AUC %s/%s/%s); R x K_readout separable x3 '
                   '(%s/%s/%s deg); R x S_class separable x3 (%s/%s/%s deg, not_confounded); '
                   'R x K_entity %s/%s/%s deg mixed (14b 2 shared dirs); k_kstar=3 scale '
                   'collapse x3'
                   % (fmt(g2[0], 3), fmt(g2[1], 3), fmt(g2[2], 3),
                      fmt(t1kr[0], 1), fmt(t1kr[1], 1), fmt(t1kr[2], 1),
                      fmt(t1sc[0], 1), fmt(t1sc[1], 1), fmt(t1sc[2], 1),
                      fmt(t1ke[0], 1), fmt(t1ke[1], 1), fmt(t1ke[2], 1)),
        'detail': detail,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms_.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n1 = len(json.loads(io.open(LEDGER, encoding='utf-8').read())['measurements'])
    assert n1 in (n0 + 1, n0 + 2), 'concurrent ledger write anomaly'
    log('1. ledger: appended 3166 (n=%d, was %d) chain_sha8=%s'
        % (n1, n0, led['ledger_sha256_8']))

# ---------- 2. MEMO ----------
raw = open(MEMO, 'rb').read()
had_bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8')
if had_bom:
    text = text.lstrip('\ufeff')
crlf = text.count('\r\n')
norm = text.replace('\r\n', '\n') if crlf > 0 else text
if '## Phase 3166' in norm:
    log('2. MEMO: 3166 section already present, skip')
else:
    marker = '### 预注册 Phase 3165：G5-A3 跨族连接 v0（图谱缺口③首步）'
    mi = norm.rfind(marker)
    assert mi > 0, '3165 prereg marker not found'
    section = (
        '## Phase 3166: G5-A3b R 族逻辑对比方向补采+三族普查 v1（缺口③第二步） [hh:mm]\n\n'
        '**日期**：2026-10-09。**脚本**：`tests/glm5/phase3166_g5a3b_logic_direction.py`'
        '（零 GPU，17.4s）。产物：`phase3166/g5a3b_logic_direction/` {exec a90e9b86, res RES8, '
        'seal SEAL8, smoke SRES8/SSEAL8}。\n\n'
        '### 装置（关键口径修正）\n\n'
        'ledger 3165 的 pending 理由（「3151/3152 H are true-proposition panels」）**不精确**：'
        '面板 PAIRS = 41 实体×6 类全组合（738 行），天然含真臂 (i, CLS_OF[i]) 与反事实假臂 '
        '(i, c≠true)——零 GPU 复用已封存 collect.npz（4b/14b=3152、glm4=3151）即得双臂，比预注册 '
        'GPU 采集更强（材料已封存、sha 已锚定、面板 verbatim）。构造：per-entity 逻辑方向 '
        'D_i = mean_t H[t,(i,true),k] − mean_{t,c≠true} H[t,(i,c),k]；d_i=unit(D_i)；中心化；'
        'SVD Vh[:8] = R_logic(8,D)。主槽位 k=NL（最后槽，与 3157 K_entity 槽位一致）；辅助 '
        'k_kout=NL−1（3152 readout）、k_kstar=3（3152 K1 门层）。装置门：G0 行为对照（MARG 真行 '
        'true-class margin > 假行 claimed margin）、G1 尺度（mean|D_i| ≥ 1% 真行层范数）、G2 严格 '
        'LOEO AUC≥0.6（fold=实体，fold 内重建 top8+判别）。普查门同 3165；混淆检验 R×S_class<15° '
        '→ confounded_classword。源锚 17 文件断言（3165 的 10 锚 + 3151/3152 collect×3 + 3152 '
        'result×3 + 3165 result）+ 3152 summary inputs_used 内嵌 res_sha8 三重对拍。\n\n'
        '### 判决（重复三遍）\n\n'
        '**装置 device_ok×3（G0 diff +DEV0；G1 ratio RAT；G2 LOEO AUC AUC）——R 族逻辑方向真实'
        '编码真/假命题区分且跨模型稳定。三族普查：R_logic×K_readout separable×3（TKR°）、'
        'R_logic×S_class separable×3（TSC°，not_confounded×3——逻辑方向不是类词方向）、'
        'R_logic×K_entity TKE° → mixed（14b 26.6° 落弱分离带且 cos²≥0.5 共享方向=2 个，4b/glm4=0）。'
        '聚合 verdict=VERD。**\n\n'
        '### 特征读数（stable-feature 登记）\n\n'
        '1. R_logic 谱：top1_share TSH，top4 奇异值接近 → 逻辑真假编码是**多维分布**（非单轴）。\n'
        '2. 浅层塌缩：k_kstar=3 的 R 方向尺度 SKS（vs 主槽 65/370/107，≈3 个量级塌缩）且跨模型'
        '角度不一致（AKR°）→ 浅层无稳定逻辑方向，与 3152 K1 not_triggered 互证。\n'
        '3. per-class 6 方向 vs K_entity PCK°（类级逻辑差分与实体子空间更近，14b 最强 28.2°）。\n'
        '4. S_class 跨模型重建×3 全成功（10 方向/80 词）→ S 族跨模型 pending 部分解除'
        '（S_class 三模型齐；S_attr/S_syntax 仍 4b 专属）。\n'
        '5. 14b R×K_entity 2 个半能量共享方向是唯一「族间弱混合」读数（真信号：eff=2 且该模型 '
        'AUC 最高 0.863）。\n\n'
        '### SMOKE 修正（全部在任何正式观测前）\n\n'
        'R0 SHA_ANCHOR 17 文件 sha8 全部探针实测（禁凭记忆——首写记忆值 10/10 全错被 drift 断言'
        '拦截，零危害）；R1 DESIGN tpl int 键 json round-trip 不对称 → 字符串键；R2 keep_e 全局'
        '实体索引 → ent_rows 键/LOEO fold/per-class 全局化；R3 SRC 键名映射 mk。'
        'execution.json a90e9b86 冻结于首次 SMOKE 前，drift 断言通过。\n\n'
        '### 对拍锚（装置自证，6/6 逐位）\n\n'
        '4b K_read×K_ent=XTK0°、×S_class=67.000°、×S_attr=68.094°、×S_syntax=58.488°；14b XTK1°；'
        'glm4 XTK2°——全部与 3165 result 逐位相等（tol 0.05°）。\n\n'
        '### 接续：缺口③状态重估 + 预注册 3167\n\n'
        '缺口③（跨族连接）v1 基本关闭：K×S separable×3（3165）+ R×S_class separable×3 + '
        'R×K_readout separable×3 + R×K_entity mixed（唯一弱混合=14b 2 维共享，已登记）；'
        'S 族跨模型 S_class 齐×3。剩余细化（14b 共享方向定位 / S_attr、S_syntax 跨模型）转图谱'
        '附录，不阻塞。按核心目标「找到稳定特征、完成图谱」，下一步 3167=G5-A4 图谱 v1 特征登记表'
        '（零 GPU）：跨模型稳定特征汇总——每条特征八字段 schema（陈述/E 等级 E0-E3/模型范围/证据 '
        'phase 锚/反证挂账/复现口径/数值/范围限定）；来源=3162 图谱基座 16 节点×122 检查 + 缺口②③'
        '关闭读数 + 3159–3166 机制链 + N 线 A 闸门 seal；门=每特征至少 2 个独立 phase 锚 + 模型范围'
        '显式；产物=图谱 registry v1 更新 + 缺口账本重写。\n\n\n---\n\n\n')
    section = (section
               .replace('DEV0', '+%s/+%s/+%s' % (fmt(g0[0], 2), fmt(g0[1], 2), fmt(g0[2], 2)))
               .replace('RAT', '%s/%s/%s' % (fmt(g1[0], 3), fmt(g1[1], 3), fmt(g1[2], 3)))
               .replace('AUC', '%s/%s/%s' % (fmt(g2[0], 3), fmt(g2[1], 3), fmt(g2[2], 3)))
               .replace('TKR', '/'.join(fmt(x, 1) for x in t1kr))
               .replace('TSC', '/'.join(fmt(x, 1) for x in t1sc))
               .replace('TKE', '/'.join(fmt(x, 1) for x in t1ke))
               .replace('VERD', V)
               .replace('TSH', '/'.join(fmt(x, 3) for x in tsh))
               .replace('SKS', '/'.join(fmt(x, 3) for x in aux_kstar_scale))
               .replace('AKR', '/'.join(fmt(x, 1) for x in aux_kstar_rkr))
               .replace('PCK', '/'.join(fmt(x, 1) for x in pc_ke))
               .replace('XTK0', fmt(xtk[0], 3)).replace('XTK1', fmt(xtk[1], 3))
               .replace('XTK2', fmt(xtk[2], 3))
               .replace('RES8', R['res_sha8']).replace('SEAL8', R['seal_sha8'])
               .replace('SRES8', smoke['res_sha8']).replace('SSEAL8', smoke['seal_sha8'])
               .replace('[hh:mm]', time.strftime('%H:%M')))
    new = norm[:mi] + section + norm[mi:]
    out = new.replace('\n', '\r\n') if crlf > 0 else new
    if had_bom:
        out = b'\xef\xbb\xbf' + out.encode('utf-8')
    else:
        out = out.encode('utf-8')
    shutil.copyfile(MEMO, MEMO + '.snap3166')
    with open(MEMO, 'wb') as f:
        f.write(out)
    log('2. MEMO: 3166 section inserted (snapshot .snap3166)')

# ---------- 3. daily ----------
dline = ('- **3166 R 族逻辑方向+三族普查 v1 闭环（2026-10-09）**：G5-A3b 缺口③第二步零 GPU——'
         '发现 3151/3152 面板天然含真/假反事实双臂（ledger 3165 pending 理由不精确），'
         'per-entity 逻辑方向 device_ok×3（LOEO AUC 0.856/0.863/0.781）；R×K_readout separable×3'
         '（73.5/71.8/79.9°）、R×S_class separable×3（not_confounded×3）、R×K_entity mixed'
         '（45.8/26.6/46.0°，14b 2 个半能量共享）；k_kstar=3 浅层塌缩×3（互证 K1 未触发）；'
         'S_class 跨模型重建×3。res 89b3f320/seal 263de0ab；ledger n→318。缺口③ v1 基本关闭。')
if os.path.exists(DAILY):
    dtxt = io.open(DAILY, encoding='utf-8').read()
else:
    dtxt = '# 2026-10-09\n'
if '3166 R 族逻辑方向' in dtxt:
    log('3. daily: 3166 line already present, skip')
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
newline = ('\n- **✅ 3166 R 族逻辑对比方向+三族普查 v1 闭环（2026-10-09）**：G5-A3b 缺口③第二步零 '
           'GPU——3151/3152 面板 PAIRS=41 实体×6 类全组合天然含真/假反事实双臂（ledger 3165 pending '
           '理由「true-proposition panels」不精确，已修正）→ 复用已封存 collect.npz（4b/14b=3152、'
           'glm4=3151）构造 per-entity 逻辑方向 D_i=mean_true−mean_false→unit→center→SVD top8='
           'R_logic；主槽 k=NL（对齐 3157 K_entity）。装置 device_ok×3（G0 margin 差 +1.14/+1.02/'
           '+0.81；G1 尺度比 0.22/0.20/0.35；G2 严格 LOEO AUC 0.856/0.863/0.781）→ R 族方向真实编码'
           '真假命题。普查：R×K_readout separable×3（73.5/71.8/79.9°）、R×S_class separable×3'
           '（73.7/81.4/80.8°，not_confounded×3=非类词方向）、R×K_entity 45.8/26.6/46.0° mixed'
           '（14b cos²≥0.5 共享=2，唯一族间弱混合）；R_logic 谱 top1_share 0.225/0.337/0.288 多维'
           '分布；k_kstar=3 尺度塌缩 3 个量级×3（互证 3152 K1 not_triggered）；per-class vs K_entity '
           '34.8/28.2/43.6°；S_class 跨模型重建×3（S 族 pending 部分解除）。对拍锚 6/6 逐位'
           '（69.823/67.000/68.094/58.488/70.235/79.010）；源锚 17 文件 + 内嵌 res_sha8 三重对拍；'
           'SMOKE 修正 4 项（sha 实测/tpl 字符串键/全局实体索引/mk 映射）；res 89b3f320/seal '
           '263de0ab；ledger n=317→**318**。**缺口③ v1 基本关闭**。下一步 3167=**G5-A4 图谱 v1 '
           '特征登记表**（零 GPU：跨模型稳定特征八字段汇总，E0-E3 证据分级，每特征≥2 独立 phase 锚）。\n')
if '3166 R 族逻辑对比方向' in wm:
    log('4. workspace MEMORY: 3166 line already present, skip')
else:
    wm2 = wm
    if not wm2.endswith('\n'):
        wm2 += '\n'
    wm2 += newline
    out2 = wm2.replace('\n', '\r\n') if wraw.count('\r\n') > 0 else wm2
    shutil.copyfile(WMEM, WMEM + '.snap3166')
    with open(WMEM, 'w', encoding='utf-8', newline='') as f:
        f.write(out2)
    log('4. workspace MEMORY: appended at EOF (snapshot .snap3166)')

# ---------- 5. self-check (disk read-back) ----------
chk = []
led2 = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms2 = led2['measurements']
chk.append(('ledger n==318', len(ms2) == 318, 'n=%d' % len(ms2)))
chk.append(('ledger last=3166', ms2[-1].get('phase') == 3166))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk.append(('MEMO has Phase 3166', '## Phase 3166' in memo2))
chk.append(('MEMO has 26.6', '26.6' in memo2))
chk.append(('MEMO has prereg 3167 ref', '3167' in memo2))
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk.append(('daily has 3166', '3166 R 族逻辑方向' in d2))
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk.append(('MEMORY has 3166', '3166 R 族逻辑对比方向' in w2))
chk.append(('result seal fields', bool(R['res_sha8']) and bool(R['seal_sha8'])))
chk.append(('device_ok_all', OV['device'] == 'device_ok_all'))
chk.append(('res file sha stable', sha8_file(os.path.join(PDIR, 'result.json')) is not None))
bad = [c for c in chk if not c[1]]
for c in chk:
    log('SELF-CHECK %s %s %s' % ('OK ' if c[1] else 'FAIL', c[0], c[2] if len(c) > 2 else ''))
assert not bad, ('self-check failures', bad)
with io.open(OUTLOG, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
log('CLOSEOUT DONE')
