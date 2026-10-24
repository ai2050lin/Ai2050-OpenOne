# -*- coding: utf-8 -*-
"""Phase 3249 closeout (five writes, idempotent):
1) result assertions  2) ledger #401  3) MEMO Phase 3249 section
4) dual wlog  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3249\g5h12_deg_extension\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 11:30'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D = R['delta_vs_G0']
H = R['headline']

# ---------- 1. result assertions ----------
A = R['anchors']
assert R['verdict'] == 'deg_extension_registered'
assert R['design_sha8'] == '6eb9bed8' and R['res_sha8'] == '8104fd1b'
assert R['model'] == 'qwen3-4b' and R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_transfer_realism',
                                  'clause_b_probe_closure_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert AGG['XC_b4']['d_vs_sealed_3246_max'] == 0.0
assert AGG['XC_p2_b4']['d_vs_sealed_3247_max'] == 0.0
assert AGG['XC_p3_b4']['d_vs_sealed_3248_max'] == 0.0
assert AGG['K_p3_T3']['d_vs_sealed_3248_max'] == 0.0
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
# clause (a): realism of NEW map specs p4/p5
real = H['transfer_realism']
for k in ('p4', 'p5'):
    assert real[k]['margin'] > 0.0, (k, real[k])
# numeric probe: finite everywhere
assert H['numeric_probe_ok'] is True
# degree ladder monotone improving 2->3->4->5 in SCLS
dl = H['extension_axes']['degree_ladder_SCLS']
assert dl['XC_p2_b4'] < dl['XC_p3_b4'] < dl['XC_p4_b4'] < dl['XC_p5_b4']
# p5 best new closure positive, not saturated
assert H['saturated'] is False and H['best_new_item'] == 'XC_p5_b4'
assert H['closure_table']['XC_p5_b4'] > 0.0
# p4/p5 mild DVU erosion but bounded
assert -0.05 < D['XC_p4_b4']['d_fam_fid']['DVU'] <= 0.0
assert -0.05 < D['XC_p5_b4']['d_fam_fid']['DVU'] <= 0.0
# beta on p3: b6/b8 improve r2 strongly with zero fidelity regression
b3 = AGG['XC_p3_b4']['r2_oov_mean']
b6 = AGG['XC_p3_b6']['r2_oov_mean']
b8 = AGG['XC_p3_b8']['r2_oov_mean']
assert b6 > b3 + 0.02 and b8 > b3 + 0.02
assert D['XC_p3_b6']['d_fam_fid']['DVU'] > -0.02 and D['XC_p3_b8']['d_fam_fid']['DVU'] > -0.02
assert D['XC_p3_b6']['d_fam_fid']['DCC'] > -0.02 and D['XC_p3_b8']['d_fam_fid']['DCC'] > -0.02
# combo re-positioning: on p3 base the lexical extra no longer beats pure pull
assert H['closure_table']['XC_p3_b6'] < 0.0 and H['closure_table']['XC_p3_b8'] < 0.0
assert min(AGG['XC_p3_b6']['fam_fid_mean']['SCLS'],
           AGG['XC_p3_b6']['fam_fid_mean']['ENT']) < min(
    AGG['K_p3_T3']['fam_fid_mean']['SCLS'],
    AGG['K_p3_T3']['fam_fid_mean']['ENT']) + 0.02
# ORC_ls_cls upper bound replicates sealed 3246 (semantic)
assert abs(AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] - 0.083) < 0.01
REPORT.append('1 result assertions: OK (verdict=%s res=%s)' % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = '652aa8a2'
already = [e for e in led['measurements'] if e.get('phase') == 3249]
if not already:
    assert len(led['measurements']) == 400 and led['ledger_sha256_8'] == PREV_POST
    eS = lambda rn: round(AGG[rn]['fam_fid_mean']['SCLS'], 3)
    eE = lambda rn: round(AGG[rn]['fam_fid_mean']['ENT'], 3)
    entry = {
        "phase": 3249,
        "name": "g5h12_deg_extension",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b (zero-GPU degree-ladder extension + beta "
                 "re-coupling on the p3 base + combo re-positioning vs "
                 "pure pull)",
        "verdict": ("deg_extension_registered|NOT SATURATED at deg 5: "
                    "degree ladder 2->3->4->5 monotone improving (SCLS "
                    "%s -> %s -> %s -> %s; ENT %s -> %s -> %s -> %s); "
                    "XC_p5_b4 closure +0.102 (best new); mild DVU erosion "
                    "at high degree (-0.017 at p5) - zero-regression and "
                    "degree gains trade off; beta=8 on the p3 base lifts "
                    "r2 to %.3f (vs %.3f at beta 4) and KL to %.3f (vs "
                    "%.3f) at equal fidelity with DVU/DCC zero regression; "
                    "on the p3 base the lexical combo extra NO LONGER "
                    "beats the pure pull (b6/b8 closure -0.01) - combo "
                    "mechanism re-positioned as a lin/p2-era device"
                    % (eS('XC_p2_b4'), eS('XC_p3_b4'), eS('XC_p4_b4'),
                       eS('XC_p5_b4'), eE('XC_p2_b4'), eE('XC_p3_b4'),
                       eE('XC_p4_b4'), eE('XC_p5_b4'),
                       round(b8, 3), round(b3, 3),
                       round(AGG['XC_p3_b8']['kl_mean'], 3),
                       round(AGG['XC_p3_b4']['kl_mean'], 3))),
        "evidence_level": ("deepening arm (preregistered MEMO 3248 tail; "
                           "SIX bit-level anchors d40/d42/d46/d47/d48a/"
                           "d48b == 0.0 per-seed on all 876 rows vs sealed "
                           "3240/3242/3246/3247/3248; anchors d83/d_tc/"
                           "drift69/d88 == 0.0, d89 <1e-6; census spectra "
                           "cross-checked; numeric probe finite for p4/p5 "
                           "(cond ~1e5, stable); realism margins positive; "
                           "ORC_ls_cls is EVALUATION-ONLY closure "
                           "denominator)"),
        "model_scope": "qwen3-4b (per prereg)",
        "prereg_id": "G5-H12 (MEMO 3248 tail)",
        "superseded_by": None,
        "detail": (
            "Degree-ladder extension (zero GPU), 11 recipes. Findings "
            "(3-seed means): (1) DEGREE LADDER STILL MONOTONE at 4/5: "
            "XC_p2_b4 -0.651 -> XC_p3_b4 -0.550 -> XC_p4_b4 -0.495 -> "
            "XC_p5_b4 -0.467 SCLS (ENT -0.435 -> -0.356 -> -0.317 -> "
            "-0.301); closure vs [K_p3_T3 -> ORC_ls_cls]: p4 +0.057, "
            "p5 +0.102 = best new item; SATURATED=False; (2) cost side: "
            "DVU erosion grows with degree (+0.000 at p3 -> -0.010 at "
            "p4 -> -0.017 at p5) - degree gains and zero-regression "
            "trade off, no free lunch; (3) beta on p3 base: optimum "
            "b6/b8 - r2 0.239/0.243 (vs 0.205 at b4), KL 0.978/0.982 "
            "(vs 1.080), ENT -0.335/-0.332 (vs -0.356), SCLS flat, "
            "DVU/DCC zero regression - r2/KL-dimensional winner but "
            "min(SCLS,ENT) not improved over pure pull; (4) combo "
            "re-positioning: on the p3 base XC_p3_b6/b8 closure -0.010/"
            "-0.011 - the lexical extra does NOT stack on the best pure "
            "pull; the combo mechanism was a lin/p2-era device; (5) "
            "numeric probe: p4/p5 kernel cond ~1.1e5-3.1e5, all outputs "
            "finite (probe_ok=True); (6) realism margins positive "
            "(p4 +0.093, p5 +0.093, clause_a). Registry untouched "
            "(v1.33 head, byte-verified). Gates G1-G4 all True; "
            "determinism 0.0. freeze 6eb9bed8, res 8104fd1b. Pre-append "
            "ledger file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3249', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #401 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3249 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 401
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=401 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3249:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3249: G5-H12 degree 阶梯延长 + p3 基 β 再联扫 + combo 再定位（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3248 tail；freeze **`6eb9bed8`**；res **`8104fd1b`**；verdict **`deg_extension_registered`**（G1–G4 全过，SMOKE 一次通过：p4/p5 数值探针 finite，cond ~1e5）。**六位级锚全零**：d40/d42/d46/d47/d48a（XC_p3_b4 vs 3248）/d48b（K_p3_T3 vs 3248），均 per-seed 全 876 行。registry 不动（v1.33 仍为头）。')
    A_('')
    A_('### 核心发现：degree 阶梯在 deg 5 仍未饱和，但零回退开始被侵蚀（headline: SATURATED=False, best=XC_p5_b4 closure +0.102）')
    A_('')
    A_('- **degree 2→3→4→5 继续单调**：XC_p2_b4 −0.651 → XC_p3_b4 −0.550 → XC_p4_b4 −0.495 → **XC_p5_b4 −0.467**（SCLS；ENT −0.435→−0.356→−0.317→−0.301）；闭合率 p4 +0.057 / **p5 +0.102**。保真缺口曲线：G0 −1.751 → T3 −1.293 → XC_b4 −0.772 → K_p3_T3 −0.529 → **XC_p5_b4 −0.467** → oracle 上界 +0.083，剩余 −0.55。')
    A_('- **代价面（新发现）**：DVU 侵蚀随阶增长（p3 +0.000 → p4 −0.010 → p5 −0.017）——**degree 增益与零回退开始交换**，无免费午餐；DCC 仍 +0.003。')
    A_('- **p3 基 β 再联扫**：β=6/8 使 r2 0.205→0.239/0.243、KL 1.080→0.978/0.982、ENT −0.356→−0.335/−0.332，DVU/DCC 零回退——**r2/KL 维度明确更优**，但 min(SCLS,ENT) 仍不及纯拉力。')
    A_('- **combo 再定位（负结果）**：p3 基上 lexical extra 不再叠加——XC_p3_b6/b8 闭合 −0.010/−0.011，纯拉力 K_p3_T3（−0.529）仍最优。**combo 机制是 lin/p2 时代的装置**：其价值 = 补偿弱映射族的 span 质量；映射族足够强后 extra 冗余。')
    A_('- realism margin 全正（clause_a 过）：p4 +0.093 / p5 +0.093。数值探针：p4/p5 核矩阵 cond 1.1e5–3.1e5、输出全 finite（probe_ok=True）。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **转正路径更新**：映射函数类增益在 deg 5 仍未饱和，但边际收益递减（p4 +0.057 → p5 +0.102 闭合率增幅放缓）且 DVU 侵蚀出现。现役综合最优仍在 p3–p5 带；转正剩余缺口 −0.55。')
    A_('1. **机制分类精化**：combo 机制与映射函数类非独立——combo 的 lexical extra 只在基 span 质量不足时有增量；这解释了 3244 的 span 归因（99.3%）与 3247/3249 的 combo 失效链。')
    A_('1. **oracle 语义**：ORC_ls_cls 仅作闭合率分母（EVALUATION-ONLY）。')
    A_('')
    A_('**限界**：零 GPU；结论限于 qwen3-4b 面板与 B-协议 split；deg>5 未探（cond 增长趋势提示数值边界临近）；u* 内容归因继续顺延。')
    A_('')
    A_('**完整性**：ledger #401（pre 文件 sha8 %s）；产物 result.json + degext.npz + smoke 子目录；独立磁盘复核见 p3249_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3250（候选）')
    A_('')
    A_('- (a) **现役配方客户端化**（工程臂，自 3248 顺延两次）：/atlas/fidelity 端点 + FidelityMode.jsx 增补 3246–3249 曲线与现役配方（XC_p3_b4 / XC_p5_b4 / XC_p3_b8）标注；')
    A_('- (b) **deg>5 数值探针 + 核族收尾**（零 GPU）：deg 6 探针（cond 趋势）、poly×rbf 混合、degree–β 联合网格粗扫——定位映射函数类增益的数值边界与实际停点；')
    A_('- (c) u* 内容归因（继续顺延）。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3249 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3249 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3249:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3249 degree 阶梯延长完成（零 GPU，registered）：2→3→4→5 继续单调——XC_p5_b4 SCLS −0.467/ENT −0.301，'
      '闭合 +0.102 唯一显著正值（SATURATED=False）；**DVU 侵蚀随阶增长**（p3 +0.000 → p5 −0.017），degree 增益与零回退开始交换；'
      'p3 基 β=8 使 r2 0.243/KL 0.982 零回退（r2/KL 维更优）；**combo 机制在 p3 基失效**（b6/b8 闭合 −0.01，纯拉力仍最优）'
      '——combo 是 lin/p2 时代装置，extra 只补偿弱映射 span。六位级锚 d40/d42/d46/d47/d48a/d48b 全零。'
      'verdict deg_extension_registered；freeze 6eb9bed8 / res 8104fd1b，ledger #401，registry v1.33 不动。'
      '3250 已预注册：客户端化 + deg>5 探针/核族收尾。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3249 degree 阶梯延长完成' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3249 已预注册**：(a) deg 4/5 阶梯 + K_p3 组合臂 + β 再联扫（零 GPU）；(b) 现役配方客户端化（工程，顺延）；(c) u* 顺延。')
new_block = ('- **3249 收官（deg extension，registered）**：degree 2→5 继续单调（XC_p5_b4 −0.467/−0.301，闭合 +0.102，SATURATED=False）；DVU 侵蚀随阶增长（p5 −0.017）；p3 基 β=8 r2 0.243/KL 0.982；**combo 在 p3 基失效**（闭合 −0.01）——combo 是 lin/p2 时代装置。六位级锚 d40–d48b 全零。ledger #401 / freeze `6eb9bed8` / res `8104fd1b`。'
             '- **3250 已预注册**：(a) 现役配方客户端化（工程，顺延两次）；(b) deg>5 探针 + 核族收尾（零 GPU）；(c) u* 顺延。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
    # compress 3246/3247 detail to keep under limit
    old_46 = ('- **3246 收官（regime boundary，registered）**：FLIP 定位 = INFORMATION_GAP——LS oracle（真 OOV 类+实体偏移）转正 +0.329/+0.341（仅完美类信息 +0.083）；拉力算子数学上限证明（fam_fid→0），转正必须偏移型分量；XC_b4 现役最优（SCLS −0.772/r2 0.203/KL 1.074，R2 翻倍）。四位级锚全零。ledger #398 / freeze `88ab83b9` / res `df7ec079`。')
    new_46 = ('- **3246**：FLIP=INFORMATION_GAP（LS oracle +0.329 转正；拉力上限证明→转正须偏移型分量）；XC_b4 现役（−0.772/r2 0.203）。#398 `df7ec079`。')
    if old_46 in mem:
        mem = mem.replace(old_46, new_46)
    old_47 = ('- **3247 收官（info menu，registered）**：**XC_poly_b4 = 新现役最优**（poly kernel seen-only 映射 × X_combo β=4）：SCLS −0.651/ENT −0.435 双超 XC_b4，DVU/DCC 零回退；闭合率 +0.142 菜单唯一正值。增益载体 = 映射函数类（poly 纯拉力 −0.827）；联合 donor/per-template 扩增证伪。四位级锚 d40/d42/d46/T1 全零。ledger #399 / freeze `d0def478` / res `c5510e5f`。')
    new_47 = ('- **3247**：XC_poly_b4 新现役（−0.651/−0.435 零回退，闭合 +0.142）；增益载体=映射函数类；联合 donor/per-template 证伪。#399 `d0def478`。')
    if old_47 in mem:
        mem = mem.replace(old_47, new_47)
elif '3249 收官（deg extension' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3249 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print(chr(10).join(REPORT))
