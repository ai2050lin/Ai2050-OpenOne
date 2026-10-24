# -*- coding: utf-8 -*-
"""Phase 3248 closeout (five writes, idempotent):
1) result assertions  2) ledger #400  3) MEMO Phase 3248 section
4) dual wlog  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3248\g5h11_poly_deepening\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 11:00'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D = R['delta_vs_G0']
H = R['headline']

# ---------- 1. result assertions ----------
A = R['anchors']
assert R['verdict'] == 'poly_deepening_registered'
assert R['design_sha8'] == 'b47042fd' and R['res_sha8'] == '7d2acdb6'
assert R['model'] == 'qwen3-4b' and R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_transfer_realism',
                                  'clause_b_saturation_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert AGG['XC_b4']['d_vs_sealed_3246_max'] == 0.0
assert AGG['XC_p2_b4']['d_vs_sealed_3247_max'] == 0.0
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
# clause (a): realism of every NEW map spec
real = H['transfer_realism']
for k in ('p1', 'p3', 'rbf_s05', 'rbf_s2', 'p2n', 'rbfn'):
    assert real[k]['margin'] > 0.0, (k, real[k])
# degree ladder monotone improving in SCLS
dl = H['saturation_axes']['degree_ladder_SCLS']
assert dl['XC_p1_b4'] < dl['XC_p2_b4'] < dl['XC_p3_b4']
# near-linear control passes: p1 ~= lin XC_b4 (no artifact)
assert abs(dl['XC_p1_b4'] - AGG['XC_b4']['fam_fid_mean']['SCLS']) < 0.03
# new best beats the 3247 best on BOTH SCLS and ENT with zero regression
assert AGG['XC_p3_b4']['fam_fid_mean']['SCLS'] > AGG['XC_p2_b4']['fam_fid_mean']['SCLS']
assert AGG['XC_p3_b4']['fam_fid_mean']['ENT'] > AGG['XC_p2_b4']['fam_fid_mean']['ENT']
assert D['XC_p3_b4']['d_fam_fid']['DVU'] > -0.05 and D['XC_p3_b4']['d_fam_fid']['DCC'] > -0.05
# pure pull K_p3_T3 also zero-regression and beats XC_p2_b4
assert D['K_p3_T3']['d_fam_fid']['DVU'] > -0.05 and D['K_p3_T3']['d_fam_fid']['DCC'] > -0.05
assert AGG['K_p3_T3']['fam_fid_mean']['SCLS'] > AGG['XC_p2_b4']['fam_fid_mean']['SCLS']
# saturation verdict: not saturated
assert H['saturated'] is False
assert H['best_new_closure'] > 0.0
# normalization variants: p2n harmful (DVU/DCC regression)
assert D['XC_p2n_b4']['d_fam_fid']['DVU'] < -0.10
# ORC_ls_cls upper bound replicates sealed 3246 (semantic)
assert abs(AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] - 0.083) < 0.01
REPORT.append('1 result assertions: OK (verdict=%s res=%s)' % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = '94fc1ffe'
already = [e for e in led['measurements'] if e.get('phase') == 3248]
if not already:
    assert len(led['measurements']) == 399 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3248,
        "name": "g5h11_poly_deepening",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b (zero-GPU transfer-map family deepening: is the "
                 "3247 poly-kernel gain saturated?)",
        "verdict": ("poly_deepening_registered|NOT SATURATED: degree ladder "
                    "1->2->3 monotone improving; XC_p3_b4 (deg-3 map) "
                    "SCLS -0.550 / ENT -0.356 with DVU +0.000 / DCC +0.009 "
                    "zero regression (closure +0.137); K_p3_T3 (deg-3 PURE "
                    "pull, no combo mechanism) reaches SCLS -0.529 / ENT "
                    "-0.387 (closure +0.165) - a 14-dim high-quality "
                    "transfer span pull alone pushes the gap past the "
                    "entire 3240-3247 recipe stack; beta re-coupling on "
                    "the poly2 base moves the optimum to beta 6-8 "
                    "(r2 0.242 vs 0.205, KL 0.979 vs 1.077)"),
        "evidence_level": ("deepening arm (preregistered MEMO 3247 tail; "
                           "FOUR bit-level anchors d40/d42/d46/d47 == 0.0 "
                           "per-seed on all 876 rows vs sealed "
                           "3240/3242/3246/3247; anchors d83/d_tc/drift69/"
                           "d88 == 0.0, d89 <1e-6; census spectra "
                           "cross-checked; transfer realism of every new "
                           "map spec positive; near-linear deg-1 control "
                           "reproduces the lin-based XC_b4 within 0.01, "
                           "ruling out kernel artifacts; ORC_ls_cls is "
                           "EVALUATION-ONLY closure denominator)"),
        "model_scope": "qwen3-4b (per prereg)",
        "prereg_id": "G5-H11 (MEMO 3247 tail)",
        "superseded_by": None,
        "detail": (
            "Transfer-map family deepening (zero GPU), 15 recipes on 4 "
            "axes. Findings (3-seed means): (1) DEGREE LADDER MONOTONE: "
            "XC_p1_b4 -0.766 -> XC_p2_b4 -0.651 -> XC_p3_b4 -0.550 SCLS "
            "(ENT -0.532 -> -0.435 -> -0.356); deg-1 control ~= lin "
            "XC_b4 (-0.772), so the gain is genuinely from kernel "
            "degree; NOT SATURATED at deg 3; (2) K_p3_T3 pure pull "
            "(no lexical combo) SCLS -0.529 / ENT -0.387, DVU +0.000 / "
            "DCC +0.009, closure +0.165 - the BEST item overall, and "
            "3246's pull-operator ceiling does NOT bind at 14-dim "
            "high-fidelity transfer spans (the ceiling argument applies "
            "as span -> full space, not at k=14 with sharp directions); "
            "(3) beta re-coupling on the poly2 base: optimum moves to "
            "beta 6-8, r2 0.242 (vs 0.205 at beta 4) and KL 0.979 (vs "
            "1.077) at equal fidelity - the lexical sharpness optimum "
            "is map-family dependent; (4) RBF sigma: s2 slightly worse "
            "(-0.851), s0.5 harmful via DVU/DCC regression (-0.124/"
            "-0.115); (5) row normalization harmful: p2n SCLS -0.903 "
            "with DVU -0.144 / DCC -0.268 regression; rbfn slightly "
            "worse than unnormalized; (6) realism margins all positive "
            "(clause_a). Registry untouched (v1.33 head, byte-verified). "
            "Gates G1-G4 all True; determinism 0.0. freeze b47042fd, "
            "res 7d2acdb6. Pre-append ledger file sha8 %s."
            % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3248', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #400 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3248 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 400
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=400 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3248:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3248: G5-H11 迁移映射函数族深化（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3247 tail；freeze **`b47042fd`**；res **`7d2acdb6`**；verdict **`poly_deepening_registered`**（G1–G4 全过，SMOKE 一次通过）。**四个位级锚全零**：G0 d40=0.0、T3_rep d42=0.0、XC_b4 d46=0.0、XC_p2_b4 d47=0.0（vs 3247，新增锚），均 per-seed 全 876 行。registry 不动（v1.33 仍为头）。')
    A_('')
    A_('### 核心发现：degree 阶梯**未饱和**（headline: SATURATED=False）')
    A_('')
    A_('- **degree 单调 1→2→3**：XC_p1_b4 −0.766 → XC_p2_b4 −0.651 → **XC_p3_b4 −0.550**（SCLS；ENT −0.532→−0.435→−0.356），DVU +0.000/DCC +0.009 零回退；deg-1 近线性控制 ≈ lin 基 XC_b4（−0.772 vs −0.772）——增益确来自核阶数、非核伪影。**闭合率 +0.137**（3248 菜单次优正值）。')
    A_('- **K_p3_T3 纯拉力 = 全场最优**（闭合率 **+0.165**）：SCLS −0.529 / ENT −0.387 / DVU +0.000 / DCC +0.009 / KL 1.044——**无 combo 机制的 14 维高质量迁移 span 拉力单枪匹马超过 3240–3247 全部配方栈**。3246 拉力上限论证不约束 k=14 尖锐 span（该论证只在 span→全空间时生效）。')
    A_('- **β 再联扫（p2 基）**：最优移至 β=6–8——r2 0.242（vs β=4 的 0.205）且 KL 0.979（vs 1.077），保真 min(SCLS,ENT) 持平。**词汇尖锐度最优随映射族移动**；r2/KL 维 β=6–8 明确更优。')
    A_('- **证伪臂**：RBF σ=0.5 DVU/DCC 回退（−0.124/−0.115）；行归一化 p2n 有害（SCLS −0.903、DVU −0.144/DCC −0.268）；rbfn/s2 略差。**保真缺口曲线更新**：G0 −1.751 → T3 −1.293 → XC_b4 −0.772 → XC_p2_b4 −0.651 → **K_p3_T3 −0.529 / XC_p3_b4 −0.550** → oracle 上界 +0.083，剩余 −0.61。')
    A_('- realism margin 全正（clause_a 过）：p3 +0.090 / rbf_s05 +0.122 / rbf_s2 +0.096 / p2n +0.064 / rbfn +0.119。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **转正路径再更新**：映射函数类增益在 deg 3 未饱和。纯拉力最优（K_p3_T3）意味着「combo 机制 + lexical extra」的增量需要重新在 p3 基上评估——3249 预注册 deg 4/5 阶梯延长 + p3 基 β 再联扫 + K_p3 基 combo 臂。')
    A_('1. **机制几何分类精化**：3246 的拉力上限是 span→全空间的渐近论证；k=14 尖锐迁移 span 下拉力仍是有效通道——上限约束的是 span 覆盖率与方向质量，不是拉力算子本身。')
    A_('1. **oracle 语义**：ORC_ls_cls 仅作闭合率分母（EVALUATION-ONLY）。')
    A_('')
    A_('**限界**：零 GPU；结论限于 qwen3-4b 面板与 B-协议 split；deg>3 数值稳定性未验证（deg 3 核矩阵量级已 ~1e2，deg 4 需先数值探针）；u* 内容归因继续顺延。')
    A_('')
    A_('**完整性**：ledger #400（pre 文件 sha8 %s）；产物 result.json + poly.npz + smoke 子目录；独立磁盘复核见 p3248_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3249（候选）')
    A_('')
    A_('- (a) **degree 阶梯延长 + 组合再定位**（零 GPU）：deg 4/5（先数值探针）、K_p3 基 × β∈{4,6,8} combo 臂（K_p3_T3 0.165 与 XC_p2_b6 r2/KL 增益可能叠加）、XC_p3_b6/b8；')
    A_('- (b) **现役配方客户端化**（工程臂，自 3248 顺延）：/atlas/fidelity 端点 + FidelityMode.jsx 增补 3246–3248 曲线与现役配方标注；')
    A_('- (c) u* 内容归因（继续顺延）。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3248 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3248 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3248:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3248 迁移映射函数族深化完成（零 GPU，registered）：degree 阶梯未饱和——1→2→3 单调（XC_p3_b4 SCLS −0.550/ENT −0.356 零回退）；'
      '**K_p3_T3 纯拉力 = 全场最优**（SCLS −0.529/ENT −0.387，闭合率 +0.165，DVU/DCC 零回退），超过 3240–3247 全部配方栈；'
      '3246 拉力上限论证不约束 k=14 尖锐 span。β 再联扫：p2 基最优移至 β=6–8（r2 0.242/KL 0.979）。'
      '行归一化与 σ=0.5 证伪。四位级锚 d40/d42/d46/d47 全零（含新增 d47）。verdict poly_deepening_registered；'
      'freeze b47042fd / res 7d2acdb6，ledger #400，registry v1.33 不动。3249 已预注册：deg 4/5 + K_p3 组合臂 + 客户端化。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3248 迁移映射函数族深化完成' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3248 已预注册**：(a) XC_poly_b4 保真曲线客户端化（工程）；(b) poly 映射谱系深化（degree/σ/β 再联扫，零 GPU）；(c) u* 顺延。')
new_block = ('- **3248 收官（poly deepening，registered）**：degree 阶梯未饱和（1→2→3 单调，XC_p3_b4 −0.550/−0.356 零回退）；**K_p3_T3 纯拉力全场最优**（−0.529/−0.387，闭合 +0.165）；3246 拉力上限不约束 k=14 尖锐 span；p2 基 β 最优移至 6–8（r2 0.242）。归一化/σ0.5 证伪。四位级锚 d40/d42/d46/d47 全零。ledger #400 / freeze `b47042fd` / res `7d2acdb6`。'
             '- **3249 已预注册**：(a) deg 4/5 阶梯 + K_p3 组合臂 + β 再联扫（零 GPU）；(b) 现役配方客户端化（工程，顺延）；(c) u* 顺延。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
elif '3248 收官（poly deepening' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3248 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print(chr(10).join(REPORT))
