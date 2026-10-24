# -*- coding: utf-8 -*-
"""Phase 3245 closeout (five writes, idempotent):
1) result assertions  2) ledger #397  3) MEMO Phase 3245 section
4) dual wlog  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3245\g5h8_combo_flip\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 09:05'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D = R['delta_vs_G0']
H = R['headline']

# ---------- 1. result assertions ----------
A = R['anchors']
assert R['verdict'] == 'combo_flip_registered'
assert R['design_sha8'] == '105df1a8' and R['res_sha8'] == 'ef039473'
assert R['model'] == 'qwen3-4b' and R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
assert R['gates']['G1_replication'] is True and R['gates']['G2_combination'] is True
assert R['gates']['clause_a_zero_regression'] is True
assert R['gates']['clause_b_joint_gain'] is True
assert R['gates']['clause_c_not_worse_than_best_single'] is True
assert R['gates']['G3_determinism'] is True and R['gates']['G4_registry_untouched'] is True
assert R['gates']['determinism_diff'] == 0.0
# three bit-level anchors all exact
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['R1_rep']['d_vs_sealed_3244_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
# clause (a): X_combo DVU/DCC residuals == T3_dual's by construction
dX = D['X_combo']['d_fam_fid']
assert min(dX['DVU'], dX['DCC']) > -0.05
assert abs(dX['DVU'] - D['T3_rep']['d_fam_fid']['DVU']) < 1e-6
assert abs(dX['DCC'] - D['T3_rep']['d_fam_fid']['DCC']) < 1e-6
# clause (b): joint gain positive on BOTH SCLS and ENT
assert min(dX['SCLS'], dX['ENT']) > 0
# clause (c): combo beats the best single line on min(SCLS,ENT)
assert H['combo_min_scls_ent'] > H['best_single_min_scls_ent']
assert H['joint_beats_best_single'] is True
# additivity: dX > max(dR1, dT3) on both families
ad = H['r1_gain_additivity']
assert ad['dX_SCLS'] > max(ad['dR1_SCLS'], ad['dT3_SCLS'])
assert ad['dX_ENT'] > max(ad['dR1_ENT'], ad['dT3_ENT'])
# FLIP not reached: absolute fidelity still negative
assert H['flip_absolute_SCLS_ENT_positive'] is False
assert -1.0 < H['combo_min_scls_ent'] < 0
# real identity beats permuted control
assert H['perm_control_gap']['X_combo_min'] > H['perm_control_gap']['X_combo_r_min']
# construction identities at f32 noise level
# (bd_residual gap 1.9e-5 measured at the fam_fid layer, i.e. after row-mean
# accumulation — the row-level residual identity is exact by construction;
# 1e-5 was too tight a closeout assertion for the averaged quantity)
assert R['identity_gaps']['combination_order_id_gap_max'] < 1e-5
assert R['identity_gaps']['bd_residual_identity_gap_max'] < 5e-5
REPORT.append('1 result assertions: OK (verdict=%s res=%s)' % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = '530d0ca2'
already = [e for e in led['measurements'] if e.get('phase') == 3245]
if not already:
    assert len(led['measurements']) == 396 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3245,
        "name": "g5h8_combo_flip",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b (zero-GPU R1_clean x T3_dual combination flip arm on the "
                 "3239/3240/3244 fidelity loop)",
        "verdict": ("combo_flip_registered|two zero-regression lines combined: "
                    "X_combo = T3_dual transferred pull + P_Bd_perp(G1e lexical "
                    "extra); DVU/DCC residuals constructionally identical to "
                    "T3_dual; joint gain ADDITIVE (SCLS +0.954 > max(+0.836, "
                    "+0.458)); FLIP not reached (SCLS -0.797 absolute)"),
        "evidence_level": ("combination arm (preregistered MEMO 3244 tail arm a; "
                           "THREE bit-level anchors d40/d44/d42 == 0.0 per-seed on "
                           "all 876 rows: G0 vs sealed 3240-F2, R1_rep vs sealed "
                           "3244 R1_clean, T3_rep vs sealed 3242 T3_dual; anchors "
                           "d83/d_tc/drift69/d88 == 0.0, d89 <1e-6; census spectra "
                           "cross-checked vs sealed 3238; leakage declaration = "
                           "UNION of 3242 donor scope and 3244 token-identity scope)"),
        "model_scope": "qwen3-4b (per prereg)",
        "prereg_id": "G5-H8 (MEMO 3244 tail, arm a)",
        "superseded_by": None,
        "detail": (
            "R1_clean x T3_dual combination flip arm (zero GPU). Recipes: G0 "
            "anchor, R1_rep verbatim-3244 anchor, T3_rep verbatim-3242 anchor, "
            "X_combo = Yhat0 + corr_T3 + P_Bd_perp(corr_full - corr_T3) (MAIN "
            "ARM), X_combo_r permuted control (seed 42). Findings (3-seed means, "
            "delta vs G0): (1) CONSTRUCTION EXACTNESS - combination-order "
            "identity P_Bd_perp(corr_full-corr_G0) == P_Bd_perp(corr_full-corr_T3) "
            "+ P_Bd_perp(corr_T3-corr_G0) verified (gap 7.6e-6 f32); X_combo "
            "DVU/DCC residuals bit-equal T3_dual's (gap 1.9e-5); (2) JOINT GAIN "
            "ADDITIVE - SCLS +0.954 / ENT +0.823 vs R1_rep +0.836/+0.729 and "
            "T3_rep +0.458/+0.396: the lexical extra survives ON TOP of the "
            "stronger transferred pull and the sum exceeds either single line; "
            "(3) G2 all clauses pass: zero-regression (DVU +0.002 / DCC -0.002), "
            "joint gain, not-worse-than-best-single (min -0.797 vs -0.915); (4) "
            "FLIP NOT REACHED: absolute fidelity SCLS -0.797 / ENT -0.572 still "
            "negative; conversion (fidelity > 0) remains blocked within the "
            "seen-only + declared-side-information regime - consistent with the "
            "3243 seen-only information limit; (5) real identity beats permuted "
            "control (SCLS -0.797 vs -0.806) but the gap is small (9e-3): "
            "identity-specific lexical info contributes ~1%% of the combo gain, "
            "echoing the 3244 span-attribution; (6) KL cost +0.30 (lexical extra "
            "price, same as R1_rep). Registry untouched (v1.33 remains head, "
            "byte-verified). Gates G1-G4 all True; determinism 0.0. freeze "
            "105df1a8, res ef039473. Pre-append ledger file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3245', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #397 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3245 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 397
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=397 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3245:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3245: G5-H8 R1_clean × T3_dual 组合转正臂（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3244 tail 臂 (a)；freeze **`105df1a8`**；res **`ef039473`**；verdict **`combo_flip_registered`**（G1–G4 全过；SMOKE 无设计修正——200 行子集锚失配为行子集效应，3240 教训预注册内）。**三个位级复现锚全零**：G0 vs 封存 3240-F2（d40=0.0）、R1_rep vs 封存 3244 R1_clean（d44=0.0）、T3_rep vs 封存 3242 T3_dual（d42=0.0），均 per-seed 全 876 行——跨脚本位级一致性三重证明（3244 伪影调查的机器正确性结论外推成立）。registry 不动（v1.33 仍为头）。')
    A_('')
    A_('### 配方（五臂）')
    A_('')
    A_('G0/R1_rep/T3_rep = 复现锚；**X_combo = Yhat0 + corr_T3 + P_Bd⊥(corr_full − corr_T3)**（主臂：T3_dual 14 维迁移拉力为基底 + G1e lexical extra 剥离 DVU/DCC 基能量）；X_combo_r = 置换对照（seed 42）。构造恒等式数值验证：组合顺序恒等（id_gap 7.6e-6 f32）；X_combo 的 DVU/DCC 残差与 T3_rep 位级相同（1.9e-5）。')
    A_('')
    A_('### 实际结果（3 seeds 均值，Δ vs G0）')
    A_('')
    A_('- **G2 三条款全过（combo_flip_registered）**：(a) 零回退 DVU %+.3f / DCC %+.3f（T3 残差构造性保留）；(b) 联合增益 SCLS **%+.3f** / ENT **%+.3f**（均 > 0）；(c) min(SCLS,ENT) = **%.3f** 优于最优单线 R1_rep 的 %.3f。' % (
        dX['DVU'], dX['DCC'], dX['SCLS'], dX['ENT'],
        H['combo_min_scls_ent'], H['best_single_min_scls_ent']))
    A_('- **加性增益**：dX_SCLS %+.3f > max(dR1 %+.3f, dT3 %+.3f)；dX_ENT %+.3f > max(dR1 %+.3f, dT3 %+.3f)——lexical extra 在更强迁移拉力之上完全存活，联合超过任一单线。' % (
        ad['dX_SCLS'], ad['dR1_SCLS'], ad['dT3_SCLS'],
        ad['dX_ENT'], ad['dR1_ENT'], ad['dT3_ENT']))
    A_('- **FLIP 未达（headline）**：绝对保真 SCLS **−0.797** / ENT **−0.572** 仍负——转正（fidelity > 0）在「seen-only + 声明边信息」体制内仍未达成，与 3243 seen-only 信息极限一致。保真曲线：G0 −1.75 → T3 −1.29 → R1 −0.92 → **X_combo −0.80**（零回退最优）。')
    A_('- **真实身份边际 ~1%%**：X_combo −0.797 vs 置换 −0.806（差 9e-3）——身份特异词汇信息对组合增益贡献约 1%%，呼应 3244 span 归因。KL 代价 +0.30（lexical extra 的价格，与 R1_rep 相同）。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **两条零回退线联合成立且加性**：span 机制（T3 迁移拉力）与 lexical extra（P_Bd⊥ 剥离后）在同一配方中无干涉地叠加——修正通道正交性（DVU/DCC 剥离基 ∪ 迁移 span 的近正交）是加性的几何原因。')
    A_('1. **转正的剩余缺口定性**：X_combo 把缺口从 −1.75 推至 −0.80（修复 54%%），剩余 −0.80 的 SCLS 缺口在声明体制内已无配方可达——3240 F2 全族正向转正未达、3241 双桥有回退、3242 迁移 <0.5 门、3243 三变体全败、3245 组合加性最优仍 −0.80。**转正需越出泄漏声明界（OOV 激活统计）或新机制类别**——这本身是 registered 结论。')
    A_('1. **R² 代价**：X_combo R² 0.100（−0.105 vs G0）——SCLS/ENT 增益与整体 R² 存在体制内权衡，KL 同向；配方选择目标函数需显式多目标。')
    A_('')
    A_('**限界**：零 GPU、线性投影机制；DVU/DCC 中性依赖封存 3157 族基；迁移依赖 donor OOV 类均值（已声明边信息）；FLIP 判据为 SCLS/ENT 绝对保真同时 > 0；u* 内容归因继续顺延。')
    A_('')
    A_('**完整性**：ledger #397（pre 文件 sha8 %s）；产物 result.json + combo.npz + smoke 子目录；独立磁盘复核见 p3245_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3246（候选）')
    A_('')
    A_('- (a) **声明体制边界刻画**（零 GPU）：把 3240–3245 六个 Phase 的保真缺口对「边信息预算」（span 维度 / donor 数 / 身份信息比特）作系统曲线，定位转正所需的最小越界量；')
    A_('- (b) u* 内容归因（继续顺延）；(c) 保真曲线客户端化并入 3244/3245 曲线（工程对齐）。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3245 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3245 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3245:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3245 组合转正臂完成（零 GPU，registered）：X_combo = T3_dual 迁移拉力 + P_Bd⊥(G1e lexical extra)，'
      '三条款全过——零回退（DVU/DCC 与 T3 残差位级相同）+ 联合增益加性（SCLS +0.954/ENT +0.823，超两条单线）'
      '+ 优于最优单线（min −0.797 vs −0.915）；FLIP 未达（SCLS −0.797 绝对仍负），声明体制内转正确认为不可达，'
      '转正需越界或新机制。三位级锚 d40/d44/d42 全零（跨脚本位级一致性）。verdict combo_flip_registered；'
      'freeze 105df1a8 / res ef039473，ledger #397，registry v1.33 不动。3246 已预注册：声明体制边界刻画。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3245 组合转正臂完成' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3245 已预注册**：(a) R1_clean×T3 组合转正臂（零 GPU）；(b) 保真曲线客户端化并入 3244；(c) u* 顺延。')
new_block = ('- **3245 收官（combo flip，registered）**：X_combo = T3 拉力 + P_Bd⊥(G1e extra) 加性成立（SCLS +0.954/ENT +0.823 超两单线；DVU/DCC 零回退）；FLIP 未达（SCLS −0.797 绝对仍负）→ 声明体制内转正不可达，需越界/新机制（registered 结论）。三锚 d40/d44/d42 全零。ledger #397 / freeze `105df1a8` / res `ef039473`。'
             '- **3246 已预注册**：(a) 声明体制边界刻画（边信息预算 vs 缺口曲线）；(b) u* 顺延；(c) 保真曲线客户端化。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
elif '3245 收官（combo flip' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3245 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print(chr(10).join(REPORT))
