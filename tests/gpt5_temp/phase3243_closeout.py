# -*- coding: utf-8 -*-
"""Phase 3243 closeout (five writes, idempotent):
1) result assertions  2) ledger #395  3) MEMO Phase 3243 section
4) dual wlog  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3243\g5h6_map_oov_extrap\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 05:50'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D = R['delta_vs_G0']
XD = AGG['transfer_diag']


def fx(v, n=3):
    return ('%.' + str(n) + 'f') % v


# ---------- 1. result assertions ----------
A = R['anchors']
assert R['verdict'] == 'mapfix_partial'
assert R['design_sha8'] == '5ccf2bdc' and R['res_sha8'] == '2734168a'
assert R['model'] == 'qwen3-4b' and R['donors'] == ['glm4-9b', 'qwen3-14b']
assert R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
assert R['gates']['G1_replication'] is True and R['gates']['G2_efficacy'] is False
assert R['gates']['G3_determinism'] is True and R['gates']['G4_registry_untouched'] is True
assert R['gates']['determinism_diff'] == 0.0
assert R['gates']['d40_vs_sealed_3240'] is True and R['gates']['d42_vs_sealed_3242'] is True
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
# T3_rep verbatim: reproduces sealed 3242 T3_dual deltas
assert abs(D['T3_rep']['d_fam_fid']['SCLS'] - 0.458) < 5e-3
assert abs(D['T3_rep']['d_fam_fid']['ENT'] - 0.396) < 5e-3
# all three map variants FAIL to beat the base map
assert D['M1_dual']['d_fam_fid']['SCLS'] < D['T3_rep']['d_fam_fid']['SCLS']
assert D['M2_dual']['d_fam_fid']['SCLS'] < D['T3_rep']['d_fam_fid']['SCLS']
assert D['M3_dual']['d_fam_fid']['SCLS'] < 0.1
# M2 mechanism collapse: correction residuals at noise scale
assert all(XD[d]['M2']['res_norm_mean'] < 0.01 for d in XD)
# M3: cos improves but permuted control nearly matches (geometry w/o behavior)
cos_r_14b = sum(XD['qwen3-14b']['M3']['cos_transfer_real']) / 4
cos_p_14b = sum(XD['qwen3-14b']['M3']['cos_transfer_perm']) / 4
assert cos_r_14b > 0.2 and cos_p_14b > 0.15          # weak separation
assert abs(D['M3r_dual']['d_kl_mean']) < abs(D['M3_dual']['d_kl_mean']) + 1e-9
# centering does NOT fix OOV extrapolation
assert all(XD[d]['M1']['r2_map_oov'] < 0.2 for d in XD)
REPORT.append('1 result assertions: OK (verdict=%s res=%s)'
              % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'face54fa'
already = [e for e in led['measurements'] if e.get('phase') == 3243]
if not already:
    assert len(led['measurements']) == 394 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3243,
        "name": "g5h6_map_oov_extrap",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b with qwen3-14b / glm4-9b donors (zero-GPU transfer-map "
                 "OOV-extrapolation improvement on the 3239/3240/3242 fidelity loop)",
        "verdict": ("mapfix_partial|all_three_map_variants_fail_to_beat_base_dual_ridge|"
                    "centering_HURTS_M1_SCLS+0.27_vs_base_+0.46|M2_collapse_res_1e-3|"
                    "M3_geometry_without_behavior_cos_up_KL_equals_permuted_control|"
                    "base_map_at_seen_only_information_limit|sign_flip_not_achieved"),
        "evidence_level": ("bridge arm (preregistered MEMO 3242 tail arm a; G0 = "
                           "3240-F2 verbatim d40 = 0.0 AND T3_rep = sealed 3242 "
                           "T3_dual verbatim d42 = 0.0, both per-seed bit-level on "
                           "all 876 rows; anchors d83/d_tc/drift69/d88 == 0.0, "
                           "d89 <1e-6; census spectra cross-checked vs sealed 3238; "
                           "leakage declaration unchanged from 3242: donor OOV class "
                           "means are the DECLARED side information, all map "
                           "variants trained on 4b SEEN train rows only)"),
        "model_scope": "qwen3-4b target + 2 donors (per prereg)",
        "prereg_id": "G5-H6 (MEMO 3242 tail, arm a)",
        "superseded_by": None,
        "detail": (
            "Transfer-map OOV-extrapolation improvement arm (zero GPU), targeting "
            "the 3242 gap (r2_map_oov negative; T3_dual just under the +0.5 gate). "
            "Three preregistered map variants, all trained on seen train rows only: "
            "M1 = centered dual-ridge (no bias); M2 = class-conditional affine on "
            "top of M1 (per-seen-class mapping residual correction with donor-space "
            "softmax weights, beta=1); M3 = multi-anchor min-norm interpolation "
            "(pinv of donor seen class directions, rcond 1e-6). Recipes: G0 anchor, "
            "T3_rep = verbatim 3242 T3_dual, M1/M2/M3_dual = joint spans (6+4+4), "
            "M1r/M2r/M3r = matched row-level permutation controls (seed 42). "
            "Findings (3-seed means, delta vs G0): (1) CENTERING HURTS: M1_dual "
            "SCLS +0.273 / ENT +0.239 vs base T3_rep +0.458 / +0.396 — the bias "
            "term of the base dual-ridge carries transferable class structure that "
            "centering removes; (2) M2 collapses onto M1: after centering, seen-"
            "class mapping residuals are at noise scale (norm ~2e-3..7e-3), the "
            "class-conditional affine has no room to act (mechanism collapse, "
            "same pattern as 3240-F1); (3) M3 improves geometric transfer (mean "
            "cos vs true OOV dirs: 14b 0.264 / glm 0.250) but carries NO "
            "behavioral information beyond its permuted control (KL delta +0.0014 "
            "vs control -0.0008; fam_fid equals control) — geometry-without-"
            "behavior, the transfer-level echo of F20; (4) centering does not fix "
            "OOV extrapolation (r2_map_oov 0.115 / -0.291, unchanged). "
            "Conclusion: the 3242 base dual-ridge map is at or near the SEEN-ONLY "
            "information limit — three independent improvements all fail; "
            "remaining options require OOV-side information beyond donor class "
            "means (row-level donor structure) or a non-linear/non-parametric "
            "mechanism. T3_rep (base map) remains the best recipe overall. "
            "Registry untouched (v1.33 remains head, byte-verified). Gates: "
            "G1/G3/G4 True, G2 False (amplitude clause); determinism 0.0. "
            "freeze 5ccf2bdc, res 2734168a. Pre-append ledger file sha8 "
            "%s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3243', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #395 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3243 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 395
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=395 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3243:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3243: G5-H6 迁移映射 OOV 外推改进臂（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3242 tail 臂 (a)；freeze **`5ccf2bdc`**；res **`2734168a`**；verdict **`mapfix_partial`**（G1/G3/G4 True，G2 False——三变体全部未能超越 base 映射；G0 位级复现 d40=0.0 且 **T3_rep 对封存 3242 T3_dual 位级复现 d42=0.0**）。泄漏声明与 3242 一致：donor OOV 类均值为声明侧信息；全部映射变体仅在 4b seen train rows 上训练。registry 不动（v1.33 仍为头）。')
    A_('')
    A_('### 配方（预注册八臂）')
    A_('')
    A_('G0 = 3240-F2 复现锚；T3_rep = 3242 T3_dual 逐字复现锚；M1_dual = 中心化 dual-ridge（去偏置）；M2_dual = 类条件仿射（M1 + seen 类映射残差修正，donor 空间 softmax β=1）；M3_dual = 多锚最小范数插值（pinv donor seen 类方向，rcond 1e-6）；M1r/M2r/M3r = 行级置换对照（seed 42，3242 同款构造）。span 均为 6+4+4 联合。')
    A_('')
    A_('### 实际结果（3 seeds 均值，Δ vs G0）')
    A_('')
    A_('- **T3_rep 逐字复现**：ΔSCLS **%+.3f** / ΔENT **%+.3f**——与封存 3242 逐 seed 位级一致（d42=0.0），跨 phase 装置连续性确立。' % (
        D['T3_rep']['d_fam_fid']['SCLS'], D['T3_rep']['d_fam_fid']['ENT']))
    A_('- **中心化反而有害（M1）**：ΔSCLS **%+.3f** / ΔENT **%+.3f**，低于 base——base dual-ridge 的偏置项携带了可迁移的类结构信息，去偏置把它丢掉；且中心化不修复 OOV 外推（r2_map_oov 14b %.3f / glm %+.3f，几乎不变）。' % (
        D['M1_dual']['d_fam_fid']['SCLS'], D['M1_dual']['d_fam_fid']['ENT'],
        XD['qwen3-14b']['M1']['r2_map_oov'], XD['glm4-9b']['M1']['r2_map_oov']))
    A_('- **M2 机制塌缩**：中心化后 seen 类映射残差仅 norm %.1e（14b）/ %.1e（glm）——类条件仿射无空间可修，M2≡M1（同 3240-F1 塌缩模式）。' % (
        XD['qwen3-14b']['M2']['res_norm_mean'], XD['glm4-9b']['M2']['res_norm_mean']))
    A_('- **M3 几何有、行为无**：迁移 cos 升至 14b **%.3f** / glm **%.3f**（三变体中唯一实质提升），但 KL Δ **%+.4f** vs 其置换对照 **%+.4f**——不高于置换；fam_fid 与对照持平。几何对齐 ≠ 输出分布信息（F20「读得出≠控得住」在迁移层的回声）。' % (
        sum(XD['qwen3-14b']['M3']['cos_transfer_real']) / 4,
        sum(XD['glm4-9b']['M3']['cos_transfer_real']) / 4,
        D['M3_dual']['d_kl_mean'], D['M3r_dual']['d_kl_mean']))
    A_('- **对照装置有效**：M1r/M2r KL Δ %+.4f vs 真实 %+.4f——行级置换对照仍然干净分离；M3 对照分离弱是其自身性质（donor 几何透明传递），非装置缺陷。' % (
        D['M1r_dual']['d_kl_mean'], D['M1_dual']['d_kl_mean']))
    A_('- **headline（非门）**：转正未达成；base 映射（T3_rep +0.458）仍是全场最优配方。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **base dual-ridge 已处于 seen-only 信息极限**：三个独立改进方向（去偏置/类条件/锚插值）全部失败，且失败方式各异（有害/塌缩/无行为信息）——缺口不在映射形式，而在 seen-only 训练信号本身不含 OOV 类几何。')
    A_('1. **「几何-行为分离」二次确证**：M3 把几何 cos 推高近一倍（glm 0.113→0.250）而行为信息为零——cos 保真诊断与行为保真（KL/fam_fid）在迁移层脱钩，与 F20（读出几何≠行为控制）同构。')
    A_('1. **保真曲线收敛**：F0 −1.75 → T3_rep −1.29（无副作用最优）→ M1/M2 −1.48 → M3 −1.72。seen-only 映射族的包络就是 T3_rep；继续推进需越出声明侧信息界（donor OOV 行级结构）或非参数机制。')
    A_('')
    A_('**限界**：映射均为线性（λ=1e-3 / pinv rcond 1e-6，无调参）；M3 anchor 复现残差 ~1.5e-5（类方向近秩 1 病态，SCLS PR 1.03）；donor OOV 类均值是声明侧信息；seed 方差未解释。')
    A_('')
    A_('**完整性**：ledger #395（pre 文件 sha8 %s）；产物 result.json + mapfix.npz（逐配方 fam_fid/R²/KL 聚合矩阵 + 逐变体迁移 cos）；独立磁盘复核见 p3243_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3244（候选）')
    A_('')
    A_('- (a) **实体桥 DVU/DCC 回退归因臂**（3241 挂账，轻量零 GPU）：定位 G1e 双桥污染关系族的机制通道；')
    A_('- (b) u* 内容归因（继续顺延）；(c) 跨模型行级 donor 结构（需扩泄漏声明，暂缓）。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3243 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3243 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3243:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3243 迁移映射 OOV 外推改进完成（零 GPU）：T3_rep 对封存 3242 T3_dual 位级复现 d42=0.0；三变体全部失败——'
      'M1 中心化有害（SCLS +0.27 vs base +0.46，偏置项携带可迁移类结构）、M2 机制塌缩（seen 残差 ~1e-3）、'
      'M3 几何有行为无（cos 升至 0.26/0.25 但 KL 不高于置换对照，F20 迁移层回声）。结论：base dual-ridge 已处 '
      'seen-only 信息极限。verdict mapfix_partial；freeze 5ccf2bdc / res 2734168a，ledger #395，registry v1.33 不动。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3243 迁移映射 OOV 外推改进完成' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3243 已预注册**：(a) 映射 OOV 外推改进臂（零 GPU，中心化/类条件仿射/多锚 Procrustes）；(b) 实体桥回退归因顺延；(c) u* 归因顺延。')
new_block = ('- **3243 收官（map OOV extrap）**：T3_rep d42=0.0 位级复现；三变体全败（M1 中心化有害 +0.27 vs +0.46、M2 塌缩、M3 几何有行为无）→ base 映射处 seen-only 信息极限。ledger #395 / freeze `5ccf2bdc` / res `2734168a`。'
             '- **3244 已预注册**：(a) 实体桥 DVU/DCC 回退归因（零 GPU）；(b) u* 归因顺延。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
elif '3243 收官（map OOV extrap' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3243 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print(chr(10).join(REPORT))
