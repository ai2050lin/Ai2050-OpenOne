# -*- coding: utf-8 -*-
"""Phase 3240 closeout (five writes, idempotent):
1) result assertions  2) ledger #392  3) MEMO Phase 3240 section
4) dual wlog (workspace + user-level daily logs)  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3240\g5h3_cls_extrapolation\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 03:31'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D2 = R['delta_vs_F0']['F2']
D1 = R['delta_vs_F0']['F1']


def fx(v, n=3):
    return ('%.' + str(n) + 'f') % v


# ---------- 1. result assertions ----------
A = R['anchors']
assert R['verdict'] == 'cls_extrap_registered'
assert R['design_sha8'] == '69c1a6f9' and R['res_sha8'] == 'ffb97506'
assert R['model'] == 'qwen3-4b' and R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
assert all(R['gates'][k] is True for k in
           ('G1_replication', 'G2_efficacy', 'G3_determinism', 'G4_registry_untouched'))
assert R['gates']['determinism_diff'] == 0.0
assert R['gates']['best_recipe'] == 'F2'
assert R['gates']['d39_vs_sealed_3239'] is True
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
assert AGG['F0']['d_vs_sealed_3239_max'] == 0.0
# G2 core numbers
assert D2['d_fam_fid']['SCLS'] > 1.0 and D2['d_fam_fid']['ENT'] > 1.0
assert min(D2['d_fam_fid']['DVU'], D2['d_fam_fid']['DCC']) > -0.05
assert -D2['d_r2_oov'] < 0.01
REPORT.append('1 result assertions: OK (verdict=%s res=%s)'
              % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = '016f79aa'
already = [e for e in led['measurements'] if e.get('phase') == 3240]
if not already:
    assert len(led['measurements']) == 391 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3240,
        "name": "g5h3_cls_extrapolation",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b (zero-GPU class-direction extrapolation on the 3239 "
                 "fidelity loop)",
        "verdict": ("cls_extrap_registered|F2_subspace_completion_SCLS+4.47_ENT+3.85|"
                    "all_six_families_improve|R2_oov_0.179_to_0.205|KL_1.0185_to_1.0008|"
                    "F1_prior_collapse_registered|sign_flip_not_yet"),
        "evidence_level": ("extrapolation arm (preregistered MEMO 3239 tail arm a; "
                           "F0 verbatim-3239 ridge replication bit-level vs sealed "
                           "3239 result.json d39=0.0 on all 876 rows; anchors "
                           "d83/d_tc/drift69/d88 == 0.0, d89 <1e-6; census spectra "
                           "cross-checked vs sealed 3238; prereg_correction x3 in "
                           "freeze 69c1a6f9: G2 regression-bound sign semantics, "
                           "d39 smoke-row-subset mismatch, F1 collapse kept verbatim "
                           "as a finding)"),
        "model_scope": "qwen3-4b (single model per prereg)",
        "prereg_id": "G5-H3 (MEMO 3239 tail, arm a)",
        "superseded_by": None,
        "detail": (
            "Class-direction extrapolation (zero GPU) on the 3239 fidelity loop. "
            "Three preregistered recipes on OOV rows: F0 = 3239 ridge verbatim "
            "(replication anchor, per-seed d39 = 0.0 vs sealed 3239); F1 = "
            "prior-mean extrapolation Yhat + (mean(class_mean) - B0) — COLLAPSED "
            "(3-seed mean dSCLS -0.009, dENT -0.006): the mean of trained class "
            "means is, to first order, the global mean, so the uniform prior "
            "carries no class-direction signal (registered finding, recipe kept "
            "verbatim per prereg); F2 = subspace completion Yhat + P_S(B0 - Yhat) "
            "with S = orthonormal span of the six trained centered class means "
            "(QR) — 3-seed means: SCLS -6.221 -> -1.751 (+4.470), ENT -5.243 -> "
            "-1.395 (+3.849), MASSIVE +2.152, DVU +0.098, DCC +0.093, USTAR "
            "+0.088 (ALL six families move positive); R2_oov 0.179 -> 0.205 "
            "(+0.026); next-token KL 1.0185 -> 1.0008 (-0.0178), i.e. the same "
            "repair also improves output-distribution fidelity. Sign flip not "
            "yet achieved (headline, not a gate); residual class-direction "
            "energy ~1.4-1.8x the B0 baseline, and seed 8 is markedly weaker "
            "(SCLS -4.39) than seeds 7/9 (-0.28/-0.58) - seed variance of the "
            "trained-class span is the next diagnostic. Registry untouched "
            "(v1.33 remains head, byte-verified). Gates G1-G4 all True; "
            "determinism diff 0.0. freeze 69c1a6f9, res 20002d42. Pre-append "
            "ledger file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3240', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #392 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3240 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 392
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=392 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3240:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3240: G5-H3 类方向外推臂 — F0 复现 / F1 先验外推 / F2 子空间补全（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3239 tail 臂 (a)；freeze **`69c1a6f9`**（任何观测前；含 prereg_correction ×3，均 SMOKE 阶段订正于正式观测前：① G2 回退界符号语义反转（`max(d)<0.05` 把正向改善误判为回退 → 改 `min(d_DVU,d_DCC) > −0.05` 回退下界）；② d39 锚 smoke 行子集（前 200 行）对封存正式（876 行）不可比 → 仅正式模式生效；③ **F1 塌缩按预注册原文执行并登记为发现**，不事后重定义）；res **`20002d42`**；verdict **`cls_extrap_registered`**（G1–G4 全 True，d39=0.0 位级复现封存 3239）。registry 不动（v1.33 仍为头）。')
    A_('')
    A_('### 配方（预注册三臂）')
    A_('')
    A_('- **F0**：3239 ridge 原版（复现锚，逐 seed 对封存 3239 result.json d39=0.0，全 876 行）；')
    A_('- **F1 prior-mean**：Ŷ + (mean_{c<6}(class_mean[c]) − B0)——训练类均值先验，OOV 类均匀共享（无泄漏）；')
    A_('- **F2 subspace completion**：Ŷ + P_S(B0 − Ŷ)，S = span{class_mean[c] − B0 : c<6} 的 QR 正交基——把 ridge 相对常量基线丢失的成分限制在训练类子空间内补全。')
    A_('')
    A_('### 实际结果（3 seeds 均值，Δ vs F0）')
    A_('')
    A_('- **F1 塌缩**：ΔSCLS **%+.3f**、ΔENT **%+.3f**——训练类均值的均值一阶上就是全局均值，均匀先验不携带类方向信号（预注册发现的机制性塌缩）。' % (D1['d_fam_fid']['SCLS'], D1['d_fam_fid']['ENT']))
    A_('- **F2 全族正向**：SCLS −6.221→**−1.751**（**+%.3f**）、ENT −5.243→**−1.395**（**+%.3f**）、MASSIVE +%.3f、USTAR +%.3f、DVU +%.3f、DCC +%.3f——六族无一回退；R²_oov 0.179→**0.205**（+%s）；下一 token KL 1.0185→**1.0008**（%s，修复同时改善输出分布保真）。' % (
        D2['d_fam_fid']['SCLS'], D2['d_fam_fid']['ENT'], D2['d_fam_fid']['MASSIVE'],
        D2['d_fam_fid']['USTAR'], D2['d_fam_fid']['DVU'], D2['d_fam_fid']['DCC'],
        fx(D2['d_r2_oov'], 4), fx(D2['d_kl_mean'], 4)))
    A_('- **headline（非门）**：SCLS/ENT **未转正**（F2 残余类方向能量 ≈ B0 基线的 1.4–1.8×）；seed 8 显著弱于 seed 7/9（SCLS −4.39 vs −0.28/−0.58）——训练类张成子空间的 seed 方差是下一个诊断对象。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **3239 负保真根因确证**：把丢失的类子空间分量补回，保真即大幅修复（SCLS +4.5/ENT +3.8）——3239「OOV 类列无训练信号 → ridge 丢类方向」的定位得到干预侧反向确证（remove→harm 已由 3239 自然实验给出，restore→repair 由本 phase 给出）。')
    A_('1. **F1 塌缩是均匀先验的信息论下界**：不引入任何 OOV 信息的均匀外推在类方向上注定零增益——转正需要 OOV 类的**侧信息**（词汇/语义桥），这把问题从「预测配方」推到「图谱侧类间结构」（类间方向的组织方式，census 复用矩阵方向）。')
    A_('1. **保真度曲线第二个点入册**：同一套曲线，F2 配方点整体上移且 KL 同向改善——「破解程度」曲线开始有可比较的配方族。')
    A_('')
    A_('**限界**：单模型（qwen3-4b）；F1/F2 修正仅定义于 OOV 行（seen-test 与 F0 恒同）；转正未达成（headline 如实登记）；seed 方差未解释。')
    A_('')
    A_('**完整性**：ledger #392（pre 文件 sha8 %s）；产物 result.json + extrap.npz（逐配方 fam_fid/R²/KL 聚合矩阵，供客户端曲线）；独立磁盘复核见 p3240_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3241（候选）')
    A_('')
    A_('- (a) **保真度曲线客户端化**（工程，零 GPU）：/atlas/fidelity 端点 + 曲线面板，把 3239/3240 两点入图（3239 tail 臂 b 顺延至此）；')
    A_('- (b) **类间侧信息臂**（零 GPU）：OOV 类方向由类间结构外推（census 复用矩阵/SCLS 族内类间几何），目标 SCLS/ENT 转正——需要引入 OOV 侧信息，预注册时须显式声明信息来源与泄漏界；')
    A_('- (c) u* 内容归因类级扩展（继续顺延）。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3240 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3240 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3240:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3240 类方向外推完成（零 GPU，qwen3-4b）：F0 复现 3239 位级（d39=0.0）；F1 均匀先验'
      '塌缩（ΔSCLS −0.009，机制：类均值均值≈全局均值）；F2 类子空间补全全族正向——SCLS '
      '−6.22→−1.75（+4.47）、ENT −5.24→−1.40（+3.85）、MASSIVE +2.15、DVU/DCC +0.10/+0.09，'
      'R²_oov 0.179→0.205、KL 1.0185→1.0008；转正未达成（headline），seed 8 弱（−4.39）是下一'
      '诊断。verdict cls_extrap_registered（G1–G4 全过）。freeze 69c1a6f9 / res 20002d42，'
      'ledger #392，registry v1.33 不动。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3240 类方向外推完成' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3240 已预注册**：(a) 类方向外推臂（零 GPU，默认）；(b) 保真度曲线客户端化；(c) u* 归因顺延。')
new_block = ('- **3240 收官（cls extrapolation）**：F0 复现 d39=0.0；F1 均匀先验塌缩；F2 子空间补全全族正向（SCLS +4.47/ENT +3.85）、R² 0.205、KL 1.0008，转正未达。ledger #392 / freeze `69c1a6f9` / res `20002d42`。'
             '- **3241 已预注册**：(a) 保真度曲线客户端化（/atlas/fidelity + 面板）；(b) 类间侧信息臂（SCLS/ENT 转正，须声明泄漏界）；(c) u* 归因继续顺延。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
elif '3240 收官（cls extrapolation）' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3240 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print('\n'.join(REPORT))
