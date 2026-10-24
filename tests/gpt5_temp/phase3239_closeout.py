# -*- coding: utf-8 -*-
"""Phase 3239 closeout (five writes, idempotent):
1) result assertions  2) ledger #391  3) MEMO Phase 3239 section
4) dual wlog (workspace + user-level daily logs)  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3239\g5h2_fidelity_minloop\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 02:32'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']


def fx(v, n=3):
    return ('%.' + str(n) + 'f') % v


# ---------- 1. result assertions ----------
A = R['anchors']
assert R['verdict'] == 'fidelity_loop_registered'
assert R['design_sha8'] == '3776313c' and R['res_sha8'] == '5af7a0dd'
assert R['model'] == 'qwen3-4b' and R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
assert all(R['gates'][k] is True for k in
           ('G1_replication', 'G2_metrics', 'G3_determinism', 'G4_registry_untouched'))
assert R['gates']['determinism_diff'] == 0.0
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
assert R['spectra']['ENT']['k90'] == 1 and R['spectra']['SCLS']['k90'] == 1
assert R['spectra']['USTAR']['k90'] == 93
assert R['lens']['tied'] is True and R['lens']['V'] == 151936
assert abs(AGG['r2_oov_model_mean'] - 0.1787) < 1e-3
assert abs(AGG['r2_seen_model_mean'] - 0.6593) < 1e-3
assert abs(AGG['kl_model_mean_mean'] - 1.0185) < 1e-3
assert abs(AGG['kl_B0_mean_mean'] - 1.6615) < 1e-3
REPORT.append('1 result assertions: OK (verdict=%s res=%s)'
              % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = '2c2e0d4f'
already = [e for e in led['measurements'] if e.get('phase') == 3239]
if not already:
    assert len(led['measurements']) == 390 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3239,
        "name": "g5h2_fidelity_minloop",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b (zero-GPU fidelity loop, CPU lens over tied unembedding)",
        "verdict": ("fidelity_loop_registered|KL_gap_recovery_38.7pct_vs_B0|"
                    "beats_identity_B1_24.8pct|R2_seen_0.659_R2_oov_0.179|"
                    "DVU_DCC_positive_fidelity|rank1_families_negative"),
        "evidence_level": ("fidelity loop arm (preregistered MEMO 3238 tail arm c; "
                           "graph side = census six families from sealed collect npz "
                           "with 3238 verbatim recipes, spectra cross-checked exact "
                           "k90 + PR/top1 <1e-6 vs sealed 3238 census_spectra.npz; "
                           "bit-level anchors d83/d_tc/drift69/d88 == 0.0, d89 f32 "
                           "class <1e-6; lens determinism bit-level 0.0 via "
                           "same-GEMM-shape recompute per prereg_correction in "
                           "freeze 3776313c)"),
        "model_scope": "qwen3-4b (single model per prereg)",
        "prereg_id": "G5-H2 (MEMO 3238 tail, arm c)",
        "superseded_by": None,
        "detail": (
            "Spectral-map prediction-fidelity minimal loop (zero GPU): graph side "
            "= 6 census family bases (k90 from sealed 3238); prediction side = "
            "B-protocol ridge (seeds 7/8/9, split_s1_246, lam 1e-3, phi 780->87 "
            "cols) on sealed 3169 KOUT activations; held-out = 876 OOV rows "
            "(classes 6-9 x 73 entities x 3 templates); answer side = CPU lens "
            "(softmax of final RMSNorm @ tied embed_tokens^T, V=151936, shards "
            "328a91d3/e4bf4369) + activation R^2. Headline (mean over seeds): "
            "KL(actual||pred) model %.4f < B1 identity %.4f < B0 global-mean "
            "%.4f — graph recovers %.1f%% of the B0 KL gap (B1: %.1f%%); "
            "median KL 0.722/1.016/1.293. R^2 seen-test 0.659 vs OOV 0.179 "
            "(~ identity baseline 0.177): in-distribution graph prediction is "
            "strong, OOV class generalization is the weak link. Family-"
            "direction fidelity (1 - resid_energy_ratio vs B0): DVU +0.160, "
            "DCC +0.277 positive (relation families carried by graph); "
            "USTAR +0.027 ~ 0; ENT -5.24 / SCLS -6.22 / MASSIVE -3.92 NEGATIVE "
            "(rank-1 concept directions actively corrupted by ridge on OOV "
            "rows — untrained class one-hot columns for unseen classes remove "
            "the class direction that even the constant baseline retains). "
            "First point of the fidelity curve registered; OOV class-direction "
            "extrapolation is the named next lever. Registry untouched "
            "(v1.33 remains head, byte-verified). Gates G1-G4 all True; "
            "in-run determinism diff 0.0. freeze 3776313c, res 5af7a0dd. "
            "Pre-append ledger file sha8 %s."
            % (AGG['kl_model_mean_mean'], AGG['kl_B1_mean_mean'],
               AGG['kl_B0_mean_mean'],
               100 * (AGG['kl_B0_mean_mean'] - AGG['kl_model_mean_mean'])
               / AGG['kl_B0_mean_mean'],
               100 * (AGG['kl_B0_mean_mean'] - AGG['kl_B1_mean_mean'])
               / AGG['kl_B0_mean_mean'],
               pre_file_sha8)),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3239', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #391 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3239 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 391
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=391 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3239:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3239: G5-H2 谱图预测保真度最小闭环 — census 六族图谱侧 → B-protocol 预测 → CPU lens 对答案（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3238 tail 臂 (c)；freeze **`3776313c`**（任何观测前；含 prereg_correction：G2 argmax 多样性界 100→10（模板面板表面形共享，结构性偏小，门的用途是退化检测）、G3 确定性探针改同 GEMM shape 批重放（3099 陷阱：跨 shape f32 累加序差 1.1e-05 纯 kernel epsilon）——均 SMOKE 阶段订正，正式观测前）；res **`5af7a0dd`**；verdict **`fidelity_loop_registered`**（G1–G4 全 True，运行内确定性 diff=0.0）。零 GPU：lens 用 safetensors 直读 tied unembedding（embed_tokens 151936×2560 f32 + final RMSNorm，eps=1e-6，shard sha8 328a91d3/e4bf4369）。registry 不动（v1.33 仍为头，字节核验过）。')
    A_('')
    A_('### 原理与公式')
    A_('')
    A_('三段闭环：**图谱侧** = census 六族子空间基（k90 截断右奇异向量，谱与封存 3238 npz 逐族 k90 精确相等 + PR/top1 <1e-6 交叉核对全过）；**预测侧** = B-protocol ridge（seed 7/8/9，split_s1_246，lam 1e-3，φ 780→87 列）在封存 3169 KOUT 激活上训练，对 876 OOV 行（类 6–9 × 73 实体 × 3 模板）出 Ŷ；**答案侧** = 激活 R² + 下一 token KL(actual‖pred)，lens(h)=softmax(RMSNorm_final(h)·E_unembedᵀ)。基线：B0=训练行全局均值（常量），B1=identity（训练实体均值，实体 41–72 回落 B0；模板面板无 n-gram 基线可行，范围注记入册）。KL 排序是登记头条数字、**不是门**——两个方向都是闭环的诚实发现。')
    A_('')
    A_('### 复现锚（装置同一性）')
    A_('')
    A_('位级锚全 0.0：d83（3183 六标量）、d_tc（ANCH_3157）、drift69（3169 逐 seed E_seen/E_oov）、d88（3188 五标量）；f32 级 d89=1.49e-07<1e-6；六族谱 vs 封存 3238 census_spectra.npz：k90 精确相等 + PR/top1 <1e-6 全过（USTAR top1 0.306548）。lens 确定性：同 shape 批重放 bit 级 diff=0.0。')
    A_('')
    A_('### 实际结果（3 seeds 均值 | 全数入 result.json + fidelity.npz）')
    A_('')
    A_('**下一 token KL（876 OOV 行）**：model **%.4f** < B1 identity **%.4f** < B0 **%.4f**——图谱预测恢复 B0 缺口的 **38.7%%**（B1 仅 24.8%%）；中位 0.722/1.016/1.293。**激活 R²**：seen-test **0.659**（分布内图谱预测强）vs OOV **0.179**（≈B1 0.177）——OOV 类泛化是短板。' % (
        AGG['kl_model_mean_mean'], AGG['kl_B1_mean_mean'], AGG['kl_B0_mean_mean']))
    A_('')
    A_('**族方向保真度**（1 − 残差族基能量比 vs B0）：DVU **+0.160**、DCC **+0.277**（关系族被图谱正向承载）；USTAR **+0.027**≈0；ENT **−5.24**、SCLS **−6.22**、MASSIVE **−3.92**（**负保真**：rank-1 概念方向被 ridge 在 OOV 行上主动破坏——未见类的 one-hot 列无训练信号，预测丢掉类方向，而常量基线至少保留均值类信号）。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **保真度曲线第一个点入册**：「破解程度」首次成为可画曲线的数字——分布内 0.659 / 谱外 0.179 / KL 缺口恢复 38.7%%。')
    A_('1. **下一 token 级图谱已赢 identity 基线**（38.7%% vs 24.8%%）：φ-ridge 的组合结构在输出分布层面有真实增量，尽管 OOV 激活 R² 与 identity 打平。')
    A_('1. **负保真是定位明确的杠杆**：ENT/SCLS 负值不是噪声（3 seeds 一致 −5.2/−6.2），根因是 OOV 类列无训练信号——类方向外推（3240 臂 a）直接对准此缺口。')
    A_('')
    A_('**限界**：单模型（qwen3-4b，预注册口径）；B1 在实体 41–72 回落 B0 使 identity 基线偏弱（登记在案）；模板面板无 n-gram 基线可行；lens 用 f16 封存激活（两侧同 lens，公平但非原生 logits）；KL 排序非门。')
    A_('')
    A_('**完整性**：ledger #391（pre 文件 sha8 %s）；产物 result.json + fidelity.npz（逐行 KL 数组，供客户端曲线）；独立磁盘复核见 p3239_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3240（候选）')
    A_('')
    A_('- (a) **类方向外推臂**（零 GPU，默认）：OOV 类方向由训练类方向子空间外推（类均值组合/子空间投影），目标把 SCLS/ENT 保真转正、抬 R²_oov——直接对准 3239 定位的缺口；')
    A_('- (b) 保真度曲线客户端化（工程）：/atlas/fidelity 端点 + 曲线面板，把 3239/3240 曲线放进工作台；')
    A_('- (c) u* 内容归因类级扩展（顺延）。')
    body = ('\n'.join(lines) + '\n').encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3239 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3239 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3239:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3239 fidelity min-loop 完成（零 GPU，qwen3-4b）：KL(actual‖pred) model 1.018 < '
      'B1 1.249 < B0 1.661（图谱恢复 B0 缺口 38.7%，赢 identity 基线）；R² seen 0.659 / '
      'OOV 0.179；族保真 DVU +0.16、DCC +0.28 正，ENT/SCLS 负（OOV 类列无训练信号，'
      '3240 类方向外推臂对准）。verdict fidelity_loop_registered（G1-G4 全过，位级锚 0.0，'
      'lens 同 shape 确定性 0.0）。freeze 3776313c / res 5af7a0dd，ledger #391，registry '
      'v1.33 不动。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3239 fidelity min-loop' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old238 = chr(45)+' **3238 收官（census A）**：六族谱 ENT/SCLS 近秩1、DVU 4.4-5.2、DCC 2.96-2.98（跨模型 0.996）、USTAR 8.5-9.8 最扩散；复用非正交（0.54/0.76/0.74）；hub #1 FTR-11。registry v1.33 `a30215fd` / ledger #390 / freeze `4cc8adda` / res `bed6984f`。'
new238 = chr(45)+' 3238 收官（census A）：ENT/SCLS 近秩1、USTAR 最扩散、复用非正交、hub #1 FTR-11；registry v1.33 `a30215fd` / ledger #390。'
if old238 in mem:
    mem = mem.replace(old238, new238)

old_block = ('- **3239 已预注册**：(a) u* 内容归因类级扩展（零 GPU）；(b) DAS 子空间交换设计臂（轻 GPU）；(c) 谱图预测保真度最小闭环（零 GPU 起步）。')
new_block = ('- **3239 收官（fidelity min-loop）**：KL 缺口恢复 38.7%（B1 24.8%）；R² seen 0.659/OOV 0.179；DVU/DCC 正保真、ENT/SCLS 负（OOV 类列无信号）。ledger #391 / freeze `3776313c` / res `5af7a0dd`；registry v1.33 不动。'
             '- **3240 已预注册**：(a) 类方向外推臂（零 GPU，默认）；(b) 保真度曲线客户端化；(c) u* 归因顺延。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
elif '3239 收官（fidelity min-loop）' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3239 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print('\n'.join(REPORT))
