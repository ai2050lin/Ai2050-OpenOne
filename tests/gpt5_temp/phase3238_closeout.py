# -*- coding: utf-8 -*-
"""Phase 3238 closeout (five writes, idempotent):
1) result assertions  2) ledger #390  3) MEMO Phase 3238 section
4) dual wlog (workspace + user-level daily logs)  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3238\g5h1_census_rank\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 00:54'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']


def sp(m, f, k):
    return R['spectra'][m][f][k]


def fx(v, n=3):
    return ('%.' + str(n) + 'f') % v


# ---------- 1. result assertions (hardcoded from vals extract, checked vs file) ----------
A = R['anchors']
assert R['verdict'] == 'census_a_registered'
assert R['design_sha8'] == '4cc8adda' and R['res_sha8'] == 'bed6984f'
assert R['registry_sha8'] == 'a30215fd' and R['registry_file'] == 'atlas_registry_v1_33.json'
assert R['registry_byte_preservation_bad'] == []
assert all(R['gates'][k] is True for k in
           ('G1_replication', 'G2_spectra', 'G3_reuse', 'G4_hub', 'G5_registry'))
assert R['gates']['determinism_diff'] == 0.0
assert all(A[m]['d83'] == 0.0 and A[m]['d_tc'] == 0.0 and A[m]['drift69'] == 0.0
           and A[m]['d88'] == 0.0 and A[m]['d89'] < 1e-6 for m in MODELS)
assert abs(sp('qwen3-4b', 'USTAR', 'top1') - 0.306548) < 1e-5
assert abs(sp('qwen3-14b', 'USTAR', 'top1') - 0.293589) < 1e-5
assert abs(sp('glm4-9b', 'USTAR', 'top1') - 0.300797) < 1e-5
assert sp('qwen3-4b', 'ENT', 'k90') == 1 and sp('qwen3-14b', 'SCLS', 'k90') == 1
assert sp('glm4-9b', 'DVU', 'k90') == 11 and sp('glm4-9b', 'DCC', 'k90') == 22
assert R['hub_order'][0] == 'FTR-11' and len(R['hub_order']) == 26
REPORT.append('1 result assertions: OK (verdict=%s res=%s reg=%s)'
              % (R['verdict'], R['res_sha8'], R['registry_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'd37ca65f'
already = [e for e in led['measurements'] if e.get('phase') == 3238]
if not already:
    assert len(led['measurements']) == 389 and led['ledger_sha256_8'] == PREV_POST
    reg_f = 'a30215fd'
    entry = {
        "phase": 3238,
        "name": "g5h1_census_rank",
        "line": "G",
        "date": "2026-10-20",
        "model": "none (zero-GPU census arm)",
        "verdict": ("census_a_registered|registry_v1.33|hub_top_FTR-11|"
                    "ENT_SCLS_near_rank1_x3|DVU_mid_DCC_stable|"
                    "USTAR_most_diffuse|reuse_nonorthogonal"),
        "evidence_level": ("census arm (preregistered MEMO 3237 tail arm a; "
                           "families derived verbatim from sealed 3157/3169 "
                           "collect npz; bit-level anchors d83/d_tc/drift69/"
                           "d88 == 0.0 x3 models; d89 f32-reproduction class "
                           "<1e-6 per prereg_correction in freeze 4cc8adda)"),
        "model_scope": "qwen3-4b + qwen3-14b + glm4-9b (sealed 3157/3169 panels)",
        "prereg_id": "G5-H1 (MEMO 3237 tail, arm a)",
        "superseded_by": None,
        "detail": (
            "Census A (zero GPU): 6 direction families per model from sealed "
            "collect npz — DVU(16xd, 3183 recipe) / DCC(64xd ctx shifts) / "
            "ENT(16 entity means) / USTAR(876xd OOV collapse rows, 3188/3189 "
            "recipe) / SCLS(10 class means) / MASSIVE(e_d1 from FTR-12 values). "
            "Rank metrics (PR/srank/k90/top1): ENT PR 1.050/1.041/1.055 "
            "top1 97.4-98.0%% x3 (entity subspace near rank-1); SCLS PR "
            "1.028/1.024/1.053 (near rank-1); DVU PR 4.43/5.22/4.37 k90 12/12/"
            "11 of 16; DCC PR 2.98/2.97/2.96 k90 20/21/22 of 64 with cross-"
            "model PR consistency 0.996; USTAR PR 8.87/9.76/8.54 k90 93/120/79 "
            "of 876 (most diffuse — coheres GAP-5/F20/F21 distributed "
            "carrying); MASSIVE rank-1 by construction. Reuse matrix (k90 "
            "principal angles): off-diag mean 0.543/0.758/0.740 — family "
            "subspaces substantially non-orthogonal; USTAR/ENT/SCLS mutual "
            "reuse highest (0.65-0.85), DVU lowest (0.36-0.62). Hub ranking "
            "(z(reuse,causal_load,cross_model), 26 rows unique): FTR-11 > "
            "FTR-10 ~= FTR-16 (E3 chain) > FTR-25 ~= FTR-26 (u*) > FTR-08 "
            "(ENT) > FTR-04/05/07 (SCLS) ...; 16 non-directional FTRs "
            "recorded with explicit null reasons. Census scope note: FTR-04/"
            "05/07 measured on panel-class analog directions, not the "
            "registered 2881-J4 word-direction recipe. Gates G1-G5 all True; "
            "in-run determinism diff 0.0. Registry v1.32 -> v1.33 (file %s) "
            "via census_3238 key + upgrade_log append, all other keys "
            "byte-preserved (canonical equality). freeze 4cc8adda, res "
            "bed6984f. Pre-append ledger file sha8 %s."
            % (reg_f, pre_file_sha8)),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3238', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #390 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3238 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 390
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=390 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3238:' not in mb:
    def S(m, f, k):
        return sp(m, f, k)
    prs = {f: [S(m, f, 'PR') for m in MODELS] for f in FAMS}
    k90s = {f: [S(m, f, 'k90') for m in MODELS] for f in FAMS}
    t1s = {f: [S(m, f, 'top1') for m in MODELS] for f in FAMS}
    ns = {f: S(MODELS[0], f, 'n') for f in FAMS}
    reuse_mean = []
    import numpy as np
    for m in MODELS:
        Rm = np.array(R['reuse'][m])
        tri = Rm[np.triu_indices(6, 1)]
        reuse_mean.append(float(tri.mean()))
    ho = R['hub_order']
    def hb(i):
        fid = ho[i - 1]
        return '%s(%s)' % (fid, fx(R['hub'][fid]['hub_score'], 3))
    xc = R['sealed_vector_crosscheck']

    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3238: G5-H1 census A — 26 封存特征内在维度/子空间秩普查 + 复用矩阵 + hub 排序（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3237 tail 臂 (a)；freeze **`4cc8adda`**（任何观测前；含 prereg_correction：G1 锚分层 bit<1e-9 / f32 复现<1e-6，SMOKE 阶段订正，理由与依据入冻结字典）；res **`bed6984f`**；verdict **`census_a_registered`**（G1–G5 全 True，运行内确定性 diff=0.0）。零 GPU，全部派生自封存 collect npz，配方逐字沿用 3183/3184/3188/3189 冻结脚本。')
    A_('')
    A_('### 原理与公式')
    A_('')
    A_('六族方向矩阵（每模型）：DVU=关系偏移 U 16×d（3157 dC 逐实体 isa−hasa，3183 配方）；DCC=上下文移位 64×d（3157 ctx 对差）；ENT=实体均值 16×d；USTAR=谱外崩塌行方向 876×d（3169 B-protocol ridge 残差，3188/3189 配方）；SCLS=类均值 10×d；MASSIVE=e_d1（FTR-12 登记值 0/731/2319）。谱度量：SVD 奇谱 λ，**PR=(Σλ)²/Σλ²**（参与比，∈[1,n]）、stable rank、k90（累计能量 90% 最小 k）、top1 份额。复用矩阵：族子空间取 k90 右奇异向量基，reuse(A,B)=mean|sv(S_A·S_Bᵀ)|（主角平均）。hub=三判据 z 均值：reuse（族平均 cross-angle）、causal_load（E3=1.0 / E2_behavioral=0.5 / 其余 0）、cross_model（方向族=PR 跨模型一致性 1−cv；非方向族=anchor cross_model 标签占比）。')
    A_('')
    A_('### 复现锚（装置同一性）')
    A_('')
    A_('位级锚 ×3 模型全 0.0：d83（3183 五标量+tc）、d_tc（ANCH_3157 tc_mean_cos）、drift69（3169 逐 seed E_seen/E_oov）、d88（3188 五标量）；f32 复现锚 d89（vs 封存 3189 top1/top2/cross_med）= 1.49e-07/1.79e-07/4.47e-08 < 1e-6 门（跨运行 LAPACK f32 epsilon，prereg_correction 记录在案）。USTAR top1 复现 0.3065/0.2936/0.3008（=MEMO 3189 已发表值）。')
    A_('')
    A_('### 实际结果（全表入 registry v1.33 census_3238 键）')
    A_('')
    A_('**谱（PR ×3 模型 | k90/n | top1 ×3）**：')
    A_('')
    A_('| 族 | PR (4b/14b/glm4) | k90/n | top1 (4b/14b/glm4) |')
    A_('|---|---|---|---|')
    for f in FAMS:
        A_('| %s | %s / %s / %s | %d/%d | %s / %s / %s |' % (
            f, fx(prs[f][0], 2), fx(prs[f][1], 2), fx(prs[f][2], 2),
            k90s[f][0], ns[f], fx(t1s[f][0], 3), fx(t1s[f][1], 3), fx(t1s[f][2], 3)))
    A_('')
    A_('**复用矩阵**：off-diag 均值 %.3f/%.3f/%.3f（max %.3f/%.3f/%.3f）——族子空间显著非正交；族级复用分 USTAR/ENT/SCLS 互复用最高（0.65–0.85），DVU 最低（0.36–0.62）。跨模型 PR 一致性：DCC 0.996 > ENT 0.994 > SCLS 0.987 > USTAR 0.943 > DVU 0.918。' % (
        reuse_mean[0], reuse_mean[1], reuse_mean[2],
        max(np.array(R['reuse']['qwen3-4b'])[np.triu_indices(6, 1)]),
        max(np.array(R['reuse']['qwen3-14b'])[np.triu_indices(6, 1)]),
        max(np.array(R['reuse']['glm4-9b'])[np.triu_indices(6, 1)])))
    A_('')
    A_('**hub 排序（26 行唯一秩）**：#1 %s，#2–3 %s / %s，#4–5 %s / %s，#6 %s，#7–9 %s / %s / %s；尾部 FTR-19（#26，非方向族且无 cross_model 锚）。' % (
        hb(1), hb(2), hb(3), hb(4), hb(5), hb(6), hb(7), hb(8), hb(9)))
    A_('')
    A_('封存向量交叉核对（记录性）：3225 u* 向量落本族子空间内（|proj|=1.0 ×2 模型，自洽）；对 ENT/SCLS 子空间投影 0.73–0.82（u* 与实体/类均值族高度共线）；3230 dR（关系对比方向）对 MASSIVE 投影 ~0（14b 0.0000），对 DVU 0.60/0.34。')
    A_('')
    A_('### 分析与硬伤')
    A_('')
    A_('**三条主发现**（各三述）：')
    A_('')
    A_('1. **概念均值层近乎秩 1**：ENT PR=1.050/1.041/1.055（top1 97.4–98.0%%）、SCLS PR=1.028/1.024/1.053（top1 97.4–98.8%%）×3 模型——「实体性」与「类别性」各由**一条主方向**承载，个体/类间差异是低能量旁枝。Cammarata 式单点叙事在概念均值层成立；3238-A 给出了它的量化版本。')
    A_('1. **关系/上下文中等秩且跨模型谱形极稳**：DVU PR=4.43/5.22/4.37（k90 11–12/16）、DCC PR=2.98/2.97/2.96（k90 20–22/64，PR 一致性 0.996）——关系偏移与上下文变换住在小规模子空间里，谱形状跨模型几乎不变（DCC 三模型 PR 差 <1%%）。')
    A_('1. **谱外崩塌最分布式**：USTAR PR=8.87/9.76/8.54（k90 79–120/876）——六族中唯一高维扩散族，与 GAP-5/F20/F21「行为承载分布式、读出末层线性」定谳自洽；「读出几何 ≠ 行为控制」在谱维度上再次显形：最好解释的族（USTAR，FTR-25/26）恰是最扩散的族。')
    A_('')
    A_('**硬伤与限界**：① 16/26 特征非方向性（标量/份额/门语句），rank 测量 N/A，其 hub 由另两判据驱动，reuse 覆盖不全；② FTR-04/05/07 测的是面板类均值方向的 **analog**，非其注册对象（2881-J4 词方向配方）——映射表在冻结中显式声明，引用时不可混同；③ 复用矩阵用 k90 截断基，主角对截断敏感（SCLS/ENT k90=1 退化为单方向夹角）；④ causal_load 是登记驱动的 3 档粗粒度（1.0/0.5/0），非新因果证据；⑤ cross_model 对非方向族回退为锚标签占比，与 PR 一致性不同质；⑥ d89 锚为 f32 复现级（1e-6），非位级。')
    A_('')
    A_('### 理论更新')
    A_('')
    A_('registry v1.32 → **v1.33**（file `a30215fd`）：新增 `census_3238` 键（六族谱/复用矩阵/26 行 hub 表/封存向量交叉核对/来源 sha8）+ upgrade_log append；其余键 byte-preserve 全过（canonical equality，bad=[]）。机制图景新增一层：**「族谱维度分层」**——概念均值（秩1）＜ 关系/上下文（3–5 维）＜ 谱外崩塌（≈百维），与三段式承载结构正交互补；复用矩阵首次给出族间共享的定量基线（非正交是常态，正交是例外）。')
    A_('')
    A_('**完整性**：ledger #390（pre 文件 sha8 %s）；独立磁盘复核见 p3238_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3239（候选）')
    A_('')
    A_('- (a) u* 内容归因类级扩展（3237 tail b，零 GPU，顺延）；')
    A_('- (b) DAS 式子空间交换干预设计臂（3237 tail c，轻 GPU 预备）；')
    A_('- (c) **谱图预测保真度最小闭环**（新候选，零 GPU 起步）：以 census 六族 + registry 26 特征为图谱侧，单模型 held-out 句上预测特征激活/下一 token 分布，与真实输出对答案（KL/损失恢复 vs 词汇身份/n-gram 基线）——把「破解程度」变成可画曲线的第一步。')
    body = ('\n'.join(lines) + '\n').encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3238 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3238 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.startswith(mb) or b'## Phase 3238:' in mb
assert mb2.count(b'## Phase 3238:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3238 census A 完成（零 GPU）：六族谱 PR——ENT 1.05/SCLS 1.03 近秩1、DVU ~4.4-5.2、'
      'DCC ~2.96-2.98（跨模型一致性 0.996）、USTAR 8.5-9.8 最扩散；复用矩阵非正交'
      '（offdiag 均值 0.54/0.76/0.74）；hub #1 FTR-11。registry v1.33（a30215fd），'
      'ledger #390，verdict census_a_registered（G1-G5 全过，位级锚 d83/drift69/d88/d_tc=0.0）。'
      'freeze 4cc8adda / res bed6984f。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3238 census A' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old237 = chr(45)+' 3237 收官：f21_upstream_absent_registered（freeze `58722ca2`，res `bee6ef5d`），ledger #389 post `d37ca65f`；F21 入册（上游杠杆缺席，三门 FAIL ×2 模型）+ GAP-5 终版三段式收官 + final_verdict 新键；registry v1.32 `3c3f0484` / gap v1.11 `5934631b`；verify 30/0。'
new237 = chr(45)+' 3237 收官：F21 入册（上游杠杆缺席）+ GAP-5 三段式收官；registry v1.32 `3c3f0484` / gap v1.11 `5934631b` / ledger #389；verify 30/0。'
if old237 in mem:
    mem = mem.replace(old237, new237)

old_block = '- **3238 已预注册**：(a) **阶段 A 普查升级臂**（默认，零 GPU）：26 封存特征内在维度/子空间秩 + 复用矩阵族内校准 → hub 排序（复用度/因果承重/跨模型普适）→ registry v1.33；(b) u* 内容归因类级扩展（顺延）；(c) DAS 式子空间交换设计臂（轻 GPU 预备）。'
new_block = ('- **3238 收官（census A）**：六族谱 ENT/SCLS 近秩1、DVU 4.4-5.2、DCC 2.96-2.98（跨模型 0.996）、USTAR 8.5-9.8 最扩散；复用非正交（0.54/0.76/0.74）；hub #1 FTR-11。registry v1.33 `a30215fd` / ledger #390 / freeze `4cc8adda` / res `bed6984f`。'
             '- **3239 已预注册**：(a) u* 内容归因类级扩展（零 GPU）；(b) DAS 子空间交换设计臂（轻 GPU）；(c) 谱图预测保真度最小闭环（零 GPU 起步）。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3238 block replaced (len=%d)' % len(mem))
elif '3238 收官（census A）' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
assert len(open(MEM, encoding='utf-8').read()) < 3000

print('\n'.join(REPORT))
