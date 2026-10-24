# -*- coding: utf-8 -*-
"""Phase 3250 closeout (five writes, idempotent) - dual arm:
(b) science: kernel numerical boundary + joint grid
(a) engineering: /atlas/fidelity v2 endpoint + FidelityMode.jsx incumbent
1) result+engineering assertions  2) ledger #402  3) MEMO Phase 3250 section
4) dual wlog  5) workspace MEMORY.md replace
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3250\g5h13_kernel_boundary\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 15:45'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D = R['delta_vs_G0']
H = R['headline']

# ---------- 1. result assertions (science arm) ----------
A = R['anchors']
assert R['verdict'] == 'kernel_boundary_registered'
assert R['design_sha8'] == 'c3aa103e' and R['res_sha8'] == '9d0f1825'
assert R['model'] == 'qwen3-4b' and R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_transfer_realism',
                                  'clause_b_probe_closure_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
for key, ph in (('G0', 3240), ('T3_rep', 3242), ('XC_b4', 3246),
                ('XC_p2_b4', 3247), ('XC_p3_b4', 3248), ('K_p3_T3', 3248),
                ('XC_p5_b4', 3249), ('XC_p3_b6', 3249)):
    assert AGG[key]['d_vs_sealed_%d_max' % ph] == 0.0, (key, ph)
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
real = H['transfer_realism']
for k in ('p6', 'hyb'):
    assert real[k]['margin'] > 0.0, (k, real[k])
assert H['numeric_probe_ok'] is True
# deg-6 probe: finite, cond flat (no numerical explosion)
ct = H['cond_trend_max']
assert ct['p6'] < 1e6 and max(ct.values()) / min(ct.values()) < 2.0
# joint grid: p5_b8 best new, positive closure
assert H['best_new_item'] == 'XC_p5_b8' and H['saturated'] is False
assert 0.0 < H['best_new_closure'] < 0.2
# p5_b8 combines degree fidelity gain with beta r2/KL gain
assert AGG['XC_p5_b8']['r2_oov_mean'] > AGG['XC_p5_b4']['r2_oov_mean'] + 0.02
assert AGG['XC_p5_b8']['kl_mean'] < AGG['XC_p5_b4']['kl_mean'] - 0.05
assert min(AGG['XC_p5_b8']['fam_fid_mean']['SCLS'],
           AGG['XC_p5_b8']['fam_fid_mean']['ENT']) > min(
    AGG['XC_p5_b4']['fam_fid_mean']['SCLS'],
    AGG['XC_p5_b4']['fam_fid_mean']['ENT'])
# DVU erosion bounded, grows with degree
assert -0.05 < D['XC_p5_b8']['d_fam_fid']['DVU'] <= 0.0
assert D['XC_p6_b4']['d_fam_fid']['DVU'] <= D['XC_p5_b4']['d_fam_fid']['DVU'] + 1e-9
# hybrid degenerates to poly3 (metric-level identity)
assert abs(AGG['XC_hyb_b4']['fam_fid_mean']['SCLS']
           - AGG['XC_p3_b4']['fam_fid_mean']['SCLS']) < 1e-3
# ORC replicates sealed 3246 (semantic)
assert abs(AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] - 0.083) < 0.01
REPORT.append('1 science assertions: OK (verdict=%s res=%s)' % (R['verdict'], R['res_sha8']))

# ---------- 1b. engineering arm assertions ----------
svc = open(os.path.join(ROOT, 'server', 'ai_rnd_service.py'), encoding='utf-8').read()
assert svc.count('_FIDELITY_SOURCES') >= 1 and '"3250"' in svc
assert 'atlas_fidelity_v2' in svc and '_FIDELITY_INCUMBENT' in svc
fj = open(os.path.join(ROOT, 'frontend', 'src', 'components', 'app', 'rdc_fusion', 'FidelityMode.jsx'), encoding='utf-8').read()
assert 'atlas_fidelity_v2' in fj.replace('fidelity.v2', 'atlas_fidelity_v2') or 'fidelity.v2' in fj
for marker in ('XC_p5_b8', 'SCLS 缺口收线', 'GapLineChart', '3250 现役 XC_p5_b8'):
    assert marker in fj, marker
ep = open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3250_endpoint_out.txt'), encoding='utf-8').read()
assert 'ENDPOINT CHECK OK' in ep
ui = os.popen('cd /d %s && dir /b tests\\gpt5_temp\\p3250_ui_check.js' % ROOT).read()
assert 'p3250_ui_check.js' in ui
REPORT.append('1b engineering assertions: OK (endpoint v2 + FidelityMode v2 + endpoint-check sealed numbers)')

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = '87e2762c'
already = [e for e in led['measurements'] if e.get('phase') == 3250]
if not already:
    assert len(led['measurements']) == 401 and led['ledger_sha256_8'] == PREV_POST
    eS = lambda rn: round(AGG[rn]['fam_fid_mean']['SCLS'], 3)
    eE = lambda rn: round(AGG[rn]['fam_fid_mean']['ENT'], 3)
    entry = {
        "phase": 3250,
        "name": "g5h13_kernel_boundary",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b (zero-GPU deg-6 numeric probe + poly-rbf hybrid + "
                 "degree-beta joint grid; + engineering arm: /atlas/fidelity "
                 "v2 endpoint and FidelityMode.jsx incumbent client)",
        "verdict": ("kernel_boundary_registered|NUMERICAL BOUNDARY NOT "
                    "REACHED at deg<=6 (cond flat 3.0e5-3.9e5, all finite): "
                    "the ladder stops for VALUE reasons, not numerical ones "
                    "- p6 closure +0.024 with DVU erosion -0.022; joint grid "
                    "XC_p5_b8 (poly5 x beta8) is the new incumbent: SCLS %s / "
                    "ENT %s, r2 0.241, KL 0.987, DVU -0.017 / DCC +0.003 "
                    "(closure +0.032) - degree fidelity gain and beta r2/KL "
                    "gain STACK; hybrid kernel degenerates to poly3 "
                    "(metric identity, 5e-5 f32 noise); engineering arm "
                    "shipped: fidelity endpoint v2 (7 sealed phases + "
                    "incumbent annotations), FidelityMode.jsx gap-closing "
                    "curve, playwright UI 19/19"
                    % (eS('XC_p5_b8'), eE('XC_p5_b8'))),
        "evidence_level": ("deepening arm (preregistered MEMO 3249 tail; "
                           "EIGHT bit-level anchors d40/d42/d46/d47/d48a/"
                           "d48b/d49a/d49b == 0.0 per-seed on all 876 rows "
                           "vs sealed 3240/3242/3246/3247/3248/3249; anchors "
                           "d83/d_tc/drift69/d88 == 0.0, d89 <1e-6; census "
                           "spectra cross-checked; probe finite for p6/hyb; "
                           "realism margins positive; endpoint numbers "
                           "cross-checked against sealed result.json in a "
                           "direct-coroutine test; ORC_ls_cls is "
                           "EVALUATION-ONLY closure denominator)"),
        "model_scope": "qwen3-4b (per prereg)",
        "prereg_id": "G5-H13 (MEMO 3249 tail)",
        "superseded_by": None,
        "detail": (
            "Kernel-family numerical boundary + joint grid (zero GPU), 15 "
            "recipes + client arm. Findings (3-seed means): (1) DEG-6 "
            "PROBE: ladder still monotone (XC_p6_b4 SCLS %s vs p5 %s) but "
            "closure +0.024 and DVU erosion -0.022 (p5 -0.017, p3 +0.000) "
            "- marginal gain shrinking while regression grows; (2) "
            "NUMERICAL BOUNDARY NOT REACHED: cond trend p3 3.91e5 / p4 "
            "3.12e5 / p5 3.00e5 / p6 3.45e5 (flat, all finite) - the "
            "practical stopping point is VALUE-based (gain/regression "
            "trade-off), around deg 5-6; (3) JOINT GRID: beta 6-8 gains "
            "STACK with degree gains - XC_p5_b8 SCLS %s / ENT %s / r2 "
            "0.241 / KL 0.987 (closure +0.032, best new) vs XC_p5_b4 "
            "-0.467/-0.301/0.204/1.087; p4_b6/b8 closure -0.021; (4) "
            "HYBRID DEGENERATES: 0.5*poly3+0.5*rbf produces metric-level "
            "identity with pure poly3 (5e-5, f32 noise) - the rbf half "
            "adds nothing outside the poly3 fitted span; (5) realism "
            "margins positive (p6 +0.091, hyb +0.090). Engineering arm: "
            "/atlas/fidelity schema v2 (7 sealed phases 3239/3240/"
            "3246-3250 + agg_kl from sealed results + closure where npz "
            "carries it + incumbent annotations XC_p5_b8/XC_p5_b4/"
            "XC_p3_b6/K_p3_T3); FidelityMode.jsx v2 (incumbent card, "
            "SCLS gap-closing milestone curve 3240->3250, new chart "
            "series); direct-coroutine endpoint check OK (sealed numbers "
            "match); playwright live UI check 19/19 (one check-string "
            "fixed to the sealed 0.986 during verification, no product "
            "change). Registry untouched (v1.33 head, byte-verified). "
            "Gates G1-G4 all True; determinism 0.0. freeze c3aa103e, "
            "res 9d0f1825. Pre-append ledger file sha8 %s."
            % (eS('XC_p6_b4'), eS('XC_p5_b4'), eS('XC_p5_b8'), eE('XC_p5_b8'),
               pre_file_sha8)),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3250', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #402 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3250 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 402
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=402 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3250:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3250: G5-H13 核函数族数值边界 + 联合网格 + 现役客户端化（零 GPU + 工程臂） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3249 tail（双臂）；freeze **`c3aa103e`**；res **`9d0f1825`**；verdict **`kernel_boundary_registered`**（G1–G4 全过，SMOKE 一次通过）。**八位级锚全零**：d40/d42/d46/d47/d48a/d48b/d49a（XC_p5_b4 vs 3249）/d49b（XC_p3_b6 vs 3249），均 per-seed 全 876 行。registry 不动（v1.33 仍为头）。')
    A_('')
    A_('### 核心发现（科学臂）：数值边界未到，停点是价值停点（headline: best=XC_p5_b8 closure +0.032, SATURATED=False）')
    A_('')
    A_('- **deg-6 探针**：阶梯仍单调（XC_p6_b4 SCLS −0.453 vs p5 −0.467），但闭合率降至 +0.024、DVU 侵蚀加深（−0.022）——**边际收益收缩与回退增长同时发生**。')
    A_('- **数值边界未到**：cond 趋势平坦（p3 3.91e5 / p4 3.12e5 / p5 3.00e5 / p6 3.45e5，全 finite）——**deg≤6 无数值爆炸，实际停点是价值停点（增益/回退交换），位于 deg 5–6 带**。')
    A_('- **联合网格（新现役）**：**XC_p5_b8 = poly5×β8，degree 保真增益与 β r2/KL 增益可叠加**——SCLS **−0.449** / ENT **−0.273**（closure +0.032 最优）/ r2 0.241 / KL 0.987 / DVU −0.017 / DCC +0.003。保真缺口曲线：G0 −1.751 → T3 −1.293 → XC_b4 −0.772 → K_p3_T3 −0.529 → **XC_p5_b8 −0.449** → oracle 上界 +0.083，剩余 −0.53。')
    A_('- **混合核退化**：0.5·poly3+0.5·rbf 与纯 poly3 度量恒等（差 5e-5，f32 噪声级）——**rbf 半核在 poly3 拟合 span 外无增量**，核族互补性证伪。')
    A_('- realism margin 全正（p6 +0.091 / hyb +0.090）。')
    A_('')
    A_('### 工程臂（现役配方客户端化，自 3248 顺延两次后落地）')
    A_('')
    A_('- **`/api/ai-rnd/atlas/fidelity` schema v2**：封存源扩至 7 相位（3239/3240/3246–3250）；3246+ 的 agg_kl 取自各自封存 result.json（npz immutable 不补写）；closure 按 npz 有无透传（3246 封存 npz 无该字段）；新增 **现役标注块**（XC_p5_b8 现役综合最优 / XC_p5_b4 保真维 / XC_p3_b6 R²KL 维 / K_p3_T3 纯拉力）。')
    A_('- **FidelityMode.jsx v2**：现役配方摘要卡、**SCLS 缺口收线里程碑曲线**（G0→T3→XC_b4→XC_poly_b4→XC_p3_b4→现役→oracle，封存查表零口径计算）、族图新增 3246 XC_b4 与 3250 现役序列。')
    A_('- **验证**：端点直调校验（封存数字交叉核对）OK；playwright 实机 **19/19**（一处校验字符串按封存值 0.986 修正，无产品改动）。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **映射函数类收线**：3247–3250 四连击把声明体制内缺口从 −0.772 推至 **−0.449**（ORC 上界 +0.083，剩余 −0.53）。核族菜单（degree×β×hybrid×joint）已系统扫毕：**β=8×poly5 为价值最优停点**；继续 deepening 的期望收益低于回退增长。')
    A_('1. **转正路径**：剩余缺口仍为 INFORMATION_GAP（3246 定位）；声明体制内逼近极限，后续转正需越出泄漏声明界（ORC 语义）或新机制类别（偏移型分量）。')
    A_('1. **oracle 语义**：ORC_ls_cls 仅作闭合率分母（EVALUATION-ONLY）。')
    A_('')
    A_('**限界**：零 GPU；结论限于 qwen3-4b 面板与 B-协议 split；deg>6 未探（cond 平坦提示数值空间仍有余量，但价值停点已现）；u* 内容归因继续顺延。')
    A_('')
    A_('**完整性**：ledger #402（pre 文件 sha8 %s）；产物 result.json + kbound.npz + smoke 子目录 + 端点/UI 校验脚本；独立磁盘复核见 p3250_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3251（候选）')
    A_('')
    A_('- (a) **G5 保真收线小结段**（文档臂）：把 3239–3250 保真收线链写成 MEMO 小结 + registry gap 附录候选条目（零 GPU）；')
    A_('- (b) **u* 内容归因**（自 3244 顺延多次，优先级提升）：u* 方向的词汇/类内容分解（零 GPU）；')
    A_('- (c) checkpoint 轨迹与 5 appendix GPU 项（维持挂账）。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3250 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3250 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3250:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3250 核函数族数值边界+联合网格+现役客户端化完成（双臂，registered）：deg≤6 cond 平坦无数值爆炸——停点是价值停点（deg 5–6 带，p6 闭合 +0.024/DVU −0.022）；'
      '**XC_p5_b8（poly5×β8）= 新现役**（SCLS −0.449/ENT −0.273/r2 0.241/KL 0.987，闭合 +0.032）——degree 与 β 增益可叠加；混合核与 poly3 度量恒等（互补性证伪）。'
      '工程臂：fidelity 端点 v2（7 相位+现役标注）+ FidelityMode v2（现役卡+缺口收线曲线），端点直调封存核对 OK，playwright 实机 19/19。'
      '八位级锚 d40–d49b 全零。verdict kernel_boundary_registered；freeze c3aa103e / res 9d0f1825，ledger #402，registry v1.33 不动。'
      '3251 已预注册：保真收线小结段 + u* 内容归因（顺延多次后提级）。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3250 核函数族数值边界' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3250 已预注册**：(a) 现役配方客户端化（工程，顺延两次）；(b) deg>5 探针 + 核族收尾（零 GPU）；(c) u* 顺延。')
new_block = ('- **3250 收官（kernel boundary + 客户端化，双臂 registered）**：deg≤6 cond 平坦无数值爆炸——停点是价值停点（deg 5–6 带）；**XC_p5_b8（poly5×β8）= 新现役**（−0.449/−0.273/r2 0.241/KL 0.987，闭合 +0.032，degree×β 增益叠加）；混合核≡poly3（互补性证伪）。工程臂：fidelity 端点 v2（7 相位+现役标注）+ FidelityMode v2（现役卡+缺口收线），UI 实机 19/19。八位级锚全零。ledger #402 / freeze `c3aa103e` / res `9d0f1825`。'
             '- **3251 已预注册**：(a) 保真收线小结段（文档臂）；(b) u* 内容归因（提级）；(c) checkpoint/appendix 挂账。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
    # compress older phase entries to keep under the 3000-char limit
    old_early = ('3238 census A（hub #1 FTR-11，registry v1.33 `a30215fd` #390）；3239 KL 恢复 38.7%（#391）；3240 F2 转正未达（#392）；3241 G1e 回退越界 partial（#393）；3242 T3_dual 零回退正向 partial（#394）；3243 seen-only 信息极限（#395）；3244 归因反转+R1_clean 零回退+sealed-3241 伪影发现（#396 `482eeaa4`）；3245 X_combo 组合加性但 FLIP 未达 −0.797（#397 `ef039473`）——细节回 MEMO。')
    new_early = ('3238–3245 收线链（census/KL 恢复/F2/G1e/T3_dual/信息极限/归因反转/combo，#390–#397）——细节回 MEMO。')
    if old_early in mem:
        mem = mem.replace(old_early, new_early)
    old_48 = ('- **3248 收官（poly deepening，registered）**：degree 阶梯未饱和（1→2→3 单调，XC_p3_b4 −0.550/−0.356 零回退）；**K_p3_T3 纯拉力全场最优**（−0.529/−0.387，闭合 +0.165）；3246 拉力上限不约束 k=14 尖锐 span；p2 基 β 最优移至 6–8（r2 0.242）。归一化/σ0.5 证伪。四位级锚 d40/d42/d46/d47 全零。ledger #400 / freeze `b47042fd` / res `7d2acdb6`。')
    new_48 = ('- **3248**：degree 1→3 单调未饱和；K_p3_T3 纯拉力最优（闭合 +0.165）；p2 基 β 最优移至 6–8。#400 `b47042fd`。')
    if old_48 in mem:
        mem = mem.replace(old_48, new_48)
    old_49 = ('- **3249 收官（deg extension，registered）**：degree 2→5 继续单调（XC_p5_b4 −0.467/−0.301，闭合 +0.102，SATURATED=False）；DVU 侵蚀随阶增长（p5 −0.017）；p3 基 β=8 r2 0.243/KL 0.982；**combo 在 p3 基失效**（闭合 −0.01）——combo 是 lin/p2 时代装置。六位级锚 d40–d48b 全零。ledger #401 / freeze `6eb9bed8` / res `8104fd1b`。')
    new_49 = ('- **3249**：degree 2→5 单调（−0.467，闭合 +0.102）；DVU 侵蚀随阶增长；combo 在 p3 基失效。#401 `6eb9bed8`。')
    if old_49 in mem:
        mem = mem.replace(old_49, new_49)
elif '3250 收官（kernel boundary' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3250 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print(chr(10).join(REPORT))
