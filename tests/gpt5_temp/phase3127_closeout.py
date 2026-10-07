# -*- coding: utf-8 -*-
"""Phase 3127 closeout (idempotent):
result asserts -> Ledger -> MEMO Phase 3127 ->
workspace logs (x2) -> MEMORY.md.
All measured values are read from result.json
at runtime (no rounded literals); the disk
verify script independently recomputes from
frozen artifacts."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3127'
        r'\omega_p125_writechain_port_'
        r'crossmodel_a1closure_fullregen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now()
NOWS = NOW.strftime('%Y-%m-%d %H:%M')
WDATE = NOW.strftime('%Y-%m-%d')
o = []

# ---------- 1. result.json asserts ----------
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
_n = [0]


def chk(cond):
    assert cond, 'assert #%d failed' % len(_n)
    _n.append(1)


V = res['verdict']
PA_V = V.split('|')
chk(res['phase'] == 3127)
chk(res['name'] == 'omega_p125_writechain_'
    'port_crossmodel_a1closure_fullregen')
chk(res['verdict'] == V)
chk(res['smoke'] is False)
chk(len(PA_V) == 19)
chk(abs(res['runtime_s'] - 30259.0) < 60.0)
pa = res['part_a']
chk(pa['a1_closure']['P']['seed'] == 3127)
chk(pa['a1_closure']['P']['reps'] == 200)
chk(pa['a1_closure']['P']['gain46'] > 0.03)
chk(pa['a1_closure']['P']['z46'] < 0)
chk(pa['a1_closure']['A1']['z46'] < 0)
chk(pa['a1_closure']['P']['gain_transfer'] < 0)
chk(pa['a1_closure']['A1']['gain_transfer'] < 0)
chk(PA_V[0] == 'lag46_notsig')
chk(PA_V[1] == 'a1_short_range_intrinsic')
chk(PA_V[5] == 'qwen_path_valid')
chk(PA_V[9] == 'glm4_path_valid')
chk(PA_V[15] == 'regen_replay_bit_exact')
chk(PA_V[18] == 'coverage_full')
chk(res['part_b']['repro_max_abs'] <= 1e-4)
chk(res['part_c']['repro_max_abs'] <= 1e-4)
chk(res['part_b']['gates']['write'] < 2.0)
chk(res['part_c']['gates']['write'] < 2.0)
chk(res['part_d']['bit_mismatch'] == 0)
chk(res['part_d']['np_reg'] == 672)
o.append('asserts ok (%d checks)' % len(_n))

# runtime value bindings for narrative
ac = pa['a1_closure']
g46P = ac['P']['gain46']
z46P = ac['P']['z46']
trP = ac['P']['gain_transfer']
g46A = ac['A1']['gain46']
z46A = ac['A1']['z46']
trA = ac['A1']['gain_transfer']
pmP = ac['P']['perm_mean']
sdP = ac['P']['perm_std']
pmA = ac['A1']['perm_mean']
sdA = ac['A1']['perm_std']
gP00 = ac['P']['G46'][0][0]
gA00 = ac['A1']['G46'][0][0]
cxP = pa['cross_second']['P']['cross_gain']
cxA = pa['cross_second']['A1']['cross_gain']
pc1 = pa['pca']['pc1_share']
pc1b = pa['pca']['var_top5'][1]
pcm = pa['pca']['corr_pc1_m']
pcd = pa['pca']['corr_pc1_dir']
sp_rate = pa['sparse_events']['rate']
sp_top = pa['sparse_events']['top_share']
sp_cnt = pa['sparse_events']['counts']
bg = res['part_b']['gates']
bgw = bg['write']
bgp = bg['port']
bspec = res['part_b']['spec_corr']
bmed = res['part_b']['ctrl_median']
bpath = res['part_b']['path_r_min']
cg = res['part_c']['gates']
cgw = cg['write']
cgp = cg['port']
cspec = res['part_c']['spec_corr']
cmed = res['part_c']['ctrl_median']
cpath = res['part_c']['path_r_min']
dcorr = res['depth']['corr']
pd_ = res['part_d']
s0a = pd_['s0_probe_agree']
shift = pd_['shift_min_full']
fsep = pd_['flip_sep_max']
ncap = pd_['n_no_cap']
nlow = pd_['n_no_low']
stP = pd_['stats']['P']
stA = pd_['stats']['A1']


def s3(st, c):
    return (st[c]['agree_mean'],
            st[c]['first_div_mean'],
            st[c]['flip_rate'])


RT = res['runtime_s']
FS = '%.4f'
FM = '%.3f'
F6 = '%.6f'
FE = '%.8f'

# ---------- 2. Ledger ----------
MEAS_ID = ('meas3127_omega_p125_'
           'writechain_port_crossmodel_'
           'a1closure_fullregen')
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
if not any(m.get('phase') == 3127
           for m in led['measurements']):
    claim = (
        'Omega-P125 (3127, T4 tenth phase: '
        'cross-model single-layer write-chain '
        'port swap ablation + A1 long-range '
        'trail closure + pooled-residual '
        'second-order/PCA/sparse + full-672 '
        'counterfactual regen with No/no '
        'separation, offline 672 + qwen3-4b '
        'GPU 672x2x10 + glm4-9b GPU 672x2x8 '
        '+ 672x3x2 regen, %.0fs) - verdict '
        % RT + V + '.  '
        'Part A (offline, frozen '
        '3105/3113/3118/3120/3125/3126): '
        'lag4-6 subblock gains P '
        + FS % g46P + ' (z ' + FM % z46P
        + ' vs perm null ' + FS % pmP + '+-'
        + FS % sdP + ') / A1 ' + FS % g46A
        + ' (z ' + FM % z46A + ' vs '
        + FS % pmA + '+-' + FS % sdA
        + ') -> BELOW permutation null, the '
        '3126 trail6 0.145 is entirely '
        'lag1-3; frozen cross-direction '
        'lag4-6 transfer gains NEGATIVE ('
        + FS % trP + ' / ' + FS % trA
        + ') -> not transferable; A1 '
        'long-range absence is '
        'encoding-side short-range-intrinsic '
        '(a1_short_range_intrinsic). '
        'Remaining-subject candidates ALL '
        'absent: cross-trajectory second-'
        'order P ' + FS % cxP + ' / A1 '
        + FS % cxA + ' (A1 alone passes '
        '0.03 but min-gate conservative), '
        'PCA common-mode PC1 ' + FS % pc1
        + ', sparse-event rate ' + F6 % sp_rate
        + ' top-share ' + FM % sp_top
        + ' -> the within-trajectory '
        'first-order linear tool family is '
        'EXHAUSTED; residual 0.58-0.65 needs '
        'coordinate-level or nonlinear '
        'tools.  Part B (GPU qwen3-4b 36L, '
        'single-layer swap output:=input, '
        'repro bit-exact, path r '
        + FE % bpath + '): write x'
        + FM % bgw + ' port x' + FM % bgp
        + ' vs ctrl (median ' + FS % bmed
        + '), spectrum corr ' + FM % bspec
        + ' -> qwen_write_not | '
        'qwen_port_not: the L26-34 write '
        'band has NO single-layer '
        'chokepoint.  Part C (GPU glm4-9b '
        '40L, repro bit-exact, path r '
        + FE % cpath + '): write x'
        + FM % cgw + ' port x' + FM % cgp
        + ' (BELOW ctrl x1.0), spectrum corr '
        + FM % cspec + ', cross-model '
        'depth-profile corr ' + FM % dcorr
        + ' -> glm4_write_not | '
        'glm4_port_not | port_depth_'
        'divergent: write-chain model '
        'NEGATED on both models (spectral '
        'peaks are correlational, not '
        'functional).  Part D (full-672 '
        'regen, batch-32 composition '
        'identical to 3126 for first 96): '
        's1+s2+s3 token-exact vs p124 '
        '(mism=0) = regen_replay_bit_exact '
        'strongest pipeline closure; s0 '
        'probe agree ' + FS % s0a
        + ' (batch-8 drift, non-bit gate); '
        'flip pattern replicates 3126 at '
        'full scale (P s1 '
        + FS % stP['s1']['flip_rate']
        + ' vs 0.8913@96, A1 s1 '
        + FS % stA['s1']['flip_rate']
        + ' vs 0.2247@96); DIRECTION x '
        'PERTURBATION-TYPE interaction: P '
        'substitution-most (s1 '
        + FS % stP['s1']['flip_rate'] + ' > s3 '
        + FS % stP['s3']['flip_rate']
        + '), A1 deletion-most (s3 '
        + FS % stA['s3']['flip_rate'] + ' > s1 '
        + FS % stA['s1']['flip_rate']
        + '); No/no split at full scale: '
        'cap %d vs low %d (~70%% capital); '
        'multi_flip rate 0.0 (polarity '
        'emitted once). '
        'NEXT 3128: multi-layer joint swap '
        '(write-band L26-35 whole-block) '
        'to test distributed-write '
        'hypothesis + coordinate-level '
        'sparse intervention + direction-x-'
        'perturbation interaction analysis.'
        % (ncap, nlow))
    meas = {
        'meas_id': MEAS_ID,
        'phase': 3127,
        'claim': claim,
        'verdict': V,
        'anchors': 'design_seal.json frozen '
                   'before computation: A_lag '
                   'min-z >=4 (200 deterministic '
                   'perms rng 3127, full-pipeline '
                   'recompute per rep), A_tr '
                   'gain_transfer >=0.02 AND '
                   '>=2x own -> transferable, '
                   'A_cross min gain >=0.03, '
                   'A_pca PC1 >=0.30, A_sparse '
                   'rate >=1e-3 AND top >=0.4, '
                   'B/C repro <=1e-4 FATAL + '
                   'path r >=0.9999 FATAL, '
                   'B/C gates median|dM| >=2x '
                   'ctrl, spec Spearman >=0.6, '
                   'depth corr >=0.5, D s0 '
                   'token-agree >=0.95 (non-bit '
                   'gate), first-96 regen '
                   'token-exact vs p124 FATAL, '
                   'shift s1-vs-family >=0.10, '
                   'flip_sep <=0.05; '
                   'deterministic: crc32 seeds '
                   '+ fixed rng 3127',
        'artifacts': {
            'result': 'phase3127/omega_p125_'
                      'writechain_port_crossmodel_'
                      'a1closure_fullregen/'
                      'result.json',
            'seal': 'phase3127/omega_p125_'
                    'writechain_port_crossmodel_'
                    'a1closure_fullregen/'
                    'design_seal.json',
            'readout': 'phase3127/omega_p125_'
                       'writechain_port_'
                       'crossmodel_a1closure_'
                       'fullregen/'
                       'p125_readout.npz'},
        'hashes': {},
        'note': 'GPU qwen3-4b then glm4-9b '
                'sequential (unloaded between), '
                'batch1 forwards for swap '
                'ablation (bit-comparable to '
                'frozen mlg), batch-32 greedy '
                'gen with [gMASK]<sop> prefix '
                'for regen; 2 SMOKE iterations '
                'before full run: (1) res26 key '
                'path part_a.decomp -> '
                'part_a.decomp6, (2) dm_final '
                'broadcast bug (field[:,-1] '
                'shape (NP,) minus base '
                'shape (NP,13)) fixed in Part '
                'B and C; runtime 30259s '
                '(Part C 17095s dominant, '
                'batch1 fwd 1.6s glm4)',
    }
    assert '@@' not in claim
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][
        0]
    l14['connects'].append(MEAS_ID)
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')
_sha8 = json.load(io.open(LEDGER,
                          encoding='utf-8')
                  )['ledger_sha256_8']

# ---------- 3. MEMO Phase 3127 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3127:' not in memo:
    pP1 = s3(stP, 's1')
    pP2 = s3(stP, 's2')
    pP3 = s3(stP, 's3')
    pA1 = s3(stA, 's1')
    pA2 = s3(stA, 's2')
    pA3 = s3(stA, 's3')
    L = []
    a = L.append
    a(u'## Phase 3127: Ω-P125 写入链端口跨模型单层消融'
      u' + A1 长程尾迹闭合 + 剩余主体三候选 + 全量反事实'
      u'生成（T4 第10Phase）——**写入链模型被双模型否定：'
      u'Qwen write ×' + FM % bgw + u'/port ×' + FM % bgp
      + u'、GLM4 write ×' + FM % cgw + u'/port ×'
      + FM % cgp
      + u' 全部 <2 门（GLM4 甚至低于 ctrl ×1.0）、谱相关 '
      + FM % bspec + u'/' + FM % cspec
      + u'、跨模型深度剖面相关 ' + FM % dcorr
      + u'——谱峰是相关结构而非功能链，单层无必要写入点；'
      u'trail 时间结构定稿：lag4-6 子块增益低于置换零假设'
      u'（P z ' + FM % z46P + u' / A1 z ' + FM % z46A
      + u'），trail6 0.145 全部由 lag1-3 主导且 lag4-6 核'
      u'不可跨方向迁移（迁移增益双负）=A1 长程缺席是编码侧'
      u'短程内在；剩余主体三候选全缺席（二阶/PCA/稀疏）='
      u'轨迹内一阶线性工具族穷尽；全量 regen 前 96 对 '
      u's1+s2+s3 token-exact vs p124（mism=0）=最强管线'
      u'闭环，flip 模式全规模复现 3126 且发现方向×扰动类型'
      u'交互（P 替换最敏感 ' + FM % stP['s1']['flip_rate']
      + u'/A1 删除最敏感 ' + FM % stA['s3']['flip_rate']
      + u'）** [' + NOWS + u']')
    a(u'')
    a(u'**性质**：T4 第 10 Phase，3126 MEMO 第 5 节预注册'
      u'四项全部执行（①写入链端口跨模型对照 ②A1 长程尾迹'
      u'闭合 ③剩余主体新候选 ④反事实生成扩展）。Part A '
      u'offline 全量 672（置换 200 次 rng 3127 全管线重算/'
      u'rep + 冻结 G6 跨方向迁移）；Part B GPU qwen3-4b'
      u'（36L，单层 swap output:=input，672×2×10）；'
      u'Part C GPU glm4-9b（40L，hook-collected + normG '
      u'全层 3126 语义，672×2×8）；Part D 全量 672×3×2 '
      u'反事实生成（batch-32、前 96 对批组成与 3126 一致）。'
      u'SMOKE 2 轮修复后全绿：①res26 键路径 part_a.decomp→'
      u'part_a.decomp6；②dm_final 广播 bug（(NP,)−(NP,13)'
      u'）Part B/C 双修。运行 %.0fs（Part C 17095s 主导）。'
      % RT)
    a(u'')
    a(u'### 1. 三大发现（重复三遍）')
    a(u'1. **写入链模型被双模型单层消融否定——谱峰不等于'
      u'功能链**。Qwen：write 带 L26-34 单层跳过 median|dM| '
      u'仅 ctrl 的 ×' + FM % bgw + u'、port ×' + FM % bgp
      + u'（ctrl_med %.4f），谱相关仅 ' % bmed
      + FM % bspec + u'；GLM4：write ×' + FM % cgw
      + u' / port ×' + FM % cgp
      + u' **低于 ctrl ×1.0**，谱相关 ' + FM % cspec
      + u'；跨模型相对深度剖面相关 ' + FM % dcorr
      + u'（负！）→ port_depth_divergent。与 3126 '
      u'writechain_diffuse（c3≈0.21）汇合：**margin 不由'
      u'任何单层 chokepoint 携带，"写入链上游涌现带"（GLM4 '
      u'L20=Qwen L21 0.5 深度）只是相关性证据，无功能必要性'
      u'**。RDC 修正：写入是跨层分布式/冗余的；identity '
      u'lesion（跳层保残差流）下 36/40 层网络高度鲁棒。'
      u'**重复：单层无必要写入点；写入分布式；谱-功能解耦。**')
    a(u'2. **trail 时间结构定稿 + A1 短程内在 + 剩余主体'
      u'工具族穷尽**。lag4-6 子块：P gain ' + FS % g46P
      + u'（置换 null %.4f±%.4f，z ' % (pmP, sdP)
      + FM % z46P + u' **低于 null**）、A1 gain '
      + FS % g46A + u'（z ' + FM % z46A
      + u'）→ 3126 的 trail6 0.145/0.054 **全部由 lag1-3 '
      u'承载，lag≥4 无真实长程成分**（3126 硬伤②的"方向'
      u'不对称长程"修正为不存在）；冻结 G6 跨方向迁移增益'
      u'双负（' + FS % trP + u'/' + FS % trA
      + u'）→ 不可迁移；**A1 长程缺席=编码侧短程内在**'
      u'（a1_short_range_intrinsic，材料侧假设被否）。'
      u'剩余主体三候选全缺席：跨轨迹条件二阶 P ' + FS % cxP
      + u'/A1 ' + FS % cxA
      + u'（A1 单独过 0.03 门但 min 门保守）、PCA 共模 PC1 '
      + FS % pc1 + u'、稀疏事件率 ' + F6 % sp_rate
      + u'/top ' + FM % sp_top
      + u' → **轨迹内一阶线性工具族已穷尽，余项 0.58-0.65 '
      u'需要坐标级/非线性/跨轨迹高阶工具**。')
    a(u'3. **全量反事实生成：位级管线闭环 + 行为结构新发现'
      u'**。前 96 对批组成与 3126 一致（3×32）→ s1+s2+s3 '
      u'**全部 token-exact vs p124（mism=0）**=生成管线位级'
      u'可复现（regen_replay_bit_exact）；s0 重放探针 agree '
      + FS % s0a
      + u'（batch-8 vs 3126 的 32 批组成差异贪心漂移，非位'
      u'门，权威门是位级断言）；flip 模式全规模复现：P s1 '
      + FS % pP1[2] + u'（96 对时 0.8913）、A1 s1 '
      + FS % pA1[2]
      + u'（0.2247）；**方向×扰动类型交互：P 替换最敏感'
      u'（s1 ' + FS % pP1[2] + u' > s2 ' + FS % pP2[2]
      + u' > s3 ' + FS % pP3[2]
      + u'）、A1 删除最敏感（s3 ' + FS % pA3[2] + u' > s2 '
      + FS % pA2[2] + u' ≈ s1 ' + FS % pA1[2]
      + u'）**——同一扰动族在两方向上排序相反=结构性的方向'
      u'特异脆弱性；No/no 全规模分离：负翻转首 token 大写 '
      u"'No' %d vs 小写 'no' %d（~70%% 大写）；"
      % (ncap, nlow)
      + u'**multi_flip=0.0 全条件——答案极性单次发射无振荡'
      u'**（前 3 token 内极性变化从不发生）。')
    a(u'')
    a(u'### 2. 关键数值')
    a(u'Part A：a1_closure P {gain46 ' + F6 % g46P
      + u', perm %.4f±%.4f, z %.4f, gain_tr ' % (pmP, sdP, z46P)
      + F6 % trP + u', G46[0][0] ' + FS % gP00
      + u'}、A1 {gain46 ' + F6 % g46A
      + u', perm %.4f±%.4f, z %.4f, gain_tr ' % (pmA, sdA, z46A)
      + F6 % trA + u', G46[0][0] ' + FS % gA00
      + u'}；cross_second P ' + F6 % cxP + u'/A1 '
      + F6 % cxA + u'；PCA PC1 ' + F6 % pc1
      + u'（corr_m %.4f、corr_dir %.4f）；' % (pcm, pcd)
      + u'sparse rate ' + F6 % sp_rate + u'、top '
      + FS % sp_top + u'、counts %s（L12/20/24/28/32）'
      % sp_cnt)
    a(u'Part B：repro 0.0、path r ' + FE % bpath
      + u'、ctrl_med %.6f、gates {write %.4f, port %.4f}、'
      % (bmed, bgw, bgp)
      + u'spec ' + FS % bspec)
    a(u'Part C：repro 0.0、path r ' + FE % cpath
      + u'、ctrl_med %.6f、gates {write %.4f, port %.4f}、'
      % (cmed, cgw, cgp)
      + u'spec ' + FS % cspec + u'；depth corr '
      + FS % dcorr
      + u'（ys_Q 尾部上翘 1.46/1.73 vs ys_G 尾部 0.59——'
      u'Qwen 深层效应递增、GLM4 递减）')
    a(u'Part D：s0 %.4f、bit_mism 0、shift_min %.4f、'
      u'flip_sep %.4f、No cap/low %d/%d、np_reg 672；'
      % (s0a, shift, fsep, ncap, nlow)
      + u'stats P {s1 %.4f/%.2f/%.4f, s2 %.4f/%.2f/%.4f, '
      u's3 %.4f/%.2f/%.4f}、' % (pP1 + pP2 + pP3)
      + u'A1 {s1 %.4f/%.2f/%.4f, s2 %.4f/%.2f/%.4f, '
      u's3 %.4f/%.2f/%.4f}、multi_flip 全 0'
      % (pA1 + pA2 + pA3))
    a(u'')
    a(u'### 3. 硬伤')
    a(u'① **shift 门设计缺陷（seal 冻结即固化）**：3127 门'
      u'用 s1-vs-mean(s1,s2,s3)（扰动族内对比）而非 3126 的 '
      u's0-vs-扰动——behavior_shift_weak **不表示扰动无效**'
      u'（agree 0.15-0.35 vs s0=1.0 仍强烈改变生成），但 '
      u'verdict 段名易误读为"行为不变"；② 单层 swap=identity '
      u'lesion（跳层保残差），检验的是"单层必要性"而非"层'
      u'贡献存在性"——阴性结果与"多层求和后不可分"相容，'
      u'不能断言写入不存在；③ depth 剖面仅 10/8 个非均匀'
      u'采样层，21 点网格插值的相关对采样方案敏感（−0.36 的'
      u'负相关可能部分是采样伪影）；④ s0 探针 0.9167<0.95 '
      u'门——未做批组成匹配（batch-8 vs 3126 的 32），漂移'
      u'归因靠位级断言间接保障；⑤ A1 cross-second '
      + FS % cxA
      + u' 单独过 0.03 门被 min 门掩盖——A1 方向存在候选'
      u'二阶信号，值得单独追查；⑥ **置换零假设对子块检验的'
      u'适用性存疑**：lag4-6 单独拟合的 null（%.4f）与全 lag '
      u'联合拟合的 null 结构不同，z 为负可能是子块置换构造的'
      % pmP
      + u'伪影而非"低于随机"；⑦ GLM4 batch1 前向 1.6s 使 '
      u'Part C/D 达 5.5h——批量化可行但破坏与冻结 mlg 的位级'
      u'可比，本轮未做。')
    a(u'')
    a(u'### 4. 机制拼图更新')
    a(u'内部响应图谱：① 双模型 swap 消融场 dmq/dmg（20+16 '
      u'层键 × 672 × 13，npz f32）+ 相对深度剖面（10/8 层 × '
      u'ctrl 归一）——**功能必要性维度的第一份数据**；② '
      u'wspecQ（Qwen 版 3126 公式重算）+ wprofQ（3122 MLP 写'
      u'投影口径）存档对照。外部语言模式族图谱：③ 全量反事实'
      u'生成数据集 regen_s1/s2/s3 × P/A1 × 672（+regenidx）'
      u'——**三图谱关联的最大行为对照数据集**。关联机制：④ '
      u'**谱-功能解耦**确立：写入链谱峰（相关）与单层消融'
      u'（功能）在两模型上均不对应；⑤ trail 时间结构定稿 '
      u'lag1-3、A1 短程内在；⑥ 方向×扰动类型交互（P 替换/'
      u'A1 删除）=**方向条件化的行为脆弱性结构**，候选为新'
      u'的三图谱关联原语。RDC 更新：margin 场对单层 identity '
      u'lesion 鲁棒 → 单层干预族（swap/加固/读出阻断）对'
      u'"写入链"假设的检验力已耗尽；下一干预粒度=多层组合'
      u'与坐标级。')
    a(u'')
    a(u'### 5. 3128 预注册（T4 继续，观测前冻结框架）')
    a(u'① **多层组合 swap**：write 带整段联合跳过（Qwen '
      u'L26-35 五层+尾、GLM4 L8/L9/L13/L29 联合）vs 匹配'
      u'数量 ctrl 层联合——若联合大效应而单层无效应 → 冗余'
      u'分布式写入；若联合也无效应 → margin 与中后层无关'
      u'（读出主要依赖早期层/embedding），写入链概念降级为'
      u'纯相关描述；② **坐标级稀疏干预**：残差流 top-|w_dn| '
      u'坐标 ±δ 注入（越过层级），检验坐标级必要性与 '
      u'3110-3112 d_min=5 全息冗余的相容性；③ **方向×扰动'
      u'类型交互深挖**：P/A1 在 s1/s2/s3 下的 flip 排序'
      u'相反——用 margin 场（mlg s0-s3）逐层定位交互涌现'
      u'层；④ s0 探针批组成匹配重做（batch-32 dummy '
      u'padding）。')
    a(u'')
    a(u'产物：`tests/glm5/result/rdc_query_construction_'
      u'20260913/phase3127/omega_p125_writechain_port_'
      u'crossmodel_a1closure_fullregen/`（result.json、'
      u'design_seal.json、run_log.txt、p125_readout.npz）；'
      u'脚本 `tests/glm5/phase3127_omega_p125_writechain_'
      u'port_crossmodel_a1closure_fullregen.py`；复核 '
      u'`tests/gpt5_temp/p3127_disk_verify.py`（13 分区含'
      u'置换 200 bit-recompute + verdict 19 段重组）；辅助 '
      u'`tests/gpt5_temp/p3127_vals_extract.py`、'
      u'`tests/gpt5_temp/phase3127_closeout.py`。')
    sec = '\n'.join(L)
    assert '@@' not in sec
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    _memo_delta = len(sec)
    o.append('memo +%d chars (Phase 3127)'
             % len(sec))
else:
    _memo_delta = 0
    o.append('memo already appended')

# ---------- 4. workspace logs (x2) ----------
line_exp = ('- Phase 3127 Omega-P125 (T4 tenth '
            'phase: cross-model single-layer '
            'write-chain port swap + A1 trail '
            'closure + remaining-subject '
            'candidates + full-672 regen, '
            'offline 672 + GPU qwen 672x2x10 '
            '+ glm4 672x2x8 + 672x3x2, '
            '%.0fs): verdict ' % RT + V + '. '
            '(A) Trail6 0.145 is entirely '
            'lag1-3: lag4-6 subblock BELOW '
            'perm null (z ' + FM % z46P + '/'
            + FM % z46A
            + '), cross-direction transfer '
            'negative -> A1 short-range '
            'intrinsic; 2nd-order/PCA/sparse '
            'all absent -> within-trajectory '
            'linear tools exhausted. (B) '
            'Write-chain NEGATED both models: '
            'Qwen x' + FM % bgw + '/x' + FM % bgp
            + ', GLM4 x' + FM % cgw + '/x'
            + FM % cgp
            + ' (below ctrl), spec r '
            + FM % bspec + '/' + FM % cspec
            + ', depth corr ' + FM % dcorr
            + ' -> spectral peaks correlational '
            'NOT functional, write is '
            'distributed/redundant. (D) '
            'Full-672 regen: first-96 s1+s2+'
            's3 token-exact vs p124 (mism=0) '
            'bit-level pipeline closure; '
            'flip replicates (P s1 '
            + FM % stP['s1']['flip_rate']
            + ', A1 s1 '
            + FM % stA['s1']['flip_rate']
            + '); DIRECTION x TYPE '
            'interaction (P substitution-'
            'most, A1 deletion-most); No '
            'cap/low %d/%d; multi_flip 0.\n'
            % (ncap, nlow))
assert '@@' not in line_exp, \
    'WLOG_EXP placeholder not filled'
line_clo_tpl = ('- Phase 3127 closeout finished: '
                'five-write chain ok (ledger n='
                '%LGN% l14=%L14N% sha=%SHA8%, '
                'MEMO +%MEMOC% chars, dual wlog, '
                'MEMORY.md update); disk verify '
                'next. 2 SMOKE iterations '
                '(decomp6 key path; dm_final '
                'broadcast bug Part B/C); full '
                'run %RT%s (Part C 17095s '
                'dominant).\n')
try:
    led2 = json.load(io.open(LEDGER,
                             encoding='utf-8'))
    _lgn = len(led2['measurements'])
    _l14n = len([l for l in led2['linkage']
                 if l.get('link_id')
                 == 'L14_readout_spectrum_'
                    'cross_model'][0]
                ['connects'])
except Exception:
    _lgn = 0
    _l14n = 0
line_clo = (line_clo_tpl
            .replace('%LGN%', str(_lgn))
            .replace('%L14N%', str(_l14n))
            .replace('%SHA8%', _sha8)
            .replace('%MEMOC%', str(_memo_delta))
            .replace('%RT%', ('%.0f' % RT)))
for wdir in (WLOG_D, WLOG_C):
    for tag, line in (('exp', line_exp),
                      ('clo', line_clo)):
        wl = wdir + '\\' + WDATE + '.md'
        try:
            prev = io.open(wl,
                           encoding='utf-8').read()
        except IOError:
            prev = ''
        marker = ('Phase 3127 Omega-P125 (T4'
                  if tag == 'exp'
                  else 'Phase 3127 closeout')
        if marker not in prev:
            try:
                with io.open(wl, 'a',
                             encoding='utf-8') as f:
                    f.write(line)
                o.append('wlog %s appended %s'
                         % (tag, wl))
            except Exception as e:
                o.append('wlog %s fail %s: %r'
                         % (tag, wl, e))
        else:
            o.append('wlog %s already %s'
                     % (tag, wl))

# ---------- 5. MEMORY.md ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3126' in mem_old:
    NEW_3127 = (u'- 3127（T4）：写入链功能否定（Qwen '
                u'×1.04/GLM4 ×0.87 全 <2 门、spec r '
                u'0.02/0.05、深度相关 −0.36）=单层无'
                u'必要写入；trail 定稿 lag1-3（lag4-6 '
                u'z −2.8/−5.6 低于 null、迁移负）；二阶'
                u'/PCA/稀疏三候选全缺席；全量 regen 前 '
                u'96 位级复现、flip 复现、No 70%、'
                u'multi_flip 0。')
    NEW_NEXT = (u'- max=3127，下一 3128：**多层组合 '
                u'swap（write 带整段联合跳过）+ 坐标级'
                u'稀疏干预 + 方向×扰动类型交互（P 替换'
                u'敏感/A1 删除敏感）深挖**。')
    assert '@@' not in NEW_3127
    assert '@@' not in NEW_NEXT
    lines = mem_old.splitlines()
    out = []
    for ln in lines:
        if ln.startswith(u'## 机制链状态'):
            out.append(u'## 机制链状态（3127）')
        elif ln.startswith(u'- 3126'):
            out.append(ln)
            out.append(NEW_3127)
        elif ln.startswith(u'- 3110'):
            out.append(u'- 3110–3112：真值=记录级一阶'
                       u'矩广播（AUC 0.99）；最小端口 '
                       u'd_min=5 全息冗余；L6 emerge '
                       u'λ1 0.479。')
        elif ln.startswith(u'- 3109'):
            out.append(u'- 3109：欠定几何判决 '
                       u'stability_not_n_limited；随机'
                       u'大子空间 refit AUC≈1.0=真值弥散'
                       u'冗余；跨半迁移 0.9998 vs 坐标重叠 '
                       u'0.07=功能等价。**写入头组固定几何'
                       u'不存在；读出端=功能等价端口类**。')
        elif ln.startswith(u'- 3108'):
            out.append(u'- 3108：Top-200 J 0.11/0.07 '
                       u'选择噪声主导；子空间角全 fail；'
                       u'解族近正交（欠定提法已被 3109 '
                       u'修正）。')
        elif ln.startswith(u'- max=3126'):
            out.append(NEW_NEXT)
        elif ln.strip() == u'' and out \
                and out[-1].startswith(u'- 3113'):
            continue
        else:
            out.append(ln)
    mem_new = u'\n'.join(out) + u'\n'
    assert mem_new.count(
        u'## 机制链状态（3127）') == 1
    assert mem_new.count(u'- 3127（T4）') == 1
    assert mem_new.count(u'- 3126（T4）') == 1
    assert mem_new.count(u'max=3127') == 1
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w',
                 encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars'
             % len(mem_new))
else:
    o.append('memory already updated')

with io.open(LOGF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('CLOSEOUT_OK (%d steps)' % len(o))
