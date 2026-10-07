# -*- coding: utf-8 -*-
"""Phase 3056 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3056'
     r'\omega_p53_gamma_spectrum_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory')
MEMO_W = WLOG_DIR + r'\MEMORY.md'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['verdict']
assert verdict == 'gamma_vocab_null_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a174_z55_replay_ok'] is True
assert an['a175_gamma_stats_diff'] == 0.0
assert an['a176_tt_diff'] == 0.0
t2 = st['T2_token_gain']
assert abs(t2['min'] - 2.4908726203825773) < 1e-12
assert abs(t2['med'] - 2.8320266492151696) < 1e-12
assert abs(t2['max'] - 4.117437677681102) < 1e-12
t3 = st['T3_mass_enrich']
assert abs(t3['mass_top']
           - 0.0007487806741309248) < 1e-12
assert abs(t3['mass_bot']
           - 0.001471529670905527) < 1e-12
assert abs(t3['null_med']
           - 0.0013115224668679769) < 1e-12
assert t3['p_top'] == 1.0
assert t3['G_TOP'] == 200 and t3['R_NULL'] == 2000
t4 = st['T4_vote_overlap']
assert abs(t4['spearman_abs_gamma_abs_votes']
           - 0.010388947403174956) < 1e-12
assert abs(t4['p_enr']
           - 0.07946026986506746) < 1e-12
assert t4['sig_in_top'] == 119
assert abs(t4['sig_expected'] - 80.5) < 1e-9
t5 = st['T5_layer_profiles']
assert abs(t5['med_cos_post']
           - 0.971615109684663) < 1e-12
assert abs(t5['corr_gamma_colstd_pearson']
           - -0.8200857026001915) < 1e-12
assert abs(t5['corr_gamma_colstd_spearman']
           - -0.6763770327285052) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3056
           for m in led['measurements']):
    claim = (
        'Omega-P53 (plan 3056 A) - gamma '
        'pre-alignment direction-spectrum '
        'inventory, run1 fp32 weights-only '
        'authoritative (12.5s, zero forwards; '
        'offline from model weights + z48 TT/LG '
        '+ z55 T2 arrays + z2802 votes). '
        'RESULTS: (1) VERDICT '
        'gamma_vocab_null_qwen - NO vocab-level '
        'pre-alignment: the 24 target logit '
        'directions carry LESS mass on the '
        '200 top-gain tokens (0.000749) than on '
        'random 200-token sets (null med '
        '0.001312, p_top = 1.0) and less than '
        'the bottom-200 negative control '
        '(0.001472); Spearman(|gamma|,|votes|) '
        '= 0.0104 and top-256-gamma-channel '
        'vote enrichment p = 0.079 n.s. (sig '
        'count 119 vs expected 80.5, mild '
        'descriptive only). (2) Token gain '
        'spectrum g_v = ||gamma*W_U[v]|| / '
        '||W_U[v]||: min 2.49 / med 2.83 / max '
        '4.12 - gamma amplifies every token '
        'row 2.5-4x; top tokens are chat/tool '
        'template markers (think/im_start/'
        'tool_call) and rare CJK glyphs, not '
        'semantic content words. (3) STRONGEST '
        'STRUCTURE: corr(gamma, colstd(W_U)) = '
        '-0.820 Pearson / -0.676 Spearman - '
        'gamma is ANTI-correlated with the '
        'unembedding column variance, i.e. the '
        'readout pre-alignment is a WHITENING '
        'filter (suppress the dominant raw '
        'channel variance, boost low-variance '
        'channels), not a token-row aligner. '
        '(4) gamma_final cos vs 72 layer RMS '
        'gammas: post-attn 0.97 / input 0.93 '
        'raw but mean-centered post +0.31 / '
        'in -0.19 - mild shared channel '
        'structure with post-attention norms. '
        'Reconciles with 3055: the target-'
        'specific gain lives in the d_tan x '
        'whitened-readout interaction geometry, '
        'not in static token identity.')
    meas = {
        'meas_id': 'meas3056_omega_p53_gamma_'
                   'spectrum_qwen',
        'phase': 3056,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a116d source seals 3044-3055; '
                   'a174 z55 T2 replay bit-equal; '
                   'a175 gamma stats diff 0.0 vs '
                   'z55 GAMMA_STATS; a176 TT '
                   'recompute diff 0.0',
        'artifacts': {
            'result': 'phase3056/omega_p53_'
                      'gamma_spectrum_qwen/'
                      'result.json',
            'npz': 'phase3056/omega_p53_'
                   'gamma_spectrum_qwen/'
                   'omega_p53_gamma_spectrum_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (12.5s, fp32 '
                'weights-only, zero forwards); null '
                'R = 2000 random 200-token sets; '
                'enrichment null permutes channel '
                'identity',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 195
    l14['connects'].append({
        'meas_id': 'meas3056_omega_p53_gamma_'
                   'spectrum_qwen',
        'phase': 3056,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P53: gamma '
                        'pre-alignment spectrum '
                        'inventory - NO vocab-level '
                        'pre-alignment (target mass '
                        'on top-gain tokens 0.000749 '
                        '< random null 0.001312, '
                        'p = 1.0; vote-spectrum '
                        'overlap rho 0.0104, '
                        'enrichment p 0.079 n.s.); '
                        'the real structure is '
                        'corr(gamma, colstd(W_U)) = '
                        '-0.82: gamma is a readout '
                        'WHITENING filter '
                        '(anti-variance channel '
                        'reweighting), so the 3055 '
                        'target-specific gain lives '
                        'in interaction geometry, '
                        'not static token identity '
                        '(gamma_vocab_null_qwen)'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted n=%d l14=%d'
             % (len(led['measurements']),
                len(l14['connects'])))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3056:' not in memo:
    sec = u'''## Phase 3056: Ω-P53 γ 预对齐方向谱清点——词表级 null + 白化滤波结构（gamma_vocab_null_qwen） [%(created)s]

**判决：`gamma_vocab_null_qwen`**（run1 fp32 weights-only 权威 12.5s，零前向，锚核心一次全过）。锚：a116d 源封印 3044-3055 / a174 z55 T2 逐量 bit 复放 / a175 γ 统计 vs z55 GAMMA_STATS bit 0.0 / a176 TT 逐对重算 bit 0.0。

### 设计
**T2 词表读出增益谱（离线）**：g_v = ‖γ⊙W_U[v,:]‖/‖W_U[v,:]‖ 全词表；**T3 目标方向质量富集（主检验）**：24 目标方向在 top-200 增益 token 上的质量占比 vs 随机 200-token 集 null（R=2000）+ bottom-200 负对照；**T4 2802 投票谱通道重叠**：Spearman(|γ|,|votes|) + top-256 γ 通道 votes 富集置换检验；**T5 跨层 norm γ 剖面**：γ_final vs 72 层内 RMSNorm γ + corr(γ, colstd(W_U))。

### 核心结果（重复三遍）
**① 词表级预对齐不存在（主检验 null）**：24 目标方向在 top-200 增益 token 上的质量 0.000749 **低于**随机 null med 0.001312（p_top=1.0）也低于 bottom-200 负对照 0.001472——γ 的高增益 token 不承载目标方向质量；投票谱重叠 Spearman=0.0104、富集 p=0.079 不显著（sig 119 vs 期望 80.5，仅描述性）。**② 增益谱均匀偏置**：g_v min 2.49 / med 2.83 / max 4.12——γ 把**每个** token 读出行放大 2.5-4 倍；top token 是 chat/tool 模板标记（think/im_start/tool_call）与生僻字形，非语义内容词。**③ 最强结构 = 白化滤波**：corr(γ, colstd(W_U)) = **−0.820**（Pearson）/ −0.676（Spearman）——γ 与 unembedding 列方差**反相关**：读出预对齐的本质是**压低高方差主通道、抬升低方差通道的反方差重加权（whitening）**，不是词表行预对齐。**④ 跨层剖面**：raw cos 高（post 0.97 / in 0.93，均值主导），去均值后 post +0.31 / in −0.19——γ 与 post-attention norm 有温和共享通道结构。

### 机制链定版
γ 预对齐与 3055 的"方向特异增益"如何共存：**增益不住在静态 token 身份里，住在 d_tan × 白化读出的交互几何里**。γ 先做反方差白化（把读出基从原始偏置谱拉平），语境写入的 d_tan 恰好在这个白化基里对准目标方向——"γ 定基"的确切含义是**白化定基**，方向特异由写入侧（KV 门位+语境组装）供给。写入-读出闭环修订：写入=语境组装造 d_tan，读出=γ 白化放大其对准。

### 方法论入册
- **weights-only 相位**：零前向设计（全部量从权重+上游 npz 离线导出），锚全部离线 bit 复核——最廉价的一类相位，适合清点/库存型问题。
- **token 增益谱 g_v**：γ⊙W_U 行范数比——判别"预对齐住在哪一层"（token 行 vs 通道几何）的直接仪表。
- **负对照带符号解读**：mass_top < mass_bot 说明 top-γ token 不仅不富集反而抑制目标质量（白化把质量搬去低方差通道）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3056/omega_p53_gamma_spectrum_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3057 菜单**——A（主选）**白化几何验证**：γ⊙W_U 的有效谱/条件数 vs W_U（白化定量）+ γ 反比于 colstd 的解析拟合 + d_tan 在白化基下的对齐增益重演（3055 T2 的白化解释）；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区+γ 白化）；D 门区 2D 易感图。"好的，继续"即进 3057 A。
''' % {'created': created,
           'script8': seal['script_sha256_8'],
           'result8': seal['result_sha256_8'],
           'npz8': seal['npz_sha256_8'],
           'exec8': seal['exec_sha256_8'],
           'n': len(led['measurements']),
           'l14': len(l14['connects'])}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 十八、3056 增补' not in aud:
    add = u'''
    
---

## 十八、3056 增补：γ 预对齐方向谱清点——词表 null + 白化滤波（Omega-P53，判决 gamma_vocab_null_qwen）

1. **词表级预对齐不存在**：24 目标方向在 top-200 γ 增益 token 上的质量低于随机 null（p=1.0），投票谱重叠 ρ=0.0104、富集 p=0.079 均不显著——γ 的高增益 token 是模板标记与生僻字形，非语义方向。
2. **γ = 读出白化滤波器**：corr(γ, colstd(W_U)) = −0.82——反方差通道重加权，压低原始读出谱的主导通道、抬升低方差通道；g_v 全词表 2.5-4 倍均匀放大。
3. 3055 修订：方向特异增益不住在 token 身份里，而在 d_tan × 白化基的交互几何——"γ 定基"= 白化定基，方向特异由写入侧供给。
4. HDMCC 修正：读出端应画为"γ 白化读出基"而非"γ 词表预对齐"；两图谱接口的读出端是频谱整形器，语义方向对准全部由写入侧完成。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3056' not in prev:
    line = ('- Phase 3056 Omega-P53 gamma spectrum '
            'inventory: verdict gamma_vocab_null_'
            'qwen (run1 fp32 weights-only 12.5s, '
            'zero forwards; anchors a116d/a174/'
            'a175/a176 all bit-pass). RESULTS: NO '
            'vocab-level pre-alignment - target '
            'mass on top-200 gain tokens 0.000749 '
            '< random null 0.001312 (p=1.0), vote-'
            'spectrum rho 0.0104, enrichment p '
            '0.079 n.s.; STRONGEST STRUCTURE corr'
            '(gamma, colstd(W_U)) = -0.82: gamma '
            'is a readout WHITENING filter (anti-'
            'variance reweighting), gain_v 2.5-4x '
            'uniform; top tokens are template '
            'markers. 3055 target-specific gain '
            'lives in d_tan x whitened-readout '
            'interaction geometry. Audit addendum '
            '18; ledger 195/L14 163.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md rewrite (<=3000 chars) ----------
mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧产物；负结果/锚失败/崩溃如实登记；verdict 单分支赋值；非权威 run 登记进 corrections+seal。
4. 统计纪律：obs/null 同量纲同范围；null 限同一行集；**负对照带符号解读（top 低于 null 时写抑制而非富集）**；近似质量判据可替代 null（阈值预注册）；随机方向对照判增益特异性。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 锚形状核对纪律；捕获库跨相位复用（上游 npz+抽样重捕获 bit 锚+派生量逐对复现锚）。
- 二维输出索引（3050）：hook out[0] 剥 batch 后 lm_head 输出 (n,V)——末 token 是 lg[-1]。
- 头块切片：1024=8×128 头主序；块级 null 逐行 norm 匹配。
- 无替换基线走专用 forward_plain（3055 教训）；γ 扰动协议：weight swap+restore sham+扰动下 self-replacement 恒等锚。
- **weights-only 相位（3056）**：零前向离线设计，锚全离线 bit 复核（源封印+上游数组逐量复放+权重统计重算）——清点/库存型问题首选。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→门位易感+内容自由→norm 投影（径向稀释+γ 重整）→γ 读出预对齐（通道分配载荷+方向特异）→**γ 谱清点（3056：词表级 null；γ=白化滤波 corr(γ,colstd W_U)=−0.82；方向特异住在 d_tan×白化基交互几何）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- final norm γ=2560 维强偏斜（max 9.75，均值主导 raw cos）；γ⊙W_U=白化读出基；token 增益谱 g_v=行范数比。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3056）
Ω-P2（3011-3056）：3045-3048 KV 五级阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 h7 通用门槽；3053 内容自由门+场易感；3054 径向稀释+γ 重整；3055 γ 读出预对齐；**3056 γ 谱清点=词表 null+白化滤波（gamma_vocab_null_qwen）**。终局：承载=门位易感，方向=语境组装，读出=γ 白化基（反方差重加权）。

## 下一步
- max=3056，下一个 3057（A 主选 **白化几何验证**——γ⊙W_U 有效谱/条件数 vs W_U、γ~colstd 反比解析拟合、d_tan 白化基对齐增益重演；B h0 对抗成分溯源；C DS7B 复刻全链；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
