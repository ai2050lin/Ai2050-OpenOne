# -*- coding: utf-8 -*-
"""Phase 3055 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3055'
     r'\omega_p52_gamma_prealign_qwen')
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
assert verdict == 'gamma_prealigned_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a169_recapture_diff'] == 0.0
assert an['a172_tt_diff'] == 0.0
assert an['a170_chain_diff'] == 0.0
assert an['a171_restore_diff'] == 0.0
assert an['a173_fail'] == 0
assert an['a173_checked'] == 2
t2 = st['T2_preimage']
assert abs(t2['med_cos_dtan_gamma_wt']
           - 0.11143683246599181) < 1e-12
assert abs(t2['med_cos_dtan_wt']
           - 0.027109528168621812) < 1e-12
assert abs(t2['gain_t']
           - 0.08432730429737001) < 1e-12
assert abs(t2['med_gain_u']
           - -0.0001584192034687691) < 1e-12
assert abs(t2['pct95_gain_u']
           - 0.05238319642821795) < 1e-12
assert t2['target_specific'] is True
t3 = st['T3_gamma_perturb']
assert abs(t3['med_cos_orig']
           - 0.6804954382830863) < 1e-12
assert abs(t3['med_cos_shuf']
           - 0.20314887613828342) < 1e-12
assert abs(t3['med_cos_ones']
           - 0.1899329367435097) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3055
           for m in led['measurements']):
    claim = (
        'Omega-P52 (plan 3055 A) - gamma readout '
        'pre-alignment source and universality, '
        'run2 fp32 authoritative (51.6s; run1 '
        '55s crashed pre-verdict at the a173 '
        'baseline: zeros-repl + all-False mask '
        'hits the hook assignment branch with '
        '0-row boolean index vs 6-row repl, '
        'fixed by a dedicated forward_plain '
        'path, registered in corrections). '
        'RESULTS: (1) CHANNEL ASSIGNMENT '
        'CAUSALLY CARRIES the readout '
        'pre-alignment: permuting the gamma '
        'channels (keeping the amplitude '
        'profile) drops the alignment vs the '
        'ORIGINAL target from 0.6805 to 0.2031, '
        'setting gamma to ones drops it to '
        '0.1899 - both far below the 0.3402 '
        'threshold -> verdict '
        'gamma_prealigned_qwen; shuffled and '
        'ones are EQUALLY low, so the amplitude '
        'profile shape carries nothing on its '
        'own - WHICH channels are amplified is '
        'the payload. (2) TARGET-SPECIFIC '
        'GAIN: residual-preimage alignment '
        'cos(d_tan, gamma*w_t) = 0.1114 vs '
        'cos(d_tan, w_t) = 0.0271 (gain 0.0843) '
        'while 200 norm-matched random logit '
        'directions give median gain -0.0002 '
        '(p95 0.0524) - the gamma gain is '
        'concentrated on behaviorally relevant '
        'target directions, not a global '
        'readout effect. (3) gamma stats: min '
        '-0.022, max 9.75, mean 2.76, std 0.455 '
        '- a strongly skewed per-channel scale. '
        '(4) a173 self-replacement bit identity '
        'holds under each perturbed gamma; '
        'a171 gamma-restore sham bit 0.0.')
    meas = {
        'meas_id': 'meas3055_omega_p52_gamma_'
                   'prealign_qwen',
        'phase': 3055,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a169 re-capture bit 0.0 vs '
                   'z48 + a172 TT 0.0; a170 '
                   'COS_orig diff 0.0 vs z51 '
                   'COS_H[7]; a171 gamma-restore '
                   'sham bit 0.0; a173 '
                   'self-replacement bit identity '
                   'under both perturbed gammas '
                   '(2 checked, 0 fail)',
        'artifacts': {
            'result': 'phase3055/omega_p52_'
                      'gamma_prealign_qwen/'
                      'result.json',
            'npz': 'phase3055/omega_p52_'
                   'gamma_prealign_qwen/'
                   'omega_p52_gamma_prealign_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run2 authoritative (51.6s, fp32); '
                'gamma perturbation via '
                'model.model.norm.weight swap with '
                'restore sham; option A (original '
                'TT target fixed across arms); run1 '
                'crash registered in corrections',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 194
    l14['connects'].append({
        'meas_id': 'meas3055_omega_p52_gamma_'
                   'prealign_qwen',
        'phase': 3055,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P52: gamma readout '
                        'pre-alignment - permuting '
                        'gamma channels drops the '
                        'alignment 0.6805 -> 0.2031 '
                        'and ones -> 0.1899 (both '
                        'below 0.3402, shuffled = '
                        'ones so the amplitude '
                        'profile alone carries '
                        'nothing): the channel '
                        'assignment IS the payload; '
                        'gain is target-specific '
                        '(t-gain 0.0843 vs random-'
                        'direction median -0.0002, '
                        'p95 0.0524) - training '
                        'pre-aligned the final-norm '
                        'readout for behaviorally '
                        'relevant logit directions '
                        '(gamma_prealigned_qwen)'})
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
if '## Phase 3055:' not in memo:
    sec = u'''## Phase 3055: Ω-P52 γ 读出预对齐——通道分配因果承重 + 方向特异增益（gamma_prealigned_qwen） [%(created)s]

**判决：`gamma_prealigned_qwen`**（run2 fp32 权威 51.6s，锚核心全过）。run1（55s）判决前崩溃于 a173 基线构造（zeros-repl + 全 False mask 误入 hook 赋值分支，0 行布尔索引 vs 6 行 repl 广播失败），改专用 forward_plain 路径，corrections 如实入册。链锚：a169 重捕获 / a170 COS_orig vs z51 COS_H[7] bit 0.0 / a172 TT bit 0.0 / a171 γ 还原 sham bit 0.0 / a173 两个扰动 γ 下 self-replacement bit 恒等。

### 设计
**T3 γ 因果扰动（主检验）**：三臂（orig / shuffled 置换 γ 通道保持幅度谱 / ones），各 24 对 canonical h7 joint 替换，r_new 对**原 TT 方向**的 cos（选项 A 固定目标方向——掉幅度量的是读出预对齐的损失而非目标改变）。**T2 读出预像对齐（普适性）**：w_t=W_U^T t，逐对 cos(d_tan, γ⊙w_t) vs cos(d_tan, w_t)；随机对照 U=200 norm 匹配 logit 方向（seed 9951）。

### 核心结果（重复三遍）
**① γ 通道分配因果承重**：置换 γ 通道（幅度谱不变）后对齐 **0.6805→0.2031**，γ→1 后 **0.1899**——双双远低于阈值 0.3402；且 **shuf ≈ ones**：幅度谱形状自身不承载任何东西，**"哪些通道被放大"才是载荷**。剩余 ~0.20 是无预对齐读出的基线可读性。**② 增益方向特异**：残流预像对齐 cos(d_tan, γ⊙w_t)=**0.1114** vs cos(d_tan, w_t)=**0.0271**（gain **0.0843**）；200 个随机 logit 方向 med gain **−0.0002**（p95 0.0524）——**γ 增益集中于行为相关的目标方向，非全局读出效应**。**③ γ 剖面强偏斜**：min −0.022 / max 9.75 / mean 2.76 / std 0.455。

### 机制链定版
**读出预对齐 = 训练进 final-norm γ 通道分配的静态任务几何**：写入侧（KV 门位易感 + 语境组装）与读出侧（γ 预对齐基）是两个独立训练出的自由度；"深层 KV 写方向 + γ 读基"构成完整写入-读出闭环。与 P1-P8 合并：模型不仅学了"往哪写"（对比基+投票表），还学了"怎么读"（γ 通道分配放大行为相关读出方向）。

### 方法论入册
- **γ 扰动协议（3055）**：weight swap + restore sham（a171 bit 复核）+ 扰动 γ 下 self-replacement 恒等锚（a173）；选项 A（固定原 TT 目标）判读掉幅语义。
- **无替换基线必须走专用路径（3055 run1 教训）**：zeros-repl + 全 False mask 会误入 hook 赋值分支（布尔索引 0 行 vs repl 行数广播失败）——forward_plain 显式清空替换状态。
- **随机方向对照判普适性（3055）**：norm 匹配 + 同一 d_tan、只换读出方向 u——增益差直接归因于方向特异性。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3055/omega_p52_gamma_prealign_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3056 菜单**——A（主选）**γ 预对齐的方向谱清点**：γ⊙W_U 组合读出基的 top 方向谱（哪些词表/语义方向被预对齐）+ 与 24 目标方向、2802 投票谱方向的重叠 + 跨层 norm γ 剖面对比；B h0 对抗成分溯源（V-only −0.34 与 h0 切向对抗同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区+γ）；D 门区 2D 易感图。"好的，继续"即进 3056 A。
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
if '## 十七、3055 增补' not in aud:
    add = u'''
    
---

## 十七、3055 增补：γ 读出预对齐——通道分配承重、方向特异（Omega-P52，判决 gamma_prealigned_qwen）

1. **γ 通道分配是载荷**：置换通道（保持幅度谱）0.6805→0.2031，γ→1 →0.1899，shuf ≈ ones——幅度谱形状无承载，通道分配即预对齐内容。
2. **增益方向特异**：目标方向预像对齐增益 0.0843，随机方向 med −0.0002（p95 0.0524）——训练把 final-norm γ 调成对行为相关读出方向的放大器，非全局标度。
3. HDMCC 修正升级：读出侧应画为"γ 预对齐读出基（方向特异）"——读出几何是可学习参数的一部分，与写入门位（KV 场易感）构成写入-读出双自由度。
4. 图谱关联含义：词族信息 → KV 写入（门位+语境组装）→ γ 预对齐读出（方向特异放大）——两图谱的接口在读出端也是学习产物。
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
if 'Phase 3055' not in prev:
    line = ('- Phase 3055 Omega-P52 gamma readout '
            'pre-alignment: verdict '
            'gamma_prealigned_qwen (run2 fp32 '
            '51.6s; run1 crashed pre-verdict on '
            'a173 zeros-repl+all-False-mask hook '
            'broadcast, fixed via forward_plain, '
            'registered). RESULTS: gamma channel '
            'assignment CAUSALLY carries - permute '
            '0.6805->0.2031, ones ->0.1899 (shuf = '
            'ones: amplitude profile carries '
            'nothing, WHICH channels is the '
            'payload); gain TARGET-SPECIFIC '
            '(t-gain 0.0843 vs random med -0.0002, '
            'p95 0.0524); gamma skew min -0.022 '
            'max 9.75. Training pre-aligned the '
            'final-norm readout for behaviorally '
            'relevant directions. Audit addendum '
            '17; ledger 194/L14 162.\n')
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
4. 统计纪律：obs/null 同量纲同范围；null 限同一行集；loo 全谱；跨对迁移 A/B+配对符号翻转；块内随机 null+跨头对照；近似质量判据可替代 null（阈值预注册）；**随机方向对照判增益特异性（3055：同 d_tan 换读出方向 u）**。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- 锚形状核对纪律；捕获库跨相位复用（上游 npz+抽样重捕获 bit 锚+派生量逐对复现锚）。
- 二维输出索引（3050）：hook out[0] 剥 batch 后 lm_head 输出 (n,V)——末 token 是 lg[-1]。
- 头块切片：1024=8×128 头主序；块级 null 逐行 norm 匹配。
- **无替换基线走专用 forward_plain（3055 run1 教训）**：zeros-repl+全 False mask 误入 hook 赋值分支（0 行布尔索引广播失败）。
- **γ 扰动协议（3055）**：weight swap+restore sham bit 锚+扰动 γ 下 self-replacement 恒等锚；选项 A 固定原 TT 目标。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯→体位均匀→末层 L35→K 场+h7/h6→h7 通用门槽→内容自由门+场易感分级→norm 投影（径向稀释+γ 重整）→**γ 读出预对齐（3055：通道分配因果承重 shuf/ones 双掉 0.68→0.20；增益方向特异 t-gain 0.0843 vs 随机 −0.0002）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX；fp32 logit 级测量。
- final norm γ=2560 维强偏斜（max 9.75）；γ⊙W_U=读出预对齐基；stage 捕获配 pre 恒等+postn 一致性锚。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3055）
Ω-P2（3011-3055）：3045-3048 KV 五级阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 h7 通用门槽；3053 内容自由门+场易感；3054 径向稀释+γ 重整；**3055 γ 读出预对齐=通道分配载荷+方向特异（gamma_prealigned_qwen）**。终局：承载=门位易感，方向=语境组装，读出=γ 预对齐静态基。

## 下一步
- max=3055，下一个 3056（A 主选 **γ 预对齐方向谱清点**——γ⊙W_U top 方向谱、与 24 目标/2802 投票谱方向重叠、跨层 γ 剖面；B h0 对抗成分溯源；C 跨模型 DS7B 复刻；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
