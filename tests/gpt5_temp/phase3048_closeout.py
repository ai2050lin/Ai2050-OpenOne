# -*- coding: utf-8 -*-
"""Phase 3048 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3048'
     r'\omega_p45_kvpos_full_replay_qwen')
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
assert verdict == 'kvpos_carries_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a126_chain_diff'] == 0.0
assert an['a127_dup_diff'] == 0.0
assert an['a128_sham_diff'] == 0.0
assert abs(an['a129_key_diff']
           - 3.0517578125e-05) < 1e-12
assert an['a129_v_bit'] == 0.0
assert abs(an['a129_group_diff']
           - 4.76837158203125e-07) < 1e-12
assert an['a130_fail'] == 0
assert abs(an['a131_max_dlg']
           - 3749.0751618558897) < 1e-6
add = st['T2_ADD']
assert abs(add['med_cos']
           - 0.7353763951997888) < 1e-12
assert abs(add['med_frac']
           - 4.720301638986339) < 1e-12
rep = st['T2_REP']
assert abs(rep['med_cos']
           - 0.6066816593084429) < 1e-12
assert abs(rep['med_frac']
           - 3.1957734906639392) < 1e-12
nu = st['null']
assert abs(nu['p_rep']
           - 0.004975124378109453) < 1e-12
assert abs(nu['med_cos']
           - 0.36271359042355694) < 1e-12
assert abs(nu['max_cos']
           - 0.5678279743937804) < 1e-12
assert abs(nu['med_frac']
           - 4.528726202273663) < 1e-12
assert nu['R'] == 200 and nu['seed'] == 9851
t4 = st['T4_SCR']
assert abs(t4['med_kill_frac']
           - 2.2128508596306338) < 1e-12
assert abs(t4['med_kill_cos']
           - 0.07918879803189052) < 1e-12
t3 = st['T3_profile']
assert t3['argmax_layer'] == 33
assert abs(t3['med_frac_at_argmax']
           - 1.1721915952506925) < 1e-12
assert abs(t3['low_half_med']
           - 0.34268752206734165) < 1e-12
assert abs(t3['high_half_med']
           - 0.4987827551068936) < 1e-12
assert abs(st['t_norms']['med']
           - 367.0807715900762) < 1e-9

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3048
           for m in led['measurements']):
    claim = (
        'Omega-P45 (plan 3048 A) - full-position '
        'exact KV replay and decomposition, run3 '
        'fp32 authoritative. RUN1 crashed '
        'pre-anchor on prompt-assembly alignment '
        '(BPE leading-space boundary effect: '
        'first body token is the no-space variant '
        'in the base prompt but the leading-space '
        'variant inside the prefix prompt); '
        'alignment redefined as tail matching + '
        'first-token strip-equality. RUN2 crashed '
        'mid-T4 on a replacement-array length '
        'mismatch AND its null was found '
        'single-pair (accidental reuse of the '
        'last T2 iteration) instead of the '
        'preregistered per-mc 24-pair median - '
        'null implementation corrected to the '
        'preregistered statistic; observed '
        'statistics frozen-seed bit-stable. '
        'DESIGN: post-norm keys captured at the '
        'k_norm output (pre-RoPE), rotated by the '
        'prompt offset (RoPE relative correction) '
        'and written into the base prompt at ALL '
        'body positions, all 36 layers, all 8 kv '
        'heads; values copied verbatim; exactness '
        'verified empirically (a129: replaced '
        'past keys match prefix keys 3.05e-05, '
        'values bit 0.0). RESULTS: (T2_ADD '
        'companion) all-position pre-norm '
        'displacement: med cos 0.7354, frac '
        '4.72. (T2_REP PRIMARY) exact replay: '
        'med cos 0.6067 vs norm-matched random-'
        'replacement null med 0.3627 max 0.5678, '
        'p = 0.00498; frac med 3.20 (overshoot; '
        'null frac med 4.53 - norm-matched '
        'replacement is causally massive, the '
        'discriminating statistic is cos). (T4 '
        'SCR) scrambling the prefix-token KV '
        'moves logits 2.21x the target norm but '
        'NOT along -t (kill_cos 0.079) - '
        'prefix-token KV carries magnitude, not '
        'the specific direction. (T3) single-'
        'layer replacement profile rises toward '
        'late layers, argmax L33 (med frac '
        '1.17), high half 0.50 vs low 0.34. '
        'CONCLUSION: kvpos_carries_qwen - at '
        'full-position scope the KV writes are '
        'for the first time causally load-'
        'bearing AND directionally informative '
        '(cos 0.61 > null max), but the replay '
        'is NOT a faithful reproduction (cos '
        'far from 1, overshoot 3.2x): the '
        'prefix effect is KV-potent but '
        'distributed, not KV-exclusive. Ladder: '
        'single V 3045 null, single K 3046 '
        '0.3pct, joint target 3047 null 7.6pct, '
        'full-position 3048 significant-'
        'partial.')
    meas = {
        'meas_id': 'meas3048_omega_p45_kvpos_'
                   'full_replay_qwen',
        'phase': 3048,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a126 bit 0.0 (KPRE/VPRE '
                   'kv7-slice + DPREK/DPREV vs '
                   'z47); a127 bit 0.0; a128 sham '
                   'bit 0.0; a129 key 3.05e-05 / '
                   'values bit 0.0 / group '
                   '4.77e-07; a130 bit-exact '
                   'fails 0; a131 max dlg 3749',
        'artifacts': {
            'result': 'phase3048/omega_p45_'
                      'kvpos_full_replay_qwen/'
                      'result.json',
            'npz': 'phase3048/omega_p45_'
                   'kvpos_full_replay_qwen/'
                   'omega_p45_kvpos_full_replay_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run3 authoritative (599.7s, fp32); '
                'exact-replay machinery: post-norm '
                'capture + RoPE offset rotation '
                '(k_norm elementwise weight breaks '
                'rotation equivariance, so the '
                'rotation must be applied POST-norm); '
                'null scope corrected to the '
                'preregistered 24-pair median',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 187
    l14['connects'].append({
        'meas_id': 'meas3048_omega_p45_kvpos_'
                   'full_replay_qwen',
        'phase': 3048,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P45: full-position '
                        'exact KV replay (post-norm '
                        '+ RoPE offset rotation, all '
                        '36 layers, all 8 kv heads) '
                        'is the FIRST causally '
                        'significant KV result: '
                        'med cos 0.6067 vs null max '
                        '0.5678 (p 0.00498) with '
                        'overshoot frac 3.20; ADD '
                        'companion cos 0.7354; '
                        'prefix-token scramble moves '
                        'logits 2.2x but NOT along '
                        '-t (kill_cos 0.079) - '
                        'magnitude not direction; '
                        'single-layer profile argmax '
                        'L33; KV is causally potent '
                        'and partially direction-'
                        'informative at full '
                        'position scope, but not a '
                        'faithful carrier (cos far '
                        'from 1); kvpos_carries_qwen'})
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
if '## Phase 3048:' not in memo:
    sec = u'''## Phase 3048: Ω-P45 全位置精确 KV 复放——KV 载荷首次显著承载方向，但非忠实复制（kvpos_carries_qwen） [%(created)s]

**判决：`kvpos_carries_qwen`**（run3 fp32 权威 599.7s；run1 崩于 BPE 前导空格对齐断言——基座句首 token 是无空格变体而前缀句内是带空格变体，改为尾部对齐+首 token 语义校验；run2 崩于 T4 替换数组长度不匹配，且其 null 被发现是单 pair（误用 T2 循环残留变量）而非预注册的逐 mc 24-pair 中位数——null 实现修正回预注册统计量；观测统计量冻结种子逐位稳定）

### 设计（精确复放机械）
三钩点捕获（k_proj pre-norm 链锚 / **k_norm post-norm** 精确复放源 / v_proj），全部 36 层 × 全部体位 × 全部 8 kv 头。关键：**k_norm 逐元素权重破坏旋转等变性**——旋转必须在 post-norm 点施加：替换值 = rot_apply(KPpost[pref, pos+off], off)，模型自身 RoPE(p) 恰好把它映到前缀的 post-RoPE key（a129 实证：替换后 past key vs 前缀 past key diff 3.05e-05，values bit 0.0）。REP 臂主检验 null = R=200 norm 匹配随机替换、逐 mc 24-pair 中位数。

### 核心结果（重复三遍）
**① KV 首次显著承载方向**：全位置精确替换 med cos=**0.6067** vs null med 0.3627 / max 0.5678，**p=0.00498**——四级阶梯（单层 V 3045 → 单层 K 3046 → 全层联合目标位 3047 → 全位置 3048）中第一个显著结果。**② 但非忠实复制**：frac med **3.20**（超调；null frac med 4.53——norm 匹配替换本身因果巨大，判别统计量是 cos），cos 0.61 远离 1；ADD 伴随臂 cos 0.7354 / frac 4.72。**③ 前缀位 KV 承载幅度不承载方向**：SCR 打乱前缀 token 位 KV 使 logits 移动 2.21× 目标范数，但 kill_cos 仅 **0.079**——不是沿 −t 方向的擦除。**④ 层面剖面后段抬升**：单层替换 argmax **L33**（med frac 1.17），高半层 0.50 vs 低半层 0.34。

### 机制链定版
**KV 在全位置范围是因果承重且部分方向可信息的，但效应分布式、非 KV 独占**：精确替换响应 = 显著方向分量（cos 0.61）+ 大量非方向分量（超调）；前缀 logit 位移方向不完全住在任何 KV 写入集合里。结合 3044-3047：前缀效应 = 多通路联合（KV 全位置 + 残流/MLP），KV 贡献真实但非充分。

### 方法论入册
- **精确 KV 复放配方**：post-norm 捕获 + 位置偏移 RoPE 旋转（pre-norm 旋转会被 k_norm 权重解等变化）；实证验证用替换后 past key vs 源 past key 直接比对。
- **BPE 对齐纪律**：跨前缀 token 对齐禁用朴素子序列搜索（首 token 空格变体）；尾部对齐 + 首 token strip 等值。
- **null 实现三查**：null 循环变量严禁复用外层循环残留（run2 单 pair 事故）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3048/omega_p45_kvpos_full_replay_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3049 菜单**——A（主选）**KV 载荷定位**：位置×层贡献矩阵（逐位置替换/擦除子集，定位哪些位置承载 cos 分量 vs 超调分量）；B 阻尼场通道分解；C 跨模型 DS7B 复刻（KV 阶梯协议）；D 跨语言共享子空间。
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
if '## 十、3048 增补' not in aud:
    add = u'''
    
---

## 十、3048 增补：KV 载荷首次显著——全位置精确复放部分承载方向（Omega-P45，判决 kvpos_carries_qwen）

1. **四级 KV 阶梯终章**：单层 V（3045 null）→ 单层 K（3046 0.3pct）→ 全层联合目标位（3047 null 7.6pct）→ **全位置精确复放（3048：cos 0.6067 vs null max 0.5678，p=0.005，首次显著）**——KV 写入在全位置范围因果承重。
2. **对 HDMCC 的最终裁定**："Attention 语法路由"有真实 KV 载荷（方向显著），但**非忠实**（cos 0.61 远离 1、幅度超调 3.2×）；前缀位 KV 打乱移动幅度不移动方向（kill_cos 0.079）——"引力场扭曲指纹竞争"过强，实际是"分布式多通路 + KV 部分方向信息"。
3. **精确复放配方**：post-norm 捕获 + RoPE 位置偏移旋转（qk-norm 逐元素权重破坏 pre-norm 旋转等变性）——任何 KV 复放实验的必需机械；a129 式 past-key 实证验证先于统计。
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
if 'Phase 3048' not in prev:
    line = ('- Phase 3048 Omega-P45 full-position '
            'exact KV replay: verdict '
            'kvpos_carries_qwen (run3 fp32 599.7s; '
            'run1 BPE leading-space alignment crash '
            '-> tail-matching fix; run2 repl length '
            'mismatch crash + null found single-'
            'pair instead of preregistered 24-pair '
            'median -> corrected). RESULTS: exact '
            'post-norm replacement (k_norm capture '
            '+ RoPE offset rotation; a129 past-key '
            'verification 3.05e-05) med cos 0.6067 '
            'vs null med 0.3627/max 0.5678 p '
            '0.00498, frac 3.20 overshoot (null '
            'frac med 4.53); ADD companion cos '
            '0.7354/frac 4.72; prefix-token '
            'scramble kill_frac 2.21 but kill_cos '
            '0.079 (magnitude not direction); T3 '
            'profile argmax L33 1.17, high half '
            '0.50 vs low 0.34. KV causally potent '
            'at full-position scope, partially '
            'direction-informative, not a faithful '
            'carrier; audit addendum 10; ledger '
            '187/L14 155.\n')
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
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%；log 占位符数=实参数。
3. 重跑先删旧产物；负结果与判据作废如实登记；verdict 单分支赋值。
4. **统计量纪律（3044-3048）**：obs 与 null 同量纲同范围（null 循环严禁复用外层残留变量，3048 单 pair 事故）；统计量先量纲自检；logit 级因果读出必须 fp32；改函数返回值后全文件查解包。

## 标准锚与精度
- bit 级仅限同文件链/同精度；跨精度 cos 门 ≥0.999。
- **干预点纪律（3046）**：先探针 norm 管线（qk-norm 在 k_proj 与 RoPE 之间）；自然尺度/场轴定义在干预点所在空间；捕获顺序先改后录。
- **完整性检查纪律（3047）**：禁用 (x+d)−x==d；位级 mod == orig+delta / where(mask,repl,orig) 散射式。
- **精确 KV 复放配方（3048）**：post-norm（k_norm 输出）捕获+替换；RoPE 相对旋转必须在 post-norm 点施加（k_norm 逐元素权重破坏 pre-norm 旋转等变性）；a129 式 past-key 实证验证先于统计。
- **BPE 对齐纪律（3048）**：跨前缀 token 对齐禁用朴素子序列搜索（首 token 空格变体）；尾部对齐+首 token strip 等值。

## 统计判据纪律
- 判据可达性先检；退化行先剔；构造匹配置换；池内标签置换 MC。

## 机制解释审计链（命名前依次检查）
…→KV 多分量→谱水平复核→场方差分解→相关可测层≠因果作用层→**KV 四级阶梯**：单层 V（3045）→单层 K（3046）→全层联合目标位（3047 null 7.6pct）→**全位置精确复放（3048：cos 0.607 显著、frac 3.2 超调、非忠实）**→前缀位 KV 承载幅度不承载方向（kill_cos 0.079）→效应=多通路联合分布式。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出形状 (1,s,8,128)；RoPE NeoX 配对 (i,i+64)、inv_freq 缓冲、attention_scaling==1；V 无 norm/RoPE。
- output_attentions 张量 detach().cpu()；fp32 用于 logit 级因果测量。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3048）
Ω-P2（3011-3048）：3011 门控=L3 KV；3018-3019 抑制场；3020 注入特异 944×；3021-3024 联盟中继/承重；3028 剂量凸增长；3031-3036 异质性伪影/头集中/指纹 logistic/非正交；3037 KV 多分量；3038/3039 协议分层；3040-3041 情景分量+谱复核；3042-3043 风格场（4.5×共享、体主效应 64pct、轴共性 0.647、库外迁移）；3044 作废；3045 fp32 终审 V 无特权+bf16 噪声地板；3046 qk-norm 发现、K 场真实但单层复放 0.3pct；3047 全层联合目标位 null；**3048 全位置精确复放首次显著（cos 0.607 p=0.005、超调 3.2×、非忠实；前缀位打乱 kill_cos 0.079）**。

## 下一步
- max=3048，下一个 3049（A 主选 **KV 载荷定位**——位置×层贡献矩阵，定位 cos 分量 vs 超调分量的位置归属；B 阻尼场通道分解；C 跨模型 DS7B 复刻；D 跨语言共享子空间）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
