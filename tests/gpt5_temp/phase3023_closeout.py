# -*- coding: utf-8 -*-
"""Phase 3023 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3023'
     r'\omega_p2q_relay_causal_ablation_qwen')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
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
verdict = res['final_verdict']
assert verdict == 'relay_causal_toxic_void', verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_content'] == 11
assert t2['med_js_erase'] == 0.003332
assert t2['med_js_coal'] == 0.008028
assert t2['med_js_bot'] == 0.002496
assert t2['med_js_abl_only'] == 0.002352
assert t2['med_js_rnd'] == 0.003436
assert t2['collapse_med'] == -1.4726
assert t2['ratio_coal_med'] == 2.4726
assert t2['ratio_rnd_med'] == 0.9355
assert t2['ratio_bot_med'] == 0.9917
assert t2['n_plus'] == 3
assert t2['p_binom'] == 0.9673
assert t2['a28_erase_chain_diff'] == 0.0
t2b = res['T2b']
assert t2b['med_abl_rel'] == 0.555
assert t2b['toxic_ratio'] == 0.7059
assert t2b['rnd_seed'] == 5919
t2c = res['T2c']
assert t2c['med_js_erase_content'] == 0.000118
assert t2c['med_js_coal_content'] == 0.003546
assert t2c['collapse_content'] == -28.2673
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a27_3022'] is True
assert res['anchors']['a28_erase_chain_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3023
           for m in led['measurements']):
    claim = (
        'Omega-P2q (plan v5 P2) - causal ablation '
        'leg of the 3022 relay coalition: exact '
        'down_proj output subtraction (output -= '
        'h[:,:,idx] @ W_down[:,idx].T, hook-input h, '
        'causal, bf16) of the per-tag top-32 '
        'coalitions reconstructed from the sealed '
        '3022 npz; arms per logic tag = ERASE / COAL '
        '/ RND(12) / BOT / ABL_ONLY; content tags '
        'ERASE_C / COAL_C.  Verdict '
        'relay_causal_toxic_void (frozen map: '
        'med js_abl_only 0.002352 > 0.5*med '
        'js_erase 0.001666, toxic_ratio 0.706).  '
        'RESULTS: (i) the coalition has a HUGE '
        'baseline function - these 32 neurons '
        '(0.33 pct) carry med 55.5 pct of the L3 '
        'MLP output norm at the step-2 position, '
        'so zero-ablation removes their normal '
        'output together with the relay and '
        'injects a perturbation COMPARABLE TO THE '
        'SIGNAL; (ii) the toxicity is '
        'COALITION-SPECIFIC, not generic - random-'
        '32 (ratio 0.936) and bottom-32 (0.992) '
        'ablations leave the erase JS untouched '
        'while the coalition ablation RAISES it '
        '(ratio_coal 2.473, collapse negative in '
        '9/11 tags, p_binom 0.967 - sign test '
        'uninterpretable under toxicity); (iii) '
        'content positions: erase JS 1.2e-04 '
        'rises to 3.5e-03 (30x) under coalition '
        'ablation - the coalition baseline output '
        'is consumed at ALL positions, it is not '
        'a logic-specific static circuit; (iv) '
        'chain identity a28: ERASE arm js == '
        'sealed 3022 js bit-level (max diff 0.0).  '
        'INTERPRETATION: attribution mass (3022: '
        '82 pct) != causal isolability - the '
        'relay coalition is the ATTRIBUTION main '
        'channel but is causally REDUNDANT/'
        'REDISTRIBUTIVE (removal reroutes rather '
        'than removes, consistent with the '
        'ablation = direct + competitive '
        'rebalancing audit item and contrasting '
        'with the genuinely necessary h12 carrier '
        'of 2980); zero-ablation collapse is '
        'UNMEASURABLE at this coalition size.  '
        'REGISTERED DEFECT: the reachability '
        'precondition "ablation side-effect << '
        'signal" was not pre-checked - the '
        'ablation-only control (third recurrence '
        'of the lesson) is what saved the '
        'interpretation.  NEXT: baseline-'
        'restoration ablation (restore coalition '
        'output to its no-erase values in the '
        'erase chain) isolates the erase-induced '
        'change without the toxicity.')
    meas = {
        'meas_id': 'meas3023_omega_p2q_relay_'
                   'causal_ablation_qwen',
        'phase': 3023,
        'claim': claim,
        'verdict': verdict,
        'anchors': '28/28 (a0-a24 as 3022; a26 3021; '
                   'a27 3022 integrity; a28 erase '
                   'chain vs 3022 js bit-level 0.0)',
        'artifacts': {
            'result': 'phase3023/omega_p2q_'
                      'relay_causal_ablation_qwen/'
                      'result.json',
            'npz': 'phase3023/omega_p2q_'
                   'relay_causal_ablation_qwen/'
                   'omega_p2q_relay_causal_'
                   'ablation_qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'runs 1-3 crashed (0-d directions '
                'load; content-arm tmap lookups), run4 '
                'authoritative; negative result '
                'registered: attribution mass != '
                'causal isolability; toxicity is '
                'coalition-specific (random/bottom '
                'clean).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 162
    l14['connects'].append({
        'meas_id': 'meas3023_omega_p2q_relay_'
                   'causal_ablation_qwen',
        'phase': 3023,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2q: zero-ablation of '
                        'the 3022 relay coalition is '
                        'TOXIC and coalition-'
                        'SPECIFIC (32 neurons = 0.33 '
                        'pct carry 55.5 pct of L3 MLP '
                        'output norm; ablating raises '
                        'erase JS 2.5x, random/bottom '
                        'clean; content JS 30x) - '
                        'attribution mass != causal '
                        'isolability; relay is '
                        'redistributive, need '
                        'baseline-restoration '
                        'design'})
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
if '## Phase 3023:' not in memo:
    sec = u'''## Phase 3023: Ω-P2q 中继联盟因果消融——零消融有毒且集合特异：归因质量 ≠ 因果可分离性 [%(created)s]

**判决：`relay_causal_toxic_void`**（run4 权威，169.3s，锚 28/28：a1 轴身份 0.0 位级、a10 rel 0.0、a13 diff 0.0、a27 3022 完整性、**a28 擦除链 vs 3022 js 位级 diff 0.0**；correction_note 空。run1-3 崩溃：directions 0-d 加载、content 臂 tmap 查找 ×2）

### 设计（3022 机器 verbatim + 精确 down_proj 输出消融）
从 sealed 3022 npz 的 s_relay 重构每 tag 的 top-32 正性联盟；干预 = **down_proj forward hook 内精确减去选中神经元贡献：output -= h[:,:,idx] @ W_down[:,idx]·T**（h 取 hook 自身输入 → 对真实擦除链因果，bf16；prefill 期 idx=None 自动空操作，仅 step-2 生效=与 3022 捕获口径一致）。每 logic tag 五臂：ERASE / COAL / RND(12 组) / BOT(|s| 最小 32) / **ABL_ONLY（只消融不擦除=毒性对照）**；content 臂用同 prompt 首个 logic tag 的联盟。collapse = 1 − js_coal/js_erase；配对符号检验 vs RND。

### 核心结果（重复三遍——负结果但高信息量）
**① 零消融有毒且集合特异**：联盟消融使擦除 JS **不降反升 2.47×**（med co 0.0080 vs er 0.0033；collapse 9/11 tag 为负），而随机 32（ratio 0.936）与 bottom-32（0.992）完全无效——"消融 32 个神经元"本身无毒，**毒在这批特定神经元**；**② 联盟基线功能巨大**：这 0.33pct 的神经元承载 **L3 MLP 输出范数的 55.5pct**（med abl_rel）——零消融把"正常功能+中继"一起移除，注入与信号同量级的扰动（toxic_ratio 0.706 > 0.5 门触发判决）；**③ content 位 30× 副作用**：擦除 JS 1.2e-04 → 消融后 3.5e-03——联盟的基线输出在**所有位置**被大量消费，它不是 logic 特异的静态电路；**④ 链身份**：ERASE 臂与 sealed 3022 js 位级一致（a28 diff 0.0），随机/bottom 对照干净，测量管线本身无懈可击。

### 判决与统计缺陷（如实登记）
判决按冻结映射 = toxic_void（med js_abl_only 0.002352 > 0.5×med js_erase 0.001666）。**缺陷：判据可达性先检第三次复发**——未预检"消融副作用 << 信号"这一前提，collapse 分数在毒性污染下不可解释（p_binom 0.967 无意义）。毒性对照（ABL_ONLY）是救命的：没有它会把"消融噪声"误读为"无因果作用"。

### 结论：归因质量 ≠ 因果可分离性
3022 的"专属正性联盟（82pct 归因质量）"必须限定为**归因主通道**，而非**必要载体**——移除主通道后误差被重路由而非消失（与"消融差分 = 直接 + 竞争重平衡"审计链一致；对比 2980 的 h12 = 真必要载体，消融即消失 97pct）。**零消融在这个联盟规模上原理性不可用**，下一步需用**基线恢复消融**（把联盟输出替换为其无擦除值，而非清零——只隔离擦除诱导的变化）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3023/omega_p2q_relay_causal_ablation_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3024 = A（主选）**基线恢复消融**——擦除链中把联盟 32 神经元的 down_proj 输出替换为其无擦除链的对应值（patch 而非 zero），隔离擦除诱导的中继变化、消除毒性，测 JS 塌缩；B 只恢复 e4-对齐分量（把 3022 的 s 分解做成干预）；C 联盟基线功能刻画（55pct 范数在算什么：下游消费头定位）；D L31 次峰定位。
''' % {'created': created,
           'script8': exe['script_sha256_8'],
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

# ---------- workspace log ----------
wl = WLOG_DIR + r'\2026-09-20.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3023' not in prev:
    line = ('- Phase 3023 Omega-P2q: verdict '
            'relay_causal_toxic_void (runs 1-3 '
            'crashed, run4 authoritative, a28 erase '
            'chain vs 3022 bit-level 0.0); NEGATIVE '
            'RESULT: zero-ablation of the 3022 '
            'relay coalition is TOXIC and '
            'coalition-SPECIFIC - 32 neurons (0.33 '
            'pct) carry 55.5 pct of L3 MLP output '
            'norm, ablating raises erase JS 2.47x '
            '(collapse negative 9/11), random-32 '
            '(0.936) / bottom-32 (0.992) clean, '
            'content JS 30x side-effect = coalition '
            'baseline consumed at all positions; '
            'attribution mass (82 pct) != causal '
            'isolability; relay redistributive '
            '(vs 2980 h12 truly necessary); defect: '
            'reachability pre-check (ablation '
            'side-effect << signal) missed, third '
            'recurrence, ABL_ONLY control saved '
            'interpretation; next = baseline-'
            'restoration ablation; ledger '
            '162/L14 %d.\n'
            % len(l14['connects']))
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
- 脚本 tests\\glm5\\phase{N}_*.py；产物 ...\\phase{N}\\{arm}\\；临时 gpt5_temp\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14；hash=去 ledger_sha256_8 后 dumps(sort_keys) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结（PREREG/锚/判决）→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present_files→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）。
3. 重跑先删旧 execution/result/npz；负结果如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格（{key} 不替换，2971）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；大内积阈 1e-8；链身份锚多点（‖e4‖=3.6017、p_m=.686、js 序列，均 0.0）。

## 统计判据纪律
- 判据可达性先检（永真/不可达禁用）：置换不变 null p≡1（3021）；随机子集 Jaccard null 中位=0→以 p 值为主（3022）；**消融类设计必须预检 ablation-only 副作用 << 信号（毒性门），否则 collapse 不可解释（3023 第三次复发）**；归因质量 ≠ 因果可分离性——零消融大基线功能集合=重路由非移除，用基线恢复（patch to no-erase value）替代清零；margin n≳40；quasi-post-hoc 标注；maxT；镜像 −dirs 必配；跨产物对照先查类目对齐。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+竞争重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；args 空用 kwargs；单样本保 batch 维；真残差流=decoder-layer pre-hook；恒等门分母与噪声同尺度；npz 存 dict→0-d 对象数组，读回须 .item()（3023 run1）；hook 修改输出用返回值，prefill 期 idx=None 空操作=只 step-2 生效；CUDA 张量与权重同 device；np.savez 后字典键不展开。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm/grep shim 损坏→用 Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁；replace 未命中→先 Grep；改后必编译检查。
- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接。
- numpy 标量入 json 转 int()/float()；GQA：KV 缓存 8 头。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3023）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/GLM4 86 vs qwen 63pct/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3023）：3011 **门控=L3 KV**（次峰 L31）；3013 情景码 LOO .397；3014 剂量非单调=K 路由；3015 K 消费=领先 g7（.390）；3016 放大分布式；3017 吸收混合；3018 **抵消主导**（credit 2.51 nats，MLP 承载）；3019 抵消带=通用抑制场（top32 6.7pct）；3020 **读出特异=注入特异**（L4 即时 JS 944×→稀释 28.3×）；3021 **注入=MLP 中继主导 69pct**（头 31pct GQA 限 group7）；3022 **中继=正性专属稀疏联盟**（top32/9728 承载 82pct，跨 tag Jaccard .362 p≈0，vs 3019 抑制场 Jaccard 0.0 且反号=独立回路）；3023 **零消融有毒且集合特异**（联盟 32 神经元=55.5pct MLP 输出范数，消融反升擦除 JS 2.47×，随机/bottom 干净，content 30× 副作用）——**归因质量≠因果可分离性，中继=重路由型**（对比 h12 真必要）。门控三级=种子→L3 中继→负性均衡场；核心：null 重编码全层分布式涌现；头级重要性=关系属性。

## 下一步
- max=3023，下一个 3024（A 主选 **基线恢复消融**——擦除链中把联盟输出 patch 到无擦除值，隔离中继变化消除毒性，测 JS 塌缩；B 只恢复 e4 对齐分量；C 联盟基线功能刻画；D L31 次峰）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
