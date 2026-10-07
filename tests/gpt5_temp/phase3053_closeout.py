# -*- coding: utf-8 -*-
"""Phase 3053 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3053'
     r'\omega_p50_gate_source_qwen')
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
assert verdict == 'gate_mixed_qwen', verdict
st = res['stats']
an = st['anchors']
assert an['anchor_core_ok'] is True
assert an['a116_seals_ok'] is True
assert an['a155_recapture_diff'] == 0.0
assert an['a156a_kdiag_diff'] == 0.0
assert an['a156b_vdiag_diff'] == 0.0
assert an['a156c_jdiag_diff'] == 0.0
assert an['a157_sham_diff'] == 0.0
assert an['a158_fail'] == 0
assert an['a158_checked'] == 96
assert abs(an['a159_max_dlg']
           - 864.0003649283559) < 1e-6
assert an['a160_pre_id_max'] == 0.0
assert an['a161_postn_dev'] < 1e-3
t2a = st['T2a_k_cross']
assert abs(t2a['med_diag']
           - 0.6871243793038855) < 1e-12
assert abs(t2a['med_A_off']
           - 0.674142444522372) < 1e-12
assert abs(t2a['med_D_off']
           - -0.011555579068341981) < 1e-12
assert abs(t2a['p_perm']
           - 0.8490754622688655) < 1e-12
assert abs(t2a['g1']
           - 0.9811068633677862) < 1e-12
t6 = st['T6_controls']
assert abs(t6['med_exact_h0']
           - -0.24919460797697768) < 1e-12
assert abs(t6['med_exact_h3']
           - 0.4194891522570098) < 1e-12
assert abs(t6['med_null_h7']
           - 0.6608560952615515) < 1e-12
assert abs(t6['med_null_h0']
           - 0.48995892650749406) < 1e-12
assert abs(t6['med_null_h3']
           - 0.3097709175435721) < 1e-12
assert abs(t6['max_null_h7']
           - 0.7075643906612268) < 1e-12
assert abs(t6['p_h7']
           - 0.36318407960199006) < 1e-12
assert abs(t6['p_h0'] - 1.0) < 1e-12
assert abs(t6['p_h3']
           - 0.27860696517412936) < 1e-12
t3 = st['T3_stages']
assert t3['med_cos_pre'] == 0.0
assert abs(t3['med_cos_att']
           - 0.43050748336450706) < 1e-12
assert abs(t3['med_cos_mlp']
           - -0.1372528281075465) < 1e-12
assert abs(t3['med_cos_mlpn']
           - 0.6804954615601925) < 1e-12
t4 = st['T4_attn_transfer']['rows']
assert abs(t4[0]['corr_dmass_AK']
           - -0.742455862181155) < 1e-12
assert abs(t4[2]['corr_dmass_AK']
           - -0.756851677630877) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3053
           for m in led['measurements']):
    claim = (
        'Omega-P50 (plan 3053 A) - gate source '
        'dissection, run3 fp32 authoritative '
        '(932.9s; run1 393.6s non-authoritative '
        'block-vs-fullrow anchor mismatch, run2 '
        '931.3s non-authoritative scalar-index '
        'bug on a156a/b, both registered in '
        'corrections; corrected anchors '
        'verified bit 0.0 offline before run3). '
        'RESULTS: (1) CONTENT-FREE GATE within '
        'h7: the h7-block K-only restricted null '
        'is NOT significant - exact obs 0.6871 '
        'vs random-K null med 0.6609 max 0.7076, '
        'p = 0.363: norm-matched RANDOM K in the '
        'h7 block opens the gate as well as the '
        'exact fields. (2) NOT h7-EXCLUSIVE: '
        'random K in h0 gives null med 0.4900 '
        '(max 0.6847), in h3 0.3098 (max '
        '0.6020) - ANY head block at L35 x '
        'FRONT responds to K perturbation with '
        'a graded magnitude (h7 > h0 > h3); '
        'mn0 = 0.74 x mn7 is neither >= 0.8 nor '
        '< 0.5 -> verdict gate_mixed_qwen. '
        '(3) Exact-field counter-example: h0 '
        'exact K-only diag is NEGATIVE -0.2492 '
        'while its random null is +0.49 (p 1.0) '
        '- same signature as the 3051 V-only '
        '-0.3409: the exact source fields carry '
        'an opposing component that random '
        'perturbation lacks. (4) Assembly '
        'locus (T3, a160 pre-identity 0.0 / '
        'a161 post-norm dev 1e-7): pre-norm '
        'stage alignment att 0.4305, L35 MLP '
        'output delta ANTI-aligns -0.1373, '
        'post-final-norm 0.6805 - the attention '
        'writes a partially aligned shift, the '
        'L35 MLP reshapes it, and the final '
        'RMSNorm projection restores the '
        'alignment. (5) T4: corr(Delta-mass g7, '
        'A_K) = -0.74..-0.76 - attention mass '
        'REDISTRIBUTION anti-predicts the '
        'effect (pressure-valve signature). '
        '(6) T2a cross-pair transfer replicates '
        '3052 genericity: g1 = 0.981, p_perm '
        '0.849; T2b V-block cross ~ 0 (A_off '
        '0.088).')
    meas = {
        'meas_id': 'meas3053_omega_p50_gate_'
                   'source_qwen',
        'phase': 3053,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a155 re-capture bit 0.0 vs '
                   'z48 + TT 0.0; a156a/b full-row '
                   'K/V diag diff 0.0 vs z51 '
                   'COS_K/COS_V; a156c joint h7 '
                   'diag diff 0.0 vs z51 COS_H[7]; '
                   'a157 sham bit 0.0; a158 '
                   'integrity fails 0 (96 checked); '
                   'a159 max dlg 864.0; a160 '
                   'stage-pre identity 0.0; a161 '
                   'post-norm dev 1.0e-7',
        'artifacts': {
            'result': 'phase3053/omega_p50_'
                      'gate_source_qwen/'
                      'result.json',
            'npz': 'phase3053/omega_p50_'
                   'gate_source_qwen/'
                   'omega_p50_gate_source_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run3 authoritative (932.9s, fp32); '
                'block nulls h7/h0/h3 R=200 each '
                '(seeds 9935/9941/9942); paired '
                'sign-flip permutation R=2000; run1 '
                'anchor-mismatch and run2 scalar-'
                'index bug registered in the prereg '
                'corrections',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 192
    l14['connects'].append({
        'meas_id': 'meas3053_omega_p50_gate_'
                   'source_qwen',
        'phase': 3053,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P50: gate source - '
                        'h7 K-block effect is '
                        'CONTENT-FREE (exact 0.6871 '
                        'vs random null 0.6609, p '
                        '0.363) and NOT h7-exclusive '
                        '(random K in h0/h3 gives '
                        '0.49/0.31) - L35 x FRONT K '
                        'field is a graded '
                        'susceptibility zone, head '
                        'identity only scales the '
                        'response; exact h0 fields '
                        'NEGATIVE -0.249 (opposing '
                        'component, same signature '
                        'as V-only -0.34); assembly: '
                        'attention writes 0.43, L35 '
                        'MLP reshapes (-0.14), final '
                        'RMSNorm restores 0.68; '
                        'attention mass '
                        'redistribution '
                        'anti-predicts (corr -0.74) '
                        '(gate_mixed_qwen)'})
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
if '## Phase 3053:' not in memo:
    sec = u'''## Phase 3053: Ω-P50 门控源解剖——内容自由门 + 场易感分级 + norm 投影组装（gate_mixed_qwen） [%(created)s]

**判决：`gate_mixed_qwen`**（run3 fp32 权威 932.9s，锚核心全过）。run1（393.6s，块-全行锚错位）与 run2（931.3s，a156a/b 标量索引 bug：COS_K51[h] 取到 (24,) 向量第 7 对标量）均非权威、如实入册；run2 离线 npz 验证修正后 diff 0.0 才进 run3。

### 设计
K-block-only / V-block-only 跨对迁移矩阵（h7，24×24，A/B 双目标）+ 全行 K/V 对角链锚（a156a/b）+ **T6 判别组合**：h0/h3 精确 K-only 对角（门位特异性）+ h7/h0/h3 随机 K 块限制性 null（R=200 each，norm 匹配）+ T3 三级 stage 差分（a160 pre 恒等 / a161 post-norm 一致性）+ T4 注意力质量迁移相关。

### 核心结果（重复三遍）
**① 门效应内容自由（h7 内）**：h7 块 K-only obs=**0.6871** vs 随机 K null med=**0.6609** / max 0.7076，**p=0.363 不显著**——norm 匹配随机 K 与精确场同样开门，场内容无关。**② 门位非 h7 独占（幅度分级）**：随机 K 在 h0 给 null med **0.4900**、h3 **0.3098**（h7 0.6609）——**L35×FRONT K 场是分级易感区（h7>h0>h3），头身份只是幅度系数**；mn0=0.74×mn7 非 high 非 low → gate_mixed。**③ 精确场对抗成分**：h0 精确 K-only=**−0.2492（负）**而随机 +0.49（p=1.0）——与 3051 V-only −0.34 同签名：精确源场携带对抗性成分，随机扰动反而无此负担。**④ 组装定位（T3）**：pre-norm 三级对齐 att=**0.4305** / L35 MLP 差分=**−0.1373（反向）** / post-norm=**0.6805**（a161 dev 1e-7）——注意力写入部分对齐、L35 MLP 重塑、最终 RMSNorm 投影恢复并放大对齐。**⑤ T4 反相关**：corr(Δmass_g7, A_K)=**−0.74..−0.76**——g7 组 FRONT 注意力质量减少越多对齐越强（压强阀签名）。**⑥ 通用性复制**：T2a g1=**0.981**、p_perm 0.849；T2b V 块跨对 ≈0（A_off 0.088）。

### 机制链定版
**KV 承载链终局图景：L35×FRONT 的 K 场是一个分级易感门区——任何头块的 K 扰动都开门（内容无关），方向由目的语境+读出侧组装（注意写入→MLP 重塑→norm 投影放大），头维只是幅度梯度**。3051 的"头 7/6 承载"须重述为"h7/h6 精确场替换效应最强，但效应强度来自门位易感度而非场内容"；与 3027（读出=上下文属性）、3052（头身份=门）合并：**承载=门位易感，方向=语境组装，读出=独立自由度**。

### 方法论入册
- **锚形状核对纪律（3053 run1/run2 教训，两次 run 报废）**：跨相位锚比较前必须 assert 上游数组形状——z51 的 COS_K/COS_V 是 (24,) 逐对向量无头维，误加 [h] 即取标量。
- **内容自由判别组合（3053）**：块内限制性 null（随机 vs 精确，同头同行同范数匹配）判内容自由性 + 跨头随机对照（h0/h3）判门位特异性——两null缺一不可，否则"通用"与"易感"混淆。
- **stage 捕获校验（3053）**：a160（d_pre≡0 恒等）+ a161（post-norm 差分与 logits cos 一致）是 stage 管线必配锚；pre-norm stage cos 受最终 RMSNorm 投影混淆，只能作定性对比。
- **块级分接头切片**：K 块 null 随机化须逐行 norm 匹配该头块精确场（NK = ‖rotK[L35,:,h,:]‖ 按行）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3053/omega_p50_gate_source_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3054 菜单**——A（主选）**门区的 norm 投影机制**：RMSNorm Jacobian 为何把 0.43 的 pre-norm 对齐放大到 0.68——门开后 logits 移位的解析分解（径向/切向分量 + W_U 有效方向）；B h0 精确场负成分溯源（与 V-only −0.34 同源检验）；C 跨模型 DS7B 复刻全链（KV 阶梯+门区）；D 门区 2D 易感图（体位×层位扰动响应面）。
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
if '## 十五、3053 增补' not in aud:
    add = u'''
    
---

## 十五、3053 增补：门控源——内容自由门、场易感分级、norm 投影组装（Omega-P50，判决 gate_mixed_qwen）

1. **门是内容自由的**：h7 块 K-only 精确替换 0.6871 与随机 K null 0.6609 不可区分（p 0.363）——承载"内容"不在场值里；结合 3052（跨对可互换），KV 承载 = 门位扰动的结构效应，方向全部来自目的语境+读出侧。
2. **门位非独占**：随机 K 在 h0/h3 也产生 0.49/0.31 对齐——L35×FRONT K 场整体易感，头维只是幅度梯度；HDMCC 若画"门控节点"应画成**场属性**（位置×层的易感度面）而非头身份。
3. **组装机制**：attention 写入 0.43 → L35 MLP 反向重塑 −0.14 → RMSNorm 投影恢复 0.68——读出侧 norm 非线性是方向成形的最后一环；"深层 KV 是方向写入器"（3050）在 pre-norm 层面只成立一半，完整叙述是"KV 开门 + MLP 重塑 + norm 放大"。
4. **精确场负成分再证**：h0 精确 K −0.249 vs 随机 +0.49——源场（pref 位 K）携带对抗性成分与 3051 V-only −0.34 同签名，指向"源场含读出侧不知道如何使用的成分"，待 B 线溯源。
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
if 'Phase 3053' not in prev:
    line = ('- Phase 3053 Omega-P50 gate source: '
            'verdict gate_mixed_qwen (run3 fp32 '
            '932.9s; run1 anchor-mismatch + run2 '
            'scalar-index bug non-authoritative, '
            'registered). RESULTS: h7 K-block '
            'effect CONTENT-FREE (exact 0.6871 vs '
            'random null 0.6609, p 0.363) and NOT '
            'h7-exclusive (random K h0 0.49 / h3 '
            '0.31) - L35 x FRONT K field = graded '
            'susceptibility zone, head identity '
            'only scales; exact h0 fields NEGATIVE '
            '-0.249 (opposing component, same as '
            'V-only -0.34); assembly: attention '
            '0.43 -> L35 MLP reshape -0.14 -> '
            'RMSNorm restores 0.68; attention '
            'mass anti-predicts (corr -0.74); '
            'cross transfer replicates 3052 (g1 '
            '0.981, p_perm 0.849). Audit addendum '
            '15; ledger 192/L14 160.\n')
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
4. 统计纪律：obs/null 同量纲同范围；null 限同一行集（3050）；loo 全谱（3051）；跨对迁移 A/B+配对符号翻转（3052）；**块内随机 null + 跨头随机对照判内容自由 vs 场易感（3053）**。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- **锚形状核对纪律（3053，两次 run 报废教训）**：跨相位锚比较前 assert 上游数组形状——z51 COS_K/COS_V 是 (24,) 逐对向量无头维，误加 [h] 即取标量。
- 捕获库跨相位复用：上游 npz+抽样重捕获 bit 锚+派生统计量逐对复现锚。
- 二维输出索引（3050）：hook out[0] 剥 batch 后 lm_head 输出 (n,V)——末 token 是 lg[-1]；layer 返回裸 tensor；output_hidden_states 不可信。
- 头块切片（3051）：1024=8×128 头主序；块级 null 随机化逐行 norm 匹配该头精确场。

## 机制解释审计链（命名前依次检查）
…→KV 五级阶梯（3045-3048）→载荷定位（3049）→层定位（3050 末层 L35）→通道定位（3051 K 场+h7/h6）→头身份（3052 h7 通用门槽）→**门控源（3053：内容自由 p 0.363、场易感分级 h7>h0>h3、组装=注意 0.43→MLP 重塑 −0.14→norm 放大 0.68、注意力质量反预测）**。

## 工程规范（Qwen3-4B）
- hidden=2560；GQA 32q/8kv；k_norm 输出 (1,s,8,128)；v_proj (1,s,1024)；RoPE NeoX、attention_scaling==1；fp32 logit 级测量。
- 头 h OV 写出：M_h=Σ₄ o_proj 列组 @v̄；3022 联盟写入=s_relay·down_proj[:,j] 重构；stage 捕获配 a160 pre 恒等 + a161 post-norm 一致性锚。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3053）
Ω-P2（3011-3053）：3011 门控=L3 KV；3018-3024 联盟中继；3045-3048 KV 五级阶梯；3049 体位均匀；3050 末层 L35；3051 K 场+h7/h6；3052 h7 通用门槽；**3053 门控源：内容自由+场易感分级+norm 投影组装（gate_mixed_qwen）**。终局：承载=门位易感，方向=语境组装，读出=独立自由度。

## 下一步
- max=3053，下一个 3054（A 主选 **门区 norm 投影机制**——RMSNorm Jacobian 把 0.43 放大到 0.68 的解析分解；B h0 精确场负成分溯源；C 跨模型 DS7B 复刻全链；D 门区 2D 易感图）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
