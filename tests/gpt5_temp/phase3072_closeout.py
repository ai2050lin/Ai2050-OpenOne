# -*- coding: utf-8 -*-
"""Phase 3072 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3072'
     r'\omega_p69_focal_head_lineage')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
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
assert verdict == 'focal_lineage_full', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a7_diff'] == 0.0 and an['a7_ok'] is True
assert an['a8_diff'] == 0.0 and an['a8_ok'] is True
assert an['a9_diff'] == 0.0 and an['a9_ok'] is True
assert an['b9_diff'] == 0.0 and an['b9_ok'] is True
assert an['b10_diff'] == 0.0 and an['b10_ok'] is True
assert an['b8_diff'] == 0.0 and an['b8_ok'] is True
assert an['aref_diff'] == 0.0 and an['aref_ok'] is True
assert an['a2lens_max'] == 0.062412261962890625
assert an['a2lens_ok'] is True
assert an['b0_diff'] == 0.0 and an['b0_ok'] is True
assert an['b1_diff'] == 0.0 and an['b1_ok'] is True
assert an['b3_ok'] is True
assert an['b4_diff'] == 0.0 and an['b4_ok'] is True
assert an['b5_diff'] == 0.0
assert an['b6_diff'] == 0.0 and an['b6_ok'] is True
assert an['b7a_diff'] == 0.0 and an['b7c_diff'] == 0.0
assert an['b7_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_34'] == 0.1487826048372403
assert st['pa34'] == 0.29685845971107483
assert st['pf34'] == 0.2371114194393158
assert st['d34_cproj']['x'] == 0.0
assert abs(st['d34_cproj']['attn']
           - 639.8104587682893) < 1e-9
assert abs(st['d34_cproj']['mlp']
           + 95.91242909117048) < 1e-9
assert abs(st['d34_cproj']['out']
           - 735.8797832320352) < 1e-9
assert abs(st['d35_cproj']['attn']
           + 14.567961614698053) < 1e-9
assert abs(st['d35_cproj']['mlp']
           + 924.1991208657355) < 1e-9
assert abs(st['d35_cproj']['out']
           + 191.78616219035163) < 1e-9
assert abs(st['lincheck_34']
           - 0.9999932519350134) < 1e-12
pb = st['probe']
assert pb['b9_diff'] == 0.0
assert pb['b10_diff'] == 0.0
assert abs(pb['b10_35_max']
           - 0.76220703125) < 1e-12
assert pb['probe_ok'] is True
ln = st['lineage']
assert ln['focal5'] == [20, 7, 1, 14, 26]
assert abs(ln['med_rel_joint']
           - 0.001736582162660769) < 1e-12
assert abs(ln['top8_min_cos']
           - 0.9999986141965966) < 1e-12
assert abs(ln['all32_med_cos']
           - 0.9999986168914943) < 1e-12
ls = st['listener']
assert abs(ls['sp_m34_absR']
           - 0.5454545454545454) < 1e-12
assert abs(ls['sp_m34_absDah']
           - 0.40725806451612906) < 1e-12
assert abs(ls['sp_m35_absR35']
           + 0.4274193548387097) < 1e-12
m34 = ls['m34_med']
assert abs(m34[20] - 0.988) < 2e-3
assert abs(m34[7] - 0.979) < 2e-3
assert abs(m34[1] - 0.978) < 2e-3
assert abs(m34[14] - 0.990) < 2e-3
assert abs(m34[26] - 0.994) < 2e-3
ov = st['ov']
assert ov['sel_heads'] == [1, 6, 7, 14,
                           20, 24, 26]
assert abs(ov['target_rank_med'][0]
           - 74738.5) < 1e-9
assert res['forwards'] == 111

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3072
           for m in led['measurements']):
    claim = (
        'Omega-P69 (plan 3072 A) - qwen3-4b '
        'bf16 focal-head lineage: WHY are '
        'h20/7/1/14/26 the focal responders? '
        '(12.3s, 111 forwards; anchors: a1 row '
        '34 bit-exact vs 3066 npz; a7 E2 cproj '
        'bit-exact vs 3070 refs; a8 ZH34/ZH35 '
        'bit-exact vs 3071 npz; a9 DAH medians '
        'bit-exact vs 3071 npz; aref PA34/PF34 '
        'bit-exact vs 3069; b9 probe-logits '
        'bit 0.0 (output_attentions=True does '
        'not change numerics); b10 L34 '
        'attention weights bit 0.0 base-vs-inj '
        '(V-clamp leaves Q/K unchanged); b8 '
        'bit 0.0; b0/b1/b4/b6/b7 bit 0.0). '
        'RESULTS: verdict focal_lineage_full. '
        '(1) LINEAGE IDENTITY HOLDS FOR ALL 32 '
        'HEADS: dzH_h = sum_p w_h(p) * '
        'dV_p[g(h)] with g(h)=h//4 (GQA 32Q/'
        '8KV) predicts the per-head response '
        'at cos med 1.0000 (top8 min 0.9999986, '
        'all-32 median 0.9999986) and joint '
        'rel err med 1.7e-3 (gate 0.05) - the '
        'focal heads respond to the injection '
        'by PURE LINEAR READING of the '
        'injected V signature; no nonlinear '
        'term is needed (real-arithmetic '
        'exact, bf16 rounding only). (2) '
        'LISTENER PROFILE: focal5 put 97.8-99.4 '
        'percent of their L34 attention mass '
        'on the injected positions 0..3 (m34 '
        'med h20 0.988 h7 0.979 h1 0.978 h14 '
        '0.990 h26 0.994) vs block min 0.49 '
        '(h28/h29); spearman(m34,|R34|)=0.545 '
        'and spearman(m34,|dAh|)=0.407 - '
        'attention mass explains about half '
        'of the write magnitude (downstream '
        'transmission is the other half, '
        'consistent with 3071 obs-causal '
        'decoupling). L35 contrast: '
        'spearman(m35,|R35|)=-0.427 and focal5 '
        'm35 scattered 0.48-0.92 - L35 heads '
        'are not listeners, matching 3071 L35 '
        'flat. (3) b10_35_max 0.762: the '
        'injection does NOT change L34 Q/K '
        'weights (bit 0.0) but DOES shift L35 '
        'attention weights up to 0.762 - '
        'direct evidence that the L34 response '
        'is transmitted forward as input to '
        'L35 computation. (4) OV CIRCUIT ('
        'descriptive): dlogit_h = W_U @ out_h '
        'top-10 tokens - h1 reads the '
        'theatre-semantic family (theatre/'
        'Theatre/theatrical/theaters/theater), '
        'h7 reads Th/The prefix tokens, h26 '
        'reads <|endoftext|> and code '
        'punctuation, h6 reads digits/'
        'punctuation; per-pair target rank '
        'medians are large (17k-116k of '
        '151936) - single-head OV output '
        'aligns with the TARGET SEMANTIC '
        'FAMILY, not the exact target token; '
        'selection happens in composition/'
        'downstream, consistent with the 3071 '
        'competing-writers picture. '
        'Conclusion: the responder gears are '
        'LINEAR READ-HEADS: attention-weighted '
        'read of the injected V signature '
        '(listener end, exact) -> W_O write-'
        'back of semantic-family directions '
        '(responder end) -> nonlinear '
        'transmission selects (3071). '
        'Head-level response generation is '
        'fully linear under the V-clamp '
        'protocol; nonlinearity lives '
        'downstream.')
    meas = {
        'meas_id': 'meas3072_omega_p69_focal_'
                   'head_lineage',
        'phase': 3072,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 row 34 bit 0.0 vs 3066 '
                   'npz (hard); a7 E2 cproj '
                   'medians bit 0.0 vs 3070 '
                   'refs (hard); a8 ZH34/ZH35 '
                   'bit 0.0 vs 3071 npz (hard); '
                   'a9 DAH34/35 medians bit 0.0 '
                   'vs 3071 npz (hard); aref '
                   'PA34/PF34 bit 0.0 vs 3069 '
                   '(hard); b9 probe logits bit '
                   '0.0 (hard); b10 L34 weights '
                   'bit 0.0 base vs inj (hard; '
                   'L35 shift 0.762 '
                   'descriptive); b8 dzX_35='
                   'dzP_34 bit 0.0 (hard); '
                   'med_c_34 reference assert; '
                   'b0/b1/b4/b6/b7 bit 0.0; b3 '
                   'finite',
        'artifacts': {
            'result': 'phase3072/omega_p69_'
                      'focal_head_lineage/'
                      'result.json',
            'npz': 'phase3072/omega_p69_'
                   'focal_head_lineage/'
                   'omega_p69_focal_head_'
                   'lineage.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16; smoke found '
                'one dtype bug - OUTH34 stored '
                'float64, F.linear against fp32 '
                'W32 crashed; fixed dtype='
                'torch.float32 at the E6 call '
                'and rerun; one fix, clean '
                'authoritative)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 211
    l14['connects'].append({
        'meas_id': 'meas3072_omega_p69_focal_'
                   'head_lineage',
        'phase': 3072,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P69: focal-head '
                        'lineage - the responder '
                        'gears (h20/7/1/14/26) '
                        'are LINEAR READ-HEADS. '
                        'Lineage identity dzH_h = '
                        'sum_p w_h(p)*dV_p[g(h)] '
                        '(GQA g=h//4) holds for '
                        'ALL 32 heads at cos med '
                        '1.0000 (joint rel 1.7e-3, '
                        'gate 0.05): the focal '
                        'response is pure linear '
                        'reading of the injected '
                        'V signature. Listener '
                        'profile: focal5 put '
                        '97.8-99.4 percent of L34 '
                        'attention mass on '
                        'injected positions (block '
                        'min 0.49); spearman(mass,'
                        '|write|)=0.545 - mass '
                        'explains half, '
                        'transmission the rest. '
                        'b10: L34 weights bit 0.0 '
                        '(V-clamp spares Q/K) but '
                        'L35 weights shift 0.762 - '
                        'the response is forwarded '
                        'as L35 input. OV circuit: '
                        'h1 reads the theatre '
                        'semantic family, h7 Th '
                        'prefix, h26 endoftext; '
                        'per-pair target ranks '
                        'large (17k-116k) - '
                        'family-level alignment, '
                        'token selection happens '
                        'downstream. Opens 3073: '
                        'A head interaction matrix '
                        '(pairwise joint swaps, '
                        '3071 non-additivity test); '
                        'B focal-set cross-prompt-'
                        'family stability; C DS7B '
                        'head-level control; D '
                        'lineage under natural '
                        'generation (Q/K drift); '
                        'E neuron identity of '
                        'S_TOP'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
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
if '## Phase 3072:' not in memo:
    sec = u'''## Phase 3072: Ω-P69 焦点头谱系——线性读取恒等式全头成立（focal_lineage_full） [%(created)s]

**判决：`focal_lineage_full`**（qwen3-4b 单模型 bf16 12.3s，111 次前向；12 重 bit 锚全过：**a1：row 34 与 3066 npz diff=0.0；a7：E2 cproj 与 3070 refs diff=0.0；a8：ZH34/ZH35 与 3071 npz diff=0.0；a9：DAH 中位与 3071 npz diff=0.0；aref：PA34/PF34 与 3069 diff=0.0；b9：output_attentions=True 探针 logits diff=0.0；b10：V-clamp 下 L34 attention 权重 base-vs-inj diff=0.0**；b8 bit 0.0；b0/b1/b4/b6/b7 bit 0.0）。调试史如实入册：smoke 崩溃 1 次——OUTH34 以 float64 存储（out_np=outh.double()），E6 的 F.linear 对 fp32 W32 报 dtype mismatch；修复 `dtype=torch.float32` 重编译后 smoke 全绿、权威一次通过。

### 问题与设计（3072 A，3071 菜单主选）
**问题**：为什么是 h20/7/1/14/26 这五个头在应答？3071 打开了输出端（recov），3072 打开输入端与机制本身，三个子问题：(i) **listener**——焦点头把 attention 质量放在注入位置吗？(ii) **lineage**——焦点头的 dzH_h 是否 = 注入 V 签名经 W_V/W_O 的线性像（谱系恒等式）？(iii) **OV**——头的输出直读哪个词表方向？
**协议（预注册）**：V-clamp——只替换 L34 位置 0..3 的 V 矩阵输入为 prefix 值（FRONT=4），残差流与 Q/K 输入不变。由此推出两个机制锚：Q/K 不变 → **b10（L34 attention 权重 bit 0.0）**；probe 协议不改数值 → **b9（oatt=True logits bit 0.0）**。谱系恒等式：在 V-clamp 下 L34 头 h 的响应有精确闭式
$$dzH_h = \\sum_{p=0}^{3} w_h(p)\\cdot \\Delta V_p[g(h)],\\quad g(h)=h/4\\ (\\mathrm{GQA}\\ 32Q/8KV)$$
其中 w_h(p) 是 base/inj 共享的 attention 权重（b10 保证），ΔV_p[g(h)]=注入组 V − base 组 V 从 bank 直接取——**实数意义精确，唯一误差是 bf16 舍入**。预测 pred_h 与观测 dzH_h 比较：联合 rel 门 0.05，focal5 每头 cos med 门 0.99。E6 OV：dlogit_h = W_U @ out_h（out_h 为 3071 同款 per-head o_proj 分解，a9 锚绑死），top-10 token 与目标词排名，只记录不设门。

### 核心结果（重复三遍）
**① 线性谱系全头成立（一）**：med_rel_joint=**1.7e-3**（门 0.05 的 1/29），focal5 每头 cos med 全 **1.0000**，top8 min cos=0.9999986，**all-32 median cos=0.9999986**——**dzH_h = Σ_p w_h(p)·ΔV_p[g(h)] 对全部 32 头精确成立**：焦点头的响应就是"attention 加权读取注入 V 签名"，没有任何隐藏的非线性项。**② 焦点头=倾听者（二）**：focal5 把 L34 attention 质量的 **97.8-99.4 percent** 压在注入位置 0..3（m34 med：h26 0.994、h14 0.990、h20 0.988、h7 0.979、h1 0.978），而全块最低的头只有 0.49（h28/h29）；spearman(m34,|R34|)=**0.545**、spearman(m34,|dAh|)=**0.407**——attention 质量解释约一半写入量，另一半是下游传输（与 3071 obs-causal 脱钩一致）。L35 对照：spearman(m35,|R35|)=**-0.427**（负），focal5 的 m35 分散在 0.48-0.92——**L35 头不是倾听者**，与 3071 L35 全头平坦互证。**③ 传导与应答方向（三遍）**：b10_35_max=**0.762**——注入不改 L34 Q/K 权重（bit 0.0）但把 L35 attention 权重最多推移 0.762：**L34 的响应作为输入直接进入 L35 计算**，这是"响应被向下游传导"的直接证据。OV 电路（描述性）：**h1 直读 theatre 语义族**（theatre/Theatre/theatrical/theaters/theater），h7 读 Th/The 前缀族，h26 读 <|endoftext|> 与代码标点，h6 读数字标点；但逐对目标词排名中位很大（1.7 万-11.6 万 / 151936）——**单头 OV 与目标语义族对齐，与具体目标 token 不重合；token 选择发生在组合与下游**（与 3071 竞争写入图景一致）。

### 机制综合（Ω-P69 拼图）
应答器齿轮的完整机制现在两端都打开了：**倾听端（输入）**——focal5 是注入位置的专用倾听头，attention 质量 97-99 percent 压在位置 0..3；**读取机制**——线性谱系恒等式全头 cos=1.0，dzH 就是 V 签名的 attention 加权线性像；**应答端（输出）**——OV 电路直读语义族方向（h1→theatre 族最干净）；**选择机制**——单头写入是候选方向，token 级选择留给组合与下游非线性（3071 的传输链）。四层分辨率图景完整：3067 神经元级（神经元集×输入方向）→ 3070 组件级（写入×相对幅度）→ 3071 头级（直接写入×传输）→ **3072 头级响应机制（=线性读取，无隐藏非线性）**。

### 硬伤与边界
- 谱系是 **V-clamp 协议下**的精确结论（Q/K 不变由协议保证）；自然生成中 Q/K 会漂移，谱系近似偏离多少未测（3073 D）。
- OV top10 是无 final-norm 的声明性词表投影（dlogit_h = W_U @ out_h），是"头的直接读出方向"而非完整读出。
- E5 spearman 0.545/0.407 中等——attention 质量只解释一半写入量变异；"传输系数"仍是黑箱（头间交互 3073 A）。
- 单 prompt 族（TT）、单注入层 L34、单模型；GQA 组映射 g(h)=h//4 是 Qwen3-4B 结构事实，DS7B 组数不同（3073 C）。
- E6 排名统计基于 24 对的中位，vocab 15 万下排名波动大，只做定性参考。

### 方法论入册
- **V-clamp 推锚法**：从协议结构（只改 V）直接推出 bit 锚（b10 权重不变、b9 探针不改数值），锚不是"额外验证"而是协议正确性的必然推论。
- **谱系恒等式测试范式**：协议保证权重共享（b10）+ bank 直取 ΔV + GQA 组映射 → 预测是闭式的，cos=1.0 是"实数意义精确"的强判决。
- GQA 头-组映射 g(h)=h//4：V 切片按 KV 头分组，8 组×4 Q 头；跨模型复用需查各模型 GQA 配置。
- OUTH34 等 per-head 数组统一存 float32（或调用处显式 dtype），避免 float64 银行与 fp32 权重矩阵的 F.linear dtype mismatch。

### 智能理论洞察（第一性原理）
**"条件化齿轮组"的齿轮本身是线性读头，齿轮箱才是非线性的。** 3072 证明：头的响应生成（读取注入信号）是纯线性算子（cos=1.0，无隐藏项）——一个头 = 一组 attention 权重（决定读哪里）× 一个 V 组切片（决定读什么）× 一个 W_O 写回方向（决定写什么）；头的全部"个性"都在这三个静态参数选择里，动态行为只是它们的线性组合。非线性不在齿轮里，在齿轮之间的传动（3071 的传输链、竞争写入、组合选择）中。这与大脑图景同构：单个神经元的输入-输出传递函数相对简单（加权求和+阈值），智能来自连接结构与回路动力学的组合。**语言能力的极小单元是"线性读头+非线性传动"的分工**：参数级做线性读取，结构级做非线性选择。下一步缝隙：齿轮间的传动矩阵——top-8 头成对联合换回，直接测量头间交互项（3071 已证非可加：top-5 中位和 -0.968 vs 全换回 -0.572，交互项占比 41 percent）；以及自然生成下谱系的稳健性（Q/K 漂移时 cos 掉多少——线性读取在多大范围内是"机制"而非"协议产物"）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3072/omega_p69_focal_head_lineage/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3073 菜单**——A（主选）**头间交互矩阵**：top-8 头成对联合换回（8×7/2=28 对×24 对样本），分解交互项，检验 3071 非可加性（top-5 和 -0.968 vs 全换回 -0.572）的来源。B **焦点头集合跨 prompt 族稳定性**：新 prompt 族复测头级 recov 谱与 listener 剖面。C **DS7B 头级对照**：跨模型检验焦点化+线性谱系图景。D **自然生成谱系稳健性**：Q/K 漂移下 dzH_h = Σ w·ΔV 的近似偏离。E **神经元身份**：S_TOP 的 up/gate 权重结构。"好的，继续"即进 3073 A。
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
if '## 三十四、3072 增补' not in aud:
    add = u'''

---

## 三十四、3072 增补：焦点头谱系——线性读取恒等式（Omega-P69，判决 focal_lineage_full）

1. **线性谱系全头成立**：V-clamp 协议（只替换 L34 位置 0..3 的 V 输入，Q/K 不变由 b10 bit 锚保证）下，dzH_h = sum_p w_h(p)·dV_p[g(h)]（GQA g=h//4）对全部 32 头 cos med 1.0000（top8 min 0.9999986；联合 rel 1.7e-3，门 0.05）——焦点头的响应是注入 V 签名的纯线性读取，无隐藏非线性项。
2. **倾听者剖面**：focal5 把 97.8-99.4 percent 的 L34 attention 质量压在注入位置（全块最低 0.49）；spearman(mass,|write|)=0.545/0.407——质量解释一半写入量，另一半是传输；L35 对照负（-0.427）+ focal5 m35 分散（0.48-0.92）——L35 头不是倾听者。
3. **传导证据**：b10_35_max 0.762——注入不改 L34 权重但推移 L35 attention 权重至 0.762：L34 响应作为输入直接进入 L35 计算。
4. **OV 电路（描述性）**：h1 直读 theatre 语义族、h7 读 Th 前缀、h26 读 endoftext/代码标点；逐对目标词排名中位 1.7 万-11.6 万（vocab 15.2 万）——单头与语义族对齐、与目标 token 不重合，选择在组合/下游。
5. HDMCC 修正：V-clamp 推锚法（协议结构直接推出 bit 锚）；谱系恒等式测试范式（权重共享+bank 直取 ΔV+GQA 组映射=闭式预测）；**齿轮是线性读头、齿轮箱才是非线性**——头的个性全在（读哪里×读什么×写什么）三个静态参数选择里，动态行为是它们的线性组合，非线性在传动与选择。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-21.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3072' not in prev:
    line = ('- Phase 3072 Omega-P69 focal-head '
            'lineage (qwen3-4b bf16 single, '
            '12.3s, 111 forwards): verdict '
            'focal_lineage_full. Twelve bit '
            'anchors exact (a1 vs 3066, a7 vs '
            '3070, a8/a9 vs 3071 npz, aref vs '
            '3069, b9 probe-logits 0.0, b10 L34 '
            'weights 0.0 base-vs-inj, b8, b0/b1/'
            'b4/b6/b7). Lineage identity dzH_h = '
            'sum_p w_h(p)*dV_p[g(h)] (GQA g=h//4) '
            'holds for ALL 32 heads at cos med '
            '1.0000 (joint rel 1.7e-3): focal '
            'response = pure linear reading of '
            'the injected V signature. Listener: '
            'focal5 mass 97.8-99.4 percent on '
            'injected positions; spearman(mass,'
            '|write|)=0.545; L35 contrast '
            'negative (-0.427). b10_35_max 0.762 '
            '= response forwarded into L35 '
            'computation. OV: h1 reads the '
            'theatre semantic family, h7 Th '
            'prefix, h26 endoftext; target ranks '
            'large (17k-116k) - family-level '
            'alignment, selection downstream. '
            'Smoke found one dtype bug (OUTH34 '
            'float64 vs fp32 W32; fixed dtype '
            'at call). Audit 34; ledger 211/L14 '
            '179.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W, encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3072' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃/smoke 推翻均如实登记；verdict 单分支赋值。
4. 统计纪律：阈值预注册；重叠算随机期望；capture 设效应量 floor；SMOKE 判决仅供管线验证。

## 标准锚与精度
- 跨脚本 bit 级锚家族：标量行（a1）、全层多行（a1ext）、集合相等（a2）、因果置换（a3/a4）、跨 phase 参考数（l35ctl/aref）、块链恒等（b8）、hook 互证（a5/a6）、**probe 协议锚（b9：output_attentions 不改 logits；b10：V-clamp 下 L34 权重 bit 0.0）**。
- a2b 型校准断言只在末层读出层有效；中间层用 aref。SMOKE 跳过的锚在 setup_ok 中视为通过。
- **3070**：ATTN 换回=o_proj 输入 H 末位 pre-hook（stateATN[li]）。**3071**：per-head 换回=mask 头切片索引；o_proj 线性无 bias→per-pair 头分解精确。**3072**：V-clamp 推锚法（只改 V→b10 权重 bit 0.0 可预注册）；**谱系恒等式 dzH_h=sum_p w_h(p)·dV_p[g(h)]（GQA g=h//4）cos=1.0 全头成立**；per-head 数组勿存 float64（F.linear 对 fp32 权重会 dtype mismatch）。

## 机制解释审计链（命名前依次检查）
…→3070 L34 压制解剖（压制者=同块 attention）→3071 头级分解（焦点头集合 h20/7/1/14/26，capture8 0.836；头级效应=直接写入×传输）→**3072 焦点头谱系：响应=注入 V 签名的纯线性读取（cos 1.0 全头）；focal5 倾听注入位置 97.8-99.4 percent；OV 电路读语义族（h1→theatre 族）；token 选择在组合/下游**。齿轮=线性读头（读哪里×读什么×写什么），齿轮箱=非线性传动。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→脚本自带 run_log 用 Read 读；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核；改后必编译检查。
- **median 轴陷阱（3071）**：smoke 维度裁剪可使两轴同尺寸、轴错误不可见——语义轴尺寸应与 smoke 裁剪解耦或断言形状。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3072）
Ω-P2（3011-3072）：…3070 suppression_same_block_attn；3071 attn_heads_focal；**3072 focal_lineage_full（应答器齿轮=线性读头；倾听-读取-应答-选择四段机制）**。

## 下一步
- max=3072，下一个 3073（A 主选 **头间交互矩阵**：top-8 成对联合换回分解交互项，检验 3071 非可加性来源；B 焦点头跨 prompt 族稳定性；C DS7B 头级对照；D 自然生成谱系稳健性；E 神经元身份）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3072')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
