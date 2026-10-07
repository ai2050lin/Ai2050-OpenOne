# -*- coding: utf-8 -*-
"""Phase 3074 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3074'
     r'\omega_p71_capacity_law')
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
assert verdict == 'capacity_law_hill', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
for nm in ('a1', 'a8', 'a9', 'a10', 'a11', 'b8',
           'b0', 'b1', 'b4', 'b6'):
    assert an[nm + '_diff'] == 0.0 \
        and an[nm + '_ok'] is True, nm
assert an['aref_diff'] == 0.0 and an['aref_ok'] is True
assert an['a2lens_max'] == 0.062412261962890625
assert an['a2lens_ok'] is True
assert an['b3_ok'] is True
assert an['b5_diff'] == 0.0
assert an['b7a_diff'] == 0.0 and an['b7c_diff'] == 0.0
assert an['b7_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_34'] == 0.1487826048372403
assert st['pa34'] == 0.29685845971107483
assert st['pf34'] == 0.2371114194393158
assert st['top8'] == [20, 7, 1, 14, 26, 0, 2, 24]
assert st['r_all32'] == -0.5717521069904176
assert st['a_u8'] == 0.5322981028358011
assert st['a_best_subset'] == {
    'mask': 255, 'amp': 0.5322981028358011}
assert st['x_u8'] == 1.219138699856578
assert st['x_all32'] == 1.9917079240985371
assert st['n_sub_checked'] == 16472
assert st['n_viol'] == 4359
assert abs(st['viol_rate']
           - 0.26463088878096164) < 1e-12
assert st['submodular_ok'] is False
fits = st['fits']
assert set(fits) == {'add', 'exp', 'log', 'hill'}
assert fits['add']['err_u8'] == 0.6868405970207769
assert fits['add']['err_all32'] == 1.4199558171081197
assert fits['exp']['err_u8'] == 0.03984606949404468
assert fits['exp']['err_all32'] \
    == 0.05364586169328667
assert fits['exp']['pass_u8'] is True
assert fits['exp']['pass_all32'] is False
assert fits['log']['err_u8'] == 0.1325037080319692
assert fits['hill']['p_le2'] == [
    0.7142857142857143, 0.45409444455802084,
    1.1857142857142857]
assert fits['hill']['pred_u8'] == 0.5452333231103856
assert fits['hill']['err_u8'] == 0.012935220274584491
assert fits['hill']['pred_all32'] \
    == 0.6088086522043087
assert fits['hill']['err_all32'] \
    == 0.03705654521389112
assert fits['hill']['pass_u8'] is True
assert fits['hill']['pass_all32'] is True
assert res['forwards'] == 6207

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3074
           for m in led['measurements']):
    claim = (
        'Omega-P71 (plan 3074 A) - qwen3-4b '
        'bf16 capacity law: full measurement '
        'of the saturation set function R(S) '
        '- ALL 255 non-empty subsets of the '
        'focal top-8 swapped jointly (24 '
        'pairs each, 6120 forwards), budget '
        'x(S) = sum of single-head '
        'amplitudes m_h = |r1(h)|, four '
        'candidates fit on |S|<=2 (36 pts) '
        'and extrapolated to the 8-head and '
        '32-head joints (double gate 0.05). '
        '(359.8s, 6207 forwards; anchors: '
        'a1 row 34 bit-exact vs 3066 npz; '
        'a8/a9 ZH + DAH medians bit-exact '
        'vs 3071 npz; a10 family = all 38 '
        '3073 conditions are subsets of the '
        'sweep, bit 0.0 vs 3073/3071 refs - '
        'cross-phase set-function '
        'reproducibility perfect; a11 full '
        '32-head swap bit 0.0 vs 3071 gA '
        'recov -0.5717521069904176; aref '
        'bit vs 3069; b8/b0/b1/b4/b6/b7 bit '
        '0.0). RESULTS: verdict '
        'capacity_law_hill. (1) LINEAR '
        'BASELINE FAILS HARD: add predicts '
        'A(S8)=1.219 vs measured 0.532 (err '
        '0.687, 2.3x overshoot) and '
        'A(all32)=1.992 vs 0.572 (err 1.420) '
        '- saturation is real and large. '
        '(2) HILL WINS THE DOUBLE GATE: '
        'A(x) = a x^g/(b^g+x^g) with '
        'a=0.7143, b=0.4541, g=1.1857 '
        '(le2 fit, sse 0.0105); extrapolates '
        'S8 to 0.5452 (err 0.0129 = 26 '
        'percent of gate) and all-32 to '
        '0.6088 (err 0.0371) - the only '
        'candidate passing both.  exp is '
        'partial (0.0398 pass / 0.0536 fail); '
        'log fails (0.133/0.276).  hill is '
        'also the best descriptive fit on '
        'all 255 points (sse 0.659).  (3) '
        'NOT STRICTLY SUBMODULAR: 4359 of '
        '16472 inequalities A(S+x)-A(S) <= '
        'A(T+x)-A(T) violated at tol 0.02 '
        '(26.5 percent, worst excess 0.337) '
        '- R(S) carries supermodular '
        'components in set space; the Hill '
        'law is a valid 1-D budget '
        'projection, not a complete set '
        'theory.  (4) No magic subset: the '
        'largest amplitude among all 255 is '
        'the full set itself (mask 255, '
        '0.5323) - monotone with no '
        'small-set winner.  (5) Asymptote '
        'a=0.714 > measured all-32 0.572: '
        'the fitted channel capacity '
        'exceeds the full swap-back - the '
        'TT channel saturates below its '
        'nominal capacity.  Conclusion: the '
        'head-level causal effect follows a '
        'CAPACITY LAW in the write-budget '
        'variable - collective saturation '
        'with gamma ~= 1.19 (near-linear '
        'onset, early saturation), the '
        'first quantitative law of the '
        'shared TT channel.')
    meas = {
        'meas_id': 'meas3074_omega_p71_capacity_'
                   'law',
        'phase': 3074,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 row 34 bit 0.0 vs 3066 '
                   'npz (hard); a8 ZH34/ZH35 '
                   'bit 0.0 vs 3071 npz (hard); '
                   'a9 DAH34/35 medians bit 0.0 '
                   'vs 3071 npz (hard); a10 '
                   'family = 8 singles bit 0.0 '
                   'vs 3071 r34[top8], 28 pairs '
                   '+ triplet mask7 + top-8 '
                   'mask255 bit 0.0 vs 3073 '
                   'r2/r_t3/r_u8 (all 38 3073 '
                   'conditions are subsets of '
                   'the sweep; cross-phase '
                   'set-function '
                   'reproducibility); a11 full '
                   '32-head swap bit 0.0 vs '
                   '3071 gA recov '
                   '-0.5717521069904176 (hard); '
                   'aref PA34/PF34 bit 0.0 vs '
                   '3069 (hard); b8 dzX_35='
                   'dzP_34 bit 0.0 (hard); '
                   'med_c_34 reference assert; '
                   'b0/b1/b4/b6/b7 bit 0.0; b3 '
                   'finite',
        'artifacts': {
            'result': 'phase3074/omega_p71_'
                      'capacity_law/'
                      'result.json',
            'npz': 'phase3074/omega_p71_'
                   'capacity_law/'
                   'omega_p71_capacity_'
                   'law.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16), rerun once '
                'before sealing to fix PREREG '
                'wording of the E4 inequality '
                'count (16472 = T a proper '
                'subset incl. empty; both runs '
                'bit-identical results).  '
                'Deterministic: two full '
                'authoritative runs agree '
                'bit-exact on every reported '
                'statistic',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 213
    l14['connects'].append({
        'meas_id': 'meas3074_omega_p71_capacity_'
                   'law',
        'phase': 3074,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P71: capacity '
                        'law of the shared TT '
                        'channel - full R(S) '
                        'measurement over all '
                        '255 subsets of the '
                        'focal top-8 (6120 '
                        'forwards).  Linear '
                        'additive baseline '
                        'overshoots 2.3x (err '
                        '0.687 at S8, 1.420 at '
                        'all-32) - saturation '
                        'is real.  Hill form '
                        'A(x)=a x^g/(b^g+x^g), '
                        'budget x = sum of '
                        'single-head amplitudes: '
                        'a=0.7143 b=0.4541 '
                        'g=1.1857 (fit on |S|<=2 '
                        'only) extrapolates both '
                        'the 8-head (err 0.0129) '
                        'and 32-head (err 0.0371) '
                        'joints within the 0.05 '
                        'double gate - only '
                        'candidate to pass; exp '
                        'partial, log/add fail.  '
                        'R(S) is NOT strictly '
                        'submodular (26.5 percent '
                        'of 16472 inequalities '
                        'violated at tol 0.02) - '
                        'supermodular components '
                        'exist in set space; the '
                        'Hill law is a 1-D budget '
                        'projection.  No magic '
                        'subset (max amplitude = '
                        'full set).  Fitted '
                        'asymptote 0.714 > full '
                        'swap-back 0.572 - '
                        'channel saturates below '
                        'nominal capacity.  '
                        'Model: write-budget -> '
                        'recovery conversion is a '
                        'quantitative capacity '
                        'law (gamma ~= 1.19).  '
                        'Opens 3075: A '
                        'supermodular-structure '
                        'localization from the '
                        '16472 marginal '
                        'inequalities; B write-'
                        'direction collinearity '
                        '(3072 npz, no forwards); '
                        'C focal-set cross-prompt '
                        'stability; D DS7B '
                        'control; E neuron '
                        'identity'})
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
if '## Phase 3074:' not in memo:
    sec = u'''## Phase 3074: Ω-P71 饱和曲线 R(S) 全测量与容量定律——Hill 形式双门外推成立（capacity_law_hill） [%(created)s]

**判决：`capacity_law_hill`**（qwen3-4b 单模型 bf16 359.8s，**6207 次前向**；12 重 bit 锚全过：**a1：row 34 与 3066 npz diff=0.0；a8/a9：ZH 与 DAH 中位与 3071 npz diff=0.0；a10 家族：3073 全部 38 个条件（8 单头 vs 3071 r34[top8]、28 对 + 三元组 mask7 + top-8 mask255 vs 3073 r2/r_t3/r_u8）作为 255 子集扫描的子集逐一 bit 复现 diff=0.0——跨 phase 集合函数复现性完美；a11：全 32 头换回 recov 与 3071 gA -0.5717521069904176 diff=0.0**；aref diff=0.0；b8/b0/b1/b4/b6/b7 bit 0.0）。设计史如实入册：权威运行跑过一次后（结果与本跑 bit 一致）发现 PREREG 中 E4 不等式计数文字有误（实际枚举 T 为真子集含空集 = 16472 条），修正措辞后重跑 seal——两次完整权威运行的每一个报告统计量 bit 级一致（确定性复现）。另：编译前自查修复 4 处 bug（L35 head_decomp 形状不匹配补捕获 ZA35、AMP_keys_255 未定义死块、E4 死循环、smoke verdict 占位）。

### 问题与设计（3074 A，3073 菜单主选）
**问题**：3073 证明头级因果效应是换回头集合 S 的集合函数 R(S)、边际递减强，且二阶展开不完备——**R 的函数形式是什么？能否用一个低维定律预测任意规模的联合效应？** **设计**：（E1）24 对 repV 注入 ladder（协议 3065/.../3073 全同，锚 a1/aref/med_c_34 参考）；（E3）**top-8 焦点集的全部 255 个非空子集**（bitmask over TOP8=[20,7,1,14,26,0,2,24]）联合换回，每子集 24 对，R(S)=median cos−med_c_34；嵌入 a10 家族锚（3073 全部条件是本扫描的子集）；（a11）全 32 头换回 vs 3071 gA；（E4）次模性全查：A(S+x)−A(S) ≤ A(T+x)−A(T)，T 为真子集含空集，共 16472 条不等式，tol 0.02（记录不设门）；（E5）容量定律拟合：预算参数化 x(S)=Σ_{h∈S} m_h，m_h=|r1(h)|（单头振幅，3071/3073 bit 同源），四候选 A(x)=x（线性）/ a(1−e^(−x/b))（指数）/ a·ln(1+x/b)（对数）/ a·x^γ/(b^γ+x^γ)（Hill），网格+两轮细化最小二乘；**fit1=全 255 点（描述性）；fit2=仅 |S|≤2 的 36 点（8 单头+28 对）外推 S8（实测 0.5323）与 all-32（实测 0.5718），双门 |err|<0.05**（≈全换回恢复量的 9 percent）。预注册判决树 13 分支；submodular 率与拟合表记录不设门。

### 核心结果（重复三遍）
**① 线性基线惨败——饱和确凿（一）**：线性模型外推 S8=1.219 vs 实测 0.532（误差 0.687，**超出 2.3 倍**）、all-32=1.992 vs 0.572（误差 1.420，超出 3.5 倍）——**联合效应绝不按单头效应求和**。**② Hill 形式双门唯一通过（二）**：A(x)=a·x^γ/(b^γ+x^γ)，le2 拟合（仅 36 个 ≤2 头点）**a=0.7143、b=0.4541、γ=1.1857**（sse 0.0105），外推 **S8 误差 0.0129（门的 26 percent）、all-32 误差 0.0371**——唯一双门候选；exp 部分通过（0.0398 过/0.0536 败），log 失败（0.133/0.276）；Hill 同时是全 255 点的描述性最优（sse 0.659 < exp 0.667 < log 0.929 << add 28.19）。**③ R(S) 非严格次模（三）**：16472 条次模不等式 **4359 条违反（26.5 percent）**，worst excess 0.337——集合空间存在超模成分（特定头对协同超过边际递减的普遍趋势）；**Hill 定律是预算变量上的一维有效投影，不是完备集合理论**。**④ 无魔法子集**：255 个子集中最大振幅恰为全集 mask 255（0.5323）——曲线单调、无小集合反超。**⑤ 拟合渐近 a=0.714 > 实测全换回 0.572**：TT 通道的名义容量高于全换回所能实现的恢复量——**通道在名义容量之下就已饱和**（下游非线性截断）。

### 数学公式
- R(S) = median_k cos(Δlogit_k(S), TT_k) − med_c_34，A(S) = −R(S) ≥ 0；
- 预算参数化：x(S) = Σ_{h∈S} m_h，m_h = |R1(h)|（3071 单头 recov 中位振幅）；x_u8 = 1.219138699856578（Σ top8）、x_all32 = 1.9917079240985371（Σ 全 32 头）；
- **Hill 容量定律：A(x) = a·x^γ / (b^γ + x^γ)**，a=0.7142857142857143（渐近容量）、b=0.45409444455802084（半饱和预算：x=b 时恢复 a/2）、γ=1.1857142857142857（Hill 系数，>1 为 sigmoid 型启动、≈1 退化为 Michaelis-Menten）；
- 对照：exp A=a(1−e^(−x/b))（a=0.6429, b=0.5523）、log A=a·ln(1+x/b)（a=0.4643, b=0.3826）、add A=x；
- 双门：|pred(S8) − 0.5322981028358011| < 0.05 ∧ |pred(all32) − 0.5717521069904176| < 0.05。

### 硬伤与边界
- **γ≈1.19 与 exp 的区分度有限**：exp 只差 0.0036 过第二个门（0.0536 vs 0.05）——"Hill 严格优于 exp"在当前数据强度下是弱结论；需要更大预算范围或更多中间规模点才能定形。
- **预算参数化的信息损失**：x(S)=Σm_h 把"哪些头"压缩成标量——26.5 percent 的次模违反正是被压缩掉的结构（3073 的 spearman 0.822 也说明结构不是零，只是次要）；Hill 定律预测的是"预算换恢复"，不预测"协同超模对在哪"。
- fit2 外推跨 4-64 倍预算范围，参数受模型形式支配；a=0.714 的"渐近"是拟合形式的外推上界，不是实测（实测最大 0.572）。
- 单 prompt 族、单模型、单注入层（L34）；Hill 参数对 prompt 族的稳定性未测（3075 C）。
- 拟合用手写网格+两轮细化（a 35 档/b 45 档/γ 43 档），参数精度受网格分辨率限制。
- 次模违反率 26.5 percent 的显著性未做置换检验（tol 0.02 ~24 对中位 SE，0.337 的 worst excess 远超噪声，但 0.02-0.05 区间的违反可能部分是噪声）。

### 方法论入册
- **全子集扫描 + 嵌入式锚**：255 子集不只测曲线，还把上一 phase 的全部 38 个条件作为子集逐一 bit 复现（a10 家族）——扫描本身成为跨 phase 因果一致性检验，无额外前向成本。
- **预算参数化范式**：集合函数 → 1D 守恒量（写入预算）→ 低维容量定律；这是从 3073"集合函数不可加"到本 phase"可定量预测"的关键一跃。
- 双门外推判决：只允许用 ≤2 头数据拟合、预言 8 头与 32 头——防"用数据拟合数据"；门 0.05 ≈ 全恢复量的 9 percent。
- 权威运行 bit 级复跑一致性（PREREG 文字修正后重跑，全部统计量两跑一致）——确定性管线的最终检查。

### 智能理论洞察（第一性原理）
**"条件化齿轮组"的传动轴有了容量定律。** 3072 说每个头是线性读头，3073 说它们写共享 TT 通道、因果效应是集合函数，3074 说这个集合函数在"写入预算"变量上服从 **Hill 容量定律 A(x)=a·x^γ/(b^γ+x^γ)，γ≈1.19、b≈0.45、a≈0.71**——单头写入幅度（可测）通过一个固定非线性映射决定任意联合的恢复份额（可预测）。这是机制链第一个**带参数的定量预测定律**：给定 8 个头的单头效应，不跑任何联合前向就能预言 8 头联合到 1.3 percent 精度、32 头联合到 3.7 percent。从智能理论看：**有限参数支撑无限组合的机制之一可能正是这种"共线写入 + 饱和容量"结构**——大量并行写入器（头）竞争一条低维输出通道，系统层面的响应由守恒预算与容量非线性决定，而不是逐器件求和；这与热力学把微观态约化为宏观量的逻辑同构，也解释了鲁棒性（单个头的损失被通道容量缓冲）。γ>1 的 sigmoid 启动提示通道有轻微"协同门槛"（弱写入几乎无效，达到 b≈0.45 预算后迅速进入饱和段）——**弱头的"翻转"行为（3073 ⑤）在容量定律图景下是自然的：它们处在门槛以下的噪声区**。未解释的核心：超模成分的来源（哪些头对协同、为什么）与通道的物理身份（下游 MLP/L35 的哪段非线性施加饱和）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3074/omega_p71_capacity_law/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3075 菜单**——A（主选）**超模结构定位**：从本 phase PAIRS4（16472 条边际数据）反解违反次模的头对/三元组集合，定位"协同写入"结构并与 GQA/OV 方向对照（免前向，npz 已存）。B **写入方向共线性**：focal5 的 out_h/dAh 方向互相余弦，直接测"共线写入"假设（3072 npz 免前向）。C **焦点头集合跨 prompt 族稳定性**：新 prompt 族复测单头谱 + Hill 参数。D **DS7B 头级对照**：跨模型检验焦点化+容量定律图景。E **神经元身份**：S_TOP 的 up/gate 权重结构。"好的，继续"即进 3075 A。
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
if '## 三十六、3074 增补' not in aud:
    add = u'''
---
## 三十六、3074 增补：饱和曲线全测量与容量定律（Omega-P71，判决 capacity_law_hill）
1. **容量定律**：255 个非空子集全测量（6120 前向）+ 预算参数化 x(S)=Σ 单头振幅；Hill 形式 A(x)=a·x^γ/(b^γ+x^γ)（a=0.714, b=0.454, γ=1.186，仅用 ≤2 头数据拟合）外推 8 头（err 0.0129）与 32 头（err 0.0371）联合双门通过——唯一候选；线性基线超出实测 2.3-3.5 倍，饱和确凿。
2. **非严格次模**：16472 条不等式 26.5 percent 违反（worst 0.337）——集合空间有超模成分；Hill 定律是一维预算投影，不是完备集合理论；255 子集中最大振幅=全集（无魔法子集）。
3. **容量低于名义**：拟合渐近 a=0.714 > 全换回复现 0.572——TT 通道在名义容量之下饱和（下游非线性截断）。
4. HDMCC 修正：**预算参数化范式**（集合函数→守恒量→低维定律）与**双门外推判决**（≤2 头拟合、预言 8/32 头）入册；嵌入式锚（上一 phase 全部条件作为扫描子集 bit 复现）成为零成本跨 phase 一致性检验。
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
if 'Phase 3074' not in prev:
    line = ('- Phase 3074 Omega-P71 capacity '
            'law (qwen3-4b bf16 single, '
            '359.8s, 6207 forwards): verdict '
            'capacity_law_hill. Twelve bit '
            'anchors exact (a1/a8/a9/a10 '
            'family/a11/aref/b8/b0/b1/b4/b6/'
            'b7); a10 family = all 38 3073 '
            'conditions reproduced bit-exact '
            'as subsets of the 255-subset '
            'sweep; a11 full 32-head = 3071 '
            'gA. Linear baseline overshoots '
            '2.3x (err 0.687 at S8, 1.420 at '
            'all-32). Hill A(x)=a x^g/(b^g+'
            'x^g) a=0.7143 b=0.4541 g=1.1857 '
            '(fit on |S|<=2, 36 pts) '
            'extrapolates both joints within '
            'the 0.05 double gate (err '
            '0.0129/0.0371) - only candidate; '
            'exp partial, log/add fail. NOT '
            'strictly submodular: 26.5 '
            'percent of 16472 inequalities '
            'violated. Max amplitude = full '
            'set (no magic subset). Fitted '
            'asymptote 0.714 > measured '
            'all-32 0.572. PREREG wording '
            'fixed (16472 = proper subsets '
            'incl. empty) and rerun - two '
            'authoritative runs bit-identical. '
            'Audit 36; ledger 213/L14 181.\n')
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
if 'max=3074' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model；旧条目可能混有 str，verify 需 isinstance 防御）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃/smoke 推翻均如实登记；verdict 单分支赋值。
4. 统计纪律：阈值预注册；**交互/组合分析必须在效应量同尺度（中位簿记）**；样本级噪声项求和放大 sqrt(n) 倍（3073 教训）；SMOKE 判决仅供管线验证。

## 标准锚与精度
- bit 级锚家族：标量行（a1）、因果置换（a3/a4）、跨 phase 参考数（aref）、块链恒等（b8）、hook 互证（a5/a6）、probe 协议（b9/b10）、跨 phase 因果复现（a10）、**嵌入式家族锚（3074：上一 phase 全部条件作为全子集扫描的子集 bit 复现，零额外前向）**。
- a2b 型校准断言只在末层读出层有效；中间层用 aref。SMOKE 跳过的锚在 setup_ok 中视为通过。
- **3070**：ATTN 换回=o_proj 输入 H 末位 pre-hook。**3071**：per-head 换回=mask 头切片。**3072**：V-clamp 推锚法；谱系恒等式 dzH_h=sum_p w_h(p)·dV_p[g(h)] cos=1.0；per-head 数组勿存 float64。**3073**：中位簿记 I(AB)=R(AB)-R(A)-R(B)；recov 和的基线减 N*med_c_34。**3074**：预算参数化 x(S)=Σ 单头振幅；双门外推（≤2 头拟合预言 8/32 头）；L35 head_decomp 需补捕获 ZA35。

## 机制解释审计链（命名前依次检查）
…→3072 焦点头谱系（线性读取 cos 1.0）→3073 头间交互（强次可加、二阶展开不完备、交互按效应量驱动、top8=全换回 93 percent、共享 TT 通道）→**3074 容量定律（判决 capacity_law_hill）：R(S) 在预算 x=Σ 单头振幅上服从 Hill 定律 A(x)=a·x^γ/(b^γ+x^γ)（a=0.714 b=0.454 γ=1.186），8/32 头联合双门外推成立（err 0.013/0.037）；非严格次模（26.5 percent 违反，超模成分在集合空间）；渐近容量 0.714>全换回 0.572（通道在名义容量下饱和）**。齿轮=线性读头，齿轮箱=非线性传动，传动轴共享，**传动轴有容量定律**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 D:/ 风格；管道工具缺失→脚本自带 run_log 用 Read 读；-c stdout 丢→写文件再 Read。
- 关键写入后必须 Grep/Read 复核（3074 出现 8 连 Edit 仅 2 处落盘的沙箱混合态，重应用后复核通过）；改后必编译检查。
- median 轴陷阱（3071）：smoke 维度裁剪可隐藏轴错误——断言形状。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3074）
Ω-P2（3011-3074）：…3071 attn_heads_focal；3072 focal_lineage_full；3073 higher_order_required；**3074 capacity_law_hill（共享通道容量定律）**。

## 下一步
- max=3074，下一个 3075（A 主选 **超模结构定位**：PAIRS4 数据反解协同头对，免前向；B 写入方向共线性（3072 npz 免前向）；C 焦点头跨 prompt 族稳定性；D DS7B 头级对照；E 神经元身份）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3074')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
