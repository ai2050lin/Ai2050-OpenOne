# -*- coding: utf-8 -*-
"""Phase 3144 closeout: five writes
(ledger 280->281 + MEMO append incl.
3145 prereg + workspace daily log +
MEMORY.md next-step) with idempotent
guards throughout."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D44 = os.path.join(
    RDIR, 'phase3144',
    'omega_p142_readouttraj_co36sign_'
    'unembed_d19resid')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')

# ---- load result + verify hashes ----
raw = io.open(os.path.join(D44,
                           'result.json'),
              'rb').read()
res_sha = hashlib.sha256(raw).hexdigest()[:8]
assert res_sha == 'def947d8', res_sha
res = json.loads(raw.decode('utf-8'))
assert res['smoke'] is False
V = res['verdict']
EXP = ('a_3143_ok|repro_bit_6|'
       'repro_bit_ok|dvec19_repro_6a0332|'
       'field_self_ok|z35_cos17_ok|'
       'traj_gradual|pc1_channel_format|'
       'tail_neg_confirmed|tail_sign_sym|'
       'identity_unembed_orthogonal|'
       'd19resid_readout|xphase_ok|'
       'coverage_full')
assert V == EXP, V
SEAL_SHA = str(res['seal_sha8'])
assert SEAL_SHA == 'bcd6fc5e', SEAL_SHA
assert os.path.exists(
    os.path.join(D44, 'design_seal.json'))
assert os.path.exists(
    os.path.join(D44, 'p142_readout.npz'))

# ---- 1. ledger ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
_n0 = len(led['measurements'])
assert _n0 in (280, 281), _n0
_has = any(m.get('phase') == 3144
           for m in led['measurements'])
if not _has:
    assert _n0 == 280, _n0
    entry = {
        'phase': 3144,
        'name': ('omega_p142_readouttraj_'
                 'co36sign_unembed_'
                 'd19resid'),
        'date': '2026-09-29',
        'kind': ('readouttraj_co36sign_'
                 'unembed_d19resid'),
        'verdict': V,
        'runtime_s': round(
            float(res['runtime_s']), 1),
        'hashes': {
            'result_sha256_8': res_sha,
            'seal_sha256_8': SEAL_SHA},
        'anchors': {
            'res43_sha8': 'ce348ff5',
            'seal43': 'e7f52d03',
            'dvec19_sha8': '6a0332a6',
            'xphase': 'P1.0/A1.1.0',
            'repro_bit': '6/6',
            'extra_bit': 'co50ex 0.421875 '
                         '= 3135/3137 co50 '
                         'anchor',
            'pc1_sign': '+1'},
        'summary': (
            'readout trajectory: L29-39 '
            'w_dn gap all-negative '
            '(max -0.219) -> blocking '
            'invisible in first-order '
            'projection; joint v1 '
            'component amplified 6.7x '
            'downstream (-10.4 -> -69.5 '
            'at L39); pc1 channel = '
            'format tokens (17/26, '
            'answer 0) + systematic '
            'content-token logit lift '
            '(131401 " preoc"); co36 '
            'sign: head 0.2188 (bit) '
            'tail 0.0938 tailpos 0.1016 '
            '-> tail net-suppressive '
            'AND sign-symmetric '
            '(noise-like, not '
            'directional signal); '
            'overlap HEAD-GT 16/50 '
            '(J 0.47, order_rho 0.186); '
            'unembed: self 0.00084 < '
            'rand 0.00117, top1-in-ents '
            '0/672 -> identity '
            'direction ORTHOGONAL to '
            'entity-token output '
            'geometry; dh19 residual '
            '~2/3 energy, perp@w_dn '
            'rises monotonically '
            '0.046(L20)->2.50(L39), '
            'dominates dh19 readout at '
            'L39 (2.50 vs 1.40 full) -> '
            'L19 independent component '
            'destined to answer '
            'readout.')}
    led['measurements'].append(entry)
    io.open(LEDGER, 'w',
            encoding='utf-8').write(
        json.dumps(led, ensure_ascii=False,
                   indent=1))
led2 = json.load(io.open(LEDGER,
                         encoding='utf-8'))
assert len(led2['measurements']) == 281
assert led2['measurements'][-1][
    'phase'] == 3144
print('1. ledger n=281 OK')

# ---- 2. MEMO append ----
block = '''

## Phase 3144: 读出轨迹+co36符号+unembed正交+dh19残差读出（T4 第27 Phase）[21:15]

**执行**：phase3144_omega_p142_readouttraj_co36sign_unembed_d19resid.py；SMOKE 198s（xphase P 3/4 为 batch=4 小样本批次伪象，正式 32-batch 全复现）→ 正式跑一次通过 4472s。设计四组：①高阶读出竞争定位（b_pc1/b_dvec29/b_joint 3 生成位级锚+sign 消解+L29-39 逐层 w_dn/v1/norm 轨迹+干扰 token 分类+teacher-forced logit 分解 26 行）②co36 正/负分离（order_e HEAD25/TAIL25 vs 3137 co36_rank top25/bot25 精确列表断言+交集/秩相关+6 注入@L17 A1 d1.0）③IDEINT_P unembed 溯源（实体 token 均值向量 self/obj/rand cos+logit-lens top1 命中+top25 能量份额）④dh19 残差场（field 重建+cond19/17 捕获 dose2.0→逐层行级投影残差归宿）。

**锚链**：PART A 3143 sha8=ce348ff5/seal=e7f52d03 全数值断言通过；dvec19 重捕获 sha8=6a0332a6（drift 0.00e+00，第 2 次跨会话 672 行位级）；field 自洽 cos 双 1.0000；xphase P/A1=1.0（128/128）；z35 cos_17 软锚 max|diff|=0.0102（与 3143 完全一致）；**bit 锚 6/6**（b_pc1 0.203125 + b_dvec29 0.640625 + b_joint 0.4765625 双 hook 构造兼容 + d_head 0.21875=k25 + d_gbot 0.1171875=3137 + d_co36full 0.140625 三重）+ 额外 d_co50ex 0.421875 位级命中 3135/3137 co50 锚（co50∩co36=∅ → co50ex=co50）。

### §1 发现1：blocking 不在 L29-39 逐层 w_dn 轨迹上；pc1 分量下游放大 6.7×
- w_dn 轨迹（128 行 median）：pc1 全程 0.04-0.37（暗）；dv29 6.51→2.18(L38 谷)→3.97(L39)；joint 全程 ≥ dv29（gap=max(dv29−joint) 全负，max −0.219 @L29）→ **traj_gradual：一阶读出投影轨迹上不存在 blocking 层位**——行为 blocking（joint 0.4766 < dv29 0.6406）不来自逐层 w_dn 递减。
- **v1 轨迹（joint）**：dh.v1 从 L29 −10.39 单调放大至 L39 **−69.51**（6.7×）——pc1 分量在 L30-39 下游被动态放大并主导状态空间（|v1| 69.5 vs w_dn 4.7 @L39）。高阶竞争的候选机制=**形态分量支配下游动态、把状态拉离答案读出**，而非任何单层的投影削减。

### §2 发现2：pc1 通道=format token 挤入+系统性 content-token logit 抬升（×3 强调）
- 干扰 token 分类（26 被干扰行）：pc1 条件 format 17/other 9/**answer 0**（format_frac 0.654，med t*=6.0）→ **pc1_channel_format**；对照 dvec29 全 other（82/82，med t*=4.0）——dvec29 改写内容、pc1 破坏句法节奏，两者行为通道不同。
- teacher-forced logit 分解：med Δlogit(t*)=+0.596（73% 为正）；top1Δ token 高度集中：' preoc'(131401) 6/8 抽样行、' tremend'/' tenden'——**pc1 注入系统性抬高特定实体风格 token 的 logit**，把句中位置挤入词片段而非翻转答案。瞬时注入下 base token logit 反而上升（+0.596）但被抬得更高的干扰 token 超过——单步瞬时与 allstep 累积的竞争结构不同（3145 用 allstep 历史重放对照）。

### §3 发现3：尾部坐标=符号对称噪声，非方向性负信号（×3 强调）
- 注入矩阵（A1，L17，d1.0）：head 0.2188（=k25 位级锚）/ tail(sgn−1) 0.0938 / tailpos(sgn+1) 0.1016 / gbot 0.1172（=3137 锚）/ co36full 0.1406 / co50ex 0.421875（=co50 锚，first=51 亦复现）。
- **tail_neg_confirmed + tail_sign_sym**：尾部净抑制确认（0.0938 << head 0.2188），但反向注入 chg 几乎不变（|Δ|=0.0078<0.03）——**尾部坐标的行为效应与注入符号无关=类噪声扰动，不是方向性负贡献信号**；头部是方向特异信号（3143 k25>全集的结构来源），尾部只是稀释/噪声底。
- 坐标身份：|HEAD∩GT|=16、|TAIL∩GB|=16（对角互补），J(HEAD,GT)=0.471，两序秩相关仅 0.186——3137 co36_rank 序与 3143 IDEINT 富集序是同 50 坐标的两种弱相关切割（top25 重叠 16 超几何期望 12.5，+1.6σ）；富集分数 head_med 2.78e-4 vs tail_med 1.88e-4（1.5×）。

### §4 发现4：身份方向与实体 token 输出几何正交；dh19 独立成分归宿=答案读出（×3 强调）
- unembed 溯源（672 行，65 实体 token id）：cos(IDEINT_P[17], 自身实体) 0.00084 **<** 随机 0.00117（obj 0.00062）——self 不高于随机反而更低；logit-lens top1 命中实体 token 0/672（chance 0.000429→期望 0.29）；top25 能量份额 1.25%（均匀 0.61% 的 2×，弱）→ **identity_unembed_orthogonal：IDEINT 身份方向不在 unembed 可读空间、不与任何实体 token 输出几何对齐**（结合 3143 iself_session：身份成分既不可迁移也不可读出——纯内部权重级痕迹）。
- dh19 残差（行级投影去 dh17 后）：perp share 全程 ~2/3（L20 0.607→L29 0.680→L38 0.640）；**perp@w_dn 单调上升 L20 0.046 → L29 0.613 → L33 1.262 → L38 1.576 → L39 2.496**，而 dh19 全量 w_dn L39 仅 1.40——**残差（独立成分）主导 yes-no 读出位移，dh17 通路部分反向抵消（−1.10）**；resid-field cos 由 0.563(L20) 衰减至 0.255(L39)（不与 swap 场共线）；L38 残差 top-PC 对齐 v1 0.229/w_dn 0.0004/dv29 0.091=新方向。
- 判决 **d19resid_readout**：L19 载体的独立成分直接写答案读出方向——机制解释 3143"L19 场更强（行为 chg 0.7891 的几何基础）"：**L19=行为直通载体，L17 通路=共享消费但读出贡献为负**。

### §5 综合 + 3145 预注册
3144 四问四答：(1) blocking 不在逐层 w_dn 轨迹，pc1 分量下游放大 6.7×（形态支配）；(2) pc1 通道=format 挤入+content-token logit 抬升，answer 0；(3) 尾部=符号对称噪声（非负信号），头部/尾部切割与 3137 序弱相关（ρ 0.186）；(4) 身份方向 unembed 正交 + dh19 残差归宿=答案读出（perp@wdn L39 2.50 主导）。

3145（Ω-P143）预注册：
1. v1 放大因果：dv29 单独 v1 轨迹 vs joint 对比 + joint 注入下 L38/L39 逐层 clip v1 分量（投影去除后重生成）——chg 是否恢复 dv29 单独水平（形态支配假说的因果检验）；
2. pc1 通道 token 谱：top1d token（' preoc' 等）逐层 logit 位移谱（从哪层开始抬升）+ allstep 历史重放 vs 瞬时注入对照（解释 dlogit(t*)>0 但 argmax 翻转）；
3. 头部符号补全：d_headpos（sgn+1）注入 → 头部方向特异性直接检验；tail 剂量曲线 d{2,4}（噪声注入是否剂量饱和/是否永远不产生方向性信号）；
4. dh19 残差因果：perp38 top-PC 方向 @L19/@L38 注入 → 行为 chg（残差方向的行为充分性）+ perp@L39 的 yes/no 分解（残差写的是 yes 还是 no 侧）。

关键数字：repro 6/6；gap max −0.2186@L29；joint v1 −10.39→−69.51（L39）；tokcls pc1 format 17/other 9/answer 0；dlogit(t*) +0.596；d_head 0.21875/tail 0.09375/tailpos 0.1015625/co50ex 0.421875；self_cos 0.00084<rand 0.00117；perp share L38 0.6405；perp@wdn L39 2.4955 vs dh19 1.4043。

锚：result sha8=def947d8，seal sha8=bcd6fc5e，dvec19 sha8=6a0332a6，ledger n=281。'''
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3144' not in t:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(block)
t2 = io.open(MEMO, encoding='utf-8').read()
assert t2.count('## Phase 3144') == 1
assert '3145（Ω-P143）预注册' in t2
print('2. MEMO appended OK')

# ---- 3. workspace daily log ----
p_log = (ROOT + r'\.workbuddy\memory'
         r'\2026-09-29.md')
add = '''

## Phase 3144 (Omega-P142) 闭环 [21:15]
- 正式跑一次通过 4472s（SMOKE 198s，xphase 3/4 为 batch=4 批次伪象）。verdict: a_3143_ok|repro_bit_6|repro_bit_ok|dvec19_repro_6a0332|field_self_ok|z35_cos17_ok|traj_gradual|pc1_channel_format|tail_neg_confirmed|tail_sign_sym|identity_unembed_orthogonal|d19resid_readout|xphase_ok|coverage_full。
- 四发现：①blocking 不在 L29-39 逐层 w_dn 轨迹（gap 全负 max −0.219），joint v1 分量下游放大 6.7×（−10.4→−69.5）②pc1 通道=format token 挤入（17/26，answer 0）+系统性 ' preoc' logit 抬升 ③尾部=符号对称噪声（tail 0.0938/tailpos 0.1016），头部方向特异（0.2188 bit）；两序 ρ 0.186 ④身份方向 unembed 正交（self 0.00084<rand）+dh19 残差 ~2/3 能量 perp@wdn 单调升至 L39 2.50 主导读出（d19resid_readout）。
- 6/6 位级锚（b_pc1/b_dvec29/b_joint 3142 三锚 + d_head k25 + d_gbot 3137 + co36full 三重）+co50ex 0.421875 额外命中；xphase 1.0/1.0；dvec19 sha 6a0332a6 drift 0（第 2 次）。
- closeout：ledger n=281（res sha8=def947d8、seal bcd6fc5e）+ MEMO（T4 第27Phase，含 3145 预注册）+ 本日志 + MEMORY。'''
if os.path.exists(p_log):
    t = io.open(p_log, encoding='utf-8').read()
else:
    t = ''
    os.makedirs(os.path.dirname(p_log),
                exist_ok=True)
if 'Phase 3144 (Omega-P142) 闭环' not in t:
    io.open(p_log, 'w',
            encoding='utf-8').write(t + add)
t2 = io.open(p_log, encoding='utf-8').read()
assert 'Phase 3144 (Omega-P142) 闭环' in t2
print('3. daily log OK')

# ---- 4. MEMORY.md next-step line ----
p_mem = (ROOT + r'\.workbuddy\memory'
         r'\MEMORY.md')
tm = io.open(p_mem, encoding='utf-8').read()
lines = tm.splitlines()
out_lines = []
changed = False
NEWLINE = ('- max=3144，下一 3145：①v1 放大因果（dv29 v1 轨迹对比+joint 下 L38/39 clip v1 分量重生成）②pc1 通道 token 谱（top1d token 逐层 logit 位移+allstep 历史重放对照）③d_headpos 补全头部符号+tail 剂量 d{2,4} ④dh19 残差 top-PC 注入因果+perp@L39 yes/no 分解。**blocking 不在逐层 w_dn 轨迹、pc1 形态分量下游放大 6.7× 主导；pc1 通道=format 挤入+content-token 抬升（answer 0）；尾部=符号对称噪声；身份方向 unembed 正交；dh19 独立成分（~2/3 能量）归宿=答案读出（perp@wdn L39 2.50 主导，L17 通路部分反向 −1.10）。**')
for l in lines:
    if l.startswith('- max=3143，下一 3144'):
        out_lines.append(NEWLINE)
        changed = True
    else:
        out_lines.append(l)
if changed:
    io.open(p_mem, 'w',
            encoding='utf-8').write(
        chr(10).join(out_lines) + chr(10))
tm2 = io.open(p_mem, encoding='utf-8').read()
assert 'max=3144' in tm2
print('4. MEMORY.md OK (changed=%s)'
      % changed)
print('CLOSEOUT ALL DONE')
