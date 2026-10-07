# -*- coding: utf-8 -*-
"""Phase 3142 closeout: ledger n=279 +
MEMO append + wlog + MEMORY update.
Idempotent by design (3139/3140/3141
lessons applied)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D42 = os.path.join(
    RDIR, 'phase3142',
    'omega_p140_l19anat_blockmat_'
    'quartile_multitpl')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = ROOT + r'\.workbuddy\memory' \
       r'\2026-09-29.md'
WMEM = ROOT + r'\.workbuddy\memory' \
       r'\MEMORY.md'

RES_SHA = '7291cfc5'
SEAL_SHA = '449f8161'
VERDICT = ('a_3141_ok|repro_bit_9|'
           'repro_bit_ok|dvec19_sha_6a0332|'
           'dvec19_active|s1_peak_l21|'
           's1_zone_19_20_21|s1_dose_mono|'
           'bi_pc1_dose_dominant|'
           'crosslayer_blocking|'
           'order_bit_invariant|'
           'quartile_mono_fail|window_d1.0|'
           'f1fwd_bit_ok|f2gen_bit_ok|'
           'write_structural_absent|'
           'gen_retr_below|xphase_ok|'
           'coverage_full')

# ---- verify result on disk ----------
raw = io.open(os.path.join(D42,
                           'result.json'),
              'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
assert sha == RES_SHA, sha
res = json.loads(raw.decode('utf-8'))
assert res['verdict'] == VERDICT
assert str(res['seal_sha8']) == SEAL_SHA
assert res['smoke'] is False
assert os.path.exists(
    os.path.join(D42,
                 'p140_readout.npz'))
assert os.path.exists(
    os.path.join(D42,
                 'design_seal.json'))
assert os.path.exists(
    os.path.join(D42, 'run_log.txt'))
print('result artifacts ok '
      '(sha8 %s seal %s)'
      % (RES_SHA, SEAL_SHA))

# ---- 1. ledger ----------------------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
_n0 = len(led['measurements'])
assert _n0 in (278, 279), _n0
_has = any(m.get('phase') == 3142
           for m in led['measurements'])
if not _has:
    assert _n0 == 278, _n0
    entry = {
        'phase': 3142,
        'name': ('omega_p140_l19anat_'
                 'blockmat_quartile_'
                 'multitpl'),
        'date': '2026-09-29',
        'kind': ('l19anat_blockmat_'
                 'quartile_multitpl'),
        'verdict': VERDICT,
        'runtime_s': 9808.0,
        'hashes': {
            'result_sha256_8': RES_SHA,
            'seal_sha256_8': SEAL_SHA},
        'anchors': {
            'res41_sha8': 'a93c8892',
            'res37_sha8': '4f8055bb',
            'res36_sha8': '3903af46',
            'dvec_sha8': {
                '17': '5e4c3085',
                '29': 'ee9484b2',
                '33': '59fbe0d3',
                '38': 'aced803b'},
            'dvec19_sha8': '6a0332a6',
            'xphase': 'P1.0/A1.1.0',
            'repro_bit': '9/9'},
        'summary': (
            'dvec19 captured (med 5.64) '
            'and is the strongest '
            'behavioral carrier: allstep '
            'd1 0.7891 > dvec29 d2 '
            '0.6406; step1 sensitive '
            'zone [19,20,21] with L21 '
            'peak 0.3672 d4 (dose-mono '
            '0.80); blocking localized '
            'to readout competition: '
            'cross-layer joint still '
            'blocks (0.1953 vs p_add '
            '0.4141), order-reversal '
            'bit-identical, BI '
            'asymmetric pc1-dose '
            'dominant (0.698 vs 0.122); '
            'quartile split collapses '
            'enrichment structure '
            '(d0.5 flat, d1.0 weak '
            'non-mono sep 0.0547) -> '
            'halves contrast is '
            'tail-driven; write '
            'immunity confirmed on 84 '
            'pairs: f2/f4-single/f3-'
            'quad all chance-level -> '
            'identity direction is not '
            'context-writable '
            '(structural absence).')}
    led['measurements'].append(entry)
    io.open(LEDGER, 'w',
            encoding='utf-8').write(
        json.dumps(led,
                   ensure_ascii=False,
                   indent=1))
led2 = json.load(io.open(LEDGER,
                         encoding='utf-8'))
assert len(led2['measurements']) == 279
assert led2['measurements'][-1][
    'phase'] == 3142
print('ledger n=279 OK')

# ---- 2. MEMO append ------------------
block = """

## Phase 3142: L19 载体解剖 + blocking 读出竞争定位 + 四分位失败 + 写入免疫定案（T4 第25Phase）[21:40]

### §0 执行
脚本 tests/glm5/phase3142_omega_p140_l19anat_blockmat_quartile_multitpl.py（2028 行）。SMOKE 788s（rev 无）；正式跑主体 7159s 至 C3 尾（rev-3142a：bit 锚 c_all_l23_d2.0 引用未跑试验→改 c_all_l19_d2.0=0.4922，ckpt 续跑）+ 续跑 9808s 完成装配。总 ~4.7h。

### §1 PART C：dvec19 强载体 + step1 敏感带 [L19-21]
dvec19 全新捕获（3135 swap4=[8,9,13,29] 层跳过语义，P 行 672，单样本）：sha8=6a0332a6，med||d||=5.6436（vs dvec29 31.64）。

**发现 1（×3）：dvec19 是全谱最强行为载体——allstep d1 注入 chg=0.7891，超过 dvec29 d2 的 0.6406（幅度仅为其 1/5.6）；step1 d1 0.1641 亦活跃。L19 载体以极小范数实现最大行为改写——行为改写效率（chg/||d||）在 L19 达到峰值。**

step1 精扫（L17-21 × d{1,2,4}）：
- d1: L17 0.1172 / L18 0.0938 / L19 0.0703 / L20 0.0469 / L21 0.1406
- d2: L17 0.1797 / L18 0.1094 / L19 0.1406 / L20 0.0938 / L21 0.2031
- d4: L17 0.1016 / L18 0.1719 / L19 0.2422 / L20 0.1562 / L21 0.3672
- 单调敏感区=[19,20,21]（d4≥2×d1），s1_peak=L21（0.2031@d2），mono=0.80；L17 是 d2 峰回落型（非敏感）
- allstep d2 对照：L17 0.1875 / L18 0.3984 / L19 0.4922 / L20 0.1406 / L21 0.2656——L19 allstep 峰与 3141 位级一致（0.4922 ✓）

结构解读：L17→L21 单步敏感度递增而 allstep 峰在 L19——L19-21 是"写入-消费过渡带"：L19 载体注入即时改写（dvec19 强活跃佐证），L21 单步模板脉冲最敏感，L17 单步钝化（3141 已示 its step1 谱低）。

### §2 PART D：blocking=读出竞争定位 [非同层破坏]
9 格剂量矩阵（pc1 × dvec29 @ L29 allstep）+ 跨层 + 反序：

singles: pc1 d{0.5,1,2}=0.0625/0.0859/0.2031（超线性）；dvec29 d{0.5,1,2}=0.2109/0.2500/0.6406；dvec33 d1=0.2109。

joint j 矩阵（a=pc1 dose, b=dvec dose）：
- a=0.5: 0.2031/0.2266/0.6172；a=1.0: 0.1562/0.2188/0.5938；a=2.0: 0.1250/0.1953/0.4766
- BI=1−j/p_add：a=2.0 行 0.698/0.569/0.435；b=2.0 列 0.122/0.183/0.435

**发现 2（×3）：blocking 定位为读出端竞争，非同层状态破坏——①跨层 joint（pc1@L29 d2 + dvec33@L33 d1）仍 blocking（j=0.1953 << p_add 0.4141）；②注入顺序反转位级一致（Δ=0.00e+00，order_bit_invariant）；③BI 非对称：pc1（全局 WR 形态方向）剂量是 blocking 主导因子（BI 0.698→0.435 随 dvec 剂量仅缓降，而 dvec 低剂量侧 BI 随 pc1 剂量陡增 0.257→0.698）。**

机制模型修正（3141 joint_blocking → 3142 精化）：pc1 与 dvec 在答案 token 读出端竞争——pc1 幅度增大时挤占 dvec 的行为输出（BI 单调升），而 dvec 剂量增大时自身覆盖读出（j 收敛至单侧值）。两方向非加性、非独立，是同一读出瓶颈的竞争者。

### §3 PART E：四分位失败=尾部效应 [粗粒度富集]
order_e 与 3141 high25/low25 位级一致（纯 numpy 断言过）。四分位 Q1-Q4（13/13/12/12）× d{0.5,1.0} @ L17 A1 rows（3137 G 语义）：
- d0.5: 0.1016/0.0938/0.1016/0.0938（全平，sep 0.0078）
- d1.0: 0.1562/0.1641/0.1094/0.1016（sep 0.0547，但 Q2>Q1 非单调 → quartile_mono_fail）
- 3137 锚 3/3 bit-match（co36 d1 0.1406 ✓ / co36 d2 0.2266 ✓ / co50 d1 0.4219 ✓）

**发现 3（×3）：富集→因果映射是粗粒度/尾部驱动——3141 的 high25/low25 分离 2.3×（d1）在四分位细分后消失（Q1≈Q2、Q3≈Q4），分离仅存在于 halves 尺度的极端对比；单坐标富集度不线性预测因果贡献，存在簇级阈值或饱和。**

### §4 PART F：写入免疫定案 [structural_absent]
84 unseen pairs（chance 0.0357）三重写入检验：
- f2 V1 gen（3141 语义）：0.0476/0.0357/0.0595 —— 3141 retr_gen_v1 位级复现 ✓
- f4 单变体写入（变体 2 prompt+gen+V1 查询）：0.0357/0.0476/0.0357
- f3 四变体拼接写入（V0-V3 各 gen 12 token + V1 查询）：0.0357/0.0357/0.0357
- gains 中位：f2 +0.0000 / f4 −0.0238 / f3 −0.0238；gen best 0.0595 << 0.30
- f1 fwd V1 与 3141 位级复现 ✓；SMOKE 的 0.5000（6 行）为小样本假象

**发现 4（×3）：身份方向对上下文写入完全免疫（结构性缺失定案）——单模板生成、单变体模板改写、四变体 in-context 重复写入均不能使新材料行进入 IDEINT 检索域。bank I 成分不是"上下文可写状态"而是跨模板共享的稳定结构（可能权重级训练记忆或全局身份几何）；上下文工程（prompt/生成/重复）无法伪造身份。**

### §5 综合 + 3143 预注册
3142 四问四答：(1) dvec19=最强载体（效率峰值 L19）+ step1 敏感带 [19,20,21]；(2) blocking=读出竞争（跨层+反序+pc1 主导）；(3) 富集因果=尾部/粗粒度；(4) 写入免疫=结构性（三重写入全阴）。

3143（Ω-P141）预注册：
1. dvec19 下游传导场：dvec19 注入 L19（d1 allstep）→ capture 下游 L20-38 逐层状态 diff → 与 dvec17（3135 已有 dvec_full_17 参考场）下游场 cos 谱对比——L19 载体是否进入与 L17 相同的下游消费通路；
2. 读出竞争直接观测：dvec29 注入前后 L29 末位状态在 pc1 方向投影能量的位移 + w_dn 读出方向合成角——直接检验"读出挤压"假说；
3. 富集阈值定位：order_e 滑动切点 top-k（k∈{5,10,15,20,25,30}）× d{0.5,1.0} @ L17——定位分离最大的坐标数阈值（粗粒度假说的定量形式）；
4. 新材料 I 成分存在性：84 新材料行 4 模板 bank-style capture → 构造新材料 IDEINT_new → 自检索（新材料内部）+ 旧 bank 交叉检索——检验新材料是否拥有自己的 I 成分（区分"身份=旧材料权重记忆" vs "身份=通用编码机制"）。

关键数字：repro 9/9；dvec19 all_d1 0.7891/s1 0.1641；s1 d4 L21 0.3672/L19 0.2422；all d2 L19 0.4922/L18 0.3984；crosslayer j 0.1953（p_add 0.4141）；rev Δ=0；BI(2.0,0.5)=0.698；Q d1.0 [0.1562,0.1641,0.1094,0.1016]；f2/f4/f3 全 chance；gen best 0.0595。

锚：result sha8=7291cfc5，seal sha8=449f8161，dvec19 sha8=6a0332a6，ledger n=279。"""
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3142: L19 载体解剖' not in t:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(block)
t2 = io.open(MEMO, encoding='utf-8').read()
assert t2.count(
    '## Phase 3142: L19 载体解剖') == 1
assert '3143（Ω-P141）预注册' in t2
print('MEMO ok')

# ---- 3. wlog append ------------------
wadd = """

## Phase 3142 (Omega-P140) closeout [21:40]
- 正式跑完成：主体 7159s（C3 尾 rev-3142a：bit 锚 c_all_l23 未跑→改 c_all_l19_d2.0）+ 续跑 9808s 装配。verdict 19 tags，9/9 bit 锚，xphase 双 1.0。
- 四发现：①dvec19 最强载体 all_d1 0.7891（med 5.64，效率峰值）；step1 敏感带 [19,20,21]（L21 d4 0.3672）②blocking=读出竞争：跨层仍 blocking（0.1953<<0.4141）+反序位级一致+BI pc1 剂量主导 0.698 ③四分位失败：Q1≈Q2/Q3≈Q4，halves 分离=尾部效应 ④写入免疫定案：f2/f4/f3 全 chance（0.0357-0.0595），身份非上下文可写。
- closeout：ledger n=279（res sha8=7291cfc5、seal 449f8161）、MEMO 追加（T4 第25Phase）、MEMORY.md max=3142、3143（Ω-P141）预注册（dvec19 传导场/读出挤压观测/富集阈值 top-k 滑动/新材料 I 成分存在性）。"""
tw = io.open(WLOG, encoding='utf-8').read()
if 'Phase 3142 (Omega-P140) closeout' \
        not in tw:
    with io.open(WLOG, 'a',
                 encoding='utf-8') as f:
        f.write(wadd)
tw2 = io.open(WLOG, encoding='utf-8').read()
assert tw2.count(
    'Phase 3142 (Omega-P140) closeout') == 1
print('wlog ok')

# ---- 4. MEMORY.md update ------------
tm = io.open(WMEM, encoding='utf-8').read()
lines = tm.splitlines()
changed = False
out_lines = []
new_next = ('- max=3142，下一 3143：**dvec19 下游传导场（L19 注入→L20-38 diff vs dvec17 场 cos 谱）+ 读出竞争直接观测（dvec 注入后 pc1 投影位移 + w_dn 合成角）+ 富集阈值 top-k 滑动（k∈{5..30}）+ 新材料 4 模板 I 成分存在性（IDEINT_new 自检索 vs 旧 bank 交叉）。**L19 效率峰值载体确立；blocking=读出竞争（跨层+反序+pc1 主导）；富集因果=尾部粗粒度；写入免疫结构性定案（三重写入全阴）——身份=跨模板稳定结构/权重级，非上下文可写。**')
for l in lines:
    if l.startswith('- max=3141，下一 3142') \
            or l.startswith('- max=3140，下一 3141'):
        l = new_next
        changed = True
    out_lines.append(l)
if not changed:
    if '- max=3142，下一 3143' not in tm:
        out_lines.append(new_next)
        changed = True
io.open(WMEM, 'w',
        encoding='utf-8').write(
    chr(10).join(out_lines) + chr(10))
tm2 = io.open(WMEM, encoding='utf-8').read()
assert '- max=3142，下一 3143' in tm2
print('MEMORY.md ok (changed=%s)' % changed)
print('CLOSEOUT 3142 COMPLETE')
