# -*- coding: utf-8 -*-
"""p3149 closeout: idempotent five writes
(ledger + MEMO + daily log + MEMORY.md +
final check)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D49 = (ROOT + r'\tests\glm5\result'
       + r'\rdc_query_construction_20260913'
       + r'\phase3149'
       + r'\omega_p147_carrier_dlogit_'
       + r'poslate_kdose_v3amp')
RES_SHA = '108d044e'
SEAL = '29d924e2'
MEMO = (ROOT + r'\research\gpt5\docs'
        + r'\AGI_GPT5_MEMO.md')
DAILY = (ROOT + r'\.workbuddy\memory'
         + r'\2026-10-01.md')
MEM = (ROOT + r'\.workbuddy\memory'
       + r'\MEMORY.md')

raw = io.open(D49 + r'\result.json',
              'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
assert sha == RES_SHA, sha
r = json.loads(raw.decode('utf-8'))
assert r['verdict'].startswith('a_3148_ok')
assert 'repro_bit_ok' in r['verdict']
assert 'carrier_common_found' in \
    r['verdict']
print('CHECK res sha8 %s ok' % sha)

# ---- 1. ledger ------------------------
LP = (ROOT + r'\research\gpt5\atlas'
      + r'\atlas_ledger.json')
led = json.load(io.open(LP,
                        encoding='utf-8'))
has49 = any(m.get('phase') == 3149
            for m in led['measurements'])
if not has49:
    led['measurements'].append({
        'phase': 3149,
        'name': r['name'],
        'seal_sha8': r['seal_sha8'],
        'result_sha8': sha,
        'verdict': r['verdict'],
        'created': r['created'],
        'runtime_s': r['runtime_s']})
    tmp = LP + '.tmp'
    with io.open(tmp, 'w',
                 encoding='utf-8') as f:
        json.dump(led, f,
                  ensure_ascii=False,
                  indent=1)
    os.replace(tmp, LP)
led2 = json.load(io.open(LP,
                         encoding='utf-8'))
n = len(led2['measurements'])
assert n == 286, n
assert led2['measurements'][-1][
    'phase'] == 3149
print('CHECK ledger n=%d ok' % n)

# ---- 2. MEMO --------------------------
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3149:' not in t:
    memo49 = """

## Phase 3149: 载体分离+早期固化+坐标剂量互换+偏置增长（T4 第32 Phase）[02:05]

**执行**：phase3149_omega_p147_carrier_dlogit_poslate_kdose_v3amp.py；SMOKE 313s 全链贯通（patch1-3 骨架改造+3 轮补漏：G_TOP25 坐标表/PSHARE_38 门值补回、chunked fp32 unembed matmul 防 OOM）→ **正式跑有效 2498.3s**（42 min，首轮 8/8 bit 后 T2/L 连修 3 处 rev-3149a/b/c/d/e——norm_g grad 泄漏、_injp tuple 守卫、d131 标量内积、cut gen 行内容传参、kmin 改 fstep 判定；全部 ckpt 续跑零重算已存 stage）。设计四组：①neg_d0.5 8 翻转行+pos_d1 13 翻转行 step0-1 dlogit per-token 分解（chunked fp32 投影 WUG）→ top-20 共同抬升集 ②pos_d1 13 行逐 step L38/39 capture（wdn 投影 vs dlogit131401 vs |dh|）+ firstk 前缀截断 gen k{0..4} ③co50ex top-5/top-10 @d{0.5,1,2} vs full 剂量曲线互换检验 ④v1 微放大 α{−0.05,−0.10,−0.15} @L38 all 与 3147 微削减并排。

**锚链**：PART A 3148 sha8=29327d0e/seal=87842206 全数值断言通过（xphase/part_x2 双曲线 12 值/xcross [3.0,3.5]/flip 行 13+39/part_h/part_u/part_v2）；xphase P/A1=1.0（128/128，**第 7 次跨会话**）；dvec19 sha8=6a0332a6（**第 7 次跨会话位级**，drift 0.00e+00）；field 自洽双 1.0000；z35 软锚 0.0102；C4 resid 锚 4/4 精确（share38 0.6405/wdn_p39 2.4955/ra_v1 0.2287/pnorm38 55.263）；**bit 锚 8/8 全命中**（b_pc1/b_dvec29/b_joint 第 7 次、d_co50ex_d2.0 第 3 次、n_neg_d0.5 0.2422 第 4 次、s2_tailpos_d1/s2_tailneg_d1 第 2 次、v2_clip_a025 第 3 次）+ flip_rows/fsteps 冻结表精确 + neg_d1 翻转 12 行与 pos_d1 13 行双提取。

### §1 发现1：翻转载体按符号分离，131401 非主载体（×3 强调）
- **carrier_common_found**：neg 8 行共同抬升集 **13 个 token**（≥4 行 top-20 交集，非 131401）{4330, 55745, 58846, 88070, 99365, 100761, 132263, 132676, 134185, 135505, 139010, 139819, 140086}；pos 13 行共同抬升集 **9 个 token** {55, 574, 785, 1254, 10690, 14813, 54783, 95871, 145145}；**shared = 空**——正负符号的翻转载体是完全不同的 token 族。
- dlogit(131401) 中位数：neg **+0.102**（rank 12442/151936，top 8%）vs pos **−0.025**（rank 85052，≈随机）——format token 的抬升是 neg 侧的次要伴生分量，在 pos 侧几乎不存在。3148 uncancel_insufficient（去除 format 方向反而加重破坏 0.24→0.82）由此获得载体层面的解释：**131401 从来不是翻转动力的主通道**，它只是 neg 侧干扰的指纹；真正载体是分散的 token 族（neg 13 个/pos 9 个）。
- 方法论：dlogit 分解在注入行与 base 行的逐步 forward（allstep 注入语义逐步重现）上做差，chunked fp32 matmul（268MB 峰值）避免全量 fp32 unembed 的 2.5GB 显存开销——top-50 |dlogit| 与 rank 直接从 full-vocab 投影提取。

### §2 发现2：pos_d1"晚发"=早期固化+读出渐进分歧（×3 强调）
- **poslate_fixed_early**：13 翻转行前缀截断 gen（注入仅到 decode step k）显示 **kmin [0,1,0,0,0,0,1,0,−1,0,3,−1,0]**——**77%（10/13）行仅 prompt 末位一次注入（cut0）即固化翻转**（frac_fixed 0.769），仅 2 行需持续注入（kmin=−1），1 行需到 cut3。而 fstep [0,8,8,1,0,8,1,9,8,0,4,9,0] med 4.0——首个可见差异 token 晚现。
- 结论：**"晚发"不是机制延迟而是读出渐进**——注入在 prompt 末位即刻写入（固化），但生成序列与 base 的分歧在 decode 中段才首次可见（wdn 投影早期 −0.007≈0、dlogit131401 早期 −0.024≈0，到 fstep 处也仅 −0.016/−0.003——整条轨迹上 w131401 分量都接近零，再次印证 §1）。低剂量注入=单次写入+下游逐步放大读出；高剂量 neg_d4=立即压制首 token（3148 的 82% 即时）——两种时间结构是同一写入在不同剂量下的读出表现。
- 与 3147 neg_late_format_shift 对照：neg_d0.5 的"format 通道偏移"（dlogit131401 前两步 −0.412→+1.453）在逐行分解后表现为 rank 12442 的次要分量——3147 §3 的"format 载体"解释正式降级为描述性共伴（3148 §3 已否定其因果充分性，3149 §1 完成载体层否定）。

### §3 发现3：坐标数×剂量乘积守恒（×3 强调）
- **kx_interchangeable**：top-10 坐标 @d1 (chg 0.1641) ≈ full 50 坐标 @d0.5 (0.1406)，gap **0.023**；曲线族 top5 {0.5: 0.070, 1: 0.094, 2: 0.250}、top10 {0.5: 0.102, 1: 0.164, 2: 0.469} vs full {0.5: 0.141, 1: 0.422, 2: 0.852}（+3147 @d4 top10 0.891/full 0.992）。
- 结构：**坐标数×剂量≈行为剂量的乘积守恒**（10×1≈50×0.5；10×2≈50×1 的方向也成立 0.469 vs 0.422 gap 0.047）——co50ex 注入的行为效应近似随"总注入质量"（坐标数×单位幅度）累积，单坐标身份在低剂量区不敏感（与 3137"错配有效性数量驱动"一致，与 3148 top-2 仅 17% 份额互证）。高剂量区（≥d2）乘积守恒破缺（top10_d2 0.469 vs full_d1 0.422 vs full_d2 0.852——广度在 S 形阈值段仍有额外贡献）。
- 阈值结构：full 曲线的 S 形（0.141→0.422→0.852→0.992）在 top-5/10 子集上整体右移——**广度换强度是可行的**，坐标族是近似可加的剂量载体池。

### §4 发现4：v1 轴微幅度完全对称，偏置随幅度增长（×3 强调）
- **v3_bias_growing**：sym 曲线（放大/削减比）{α0.05: **1.000**, α0.10: **1.000**, α0.15: 0.706, α0.25: 0.632, α0.50: 0.576}——微幅度 α≤0.10 时**放大与削减位级等价**（v3_ampan_a005 0.0703 = 3147 v_base_a005 0.0703、a010 0.0781 = 0.0781），α≥0.15 偏置出现并单调增长。
- 结论：v1 轴（WR PC1）在小扰动区是**无方向偏置的连续幅度读出**（|扰动|决定行为破坏），方向偏置是大扰动下的非线性现象（3147 v1_micro_continuous 的"连续依赖"+3148 v1_sym_mixed 的统一：偏置不是恒定属性而是**幅度依赖的阈值后效应**）。承重轴语义：v1 是被双向使用的通路，只有大幅 push/pull 才暴露不对称。

### §5 综合 + 3150 预注册
3149 四问四答：(1) 翻转载体=按符号分离的分散 token 族（neg 13/pos 9，零交集），131401 仅 neg 侧次要分量（rank 8%）——载体层完成"format 非载体"闭环；(2) pos 晚发=prompt 末位单次注入已固化（77%）+读出渐进分歧，机制无延迟；(3) 坐标数×剂量乘积守恒（top10@d1≈full@d0.5，gap 0.023），广度换强度可行；(4) v1 偏置是幅度阈值后效应（α≤0.10 完全对称）。

3150（Ω-P148）预注册：
1. 载体 token 身份解码：neg 13/pos 9 共同 token 的 tokenizer decode + 语义分类（format/答案/标点/其他）+ 与 dvec29/dvec19 top-50 的重合——载体 token 是"内容尾迹"还是"模板残差"（3125 trail 假说回检）；
2. 载体 token 因果验证：neg 13-token 集合按 (a) 全集注入 unembed 方向 (b) 逐 token 注入 @L39 @d0.5——检验共同抬升集是否为充分的载体重建（对照 131401 单 token 注入）；
3. 乘积守恒的等值线：top-15/top-25 @d{0.25,0.5} 补齐 (k,d) 网格，拟合 chg ≈ f(k·d) 单参数曲线——守恒律的适用域与破缺点；
4. v1 偏置阈值定位：α{0.15,0.20,0.25} 微削减补齐 + 偏置量 (1−sym) 对 α 的对数斜率——阈值结构是锐利还是渐进。

关键数字：bit 8/8；neg common 13 tok/d131 +0.102 (r 12442)；pos common 9 tok/d131 −0.025 (r 85052)；shared=∅；frac_fixed 0.769 (kmin 10×0/2×1/1×3/2×−1 vs fstep med 4.0)；wdn early −0.007/d131 early −0.024；kx top5 {0.070,0.094,0.250}/top10 {0.102,0.164,0.469} best gap 0.023 (top10@d1~full@d0.5)；sym {1.000,1.000,0.706,0.632,0.576} growing。

锚：result sha8=108d044e，seal sha8=29d924e2，dvec19 sha8=6a0332a6（第 7 次），ledger n=286。
"""
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(memo49)
t2 = io.open(MEMO, encoding='utf-8').read()
assert '## Phase 3149:' in t2
print('CHECK MEMO 3149 ok')

# ---- 3. daily log ---------------------
if not os.path.exists(DAILY):
    with io.open(DAILY, 'w',
                 encoding='utf-8') as f:
        f.write('# 2026-10-01\n')
d = io.open(DAILY, encoding='utf-8').read()
if 'Phase 3149 闭环' not in d:
    with io.open(DAILY, 'a',
                 encoding='utf-8') as f:
        f.write(
            '\n## Phase 3149 闭环（Ω-P147）\n'
            '- 正式跑有效 2498.3s（3 处 '
            'rev-3149a/b/c/d/e 修复全 ckpt '
            '续跑）；bit 8/8；verdict 全绿：'
            'carrier_common_found | '
            'poslate_fixed_early | '
            'kx_interchangeable | '
            'v3_bias_growing\n'
            '- 载体按符号分离（neg 13 tok/'
            'pos 9 tok 零交集，131401 仅 neg '
            '次要分量 rank 8%）——format 非'
            '载体闭环\n'
            '- pos 晚发=早期固化 77%+读出渐'
            '进；坐标数×剂量乘积守恒 gap '
            '0.023；v1 偏置幅度阈值后效应\n'
            '- res sha8=108d044e seal='
            '29d924e2 ledger n=286\n'
            '- 3150 预注册：token 身份解码+'
            '载体因果注入+乘积等值线+偏置'
            '阈值\n')
print('CHECK daily ok')

# ---- 4. MEMORY.md ---------------------
m = io.open(MEM, encoding='utf-8').read()
OLD_MAX = '- max=3148，下一 3149'
if OLD_MAX in m:
    i0 = m.index(OLD_MAX)
    i1 = m.index('\n', i0)
    old_line = m[i0:i1]
    new_line = (
        '- 3149（T4）：载体按符号分离 '
        'carrier_common_found（neg 13 tok '
        '{4330,55745,58846,88070,99365,'
        '100761,132263,132676,134185,'
        '135505,139010,139819,140086}/pos '
        '9 tok {55,574,785,1254,10690,'
        '14813,54783,95871,145145} 零交集；'
        'd131 med neg +0.102 r12442 vs '
        'pos −0.025 r85052=131401 非主载'
        '体，3148 uncancel 载体层闭环）；'
        'poslate_fixed_early（kmin '
        '10×0/2×−1/1×3=77% prompt 单注入'
        '固化 vs fstep med 4.0=晚发是读出'
        '渐进非机制延迟，wdn/d131 全程≈0）；'
        'kx_interchangeable（top10@d1 '
        '0.164≈full@d0.5 0.141 gap 0.023='
        '坐标数×剂量乘积守恒，广度换强度'
        '可行，高剂量破缺）；v3_bias_'
        'growing（sym {1.0,1.0,0.706,'
        '0.632,0.576}=α≤0.10 完全对称，'
        '偏置幅度阈值后效应）。'
        'res sha8=108d044e seal=29d924e2 '
        'ledger n=286；3150 预注册：token '
        '身份解码+载体因果注入+(k,d) 等值'
        '线+偏置阈值定位')
    m = m[:i0] + new_line + m[i1:]
    m = m.replace(
        old_line, new_line, 1)
    io.open(MEM, 'w', encoding='utf-8',
            newline='\n').write(m)
m2 = io.open(MEM, encoding='utf-8').read()
assert '3149（T4）' in m2, 'MEMORY 3149 line missing'
print('CHECK MEMORY ok')

# ---- 5. final check -------------------
led3 = json.load(io.open(LP,
                         encoding='utf-8'))
assert len(led3['measurements']) == 286
raw2 = io.open(D49 + r'\result.json',
               'rb').read()
assert hashlib.sha256(raw2).hexdigest()[
    :8] == RES_SHA
npz = D49 + r'\p147_readout.npz'
assert os.path.exists(npz), npz
print('FINAL CHECK: ledger n=286, res '
      'sha8=%s, seal=%s, npz ok -> '
      'CLOSEOUT COMPLETE' % (RES_SHA,
                             SEAL))
