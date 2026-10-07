# -*- coding: utf-8 -*-
"""Phase 3143 closeout: five writes
(ledger 279->280 + MEMO append + two
workspace logs + MEMORY.md) with
idempotent guards throughout."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D43 = os.path.join(
    RDIR, 'phase3143',
    'omega_p141_d19field_readout_topk_'
    'newI')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')

# ---- load result + verify hashes ----
raw = io.open(os.path.join(D43,
                           'result.json'),
              'rb').read()
res_sha = hashlib.sha256(raw).hexdigest()[:8]
assert res_sha == 'ce348ff5', res_sha
res = json.loads(raw.decode('utf-8'))
assert res['smoke'] is False
V = res['verdict']
EXP = ('a_3142_ok|repro_bit_5|'
       'repro_bit_ok|dvec19_repro_6a0332|'
       'field_self_ok|z35_cos17_ok|'
       'd19_path_partial|d19_direct_hi|'
       'stat_add|readout_comp_absent|'
       'pc1_readout_dark|head_conc_k13|'
       'iself_session|xphase_ok|'
       'coverage_full')
assert V == EXP, V
SEAL_SHA = str(res['seal_sha8'])
assert SEAL_SHA == 'e7f52d03', SEAL_SHA
assert os.path.exists(
    os.path.join(D43, 'design_seal.json'))
assert os.path.exists(
    os.path.join(D43, 'p141_readout.npz'))

# ---- 1. ledger ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
_n0 = len(led['measurements'])
assert _n0 in (279, 280), _n0
_has = any(m.get('phase') == 3143
           for m in led['measurements'])
if not _has:
    assert _n0 == 279, _n0
    entry = {
        'phase': 3143,
        'name': ('omega_p141_d19field_'
                 'readout_topk_newI'),
        'date': '2026-09-29',
        'kind': ('d19field_readout_topk_'
                 'newI'),
        'verdict': V,
        'runtime_s': round(
            float(res['runtime_s']), 1),
        'hashes': {
            'result_sha256_8': res_sha,
            'seal_sha256_8': SEAL_SHA},
        'anchors': {
            'res42_sha8': '7291cfc5',
            'seal42': '449f8161',
            'dvec_sha8': {'17': '5e4c3085',
                          '29': 'ee9484b2',
                          '33': '59fbe0d3',
                          '38': 'aced803b'},
            'dvec19_sha8': '6a0332a6',
            'xphase': 'P1.0/A1.1.0',
            'repro_bit': '5/5',
            'pc1_sign': '+1'},
        'summary': (
            'dvec19 re-captured bit-exact '
            '(sha 6a0332a6, drift 0); '
            'conduction: L19 carrier '
            'downstream field partially '
            'L17-pathway (spec_corr 0.881, '
            'cross-field cos 0.778, direct '
            'cos 0.855/rho 1.898); '
            'readout competition refined: '
            'state additivity cos 1.000, '
            'dh.wdn additive (0.220+6.507='
            '6.726), pc1 readout-dark '
            '(0.220 vs 6.507) -> blocking '
            'is NOT first-order w_dn '
            'projection; top-k curve '
            'non-monotone k25 0.2188 > '
            'full-50 0.1406 -> enrichment '
            'mid-band positive, tail '
            'suppresses; new-material I: '
            'self-retrieval = chance at '
            'all RL (0.0357=1/28), cross '
            'f1 bit-match -> identity '
            'component is old-material '
            'session-specific (NOT a '
            'generic encoding mechanism); '
            'BOS gen-prefix root-caused '
            '(xphase 0->1.0); 5/5 bit '
            'anchors + pc1 sign resolved '
            '+1 by preregistered '
            'anchor.')}
    led['measurements'].append(entry)
    io.open(LEDGER, 'w',
            encoding='utf-8').write(
        json.dumps(led, ensure_ascii=False,
                   indent=1))
led2 = json.load(io.open(LEDGER,
                         encoding='utf-8'))
assert len(led2['measurements']) == 280
assert led2['measurements'][-1][
    'phase'] == 3143
print('1. ledger n=280 OK')

# ---- 2. MEMO append ----
block = '''

## Phase 3143: d19 传导场+读出竞争几何+top-k+新材料I（T4 第26 Phase）[22:40]

**执行**：phase3143_omega_p141_d19field_readout_topk_newI.py；SMOKE 231.4s → 正式跑三段（BOS 根因修复 rev-3143b/c/d）→ 有效 3516s。设计四组：①dvec19 下游传导场（128 行×40 层 base/swap 重建+自洽门+z35 谱软锚+dvec19@L19/dvec17@L17 注入捕获 dose2.0）②读出竞争（2 生成位级锚+pc1/dv29/joint 3 条件 L29 捕获→pc1/w_dn 投影+状态加法 cos）③order_e top-k 滑动 k{5,10,13,15,20,25,30}+co36 全集@L17 A1 dose1.0 ④新材料 I 成分（84 pairs V1+4 变体 prompt 捕获@RL→IDEINT_new 自检索 held-out V1 vs 旧 bank 交叉）。

**rev-3143b/c/d（BOS 根因）**：正式跑 b_pc1 chg 0.4062≠锚 0.203125 且 xphase 0/128 → 根因=3143 `_pad_batch` 丢失 PREFIX_IDS（3142 语义 tok(text) 默认 add_special_tokens=True 产生 2-id BOS 前缀，拼在行首）→ 生成基线整体偏离。patch4 恢复前缀+选择性 ckpt 保留（BOS 无关的 capture 类 dvec19/field/cond19/cond17 保留，gen 类丢弃重算）→ xphase P/A1=1.0（128/128）→ b_pc1 位级复现。**教训：teacher-forced capture 与生成管线的 BOS 语义不同——capture 无前缀（3135 语义，field 自洽 cos=1.0000 证明），生成有前缀（3126 z26 语义）。**

### §1 发现1：L19 载体下游传导=部分 L17 通路
- dvec19 重捕获 sha8=6a0332a6（drift 0.00e+00，跨会话 swap-capture 位级确定）；场重建自洽 cos(field17,dvec17)=1.0000、cos(field19,dvec19)=1.0000；z35 cos_17 谱软锚 max|diff|=0.0102（tol 0.08）。
- dvec19@L19 注入（dose 2.0）下游谱：cos L20 0.9294 单调衰减至 L39 0.5781；dvec17@L17 谱：L20 0.7408 → L29 0.4189。**L19 场全谱强于 L17 场**（rho19 1.93-0.91 vs rho17 1.80-0.84）。
- 场间 cross-cos（dh19 vs dh17）：0.72-0.80 平台（L20-27 近区 median 0.778）；谱 Spearman 0.881；d19 direct（L20-22）cos 0.855/rho 1.898 vs d17 direct 0.796。
- 判决 **d19_path_partial**：谱形高度相关（0.881）但场间 cos 0.778 未达 same 门 0.8——L19 载体下游位移部分走 L17 消费通路、部分独立；幅度上 L19 场更强（行为 chg 0.7891 的几何基础）。

### §2 发现2：blocking 不在状态层也不在一阶读出投影层（×3 强调）
- 状态加法：dh_joint vs dh_pc1+dh_dv29 的 cos=**0.9999981**（128 行 median）——L29 状态级完全线性叠加，无状态破坏（确认 3142 反序不变量）。
- 一阶读出投影加法：dh·w_dn pc1 0.2199 / dv29 6.5074 / joint 6.7260 ≈ 0.2199+6.5074=6.7273（差 0.0013）——线性 yes-no 读出方向上也是加法。
- **pc1_readout_dark**：pc1 注入位移 dh_v1=10.4805≈dh_norm=10.4811（99.99% 在 v1 自身方向）而对 w_dn 投影仅 0.2199——pc1（全局形态方向）对线性答案读出几乎正交；dvec29 位移 63.91 大得多且带 -v1 分量（-20.87）+w_dn 分量（6.51）。
- **行为 blocking（joint 0.4766 < dvec29 单独 0.6406）因此必然发生在 w_dn 一阶投影之后的环节**：logit 空间其他维度经 softmax/argmax 的竞争，或 L30-39 下游动态的非线性混合。3142 的"读出端竞争"精化为"高阶读出竞争（非一阶 w_dn）"。

### §3 发现3：富集 top-k 曲线非单调——尾部坐标净抑制
- k 曲线：k5 0.0859 → k10 0.1094 → k13 0.1562 → k15 0.1328 → k20 0.1641 → **k25 0.2188（峰）** → k30 0.2109 → k50 全集 0.1406。
- **top-25 注入强于全集 1.56×**（0.2188 vs 0.1406）：富集排序 30-50 名坐标注入时净抵消头部贡献——3137 的 k0 负贡献在富集谱系上的对应物：co36 内部混合正/负贡献坐标，头部 25 个近似纯正。
- k*=13（首个 ≥90% 全集；k13≡Q1 位级锚 0.15625 确认）；锚 2/2（co36 d1 0.140625 + Q1 0.15625）。

### §4 发现4：新材料 I 自检索=chance——身份成分是旧材料会话特有（×3 强调）
- 84 新 pairs（ents 28，chance 0.0357）、V1+4 变体模板 prompt 捕获@{17,29,38}；IDEINT_new=4 变体均值−旧 bank 层基座 B_l；held-out V1 查询自检索：3 层全 **0.0357=chance**（1.0×）。
- 交叉检索（bank=旧 IDEINT_P，q=新材料 V1）位级复现 3142 f1（0.0595/0.0357/0.0595）——检索管线验证正确，自检索阴性是真实阴性。
- **判决 iself_session：bank IDEINT 身份成分在新材料中不存在同构对应——身份编码不是通用机制，是旧材料会话特异的痕迹**（结合 3140 V2 flat/3141 gen flat/3142 三重写入阴性：prompt 形式、生成历史、多模板重复、新材料自身构造全部无法产生该结构——指向 62 命题材料相关的权重级记忆）。**指纹范式第 4 级（Compositional/身份成分）在新材料上不可达：身份"指纹"是材料绑定的一次性编码，非可迁移机制。**

### §5 综合 + 3144 预注册
3143 四问四答：(1) L19 传导=部分 L17 通路（场更强）；(2) blocking=高阶读出竞争（状态+一阶投影全加法）；(3) 富集尾部净抑制（k25>全集）；(4) 身份=旧材料会话特有（iself_session）。

3144（Ω-P142）预注册：
1. 高阶读出竞争定位：joint 条件下 L29→L39 逐层 w_dn 投影轨迹（捕捉 blocking 发生的层位）+ logit 级分解（top-k 干扰 token 识别——pc1 挤出的是答案 token 还是格式 token）；
2. co36 正/负贡献坐标分离：k25 峰值集 vs 30-50 尾集的坐标身份分析（与 3137 k0 负贡献集、co50 交叉）+ 尾集单独注入符号检验；
3. 旧材料身份成分的权重级溯源：IDEINT_P 头部坐标（top-25）的 unembed 投影与 3137 坐标基对比——身份方向是否对应特定实体 token 的输出几何；
4. L19 场独立成分：dh19 减去 dh17 在 L17 通路上的投影后的残差场归宿（L19 独立通路去向何层/何方向）。

关键数字：repro 5/5；pc1_sign=+1；stat_add 0.9999981；dh.wdn 0.2199/6.5074/6.7260；dh_v1 pc1 10.4805/dv29 −20.8695；kchg [0.0859,0.1094,0.1562,0.1328,0.1641,0.2188,0.2109]；hit_self 全 0.0357；spec_corr 0.8812；cross_near 0.7777。

锚：result sha8=ce348ff5，seal sha8=e7f52d03，dvec19 sha8=6a0332a6，ledger n=280。'''
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3143' not in t:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(block)
t2 = io.open(MEMO, encoding='utf-8').read()
assert t2.count('## Phase 3143') == 1
assert '3144（Ω-P142）预注册' in t2
print('2. MEMO appended OK')

# ---- 3. workspace daily log ----
p_log = (ROOT + r'\.workbuddy\memory'
         r'\2026-09-29.md')
add = '''

## Phase 3143 (Omega-P141) 闭环 [22:40]
- 正式跑三段完成（BOS 根因 rev-3143b/c/d 修复后有效 3516s）。verdict: a_3142_ok|repro_bit_5|repro_bit_ok|dvec19_repro_6a0332|field_self_ok|z35_cos17_ok|d19_path_partial|d19_direct_hi|stat_add|readout_comp_absent|pc1_readout_dark|head_conc_k13|iself_session|xphase_ok|coverage_full。
- 四发现：①L19 传导=部分 L17 通路（spec_corr 0.881、cross 0.778、L19 场更强）②blocking=高阶读出竞争（stat_add cos 0.9999981、dh.wdn 加法、pc1_readout_dark）③富集 k 曲线非单调、k25 0.2188>全集 0.1406=尾部净抑制④新材料 I 自检索=chance（iself_session，身份=旧材料会话特有）。
- 5/5 位级锚（b_pc1 0.203125 +v1 符号消解、b_dvec29 0.640625、co36 d1、Q1 topk13、f1 cross）；xphase P/A1=1.0；dvec19 重捕获 sha 6a0332a6 drift 0。
- closeout：ledger n=280（res sha8=ce348ff5、seal e7f52d03）+ MEMO（T4 第26Phase）+ 本日志 + MEMORY。'''
t = io.open(p_log, encoding='utf-8').read()
if 'Phase 3143 (Omega-P141) 闭环' not in t:
    io.open(p_log, 'w',
            encoding='utf-8').write(t + add)
t2 = io.open(p_log, encoding='utf-8').read()
assert 'Phase 3143 (Omega-P141) 闭环' in t2
print('3. daily log OK')

# ---- 4. MEMORY.md next-step line ----
p_mem = (ROOT + r'\.workbuddy\memory'
         r'\MEMORY.md')
tm = io.open(p_mem, encoding='utf-8').read()
lines = tm.splitlines()
out_lines = []
changed = False
NEWLINE = ('- max=3143，下一 3144：高阶读出竞争定位（joint 下 L29-39 w_dn 投影轨迹+logit 干扰 token 分解）+ co36 正/负坐标分离（k25 峰集 vs 30-50 尾集×3137 k0 集）+ IDEINT_P top-25 unembed 投影溯源（身份方向是否=实体 token 输出几何）+ dh19 残差场（L17 通路投影后）归宿。**L19 传导=部分 L17 通路且场更强；blocking=高阶（状态/一阶投影全加法，pc1 readout-dark 0.220 vs 6.507）；富集尾部净抑制 k25>全集 1.56×；身份成分=旧材料会话特有（iself_session，新材料自检索 chance）。**')
for l in lines:
    if l.startswith('- max=3142，下一 3143'):
        out_lines.append(NEWLINE)
        changed = True
    else:
        out_lines.append(l)
if changed:
    io.open(p_mem, 'w',
            encoding='utf-8').write(
        chr(10).join(out_lines) + chr(10))
tm2 = io.open(p_mem, encoding='utf-8').read()
assert 'max=3143' in tm2
print('4. MEMORY.md OK (changed=%s)'
      % changed)
print('CLOSEOUT ALL DONE')
