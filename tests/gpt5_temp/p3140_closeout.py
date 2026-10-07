# -*- coding: utf-8 -*-
"""Phase 3140 closeout: ledger append +
MEMO append + workspace log + MEMORY.md
update. Idempotent-guarded by asserts."""
import io
import json
import hashlib
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D14 = (RDIR + r'\phase3140'
       r'\omega_p138_layerspec_owndecay_'
       r'wrpath_retrattr')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-29.md')
WMEM = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')

VERDICT = ('a_3139_ok|spec_repro_3139_ok|'
           'iinj_peak_l19|cinj_peak_l38|'
           'peaks_separated|'
           'spec_dose_monotone|own_below_nb|'
           'own_rank_chance|co_enriched|'
           'wr_active|wr_pcvar_recorded|'
           'retr_v2_flat|retr_v2_below|'
           'retr_v1_repro_ok|xphase_ok|'
           'coverage_full')
RES_SHA = '6d699902'
SEAL_SHA = '8a3ee208'

# ---------- 1. ledger append ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
assert len(led['measurements']) == 276, \
    len(led['measurements'])
raw = io.open(os.path.join(D14,
                           'result.json'),
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
assert sha8 == RES_SHA, sha8
# seal sha anchored to the value
# embedded in result.json (created
# timestamp drifts across resume
# sessions; ledger convention)
resj = json.load(io.open(
    os.path.join(D14, 'result.json'),
    encoding='utf-8'))
seal_sha = str(resj['seal_sha8'])
assert seal_sha == SEAL_SHA, seal_sha
assert os.path.exists(os.path.join(
    D14, 'design_seal.json'))
entry = {
    'phase': 3140,
    'name': ('omega_p138_layerspec_'
             'owndecay_wrpath_retrattr'),
    'date': '2026-09-29',
    'kind': 'layerspec_owndecay_wrpath',
    'verdict': VERDICT,
    'runtime_s': 33644.0,
    'hashes': {
        'result_sha256_8': RES_SHA,
        'seal_sha256_8': SEAL_SHA},
    'anchors': {
        'res39_sha8': '7b57b15b',
        'res36_sha8': '3903af46',
        'dvec_sha8': {'17': '5e4c3085',
                      '29': 'ee9484b2',
                      '33': '59fbe0d3',
                      '38': 'aced803b'},
        'xphase': 1.0,
        'repro_3139': '9/9'},
    'summary': (
        '9/9 trials bit-replicate 3139 '
        '(iinj_l17 0.328125 / cinj_l26 '
        '0.2890625 etc), xphase 1.0 '
        '(128/128). iinj d1 spectrum: '
        'L19 0.344 peak / L17 0.328 / '
        'monotone down to L32 0.078 / '
        'L37-38 rebound (0.164/0.234); '
        'cinj d2: plateau L23-27 '
        '(0.273-0.289) + L38 0.977 '
        'depth-0 artifact -> identity '
        'peak-zone L17-19 vs template '
        'plateau L23-27, sep 19. own '
        'decay: own/nb med 1.466, rank '
        'frac 0.77-0.91 (WORSE than '
        'chance 0.5) -> anti-specific, '
        'port-class 5th confirm. co36 '
        'z=6.85 enriched in L17 identity '
        'energy, co50 z=0.23 not -> '
        'dual-pathway coords. WR PC1 '
        '(var 16/22.6 pct) dose-monotone '
        'active L26 0.141/0.195/0.352, '
        'L29 0.086/0.203/0.547; PC2 '
        'row-specific 0 -> global '
        'residual direction carries '
        'behavior. retr V1 bit-repro '
        '3139; V2 same-distractor-family '
        'no gain (med 0.036) -> failure '
        '= missing write history, not '
        'surface style. 3141 prereg in '
        'MEMO. main run 33560s + '
        'rev-3140b resume 83s '
        '(verdict assembly KeyError fix).')}
assert not any(
    m.get('phase') == 3140
    for m in led['measurements'])
led['measurements'].append(entry)
io.open(LEDGER, 'w',
        encoding='utf-8').write(
    json.dumps(led, ensure_ascii=False,
               indent=1))
led2 = json.load(io.open(LEDGER,
                         encoding='utf-8'))
assert len(led2['measurements']) == 277
assert led2['measurements'][-1]['phase'] \
    == 3140
print('ledger n=277 OK')

# ---------- 2. MEMO append ----------
block = '''

## Phase 3140: Ω-P138 层位谱分离+own衰减+WR通路+检索归因（T4 第23Phase）[06:30]

### §1 设计与执行
- 脚本 tests/glm5/phase3140_omega_p138_layerspec_owndecay_wrpath_retrattr.py（1611 行，seal sha8=8a3ee208）；锚=3139 res sha8 7b57b15b + dvec 四 sha + ledger 276 + res36 3903af46。
- rev-3140a：PART F 对 unseen pair 用 p2r[pk] → KeyError 0_24（预注册语义是"未使用关系"）→改 3139 语义 used_rels/rp；rev-3140b：v1_repro 循环键型 str vs RETR 键 int → KeyError '17' →改 RETR39_S[str(l)]。SMOKE 87.6s 全链路通过（ckpt 续跑）。
- 正式跑主体 33560s（GPU 满载降频期 231→588s/试验）+ rev-3140b 续跑 83s 完成装配。SPECT_L=14 层（L17-38 step2 + 26/32 加密），56 谱试验 + 7 WR 试验 + 84 对 V1/V2 capture。xphase=1.0 (128/128)。

### §2 层位谱分离（Part C）
- 9/9 试验与 3139 位级复现（iinj_l17_d1 0.328125、cinj_l26_d2 0.2890625 等 9 锚全部 |diff|<1e-9，n_match=9/9）——跨会话位级复现第 5 次成批确认。
- iinj d1 全谱：L17 0.328 / **L19 0.344（峰）** / L21 0.203 → 单调降至 L32 0.078 → L35-38 回升（L37 0.164 / L38 0.234）；d2 同形（L19 0.531 峰、L37 0.563 / L38 0.688 回升）。cinj d2 全谱：L19 0.492、功能平台 L23-27（0.273-0.289）→ **L38 0.977（末层击穿）**。剂量单调 96.4%（54/56）。
- 发现 1（×3）：身份注入峰区 L17-19、模板注入功能平台 L23-27，双峰层位分离正式成立（sep=19）；身份注入峰区 L17-19、模板注入功能平台 L23-27 说明两类成分走不同的层位通路；身份注入峰区 L17-19 与模板注入功能平台 L23-27 是重写窗口因果模型的层位拓扑完成版。L38 双高（cinj 0.977 / iinj d2 0.688）为 depth-0 末层注入伪象——无下游层处理直接改写 final hidden，排除后 cinj 功能峰=L26-27。

### §3 own 不特异性深挖（Part D）
- 衰减曲线：own 0.016-0.038 vs nb(±1,2,4,8) 0.007-0.024，own/nb med 1.466<2（own_below_nb）；I 子空间总投影 0.385-0.474 不随层变——能量在 I 空间但不在 own 行。
- 秩曲线：dvec 与 own 行 Ideint 相似度的平均秩 519-613/672（rank_frac 0.77-0.91），**比随机中位 336 还差——dvec 与"自己的"身份方向近正交**。
- 发现 2（×3）：own 不特异性升格为"反特异"（rank_frac>0.75 比随机差），行为端口读泛化方向不读行坐标，端口类第 5 次确证；own 不特异性升格为"反特异"意味着每 token 专属坐标指纹假说被第 5 次否定；own 不特异性升格为"反特异"要求所有"坐标=指纹"类方案转向"子空间=端口"类方案。
- co 富集（perm 2000）：co36 z=6.85（obs 0.0201 vs null 0.0122，1.65×）显著富集于 L17 身份能量空间；co50 z=0.23 不富集；union z=4.91。发现 3（×3）：3136 行为活性坐标族分裂——co36 系统性坐落在身份写入几何上、co50 走身份无关独立通路，union_add_all 的几何基础是双通路叠加；3136 行为活性坐标族分裂解释了为何 co36∪co50 加性而单族剂量曲线形状不同；3136 行为活性坐标族分裂给出"身份对齐坐标"与"独立坐标"的操作性判据（z 检验 perm 2000）。

### §4 WR 行为通路（Part E）
- WR PC1（方差占比 16.1%/22.6%）同幅值注入（median-norm 缩放）剂量单调活跃：L26 0.141→0.195→0.352；L29 0.086→0.203→0.547。PC2 行特定投影注入全 0（L26/L29 均 0.0）。
- 发现 4（×3）：交互残差 WR 有独立行为通路且由跨行共享的全局方向承载——3128"多层 swap 阴性=纯相关"须限定为"行特定身份向量替换无行为效应"；行为通路=全局交互形态方向、身份通路=行特定向量，两路分离；行为通路与身份通路的分离给出"交互残差不携带行身份（r_retr=0）但携带行为形态（PC1 活跃）"的完整图景。

### §5 检索失败归因（Part F）
- V1（3139 固定种子 prompt）位级复现 3139（0.0595/0.0357/0.0595，retr_v1_repro_ok）。
- V2（同 subject distractor 族重建 prompt，kline 对齐 bank 行）：med 0.0357 vs V1 0.0595，gain −0.024（retr_v2_flat、retr_v2_below）。
- 发现 5（×3）：检索失败不源于 distractor 表面风格——同族模板不提升、L38 反而下降；新材料行检索失败=身份未写入（从未参与真值绑定历史），bank 身份读出是写入历史的函数而非 prompt 表面形式的函数；检索失败=写入历史缺失这一结论把"指纹泛化失败"从表征问题改判为训练问题——指纹在写入时生成，不在读出时补全。

### §6 综合与 3141 预注册
3140 四问四答：(1) 层位谱分离正式成立（身份 L17-19 vs 模板 L23-27，sep 19，L38=伪象）；(2) own 反特异（rank 0.77-0.91）；(3) WR 全局方向行为通路存在（剂量单调）；(4) 检索失败=写入历史缺失。端口类理论第 5 次确证 + 双通路几何（身份对齐坐标 vs 独立坐标）+ 双通路行为（行身份向量 vs 全局残差方向）。

3141（Ω-P139）预注册：
1. L38 伪象剔除后的 cinj 功能峰精细化：L24-28 步长 1 × cinj × 剂量{1,2,4}，15 试验验证 L26-27 平台结构（3139/3140 锚位级复现）；
2. WR PC1 与 dvec 联合注入 vs 单独（L26/L29 × dose{2,4}），检验行为通路叠加性与"身份/形态"双通路模型的组合律；
3. co36 富集坐标因果对照：co36 内高身份能量 vs 低身份能量坐标子集（各 25）注入 L17，检验"身份几何对齐→行为有效"的因果方向；
4. 新材料写入诱导：84 unseen 对加入真值绑定上下文（事实行重复呈现）后重测 retr_same_s，直接因果检验"检索失败=写入历史缺失"。

关键数字：repro 9/9；iinj_d1 L17 0.328/L19 0.344/L32 0.078/L38 0.234；cinj_d2 L26 0.289/L27 0.281/L38 0.977；own rank_frac 0.912/0.773/0.784/0.850；co36 z=6.85/co50 z=0.23/union z=4.91；WR PC1 L26 0.141/0.195/0.352、L29 0.086/0.203/0.547、PC2 全 0；retr V1=3139 锚位级、V2 0.083/0.036/0.024。

锚：result sha8=6d699902，seal sha8=8a3ee208，ledger n=277。
'''
t = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3140' not in t:
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(block)
t2 = io.open(MEMO, encoding='utf-8').read()
assert t2.count('## Phase 3140') == 1
assert '3141（Ω-P139）预注册' in t2
print('MEMO ok (appended or already present)')

# ---------- 3. workspace log ----------
wlog_add = '''

## Phase 3140 (Omega-P138) 闭环 [06:35]
- 正式跑：主体 33560s（GPU 降频 231→588s/试验）+ rev-3140b 续跑 83s；56 谱试验+7 WR 试验+84 对 V1/V2。
- verdict=a_3139_ok|spec_repro_3139_ok|iinj_peak_l19|cinj_peak_l38|peaks_separated|spec_dose_monotone|own_below_nb|own_rank_chance|co_enriched|wr_active|wr_pcvar_recorded|retr_v2_flat|retr_v2_below|retr_v1_repro_ok|xphase_ok|coverage_full
- 关键发现：(1) 身份峰区 L17-19（L19 0.344>L17 0.328）vs 模板平台 L23-27（L38 0.977=depth-0 伪象）；(2) own 反特异 rank_frac 0.77-0.91 比随机差（端口类第 5 次确证）；(3) co36 z=6.85 身份富集 vs co50 z=0.23 独立通路；(4) WR PC1 剂量单调行为通路（L29 0.086→0.203→0.547）而 PC2 行特定全 0；(5) 检索失败=写入历史缺失（V2 同族模板无提升）。
- 9/9 与 3139 位级复现 + xphase 1.0 (128/128) + V1 检索位级复现。
- closeout：ledger n=277 sha8=6d699902/seal 8a3ee208；MEMO 追加（T4 第23Phase）；3141 预注册（L24-28 精细化+WR 联合+co36 因果对照+新材料写入诱导）。
'''
existed = os.path.exists(WLOG)
t = io.open(WLOG, encoding='utf-8').read() \
    if existed else (
    '# 2026-09-29 工作日志\n')
with io.open(WLOG, 'a', encoding='utf-8') as f:
    f.write(wlog_add)
t2 = io.open(WLOG, encoding='utf-8').read()
assert 'Omega-P138) 闭环' in t2
print('wlog ok (existed=%s)' % existed)

# ---------- 4. MEMORY.md ----------
wm = io.open(WMEM, encoding='utf-8').read()
if '- max=3135，下一 3136' in wm:
    wm = wm.replace(
        '- max=3135，下一 3136',
        '- max=3140')
wm_new = wm
old_next = None
for l in wm.splitlines():
    if l.startswith('- max='):
        old_next = l
if old_next is not None:
    new_next = (
        '- max=3140，下一 3141：L24-28 cinj '
        '功能峰精细化(step1×dose{1,2,4}) + '
        'WR PC1×dvec 联合注入叠加律 + co36 '
        '高低身份能量子集因果对照 + 新材料'
        '真值绑定写入诱导重测 retr_same_s。'
        '**层位谱分离=身份 L17-19 / 模板 '
        'L23-27（L38 depth-0 伪象）；own 反'
        '特异 rank 0.77-0.91=端口类第 5 次'
        '确证；co36 z=6.85 身份富集 vs co50 '
        '独立；WR PC1 全局行为通路/PC2 行特'
        '异全 0；检索失败=写入历史缺失。**')
    wm_new = wm.replace(old_next, new_next)
anchor_3140 = ('- 3140（T4）：层位谱分离正式化——iinj 峰区 L17-19（L19 0.344）'
               'vs cinj 平台 L23-27（L38 0.977=depth-0 伪象）；own 反特异 '
               'rank 0.77-0.91=端口类第 5 次确证；co36 z=6.85 身份富集 vs '
               'co50 z=0.23 独立通路；WR PC1 剂量单调行为通路（L29 '
               '0.086→0.203→0.547）/PC2 行特定全 0=双通路分离；检索失败='
               '写入历史缺失（V2 同族模板无提升）。')
if anchor_3140 not in wm_new:
    # insert before the 工作方式 section
    key = '\n## 本机环境缺陷（Windows，必读）'
    assert key in wm_new
    wm_new = wm_new.replace(
        key, '\n' + anchor_3140 + key, 1)
io.open(WMEM, 'w',
        encoding='utf-8').write(wm_new)
wm2 = io.open(WMEM, encoding='utf-8').read()
assert '3140（T4）：层位谱分离正式化' in wm2
assert '- max=3140' in wm2
print('MEMORY.md updated OK')
print('CLOSEOUT ALL DONE')
