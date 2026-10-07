# -*- coding: utf-8 -*-
"""Phase 3031 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3031'
     r'\omega_p2y_tag_heterogeneity_qwen')
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
assert verdict == 'hetero_idiosyncratic_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['family'] == ['erase_mag',
                        'restore_asym',
                        'coal_share', 'coal_jac',
                        'mag2', 'recruit_rho',
                        'growth_g', 'band_mid',
                        'readout_lstar']
assert t2['dropped'] == ['pos_frac:zero_var']
assert t2['eligible'] == []
assert t2['winner'] is None
assert t2['primary_confirmed'] is False
assert t2['med_rel_dev'] == 0.886534
assert t2['nperm'] == 200000
assert abs(t2['rho_obs']['readout_lstar']
           + 0.62244) < 1e-9
assert abs(t2['rho_obs']['band_mid']
           - 0.615036) < 1e-9
assert abs(t2['rho_obs']['restore_asym']
           + 0.590909) < 1e-9
assert abs(t2['rho_obs']['erase_mag']
           + 0.527273) < 1e-9
assert abs(t2['p_maxT']['readout_lstar']
           - 0.264144) < 1e-9
assert abs(t2['p_maxT']['band_mid']
           - 0.280804) < 1e-9
assert res['T2b']['med_js_ratio2'] == 3.7731
an = res['anchors']
assert an['a0_source_integrity'] is True
assert an['a1_reldev_identity'] == 0.0
assert an['a2_dose_js2_cross'] == 0.0
assert an['a3_erase_cross'] == 0.0
assert an['a4_alpha0_cross'] == 0.0
assert an['a5_traj_terminal'] < 1e-6
assert an['a6_conc_med_diff'] < 5e-5
assert an['a7_g_rho_recompute'] == {'g': 0.0,
                                    'rho': 0.0}
assert an['a8_tags_identity'] is True
assert an['a9_med_reldev_diff'] < 5e-5

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3031
           for m in led['measurements']):
    claim = (
        'Omega-P2y (plan v5 P2) - per-tag '
        'heterogeneity source: REANALYSIS phase '
        '(CPU only, no model run) over sealed '
        '3028/3029/3022/3020 npz; outcome = stored '
        '3028 rel_dev (authoritative); 10 frozen '
        'candidates (erase_mag, restore_asym, '
        'coal_share, coal_jac, mag2, recruit_rho, '
        'growth_g, pos_frac, band_mid, '
        'readout_lstar), Spearman + exact MC '
        'permutation null (200k, seed 3031) with '
        'maxT family correction at 0.05.  Anchors '
        '10/10: a0 source integrity vs own seals; '
        'a1-a4 cross-artifact bit-level 0.0 (3028 '
        'internal rel_dev identity, 3028-vs-3029 '
        'js2/js_erase/alpha0); a5 3020 traj '
        'key-aligned terminal 1.51e-07; a6 3022 '
        'conc recompute med 0.823621 vs 0.8236; a7 '
        '3029 g_exp/rho recompute bit-level 0.0; '
        'a8 tags identity; a9 med rel_dev vs '
        'recorded 0.8865.  CORRECTION registered '
        '(3028): stored pred2 == 2*js_erase '
        'exactly (js0 term absent vs design text '
        '2*js1-js0); recompute with the alpha=0 '
        'arm gives med rel_dev 1.315 > 0.15 -> '
        '3028 verdict superlinear unchanged.  '
        'Verdict hetero_idiosyncratic_qwen (frozen '
        'map: no candidate p_maxT < 0.05).  '
        'RESULTS: (i) pos_frac degenerate - ALL 11 '
        'logic tags sit at the final prompt token '
        '(pos_frac == 1.0), position context is '
        'void by design, dropped from the family; '
        '(ii) NO candidate survives maxT - '
        'heterogeneity is tag-idiosyncratic at '
        'family-wise 0.05 with n=11; (iii) '
        'exploratory trends (all p_maxT > 0.26): '
        'readout peak layer rho -0.622 (p .264 - '
        'earlier-peaking tags show LARGER convex '
        'excess), midband energy share +0.615 (p '
        '.281), restore asymmetry -0.591, erase '
        'magnitude -0.527 (rel_dev is '
        'scale-dependent: the largest-baseline '
        'tags P0/P7 show the most sublinear rel '
        'growth; js_ratio2 med 3.77x = 2*(1+'
        '0.8865) identity check rho 1.0).  '
        'NEXT: L31 secondary peak localization, '
        'log-log scale reparameterization of the '
        'dose response, situational specificity, '
        'or coalition readout decoding.')
    meas = {
        'meas_id': 'meas3031_omega_p2y_tag_'
                   'heterogeneity_qwen',
        'phase': 3031,
        'claim': claim,
        'verdict': verdict,
        'anchors': '10/10 (a0 integrity; a1-a4 '
                   'bit-level 0.0; a5 1.51e-07; '
                   'a6 2.09e-05; a7 bit-level '
                   '0.0; a8 identity; a9 3.37e-05)',
        'artifacts': {
            'result': 'phase3031/omega_p2y_'
                      'tag_heterogeneity_qwen/'
                      'result.json',
            'npz': 'phase3031/omega_p2y_'
                   'tag_heterogeneity_qwen/'
                   'omega_p2y_tag_heterogeneity_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative (3.2s, no '
                'crashes, first-pass anchors '
                '10/10); 3028 pred2 correction '
                'registered (verdict unchanged); '
                'no preregistered source survives '
                'maxT - tag-idiosyncratic; '
                'exploratory: readout peak -0.62, '
                'midband energy +0.62, erase '
                'magnitude -0.53 (scale '
                'dependence).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 170
    l14['connects'].append({
        'meas_id': 'meas3031_omega_p2y_tag_'
                   'heterogeneity_qwen',
        'phase': 3031,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2y: per-tag '
                        'heterogeneity source '
                        'reanalysis - 9-candidate '
                        'family (pos_frac degenerate: '
                        'all logic tags at final '
                        'token), NO candidate survives '
                        'maxT 0.05 => tag-'
                        'idiosyncratic heterogeneity; '
                        'exploratory trends: readout '
                        'peak layer -0.62, midband '
                        'energy +0.62, erase magnitude '
                        '-0.53 (rel_dev scale '
                        'dependence); 3028 pred2 '
                        'correction registered '
                        '(2*js_erase, verdict '
                        'unchanged); anchors 10/10 '
                        'cross-artifact'})
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
if '## Phase 3031:' not in memo:
    sec = u'''## Phase 3031: Ω-P2y 逐 tag 异质性来源——无单一显著来源、异质性 tag 特异（位置上下文设计性退化为常量）[%(created)s]

**判决：`hetero_idiosyncratic_qwen`**（run1 权威一次通过 3.2s，**无崩溃**，锚 **10/10**：a0 四源封印完整性 + a1–a4 跨产物**位级 0.0**（3028 内部 rel_dev 恒等、3028-vs-3029 js2/js_erase/α0）+ a5 3020 traj key 对齐终端 1.51e-07 + a6 3022 conc 重算 med 0.823621 vs 0.8236 + a7 3029 g_exp/rho 重算位级 0.0 + a8 tags 恒等 + a9 med rel_dev 对记账值 3.37e-05）

### 设计（纯重分析相位，无模型运行）
产出 = 3028 npz 存储 rel_dev（权威，匹配已封印判决）；10 个预注册候选特征：erase_mag / restore_asym / coal_share / coal_jac / mag2 / recruit_rho / growth_g / pos_frac / band_mid / readout_lstar（3022 s_relay→联盟特征、3020 traj key 对齐→lstar、tag 名+tokenizer→位置）。Spearman + 精确 MC 置换 null（200k，seed 3031）+ maxT 家族校正（0.05，n=11 探索性边界如实登记）。

### 核心结果（重复三遍）
**① 无候选者通过 maxT**：最强 readout_lstar ρ=**−0.622**（p .264）、band_mid **+0.615**（p .281）、restore_asym −0.591（p .326）、erase_mag −0.527（p .480）——0/9 存活，**tag 间异质性在家族级 0.05 下是 tag 特异的**，单一 preregistered 来源不存在。**② 位置上下文设计性退化**：全部 11 个 logic tag 都在 prompt 末 token（pos_frac≡1.0）——"位置上下文"作为来源**按设计即为常量**，零方差剔除（tag 名 P{i}:{pos} 的 pos 数只是 prompt 长度差异）。**③ rel_dev 有尺度依赖（探索性）**：erase_mag 负相关——基线最大的 P0/P7（erase 0.184/0.0053）恰是仅有的两个负 rel_dev tag，而 js_ratio2 med **3.77×**=2·(1+0.8865) 恒等自检（rel_dev vs js_ratio2 ρ=1.0）；读出峰位更早的 tag 倾向更大凸超额（lstar −0.62）。

### 机制解读
3028 的 tag 间幅度异质（−0.697..+3.874）**不能归约为任何单一已测特征**：联盟构成（share/jac）、中游放大（mag2）、招募（rho/g）、中带能量、读出峰位在家族校正后全部不显著。凸超额的 tag 间差异要么是未被本次候选覆盖的上下文属性，要么是 rel_dev 尺度参数化本身的伪影（小基线 tag 的比值膨胀）——后者指向**对数尺度重参数化**（log js2 vs log je）作为分离幅度与形状的下一步。与 3027/3029/3030 的"重要性=关系属性"家族一致：**连异质性本身也是关系性的，不锚定在任何单元级/构成级不变量上**。

### 缺陷与修正登记（如实）
**3028 pred2 实现偏差**：npz 存储 pred2 == 2·js_erase **精确成立**（设计文本写 2·js1−js0，js0 项未进入实现）；用 α=0 臂重算 rel_dev med=1.315 > 0.15 → **3028 判决 superlinear 不变**；3031 产出用存储 rel_dev（权威）。med_js_ratio2=3.7731=2·(1+med_rel_dev) 确认存储量自洽。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3031/omega_p2y_tag_heterogeneity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3032 = A（主选）**L31 次峰定位**——3030 深层带 28.7%% 正超额中 P2 的 lstar=32（深层峰）与 3020 遗留的 L31 次峰解剖；B rel_dev 对数尺度重参数化（log js2 vs log je 分离幅度与形状，跟进③）；C 情景性检验（同词异位 K,V 相似度）；D 联盟读出解码。
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
wl = WLOG_DIR + r'\2026-09-21.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3031' not in prev:
    line = ('- Phase 3031 Omega-P2y: verdict '
            'hetero_idiosyncratic_qwen (reanalysis '
            'run1 3.2s, no crashes, anchors 10/10 '
            'incl. a1-a4/a7 cross-artifact bit-level '
            '0.0); 9-candidate family (pos_frac '
            'degenerate - all logic tags at final '
            'token), NO candidate survives maxT '
            '0.05; exploratory: readout peak -0.62, '
            'midband energy +0.62, erase magnitude '
            '-0.53 (rel_dev scale dependence, '
            'js_ratio2 med 3.77x); correction: '
            '3028 pred2 == 2*js_erase exactly (js0 '
            'term absent), recompute med 1.315 -> '
            'verdict unchanged; ledger 170/L14 '
            '138.\n')
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
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；链身份锚多点（js 序列、js(pb,p0)=0.0、跨相位 npz js2/js_erase/α0 均 0.0；剂量两端点锚死 α=1=擦除/α=0=恢复，中间点才是新信息）。
- 跨相位数组锚按 key/索引数组对齐（3020 traj 行序=dict 序）；重分析相位锚=对源 npz/result 记账值重算恒等（a6/a9 门 5e-5）+ 源 seal.json 完整性（a0）。

## 统计判据纪律
- 判据可达性先检：置换不变 null p≡1（3021）；零方差候选先剔（3031 pos_frac）；maxT 家族校正；中位数不可加；margin n≳40（n=11 一律探索性）；镜像 −dirs 必配；校准统计与底线量同链（3029/3030 js_sham 误记教训）；**相对量（rel_dev/比值）有尺度依赖——小基线 tag 比值膨胀（3031），须同时报分子分母并考虑 log-log 重参数化**。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位（3028 R²→3029 单元级→3030 逐层 lens：凸=读出映射本征）→**异质性归因先做候选家族 maxT（3031：0/9 存活=tag 特异；位置上下文须先验证非常量）**→消融差分=直接+竞争重平衡→功能局域≠几何符号身份→读出集中须对照任意扰动 null（3027）。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；真残差流=decoder-layer pre-hook；npz dict/嵌套 dict→0-d（读回 .item()，最好扁平化键）；hook 改输出用返回值+active 门；权重列 .detach()；bf16 hook 配 bf16 列；step-2 单 token 取 [0,-1]；clear_cap 每链清→链后立即提取；位级锚 α 分支逐字复刻原表达式；docstring 内禁裸 \\U。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/行尾反斜杠被篡改（python -c 内联多行易碎→写临时脚本文件）；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3031）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3031）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018-3019 抵消=通用抑制场；3020 读出特异=注入特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 正性稀疏联盟（top32=82pct）；3023 零消融有毒；3024 联盟因果承重（collapse 0.657）；3025-3026 非联盟保护性/B1 符号 null-like；3027 消费=通用读出结构；3028 剂量凸增长（js2=2.18× 预测；pred2 实现=2·js_erase 修正已登记）；3029 凸性=读出本征（ρ 3.6pct、g 0.67）；3030 凸性=分布式读出（终端 0.45pct、mid 52pct、逐层 lens 放大 2.6-2.9×）；3031 **异质性 tag 特异**（0/9 候选过 maxT；pos_frac 常量退化；erase_mag −0.53=尺度依赖，js_ratio2 med 3.77×）。核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性；连异质性也是关系性的。

## 下一步
- max=3031，下一个 3032（A 主选 **L31 次峰定位**——深层带 28.7pct 正超额与 P2 lstar=32 解剖；B rel_dev 对数尺度重参数化 log js2 vs log je；C 情景性检验；D 联盟读出解码）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
