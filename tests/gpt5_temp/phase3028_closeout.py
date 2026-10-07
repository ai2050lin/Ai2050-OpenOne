# -*- coding: utf-8 -*-
"""Phase 3028 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3028'
     r'\omega_p2v_dose_symmetry_qwen')
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
assert verdict == 'dose_superlinear_qwen', verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['med_js_erase'] == 0.003332
assert t2['med_js_alpha0'] == 0.00117
assert t2['med_js_alpha2'] == 0.014489
assert t2['pred2_med'] == 0.006664
assert t2['med_rel_dev'] == 0.8865
assert t2['n_plus2'] == 8
assert t2['p_binom_dev'] == 0.1133
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a30_capture_self_diff'] == 0.0
assert res['anchors']['a32_dose_alpha0_diff'] == 0.0
assert res['anchors']['a31_3024'] is True
t2b = res['T2b']
assert t2b['med_r2_linear_fit'] == 0.7941
assert t2b['mag2_med'] == 0.515, t2b['mag2_med']
t2c = res['T2c']
assert t2c['collapse0_content'] == -0.0273
assert t2c['amp2_content'] == 0.9274
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a13_t3_drift_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3028
           for m in led['measurements']):
    claim = (
        'Omega-P2v (plan v5 P2) - dose symmetry of '
        'the 3022 coalition channel: in the erase '
        'chain the coalition down_proj contribution '
        'is patched to d_base + alpha*(d_cur - '
        'd_base), alphas 0/0.5/1.5/2.0 with alpha=1 '
        '= plain erase; 3024 machine verbatim; '
        'endpoints bit-anchored (a28 alpha=1 vs '
        '3022 js, a32 alpha=0 vs 3024 npz '
        'js_restore_coal, both 0.0; a30 capture '
        'self 0.0).  T2a PRIMARY: linear '
        'extrapolation pred2 = 2*js1 - js0 per tag; '
        'rel_dev = (js2 - pred2)/pred2.  Verdict '
        'dose_superlinear_qwen (frozen map: '
        'med_rel_dev 0.8865 > 0.15).  RESULTS: '
        '(i) the dose response along the coalition '
        'direction is CONVEX / superlinear: med js '
        'at alpha 0/0.5/1/1.5/2 = 0.00117/0.00178/'
        '0.00333/0.00650/0.014489 - at alpha=2 JS '
        'is 2.18x the linear prediction (med_rel_'
        'dev +0.8865; 4.35x vs erase); 10/11 tags '
        'have js2 > js1 (sign test 8/11 above '
        'pred2, p 0.113 - direction consistent, '
        'per-tag magnitudes heterogeneous -0.697 '
        'to +3.874); (ii) per-tag linear fits med '
        'R^2 0.794 - mid-curve near-linear with '
        'convex tail, i.e. RECRUITMENT (new '
        'downstream response engaged at larger '
        'coalition change), not saturation; (iii) '
        'amplification magnitude mag2 = 51.5 pct '
        'of the L3 MLP output norm - the patched '
        'change is non-degenerate; (iv) content '
        'side: collapse0 -0.027 (clean, matches '
        '3024) and alpha2 +92.7 pct relative but '
        'on a 7e-5 base (tiny absolute) - '
        'superlinearity is logic-position '
        'specific.  INTERPRETATION: third dose '
        'point completes 3014: the K-erasure dose '
        'law was non-monotone (routing), the '
        'coalition-channel response is convex '
        '(recruitment) - the relay change does '
        'not propagate linearly; mid-range (0.5-1x)'
        ' behaves quasi-linearly (R^2 0.79) so '
        '3024 single-point restoration '
        'quantification stays valid, but '
        'extrapolation beyond 1x understates the '
        'effect.  NEXT: recruitment band '
        'decomposition at alpha=2 (which '
        'downstream bands newly activate vs '
        'amplify), per-tag heterogeneity source, '
        'L31 secondary peak, or situational '
        'specificity.')
    meas = {
        'meas_id': 'meas3028_omega_p2v_dose_'
                   'symmetry_qwen',
        'phase': 3028,
        'claim': claim,
        'verdict': verdict,
        'anchors': '30/30 core (a0-a27 as 3024; '
                   'a28 alpha=1 erase vs 3022 '
                   'bit-level 0.0; a29 3023; a30 '
                   'capture self 0.0; a31 3024 '
                   'integrity; a32 dose alpha=0 '
                   'vs 3024 npz js_restore_coal '
                   'bit-level 0.0)',
        'artifacts': {
            'result': 'phase3028/omega_p2v_'
                      'dose_symmetry_qwen/'
                      'result.json',
            'npz': 'phase3028/omega_p2v_'
                   'dose_symmetry_qwen/'
                   'omega_p2v_dose_symmetry_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative (153.1s, no '
                'crashes); coalition-channel dose '
                'response is convex/superlinear: '
                'js(2x) = 0.014489 = 2.18x linear '
                'prediction (med_rel_dev +0.8865), '
                '10/11 tags js2 > js1, mid-curve '
                'R^2 0.79 = recruitment not '
                'saturation; content alpha2 +92.7 '
                'pct on 7e-5 base.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 167
    l14['connects'].append({
        'meas_id': 'meas3028_omega_p2v_dose_'
                   'symmetry_qwen',
        'phase': 3028,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2v: coalition-'
                        'channel dose response is '
                        'CONVEX (js at 2x = 2.18x '
                        'linear prediction, '
                        'med_rel_dev +0.8865, 10/11 '
                        'tags js2>js1) = recruitment '
                        'of new downstream response, '
                        'not saturation; mid-curve '
                        'quasi-linear (R^2 0.79) so '
                        '3024 restoration point '
                        'valid; endpoints bit-'
                        'anchored to 3022 and 3024 '
                        '(a28/a32 = 0.0)'})
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
if '## Phase 3028:' not in memo:
    sec = u'''## Phase 3028: Ω-P2v 联盟通道剂量对称性——响应凸增长=招募非饱和 [%(created)s]

**判决：`dose_superlinear_qwen`**（run1 权威一次通过 153.1s，**无崩溃**，锚 **30/30**：a28 α=1 擦除 vs 3022、a30 捕获自洽、**a32 α=0 vs 3024 npz js_restore_coal 三重位级 0.0**，a31 3024 完整性；correction_note 空）

### 设计（3024 机器 verbatim）
联盟通道剂量参数化：擦除链中联盟 32 神经元 down_proj 贡献 patch 到 **d_base + α·(d_cur − d_base)**，α∈{0, 0.5, 1.5, 2}；α=1 即纯擦除链（a28）、α=0 即 3024 恢复链（a32）——**两端点位级锚死，中间点才是新信息**。主检验 = 线性外推 pred2 = 2·js1 − js0 逐 tag 对比 js2。

### 核心结果（重复三遍）
**① 联盟通道剂量响应是凸的（超线性）**：med JS 在 α=0/0.5/1/1.5/2 = **0.00117 / 0.00178 / 0.00333 / 0.00650 / 0.014489**——2× 处 JS 是线性预测（0.006664）的 **2.18×**（med_rel_dev **+0.8865**，vs 擦除 4.35×）；10/11 tag js2 > js1，8/11 高于预测（p 0.113，方向一致但 tag 间幅度异质 −0.697 至 +3.874）。**② 中段准线性、尾部凸起 = 招募而非饱和**：逐 tag 五点线性拟合 med R² **0.794**——0.5–1× 区间近似线性（3024 单点恢复定量化仍有效），>1× 后新下游响应被征募；放大幅度 mag2 = **51.5%%** L3 MLP 输出范数（非退化门过）。**③ 逻辑位特异**：content 位 α=0 collapse −0.027（与 3024 一致，干净），α=2 相对 +92.7%% 但绝对量 7e-05（基数极小）——凸增长是 logic 位现象。

### 机制结论
第三个剂量点补全 3014 图景：**K 擦除本身的剂量律是非单调（路由），联盟通道的响应是凸增长（招募）**——中继变化不线性传播；恢复方向（α<1）准线性而放大方向（α>1）超线性，说明下游存在**只在大扰动时启用的非线性读出储备**。与 3016"分布式放大+深层收敛"一致：招募的是既有分布式场的新参与度，不是新通路。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3028/omega_p2v_dose_symmetry_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3029 = A（主选）**α=2 招募带分解**——用 3016 放大追踪在 α 梯度上定位哪些下游带在新激活 vs 放大（招募的解剖学）；B 逐 tag 异质性来源（rel_dev 与基线量/位置特征的相关）；C L31 次峰定位；D 情景性检验。
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
if 'Phase 3028' not in prev:
    line = ('- Phase 3028 Omega-P2v: verdict '
            'dose_superlinear_qwen (run1 '
            'authoritative 153.1s, no crashes, '
            'anchors 30/30, a28/a30/a32 all '
            'bit-level 0.0; a32 = dose alpha0 vs '
            'sealed 3024 npz js_restore_coal); '
            'coalition-channel dose response is '
            'CONVEX: med js at alpha 0/.5/1/1.5/2 = '
            '0.00117/0.00178/0.00333/0.00650/'
            '0.014489, js(2x) = 2.18x linear '
            'prediction (med_rel_dev +0.8865), '
            '10/11 tags js2>js1 (8/11 above pred, '
            'p 0.113, per-tag heterogeneity -0.697 '
            'to +3.874); mid-curve quasi-linear '
            '(med R^2 0.794) = RECRUITMENT not '
            'saturation; mag2 51.5 pct of L3 MLP '
            'output norm; content alpha2 +92.7 pct '
            'on 7e-5 base (logic-specific '
            'superlinearity); completes 3014 dose '
            'picture: K-erasure non-monotone '
            '(routing) vs coalition-channel convex '
            '(recruitment); ledger 167/L14 135.\n')
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
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；链身份锚多点（js 序列、js(pb,p0)=0.0、vs 3023 js_abl_only / js_coal_content、3024 js_restore_coal、均 0.0；**剂量参数化两端点锚死（α=1=擦除 a28、α=0=恢复 a32），中间点才是新信息**）。

## 统计判据纪律
- 判据可达性先检：置换不变 null p≡1（3021）；消融类预检毒性门（3023）→基线恢复 patch（3024）；宽 patch bf16 噪声底线（3025）；null 后处理写作期预检（3026）；干预参数须真正进链（3027）；**剂量方向/单调性先于形状检验（3028 判决树 direction→nonmonotonic→linear/super/sub）**；中位数不可加；margin n≳40；maxT；镜像 −dirs 必配。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→**凸响应区分招募 vs 饱和（中段 R² 准线性=招募，3028）**→消融差分=直接+竞争重平衡→功能局域≠几何符号身份→读出集中须对照任意扰动 null（3027）。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；真残差流=decoder-layer pre-hook；npz dict→0-d 读回 .item()；hook 改输出用返回值+active 门；权重列 .detach()；bf16 hook 配 bf16 列；step-2 单 token 取 [0,-1]；clear_cap 每链清→链后立即提取；**位级锚要求 α 分支逐字复刻原表达式（α=0 用 output−d_cur+d_base 原序，不可代数化简）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；triple-quote 内行尾 \\ 吃换行→补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3028）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3028）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018 抵消主导；3019 抵消带=通用抑制场；3020 读出特异=注入特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 正性专属稀疏联盟（top32=82pct）；3023 零消融有毒；3024 联盟因果承重（collapse 0.657）；3025 非联盟变化保护性（B1 边缘带）；3026 B1 rank 特异但符号 null-like；3027 消费=通用读出结构（头读出=上下文属性）；3028 **联盟通道剂量响应凸增长=招募**（js 2×=2.18× 线性预测 mrd+0.8865，10/11 js2>js1，中段 R² 0.794，mag2 51.5pct；content 特异）。核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性。

## 下一步
- max=3028，下一个 3029（A 主选 **α=2 招募带分解**——3016 放大追踪在 α 梯度上定位哪些下游带新激活 vs 放大；B 逐 tag 异质性来源；C L31 次峰定位；D 情景性检验）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
