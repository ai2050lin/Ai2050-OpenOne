# -*- coding: utf-8 -*-
"""Phase 3029 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3029'
     r'\omega_p2w_recruitment_decomp_qwen')
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
assert verdict == 'amplify_existing_dominant_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_sham'] == 11
assert t2['med_js_erase'] == 0.003332
assert t2['med_js_alpha0'] == 0.00117
assert t2['med_js_alpha2'] == 0.014489
assert t2['theta'] == 0.10867
assert t2['med_rho_recruit_share'] == 0.0364
assert t2['med_growth_exponent'] == 0.6714
assert t2['n_tags_rho_new_dominant'] == 0
assert t2['mag2_med'] == 0.515
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a30_capture_self_diff'] == 0.0
assert res['anchors']['a32_dose_alpha0_diff'] == 0.0
assert res['anchors']['a33_3028'] is True
t2b = res['T2b']['bands']
assert t2b['seed']['recruit_share_med'] == 0.0267
assert t2b['mid']['amp_share_med'] == 0.5722
assert t2b['deep']['recruit_share_med'] == 0.0
t2c = res['T2c']
assert t2c['collapse0_content'] == -0.0273
assert t2c['amp2_content'] == 0.9274
assert t2c['med_js_sham'] == 0.0  # DEFECT: recorded
# the capture self-js, not the sham erase js; theta
# unaffected (computed from the sham erase chain).
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a13_t3_drift_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3029
           for m in led['measurements']):
    claim = (
        'Omega-P2w (plan v5 P2) - recruitment band '
        'decomposition at alpha=2: per logic tag the '
        '3028 dose arms alpha in {0, 1, 2} each '
        'capture the step-2 o_proj INPUT per layer '
        '(4096 = 32 query heads x 128, 3016 capture; '
        'observation-only hooks, bit-identity '
        're-verified by a28/a30/a32 = 0.0); '
        'dnorm_a(l,qh) = ||o_a - o_base|| / '
        '(||o_base|| + 1e-12), l in 4..35; noise '
        'floor theta = 95th pct of pooled '
        'sham-position dnorm (11 shams, theta '
        '0.10867); U(a) = {dnorm_a > theta}; '
        'PRIMARY rho = recruited (U2\\U1) share of '
        'the alpha=2 delta energy; growth exponent '
        'g = log2(d2/d1) median over U1&U2.  '
        'Verdict amplify_existing_dominant_qwen '
        '(frozen map: med rho 0.0364 <= 0.2).  '
        'RESULTS: (i) NO head-level recruitment - '
        'med rho 0.0364, 0/11 tags new-dominant, '
        'recruited share ~0 in all bands (deep '
        'exactly 0); (ii) per-unit growth is '
        'SUBLINEAR - med g 0.6714 (d2 ~ 1.6x d1, '
        'not 2x), all 11 tags g < 1; (iii) yet JS '
        'at alpha=2 is 4.35x js1 (3028: 2.18x the '
        'linear prediction) => the convexity '
        'confirmed by 3028 is READOUT-intrinsic: '
        'neither new carriers nor superlinear '
        'carrier growth - the residual->logits '
        'mapping itself converts a ~1.6x '
        'mid-stream signal growth into a ~4.35x '
        'distribution divergence (softmax/logit '
        'competition); (iv) amplified energy '
        'concentrated mid band L8-20 (57.2 pct) + '
        'seed L4-7 (35.2 pct), deep only 4.3 pct; '
        '(v) content side clean (collapse0 -0.027 '
        'as 3024) but rho_c 0.7168 UNRELIABLE '
        '(content deltas far below theta; U2 '
        'near-empty; descriptive caveat).  DEFECT '
        'registered: med_js_sham recorded the '
        'capture self-js (0.0) instead of the sham '
        'ERASE js - calibration stat void; theta '
        'unaffected (computed from the sham erase '
        'chain deltas).  REFINEMENT of 3028: '
        'recruitment happens at the distribution/'
        'readout level, not at the head-carrier '
        'level; 3028 R^2-based "recruitment" claim '
        'is re-anchored as readout convexity.  '
        'NEXT: readout-level convexity probe (per-'
        'layer lens JS growth vs alpha gradient), '
        'per-tag heterogeneity source, L31 '
        'secondary peak, or situational '
        'specificity.')
    meas = {
        'meas_id': 'meas3029_omega_p2w_recruitment_'
                   'decomp_qwen',
        'phase': 3029,
        'claim': claim,
        'verdict': verdict,
        'anchors': '33/33 core (a0-a32 as 3028 '
                   'verbatim incl. a28/a30/a32 '
                   'bit-level 0.0; a33 3028 '
                   'integrity)',
        'artifacts': {
            'result': 'phase3029/omega_p2w_'
                      'recruitment_decomp_qwen/'
                      'result.json',
            'npz': 'phase3029/omega_p2w_'
                   'recruitment_decomp_qwen/'
                   'omega_p2w_recruitment_decomp_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative (159.0s, no '
                'crashes); no head-level recruitment '
                '(med rho 0.0364, deep band 0), '
                'per-unit growth sublinear (med g '
                '0.6714) yet JS 4.35x at alpha=2 => '
                '3028 convexity is READOUT-intrinsic; '
                'amplified energy mid L8-20 57.2 pct '
                '+ seed 35.2 pct; defect: med_js_sham '
                'stat void (recorded self-js), theta '
                'unaffected.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 168
    l14['connects'].append({
        'meas_id': 'meas3029_omega_p2w_recruitment_'
                   'decomp_qwen',
        'phase': 3029,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2w: NO head-level '
                        'recruitment on the coalition '
                        'dose gradient (med rho 0.0364, '
                        '0/11 tags; deep band 0) and '
                        'per-unit growth SUBLINEAR (med '
                        'g 0.6714) - the 3028 convex JS '
                        'response is READOUT-intrinsic '
                        '(residual->logits competition '
                        'turns ~1.6x mid-stream growth '
                        'into ~4.35x distribution '
                        'divergence); amplified energy '
                        'concentrated mid L8-20 (57 '
                        'pct) + seed L4-7 (35 pct); '
                        'endpoints bit-anchored '
                        '(a28/a30/a32 = 0.0)'})
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
if '## Phase 3029:' not in memo:
    sec = u'''## Phase 3029: Ω-P2w 招募带分解——凸增长是读出本征而非单元招募 [%(created)s]

**判决：`amplify_existing_dominant_qwen`**（run1 权威一次通过 159.0s，**无崩溃**，锚 **33/33**：a28 α=1 擦除 vs 3022、a30 捕获自洽、a32 α=0 vs 3024 三重位级 0.0，a33 3028 完整性；correction_note 空）

### 设计（3028 机器 verbatim + 3016 o_proj 捕获）
每个 logic tag 跑 α∈{0,1,2} 三臂（加 capture 共 4 链/tag），逐步捕获每层 o_proj 输入（4096=32 query 头×128）的逐头 delta 能量 dnorm_a(l,qh)=‖o_a−o_base‖/(‖o_base‖+1e-12)，l∈4..35；噪声底线 θ=sham 位置（11 个）擦除链 dnorm 池化 95 分位 = **0.10867**；U(a)={dnorm_a>θ}。主检验 = **招募能量份额 ρ**（U(2)\\U(1) 的 d2² 占 U(2) 总量）+ **逐单元增长指数 g=log2(d2/d1)**（U1∩U2 中位；g=1 线性传播，g>1 超线性）。

### 核心结果（重复三遍）
**① 头级零招募**：med **ρ=0.0364**（3.6%%），0/11 tag 新单元主导；三层带招募份额 seed 2.7%% / mid 0.9%% / **deep 精确 0.0**——α=2 处的额外 JS 完全不来自新单元上线。**② 逐单元响应是亚线性的**：med **g=0.6714**（d2≈1.6×d1，不足 2×），11/11 tag g<1——中带抑制场在阻尼下游响应。**③ 但 α=2 的 JS 是 js1 的 4.35×**（3028：线性预测的 2.18×）——两头夹逼出唯一定位：**3028 的凸性是读出本征的**（residual→logits 映射/softmax 竞争把 ~1.6× 的中游信号增长转换成 ~4.35× 的分布散度），既非新载体也非载体超线性。**④ 放大能量集中**：mid L8-20 **57.2%%** + seed L4-7 35.2%%，deep 仅 4.3%%——扰动在注入带消费、不在深层积累。

### 机制结论与 3028 修正
3028 的"招募"结论需重新锚定：**招募发生在分布/读出层，不在头载体层**。凸性解剖学定位完成：门控凸响应 = 读出映射非线性（与 3020 读出不对称、3027 消费=通用读出结构同一家族）。五级图景：种子（group7）→ L3 正性中继联盟 → 高|s| 群体保护 → 负性均衡场（L8-20，阻尼 g<1）→ **凸读出**（唯一非线性源）。

### 缺陷登记（如实）
med_js_sham 误记录为捕获自洽 js（0.0）而非 sham 擦除 JS——**校准统计作废**；θ 底线不受影响（由 sham 擦除链 delta 计算）；content ρ_c 0.7168 不可靠（content delta 远低于 θ、U2 近空，仅描述性备案）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3029/omega_p2w_recruitment_decomp_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3030 = A（主选）**读出级凸性探针**——逐层 lens JS 增长 vs α 梯度，定位凸性在 norm/lm_head 的哪一段（衔接 3020 L4 即时可读性）；B 逐 tag 异质性来源；C L31 次峰定位；D 情景性检验。
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
if 'Phase 3029' not in prev:
    line = ('- Phase 3029 Omega-P2w: verdict '
            'amplify_existing_dominant_qwen (run1 '
            'authoritative 159.0s, no crashes, '
            'anchors 33/33, a28/a30/a32 bit-level '
            '0.0, a33 3028 integrity); recruitment '
            'band decomposition on the dose '
            'gradient: NO head-level recruitment '
            '(med rho 0.0364, 0/11 tags; deep band '
            'exactly 0), per-unit growth SUBLINEAR '
            '(med g 0.6714, 11/11 g<1) yet JS 4.35x '
            'at alpha=2 => the 3028 convexity is '
            'READOUT-intrinsic (softmax/logit '
            'competition turns ~1.6x mid-stream '
            'growth into ~4.35x divergence); '
            'amplified energy mid L8-20 57.2 pct + '
            'seed L4-7 35.2 pct, deep 4.3 pct; '
            'defect: med_js_sham stat void '
            '(recorded capture self-js), theta '
            '0.10867 unaffected; 3028 "recruitment" '
            're-anchored as readout convexity; '
            'ledger 168/L14 136.\n')
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
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；链身份锚多点（js 序列、js(pb,p0)=0.0、vs 3023/3024 npz、均 0.0；**剂量参数化两端点锚死（α=1=擦除 a28、α=0=恢复 a32），中间点才是新信息**）。

## 统计判据纪律
- 判据可达性先检：置换不变 null p≡1（3021）；消融类预检毒性门（3023）→基线恢复 patch（3024）；宽 patch bf16 噪声底线（3025）；null 后处理写作期预检（3026）；干预参数须真正进链（3027）；剂量方向先于形状检验（3028）；中位数不可加；margin n≳40；maxT；镜像 −dirs 必配；**校准统计必须与底线量同链（3029 js_sham 误记自洽 0.0 作废，θ 不受影响）**。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位（3028 R² 中段准线性→**3029 单元级 g+ρ 分解：凸=读出本征，非单元招募**）→消融差分=直接+竞争重平衡→功能局域≠几何符号身份→读出集中须对照任意扰动 null（3027）。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；真残差流=decoder-layer pre-hook；npz dict/嵌套 dict→0-d 对象数组（读回 .item()，最好扁平化键）；hook 改输出用返回值+active 门；权重列 .detach()；bf16 hook 配 bf16 列；step-2 单 token 取 [0,-1]；clear_cap 每链清→链后立即提取；**位级锚要求 α 分支逐字复刻原表达式（α=0 用 output−d_cur+d_base 原序）**；docstring 内禁裸 \\U。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；triple-quote 内行尾 \\ 吃换行→补丁锚串禁行尾反斜杠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3029）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3029）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018 抵消主导；3019 抵消带=通用抑制场；3020 读出特异=注入特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 正性稀疏联盟（top32=82pct）；3023 零消融有毒；3024 联盟因果承重（collapse 0.657）；3025 非联盟保护性（B1 边缘带）；3026 B1 rank 特异符号 null-like；3027 消费=通用读出结构；3028 剂量凸增长（js2=2.18× 预测）；3029 **凸性=读出本征**（头级 ρ 3.6pct 零招募、g 0.67 亚线性、JS 4.35×；能量 mid 57pct+seed 35pct）。核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性；凸非线性唯一源=读出映射。

## 下一步
- max=3029，下一个 3030（A 主选 **读出级凸性探针**——逐层 lens JS 增长 vs α 梯度定位凸性在 norm/lm_head 哪段、衔接 3020 L4 即时可读性；B 逐 tag 异质性来源；C L31 次峰定位；D 情景性检验）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
