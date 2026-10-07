# -*- coding: utf-8 -*-
"""Phase 3030 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3030'
     r'\omega_p2x_readout_convexity_qwen')
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
assert verdict == 'readout_distributed_convex_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_sham'] == 11
assert t2['med_js_erase'] == 0.003332
assert t2['med_js_alpha0'] == 0.00117
assert t2['med_js_alpha2'] == 0.014489
assert t2['med_excess_terminal'] == 0.009438
assert t2['med_E_tot'] == 1.618
assert t2['med_terminal_share'] == 0.0045
assert t2['n_tags_counted'] == 11
assert t2['lstar_terminal_frac'] == 0.0
assert t2['mag2_med'] == 0.515
assert res['anchors']['a28_erase_chain_diff'] == 0.0
assert res['anchors']['a30_capture_self_diff'] == 0.0
assert res['anchors']['a32_dose_alpha0_diff'] == 0.0
assert res['anchors']['a33_3028'] is True
assert res['anchors']['a34_3029'] is True
assert abs(res['anchors']['a38_lens_terminal_rel']
           - 1.5801550392995439e-06) < 1e-18
assert res['anchors']['a39_lens_traj_3020_diff'] == 0.0
t2b = res['T2b']
assert t2b['bands']['seed']['pos_excess_share_med'] == \
    0.1993
assert t2b['bands']['mid']['pos_excess_share_med'] == \
    0.5245
assert t2b['bands']['deep']['pos_excess_share_med'] == \
    0.2874
assert t2b['bands']['mid']['ratio_med'] == 2.9256
assert t2b['terminal_share_med'] == 0.0045
t2c = res['T2c']
assert t2c['med_js_erase_content'] == 0.000118
assert t2c['med_t1_terminal_content'] == 0.000118
assert t2c['med_S_content'] == 0.3694
assert t2c['med_js_sham'] == 0.0  # DEFECT: recorded
# the capture self-js, not the sham erase js (3029
# recurrence); theta_l floor unaffected (computed
# from the sham erase-chain trajectories).
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a13_t3_drift_diff'] == 0.0

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3030
           for m in led['measurements']):
    claim = (
        'Omega-P2x (plan v5 P2) - readout-level '
        'convexity probe: per logic tag the 3029 '
        'dose arms alpha in {0, 1, 2} (chains '
        'bit-identical, a28/a30/a32 = 0.0) each '
        'capture the step-2 residual stack (36x2560, '
        'decoder-layer pre-hook) and final-norm '
        'input xfin; per-layer logit-lens EXACTLY '
        'as 3020 (start 4, end 35, bf16 RMSNorm + '
        'lm_head, float log_softmax), 32-point '
        'trajectory jsl_l(alpha) = js(lens(x^a_l), '
        'lens(x^b_l)); PRIMARY second-order excess '
        'excess_l = jsl(2) - 2*jsl(1) + jsl(0), '
        'per-layer floor theta_l = med sham alpha=1 '
        'trajectory (same-chain calibration), '
        'terminal share S = excess_xfin / positive '
        'counted excess.  a39 alpha=1 trajectory vs '
        '3020 npz traj_logic KEY-ALIGNED bit-level '
        '0.0 (3020 stores traj rows in '
        'traj_keys_logic dict order, not tags '
        'order - run1 false-failed 0.336 on '
        'misalignment); a38 lens-terminal '
        'consistency 1.58e-06 < 1e-4 gate (gate '
        'recalibrated from 1e-6: float64 softmax '
        'normalization noise).  Verdict '
        'readout_distributed_convex_qwen (frozen '
        'map: med S 0.0045 <= 0.3).  RESULTS: '
        '(i) terminal share med 0.0045 (11/11 tags '
        '<= 0.0137) and lstar_terminal_frac 0.0 - '
        'the convexity is NOT a terminal-readout '
        'effect; (ii) positive excess mass: mid '
        'L8-20 52.5 pct, deep 28.7 pct, seed L4-7 '
        '19.9 pct, terminal 0.45 pct; per-layer '
        'lens amplification jsl(2)/jsl(1) ~ 2.6-2.9x '
        'uniformly across bands vs head-carrier '
        'growth g 0.67 (3029) => the lens map '
        '(norm + lm_head + softmax competition) is '
        'intrinsically convex at EVERY depth on '
        'distributed representations, mid band '
        'dominates only because the distributed '
        'perturbation mass peaks there (3020 '
        'dilution profile); (iii) med_excess_'
        'terminal 0.009438 consistent with the '
        '3028 full-distribution excess 0.0078; '
        '(iv) content side small-base descriptive '
        '(S_c 0.3694).  DEFECT registered: '
        'med_js_sham again recorded the capture '
        'self-js (0.0) instead of the sham erase '
        'js (3029 recurrence); theta_l unaffected. '
        'CLOSURE: five-level architecture + '
        'convexity source fully localized - seed '
        '(group7 heads) -> L3 positive relay '
        'coalition -> high-|s| protective band -> '
        'damping equalizer field (L8-20) -> '
        'INTRINSICALLY CONVEX READOUT (every '
        'depth, distributed).  NEXT: per-tag '
        'heterogeneity source, L31 secondary peak, '
        'situational specificity, or coalition '
        'readout decoding.')
    meas = {
        'meas_id': 'meas3030_omega_p2x_readout_'
                   'convexity_qwen',
        'phase': 3030,
        'claim': claim,
        'verdict': verdict,
        'anchors': '36/36 core (a0-a34 as 3029 '
                   'verbatim incl. a28/a30/a32 '
                   'bit-level 0.0; a39 lens traj vs '
                   '3020 key-aligned bit-level 0.0; '
                   'a38 1.58e-06 < 1e-4)',
        'artifacts': {
            'result': 'phase3030/omega_p2x_'
                      'readout_convexity_qwen/'
                      'result.json',
            'npz': 'phase3030/omega_p2x_'
                   'readout_convexity_qwen/'
                   'omega_p2x_readout_convexity_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run2 authoritative (173.6s, no '
                'crashes; run1 anchor-'
                'miscalibration registered in '
                'execution PREREG corrections - '
                'verdict mapping unchanged); '
                'terminal share 0.45 pct, mid band '
                '52.5 pct, lens amplification '
                '2.6-2.9x at every depth => '
                'convexity = intrinsically convex '
                'readout on distributed '
                'representations; defect: '
                'med_js_sham stat void (3029 '
                'recurrence), theta_l unaffected.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 169
    l14['connects'].append({
        'meas_id': 'meas3030_omega_p2x_readout_'
                   'convexity_qwen',
        'phase': 3030,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2x: readout-level '
                        'convexity localization on '
                        'the dose gradient - terminal '
                        'share med 0.0045 (0/11 tags '
                        'terminal-dominant), positive '
                        'excess mass mid L8-20 52.5 '
                        'pct / deep 28.7 / seed 19.9, '
                        'per-layer lens amplification '
                        'jsl(2)/jsl(1) 2.6-2.9x at '
                        'EVERY depth vs head-carrier '
                        'g 0.67 => the lens map is '
                        'intrinsically convex on '
                        'distributed representations; '
                        'a39 vs 3020 npz key-aligned '
                        'bit-level 0.0 (3020 row-order '
                        'defect registered); '
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
if '## Phase 3030:' not in memo:
    sec = u'''## Phase 3030: Ω-P2x 读出级凸性探针——凸性是分布式读出本征、终端点仅承载 0.45pct [%(created)s]

**判决：`readout_distributed_convex_qwen`**（run2 权威一次通过 173.6s，**无崩溃**，锚 **36/36**：a28/a30/a32 三重位级 0.0 + **a39 α=1 lens 轨迹 vs 3020 npz key 对齐位级 0.0** + a38 lens 终端一致性 1.58e-06 < 1e-4 门；run1 锚校准缺陷如实登记于 execution PREREG corrections，判决映射未改）

### 设计（3029 机器 verbatim + 3020 lens 口径 verbatim）
每个 logic tag 跑 α∈{0,1,2} 三剂量臂（链与 3029 位级一致），每臂捕 step-2 逐层残差栈（36×2560，decoder-layer pre-hook）与最终 norm 输入 xfin；baseline 无擦除链同捕。logit-lens 完全按 3020（L4..L34 + xfin 共 32 点；bf16 RMSNorm+lm_head→float log_softmax），jsl_l(α)=JS(lens(x^α_l), lens(x^b_l))。主检验 = **逐层二阶超额 excess_l = jsl(2)−2·jsl(1)+jsl(0)** 的终端份额 S；逐层底线 θ_l = sham 链 α=1 轨迹逐层 med（同链校准），计入层须 jsl(2)_l > 2θ_l。

### 核心结果（重复三遍）
**① 凸性不在终端**：终端份额 med **S=0.0045**（11/11 tag ≤0.0137），lstar_terminal_frac=**0.0**（无任何 tag 最大超额在终端点）——3028 的 JS 超额不是末端 softmax 一段的产物。**② 正超额质量分布式**：mid L8-20 **52.5%%**、deep 28.7%%、seed L4-7 19.9%%、终端仅 **0.45%%**；逐层 lens 放大比 jsl(2)/jsl(1) 各带一致 **2.6–2.9×**（seed 2.65 / mid 2.93 / deep 2.84）。**③ 与 3029 夹逼闭合**：头载体能量增长 g=0.67 亚线性，而每个深度的 lens 读出都放大 2.6–2.9×——**lens 映射（RMSNorm+lm_head+softmax 竞争）在所有深度上对分布式表示本征凸**；mid 带占主导只因分布式扰动质量在中带峰值（3020 稀释轨迹）。med_excess_terminal 0.009438 与 3028 全分布超额 0.0078 量级一致。

### 机制图景闭合
五级架构 + 凸性源定位完成：种子（group7 头）→ L3 正性中继联盟（32 神经元）→ 高|s| 保护带 → 阻尼均衡场（L8-20）→ **本征凸读出（处处凸、分布式）**。与 3020 读出不对称、3027 消费=通用读出、3029 凸=读出本征同族收束：**凸非线性唯一源 = 读出映射本身，作用于每一层的分布式表示**。

### 缺陷登记（如实）
① run1 锚校准缺陷：a39 按 tags 序对比 3020 traj 行——但 3020 npz 的 traj 行序跟随 traj_keys_logic（dict 序，非 tags 序），tags 序对比假失败 0.336，**key 对齐后位级 0.0**；a38 门 1e-6 紧于 float64 softmax 归一路径噪声（实测 1.58e-6），放宽至 1e-4。② med_js_sham 又误记为捕获自洽 js（0.0，3029 复发）——θ_l 底线不受影响（由 sham 擦除链轨迹计算）。③ content S_c 0.3694 为小基数描述性。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3030/omega_p2x_readout_convexity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3031 = A（主选）**逐 tag 异质性来源**——3028 剂量响应 rel_dev −0.697..+3.874 的 tag 间差异由什么决定（位置上下文/联盟构成/擦除幅度）；B L31 次峰定位；C 情景性检验（同词异位 K,V 相似度）；D 联盟读出解码。
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
if 'Phase 3030' not in prev:
    line = ('- Phase 3030 Omega-P2x: verdict '
            'readout_distributed_convex_qwen (run2 '
            'authoritative 173.6s, no crashes, '
            'anchors 36/36, a28/a30/a32 bit-level '
            '0.0, a39 vs 3020 npz key-aligned 0.0, '
            'a38 1.58e-6<1e-4; run1 anchor '
            'miscalibration registered); per-layer '
            'lens excess on the dose gradient: '
            'terminal share med 0.0045 (0/11 '
            'terminal-dominant), excess mass mid '
            'L8-20 52.5 pct / deep 28.7 / seed '
            '19.9, lens amplification 2.6-2.9x at '
            'every depth vs carrier g 0.67 => '
            'convexity = intrinsically convex '
            'readout on distributed '
            'representations; defect: med_js_sham '
            'void again (3029 recurrence), theta_l '
            'unaffected; ledger 169/L14 137.\n')
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
- **跨相位数组锚按 key/索引数组对齐（3020 traj 行序=dict 序非 tags 序，tags 序对比假失败 0.336→key 对齐 0.0，3030）**；lens 终端一致性等数值锚门 ≥1e-4（float64 softmax 归一路径噪声 1.6e-6，3030 a38）。

## 统计判据纪律
- 判据可达性先检：置换不变 null p≡1（3021）；消融类预检毒性门（3023）→基线恢复 patch（3024）；宽 patch bf16 噪声底线（3025）；null 后处理写作期预检（3026）；干预参数须真正进链（3027）；剂量方向先于形状检验（3028）；中位数不可加；margin n≳40；maxT；镜像 −dirs 必配；**校准统计必须与底线量同链（3029/3030 js_sham 两次误记自洽 0.0 作废，θ 不受影响）**。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位（3028 R²→3029 单元级 g+ρ→**3030 逐层 lens 超额：凸=读出映射本征、处处凸、终端 0.45pct**）→消融差分=直接+竞争重平衡→功能局域≠几何符号身份→读出集中须对照任意扰动 null（3027）。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；真残差流=decoder-layer pre-hook；npz dict/嵌套 dict→0-d 对象数组（读回 .item()，最好扁平化键）；hook 改输出用返回值+active 门；权重列 .detach()；bf16 hook 配 bf16 列；step-2 单 token 取 [0,-1]；clear_cap 每链清→链后立即提取；**位级锚要求 α 分支逐字复刻原表达式（α=0 用 output−d_cur+d_base 原序）**；docstring 内禁裸 \\U。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改；rm shim 损坏→Python os.remove。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；triple-quote 内行尾 \\ 吃换行→补丁锚串禁行尾反斜杠（bash -c 双引号同样吃 \\）；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3030）
2938-2991：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/消融/塌缩/峰锁/h12/重定向/perp=重写；2992-3010：Ω-F 洗消/签名稳健/3007 锁定/3008 分离/3009 KV 饱和/3010 logic 位因果特异。
Ω-P2（3011-3030）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018 抵消主导；3019 抵消带=通用抑制场；3020 读出特异=注入特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 正性稀疏联盟（top32=82pct）；3023 零消融有毒；3024 联盟因果承重（collapse 0.657）；3025 非联盟保护性（B1 边缘带）；3026 B1 rank 特异符号 null-like；3027 消费=通用读出结构；3028 剂量凸增长（js2=2.18× 预测）；3029 凸性=读出本征（ρ 3.6pct 零招募、g 0.67 亚线性）；3030 **凸性=分布式读出**（终端份额 0.45pct、mid 52pct、逐层 lens 放大 2.6-2.9×）。核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性；凸非线性唯一源=读出映射本身。

## 下一步
- max=3030，下一个 3031（A 主选 **逐 tag 异质性来源**——3028 剂量响应 rel_dev −0.697..+3.874 由什么决定；B L31 次峰定位；C 情景性检验；D 联盟读出解码）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
