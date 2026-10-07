# -*- coding: utf-8 -*-
"""Phase 3032 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3032'
     r'\omega_p2z_deep_peak_anatomy_qwen')
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
assert verdict == 'deep_peak_head_concentrated_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_sham'] == 11
assert t2['n_deep_counted'] == 10
assert t2['med_head_top8_share'] == 0.8928
assert t2['med_mlp_top32_share'] == 0.1671
assert t2['ldp_layer_per_tag'] == [23, 31, 31, 23,
                                   25, 30, 22, -1,
                                   22, 24, 31]
assert t2['med_js_erase'] == 0.003332
assert t2['mag2_med'] == 0.515
assert t2['gates_ok'] is True
t2b = res['T2b']
assert t2b['med_late_share'] == 0.1493
assert t2b['med_head8_mid'] == 0.8617
assert t2b['med_mlp32_mid'] == 0.2438
assert t2b['med_rho_deep'] == 0.1406
t2c = res['T2c']
assert t2c['med_js_sham'] == 0.000143
assert t2c['a41_ident_med'] == 0.0001
an = res['anchors']
assert an['a0_a35_integrity'] is True
assert an['a1_signature'] == 0.0
assert an['a2_dirs'] < 1e-6
assert an['a3_vt8'] == 0.0
assert an['a4_func'] < 1e-4
assert an['a5_null0'] < 1e-4
assert an['a6_det_rel'] == 0.0
assert an['a7_xdir'] < 1e-9
assert an['a8_lwords'] is True
assert an['a10_gen_det']['ok'] is True
assert an['a13_drift_diff'] == 0.0
assert an['a28_erase_chain_diff'] == 0.0
assert an['a30_capture_self_diff'] == 0.0
assert an['a32_dose_alpha0_diff'] == 0.0
assert an['a38_lens_terminal_rel'] < 1e-4
assert an['a39_traj_3030_diff'] == 0.0
assert an['a40_dnorm_3029_diff'] == 0.0
assert an['a41_s_relay_diff'] < 1e-6

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3032
           for m in led['measurements']):
    claim = (
        'Omega-P2z (plan v5 P2) - deep '
        'secondary-peak anatomy: per logic tag '
        'arms alpha in {0,1,2} (chains bit-'
        'identical to 3029/3030) + baseline '
        'capture, grabbing o_proj input per head, '
        'residual stack, xfin, AND down_proj '
        'input h + MLP output m at L3 and L8-34 '
        '(dedicated observation-only state_mcap '
        'gate; state_hcap stays capture_base-only '
        'so the patch h_base is not overwritten). '
        'PRIMARY: at each tag deep peak layer '
        'ldp = argmax counted positive lens '
        'excess (3030 criteria, idx 17-30): '
        'head_top8_share (alpha=2 arm, top-8 '
        'heads of 32 by delta-o energy) vs '
        'mlp_top32_share (3022 SwiGLU '
        'attribution machine, top-32 of 9728 '
        'neurons).  Anchors: a0-a35 integrity '
        'chain; geometry rebuilds a1-a8 (a2 dirs '
        '2.17e-08); a10/a13 generation '
        'determinism + 3009 drift 0.0; a28 '
        'erase chain vs 3022 bit-level 0.0; a30 '
        '0.0; a32 alpha=0 vs 3024 bit-level '
        '0.0; a38 1.58e-06 < 1e-4; a39 alpha=1 '
        'trajectory vs 3030 npz bit-level 0.0; '
        'a40 per-head dnorm vs 3029 d1/d2/d0 '
        'bit-level 0.0; a41 L3 attribution vs '
        '3022 s_relay 2.33e-08 <= 1e-6 (T2c '
        'ident med 0.0001 == 3022 ident_vals '
        'med).  Verdict '
        'deep_peak_head_concentrated_qwen '
        '(frozen map: med head_top8 0.8928 >= '
        '0.5).  RESULTS: (i) the deep secondary '
        'peak (10/11 tags, ldp spread L22-31) '
        'is HEAD-CONCENTRATED: top-8 of 32 '
        'heads carry med 89.3 pct of the delta-o '
        'energy (per-tag 0.68-0.94), while the '
        'MLP side is distributed - top-32 of '
        '9728 neurons carry only med 16.7 pct; '
        '(ii) the mid excess peak shows the '
        'same contrast (head8 0.862 vs mlp32 '
        '0.244) - head concentration is NOT '
        'deep-specific, the deep peak carrier '
        'resembles the mid readout pattern and '
        'is qualitatively different from the '
        'L3 MLP relay coalition (82 pct in 32 '
        'neurons); (iii) deep recruitment at '
        'alpha=2 within deep layers med rho '
        '0.1406 (conditioned on deep-only, '
        'vs 3029 global 0.036); L31-34 share '
        'of deep excess med 0.1493 (P11 0.51); '
        '(iv) lens decode at ldp reads '
        'pronoun/content continuations (she/'
        'her/could not etc.) - descriptive.  '
        'SYNTHESIS: the five-level picture now '
        'has its carrier map complete - seed '
        '(group7 heads) -> L3 MLP relay '
        'coalition -> protective band -> '
        'damping field -> convex readout, and '
        'BOTH the mid and deep lens-excess '
        'peaks are carried by sparse HEAD '
        'sets, not neurons: the readout-'
        'relevant carriers are heads at every '
        'depth, while L3 is the unique MLP-'
        'relay depth.  NEXT: log-log scale '
        'reparameterization of the dose '
        'response, situational specificity, '
        'or coalition readout decoding.')
    meas = {
        'meas_id': 'meas3032_omega_p2z_deep_peak_'
                   'anatomy_qwen',
        'phase': 3032,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a0-a35 integrity; a1-a8 '
                   'geometry (a2 2.17e-08); a10/'
                   'a13 0.0; a28/a30/a32/a39/a40 '
                   'bit-level 0.0; a38 1.58e-06; '
                   'a41 2.33e-08 (ident med '
                   '0.0001 == 3022)',
        'artifacts': {
            'result': 'phase3032/omega_p2z_'
                      'deep_peak_anatomy_qwen/'
                      'result.json',
            'npz': 'phase3032/omega_p2z_'
                   'deep_peak_anatomy_qwen/'
                   'omega_p2z_deep_peak_anatomy_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run6 authoritative (158.1s; runs '
                '1-3 calibration: 2993 npz path, '
                'MLP capture set L8-34, a32 array '
                'mislabel, sham js self-recording; '
                'run4-5 T2c ident mis-normalized '
                'descriptive only); runs 4-6 npz '
                'bit-identical f8d1215f; deep '
                'peak = top-8 heads 89.3 pct vs '
                'MLP top-32 16.7 pct; mid shows '
                'the same head concentration.',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 171
    l14['connects'].append({
        'meas_id': 'meas3032_omega_p2z_deep_peak_'
                   'anatomy_qwen',
        'phase': 3032,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2z: deep secondary-'
                        'peak anatomy - the deep '
                        'lens-excess peak (L22-31, '
                        '10/11 tags) is carried by '
                        'top-8 of 32 heads (med 89.3 '
                        'pct) while MLP top-32 of '
                        '9728 carry only 16.7 pct; '
                        'mid peak shows the same '
                        'head concentration (0.862 '
                        'vs 0.244) => readout-'
                        'relevant carriers are '
                        'HEADS at every depth, L3 '
                        'is the unique MLP-relay '
                        'depth; anchors incl. a40 '
                        'dnorm vs 3029 and a41 '
                        'attribution vs 3022 '
                        'bit-level/1e-6'})
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
if '## Phase 3032:' not in memo:
    sec = u'''## Phase 3032: Ω-P2z 深层次峰解剖——深层峰由头承载（top-8 头 89pct vs MLP top-32 仅 17pct）[%(created)s]

**判决：`deep_peak_head_concentrated_qwen`**（run6 权威 158.1s，锚全过：a0–a35 完整性链 + 几何重建 a1–a8（a2 dirs 2.17e-08）+ a10/a13 0.0 + **a28/a30/a32/a39/a40 位级 0.0**（擦除链、捕获自洽、α=0 vs 3024、lens 轨迹 vs 3030、逐头 dnorm vs 3029）+ a38 1.58e-06 + **a41 L3 归因 vs 3022 s_relay 2.33e-08**（T2c ident med 0.0001 == 3022 ident_vals med）；run1–3 校准缺陷如实登记于 execution PREREG corrections，run4–6 npz 位级一致 f8d1215f）

### 设计（3030 机器 verbatim + 新增深带 MLP 捕获）
每 tag α∈{0,1,2} 三臂（链与 3029/3030 位级一致）+ 基线捕获；新增观察性捕获：down_proj 输入 h 与 MLP 输出 m（L3 + L8–34，独立 state_mcap 门——**state_hcap 保持仅 capture_base 开启，否则 hook_hcap 会用臂自身 h 覆盖 h_base、α=0/2 patch 被中和**）。主检验：每 tag 深层峰 ldp（3030 计层正超额 argmax，idx 17–30）处 **head_top8 份额**（α=2 臂逐头 Δo 能量 top-8/32）vs **mlp_top32 份额**（3022 SwiGLU 归因机器，9728 神经元 top-32 |s|）。

### 核心结果（重复三遍）
**① 深层峰由头承载**：10/11 tag 有深峰（ldp 散布 L22–31），**top-8 头承载 med 89.3%%** 能量（逐 tag 0.68–0.94），而 **MLP top-32（9728 中）仅 med 16.7%%**——深峰是**头集中**的，与 L3 中继联盟（32 神经元 82%%）构成互补。**② 中带峰同样头集中**（head8_mid 0.862 vs mlp32_mid 0.244）——头集中性**非深层特有**：读出相关的载体在**每个深度都是头**，L3 是唯一的 MLP 中继深度。**③ 深层内招募** med ρ_deep=0.1406（仅深层条件化；3029 全局 0.036）；L31–34 占深带超额 med 0.149（P11 0.51）；lens 解码在 ldp 读出代词/内容延续（she/her/could not 等，描述性）。

### 机制图景（载体地图补全）
种子（group7 头）→ L3 MLP 中继联盟 → 保护带 → 阻尼场 → 凸读出，现在加上**载体地图**：中带与深带的 lens 超额峰都由稀疏**头集合**承载（每深度 top-8/32 ≈ 86–89%%），MLP 在中继之后退为分布式场——**头是读出相关载体的普遍形态，L3 是唯一的 MLP 中继层**。与 3029（零头招募）、3030（凸=读出本征）一致：凸超额的头集中性是读出路径的空间组织，不是新通路招募。

### 缺陷与修正登记（如实）
run1：2993 npz 路径漏子目录。run2：mid 峰对比层不在 MLP 捕获集→扩至 L8–34。run3：**a32 误接捕获自洽 js 数组（全 0）而非 α=0 臂 js**（假失败 1.86e-02 = max(js_restore_coal)，即 3020 式 js0_all/a0_arr 分离丢失）+ med_js_sham 又录自洽 0.0（改录真实 sham 擦除 js 0.000143）。run4–5：T2c ident 公式 ref 漏除 ne2（0.78→9.89，纯描述统计），探针定位 Σs=1.2278 vs 2p_m=1.2276 后修正；a41 与判决全程不受影响。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3032/omega_p2z_deep_peak_anatomy_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3033 = A（主选）**rel_dev 对数尺度重参数化**（log js2 vs log je 分离幅度与形状，跟进 3031 的尺度依赖发现）；B 深峰头集合身份（top-8 头跨 tag Jaccard/GQA 归属，衔接 3032 载体地图）；C 情景性检验（同词异位 K,V 相似度）；D 联盟读出解码。
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
if 'Phase 3032' not in prev:
    line = ('- Phase 3032 Omega-P2z: verdict '
            'deep_peak_head_concentrated_qwen '
            '(run6 authoritative 158.1s, anchors '
            'all pass incl. a28/a30/a32/a39/a40 '
            'bit-level 0.0, a41 vs 3022 2.33e-08, '
            'ident med 0.0001 == 3022; runs 1-3 '
            'calibration registered, runs 4-6 npz '
            'bit-identical); deep lens-excess '
            'peak (L22-31, 10/11 tags) carried by '
            'top-8 of 32 heads med 89.3 pct vs '
            'MLP top-32 16.7 pct, mid peak same '
            'contrast (0.862/0.244) => heads are '
            'the readout carriers at every depth, '
            'L3 the unique MLP-relay depth; '
            'ledger 171/L14 139.\n')
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
- a1 dirs 重建 2.17e-08；跨相位 npz 数组锚位级 0.0（js2/js_erase/α0/traj/dnorm 均验证过）；**数组锚须核对语义数组（3032 run3：a32 误接 js0_all 自洽数组而非 α=0 臂 a0_arr，假失败 1.86e-02=max(对照值)——错误数组对正确数组 diff 恰=对照值量级即嫌疑）**。
- 重分析相位锚=源 seal 完整性+对记账值重算恒等（门 5e-5）；GPU 相位锚=完整 a0–a35 链+跨相位位级。

## 统计判据纪律
- 判据可达性先检；零方差候选先剔（3031 pos_frac）；maxT；n=11 探索性；相对量有尺度依赖（rel_dev/比值，小基线膨胀，报分子分母+log-log 备选）；校准统计与底线量同链（js_sham 自洽 0.0 三次教训，3032 已改录真实 sham 擦除 js）。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→凸响应定位（3030 逐层 lens）→异质性归因候选家族 maxT（3031）→**载体份额须报两通道（3032：深峰 head_top8 89pct vs mlp_top32 17pct；中带同构；头=每深度读出载体，L3 唯一 MLP 中继深度）**→归因 ident 公式 ref/scale 都须 /ne2（3032 run5 教训）→读出集中对照任意扰动 null（3027）。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；真残差流=decoder-layer pre-hook；npz 扁平化键；hook 改输出用返回值+active 门；权重列 .detach()；bf16 hook 配 bf16 列；step-2 单 token [0,-1]；clear_cap 每链清→链后立即提取；**run_chain 必须每链重新 prefill（step-2 前向即使 use_cache=False 也污染 KV cache）**；**新增捕获 hook 用独立 state 门，勿复用 state_hcap（会覆盖 patch 的 h_base 使 α=0/2 patch 中和）**；attrib: s=2·dh·(e@W_down)/‖e‖²，e=进入下一层的残差差。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道/tr 均不可靠；rm 损坏→os.remove；**后台长跑用 run_in_background（`&` 孤儿进程随 shell 会话被杀）**；python -c 内联多行易碎→写临时脚本；局部变量勿遮蔽外层（out=model(...) 覆盖 out 列表，两次踩坑）。
- 关键写入后 Grep/Read 复核磁盘；Edit 幻影→Python 补丁 assert count==1；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3032）
Ω-P2（3011-3032）：3011 门控=L3 KV；3014 剂量非单调=K 路由；3015 K 消费=领先 g7；3018-3019 抵消=通用抑制场；3020 读出特异（944×→28.3×）；3021 注入=MLP 中继 69pct；3022 正性稀疏联盟（top32=82pct）；3023 零消融有毒；3024 联盟因果承重；3025-3026 非联盟保护/B1 符号 null-like；3027 消费=通用读出；3028 剂量凸增长（pred2=2·js_erase 修正）；3029 凸=读出本征（ρ3.6pct/g0.67）；3030 凸=分布式读出（终端 0.45pct/逐层放大 2.6-2.9×）；3031 异质性 tag 特异（0/9 过 maxT；pos_frac 常量；尺度依赖）；3032 **深峰头承载**（深峰 L22-31 头 top-8=89pct vs MLP top-32=17pct；中带同构 0.86/0.24；头=每深度读出载体，L3 唯一 MLP 中继深度）。核心：null 重编码全层分布式涌现；头级/符号重要性=关系属性；载体地图=种子头→MLP 中继→分布式场→头集中读出峰。

## 下一步
- max=3032，下一个 3033（A 主选 **rel_dev 对数尺度重参数化** log js2 vs log je；B 深峰头集合身份（top-8 头跨 tag Jaccard/GQA）；C 情景性检验；D 联盟读出解码）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
