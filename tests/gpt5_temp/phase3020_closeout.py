# -*- coding: utf-8 -*-
"""Phase 3020 closeout (idempotent): Ledger -> L14 ->
MEMO append -> workspace log -> MEMORY.md rewrite."""
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3020'
     r'\omega_p2n_readout_specificity_qwen')
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
assert verdict == 'injection_readout_asymmetric_qwen', \
    verdict
assert res['anchor_all_ok'] is True
assert res['correction_note'] == ''
t2 = res['T2a']
assert t2['n_logic'] == 11
assert t2['n_content'] == 11
assert t2['n_sham'] == 11
assert t2['lens_cons_med'] == 0.0
assert t2['med_js_final_logic'] == 0.003332
assert t2['med_js_final_content'] == 0.000118
assert t2['med_js_final_sham'] == 0.000143
assert t2['js4_logic'] == 0.05016733
assert t2['js4_content'] == 5.315119e-05
assert t2['inj_ratio'] == 943.861
assert t2['final_ratio'] == 28.306
assert t2['amp_ratio'] == 0.03
assert t2['l_star'] == 4
assert t2['auc_ratio'] == 170.557
t2b = res['T2b']
assert t2b['med_e4']['logic'] == 3.6017
assert t2b['med_e4']['content'] == 0.0268
assert t2b['inj_norm_ratio'] == 134.42
assert t2b['gain_med']['logic'] == 11.294
assert t2b['gain_med']['content'] == 269.147
assert t2b['med_e35']['logic'] == 39.5545
t2c = res['T2c']
assert t2c['med_cos_e35_u35']['logic'] == 0.0471
assert res['T3']['med_drift_s_end'] == 49.5123
assert res['T3']['a13_vs_3009_diff'] == 0.0
assert res['anchors']['a1_axis_diff'] == 0.0
assert res['anchors']['a10_gen_det']['rel'] == 0.0
assert res['anchors']['a23_3019'] is True

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3020
           for m in led['measurements']):
    claim = (
        'Omega-P2n (plan v5 P2) - readout-side '
        'specificity localization: WHERE along depth '
        'does the 30x JS gap open (3019: logic '
        '0.003332 vs content 0.000118 under a '
        'GENERIC suppression field).  Layer-resolved '
        'logit-lens JS trajectory: true residual '
        'entering layer l (decoder-layer pre-hook, '
        'l=4..34) plus final-norm input x_fin (l=35), '
        'final RMSNorm + lm_head at the step-2 '
        'position, JS vs the same-layer baseline; '
        'lens(35) reproduces the true final JS at '
        'cons 0.0 (bit level).  Verdict '
        'injection_readout_asymmetric_qwen.  '
        'RESULTS: (i) the gap is INJECTION-BORN - at '
        'the FIRST divergent layer L4 the immediate '
        'JS is logic 0.0502 vs content 5.3e-05 '
        '(inj_ratio 943.9) and the error norm ||e4|| '
        'is 3.60 vs 0.027 (134x) - erasing g7-K at a '
        'logic position is visible to the model at '
        'the very next step, at a content position '
        'it is nearly invisible; (ii) the downstream '
        'is an EQUALIZER, not an amplifier - the '
        'depth trajectory dilutes 944x back to '
        '28.3x (amp_ratio 0.03): logic JS decays '
        '0.050 -> 0.0033 (the 3019 suppression field '
        'cuts it 15x) while content stays flat; '
        'L-star = 4 (persistent from the first '
        'layer), AUC ratio 170.6; (iii) relative '
        'gain is anti-parallel (content 269x vs '
        'logic 11.3x - small-base amplification) '
        'while absolute ||e35|| stays 39.6 vs 8.0; '
        'cos(e35, u35) 0.047 - the deep error does '
        'not ride the language axis.  CONCLUSION: '
        'the L3 KV gate causal specificity lives at '
        'the SEED-INJECTION end (logic-position g7-K '
        'is the direct load-bearing readout structure '
        'of step 2, linking 3015 K-consumers); the '
        'generic mid-band field then compresses it '
        'symmetrically - gate mechanism = '
        'position-specific seed x position-agnostic '
        'equalizer.  Fourth independent convergence: '
        'single-point operationalization closed, '
        'mechanism is a RELATIONAL property.')
    meas = {
        'meas_id': 'meas3020_omega_p2n_readout_'
                   'specificity_qwen',
        'phase': 3020,
        'claim': claim,
        'verdict': verdict,
        'anchors': '24/24 (a0 2993; a1 axis 0.0; a2 '
                   '2.17e-08; a3 0.0; a4 7.2e-06; a5 '
                   '6.3e-06; a6 rel 0.0; a7 9.95e-14; '
                   'a8 23/23; a9 3007; a10 gen det 0.0; '
                   'a11 3008; a12 3009; a13 T3 drift '
                   'diff 0.0; a14 3010; a15 3011 l*=3; '
                   'a16 3012; a17 3013; a18 3014; a19 '
                   '3015; a20 3016; a21 3017; a22 '
                   '3018; a23 3019)',
        'artifacts': {
            'result': 'phase3020/omega_p2n_'
                      'readout_specificity_qwen/'
                      'result.json',
            'npz': 'phase3020/omega_p2n_'
                   'readout_specificity_qwen/'
                   'omega_p2n_readout_specificity_'
                   'qwen.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': exe['script_sha256_8']},
        'note': 'run1 authoritative, no crashes; '
                'lens consistency 0.0 (bit level); '
                'pre-run self-review fixed 3 script '
                'defects before any execution (T2c '
                'placeholder, uninitialized '
                'anchor-fail branch, format-'
                'expression order).',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 159
    l14['connects'].append({
        'meas_id': 'meas3020_omega_p2n_readout_'
                   'specificity_qwen',
        'phase': 3020,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P2n: layer-resolved '
                        'logit-lens JS localization - '
                        'the 30x readout gap is '
                        'INJECTION-BORN (L4 immediate '
                        'JS 944x, ||e4|| 134x) and the '
                        'downstream is an EQUALIZER '
                        '(dilutes 944x to 28.3x; logic '
                        'JS 0.050 -> 0.0033 = 15x '
                        'suppression, content flat) - '
                        'gate = position-specific seed '
                        'x position-agnostic equalizer'})
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
if '## Phase 3020:' not in memo:
    sec = u'''## Phase 3020: Ω-P2n 读出端特异性定位——30× 差距=注入端特异，下游为均衡器 [%(created)s]

**判决：`injection_readout_asymmetric_qwen`**（run1 权威一次通过，158.3s，锚 24/24：a1 轴身份 0.0 位级、a10 生成确定性 rel 0.0、**a13 T3 漂移 vs 3009 diff 0.0 位级**、a23 3019 完整性；correction_note 空）

### 设计（3019 机器 verbatim + 层解析 logit-lens JS 轨迹）
锚链/几何/生成/位置选择/two-step 协议全继承 3019（SEED_RND=3009 显式重建）。**层解析读出**：真残差（decoder-layer pre-hook）进入层 l（l=4..34）+ final-norm 输入 x_fin（l=35）→ final RMSNorm + lm_head（bf16 全对齐）→ 与同层 baseline 分布的 JS。l=4=首个分歧层（K 擦除作用于 L3 attention 内部）。**一致性门**：lens(35) vs 真终 JS 相对差 med < 0.05——实测 **0.0（位级）**。统计：inj_ratio = med js(4)_logic / js(4)_content；final_ratio = js(35) 比；amp_ratio = final/inj；L-star = 持续 5 层窗 ratio≥5 的首层；AUC 比。判决映射：inj≥5 且 amp<2 → injection_readout_asymmetric；inj<5 且 amp≥2 → amplification_readout_asymmetric；双高 → dual_asymmetric；否则 readout_mixed。

### 核心结果（重复三遍）
**① 注入端已 944×**：首个分歧层 L4 的即时可读性 JS logic **0.0502** vs content **5.3e-05**（**inj_ratio 943.9**），误差范数 ‖e4‖ **3.60 vs 0.027（134×）**——g7-K 在 logic 位被擦除的瞬间，模型在下一步就"看见"；在 content 位几乎不可见；**② 下游=均衡器非放大器**：深度轨迹把 **944× 稀释回 28.3×**（amp_ratio 0.03）——logic 位 JS 0.050→0.0033（3019 抑制场压掉 **15×**），content 平坦（5.3e-05→1.2e-04）；L-star=4（差距自首层持续存在），AUC 比 170.6；**③ 3019 表观矛盾消解**："通用抑制场"是真的——它对两类位置同样做工，但它收到的输入误差本身差 134×；所谓"读出端特异"实为**注入端特异 + 下游对称稀释**；**④ 增益分账**：相对增益 content 269× > logic 11.3×（小基数放大，反平行）；绝对误差 ‖e35‖ logic 39.6 vs content 8.0；cos(e35,u35) logic 0.047≈0——深层误差不沿语言轴。

### 结论
L3 KV 门的因果特殊性在**种子注入端**：logic 位的 g7-K 是 step-2 读出的直接承重结构（衔接 3015 K 消费者），擦除即刻产生 0.05 量级可读扰动；中带通用抑制场随后把它对称压缩回 30× 终差。**门控机制 = "种子（位置特异）× 均衡器（位置无关）"两级架构**——第四个独立收敛：单点操作化关闭，机制=关系属性。

### 硬伤
无崩溃（correction_note 空）。脚本写作期自查出 3 处缺陷并在运行前全部修复：T2c 占位残块（cos 未真正计算）、锚失败分支变量未预初始化、float 与 %% 格式化顺序颠倒——Grep 复核后修正，未消耗运行（3018/3019 的"补丁删块跨界"教训前移到写作期生效）。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3020/omega_p2n_readout_specificity_qwen/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续**：3021 = A（主选）**注入端解剖**——L4 即时 JS 的头级分账（哪些 g7-K 消费者头在 step-2 直接改写分布；衔接 3015 share .390）；B L31 次峰定位；C 情景性检验（同词异位 vs 异词 K,V 相似度）；D 重定向终点测量（K 擦除后注意力质量流向）。
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
wl = WLOG_DIR + r'\2026-09-20.md'
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3020' not in prev:
    line = ('- Phase 3020 Omega-P2n: verdict '
            'injection_readout_asymmetric_qwen; '
            'layer-resolved logit-lens JS trajectory '
            '(res l=4..34 + x_fin l=35, bf16 aligned; '
            'lens cons 0.0 bit level); the 30x gap is '
            'INJECTION-BORN: L4 immediate JS 944x '
            '(logic 0.0502 vs content 5.3e-05), ||e4|| '
            '134x; downstream is an EQUALIZER: dilutes '
            '944x to 28.3x (logic JS 0.050 -> 0.0033 = '
            '15x suppression, content flat), L-star=4, '
            'AUC 170.6; gate = position-specific seed '
            'x position-agnostic equalizer; run1 '
            'clean (158.3s, no crashes); ledger '
            '159/L14 %d.\n'
            % len(l14['connects']))
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
3. 重跑先删旧产物；负结果如实登记；verdict 判据分支内赋值。
4. MEMO 占位符一律 %(key)s 风格（{key} 不替换，2971）。

## 标准锚与精度
- a1 dirs 重建 2.17e-08（跨相位 1e-6）；bit 级仅限同文件链/上游全精度；round 按精度设门；大内积阈 1e-8；跨相位锚 max|Δ| vs 上 Phase。

## 统计判据纪律
- 判据可达性先检（永真禁用）；margin n≳40+粒度；quasi-post-hoc 标注；置换 p 粒度×family 先验，大 family maxT；显著集重叠 null 校准；退化统计量加非退化门；镜像 −dirs 对照必配；功能主张三层分账报效应量。

## 机制解释审计链（命名前依次检查）
norm 分解→口径标签（禁混用）→子空间→SVD 能量迁移→组内混淆→倍率报分子分母→剂量分单调/峰位→消融差分=直接+重平衡。

## 工程规范（Qwen3-4B）
- hidden=2560；头仅 o_proj 输入侧；切片挂 o_proj pre-hook；args 空用 kwargs；单样本保 batch 维；逐词 LS 禁跨词聚合；**真残差流=decoder-layer pre-hook，self_attn pre-hook=post-LN 口径（3017）**；恒等门分母与噪声同尺度。
- logits /√HD；attn 输出 tuple；norm pre-hook 捕 final-norm 输入；**np.stack 单元素→(1,n,hid)（3004）**；per-(l,h) transpose(1,3,0,2)（2970）；管线恒等门；np.where 同形；跨 Phase 常量重建；logit-lens=residual→final RMSNorm→lm_head 全 bf16 对齐，x_fin 位级复现终 JS（3020 cons 0.0）。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；反引号/管道被篡改（chr(96)）。
- 关键写入后必须 Grep/Read 复核磁盘；Edit 幻影→Python 补丁；replace 未命中→先 Grep 再跑；Edit 误插裸文本=语法雷，改后必编译检查（3020 自查 3 缺陷于运行前修复）。
- bf16 batch 组成敏感：跨相位锚 bit 级一致；条件独立 batch 不拼接。
- numpy 标量入 json 转 int()/float()；GQA：KV 缓存 8 头，头级干预单位=KV 头（3015）。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3020）
2938-2970：子空间/词盲/线性壳/阈值/头集中/承重/跳变/秩1/签名/h15/消融/塌缩/峰锁；2972-2991：主调制×词类特化/fr=重写/双轴注入/剂量窗/转正/h12/三要素否定/正交重定向/局部化/perp=重写；2992-3000：符号率快照/签名稳健/轴防御/Ω-F xdir 洗消。3001-3010：GLM4 词携带 86pct vs qwen 63pct（构造 vs 破坏）；中带同构=预训练涌现；3007 锁定 .457；3008 held-out 分离；3009 KV 饱和；3010 换读出→logic 位因果特异。
Ω-P2（3011-3020）：3011 **门控=L3 KV**（D=.0137 p 1e-4，次峰 L31）；3012 混合（mean 回填不恢复）；3013 情景码 LOO .397；3014 剂量非单调（半≥全）K/V 分岔=K 路由；3015 K 消费=领先 g7+情景（share .390）；3016 放大=分布式+深层收敛（门控=种子）；3017 吸收混合（反平行中带 L5-25）；3018 **抵消主导**（credit 2.51>衰减 1.28 nats，MLP 承载带 L8-20）；3019 **抵消带=分布式通用抑制场**（top32 6.7pct p .60；spec 1.12 常开位置无关）。3020 Ω-P2n：**读出特异=注入特异**——L4 即时 JS 944×（logic .0502 vs content 5.3e-05），‖e4‖ 134×；下游把 944× 稀释回 28.3×（logic JS .050→.0033=抑制 15×，content 平坦）——抑制场=均衡器非放大器，门控=位置特异种子×位置无关均衡器。核心：null 重编码全层分布式涌现；头级重要性=关系属性。

## 下一步
- max=3020，下一个 3021（A 主选 **注入端解剖**——L4 即时 JS 的头级分账：哪些 g7-K 消费者头在 step-2 直接改写分布，衔接 3015；B L31 次峰定位；C 情景性检验（同词异位 vs 异词 K,V 相似度）；D 重定向终点）。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
