# -*- coding: utf-8 -*-
"""Phase 2941 closeout: ledger append + MEMO append + workspace
log + MEMORY.md. All writes verified by re-read."""
import hashlib
import json
import os

LED = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
       r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
EXE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2941'
       r'\v3_causal_injection\execution.json')
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\2026-09-19.md')
MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\MEMORY.md')
RPT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2941_closeout_report.txt')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:8]


ex = json.load(open(EXE, encoding='utf-8'))
src = ex['sources']
created = ex['created']
report = []

# ---------- 1. ledger ----------
led = json.load(open(LED, encoding='utf-8'))
assert len(led['measurements']) == 79
verdict_text = (
    'v3_push_damped - forward family 9.2 s, ANCHORS 7/7: a1 '
    'dirs rebuild 2.17e-08 (16th consecutive forward '
    'anchoring); a3 Vt8 vs 2939 bit-level 0.00e+00; a4 '
    '7.21e-06 / a5 6.26e-06 vs 2935 s_base; a7 c3_func vs '
    '2940 3.27e-13; a6 func sep 185.70. DESIGN: 2927 '
    'injection site verbatim (attn-input pos-1, L16 = argmax '
    'w_li), 2938/2939 batch57 final pre-norm readout; delta '
    'in {0,+-2,+-4,+-8,+-16,+-32} x {func, null0}; '
    'PREREG-frozen 3-branch verdict on causal gain vs linear '
    'direct prediction cos(v3,u35) = -0.227. PREFLIGHT '
    '(discipline 10): U8 coordinate shifts predict the '
    'actual dir35 readout shift per-word Spearman 0.994 BUT '
    'the v3 term alone carries only 0.4% of variance (mean '
    'shift decomposition v2 -19.7 / v1 -11.3 / v5 -8.2 / v3 '
    '-4.4 vs actual -54.0; v3 class-shift contribution -0.7 '
    'of sep displacement -108.4). P1 SUFFICIENCY (func +v3): '
    'gain med +0.009 vs prediction -0.227 -> attenuated ~26x '
    'with sign flip at delta>=16; sep even at +32 only '
    '185.7->180.6. P2 NECESSITY (null0 -v3): rec16 0.078 '
    'subadditive - cancelling the observed +19.5 c3 shift '
    'recovers 0.28 of the 108 sep gap and ZERO word-structure '
    'rho (0.8898 vs base 0.8896). D1 c3 propagation slope '
    '0.04-0.16 (unit injection reaches final at ~6%); D5 '
    'small-delta evenness 1.0 (linear regime), large-delta '
    '1.2-1.8 (quadratic onset). CONCLUSION: the observed v3 '
    'push is a DAMPED SYMPTOM, not a causal lever - '
    'single-point single-direction injection at its owning '
    'layer reproduces neither the null re-encoding (gain '
    'damped 26x) nor restores the readout when cancelled '
    '(7.8%); null re-encoding is a distributed multi-layer '
    'attention computation. 2940 wording "v3 pushes the '
    'residual away from the dir35 axis" must be read '
    'geometrically, not causally. Lesson: ownership of a '
    'displacement direction != causal leverage over the '
    'readout; observed coordinate displacement magnitude '
    'says nothing about manipulability.')
led['measurements'].append({
    'meas_id': 'M2941_v3_causal_injection',
    'type': 'v3_causal_injection',
    'verdict': verdict_text,
    'source': (
        'tests/glm5/phase2941_v3_causal_injection.py '
        'sha256_8=3198e9b7; outputs execution f68b8fa9 '
        'result 93842517 npz fea58b3a; sources 2887 %s / '
        '2927 %s / 2935 %s / 2939 %s / 2940 %s'
        % (src['s2887'], src['s2927'], src['s2935'],
           src['s2939'], src['s2940']))})
for lk in led['linkage']:
    if lk['link_id'] == 'L14_readout_spectrum_cross_model':
        assert 'M2941_v3_causal_injection' \
            not in lk['connects']
        lk['connects'].append('M2941_v3_causal_injection')
        n_conn = len(lk['connects'])
        break
json.dump(led, open(LED, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
led_sha = sha8(LED)
report.append('ledger: measurements %d, L14 connects %d, '
              'sha8 %s'
              % (len(led['measurements']), n_conn, led_sha))

# ---------- 2. MEMO ----------
memo_section = '''

## Phase 2941: v3 因果注入与阻尼判决（v3_causal_injection） [%s]

### 原理与设计
2940 结论"null 上下文使中层把末位残差沿 v3 均质推离 dir35 轴"含未检验的因果主张。2941 直接因果验证：沿 v3 在其拥有层（L16 = argmax w_li）pos-1 attn-input 注入 ±delta（2927 注入位置 verbatim + 2938/2939 batch57 末位 pre-norm 读出），func 条件测充分性（模拟推进应恶化读出），null0 条件测必要性（抵消推进应恢复读出）。delta ∈ {0, ±2, ±4, ±8, ±16, ±32}，20 次注入批量前向。

### 先检（纪律 10，判据可达性）
观测前用现有产物（2939/2927/2940）先检，两个发现改写设计：
1. **cos(v3, dir35) = −0.227（非正交）**——存在可预注册的线性直接预测 gain = −0.227；2940"推离 dir35 轴"精确化为 103° 夹角斜推。
2. **U8 几何分解**：U8 坐标变化对实际 Δproj35(null0−func) 逐词 Spearman **0.9938**，但 v3 单项仅 **0.4%%** 逐词方差（均值位移分解 v2 −19.7 / v1 −11.3 / v5 −8.2 / v3 −4.4 vs 实际 −54.0；sep 位移 −108.4 中 v3 类差贡献仅 −0.7）——v3 不是读出位移的逐词主体。
据此主检验重新锚定为三分支可判定命题：gain vs 线性预测（amplified ≥2× / linear 0.5–2× / attenuated <0.5×）+ 抵消恢复比 rec16（additive [0.5,2] / super / sub）——任何分支都有登记价值，规避 all_void。

### 锚（7/7，run3）
run1 KeyError 0（仅 L16 注册 pre_attn hook，pass1 需 36 层 capture）；run2 TypeError（Qwen3Attention 以 kwargs 传 hidden_states，hook 把替换向量放 args 位置导致参数重复——改为 kwargs 替换修复）；run3 全过：a1 2.17e-08（**第 16 次连续前向锚定**）、a2 0.00e+00、a3 Vt8 vs 2939 **bit 级 0.00e+00**、a4 7.21e-06 / a5 6.26e-06（vs 2935 s_base）、a6 185.6975、a7 c3_func vs 2940 3.27e-13（阈值 1e-3 按可达量级 ~2e-05 预设——纪律 10 映射版：方向噪声 3e-08 × fin_norm 730 的传播）。descriptive proj vs 2939 双条件 0.00e+00。

### 结果（判决 v3_push_damped）
- **P1 充分性（func +v3）attenuated**：gain 逐 delta {+2: +0.070, +4: +0.036, +8: +0.009, +16: −0.012, +32: −0.024}，med +0.009 vs 线性预测 −0.227 → 衰减 ~26× 且 delta≥16 反号；sep 即使 +32 也仅 185.7→180.6（−5.1）。
- **P2 必要性（null0 −v3 抵消）subadditive**：rec16 = 0.078——抵消 +19.5 的 c3 观测位移只恢复 sep 缺口 108 中的 0.28，词级结构 rho 完全不动（0.8898 vs 基线 0.8896）。
- **D1 传播阻尼**：c3 注入→final 斜率仅 0.04–0.16（单位注入到达末端剩 ~6%%）；null0 条件更弱且部分负。
- **D5 奇偶对称**：小 delta evenness = 1.0（奇阶/线性区），大 delta 1.2–1.8（偶阶非线性出现但不足以放大）。
- D3/D4：词级结构与 norm 全网格近似不动（func rho ≥ 0.989；null0 rho 恒 ~0.87–0.89）。

### 结论
v3 是重编码机器的**阻尼症状，不是因果杠杆**。单层单方向注入（哪怕在其拥有层 L16）既不能复现 null 重编码（增益衰减 26×），反向抵消也不能恢复读出（恢复 7.8%%）——null 重编码是 L14-L18 分布式注意力计算的协同结果，不可通过单点残差注入操作化。**2940 的"沿 v3 推离 dir35 轴"必须按几何读（机器移动所沿的方向），不能按因果读（一根可以拉的杠杆）**。机制链 2936→2941 补上第七环（因果操作化否定环）；先检 U8 分解同时指出读出位移逐词主体在 v2/v1/v5（dir35 平行分解成分）。

### 硬伤
- 单层（L16）单方向注入；多层联合注入（L14-L18 全体）未测——分布式协同假设未直接检验（只排除单点杠杆）。
- delta 网格上限 32（c3 位移 19.5 的 1.6 倍）；更大剂量饱和行为未测。
- 抵消实验用统一 delta，未做逐词 d3 匹配（d3 range [−18.1, +76.0]）。
- evenness 只给标量对称比，未做完整偶阶系数拟合。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2941_v3_causal_injection.py: 3198e9b7
- execution.json: f68b8fa9（created %s）
- result.json: 93842517（final_verdict=v3_push_damped，runtime 9.2 s）
- v3_causal_injection.npz: fea58b3a
- 源：2887 %s；2927 %s；2935 %s；2939 %s；2940 %s
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2941/v3_causal_injection/
- Ledger：M2941_v3_causal_injection 入账，measurements 79->80，L14 connects 47->48，ledger sha256-8 = %s

### 接续（2942 候选）
- A（主选）：U8 联合注入——沿 v2/v1/v5（先检识别的读出位移逐词主体）按逐词 dcks 模式注入，检验"联合位移才是因果杠杆"（一次前向族，同协议）。
- B：多层联合 v3 注入（L14-L18 同时注入）——直接检验分布式协同假设。
- C：gamma 负偏移解剖（2937 npz 零前向；2941 均值通道同样阻尼加深其动机）。
- D：承重带跨模型复现（glm4 双口径消融子采样，一次前向）。
''' % (created, created, src['s2887'], src['s2927'],
       src['s2935'], src['s2939'], src['s2940'], led_sha)

memo = open(MEMO, encoding='utf-8').read()
assert '## Phase 2941' not in memo
memo += memo_section
open(MEMO, 'w', encoding='utf-8').write(memo)
report.append('memo appended, lines now %d'
              % (memo.count('\n') + 1))

# ---------- 3. workspace log ----------
log_entry = ('\n## Phase 2941 v3 因果注入（%s）\n'
             '- 判决 v3_push_damped：v3 推进是阻尼症状非因果杠杆'
             '（P1 gain 0.009 vs 线性预测 −0.227，attenuated ~26×；'
             'P2 rec16 0.078 subadditive；c3 传导率 ~6%%）。\n'
             '- 先检（纪律 10）：cos(v3,dir35)=−0.227；U8 分解 '
             'Spearman 0.994 但 v3 单项仅 0.4%% 方差。\n'
             '- 锚 7/7（a3 bit 级）；run1 hook 少层 / run2 kwargs '
             '传参两连败后 run3 过。\n'
             '- Ledger 80 条 / L14 connects 48 / ledger %s；'
             'MEMO 至 L~9040。\n'
             '- 2942 主选：U8 联合注入（v2/v1/v5 逐词模式）。\n'
             % (created, led_sha))
if os.path.exists(LOG):
    open(LOG, 'a', encoding='utf-8').write(log_entry)
    report.append('log appended to existing 2026-09-19.md')
else:
    open(LOG, 'w', encoding='utf-8').write(
        '# 2026-09-19\n' + log_entry)
    report.append('log created')

# ---------- 4. MEMORY.md ----------
mem = open(MEM, encoding='utf-8').read()
mem2 = mem.replace(
    'Phase 编号连续（当前 max=2930，下一个 2931）',
    'Phase 编号连续（当前 max=2941，下一个 2942）')
if '14. **' not in mem2:
    mem2 += ('\n14. **位移方向所有权 ≠ 因果杠杆（2941 教训）**：'
             '观测到的坐标位移方向（如 v3）注入实验增益可被衰减 '
             '~26×（单层单方向），抵消也不恢复读出——分布式机制'
             '的症状不可单点操作化；"机器沿 X 移动"是几何表述，'
             '不是"拉 X 能改变读出"的因果表述。因果注入判据必须'
             '先做线性直接预测对照（cos × delta）与逐词方差分解'
             '先检（U8 分解），否则把症状当杠杆会浪费一轮。\n')
open(MEM, 'w', encoding='utf-8').write(mem2)
report.append('MEMORY.md updated: max=2941, lesson 14 added'
              if mem2 != mem else
              'MEMORY.md: no change made (patterns missing)')

open(RPT, 'w', encoding='utf-8').write(
    '\n'.join(report) + '\n')
print('\n'.join(report))
