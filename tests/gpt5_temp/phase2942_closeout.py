# -*- coding: utf-8 -*-
"""Phase 2942 closeout: ledger + MEMO + log + MEMORY."""
import hashlib
import json

LED = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
       r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
EXE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2942'
       r'\u8_joint_injection\execution.json')
LOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\2026-09-19.md')
MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
       r'\.workbuddy\memory\MEMORY.md')
RPT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2942_closeout_report.txt')


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

# ---------- ledger ----------
led = json.load(open(LED, encoding='utf-8'))
assert len(led['measurements']) == 80
verdict_text = (
    'u8_displacement_not_causal - forward family 13 s (run2 '
    'authoritative), ANCHORS 8/8: a1 dirs rebuild 2.17e-08 '
    '(17th consecutive forward anchoring); a3 Vt8 vs 2939 '
    'bit-level 0.00e+00; a4 7.21e-06 / a5 6.26e-06 vs 2935 '
    's_base; a7 c3_func vs 2940 0.00e+00; a8 injection '
    'construction 9.95e-14; a6 func sep 185.70. DESIGN: joint '
    'per-word displacement pattern xdir(w) = dcks_S @ Vt8_S '
    'with S = {v1,v2,v5} (91% of U8 mean readout contribution '
    'per 2941 preflight; dcks = 2939 npz coords null0-func), '
    'injected at L16 attn-input pos-1 (2927 site verbatim), '
    'gain sweep s in {1,2,2.5,3,3.5,4,8,16,32}. run1 '
    'correction_note: R2/D1 negative-denominator clamp bug '
    '(max(median,1e-30) clamped -29.06 to 1e-30 -> R2 -6e29) '
    'fixed; grid refined (ratio jumped 0.51@s2 -> 1.82@s4). '
    'CALIBRATION: ratio(s) 0.21/0.86/1.56/1.82/3.40/4.61/6.62 '
    '- propagation gain is superlinear then saturating; s*=2 '
    '(ratio 0.86). MAIN: R1 = 0.7401 < 0.8 -> NOT causal '
    '(run1 at undershooting s: R1 0.7129 - verdict stable '
    'across sessions); R2 = -0.03: median proj35 shift '
    '+0.87 ~= ZERO while sep collapses 185.7 -> 33.03 '
    '(OVERSHOOTS null0 77.26) - CLASS-ASYMMETRIC displacement: '
    'median-dead but group-moving. sep(s) is THRESHOLD-DROP '
    'not gradual: 176.8 (s=1) -> 33.0 (s=2) -> -36.2 (s=3) -> '
    '-51.0 (s=4); rho word-structure 0.94 -> 0.57 -> -0.21. '
    'D1 per-direction propagation: v1 +0.27..+0.47 (positive '
    'gain, ~3x the v3 slope), v5 NEGATIVE -0.19..-0.26 '
    '(anti-restoring), v2 ~+0.05..+0.29. REPRODUCIBILITY '
    'ANOMALY (probe-established): s=2 readouts differ '
    'QUALITATIVELY across sessions (sep 84.8 / 33.0 / 14.2 '
    'in three processes) while bit-level deterministic within '
    'a session (3 repeats + alternation) and s=1, s>=4 are '
    'bit-stable across sessions - the mid-gain regime sits '
    'on an unstable boundary. CONCLUSION: the joint v1/v2/v5 '
    'displacement pattern is a STRONG but UNSTABLE lever, not '
    'a faithful causal replica: injecting the null-like '
    'pattern at ~0.9 matched amplitude drives sep PAST the '
    'null level through a threshold-drop, with per-word '
    'readout shift pattern only rank-0.74 similar and '
    'median-dead. Null re-encoding is not the readout '
    'consequence of a smooth displacement field; it behaves '
    'as a regime switch whose per-word realization cannot be '
    'replayed by residual injection. Lessons: (1) '
    'negative-valued denominators must never be clamped with '
    'max(x, eps) - sign kills; (2) injected forwards in the '
    'mid-gain regime are session-unstable: repeat-within-'
    'session + cross-session spot check required before '
    'trusting a single dose-response point.')
led['measurements'].append({
    'meas_id': 'M2942_u8_joint_injection',
    'type': 'u8_joint_injection',
    'verdict': verdict_text,
    'source': (
        'tests/glm5/phase2942_u8_joint_injection.py '
        'sha256_8=9566179b; outputs execution 89134b07 '
        'result 268af050 npz 55a9107b; sources 2887 %s / '
        '2927 %s / 2935 %s / 2939 %s / 2940 %s'
        % (src['s2887'], src['s2927'], src['s2935'],
           src['s2939'], src['s2940']))})
for lk in led['linkage']:
    if lk['link_id'] == 'L14_readout_spectrum_cross_model':
        assert 'M2942_u8_joint_injection' \
            not in lk['connects']
        lk['connects'].append('M2942_u8_joint_injection')
        n_conn = len(lk['connects'])
        break
json.dump(led, open(LED, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
led_sha = sha8(LED)
report.append('ledger: measurements %d, L14 connects %d, '
              'sha8 %s'
              % (len(led['measurements']), n_conn, led_sha))

# ---------- MEMO ----------
memo_section = '''

## Phase 2942: U8 联合注入与不稳定杠杆判决（u8_joint_injection） [%s]

### 原理与设计
2941 先检指出逐词读出位移主体在 v2/v1/v5（91%% of U8 均值贡献）。2942 检验"联合位移才是因果杠杆"：逐词注入向量 xdir(w) = Σ_{k∈S} dcks[w,k]·Vt8[k]（S = {v1,v2,v5}，dcks = 2939 npz coords null0−func），L16 attn-input pos-1 注入（2927 位置 verbatim），增益扫描 s ∈ {1, 2, 2.5, 3, 3.5, 4, 8, 16, 32}，func 条件 batch57。标定：ratio(s) = median‖c_shift_S‖/median‖dcks_S‖，s* = argmin|ratio−1|；R1 = Spearman(proj35_inj(s*)−proj_func0, dp35_actual)；R2 = median 位移比。判决映射冻结：R1<0.8 → not_causal；R1≥0.8 且 R2∈[0.5,2] → causally_sufficient；否则 magnitude_mismatch。

### correction_note（run1，纪律 3/15 处理）
run1 两缺陷：① R2/D1 负分母 clamp bug——max(median(dp35), 1e-30) 把 −29.06 clamp 成 1e-30（R2 爆至 −6e29）；同款 signed 分母 clamp 污染 D1 slope。修复为直接除法 + |分母|守卫。② 判据可达性（纪律 10 窗口粒度版）：run1 ratio 从 0.508(s=2) 跳至 1.819(s=4)，匹配点落在未测区间——网格细化加入 {2.5, 3, 3.5}。run1 判决（欠匹配点 R1 0.7129）按冻结映射登记；run2 为权威 run。

### 锚（8/8，run2）
a1 2.17e-08（**第 17 次连续前向锚定**）、a2 0.00e+00、a3 Vt8 vs 2939 **bit 级 0**、a4 7.21e-06 / a5 6.26e-06、a6 185.6975、a7 c3_func vs 2940 **0.00e+00**、a8 注入构造自检 9.95e-14。

### 结果（判决 u8_displacement_not_causal）
- **标定**：ratio(s) = 0.21 / 0.86 / 1.56 / 1.82 / 3.40 / 4.61 / 6.62——传导增益超线性后饱和；s* = 2（ratio 0.86）。
- **R1 = 0.7401 < 0.8 → not_causal**（run1 在欠匹配点 R1 0.7129——判决跨 session 稳定）。
- **R2 = −0.03：中位 proj35 位移 +0.87 ≈ 0，但 sep 从 185.7 塌至 33.03（过冲 null0 的 77.26）**——类不对称位移：词级中位死区而组级大幅移动。
- **sep(s) 是陡降型而非渐进型**：176.8 (s=1) → 33.0 (s=2) → −36.2 (s=3) → −51.0 (s=4)；词级结构 rho 0.94 → 0.57 → −0.21 → −0.41。
- **D1 逐方向传导异质**：v1 +0.27..+0.47（正增益，~3× v3 斜率）、**v5 −0.19..−0.26（反传递/anti-restoring）**、v2 +0.05..+0.29。
- **复现性异常（探针确证）**：s=2 读出跨 session 定性不稳定（三进程 sep 84.8 / 33.0 / 14.2），而同 session 内 bit 级确定（3 次重复 + 交替历史后一致），s=1 与 s≥4 跨 session bit 级稳定——中间增益区骑在不稳定边界上。

### 结论
联合 v1/v2/v5 位移模式是**强但不稳定的杠杆，不是忠实的因果复制品**：注入 null 样式模式在 ~0.9 匹配幅度时把 sep 推过 null 水平（陡降/过冲），逐词读出位移模式仅秩相关 0.74 且中位死区。**null 重编码不是平滑位移场的读出后果，而是 regime 开关——其逐词实现无法通过残差注入重放**。2941（v3 单方向阻尼）+ 2942（联合模式陡降过冲 + 类不对称）共同刻画：读出失败的"因果通道"既非杠杆也非阻尼通道，而是不稳定 regime 转换，与其上游证据（2937 gamma 负偏移 + beta 斜率变化、2935 rel/raw 口径反转）自洽。

### 硬伤
- s=2 跨 session 不确定性的根因未定位（kernel 算法选择/归约顺序假说未验证）；匹配点附近读数不可单次采信。
- S 限于 3 基（v1/v2/v5）；全 U8 或含 U8 外成分的注入未测。
- 逐词 dcks 来自 null0 单组；跨 null 组（2939 cos 0.997-1.000）联合模式的稳定性未测。
- 类不对称位移（中位死区/组级移动）的机制未分解（lab0/lab1 各自的位移剖面未入账 npz）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2942_u8_joint_injection.py: 9566179b
- execution.json: 89134b07（created %s）
- result.json: 268af050（final_verdict=u8_displacement_not_causal，runtime 13.0 s）
- u8_joint_injection.npz: 55a9107b
- 源：2887 %s；2927 %s；2935 %s；2939 %s；2940 %s
- 探针 tests/gpt5_temp/phase2942_repro_probe.py（s=2 跨 session 不确定性判定）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2942/u8_joint_injection/
- Ledger：M2942_u8_joint_injection 入账，measurements 80->81，L14 connects 48->49，ledger sha256-8 = %s

### 接续（2943 候选）
- A（主选）：gamma 负偏移解剖（2937 npz 零前向）——2942 陡降/过冲与 2937 gamma 负偏移（null −11.6..−16.1 vs same +3.9）+ beta 斜率变化（0.43-0.55 vs 0.60）拼图：检验 gamma 偏移是否就是"regime 开关"的读出签名。
- B：类不对称位移分解（2942 npz 零前向）——lab0/lab1 各自的 proj35 位移剖面与逐词 dcks 的关系，解释"中位死区但组级移动"。
- C：多层联合注入（L14-L18 同时）或逐层定位（哪一层注入触发陡降）——定位 regime 开关的层位。
- D：承重带跨模型复现（glm4 双口径消融子采样，一次前向）。
''' % (created, created, src['s2887'], src['s2927'],
       src['s2935'], src['s2939'], src['s2940'], led_sha)

memo = open(MEMO, encoding='utf-8').read()
assert '## Phase 2942' not in memo
memo += memo_section
open(MEMO, 'w', encoding='utf-8').write(memo)
report.append('memo appended, total chars %d' % len(memo))

# ---------- workspace log ----------
log_entry = ('''
## Phase 2942 U8 联合注入（%s）
- 判决 u8_displacement_not_causal：联合 v1/v2/v5 位移模式是强但不稳定杠杆——s*=2（ratio 0.86）时 R1 0.7401<0.8、R2 −0.03（中位死区）而 sep 185.7→33.0 过冲 null0 77.26；sep(s) 陡降型（177→33→−36→−51）。
- 重大发现：s=2 中间增益区跨 session 定性不稳定（三进程 sep 84.8/33.0/14.2），同 session bit 级确定；s=1、s≥4 跨 session 稳定。D1：v1 正传导 +0.27..0.47、v5 反传递 −0.19..−0.26。
- run1 correction：R2/D1 负分母 clamp bug + 网格细化（窗口粒度）。锚 8/8（a8 构造自检 9.95e-14）。
- Ledger 81 条 / L14 connects 49 / ledger %s。
- 2943 主选：gamma 负偏移解剖（2937 零前向，regime 开关签名拼图）。
''' % (created, led_sha))
open(LOG, 'a', encoding='utf-8').write(log_entry)
report.append('log appended')

# ---------- MEMORY.md ----------
mem = open(MEM, encoding='utf-8').read()
mem2 = mem.replace(
    'Phase 编号连续（当前 max=2941，下一个 2942）',
    'Phase 编号连续（当前 max=2942，下一个 2943）')
mem2 = mem2.replace(
    '**位移方向所有权 ≠ 因果杠杆（2941 教训）**',
    '**位移方向所有权 ≠ 因果杠杆（2941 教训；2942 延伸：联合位移模式是陡降/分岔型杠杆**——注入 null 样式模式在 ~0.9 匹配幅度时 sep 过冲 null 水平且逐词中位死区（类不对称位移）；中间增益区剂量点跨 session 定性不稳定（同 session bit 级确定）——剂量响应必须做同 session 重复 + 跨 session 关键点抽查，负值分母禁用 max(x, eps) clamp**')
open(MEM, 'w', encoding='utf-8').write(mem2)
report.append('MEMORY.md updated: max=2942 + lesson 17 extended'
              if mem2 != mem else 'MEMORY.md no change')

open(RPT, 'w', encoding='utf-8').write(
    '\n'.join(report) + '\n')
print('\n'.join(report))
