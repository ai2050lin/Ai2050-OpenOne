# -*- coding: utf-8 -*-
"""
Phase 11 技能同步：rdc-main-axis-probe（11->12 臂 / 36->40 坑 / α-4 行与门）
               rdc-phase-closeout（8->10 条教训 / Phase 11 实证 / 参照实现）
逐处 assert count==1 + 回读复核。bytes 级读写，保留原换行。
"""
import io, os

SK = r'C:\Users\Admin\.workbuddy\skills'
P1 = os.path.join(SK, 'rdc-main-axis-probe', 'SKILL.md')
P2 = os.path.join(SK, 'rdc-phase-closeout', 'SKILL.md')


def load(p):
    raw = open(p, 'rb').read()
    nl = '\r\n' if b'\r\n' in raw else '\n'
    return raw, raw.decode('utf-8'), nl


def apply(p, pairs):
    raw, txt, nl = load(p)
    for old, new in pairs:
        old = old.replace('\n', nl)
        new = new.replace('\n', nl)
        assert txt.count(old) == 1, '%s: count=%d for %r' % (os.path.basename(p), txt.count(old), old[:60])
        txt = txt.replace(old, new)
    open(p, 'wb').write(txt.encode('utf-8'))
    back = open(p, 'rb').read().decode('utf-8')
    for old, new in pairs:
        old = old.replace('\n', nl); new = new.replace('\n', nl)
        assert back.count(new) == 1, 'readback missing %r' % (new[:60],)
        assert old not in back, 'old still present %r' % (old[:60],)
    return len(pairs), len(raw), len(back)


# ==================== 1) rdc-main-axis-probe ====================
A4_ROW = ('| **N2h1-α-4 自基全剖面 + 噪声带**（Phase 11） | 逐层累积结论在**噪声带**与**全剖面**下是否仍成立 '
          '| ① α-3 的自基臂 E3 由 4 位点铺满**全部 18 个剖面位点**（`h_ℓ + α·P_U{ℓ}(diff_ℓ)`，同 7 点网格）；'
          '② **零额外前向**——只改落盘：E1/E1b/E3 循环内记录**逐对 `dDonor`（n=24）**'
          '（`E1_pairs`/`E1b_pairs`/`E3_pairs`，F10 断言 `mean(per_pair)==dDonor`）；'
          '③ **配对 percentile bootstrap B=2000**（同一次重采样 `idx_b` 在所有位点/臂共用，只重算 J 再算跨位点 Spearman，'
          'CI=2.5/97.5，`seed=20261001`）；④ **置换零假设 B_perm=2000**（打乱位点标签得 Spearman 的 null 95% 带作阈值校准）；'
          '⑤ 分辨力诊断 B3 = 相邻位点 J 的 95% 区间重叠数；⑥ F8 硬断言 `E3(L6)≡E1(L6)`（因 `U_6≡U6`、`diff_6≡diff6`）、'
          'F9 = E1 对上一 Phase `result` 逐单元格逐位复现 |')

A4_GATE = """**N2h1-α-4 门（qwen3-4b 实测，Phase 11）**：判 **Q2_ESTABLISHED_WITH_BAND** 须三条同时成立：**B1** 绝对/相对剂量 Spearman 的 bootstrap **95% 带上界 < −0.6**；**B2** 18 位点自基剖面 Spearman ≤ −0.6 且 `spread=maxJ/minJ ≥ 3.0`；**F7'** 置换零假设带的 **|界| < 0.6**（否则阈值落在噪声可达域 → 降级 `GATE_POWER_INSUFFICIENT`，B1/B2 只作描述）。**实测**：绝对 `rho=−0.8720` 带 **[−0.9340,−0.7998]**；相对 `rho=−0.9773` 带 **[−0.9876,−0.9092]**；自基 18 位点 `rho=−0.9897` 带 [−0.9979,−0.9525]、`spread=5.587`；置换 null 带 **[−0.4655,+0.4696]**（|界| 0.47 < 0.6 ✅）⇒ **判决 Q2_ESTABLISHED_WITH_BAND**。**B3 分辨力**：相邻位点 J 的 95% 区间 **16/17 重叠**（唯一可分辨对 (L22,L24)）⇒ **剖面趋势可信、逐位点 J 高度不可排序**。**V_ownbasis=BASIS_SENSITIVE(6/18)**：类标签（离散）双基不一致（浅端 S_WEAK→S_STRONG、深端 UNREACH→GRADUAL），但**连续量趋势双基稳健（B2 ✅）** ⇒ 把 α-3 的"趋势稳健/高度不稳健"精确定位为"**连续趋势稳健 / 离散类标签不稳健 / 位点高度不可排序**"。**同位素内建复现**：`E0(ℓ=6,α=1)` 逐位 ≡ Phase 9（10.574739583333335）；**F9 E1 对 `result_phase10` 每个 (site,α) 单元 max|d|=0.000e+00**；F8 `E3(L6)≡E1(L6) max|d|=0`；F10 逐对均值自洽 3.553e-15。"""

PITS = """37. **非线性统计量的判决必须同时给误差带与零假设校准（Phase 11 的判决性一条）**：J 是"最大段斜率 / 其余段斜率中位数"的非线性统计量，**不能**用 delta 方法给区间 ⇒ 必须做**配对 bootstrap**（对发现集 n 个配对做有放回重采样，同一次 `idx_b` 在所有位点/臂共用，重采样后只重算 J）。更关键的是：**只报带不够**——阈值（如 −0.6）可能本来就落在"无效应"的可达域内。必须再跑**置换/打乱标签零假设**（打乱位点标签得 Spearman 的 null 95% 带）证明阈值**确有区分力**。实测 Phase 11：三条真带（绝对 [−0.9340,−0.7998] / 相对 [−0.9876,−0.9092] / 自基 [−0.9979,−0.9525]）上界全 < −0.6，置换 null 带 [−0.4655,+0.4696]（|界| 0.47 < 0.6）⇒ 阈值可判。**若 null 界 ≥ 阈值，判决必须降级为 `GATE_POWER_INSUFFICIENT`，真带只作描述性报告。**
38. **探针族"应由构造决定的位点"必须写成硬断言（免费区分实现缺陷 vs 物理发现）**：若某位点的量**在数学上恒等**于已知量（Phase 11：`E3(L6)` 因 `U_6≡U6`、`diff_6≡diff6` 必须**逐位等于** `E1(L6)`），就把它写成 `floors` 里的**硬断言**。实测 `max|d|=0` —— 它一次同时排除"自基实现写错"和"跨 Phase 回归"。同理，**跨 Phase 复用某位点时**必须加"逐单元格逐位复现上一 Phase result"的闸门（Phase 11 F9：E1 对 `result_phase10` 每个 (site,α) 单元 `max|d|=0.000e+00`，`bad=[]`）—— 这是零成本的装置漂移检测。
39. **逐位点区间高度重叠时只能支撑趋势、不得给位点排序**：非线性统计量（J、x*）在 n=24 下的**逐位点** bootstrap 区间可能大范围重叠（Phase 11：相邻位点 J 的 95% 区间 **16/17 重叠**，唯一不重叠是 (L22,L24)）。⇒ 报告必须把两件事分开：① "剖面**形状**可信"（跨位点 Spearman 的带上界 < −0.6，是**整体趋势**的显著性）；② "位点 J **高度**不可排序"（逐对区间重叠）。**不得**用"某位点 J 更高"支撑机制叙事，除非给出**配对**区间（`J_b(ℓ_i) − J_b(ℓ_{i+1})` 的带不含 0）。
40. **bootstrap 的带宽有边界：只覆盖"配对组成"的不确定性**：配对重采样只回答"换一批同分布实例，结论是否还在"，**不覆盖**网格点选择、面板/模板选择、基选择（固定基 vs 自基）、分类器族选择；n=24 使区间**下界偏乐观**。⇒ 任何"带下仍成立"的表述必须写清边界（"**条件于**当前设计"），并把 V_ownbasis 这类**离散判决**的双基不一致**与**连续量趋势的带**并列报告、不得互相抵消。"""

p1_pairs = [
    ('包含 11 个可复用臂（A 意义分流最小对 / B 层次 margin / C 输入端行替换 / N1b 反嵌入读出 / N1c 完成填空任务依赖 / N2 逐层逐头归因 / N2h1 置换向量消融 / N3 跨结构-跨极性-跨族通用性 / **N2h1-α 写入端组件预算**）、',
     '包含 12 个可复用臂（A 意义分流最小对 / B 层次 margin / C 输入端行替换 / N1b 反嵌入读出 / N1c 完成填空任务依赖 / N2 逐层逐头归因 / N2h1 置换向量消融 / N3 跨结构-跨极性-跨族通用性 / **N2h1-α 写入端组件预算** / **N2h1-α-2 剂量-响应** / **N2h1-α-3 剂量-深度剖面** / **N2h1-α-4 自基全剖面 + 噪声带**）、'),
    ('以及 36 条已实测的坑。', '以及 40 条已实测的坑。'),
    ('## 1 十一个臂（一次跑完，勿拆散）', '## 1 十二个臂（一次跑完，勿拆散）'),
    ('## 4 已实测的坑（36 条，逐条对应数值）', '## 4 已实测的坑（40 条，逐条对应数值）'),
    ('，分 UNREACH / S_STRONG / S_WEAK / GRADUAL / LINEAR |',
     '，分 UNREACH / S_STRONG / S_WEAK / GRADUAL / LINEAR |\n' + A4_ROW),
    ('（Phase 10 实测 **10.574739583333335 ≡ Phase 9**）。',
     '（Phase 10 实测 **10.574739583333335 ≡ Phase 9**）。\n\n' + A4_GATE),
    ('（实测 α∈{0,0.5,1,2,4,8,16}）。',
     '（实测 α∈{0,0.5,1,2,4,8,16}）。\n\n' + PITS),
]

n1, b0, b1 = apply(P1, p1_pairs)

# ==================== 2) rdc-phase-closeout ====================
LESSONS = """9. **closeout 脚本里的字符串格式化/转义坑（Phase 11 实证两条）**：① 含 `%` 的普通字符串被当格式串——`rev_note` 里写 `95% bands` 触发 `ValueError: unsupported format character 'b' (0x62)`，改为 `95%%`；② 脚本里写 `R\\*` 触发 `SyntaxWarning: invalid escape sequence '\\*'`（内容仍正确但污染日志）——用 `chr(92)+'*'` 或原始字符串。⇒ **凡含 `%` 或反斜杠的收尾脚本一律走 Python 文件化补丁、并在修改处 `assert count==1` + 回读**，勿用内联字符串拼接。
10. **跨 Phase 逐位闸门应写进 `floors`（零成本装置漂移检测）**：任何 Phase 只要**复用上一 Phase 的某个位点/向量**，就必须在 `floors` 里加"该位点逐单元格复现上一 Phase `result`"的硬断言。Phase 11 实证：**F9** E1 对 `result_phase10` 每个 `(site,α)` 单元 `max|d|=0.000e+00`（`bad=[]`）、**F8** `E3(L6)≡E1(L6)` `max|d|=0`、**F10** 逐对均值自洽 `max|d|=3.553e-15`、比特锚 `E0=10.574739583333335`、`n6=17.0613`。⇒ 这几条一行代码即可把"装置被外部改动 / 实现回归"挡在判决之前；**没有这条闸门的跨 Phase 复用等于裸奔。**"""

NLINE = ('- N 线独立复核（Phase 11）：`tests/deepseek/Phase11/disk_verify_phase11.py`'
         '（17 分区 A0/A1/B1/B2/B3/C/D/D2/E/F/G/H/I/J/K/L/M，含**同 seed 逐位复现 bootstrap 带与置换 null**），'
         '末尾 `TOTAL FAILS: 0`。')

p2_pairs = [
    ('N 线（deepseek）参照实现：Phase 10（`tests/deepseek/Phase10/` 脚本 + `tests/deepseek_temp/Phase10/` 报告/seal/校验，11 臂 36 坑）',
     'N 线（deepseek）参照实现：Phase 11（`tests/deepseek/Phase11/` 脚本 + `tests/deepseek_temp/Phase11/` 报告/seal/校验，12 臂 40 坑）'),
    ('## N 线（deepseek）Phase 收尾实证（Phase 8 / Phase 9 / Phase 10，2026-10-01）',
     '## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 11，2026-10-01）'),
    ('这条线的收尾链已跑通两次，落地脚本可直接复制（全部在 `tests/deepseek/Phase{N}/`）：',
     '这条线的收尾链已跑通四次（Phase 8/9/10/11），落地脚本可直接复制（全部在 `tests/deepseek/Phase{N}/`）：'),
    ('**Phase 8/9/10 实测的 8 条收尾教训**：', '**Phase 8/9/10/11 实测的 10 条收尾教训**：'),
    ('**不要直接信任 `Read` 的行号**。',
     '**不要直接信任 `Read` 的行号**。\n\n' + LESSONS),
    ('capture_b rho 全量重算 3e-8、6 组 flip 标签重算、ledger sha 重算）',
     'capture_b rho 全量重算 3e-8、6 组 flip 标签重算、ledger sha 重算）\n' + NLINE),
]

n2, c0, c1 = apply(P2, p2_pairs)

print('SKILL1 %d 处 %d -> %d B' % (n1, b0, b1))
print('SKILL2 %d 处 %d -> %d B' % (n2, c0, c1))
print('ALL OK')
