# -*- coding: utf-8 -*-
"""技能同步第二步：(a) 精确复核 rdc-main-axis-probe；(b) 应用 rdc-phase-closeout（修正 readback 逻辑）。"""
import io, os

SK = r'C:\Users\Admin\.workbuddy\skills'
P1 = os.path.join(SK, 'rdc-main-axis-probe', 'SKILL.md')
P2 = os.path.join(SK, 'rdc-phase-closeout', 'SKILL.md')


def load(p):
    raw = open(p, 'rb').read()
    return raw, raw.decode('utf-8'), ('\r\n' if b'\r\n' in raw else '\n')


# ---------- (a) 精确复核 P1 ----------
_, t1, _ = load(P1)
must1 = {
    '包含 12 个可复用臂': 1, '包含 11 个可复用臂': 0,
    '以及 40 条已实测的坑。': 1, '以及 36 条已实测的坑。': 0,
    '## 1 十二个臂（一次跑完，勿拆散）': 1, '## 1 十一个臂': 0,
    '## 4 已实测的坑（40 条，逐条对应数值）': 1, '（36 条，逐条对应数值）': 0,
    '**N2h1-α-4 自基全剖面 + 噪声带**（Phase 11）': 1,
    '**N2h1-α-4 门（qwen3-4b 实测，Phase 11）**': 1,
    '37. **非线性统计量的判决必须同时给误差带与零假设校准': 1,
    '38. **探针族"应由构造决定的位点"必须写成硬断言': 1,
    '39. **逐位点区间高度重叠时只能支撑趋势、不得给位点排序**': 1,
    '40. **bootstrap 的带宽有边界：只覆盖"配对组成"的不确定性**': 1,
    'N2h1-α-4 自基全剖面 + 噪声带**（Phase 11） | 逐层累积结论在**噪声带**与**全剖面**下是否仍成立': 1,
}
bad1 = [(k, t1.count(k), v) for k, v in must1.items() if t1.count(k) != v]
assert not bad1, 'P1 复核失败: %s' % bad1

# ---------- (b) 应用 P2 ----------
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

raw2, txt2, nl2 = load(P2)
for old, new in p2_pairs:
    old2, new2 = old.replace('\n', nl2), new.replace('\n', nl2)
    assert txt2.count(old2) == 1, 'P2 count=%d %r' % (txt2.count(old2), old2[:60])
    txt2 = txt2.replace(old2, new2)
open(P2, 'wb').write(txt2.encode('utf-8'))

back2 = open(P2, 'rb').read().decode('utf-8')
for old, new in p2_pairs:
    new2 = new.replace('\n', nl2)
    assert back2.count(new2) == 1, 'P2 readback missing %r' % (new2[:60],)
must2 = {'N 线（deepseek）参照实现：Phase 11（': 1, '12 臂 40 坑）': 1,
         '## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 11，2026-10-01）': 1,
         '这条线的收尾链已跑通四次（Phase 8/9/10/11）': 1,
         '**Phase 8/9/10/11 实测的 10 条收尾教训**：': 1,
         '9. **closeout 脚本里的字符串格式化/转义坑（Phase 11 实证两条）**': 1,
         '10. **跨 Phase 逐位闸门应写进 `floors`（零成本装置漂移检测）**': 1,
         '末尾 `TOTAL FAILS: 0`。': 1}
bad2 = [(k, back2.count(k), v) for k, v in must2.items() if back2.count(k) != v]
assert not bad2, 'P2 复核失败: %s' % bad2

print('P1 复核 OK (%d 项)' % len(must1))
print('P2 应用 OK: %d 处 %d -> %d B (复核 %d 项)' % (len(p2_pairs), len(raw2), len(back2.encode('utf-8')), len(must2)))
print('ALL OK')
