# -*- coding: utf-8 -*-
"""Phase 12 技能同步：rdc-main-axis-probe（12→13 臂 / 40→44 坑 / 新增 α-5 门）
   与 rdc-phase-closeout（10→12 条教训 / 参照实现加 Phase 12）。逐处 assert count==1 + 回读复核。"""
import io, os, hashlib

SK1 = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
SK2 = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
o = []


def w(s=''):
    o.append(str(s)); print(s)


def patch(path, pairs, tag):
    t = io.open(path, encoding='utf-8').read()
    h0 = hashlib.sha256(t.encode('utf-8')).hexdigest()[:8]
    for i, (a, b) in enumerate(pairs):
        c = t.count(a)
        assert c == 1, '%s[%d] old count=%d (expect 1): %r' % (tag, i, c, a[:70])
        t = t.replace(a, b, 1)
    io.open(path, 'w', encoding='utf-8').write(t)
    t2 = io.open(path, encoding='utf-8').read()
    h1 = hashlib.sha256(t2.encode('utf-8')).hexdigest()[:8]
    w('%s: %d bytes -> %d bytes ; sha8 %s -> %s' % (tag, len(t.encode('utf-8')), len(t2.encode('utf-8')), h0, h1))
    return t2


# ================= A. rdc-main-axis-probe =================
ARM_ROW = ('| **N2h1-α-5 逐层残差替换**（Phase 12） | 「栈=软门」是**少层主导**还是**逐层累积**；换探针族后结论是否存活 | '
           '把探针族由「**注入** rank-5 类轴」整体换成「**替换整段残差**」：`h_ℓ + α·diff_ℓ`（`diff_ℓ = h_ℓ(donor) − h_ℓ(recip)`，α∈[0,1]；'
           '**α=1 时该位点残差 ≡ 供体贴残差 `h_ℓ(donor)`**，用供体自身前向 ⇒ **零额外前向**）；18 位点 × **14 点 α 网格**（含 0.15/0.95，低端分辨率直接决定 xhalf 精度）；'
           '主量 = **形状量** `xhalf`（`cross_alpha`：首次达 `0.5·max y` 的 α，线性插值、不假设单调）与 `J_swap`（`jump_ratio` = max 段斜率 / 其余段斜率中位数）；'
           '端点量 `recover = y(α=1)` **按构造饱和**，只作独立正向判据 **G5**（充分性）；集中度在 **xhalf 空间**（不是 J 空间）算 `Δ_i = xhalf(ℓ_{i+1}) − xhalf(ℓ_i)` 的 `top3_share_x`（3 窗口最大占比 / `XH_RANGE`）；'
           '对照臂 E2b（相对坐标）/ E3（α>1 过冲）/ E4（全空间范数匹配随机方向地板）/ E6（读数位点 R）/ E7（置换零假设）；'
           '**新增两条构造硬断言** F11（R 位点 α=1 ≡ FULL_SWAP）与 F12（满替换 ≡ 供体贴残差）；按铁律 (r)(s) 先断言 `spearman(xhalf, depth)` 的符号再谈 G1 |\n')

GATE = ('\n**N2h1-α-5 门（qwen3-4b 实测，Phase 12）**：**G0** 前置 `curve_ok_frac ≥ 0.75 ∧ XH_RANGE ≥ 0.10`；'
        '**G2** 集中度 `top3_share_x ≥ 0.60` → 少层主导、`max_share_x ≤ 0.40` → 逐层累积、否则 mid；'
        '**G3** `Spearman(J_swap, J_inject)` ≥ 0.6 同形 / ≤ 0.2 独立；**G4** 确认集与发现集 `rho(xhalf, depth)` 同号 ∧ |rho| ≥ 0.5；'
        '**G5** `min_ℓ recover ≥ 0.90`（充分性）。组合裁决：`G2a ∧ G3_same → FEW_LAYER_DOMINANT_STACK`；`G2b ∧ G3_same → LAYERWISE_ACCUMULATE_CONFIRMED`；**其余 → `ALLOCATION_AMBIGUOUS`**。'
        '**实测**：`FULL_SWAP = +10.797396`（n=24）；`G0 ✅`（`curve_ok_frac 18/18`、`XH_RANGE = 0.1094` **仅高阈值 9%，边界脆弱**）；'
        '`G3 ✅ G3_same_gradient` `rho(J_swap, J_inject) = 0.8741`（n=18）⇒ **换族后逐层累积存活**（首次跨探针族交叉验证）；'
        '`G2 ⚠ G2_mid` `top3_share_x = 0.5745`、bootstrap 带 **[0.3867, 0.8316] 跨 0.60**、`max_share_x = 0.5895`（> 0.40）⇒ 两种结论都判不出；'
        '`G4 ❌ G4_fail` 确认集 4 位点 `rho(xhalf) = +0.80` 与发现集 `−0.7833` **反号**（4 位点落在非单调剖面不同支 = 采样密度不足，不作否定证据）；'
        '`G5 ✅ LAST_POS_STATE_SUFFICIENT` `min recover = 0.9955`（**18/18 位点**；确认集 0.9711）⇒ **末位残差是「供体答案已成形」的充分统计量**。'
        '**形状量**：`J_swap` **25.35(L6) → 1.06(L34)**（降 23.9×）；`xhalf` 0.499 → 浅端平台 0.445 → **L30 触底 0.390** → **L34 反弹 0.465**；`rho(xhalf, depth) = −0.7833`（带 [−0.9154, −0.5500]）。'
        '**端点量按构造饱和**：`recover ∈ [0.9955, 1.0010]`、`span = 0.0048` ⇒ 铁律 (r)。'
        '**G1a 是仪器伪影**：归一化 `XN=(xhalf−min)/(max−min)` 隐含「剖面随深度上升」，实测 rho 为负 ⇒ `first_reach` 全部退化返回首站点（`x_half=6.00`、`span_10_90=0.00`，恰好长得像「结晶」）⇒ 铁律 (s)，**本 Phase 不引用 G1**。'
        '**判决 = `ALLOCATION_AMBIGUOUS`。** `proj_share_u6` **0.680(L6) → 0.148(L34)**（R 处 0.1288）⇒ 浅端「全残差 ≈ 类轴」、深端「全残差主要是非类别内容」⇒ **两族剂量坐标不可比，只比秩**（⇒ 坑 43）。'
        '**同位素内建复现**：`E0(ℓ=6,α=1)` 逐位 ≡ Phase 9（`10.574739583333335`）；**F11** `E6[R](α=1) ≡ FULL_SWAP` `max|d| = 0.000e+00`；**F12** 满替换 ≡ 供体贴残差 `max = 0.000e+00`；F3 四处 `0.000e+00`；F10 逐对均值自洽 `5.329e-15`；F1 地板比 `0.3582/18.805 = 0.0190`。\n')

P4142 = """41. **端点量可能由构造决定饱和 —— 判据必须先问「这个量在端点处由数据还是由构造决定」（Phase 12 的判决性一条）**：当干预的 α=1 恰好等价于另一个**已知的满干预**时，端点量被集合包含关系**锁死**。Phase 12 实测：`recover(α=1) ∈ [0.9955, 1.0010]`、`span = 0.0048` —— 因为 Phase 8–11 已证「L6 只注入 rank-5 类轴、α=1 即得 `full_L6`」，而满替换是**严格更大**的干预（多了 rank-5 之外的全部分量）。⚠️ 后果：冻结在 seal 里的 `G0`（要求 `span(recover) ≥ 0.30`）**在浅端必然失败，死线问题将无答案** —— 这是 SMOKE 才抓出的**设计级缺陷**（不是 bug）。**对策**：① 端点量降级为**独立正向判据**（Phase 12 的 G5：`min recover ≥ 0.90`）；② 主量改用**形状量**（`xhalf` 半饱和点、`J` 陡度）；③ 修正案须在**正式运行前**冻结，并落盘 `why_this_is_a_priori_not_a_data_peek`（论证只引用已封存结论 + 集合包含关系，与正式数据无关）。**另注**：`recover` 可 > 1（是比值不是概率）；「端点不变」会被误读成「深度无关」，其实只是端点量没有分辨率。
42. **「首次达到比例」型统计量的归一化方向必须与物理方向一致（否则产出「仪器伪影」）**：把横轴（深度/位点）归一化成 `(x − min)/(max − min)` 再取 `first_reach(0.1/0.5/0.9)`，**隐含假设该剖面随深度【上升】**。Phase 12 实测 `rho(xhalf, depth) = −0.7833` 为**负** ⇒ `XN[0] = 1.0`，三个分位点**全部退化返回首站点** ⇒ `x_half = 6.00`、`span_10_90 = 0.00` —— 这个结果**恰好长得像「结晶 / 少层主导」**，正是实验想看到的形状。**对策**：任何「首次达到比例」型判据必须**先断言 `spearman(量, 深度)` 的符号**，据此选归一化方向（或直接用真实深度轴、不做归一化）；并把该符号写进报告。**这类伪影不会给出错误的方向，只会给出一条假的「强结论」——比方向反了更危险。**（⇒ 本 Phase 不引用 G1。与坑 33「物理方向与判据符号一致」互补：33 管**阈值方向**，42 管**归一化方向**。）
43. **换探针族做交叉验证时只能用秩、不能用数值**：不同族的**干预对象不同**（注入 rank-5 类轴 vs 替换整段残差），其剂量坐标**不可比**——Phase 12 实测 `proj_share_u6(ℓ) = ‖P_U6(diff_ℓ)‖/‖diff_ℓ‖` 从 **0.680(L6) 单调降到 0.148(L34)**，即浅端「全残差 ≈ 类轴」、深端「全残差主要是非类别内容」。⇒ ① 只能用 **Spearman**（秩不变）判「同形 / 不同形」，并显式声明「**只能支撑粗判定**」；② 不得把两族的 J 数值并列比较、也不得用一族的带宽去套另一族；③ 换族交叉验证的价值在于**排除系统性选择偏差**（同族内加密网格只减少抽样噪声），但它**验不了数值**。
44. **legacy payload 里的非数字键会毒死 `int(k)` 转换（SMOKE 第 1 秒崩）**：复用上一 Phase 的 `profile_abs` 等 dict 时，它可能含 `'R'`（读数位点）这类**非数字键**；`{int(k): v['jump_ratio'] for k, v in prof.items()}` 会抛 `ValueError: invalid literal for int() with base 10: 'R'`，整个 Phase 在 SMOKE 第一秒崩。**对策**：一律 `try/except (TypeError, ValueError): pass` **并打印被丢弃的键**（否则会静默漏掉位点）。**推广**：凡「跨 Phase 复用 dict / 用键做数值解析」的地方，都要先枚举键型、把非预期键显式报告。

"""

P4344 = ""
[ARM_ROW, GATE, P4142, P4344]

A = [
    # A1 frontmatter description
    ('包含 12 个可复用臂（A 意义分流最小对', '包含 13 个可复用臂（A 意义分流最小对'),
    ('/ **N2h1-α-4 自基全剖面 + 噪声带**）', '/ **N2h1-α-4 自基全剖面 + 噪声带** / **N2h1-α-5 逐层残差替换 + 层贡献分配**）'),
    ('以及 40 条已实测的坑。', '以及 44 条已实测的坑。'),
    # A2 section title
    ('## 1 十二个臂（一次跑完，勿拆散）', '## 1 十三个臂（一次跑完，勿拆散）'),
    # A3 arm row: insert after the α-4 row (which is the row ending with `逐单元格逐位复现 |\n`)
    ('逐单元格逐位复现 |\n', '逐单元格逐位复现 |\n' + ARM_ROW),
    # A4 gate paragraph: insert after the α-4 门 paragraph (ends with `3.553e-15。\n`)
    ('F10 逐对均值自洽 3.553e-15。\n', 'F10 逐对均值自洽 3.553e-15。\n' + GATE),
    # A5 pitfalls header
    ('## 4 已实测的坑（40 条，逐条对应数值）', '## 4 已实测的坑（44 条，逐条对应数值）'),
    # A6 pitfalls 41-44 before `## 5 代码骨架要点`
    ('\n## 5 代码骨架要点', '\n' + P4142 + '\n## 5 代码骨架要点'),
]

t1 = patch(SK1, A, 'SK1/main-axis-probe')

# ================= B. rdc-phase-closeout =================
B = [
    ('N 线（deepseek）参照实现：Phase 11（`tests/deepseek/Phase11/` 脚本 + `tests/deepseek_temp/Phase11/` 报告/seal/校验，12 臂 40 坑）',
     'N 线（deepseek）参照实现：Phase 12（`tests/deepseek/Phase12/` 脚本 + `tests/deepseek_temp/Phase12/` 报告/seal/校验，13 臂 44 坑；含「按字节回滚再重做」的追加修复实证）'),
    ('## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 11，2026-10-01）',
     '## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 12，2026-10-01/02）'),
    ('这条线的收尾链已跑通四次（Phase 8/9/10/11）', '这条线的收尾链已跑通五次（Phase 8/9/10/11/12）'),
    ('**Phase 8/9/10/11 实测的 10 条收尾教训**', '**Phase 8–12 实测的 12 条收尾教训**'),
    # 追加第 11、12 条（插在第 10 条之后，即 `## 参照实现` 之前）
    ('\n## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）',
     '\n11. **append-only 目标必须与「按字节回滚」脚本成对存在（Phase 12 实证）**：Phase 12 首次 `do_append_phase12.py` **已把整节写入文件**，随后自检 `assert not miss` 因正文缺一个实现名而 **exit 1**。此时若直接改锚点列表重跑，会**重复追加整节**。正确做法：① 追加前先把「已追加版」备份成 `_memo_with<N>_backup.md`；② 写 `rollback_append<N>.py`：`pre = cur[:baseline_bytes]` + **`sha256(pre) == baseline_sha` 硬断言** → 覆写（本次实测 `236092 B / 277f49da` 逐字节恢复，`前缀逐字节未变 = True`）；③ 补正文 → 重跑追加脚本（三向断言全绿）。⇒ **凡 append-only 目标，`do_append_*.py` 必须配一个 `rollback_append*.py`**，并在 MEMO 附录写「回滚路径已实证」。（本次「已追加版」备份在验证通过后清理，避免磁盘留重复副本。）\n'
     '12. **追加自检的锚点必须在写入前对「追加源文件」预检 —— 否则自检本身会假失败（Phase 12 实证）**：锚点表里写了 `cross_alpha`（函数名），但正文只写了自然语言「线性插值」⇒ `count=0` 触发断言。**更稳的顺序：先在 `memo_append_<N>.md` 上跑一遍 count 预检（只读、不落盘），全绿再追加**；锚点要逐条对应正文里**真实出现**的字符串（实现名、数值、sha8、verdict），而不是「应该出现」的名字。\n\n'
     '## 参照实现（Phase 3125，210/210 全绿；Phase 3128，19/19 全绿）'),
    ('- N 线独立复核（Phase 11）：`tests/deepseek/Phase11/disk_verify_phase11.py`（17 分区 A0/A1/B1/B2/B3/C/D/D2/E/F/G/H/I/J/K/L/M，含**同 seed 逐位复现 bootstrap 带与置换 null**），末尾 `TOTAL FAILS: 0`。',
     '- N 线独立复核（Phase 11）：`tests/deepseek/Phase11/disk_verify_phase11.py`（17 分区 A0/A1/B1/B2/B3/C/D/D2/E/F/G/H/I/J/K/L/M，含**同 seed 逐位复现 bootstrap 带与置换 null**），末尾 `TOTAL FAILS: 0`。\n'
     '- N 线独立复核（Phase 12）：`tests/deepseek/Phase12/disk_verify_phase12.py`（从冻结 `result_phase12.json` 确定性重算 `xhalf`/`J_swap`/`recover`、**同 seed 复现 bootstrap 带与置换 null**、G 族布尔、F11/F12/F13/F10）。'),
]

t2 = patch(SK2, B, 'SK2/phase-closeout')

# ================= 回读复核 =================
w('--- readback ---')
t1 = io.open(SK1, encoding='utf-8').read()
t2 = io.open(SK2, encoding='utf-8').read()
checks = [('SK1 十三个臂', t1.count('## 1 十三个臂（一次跑完，勿拆散）'), 1),
          ('SK1 13 臂 desc', t1.count('包含 13 个可复用臂（'), 1),
          ('SK1 α-5 arm row', t1.count('**N2h1-α-5 逐层残差替换**（Phase 12）'), 1),
          ('SK1 α-5 desc token', t1.count('**N2h1-α-5 逐层残差替换 + 层贡献分配**）'), 1),
          ('SK1 α-5 门', t1.count('**N2h1-α-5 门（qwen3-4b 实测，Phase 12）**'), 1),
          ('SK1 44 坑 header', t1.count('## 4 已实测的坑（44 条，逐条对应数值）'), 1),
          ('SK1 44 条 desc', t1.count('以及 44 条已实测的坑。'), 1),
          ('SK1 坑41', t1.count('\n41. **端点量可能由构造决定饱和'), 1),
          ('SK1 坑44', t1.count('\n44. **legacy payload 里的非数字键'), 1),
          ('SK1 G1a 伪影文字', t1.count('恰好长得像「结晶」'), 1),
          ('SK2 12 条教训', t2.count('**Phase 8–12 实测的 12 条收尾教训**'), 1),
          ('SK2 教训11', t2.count('\n11. **append-only 目标必须与「按字节回滚」脚本成对存在'), 1),
          ('SK2 教训12', t2.count('\n12. **追加自检的锚点必须在写入前对「追加源文件」预检'), 1),
          ('SK2 Phase12 disp', t2.count('Phase 8 → Phase 12'), 1),
          ('SK2 disk verify12', t2.count('disk_verify_phase12.py'), 1),
          ('SK2 13臂44坑', t2.count('13 臂 44 坑'), 1)]
bad = []
for nm, c, exp in checks:
    ok = (c == exp)
    w('  %-24s count=%d expect=%d %s' % (nm, c, exp, 'OK' if ok else '!! BAD'))
    if not ok:
        bad.append(nm)
assert not bad, 'readback bad: %s' % bad
w('ALL READBACK OK')
