# -*- coding: utf-8 -*-
"""Phase 10 工作区长期记忆（MEMORY.md）补丁：逐条 replace + assert count==1。"""
import os, io

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'Phase10', 'patch_memory_phase10.txt')

raw = open(MEM, 'rb').read()
eol = '\r\n' if b'\r\n' in raw else '\n'
text = raw.decode('utf-8')
lines = text.splitlines()

log = []


def rep_in_line(marker, old, new, note):
    global lines
    idx = [i for i, l in enumerate(lines) if marker in l]
    assert len(idx) == 1, '%s: marker 命中 %d 次' % (note, len(idx))
    i = idx[0]
    assert lines[i].count(old) == 1, '%s: 子串命中 %d 次' % (note, lines[i].count(old))
    lines[i] = lines[i].replace(old, new)
    log.append('OK %s (line %d)' % (note, i + 1))


def insert_after(marker, block, note):
    global lines
    idx = [i for i, l in enumerate(lines) if marker in l]
    assert len(idx) == 1, '%s: marker 命中 %d 次' % (note, len(idx))
    i = idx[0]
    lines[i + 1:i + 1] = block.split('\n')
    log.append('OK %s (inserted after line %d, +%d lines)' % (note, i + 1, len(block.split('\n'))))


# --- M1 Ledger 编号 ---
rep_in_line('n=**292**',
            'n=**292**；Phase 8 补登 291、Phase 9 补登 292；**N 线 Phase 3–7 仍待补**',
            'n=**293**；Phase 8 补登 291、Phase 9 补登 292、**Phase 10 补登 293**；**N 线 Phase 3–7 仍待补**',
            'M1 Ledger n=293')

# --- M2 Phase 10 条目 ---
BLOCK = """  - **N2h1-α-3（Phase 10，2026-10-01 22:06）软阈值深度剖面：单层产生被否证 ⇒「栈=软门」**：qwen3-4b，发现集 n=24，GPU **253.0 s**，零 OOM；面板 12 字段 × Phase8/Phase9 **双向逐元素**继承断言（exec `49f3ebda` / seal `70bb8b92` / amend1 `de8a0717`）。
    - **设计**：同一探针 `h_ℓ + α·P_U6(diff6)` 搬到 ℓ=6…34 每个层输出 + 读数位点 R（最终 LayerNorm 之后）；`y = dDonor/full_L6`，J 尺度不变可跨位点比，半饱和点须换算 `x*_rel = x*_abs·r_ℓ`。
    - **J(ℓ) 随注入深度单调下降**：绝对剂量 **5.41→1.15**、相对剂量 **14.26→1.03**；Spearman **−0.692**（12 非 UNREACH 位点）/ **−0.872**（18 剖面位点）；**全程无断崖**（最大相邻跌幅 4.83→2.85 = 1.7× < Q3 的 2×）⇒ **逐层累积，不是单层产生**。
    - **读数位点 R 严格线性（否证条款排除）**：`r_R = 0.1513`（仅深度位点的 1/3）⇒ 同网格 UNREACH，补扩展网格（α 到 16）后 `dDonor/α = 1.127` 常数、**R²_lin 1.0000 / γ 1.00 / J 1.00**，推到相对剂量 2.58 仍无拐点 ⇒ **S 形不是解嵌几何/类分数度量的读数假象**。
    - **自基旋转（最重要限界）**：`overlap(U_ℓ, U6)` **0.6892→0.0298**；**自基下 α=1 在 L7/L12/L20/L34 全部 ≈ full**（10.735/10.648/10.474/10.576 vs full 10.575）⇒ 绝对效应随深度塌掉（y_sat 1.016→**0.027**，且与 r_ℓ 同比例）**主要是方向失配，不是下游衰减**。`V_ownbasis = BASIS_SENSITIVE (2/4)`：**J 的趋势双基稳健，J 的绝对高度不是**。
    - **判决**：修正族 **Q_abs = Q_rel = Q2_accumulate ⇒ Q_ROBUST**（两剂量坐标一致）；封存 P 族 绝对 `P0_no_verdict` / 相对 `P4_once_formed` ⇒ 不一致源于"J 在相对坐标整体更高、把窗口判据 [7,10] 抬过 3.0"，属判据形式问题非物理矛盾（**不改判封存，两者并列报告**）。
    - **比特锚**：E0（ℓ=6, α=1）= `10.574739583333335` 逐位等于 Phase 9 `full`；E0b +0.3335 ≡ Phase 9 D1a；F3 四处（含 norm hook）0.000e+00；`mean‖P_U6(diff6)‖` 17.0613 与 Phase 9 相同；**F1 通过**（E4 地板 0.1227/12.444 = 0.0099）；确认集 n=17 同判 **3/4**（L11 J=1.99 vs 2.04 压在 2.0 阈值门口）。
    - **新口径陷阱**：`r_ℓ = ‖u6‖/‖h_ℓ(recip)‖`（0.0268–0.4664）**≠** Phase 9 的 `rbar`（0.2368，相邻两层范数比），两者差近 2×，**不得互相核对 drift**。
    - **记录**：deepseek 备忘录 `## Phase 10` 节起 **2036 行**，183,607→**209,316 B** / 2033→**2301 行**（前缀逐字节未变、BOM/CRLF、bare_lf 0、Phase 标题 10 个）；Ledger 补登 N 线第 3 条（292→**293**，备份 `atlas_ledger_backup_pre_phase10.json`）。"""

insert_after('  - **新参数**：承诺层的判决量从"某组件份额"改为"读出传递函数的**半饱和点 x\\***"。',
             BLOCK, 'M2 Phase 10 条目')

# --- M3 铁律 (j)-(n) ---
rep_in_line('SMOKE 截断 α 网格时必须保留定义点',
            '（否则 `full` 归一化退化）。',
            '（否则 `full` 归一化退化）。',
            'M3a 锚点确认')
insert_after('SMOKE 截断 α 网格时必须保留定义点',
             ('  - **装置铁律（Phase 10 新增）**：'
              '**（j）跨位点比较必须分「绝对/相对」双剂量坐标，半饱和点必须换算 `x*·r_ℓ` 才能比**；'
              '**（k）"注入越浅⇒经过层越多⇒非线性越强"是物理方向，判据符号必须与之一致**（封存 P2 反了，故并列修正族 Q）；'
              '**（l）位点的 r_ℓ 可差 3–17 倍，任何"打不动"结论先自检是方向旋转（自基臂）还是真衰减**；'
              '**（m）凡以"读数端/度量端"解释机制，必做一个读数位点探针（最终 norm 之后注入）**；'
              '**（n）深部位点的固定基探针必须同时报 `overlap(U_ℓ, U_ref)`**。'),
             'M3b 铁律 j-n')

# --- M4 限界 ③ ---
rep_in_line('未做权重级验证',
            '行为读数与层内读数不可互推**。',
            ('行为读数与层内读数不可互推**；**Phase 10 追加：跨深度探针给出的是"该深度的传递函数"，'
             '固定基探针的"深度衰减"必须扣除基旋转（overlap 0.689→0.030）后才是机制量**。'),
            'M4 限界 ③ 追加')

# --- M5 技能行 ---
rep_in_line('`rdc-main-axis-probe`', '（**10 臂 + 31 条坑**', '（**11 臂 + 36 条坑**', 'M5a 臂数/坑数')
rep_in_line('参数化曲线判据（禁用"是否阶跃"）',
            '**参数化曲线判据（禁用"是否阶跃"）**',
            ('**参数化曲线判据（禁用"是否阶跃"）**、**跨深度剂量剖面 + 读数位点否证探针**、'
             '**绝对/相对双剂量坐标 + r_ℓ 换算**、**自基旋转限界（overlap 必须报备）**'),
            'M5b 技能要点')

# --- M6 下一步 ---
rep_in_line('下一轮最高优先 = Phase 9 阈值增益检验',
            '**下一轮最高优先 = Phase 9 阈值增益检验**',
            '**Phase 9 阈值增益检验已完成（见上：L6 无增益）；下一轮最高优先 = Phase 11 自基全剖面 + 噪声带**',
            'M6a 下一步标题')
rep_in_line('把 I_nl 的机制含义变成判决；',
            '⇒ 把 I_nl 的机制含义变成判决；',
            '⇒ 判决已完成（Phase 9）；**Phase 11 做法**：E3 自基臂铺到全部 18 个剖面位点给**双基 J(ℓ)**，同时**记录逐对 dDonor（n=24）**用配对 bootstrap 给误差带（**零额外前向**，只改落盘），预注册判据 = Q2 的 Spearman 95% bootstrap 区间是否完全在 −0.6 以下；第二候选：用「逐层替换」而非「逐层注入」做层内贡献分配；',
            'M6b 下一步内容')
rep_in_line('Phase 9 的 D4 已部分执行',
            '范数匹配随机 5 维（**Phase 9 的 D4 已部分执行**）',
            '范数匹配随机 5 维（**Phase 9 D4 + Phase 10 E4 已在 3 个深度位点执行**）',
            'M6c R1 挂账')

out = eol.join(lines) + eol
open(MEM, 'wb').write(out.encode('utf-8'))
b2 = open(MEM, 'rb').read()
log.append('MEMORY.md bytes %d -> %d ; lines %d -> %d ; sha256 %s' %
           (len(raw), len(b2), len(lines), len(b2.decode('utf-8').splitlines()),
            __import__('hashlib').sha256(b2).hexdigest()[:16]))
# 落盘复核
t2 = b2.decode('utf-8')
for k in ['n=**293**', 'N2h1-α-3（Phase 10', 'Q2_accumulate ⇒ Q_ROBUST', '0.6892→0.0298',
          '装置铁律（Phase 10 新增）', 'Phase 11 自基全剖面', '11 臂 + 36 条坑', 'R²_lin 1.0000']:
    c = t2.count(k)
    log.append('  anchor %-34s count=%d %s' % (k, c, 'OK' if c >= 1 else '!! MISSING'))
    assert c >= 1, k
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(log))
print('\n'.join(log))
