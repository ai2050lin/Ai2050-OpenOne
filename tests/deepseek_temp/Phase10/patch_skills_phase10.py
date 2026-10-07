# -*- coding: utf-8 -*-
"""Phase 10 收尾链最后一项：同步两个技能到 11 臂 / 36 坑（幂等）。
纪律：全程 Python 读写，逐处 assert count==1，写完回读复核真实磁盘。
避免 Edit（幻影 + IME 吞字），避免内联 python -c（反引号被吃）。
幂等：检测到已打补丁的标记则跳过写入，只做回读复核。
"""
import os, hashlib

REPORT = []
def log(s):
    REPORT.append(s)
    print(s)

SK1 = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
SK2 = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'

def read_text(p):
    b = open(p, 'rb').read()
    crlf = b'\r\n' in b
    t = b.decode('utf-8')
    if crlf:
        t = t.replace('\r\n', '\n')
    return t, crlf, b

def write_text(p, t, crlf):
    if crlf:
        t = t.replace('\n', '\r\n')
    open(p, 'wb').write(t.encode('utf-8'))

def rep(t, old, new, tag):
    c = t.count(old)
    assert c == 1, 'REP %s : count=%d (expected 1) old_head=%r' % (tag, c, old[:40])
    return t.replace(old, new, 1)

def insert_after_line(t, needle, block, tag):
    c = t.count(needle)
    assert c == 1, 'ANCHOR %s : count=%d (expected 1) needle=%r' % (tag, c, needle[:50])
    i = t.index(needle)
    j = t.index('\n', i)
    return t[:j+1] + block + '\n' + t[j+1:]

# ============================================================
# 技能 1：rdc-main-axis-probe
# ============================================================
t1, crlf1, b1 = read_text(SK1)
sha_before1 = hashlib.sha256(b1).hexdigest()
log('=== SK1 %s ===' % SK1)
log('before bytes=%d sha8=%s crlf=%s' % (len(b1), sha_before1[:8], crlf1))

SK1_MARK = '## 1 十一个臂（一次跑完，勿拆散）'
if SK1_MARK in t1:
    log('SK1 ALREADY PATCHED -> skip writes (verify only)')
else:
    t1 = rep(t1, '包含 9 个可复用臂（A 意义分流最小对', '包含 11 个可复用臂（A 意义分流最小对', 'd-arms')
    t1 = rep(t1, '以及 26 条已实测的坑。', '以及 36 条已实测的坑。', 'd-pits')
    t1 = rep(t1, '## 1 十个臂（一次跑完，勿拆散）', '## 1 十一个臂（一次跑完，勿拆散）', 'sec1-title')

    ARM3 = (
    "| **N2h1-α-3 剂量-深度剖面**（Phase 10） | 软阈值是**哪一层**做的，还是**整条栈逐层累积** | "
    "唯一变量 = **注入深度** ℓ：**同一族向量** `u6 = P_U6(diff6)`（发现集 L6 输出的 rank-5 类均值子空间投影）搬到 ℓ∈{6,…,34} 每个层输出 + 位点 **R**（最终 LayerNorm 之后、lm_head 之前）；"
    "**ℓ=6 即 α-2 的站点，构成剖面零点**；两套剂量坐标 **x_abs=α**（绝对线性放大）与 **x_rel=a_rel**（按位点残差范数归一，`r_ℓ=mean‖u6‖/‖h_ℓ‖`）；"
    "**固定基臂 E1** 与 **自基臂 E3**（`h_ℓ + α·P_U{ℓ}(diff_ℓ)`，各层自算 U_ℓ=SVD(类均值)）并列；"
    "**读数位点 R = 否证探针**（R 若也长 S 形 ⇒ 软阈值只是读数假象）；"
    "判据用**参数化分类器**（logistic `y=A·sigmoid(k(x−x*))`，A=max(y)，对 (k,x*) 做二维网格最小二乘，**无 scipy**），分 UNREACH / S_STRONG / S_WEAK / GRADUAL / LINEAR |"
    )
    t1 = insert_after_line(t1, '**禁用"是否阶跃"** |', ARM3, 'arm3-row')

    GATE3 = (
    "**N2h1-α-3 门（qwen3-4b 实测，Phase 10）**：判「**单层产生**」须 **J(ℓ) 出现断崖**（相邻位点跌幅 ≥2×）且断崖处 = 某单层；判「**逐层累积**」须 **J(ℓ) 单调递减且全程无断崖**。"
    "**实测**：绝对剂量 J **5.41(L6)→1.15(L34)**、相对剂量 **14.26→1.03**，**Spearman −0.6923（12 非 UNREACH 位点）/−0.8720（18 剖面位点）**，**最大相邻跌幅仅 1.7×**（L8→L9）⇒ 判 **Q2_accumulate（逐层累积）**，"
    "「**层=软门**」须改为「**栈=软门**」；读数位点 R **严格线性**（`R²_lin=1.0000`、`γ=1.00`、`J=1.00`，扩到相对剂量 2.58 仍无拐点）⇒ 排除\"读数假象\"否证条款；"
    "**自基臂揭示** `overlap(U_ℓ,U6)` **0.6892(L7)→0.0298(L34)**，自基下 α=1 四位点**全部 ≈ full（10.47–10.58 vs 10.575）** ⇒ **绝对效应随深度塌掉主要是类子空间旋转，不是下游衰减**（`V_ownbasis=BASIS_SENSITIVE(2/4)`：J 趋势双基稳健，J 绝对高度单基依赖）。"
    "**同位素内建复现**：`E0(ℓ=6,α=1)` 必须与 α-2 的 diff6 臂**逐位相同**（Phase 10 实测 **10.574739583333335 ≡ Phase 9**）。"
    )
    t1 = insert_after_line(t1, '（Phase 9 实测 10.574739583333335 ≡ Phase 8）。', GATE3, 'arm3-gate')

    t1 = rep(t1, '## 4 已实测的坑（26 条，逐条对应数值）', '## 4 已实测的坑（36 条，逐条对应数值）', 'sec4-title')

    PITS = "\n".join([
    "32. **跨位点比较必须分双剂量坐标，半饱和点须先换算**：同一注入向量搬到不同层时，绝对坐标 `x_abs=α` 与相对坐标 `x_rel=a_rel`（按位点残差范数归一）会给出**不同的 J 高度**——实测相对坐标整体更高（把窗口判据 `[7,10]` 抬过 J≥3.0，导致两坐标下 P 族判决不一致）。⇒ **报 J(ℓ) 剖面必须两个坐标都报**，跨位点比半饱和点须换算 `x*_rel = x*_abs · r_ℓ`。⚠️ `r_ℓ = mean‖u6‖/‖h_ℓ‖`（实测 **0.0268–0.4664**）**≠** 相邻两层范数比 `rbar = mean‖P_U6(diff5)‖/mean‖P_U6(diff6)‖`（**0.2368**），两者定义不同、差近 2×，**不得互相核对 drift**。",
    "33. **物理方向与判据符号必须一致**：注入越浅 ⇒ 经过的非线性层越多 ⇒ S 形应越强 ⇒ `J` 随深度**递减**才叫\"逐层累积\"。Phase 9 的封存判据 P2 曾把方向写成\"递增\"（**反了**）⇒ Phase 10 并列新增方向修正族 **Q1_readout_origin / Q2_accumulate / Q3_single_layer**（`Q3 单层产生` 要求 **J(ℓ) 出现 ≥2× 断崖且断崖处=单层**；`Q2 逐层累积` 要求 **J 单调递减且全程无断崖**）。**判据符号写反会让整个 Phase 的判决颠倒**；封存判据不改判，与新族**并列报告**。",
    "34. **剖面设计的两条对称性铁律**：① **网格必须对称**——固定基臂 E1 与自基臂 E1b 必须用**同一 α 网格**（7 点 `[0.01,0.02,0.05,0.10,0.20,0.40,0.80]`），否则两者噪声结构不同阶、J 剖面不可比（Phase 10 冒烟抓出 E1b 只有 4 点）；② **剖面必须含参照零点**——`profile_sites` 必须包含 `ℓ = 承诺层`（= 上一 Phase 的站点，如 L6），否则既无法定位\"衰减从哪开始\"，也丢掉内建跨 Phase 复现锚点（`E0(ℓ=L6,α=1)` 必须与上一 Phase **逐位相同**）。",
    "35. **深部位点\"打不动\"先自检方向旋转，再下\"衰减\"结论**：固定基探针在深层的 overlap 会大幅下降——实测 `overlap(U_ℓ, U6)` **0.6892(L7) → 0.2270(L12) → 0.0298(L34)**；用**自基**（各层自算 `U_ℓ`）重测，`α=1` 在 L7/L12/L20/L34 **全部 ≈ full**（10.735/10.648/10.474/10.576 vs full 10.575）⇒ **固定基测到的\"绝对效应随深度塌掉\"主要是类子空间旋转，不是下游衰减**。⇒ ① 深部位点固定基探针**必须报 `overlap(U_ℓ, U_base)`**；② `V_ownbasis` 要分开写\"**趋势双基稳健**\"与\"**绝对高度单基依赖**\"（实测 2/4 ⇒ BASIS_SENSITIVE）。",
    "36. **读数端解释必须做\"读数位点探针\"（否证条款）**：任何\"某段有阈值/非线性\"的结论，都必须在本段**末端**（如最终 LayerNorm 之后、lm_head 之前）注入同一族向量，检查该位点自己是否也长出同样形状。**若读数位点也是 S 形 ⇒ \"软阈值\"只是解嵌几何/类分数度量的读数假象，整条机制叙述须撤回**。实测 qwen3-4b 的 R 位点**严格线性**（`R²_lin=1.0000`、`γ=1.00`、`J=1.00`，扩到相对剂量 2.58 仍无拐点 ⇒ 类标签 LINEAR）⇒ 排除该否证。⚠️ R 位点 `r_R` 可能远小于深度位点（实测 **0.1513 = 深度位点的 1/3**）⇒ **同网格会落 UNREACH，必须显式扩网格**（实测 α∈{0,0.5,1,2,4,8,16}）。",
    ])
    t1 = insert_after_line(t1, '必须如实写，同时附上投影校正值。', PITS, 'pits-32-36')

    write_text(SK1, t1, crlf1)

# 回读复核
t1b, crlf1b, b1b = read_text(SK1)
sha_after1 = hashlib.sha256(b1b).hexdigest()
log('after  bytes=%d sha8=%s crlf=%s' % (len(b1b), sha_after1[:8], crlf1b))
checks1 = [
 ('包含 11 个可复用臂', 1), ('以及 36 条已实测的坑。', 1),
 ('## 1 十一个臂（一次跑完，勿拆散）', 1),
 ('| **N2h1-α-3 剂量-深度剖面**（Phase 10）', 1),
 ('**N2h1-α-3 门（qwen3-4b 实测，Phase 10）**', 1),
 ('## 4 已实测的坑（36 条，逐条对应数值）', 1),
 ('32. **跨位点比较必须分双剂量坐标', 1),
 ('33. **物理方向与判据符号必须一致**', 1),
 ('34. **剖面设计的两条对称性铁律**', 1),
 ('35. **深部位点', 1),
 ('36. **读数端解释必须做', 1),
 ('**N2h1-α-2 剂量-响应**（Phase 9）', 1),
 ('**N2h1-α-2 门（qwen3-4b 实测，Phase 9）**', 1),
 ('26. **SMOKE 是', 1), ('31. **随机方向', 1),
]
for s, exp in checks1:
    c = t1b.count(s)
    ok = (c == exp)
    log('  %-52s count=%d %s' % (s[:52], c, 'OK' if ok else 'FAIL'))
    assert ok, 'SK1 verify fail: %r' % s

# ============================================================
# 技能 2：rdc-phase-closeout
# ============================================================
t2, crlf2, b2 = read_text(SK2)
sha_before2 = hashlib.sha256(b2).hexdigest()
log('')
log('=== SK2 %s ===' % SK2)
log('before bytes=%d sha8=%s crlf=%s' % (len(b2), sha_before2[:8], crlf2))

SK2_MARK = '## N 线（deepseek）Phase 收尾实证（Phase 8 / Phase 9 / Phase 10，2026-10-01）'
if SK2_MARK in t2:
    log('SK2 ALREADY PATCHED -> skip writes (verify only)')
else:
    t2 = rep(t2, '参照实现：Phase 3125',
             'N 线（deepseek）参照实现：Phase 10（`tests/deepseek/Phase10/` 脚本 + `tests/deepseek_temp/Phase10/` 报告/seal/校验，11 臂 36 坑）。glm5 线参照实现：Phase 3125',
             'd-nline')
    t2 = rep(t2, '## N 线（deepseek）Phase 收尾实证（Phase 8 / Phase 9，2026-10-01）',
             '## N 线（deepseek）Phase 收尾实证（Phase 8 / Phase 9 / Phase 10，2026-10-01）', 'sec-title')
    t2 = rep(t2, '| 复核 | `disk_verify_phase{N}.py` | `disk_verify_phase{N}.txt`，末尾 `N/N 项通过` |',
             '| 复核 | `disk_verify_phase{N}.py` | `disk_verify_phase{N}.txt`，末尾 `N/N 项通过` |\n'
             '| 技能同步 | `patch_skills_phase{N}.py` | 同步 `rdc-main-axis-probe` / `rdc-phase-closeout` 两个技能（臂数/坑数/门/实证小节），逐处 `assert count==1` + 回读复核 |',
             'table-row')
    t2 = rep(t2, '**Phase 8/9 实测的 6 条收尾教训**：', '**Phase 8/9/10 实测的 8 条收尾教训**：', 'lessons-title')

    LES = "\n".join([
    "7. **IME 会在 `Edit` 的 `old_string` 里吞汉字（Phase 10 实证）**：编辑**大段中文**的 `old_string` 时，输入法可能在提交瞬间吞掉一个字（实测 \"才能比\" → \"才 比\"），导致 `old_string` 不匹配而报错（或更坏：匹配到错误位置）。⇒ **大段中文插入一律拆两步**：短 `old_string` 做 `replace`（逐处 `assert count==1`）+ 锚点行 `insert_after`；**避免把整段中文当 `old_string`**。更稳的做法是整段用 **Python 文件化补丁脚本**（本线 Phase 10 起即如此），彻底绕开 `Edit`。",
    "8. **`Read` 的行号在空行处会 off-by-one（Phase 10 实证）**：`Read` 输出的行号与真实磁盘行号在**连续空行**附近可能偏 1 行 ⇒ 用行号定位追加/插入点时，先 `Grep -n` 拿真实行号，或改用**文本锚点**插入；**不要直接信任 `Read` 的行号**。",
    ])
    t2 = insert_after_line(t2, '并在 MEMO 附录记一笔。', LES, 'lessons-7-8')

    write_text(SK2, t2, crlf2)

t2b, crlf2b, b2b = read_text(SK2)
sha_after2 = hashlib.sha256(b2b).hexdigest()
log('after  bytes=%d sha8=%s crlf=%s' % (len(b2b), sha_after2[:8], crlf2b))
checks2 = [
 ('N 线（deepseek）参照实现：Phase 10', 1),
 ('## N 线（deepseek）Phase 收尾实证（Phase 8 / Phase 9 / Phase 10，2026-10-01）', 1),
 ('| 技能同步 | `patch_skills_phase{N}.py` |', 1),
 ('**Phase 8/9/10 实测的 8 条收尾教训**：', 1),
 ('7. **IME 会在 `Edit` 的 `old_string` 里吞汉字', 1),
 ('8. **`Read` 的行号在空行处会 off-by-one', 1),
 ('6. **时钟不可信**', 1),
]
for s, exp in checks2:
    c = t2b.count(s)
    ok = (c == exp)
    log('  %-52s count=%d %s' % (s[:52], c, 'OK' if ok else 'FAIL'))
    assert ok, 'SK2 verify fail: %r' % s

log('')
log('=== ALL SKILL PATCHES OK ===')

out = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase10\patch_skills_phase10.txt'
open(out, 'w', encoding='utf-8').write('\n'.join(REPORT) + '\n')
