# -*- coding: utf-8 -*-
"""技能同步（Phase 21）：rdc-main-axis-probe 坑 61->63；rdc-phase-closeout 教训 33->34 + N 线节标题。"""
import os
import io
import hashlib

SK_M = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
SK_C = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase21\_patch_skills_report.txt'
o = []


def w(s=''):
    o.append(str(s))


# ---------- main-axis-probe ----------
M_REP = [
    (u'## 4 已实测的坑（61 条，逐条对应数值）',
     u'## 4 已实测的坑（63 条，逐条对应数值）'),
    (u'    - **通用推论**：凡复用上一 Phase 的冻结中间量作为「可比基线」，都要问一句「**这个量在本 Phase 的实验变量下会不会变**」——不变即不可作证据，必须同时落一份**随臂重算**的版本并让判据落在它上面。',
     u'    - **通用推论**：凡复用上一 Phase 的冻结中间量作为「可比基线」，都要问一句「**这个量在本 Phase 的实验变量下会不会变**」——不变即不可作证据，必须同时落一份**随臂重算**的版本并让判据落在它上面。\n'
     u'\n'
     u'62. **直接读 bitsandbytes 4bit 模块权重会得到「反量化内部」而非实现矩阵 ⇒ 用「单位阵探针」取 `W^T`（Phase 21 实证）**：`W_o`/`W_down` 在 nf4 臂上是 `bnb.nn.Params4bit` 包装，`.weight` 拿到的是量化存储/反量化缓存，与 bf16 臂的 `.weight` **不是同一对象**，直接取来做权重实现级容量 `W` 会**跨精度不可比**。\n'
     u'    - **对策**：`get_weightT(mod) = mod(torch.eye(mod.in_features, dtype=bf16, device=mod_dev(mod)))` —— 把**单位阵**喂过模块取输出即得 `W^T`（shape `[out, in]`），绕开 4bit 内部，**两臂走同一条路径**。SMOKE 必验 shape（本轮 `(4096, 2560) -> (2560, 4096)`）。**推论**：任何「读权重」的臂（容量/范数/投影）都要先问「这个 `.weight` 在量化臂上到底是什么」。\n'
     u'\n'
     u'63. **单组件 `argmax` 身份可在「并列带」内随数值精度翻转 ⇒ 只报 `argmax` 会把「形状稳健」误判成「不稳健」；且「floors 是否达标」是跨模型不可比的装置性质（Phase 21 实证）**：qwen3-4b 写入窗单头 `argmax` 在 nf4 下由 `head14` 变 `head8`，但**前三单头彼此差 < 0.004**、33 维 `share_v` 跨精度秩相关 **ρ ≥ 0.992** ⇒ 翻转发生在**并列带内部**，分布形状未变。\n'
     u'    - **对策**：判据写在「分布层」（`max_head_share_v` 的 |Δ| + 秩相关 `ρ`），并**强制并报「前三单头及其差值」**让人看见并列宽度；「单头身份」只作描述性指标，**不得单独作判据**。\n'
     u'    - **跨模型地板不可比（同轮附带）**：glm4-9b 的 floors 在两精度下**同不达标**（`frac_M` 0.431/0.425），而 qwen3-4b 两精度**同达标**（≈0.011–0.031）——因 A1 的**单组件效应幅度极小**（`max|comp|` 0.13 vs 1.02），而 mismatch 对照的**绝对量不随之缩小** ⇒ **相对地板被抬高**。⇒ 跨模型比 floors 时必须同时报**效应绝对幅度**，否则把装置差异误读成精度效应。'),
]

# ---------- phase-closeout ----------
C_REP = [
    (u'## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 20，2026-10-01/02）',
     u'## N 线（deepseek）Phase 收尾实证（Phase 8 → Phase 21，2026-10-01/02）'),
    (u'    - **(c) 生成的留痕件（`memo_append_phase{N}.md`）必须与最终落盘正文逐字节一致**：回滚重生成后，**留痕件也要一并重写**，否则「生成器 vs 留痕件 vs MEMO」三方不一致（承教训 26）。',
     u'    - **(c) 生成的留痕件（`memo_append_phase{N}.md`）必须与最终落盘正文逐字节一致**：回滚重生成后，**留痕件也要一并重写**，否则「生成器 vs 留痕件 vs MEMO」三方不一致（承教训 26）。\n'
     u'\n'
     u'34. **收尾链有两处「工具视图 vs 真实磁盘」陷阱（Phase 21 实证）：`do_append` 的锚点预检必须对「追加源实际用词」逐条 count；复核脚本的比较式要按字符串长度配对；「改了没有」不得只信 `Read` 视图**：\n'
     u'    - **(a) 锚点预检要「对源、不对想象」**：`do_append_*` 的 `ANCHORS` 是**人工写的期望词表**，极易与渲染器实际输出**不一致**（本轮 6 个锚点在正文中 count=0：`N2h1-α-14`（正文用 ASCII `N2h1-alpha-14`）、`b_{c,ℓ}`/`w_ℓ`（正文用 `b_{c,l}`/`w_l`）、`identity-probe`（正文用「单位阵探针」）、`Phase 22`（正文用「后续死线」）、`com_B`（模型未产出该量））。**对策**：预检**必须**先对**追加源文件**做 `count>=1` 断言并逐条打印 count，缺一即停；修 `ANCHORS` 时以**源文件实际字节**为准（用 Python `repr` dump 目标行，不靠肉眼/`Read`）。**推论**：锚点表是「生成器输出的函数」，改渲染器必重跑预检。\n'
     u'    - **(b) 8 字符 `sha8` 与 64 字符 `sha256` 的比较是永假式**：`disk_verify_*` 里 `chk(\'seal sha == exec.seal_sha256\', h8(SEAL) == EX[\'seal_sha256\'], h8(SEAL), EX[\'seal_sha256\'][:8])` —— 左 8 位、右 64 位，**数学上不可能相等**，却把 `got=… exp=…` 打印成两个看起来相同的 8 位前缀，极易被误当成「数据不一致」而去误查数据。**对策**：**比较前核对两侧字符串长度**；一律 `hashlib.sha256(open(p,\'rb\').read()).hexdigest()` 与 64-hex 比、`[:8]` 与 8-hex 比；**禁止**用 `h8()` 直接对 64-hex 字段。属**生成件缺陷**（承教训 33b 的回改窗口）⇒ 修脚本重跑，**不动任何已交付产物**。\n'
     u'    - **(c) `Read` 工具对「曾被读过的大文件」可能返回陈旧缓存**（Phase 21 实证：`MEMORY.md` 的 `Read` 视图停在回填前，而 `Grep`/Python 读到的是已回填版）。**对策**：「改了没有」的判定一律用 `Grep` 或 **Python 写报告**复核，不用 `Read` 视图当唯一证据；写盘脚本设计成**幂等 + 自诊断**（`old`/`new` 双向 count、报告先行、再定 `exit code`），避免「以为没写、其实已写」或反之。'),
]


def patch(path, reps, tag):
    raw = open(path, 'rb').read()
    bom = raw[:3] == b'\xef\xbb\xbf'
    t = raw.decode('utf-8-sig')
    w('=== %s ===' % tag)
    w('  path=%s bytes=%d crlf=%d bom=%s sha8=%s'
      % (path, len(raw), raw.count(b'\r\n'), bom, hashlib.sha256(raw).hexdigest()[:8]))
    miss = []
    for i, (a, b) in enumerate(reps):
        n = t.count(a)
        w('  [%d] old_count=%d new_count=%d' % (i, n, t.count(b)))
        if n != 1:
            miss.append(i)
    if miss:
        w('  !! 缺匹配 %s ⇒ 未写盘' % miss)
        return miss
    for a, b in reps:
        t = t.replace(a, b)
    out = (('\ufeff' if bom else '') + t).encode('utf-8')
    open(path, 'wb').write(out)
    rb = open(path, 'rb').read()
    w('  bytes(after)=%d crlf=%d bare_lf=%d sha8=%s'
      % (len(rb), rb.count(b'\r\n'), rb.count(b'\n') - rb.count(b'\r\n'),
         hashlib.sha256(rb).hexdigest()[:8]))
    return []


mm = patch(SK_M, M_REP, 'rdc-main-axis-probe')
mc = patch(SK_C, C_REP, 'rdc-phase-closeout')
w('')
w('main miss=%s  closeout miss=%s' % (mm, mc))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('DONE mm=%s mc=%s' % (mm, mc))
raise SystemExit(1 if (mm or mc) else 0)
