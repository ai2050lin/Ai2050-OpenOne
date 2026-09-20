# -*- coding: utf-8 -*-
"""Append Phase 2939 section to MEMO."""
import hashlib

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2939_memo_append_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines_before = len(open(P, encoding='utf-8')
                   .read().splitlines())

sec = """## Phase 2939: 旋转目标定位——null 重编码把能量从 dir35 平行分量搬到固定的近正交方向 v3 [2026-09-19 17:55]

### 原理与设计
2938 判决 subspace_rotation_retained 后遗留：词项结构在 8 维语言子空间内被搬到哪几个基方向？本 Phase 一次前向（零消融）补 2938 缺失的逐词坐标：c(w,k) = x_w . v_k（v_k 为 dirs_word 堆叠 SVD 前 8 主成分），2938 协议 verbatim。三检验（冻结判据）：P1 逐基结构保留（Spearman func vs 条件坐标跨 57 词，rho_med = 4 组 null 中位）；P2 能量占比迁移 share_k = mean_w c_k^2 / Σ，Δe_med + 逐词标签交换置换（rng 2920，10000）；P3 类均值位移 δc 分解。判决映射：max rho_med(U8) >= 0.7 且 max Δe_med > +0.02 => rotation_target_identified；max rho_med < 0.5 => rotation_target_not_found；否则 partial。

### 锚（6/6 全过）
- a1 dirs_word 重建 diff 2.17e-08（第十五次连续前向锚定）；a2 确定性 0.00e+00；a3 7.21e-06 / a4 6.26e-06 vs 2935
- **a5 跨相位 3.61e-16**：本 run alpha_k vs 2938 npz align_k（近 bit 级，SVD 基重建协议确定的直接验证）；a6 func sep 185.70

### 结果
- **P1 结构保留谱**：dir35 **0.8809**（方向塌缩后坐标秩结构仍保留——量级减半与秩保留是不同命题）、v2 0.8186、v5 0.7940、v1 0.7504；**v6 0.3864 是唯一结构丢失基**（占比最小 0.005→0.012）。max rho_med(U8) = v2 0.8186 >= 0.7。
- **P2 能量迁移（主发现）**：k* = **v3**，Δe_med **+0.1047**（占比 0.181→0.290），置换 p **9.999e-04**（可达最小量级）；流入 v5 +0.0689、v7 +0.0223、v4 +0.0212；流出 v1 **−0.1463**、v2 −0.0739。
- **P3 类均值位移固定性**：δc(null0) 与 δc(null1/2/3) 的 cos = **0.9989/0.9974/0.9995**——重编码方向是**上下文无关的固定方向**（上下文统计驱动，非 token 身份）；same 上下文位移模式完全不同（v1 −17 vs null +17）。
- 逐词位移：||dc||/||c_func|| 中位 **0.341**（8 维坐标内）；最大位移词 war（2937 重写离群词一致）。

### 机制结论（seal 定量形态）
SVD 前两主成分 v1/v2 就是 dir35 的正负分解（func 坐标 vs dir35 投影 Spearman **−0.96/+0.97**），而 v3 与 dir35 仅弱耦合（−0.38）——**null 重编码 = 能量从"dir35 平行分量"（v1/v2，流出 −0.22）搬到"近正交固定方向 v3/v5"（流入 +0.17），词项秩结构大半保留（dir35 rho 0.88）**。这就是 2938"子空间内旋转"的定量形态：不是漫散旋转，而是朝一个固定的、跨 null 组不变的子空间内方向的线性结构化转移。读出失败机制完整闭环：随机上下文把末位残差沿固定方向 v3 推离语言读出轴，|proj on dir35| 腰斩但语言信息仍在子空间内（秩结构保留）。

### 硬伤
- "v3 承接"是 8 维基内的占比陈述；SVD 基依赖 dirs_word 堆叠（组差方向）的任意旋转，v3 的"方向身份"无独立语义锚（需对 v3 做词级解码才可命名）。
- δc 固定性的 cos 是描述性（无置换检验）；same 条件位移模式不同仅为单组观察。
- P1 秩保留（rho 0.88）与量级减半（rho 比值 0.51）的关系未形式化（秩保留下量级重标定的模型未拟合）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2939_rotation_target.py: f14811a3
- execution.json: 45f76e71（created 2026-09-19T17:55:13）
- result.json: 11183c5c（final_verdict=rotation_target_identified，runtime 14.1 s）
- rotation_target.npz: 8bae7be6（coords 6x57x8 / proj_dir35 6x57 / fin_norm 6x57 / sing_vals 36 / Vt8 8x2560）
- 源：2887 e4835a87；2927 84fec594；2929 57ed5651；2930 cb655825；2931 5307afe1；2935 3b947b5d；2938 5f1bd256
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2939/rotation_target/
- Ledger：M2939_rotation_target 入账，measurements 77->78，L14 connects 45->46，ledger sha256-8 = e35a0607

### 接续（2940 候选）
- A（主选）：**v3 方向词级解码**——v3 上的词坐标（2939 npz 已有）对词属性（lab 类、词频、2937 高 CI 词集）做关联/解码，给 v3 语义锚；零前向。
- B：gamma 负偏移解剖（2937 npz 零前向）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（roadmap 遗留项）。
- D：承重带跨模型复现（glm4 双口径消融子采样，一次前向）。
"""

with open(P, 'a', encoding='utf-8') as f:
    f.write('\n' + sec)

rep = ['lines %d -> %d' % (lines_before,
                           len(open(P, encoding='utf-8')
                               .read().splitlines())),
       'title_ok: %s'
       % ('## Phase 2939: 旋转目标定位——null 重编码把能量从 '
          'dir35 平行分量搬到固定的近正交方向 v3 '
          '[2026-09-19 17:55]'
          in open(P, encoding='utf-8').read()),
       'ledger sha8: %s' % sha8(P)]
open(REP, 'w', encoding='utf-8').write(
    chr(10).join(rep) + chr(10))
print('memo done')
