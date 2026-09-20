# -*- coding: utf-8 -*-
"""Append Phase 2938 section to MEMO."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2938_memo_append_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines_before = len(open(P, encoding='utf-8')
                   .read().splitlines())

sec = """## Phase 2938: 语言子空间主角度——塌缩严格限于 dirs_word[35] 单方向 [2026-09-19 17:46]

### 原理与设计
2937 判决 scale_collapse_rewrite 后遗留：末位残差是"旋转到语言子空间外"还是"仅离开 dirs_word[35] 单方向（子空间内部重编码）"？本 Phase 一次前向（零消融）直接测量：SVD 分解 dirs_word 堆叠（36 x 2560）得到正交语言子空间基 Vt，对齐度 alpha_k(x) = ||Vt[:k] x|| / ||x||（尺度不变纯方向量，规避纪律 16 scale 陷阱），k 网格 {1,4,8,16,36} + dir35 单方向对照；2937 协议 verbatim（pass1 dirs 重建 + 5 条件 x batch57 + 双 hook 捕获）。预注册判决映射：rho_k8 >= 0.6 且 rho_k8 >= rho_d35 + 0.2 => subspace_rotation_retained；rho_k8 < 0.5 => subspace_collapse_confirmed；否则 subspace_mixed。

### 锚（7/7 全过）
- a1 dirs_word 重建 diff 2.17e-08（第十四次连续前向锚定）
- a2 同 batch 组成确定性 0.00e+00；a3 proj_func vs 2935 7.21e-06；a4 proj_null0 vs 2935 6.26e-06
- a5 SVD 正交性 2.11e-15；a6 func separation 185.70
- **a7 跨相位 bit 级**：align_dir35 vs 2937 npz |proj|/fin_norm max diff = 0.00e+00

### 结果
- **P1 对齐梯度解离（主发现）**：rho（null/func 中位对齐比）dir35 **0.5091** < SVD PC1 0.8339 < k=4 0.9701 < **k=8 0.9991** < k=36 0.9957——塌缩严格限于 dirs_word[35] 单方向；子空间对齐（k>=4）完全保留。逐词 alpha_8：null0 min 0.173，**0/57 词离开子空间**（func min 0.151）。
- **P2 配对置换**：func vs null0 alpha_8 中位差 −0.0036，p = 0.66（rng 2919，10000 符号翻转）——子空间对齐差异不可探测。
- **P3 层剖面**：dir 比在 L18 塌至 **0.162** 而同层 alpha8 比 **0.942**；alpha8 比 >= 0.77 全层，dir 比 L6-L11 < 0.7——中层注意力在子空间内部重编码。L20 dir 比 1.187 > 1（深层部分恢复/反转）。
- SVD 奇异值剖面：top-8 能量占 **91.9%**（top-16 97.0%）——语言方向堆叠子空间低秩，8 维足以刻画。

### 机制结论
2937 的"方向塌缩"真实名字是**语言子空间内部的旋转重编码**：null 上下文经中层注意力把词项结构从 dirs_word[35] 方向搬到子空间内其他方向（~50% 线性保留 + 子空间内重组），语言信息未丢失、只是单方向投影失效。读出失败 = 单方向投影伪象。2936 rel 口径"放大"与本机制统一：子空间内旋转改变 |proj on dir35| 但不改变子空间总对齐。

### 硬伤
- 子空间由 dirs_word 堆叠定义（层间组差方向），不是数据驱动的词表征流形；PCA 于残差本身可能给出不同流形。
- "重编码到哪些方向"未定位（子空间内旋转的目标方向未测）；gamma 负偏移来源仍未解剖。
- P3 层剖面是描述性（无逐层置换检验）；主判决依赖中位比阈值（0.5/0.6 冻结门槛）而非 p 值。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2938_subspace_angles.py: b7f9bf71
- execution.json: 56632231（created 2026-09-19T17:46:06）
- result.json: 17d6ea93（final_verdict=subspace_rotation_retained，runtime 14.1 s）
- subspace_angles.npz: 5f1bd256（align_k 6x5x57 / align_dir35 6x57 / fin_norm 6x57 / sing_vals 36 / 层剖面 4x36 x2）
- 源：2887 e4835a87；2927 84fec594；2929 57ed5651；2930 cb655825；2931 5307afe1；2935 3b947b5d；2937 518cb922
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2938/subspace_angles/
- Ledger：M2938_subspace_angles 入账，measurements 76->77，L14 connects 44->45，ledger sha256-8 = 27d6dbf3

### 接续（2939 候选）
- A（主选）：**子空间内旋转目标方向定位**——null 条件下残差在子空间 8 维基上的坐标分解（零新前向，2938 npz 已含全部对齐数据可先析；补一次前向可逐基投影）——词项结构被搬到哪几个基方向。
- B：gamma 负偏移解剖（null 上下文读出偏置来源，2937 npz 零前向初探）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（roadmap 遗留项，方法与 2938 共通）。
- D：承重带跨模型复现（glm4 双口径消融子采样，一次前向）。
"""

with open(P, 'a', encoding='utf-8') as f:
    f.write('\n' + sec)

rep = ['lines %d -> %d' % (lines_before,
                           len(open(P, encoding='utf-8')
                               .read().splitlines())),
       'title_ok: %s'
       % ('## Phase 2938: 语言子空间主角度——塌缩严格限于 '
          'dirs_word[35] 单方向 [2026-09-19 17:46]'
          in open(P, encoding='utf-8').read()),
       'ledger sha8: %s' % sha8(P)]
open(REP, 'w', encoding='utf-8').write(
    chr(10).join(rep) + chr(10))
print('memo done')
