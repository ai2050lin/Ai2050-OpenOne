# -*- coding: utf-8 -*-
"""Append Phase 2937 section to AGI_GPT5_MEMO.md."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2937_memo_append_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


sec = """
## Phase 2937: scale 塌缩机制解剖——方向重写否定能量缩放 [2026-09-19 17:35]

### 原理与问题
2936 发现 null 上下文使基线读出量级塌缩（scale 92.29 → 37-49）但绝对消融扰动反而缩小，机制待解。三个候选：M1 能量缩放（残差能量缩小、词项结构保留，s_null ≈ β·s_func，β∈(0,1]）；M2 旋转（能量不变、方向偏离 dirs_word[35]）；M3 重写（读出被上下文改写，β 低）。一次前向（14 s，无消融）：pass1 dirs_word 重建（57 单前向 verbatim）+ 5 条件 baseline batch57（func/same/null0-3，pre_attn hook 逐层捕 pos-1 attn 输入 + final pre-norm 残差）+ P4 零前向 embedding 查表。

### 锚（6/6，两次跨相位 bit 级级）
- a1 dirs 重建 2.17e-08（**第十三次连续前向锚定**）；a2 determinism 0.00e+00
- a3 proj_func vs 2935 s_base **7.21e-06**；a4 proj_null0 vs 2935 **6.26e-06**（跨相位近 bit 级）
- a5 masks 469/382/295；a6 func sep 185.70（= 2935 精确复现）；scale 本 run 92.291/41.662 = 2935 精确值

### 主结果
**P1 能量不塌**：逐层残差范数比（null/func）0.94-1.07，全层无 <0.8 crossing——2936 的 scale 塌缩不是能量塌缩。**P3 方向塌**：cos(final, dirs_word[35]) median func 0.045 → null 0.002-0.024（比 0.05-0.24），4 组全部方向主导（|log norm| < |log cos|）；绝对 cos 显示语言方向即使 func 下也只解释最终残差 ~4.5%（小分量）。**P2 重写**：s_null = β·s_func + γ 跨词拟合 β 0.43-0.55（median 0.4886）、γ −11.6..−16.1、R2 0.74-0.86；same β 0.60。判决按冻结映射落 **scale_collapse_rewrite**。**P4 排除 token 身份范数解释**：null tid embed 范数 1.071±0.215 vs 词 1.085±0.073（p 0.64）。

### seal 取证（三项新发现）
1. **sep 比剖面 U 形**：语言信号塌缩最深在 L8-L16（0.11-0.36），深层部分回升（L32-L35 0.41-0.61）——**中层注意力是重写主战场**，恰是 func sep 自身峰值层（L6-L10：3.5-7.3，语言信号在此建立）。
2. **重写类不对称**：lab1 类词读出结构高保留（类内 Spearman 0.75-0.84）vs lab0 类被重写（0.13-0.24）——与 2928 L+ 类不对称/2930 L+ 方向偏置呼应：**lab0 读出更依赖上下文语义锚定**。
3. 重写离群词含高 func CI 词（light 194→131、war 198→18、city 202→33）——大读出词被重写最狠，与 2935 幸存核 L6 型高放大一致。

### 机制结论
上下文统计效应的 baseline 侧机制 = 中层注意力驱动的末位残差**旋转**（偏离语言方向，cos 0.045→0.01 量级）+ ~50% 词项保留 + 负偏移；不是能量缩放、不是 token 范数效应。2936 scale 塌缩的真实名字是"**小 cos 分量的方向塌缩**"。与 2934/2935 消融侧（rel 口径"放大"）合起来：随机上下文把读出系统从"语义锚定模式"切到"无锚定模式"——语言分量占比缩小使相对灵敏度（CI/scale）上升而绝对扰动略降。

### 硬伤
- dirs_word[35] 只是语言子空间一个方向；"旋转"是相对该方向的投影塌缩，不是完整子空间角度测量（P53/P54 遗留的子空间主角度可补全）。
- γ 负偏移 −11..−16 的来源未解剖（可能是 null 上下文的平均读出偏置）；U 形回升的深层机制未测。
- 类不对称相关（lab0 vs lab1）是描述性，未做置换检验。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2937_scale_collapse.py: 2e5cd727
- execution.json: 13b3b60e（created 2026-09-19T17:35:16）
- result.json: 5b09c835（final_verdict=scale_collapse_rewrite，runtime 13.7 s）
- scale_collapse.npz: 518cb922（attn_norm/attn_proj 5x36x57/proj 5x57/fin_norm 5x57）
- 源：2887 e4835a87；2927 84fec594；2929 57ed5651；2930 cb655825；2931 5307afe1；2935 3b947b5d
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2937/scale_collapse/
- Ledger：M2937_scale_collapse 入账，measurements 75->76，L14 connects 43->44，ledger sha256-8 = 4bbea246

### 接续（2938 候选）
- A（主选）：**语言子空间角度完整测量**——dirs_word 前 k 层堆叠子空间（非单方向）的 principal angles：func vs null 下末位残差到语言子空间的对齐度（检验"旋转到子空间外"还是"仅单方向塌"），一次前向。
- B：γ 负偏移解剖（null 上下文读出偏置的来源，零前向可初探）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（roadmap 遗留项，与 A 方法共通）。
- D：承重带跨模型复现（glm4 双口径消融子采样，一次前向）。
"""

with open(P, 'a', encoding='utf-8') as f:
    f.write('\n' + sec)

rep = ['memo lines now %d'
       % len(open(P, encoding='utf-8').read().splitlines()),
       'memo sha8 %s' % sha8(P)]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('OK memo 2937', flush=True)
