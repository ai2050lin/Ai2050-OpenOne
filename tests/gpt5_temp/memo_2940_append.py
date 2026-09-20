# -*- coding: utf-8 -*-
"""Append Phase 2940 section to AGI_GPT5_MEMO.md."""
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5'
     r'\docs\AGI_GPT5_MEMO.md')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2940_memo_append_report.txt')

created = '2026-09-19T18:08:38'

sec = """
## Phase 2940: v3 方向词级解码与层归属（v3_decode） [""" \
    + created + """]

### 原理与设计
2939 确立 rotation_target_identified（null 重编码能量流入 v3，delta_e_med +0.1047, p 9.999e-04）且 2939 硬伤明示"v3 的方向身份无独立语义锚"。2940 零前向解码：2939 npz coords (6 条件 x 57 词 x 8 基) + 2927 dirs_word（SVD 层剖面）+ 2887 labels_lang + 2937 逐词 proj（重写位移属性）。P1 v3 层归属 w_li = s3*U[li,2]；P2 位移解码四路（d3(w) = 4 组 null 的 delta-c3 中位）：类轴（标签交换置换 rng 2921）、概念锁定（组内方差 ICC，组置换 rng 2922）、尺度锁定（Spearman vs |proj_func|，rng 2923）、重写链接（vs 2937 位移，rng 2924），各 10000 置换；P3 v3 坐标语义。判决映射（冻结）：P2a p<=0.01 => v3_decoded_class_axis；elif |rho|>=0.4 且 p<=0.01 => v3_decoded_scale_locked；elif P2b p<=0.01 => v3_decoded_concept_locked；else => v3_decoder_not_established。

### 锚与 correction_note（run1 锚失败如实登记）
run1 verdict anchor_fail_all_void：a1/a2 阈值 1e-10 与 a5 阈值 1e-9 均判据不可达（纪律 10 映射版案例）——a1/a2：2939 SVD 用其前向重建 dirs_word，本 Phase 用 2927 npz dirs_word，跨相位 bf16 噪声（2.17e-08，2939 a1 锚值）传播进 SVD（Vt8 diff 3.04e-08），可达量级 1e-6；a5：2939 result.json P3 存 round(v,3)，与舍入值比较可达容差为半格 5.1e-4。修正阈值 + correction_note 入 PREREG 后 run2（注：run2 中两次 Edit 因编辑竞态未落真实磁盘，Grep 复核发现后重做——磁盘复核纪律再次生效）。run3 锚 5/5：a1 3.04e-08 / a2 4.81e-09 / a3 bit 级 / a4 vs 2937 proj 0.00e+00（跨相位）/ a5 2.21e-04。

### 结果
- **P1 层归属（机制定位）**：v3 = 双极方向——中层 L14-L18 正权重（+0.39..+0.62，top3 L16/L17/L18 ~0.62）+ 早层 L1-L10 与深层 L24-L35 全负（−0.4..−0.5）；有效层数 16.1，L15-18 质量占比 0.249。**v3 由中层重写主战场拥有**（2937 sep 塌缩最深 L8-L16、2938 L18 dir 比塌至 0.162）——三 Phase 层证据互锁。
- **P2 四路解码全部失败（判决落点）**：类轴 obs +5.22 p 0.296；概念 ICC 0.389 p 0.420；尺度锁定 rho 0.127 p 0.352；重写链接 rho 0.099 p 0.461。逐 null 类差同号（+3.3..+7.1）但词内方差支配——**v3 位移不是任何词属性的函数：均质固定方向推进**。same 条件类差反向（−12.7）。
- **P3 坐标语义（重要负结果细分）**：c3(func) 类分离显著（中位 lab0 52.4 vs lab1 82.3，p 7.0e-04）**但是语言混淆**——rho(c3, lab) 在 en 内部 = 0.0000 (n=22)：分离由语言分组驱动，非语义轴。极端词：低段 sea/foot/water/star(39-44)、高段 nuit/hombre/rey/casa/libro(121-132)。

### 结论
v3 的真名：**重编码机器的固定推进方向**——中层（L14-L18）拥有、词属性盲（类/概念/尺度/重写幅度均不预测位移）、语言读出轴的近正交补方向。null 上下文使中层注意力把所有词的末位残差沿 v3 均质推离 dir35 轴；v3 本身不携带可解码的词级语义。2936→2940 六环机制链完整：scale 塌缩（2936 rel 口径）→ 方向重写（2937）→ 子空间保留（2938）→ 目标方向 v3（2939）→ v3 层归属与属性盲性（2940）。

### 硬伤
- 概念 ICC 的组置换只覆盖 size>=2 的组（29 对），unique-ck 单词不入检验。
- "词属性"集限于 lab/lang/ck/幅度/重写位移；词频与词义范畴未测（无词频资源）。
- v3 双极层结构（早/深层负）的机制解释（为何早层反向参与读出轴）未检验——描述性登记。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2940_v3_decode.py: 02247d1e
- execution.json: 1fda5eb9（created 2026-09-19T18:08:38）
- result.json: 11dd5d1f（final_verdict=v3_decoder_not_established，runtime 1.5 s）
- v3_decode.npz: 4f5d8fa8（d3 57 / d_nulls 4x57 / c3_func 57 / w_li 36）
- 源：2887 e4835a87；2927 84fec594；2937 518cb922；2939 8bae7be6
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2940/v3_decode/
- Ledger：M2940_v3_decode 入账，measurements 78->79，L14 connects 46->47，ledger sha256-8 = b3f6b1b9

### 接续（2941 候选）
- A（主选）：v3 因果验证——沿 v3 注入（+delta 与 −delta）观察 dir35 读出恢复/加剧，直接因果确证"v3 推进 = 读出失败机制"（一次前向族，复用 2927 注入协议）。
- B：gamma 负偏移解剖（2937 npz 零前向）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（roadmap 遗留项）。
- D：承重带跨模型复现（glm4 双口径消融子采样，一次前向）。
"""

with open(P, 'a', encoding='utf-8') as f:
    f.write('\n' + sec)

rep = ['appended section Phase 2940',
       'title line: ' + sec.splitlines()[1]]
with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
print('OK memo 2940', flush=True)
