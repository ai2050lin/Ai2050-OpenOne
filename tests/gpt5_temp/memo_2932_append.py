# -*- coding: utf-8 -*-
"""Append Phase 2932 section to AGI_GPT5_MEMO.md."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2932_memo_append_report.txt')

SEC = '''
## Phase 2932: 约定不变骨架的功能判据——真实因果消融读出差 [2026-09-19 16:16]

### 目的与设计（预注册，execution.json 先落盘）
- 问题链：2929 骨架存在 -> 2930 rho 约定不变/maxT 约定相对 -> 2931 骨架重叠 1.9x 独立性。缺环：约定不变骨架是否**功能承重**（functionally load-bearing）——消融其格是否比匹配背景格更大地改变行为读出，且 lin_r 分层是否预测功能影响。
- **真实因果消融**（非解析）：o_proj 输入 pos-1 头切片置零（头 (h,l) 不向残差流写），57 条 func 条件 prompt（[the, w] verbatim 2887）batched 全前向；读出 = 末位最终残差在 dirs_word[35] 上的投影 s_w；CI_raw = mean_w|s_abl - s_base|，CI_rel = CI_raw / mean_w|s_base|；次要 CI_logit（目标词 logit 变化）。
- 格集：共享门控骨架 295（2931 S29_corrected & Smir_corrected）+ 匹配背景 353 唯一格（每骨架格同层 2 非骨架头，rng 2907 置换发牌，去重）= **648 格消融**。
- 判决映射（冻结）：anchor fail => anchor_fail_all_void；median D>0 且 p1<=0.01 => skeleton_functionally_load_bearing；median D<=0 或 p1>0.05 => skeleton_epiphenomenal；否则 skeleton_partially_load_bearing。

### 锚（5/5 全过）
- a1 dirs_word 重建 diff 2.17e-08（**第九次连续前向锚定**）；a2 batched 基线复跑 rel 0.0（确定性）；a3 全头 L18 消融 mean|ds|=1.4631 > 0.01*scale=0.9229（hook 有效性）；a4 掩码计数 469/382/295；a5 基线分离 185.70（scale 92.29，lab0>lab1）。

### 结果
- **P1 主检验通过（判决依据）**：D_s = CI_rel(skel) - mean(2 匹配 bg)，median D = 0.000568，单侧符号置换（10000，rng 2908）p = 1.0e-4（可达最小值 1/10001）；骨架 CI_rel 中位 0.005771 vs 背景 0.005135（**x1.124**）——骨架功能承重但边际温和（~12%）。**冻结判决：skeleton_functionally_load_bearing。**
- **P2 lin_r 分层预测功能**：QL 层（lin_r<0.9）0.005994（n=196）vs DEEP 层（lin_r>1.4）0.004555（n=50），diff 0.001439，双侧标签置换 p = 7.0e-4——响应结构约定不变性与因果影响同向。
- **P3 CI 与 rho 耦合**：Spearman(CI_rel, rho29) = 0.2916（p=2.0e-4）、(CI_rel, rho_mirror) = 0.3543（p=2.0e-4）。
- **P4 seal 层剖面（带状结构）**：层内 top-half 命中 L6-L12 强（L9 19/22、L10 18/24、L8 15/21、L12 10/14），L28/L29/L34 **反向**（0/3、0/2、0/1）；全格最高功能格 (12,22) CI 0.02698 是骨架格；幸存核 7 CI 中位 0.00615 vs 骨架同伴 0.00576（2/7 高于同伴，不齐整——幸存核是结构核心而非功能峰值核）；CI_logit 同向（骨架 0.04996 vs 背景 0.04921）。

### 机制链第五环闭合
2929 骨架存在 -> 2930 rho 约定不变 -> 2931 重叠超机会 -> **2932：骨架因果承重，且 lin_r（约定稳健性预测子）同时预测功能影响**——响应结构骨架是真实功能电路而非副现象；消融证据与 rho/maxT 分层登记互补（结构主张 rho 层、选拔主张 maxT 层、功能主张 CI 层，三者现在都有独立判据）。

### 硬伤
- 效应量小（x1.124）：骨架格与背景格的消融影响分布大量重叠（top-10 混有背景格）——"承重"是分布级而非逐格级命题。
- 消融只置零 pos-1 单位置单 token 序列（协议一致性要求）；多头同时消融的交互效应未测。
- 读出单一（dirs_word[35] 投影 + 目标 logit 佐证）；行为任务更宽的读出未覆盖。
- 匹配背景 2:1 同层发牌，背景格可被多骨架格共享；MID 层（lin_r 0.9-1.4）只在剖面中呈现、未进 P2。
- n=1 数据组（锚 5/5 + a2 确定性缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2932_skeleton_functional_ablation.py: 7a3b9e00
- execution.json: 368e7843（created 2026-09-19T16:16:52）
- result.json: b6df5c9d（final_verdict=skeleton_functionally_load_bearing，runtime 52 s）
- skeleton_functional_ablation.npz: b8a74702（s_base/s_abl 648x57/cells/ci_rel/ci_logit/dirs_word）
- 源：2887 language_axis_mlp.npz e4835a87；2927 probe_relativity.npz 84fec594；2929 response_structure_atlas.npz 57ed5651；2930 direction_flip_control.npz cb655825；2931 skeleton_overlap_null.npz 5307afe1
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2932/skeleton_functional_ablation/
- Ledger：M2932_skeleton_functional_ablation 入账，measurements 70->71，L14 connects 38->39，ledger sha256-8 = df5a5f0a

### 接续（2933 候选）
- A（主选）：**带状结构取证 + 全层 CI 图谱**——把消融扩展到全部 1120 非退化格（一次运行 ~13 min），检验"L6-L12 承重带 vs L28+ 反向带"是否稳定复现，并与 rho/lin_r 做全格三方耦合图谱。
- B：eps 扫描线性度——lin_r 在 eps ∈ {0.1, 0.3, 1.0} 的缩放检验（偶阶 ~eps 预测；一次前向族，层子集）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- D：准线性层骨架跨模型复现（glm4 双口径 + 镜像 + 功能消融，一次前向；只主张 lin_r<0.9 层）。
'''

lines_before = sum(1 for _ in open(P, encoding='utf-8'))
with open(P, 'a', encoding='utf-8') as f:
    f.write(SEC)
lines_after = sum(1 for _ in open(P, encoding='utf-8'))
h = hashlib.sha256()
with open(P, 'rb') as f:
    for ch in iter(lambda: f.read(1 << 20), b''):
        h.update(ch)
rep = ['lines %d -> %d' % (lines_before, lines_after),
       'memo sha8: %s' % h.hexdigest()[:8]]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('memo ok')
