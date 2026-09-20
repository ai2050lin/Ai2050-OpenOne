# -*- coding: utf-8 -*-
"""Append Phase 2933 section to AGI_GPT5_MEMO.md."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2933_memo_append_report.txt')

SEC = '''
## Phase 2933: 全格 CI 图谱——1120 格全覆盖消融与 rho/lin_r/CI 三方耦合 [2026-09-19 16:27]

### 目的与设计（预注册，execution.json 先落盘）
- 问题：2932 的承重带（L6-L12）/反向带（L28+）结论建立在 648/1120 抽样消融上——全覆盖后带状结构与 CI-rho-lin_r 三方耦合是否成立。
- 2932 协议 verbatim（真实消融：o_proj 输入 pos-1 头切片置零；57 func prompt batched；读出 dirs_word[35] 投影）；格集 = 全部门控非退化格 32×35 = **1120**（L0 按纪律 12 排除）。
- 判决映射（冻结）：p3a<=0.001 且正 且 p2b<=0.05 且方向正 => full_atlas_band_and_coupling_confirmed；p3a<=0.001 且正 => full_atlas_coupling_only；否则 full_atlas_unstructured。

### 锚（6/6 全过）
- a1 dirs_word 重建 2.17e-08（**第十次连续前向锚定**）；a2 确定性 0.0；a3 hook 有效性 1.4631；a4 掩码 469/382/295；a5 分离 185.70；**a6 跨相位复现：2932 的 648 格 CI_rel max abs diff = 0.00e+00（bit 级）**——消融测量机器全链路确定性验证。

### 结果
- **P2 带状结构决定性确认**：LOAD 带（L6-L12）中位 CI 0.005605 vs DEEP 带（L28-L35）0.002955（**×1.90**），层标签精确置换 p2b = **0.000000**（0/6435）；层中位 CI 与 lin_r 的 Spearman = **−0.6762**（p=1.0e-4）——**lin_r-CI 律：准线性层因果承重、强非线性层功能静默**（2932 抽样结论全覆盖复现且加强）。
- **P3 格级耦合弱/边界**：Spearman(CI, rho86) = 0.0989（p=1.4e-3，**超预注册阈值 1e-3 达 4e-4**）；Spearman(CI, rho_mirror) = 0.2326（p=2.0e-4）。冻结映射落 full_atlas_unstructured——判决如实登记；实质内容：带状分支决定性达成，格级 rho-CI 耦合真实但弱且口径依赖。
- **P4 seal 取证**：深带中位静默但有稀疏强离群（(15,34) CI 0.0404 为全格 max；深带 p90 0.0062 vs 承重带 p90 0.0092——承重带是"齐而不爆"，深带是"静而偶爆"）；层内 CI-rho86 耦合峰 L9 0.654 / L24 0.501 / L11 0.476（承重带内部结构-功能对齐最高）；骨架 vs 其余全覆盖 0.005771 vs 0.004877（×1.18，与 2932 抽样 ×1.124 一致）；幸存核层内 CI 排名 3..26（(14,9) 第 3、(20,8) 第 7、(1,6) 第 26）——幸存核是结构核心而非均匀功能峰。

### 判决边界登记（纪律 6）
- 冻结映射的 p3a 阈值 1e-3 预注册时未考虑 p2b 与 p3a 的分离强度；本次 p2b=0.0000 而 p3a=1.4e-3，映射无"带状确认+耦合边界"分支——verdict 按冻结映射登记为 full_atlas_unstructured，分解判读以 seal 为准。教训：多分支判决映射的阈值应先做判据可达性预估（纪律 10 的映射版）。

### 硬伤
- 格级 rho-CI 耦合弱：rho 是注入响应结构的层内选拔量、CI 是绝对量——两者量纲/归一不同，格级耦合弱可能是量纲问题而非结构问题（层内耦合 0.654 支持）。
- 深带离群格（(15,34)）提示消融读出对个别头极端敏感——单一位置（pos 1）消融。
- 消融读出单一（dirs_word[35] 投影 + logit 佐证）；n=1 数据组（a6 bit 级复现缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2933_full_atlas_ci.py: 0bd186df
- execution.json: b0d34cfe（created 2026-09-19T16:27:04）
- result.json: 88761d09（final_verdict=full_atlas_unstructured，runtime 82 s）
- full_atlas_ci.npz: 1ff6df21（s_base/s_abl 1120x57/cells/ci_rel/ci_logit/dirs_word）
- 源：2887 e4835a87；2927 84fec594；2929 57ed5651；2930 cb655825；2931 5307afe1；2932 b8a74702
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2933/full_atlas_ci/
- Ledger：M2933_full_atlas_ci 入账，measurements 71->72，L14 connects 39->40，ledger sha256-8 = b37707dc

### 接续（2934 候选）
- A（主选）：**承重带功能解剖**——对承重带（L6-L12）与深带离群格做双位置（pos 0+1）消融与条件分解（same/func/null 各自 CI），检验 lin_r-CI 律在多位置/多条件下稳定（一次运行）。
- B：eps 扫描线性度——lin_r 在 eps ∈ {0.1, 0.3, 1.0} 的缩放检验（偶阶 ~eps 预测；一次前向族，层子集）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- D：承重带跨模型复现（glm4 全格消融子采样，一次前向；只主张 lin_r<0.9 层）。
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
