# -*- coding: utf-8 -*-
"""Phase 2931 MEMO append: add the Phase 2931 section to
AGI_GPT5_MEMO.md (append-only). Title timestamp taken from
execution.json created (2026-09-19T15:59:40). Computes the
post-update ledger sha256-8 for the section registration.
"""
import hashlib

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md')
LP = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
      r'\atlas_ledger.json')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2931_memo_append_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


SECTION = '''
## Phase 2931: 骨架重叠 null 校准——骨架超机会确立 + lin_r 分层结构 [2026-09-19 15:59]

### 目的（2930 接续候选 A）
2930 报告 jaccard(skeleton_2929, skeleton_mirror) = 0.556 并解读为"骨架主体约定不变"。纪律 11（2928 重叠无信息教训）要求任何重叠主张先过 null 校准——本 Phase 零前向（2 s）执行该校准，判"约定不变骨架"是否挣得。

### 设计（冻结口径）
- 数据源：2927（B86/B_word）+ 2929（rho_grid/skel_mask/null_p95）+ 2930（B_mirror/rho_mirror_grid/skel_mask_mirror）npz，无模型加载。
- 锚 4/4：a1 rho_grid 重算 diff **2.22e-16**；a2 rho_mirror 重算 diff **0.0**；a3 集合重建 501/414/交 327；a4 2929 null_p95 置换重放（rng 2904）diff **0.0**（置换语义逐位复核）。
- **非退化门（纪律 12）**：三方向（B86/B_word/B_mirror）57 词秩向量 std <= 1e-10 的格双侧排除——恰 32 格全部在 L0（exact-zero rho 格 = 32 = 退化格，无其他层受影响）；修正骨架 469/382、交 295、obs_jacc **0.5306**。
- **P1 主检验（冻结判决映射）**：独立双重层内头置换 null（rng 2906，R=1000；两口径各抽独立置换、骨架掩码随值重排——保留列内值多重集即层热点与骨架大小，破坏跨口径格配对）；obs >= null p95 => skeleton_overlap_above_chance；p50 < obs < p95 => borderline；obs <= p50 => chance_level。
- P1b sanity：同置换配对（两口径共用同一置换）应逐位复现 obs jacc——实现正确性检验。
- P2 格级配对（描述性）：Spearman(rho29, rhomir) 全格 + 分层 + 幸存核值。

### 结果
- **冻结判决：skeleton_overlap_above_chance**——骨架重叠超机会，"约定不变骨架"主张挣得。
- **P1**：null jaccard median **0.2711**、p95 **0.2953**、max 0.3173（1000 次无一到达观测）；obs **0.5306** 超过全部 null 复本，**P(null>=obs) = 0.0000**；超额 vs 独立性期望 +139.5 格（交 295 vs 独立期望 155.5，**ratio 1.897**）。
- **P1b sanity 通过**：同置换配对 max dev 0.00e+00（实现逐位正确）。
- **P2 格级配对**：Spearman(rho29, rhomir) = **0.6709**；最好层 L11 **0.9806**（准线性区近完全复现）、最差层 L32 **−0.4223**（强非线性区反转，lin_r 1.66）；**幸存核 7/7 双修正骨架成员**。
- **seal 分层结构（本 Phase 最重要的结构性发现）**：骨架约定不变性按 lin_r 分层——**准线性层（lin_r<0.9，L4-L17）镜像骨架几乎逐格复现**（L10 24/24/24、L9 23/23/22、L8 22/23/21、L11 19/20/19）；**强非线性层（lin_r>1.4，L24-L35）镜像侧崩塌**（L32 15/2/0、L31 15/3/1、L33 8/2/1）而原口径保留自己的格——**lin_r 是约定稳健性的预测因子**：偶阶非线性越强，镜像口径的 rho 结构被重写越彻底。
- P3 审计：退化格 32（全 L0）= exact-zero rho 格 32——2929/2930 的 L0 全 32 头骨架入选完全由退化等号边界贡献，其他层无退化（2929 修正口径 469 的正式确认）。

### 解读
1. **重叠无信息定律是选拔器特异的，不是普遍的**（纪律 11 的边界确立）：2928 maxT 显著集重叠恰在机会水平（null median 7 = obs 7）；2931 rho 骨架重叠 1.9 倍独立性期望、超全部 null 复本。**rho（结构量）挣得重叠主张，maxT（选拔量）不挣**——与 2930"maxT 脆弱/rho 稳定"双层登记合流为完整的统计地位表。
2. **约定不变骨架是分层对象**：准线性层（lin_r<0.9）的骨架跨方向约定几乎逐格不变（L8-L11 峰值区全部在内），强非线性层（lin_r>1.4）的骨架是约定相对的。2929"骨架 469"的正确解读 = **约 300 格（准线性区）约定不变 + 约 170 格（深层）约定相对**。
3. 幸存核地位再加固：7/7 在双侧修正骨架内且全部位于中浅层——幸存核既是骨架∩事件（2929），又跨方向约定不变（2930 rho），又超机会共享（2931）——**幸存核 = 语言电路的约定不变响应结构核心**。
4. lin_r 作为免费诊断：一次镜像前向的 lin_r 层剖面预测骨架约定稳健性（corr(lin_r, mir_err)=0.9827 于 2930），无需额外统计机器。

### 方法论常数（新增）
- **选拔器特异性重叠校准**：重叠 null 校准（纪律 11）的结论依赖统计量类型——选拔量（maxT/族校正显著集）的重叠默认机会水平；结构量（rho/连续图谱阈值选拔）的重叠可超机会，但必须实测 null 后才可主张。
- **lin_r 分层登记**：跨口径骨架/事件主张按层内 lin_r 分层报告（准线性层 vs 强非线性层），全局 jaccard 不足以定位共享结构。
- **非退化门正式化**：2929 纪律 12 的门在双侧骨架上执行（469/382 为修正后权威口径；501/414 为含 L0 伪影的历史口径，永不混用）。

### 执行史
- 主脚本一次运行成功（2 s 零前向：锚重算 + 置换重放 + gate + 1000 双重置换 + sanity）；seal 探针：null 分布细节、分层骨架剖面（与 lin_r 对照）、幸存核双成员、SHA 登记。

### 硬伤
- 层内置换 null 保留层热点、破坏头配对——头热点跨层一致性（某头跨层都强）未被保留，null 对"头热点驱动"的共享是保守的；但 P2 格级相关 0.67 + 分层剖面提供了独立于该近似的佐证。
- 镜像只在词探针族内（2930 硬伤延续）；eps 单点。
- jaccard 单一统计量：交/并的分解（本 seal 的层剖面）是补充而非预注册主检验的一部分。
- n=1 数据组（锚 4/4 bit 级 + 零前向缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2931_skeleton_overlap_null.py: a6438941
- execution.json: 30fc0e44（created 2026-09-19T15:59:40）
- result.json: 770f0080（final_verdict=skeleton_overlap_above_chance，runtime 2 s）
- skeleton_overlap_null.npz: 5307afe1（gate、S29_corrected、Smir_corrected、null_jacc）
- 源：2927 probe_relativity.npz 84fec594；2929 response_structure_atlas.npz 57ed5651；2930 direction_flip_control.npz cb655825
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2931/skeleton_overlap_null/
- Ledger：M2931_skeleton_overlap_null 入账，measurements 69->70，L14 connects 37->38，ledger sha256-8 = __LEDGER_SHA__

### 接续（2932 候选）
- A（主选）：**约定不变骨架的功能判据**——准线性层共享骨架格（~300 格）vs 深层约定相对格的因果消融读出差（一次前向，候选 D 复活并按 lin_r 分层），检验"约定不变"是否伴随"干预功能不变"。
- B：eps 扫描线性度——lin_r 在 eps ∈ {0.1, 0.3, 1.0} 的缩放检验（偶阶 ~eps 预测；一次前向族，层子集）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- D：准线性层骨架跨模型复现（glm4 双口径 + 镜像，一次前向；只主张 lin_r<0.9 层）。
'''

ledger_sha = sha8(LP)
section = SECTION.replace('__LEDGER_SHA__', ledger_sha)

with open(P, 'r', encoding='utf-8') as f:
    old = f.read()
if '## Phase 2931:' in old:
    raise SystemExit('Phase 2931 section already present - abort')
with open(P, 'a', encoding='utf-8') as f:
    f.write(section)

with open(P, 'r', encoding='utf-8') as f:
    new = f.read()
n_lines = new.count('\n') + 1
title_ok = ('## Phase 2931: 骨架重叠 null 校准——骨架超机会确立'
            ' + lin_r 分层结构 [2026-09-19 15:59]' in new)
tail_ok = new.rstrip().endswith(
    'D：准线性层骨架跨模型复现（glm4 双口径 + 镜像，一次前向；'
    '只主张 lin_r<0.9 层）。')
rep = ['title_ok: %s' % title_ok,
       'tail_ok: %s' % tail_ok,
       'total lines: %d' % n_lines,
       'ledger sha256-8 (post-update): %s' % ledger_sha]
with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
print('memo append OK')
