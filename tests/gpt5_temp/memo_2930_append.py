# -*- coding: utf-8 -*-
"""Phase 2930 MEMO append: add the Phase 2930 section to
AGI_GPT5_MEMO.md (append-only). Title timestamp taken from
execution.json created (2026-09-19T15:44:15). Computes the
post-update ledger sha256-8 for the section registration.
"""
import hashlib

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md')
LP = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
      r'\atlas_ledger.json')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2930_memo_append_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


SECTION = '''
## Phase 2930: 方向偏置对照——反号探针镜像复测（margin 约定相对 + 幸存核 rho 约定不变）[2026-09-19 15:44]

### 目的（2929 接续候选 A）
2928 发现 L+ 类 0/4 幸存全灭并登记"词探针方向偏置待对照"；2927 的"方向约定无关"声明只对 margin 统计量成立（outer(s,s)）。本 Phase 一次前向（57 s）做镜像探针（注入 −dirs_word、readout Wo.T@(−dirs_word)）：B = einsum(r(d), Wo.T@d) 是 d 的二次型——精确线性预言 B_mirror == Bwd（双重变号抵消）；eps=1.0 有限注入下二阶项 H[d,d] 破坏之，破坏幅度即整个事件/骨架机器的方向约定敏感度。

### 设计（冻结口径）
- 2917/2927 协议 verbatim（SEED=2896、eps=1.0、pos 1、57 词、三条件、null_tids rng 序、o_proj 输入捕获、36 层），pass2 **三方向**：+dirs86（锚 a1-a4）、+dirs_word（复现锚 a5-a7 vs 2927 npz）、−dirs_word（镜像新数据；a8 断言 dirs_neg ≡ −dirs_word 且 G_neg ≡ −Gwd 逐位）。
- 锚 8/8 全过：a1 rel **3.16e-08**（第八次连续前向锚定）、a2 0.280360、a3 4.71e-08、a4 24/24 集合相等、a5 dirs_word 重建 diff **2.17e-08**、a6 Bwd 复现 rel **2.10e-08**、a7 Ewd==2927 且 pdiff **2.76e-08**、a8 逐位 ==。
- **P1（主检验）**：maxT（200 置换 rng2 2896 共享）于 B_mirror → E_mirror；E_mirror == Ewd(2927) => direction_flip_margin_invariant，否则 => direction_flip_margin_shifts。
- P2 镜像偏差：r 层 lin_r = ||r(−d)+r(d)||_F/||r(d)||_F 逐层；B 层 mir_err 格级；rho_mirror vs 2929 rho_grid dev。
- P3 骨架约定不变性：层内头置换 null（rng 2904）重放于 Bwd 与 B_mirror 秩向量；skeleton_mirror vs skeleton_2929 jaccard；L0 取证。
- P4 幸存 7 / 丢失 17（2928 seal verbatim）rho_2929 vs rho_mirror；L+ 类 {(26,6),(25,3),(24,23),(27,24)} 高亮。

### 结果
- **冻结判决：direction_flip_margin_shifts**——maxT 事件选拔是方向约定相对的。
- **P1 事件集重写**：E_mirror n=36 vs Ewd27 n=32，重叠 **20/48（jaccard 0.417）**，sign_M diff max **1.29**（margin 结构被镜像探针大幅重写）；top1 (7,19) 双口径幸存；**12 丢失**（含幸存核 (8,2) p 0.045→**0.204** 与 (20,8) 0.025→**0.090** 边界带）+ **16 新增**（含 L+ 类 **(25,3) 与 (27,24) 进入事件集**）。
- **P2 二次型破坏量化（登记级方法论发现）**：r 层 lin_r 中位 **1.1747**——偶阶非线性分量与线性分量同量级（线性预言 0）；层剖面 L4-L13 准线性（0.41-0.71）→ L30-L35 强非线性（**1.59-1.76**）；B 层 mir_err 中位 1.0192（镜像响应与原响应量级相当的重写）；**corr(lin_r_layer, mir_err_layer) = 0.9827**——机制链闭合：偶阶非线性 → 二次型破坏 → margin 重写。
- **P3 骨架部分约定不变**：skeleton_mirror 414 vs skeleton_2929 501，交 327（jaccard 0.556）；L0 退化伪影跨口径复现（全 32 头 rho≡0 经 0>=0 边界入选——纪律 12 的镜像确认）。
- **P4 幸存核 rho 约定不变（机制层核心发现）**：7/7 幸存核在镜像探针下保持高正 rho——|Δrho| 中位 **0.0154**、max 0.0703（(1,6) 0.6527→0.6988 反升；(21,6) 0.7370→0.7367 几乎不动）；**同时 2/7 丢失 maxT 显著性**——rho（响应结构）是稳定对象，maxT 是脆弱选拔器。L+ 深负 rho 跨约定保持：(25,3) **−0.6417→−0.6274**、(26,5) −0.4661→−0.5019——反相响应结构是真实的，不是伪象。
- **2928 L+ 0/4 全灭的修正**：maxT 层面部分约定相对（2/4 在镜像口径翻转进入）；(24,23)、(26,6) 双口径一致拒绝（约定稳健地拒绝）。

### 解读
1. **方向约定是 maxT 级主张的自由参数**：margin/sign-Gram 事件选拔对探针方向约定脆弱（jacc 0.417），任何"词探针口径的 maxT 显著集"必须在预注册时固定并对照方向约定；2927 的"约定无关"声明仅在 margin 统计量层面成立，事件集层面不成立。
2. **幸存核的本质在 rho 不在 maxT**（三连证据链第四环）：2928 幸存 = 双强度 + rho 不变 → 2930 rho 跨方向约定不变而 maxT 洗牌——**幸存核 = 约定不变的响应结构骨架**；(8,2)/(20,8) 的 maxT 丢失不动摇其成员资格（rho 稳定），反而支持 2929 结论"事件格是骨架的极端尾部采样"。
3. **eps=1.0 注入协议的非线性度首次量化**：lin_r 中位 1.17（偶阶项 ~ 线性项），深层达 1.76——2917→2930 全协议族的"注入响应"解释携带 O(1) 偶阶混合；B 是"二次型 + H[d,d] 修正"，镜像口径的 mir_err ~100% 是该修正的直接观测。浅层（L4-L13）准线性区（lin_r 0.4-0.7）的事件解释更干净，深层事件（如 (27,24)）处于强非线性区——层级分层解读的依据。
4. L+ 反相结构真实存在：(25,3)/(26,5) 的深负 rho 跨约定不变——L+ 类的响应结构与 lab0−lab1 探针方向反相（不是无结构）；maxT 对它们的拒绝在镜像口径翻盘（(25,3)(27,24) 进入）——**类不对称是"margin 口径 × 方向约定"的联合现象**。
5. 骨架 jaccard 0.556：响应结构骨架的主体（327 格）跨约定共享；差异部分集中在浅层与深层非线性区——与解读 3 一致。

### 方法论常数（新增）
- **镜像探针对照**：任何注入探针族（dirs_X 口径）的 maxT/骨架主张须配 −dirs_X 镜像对照（一次前向增量 ~50%）；B 的二次型结构使镜像偏差直接度量约定敏感度。
- **lin_r 非线性探针**：||r(−d)+r(d)||/||r(d)|| 作为注入线性度的标准诊断（eps 扫描的前置检查）；corr(lin_r, mir_err) ~0.98 表明 B 层偏差可由 r 层诊断预测。
- **rho vs maxT 双层登记**：maxT 显著集（脆弱、约定相对）与 rho 响应结构（稳定、约定不变）分层登记，选拔主张只给前者、结构主张只给后者。

### 执行史
- 主脚本一次运行成功（57 s：171 前向 pass1 + 3 方向 pass2 + 零前向分析；写前自查修正 2 处：Rmir_rank 残留伪分支、L_POS 集合与 2928 seal 对齐 4 元素）；seal 探针：幸存核 |Δrho| 量化、类归属跨口径表、lin_r 层剖面 + 机制链相关、丢失幸存者边界带取证、L0 复现、SHA 登记。

### 硬伤
- 镜像只在词探针族内对照方向约定，不改探针语义内容（组差构造不变）——"每极平衡对重构"（候选 A 的另一子选项）未做；约定敏感度可能被组差非对称进一步放大。
- eps=1.0 单点：lin_r 的 eps 依赖未扫（偶阶项 ~ eps 缩放预测未检验）；准线性区判定基于单 eps。
- 幸存核 n=7 小样本；jaccard 0.556 的 null 基线未做（骨架重叠的 null 校准未配，纪律 11 提示）。
- n=1 run（锚 8/8 bit 级缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2930_direction_flip_control.py: 99f29bc7
- execution.json: b701d7fa（created 2026-09-19T15:44:15）
- result.json: 8ecdbf29（final_verdict=direction_flip_margin_shifts，runtime 57 s）
- direction_flip_control.npz: cb655825（B_mirror、sign_M_mirror、p_maxT_mirror、rho_mirror_grid、mir_err_grid、lin_r_profile、skel_mask_mirror、p95_mirror、p95_bwd）
- 源：2927 probe_relativity.npz 84fec594；2929 response_structure_atlas.npz 57ed5651
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2930/direction_flip_control/
- Ledger：M2930_direction_flip_control 入账，measurements 68->69，L14 connects 36->37，ledger sha256-8 = __LEDGER_SHA__

### 接续（2931 候选）
- A（主选）：**骨架重叠 null 校准**——2930 P3 jaccard 0.556 的 null 基线（固定 Gram/秩结构 + 置换重放，零前向），判"骨架主体跨约定共享"是否超机会（纪律 11 直接应用）。
- B：eps 扫描线性度——lin_r 在 eps ∈ {0.1, 0.3, 1.0} 的缩放检验（偶阶 ~eps 预测；一次前向族，层子集）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- D：rho 骨架跨模型复现（glm4 双口径 + 镜像对照，一次前向）。
'''

ledger_sha = sha8(LP)
section = SECTION.replace('__LEDGER_SHA__', ledger_sha)

with open(P, 'r', encoding='utf-8') as f:
    old = f.read()
if '## Phase 2930:' in old:
    raise SystemExit('Phase 2930 section already present - abort')
with open(P, 'a', encoding='utf-8') as f:
    f.write(section)

with open(P, 'r', encoding='utf-8') as f:
    new = f.read()
n_lines = new.count('\n') + 1
title_ok = ('## Phase 2930: 方向偏置对照——反号探针镜像复测'
            '（margin 约定相对 + 幸存核 rho 约定不变）'
            '[2026-09-19 15:44]' in new)
tail_ok = new.rstrip().endswith(
    'D：rho 骨架跨模型复现（glm4 双口径 + 镜像对照，一次前向）。')
rep = ['title_ok: %s' % title_ok,
       'tail_ok: %s' % tail_ok,
       'total lines: %d' % n_lines,
       'ledger sha256-8 (post-update): %s' % ledger_sha]
with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
print('memo append OK')
