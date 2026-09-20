# -*- coding: utf-8 -*-
"""Phase 2929 MEMO append: add the Phase 2929 section to
AGI_GPT5_MEMO.md (append-only). Title timestamp taken from
execution.json created (2026-09-19T13:37:44). Also computes the
post-update ledger sha256-8 for the section's registration.
Disk recheck: title line + tail line verified by re-reading.
"""
import hashlib

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
     r'\AGI_GPT5_MEMO.md')
LP = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
      r'\atlas_ledger.json')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2929_memo_append_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


SECTION = '''
## Phase 2929: 全格响应结构图谱——探针不变骨架确立 + 事件耦合 [2026-09-19 13:37]

### 目的（2928 接续候选 A）
2928 把响应结构 rho 确立为幸存核的正确统计量（事件级）。本 Phase 零前向（0.9 s，无模型加载）将其推广到全格：rho(B86[h,:,l], B_word[h,:,l]) 遍历 1152 格，问三件事——探针不变骨架存在吗、多大、与事件选拔耦合还是正交。

### 设计（冻结口径）
- **P1 骨架图谱**：rho_grid[h,l] = Spearman(B86[h,:,l], Bwd[h,:,l]) over 57 词（平均秩）；层内头置换 null（rng base 2904，2000 置换/层，向量化 einsum：置换 32 头行的 Bwd 秩向量，保持 B86 配对破坏）；p_perm[h,l] = mean(rho_perm >= rho_obs)。**骨架（冻结定义）**：rho_obs >= 层内 null p95；大小检验 vs 5% 期望 57.6 格（lgamma 对数域精确二项）。
- **P2 事件耦合（主检验）**：事件格 = E17 ∪ Ewd（24+32−7=49）；rho(事件格) vs rho(背景 1103 格) 秩和 U 单侧 + 10000 标签置换（rng 2905）；coupled iff p <= 0.05 事件更高方向。
- **P3 描述性**：rho 十分位；骨架/背景 gap；事件格骨架归属。
- 锚 3/3：a1 E17 事件 rho 重算 vs 2928 rho_rows max diff **0.0**（bit 级复现）；a2 集合重建 24/32/7；a3 sign_M86 vs 2917 npz diff **0.0**。

### 结果
- **冻结判决：skeleton_event_aligned**——骨架存在且与事件选拔耦合。
- **P1 骨架存在（n_skel=501/1152，43.5%）**：5% 期望 57.6 格，观测 **8.7 倍**，精确二项 p 下溢为 0.0（lgamma 域仍为 0）；rho 全格 median 0.2039、p90 0.7297、max 0.9659；层内 null p95 中位 0.2967。
- **L0 退化伪影（登记级发现）**：L0 全部 32 头 rho=0.0000 且 null p95=0.0000（退化秩向量：spearman std 守卫与置换 einsum 分母守卫同归零），判据 rho_obs >= p95 的等号边界 0>=0 把 32 头全部收进骨架。真实骨架 = 501 − 32 = **469（40.7%）**，仍 8.1 倍期望，判决不变。
- **骨架层剖面**：峰在 L8-L11（22/23/24/19 of 32）与 L26（20）；L15/L33 谷（8/8）；深层 null p95 系统性更低（L1 0.439 → L35 0.231，深层头间响应更同质）。
- **P2 事件耦合通过**：事件格 rho 中位 **0.4862**（n=49）vs 背景 **0.1945**（n=1103），U=37096.0，置换 p=**0.0001**（1/10001 下界）——耦合。
- **分组 rho 中位**：survivor **0.8251** / lost **0.3774** / new **0.4743**——a1 锚下与 2928 P3（0.825/0.377）bit 级一致。
- **P3**：无硬 gap（骨架 min 0.0000 <= 背景 max 0.4862，软重叠连续分布）；事件格 33/49 在骨架内（67.3% vs 背景率 42.4%）；**survivor 7/7 全部在骨架（1.000）** vs lost 11/17（0.647）、new 15/25（0.600）。
- seal 取证 top-20 rho 格：最高 (1,12) 0.9659（非事件格）；幸存核 (14,9) 0.9658 居第二；top-20 中 15 格在 L8-L12 窗口。

### 解读
1. **探针不变骨架确立**：43.5%（去伪影 40.7%）的格其 57 词响应模式超出同层头间置换 null——响应结构的跨探针一致性是格级普遍性质，不是幸存核的稀有特例；maxT 事件选拔以 67% 采样骨架（背景 42%）。
2. **幸存核机制闭合（三连 Phase 链完成）**：2927 幸存核（重叠无信息修正前）→ 2928 幸存 = 双强度 + 响应结构不变（rho 0.825 vs 0.377）→ 2929 survivor 7/7 全部落在骨架内。**幸存核 = 骨架 ∩ 事件**：探针不变的响应结构就是幸存核的机制本体；"7 个"是巧合，机制才是本体。
3. **L0 等号边界教训**：退化秩向量使 rho_obs 与 null p95 同为 0，">=" 边界全收——骨架类判据必须加非退化门（如 rho_obs 的秩向量 std > eps 或 p_perm 严格小于 1）；与纪律 10（判据可达性先检）同族。
4. **无硬 gap = 选拔是程度量**：骨架与背景在 rho 上连续分布（软重叠），maxT 显著集是"骨架的极端尾部 + margin 结构"的混合采样——事件格与骨架格不是两类物体。
5. 骨架峰层（L8-L11）与 2928 null 热点层（L6/L8/L9/L19）部分重叠但非同集：热点由 margin-Gram（符号一致性）决定，骨架由响应模式跨探针一致性决定——两个选拔轴在 L8-L11 汇聚。

### 方法论常数（新增）
- **退化层等号边界检查**：格级/null 判据在退化统计量（std=0 守卫归零）下须加非退化门，等号边界是伪影之源。
- **层内头置换 null**：格级"跨条件一致性"的标配零假设——破坏配对、保持层内头间结构；与事件级 maxT、标签置换 U 并列第三件套。
- **响应结构 rho 图谱**：rho(B_probe1, B_probe2) 全格扫描 + 分层 null 选拔 = 电路骨架发现的通用流程。

### 执行史
- 主脚本 5 次运行（前 4 次运行错误如实登记：run1 rho_perm broadcast 形状、run2 精确二项 comb 大整数溢出 → lgamma 对数域、run3 局部 import log 遮蔽模块级日志函数 → UnboundLocalError、run4 numpy int64 JSON 序列化失败；run5 成功 0.9 s）；seal 探针：2x2 列联表、分组骨架率、top-20 格、L0 退化取证、层剖面 + SHA 登记。
- rerun 纪律执行：每次改脚本后 shutil.rmtree 删产物目录再跑（execution.json 先落盘冻结不变）。

### 硬伤
- 层内置换 null 检验的是"超出同层头间一般相似性"的一致性，不是绝对探针不变性；深层头间更同质使 null p95 更低，深层骨架入选更易（层间严格性不等）。
- rho 的高基线未校准（2928 已登记）：57 词共享词表 + 类差注入的共享成分可能抬高全格 rho 基线；469 的"骨架"是相对层内 null 的超额，非绝对高一致。
- L0 伪影在判决外登记（真实骨架 469 按同一定义重计），但产物 npz 的 skel_mask 保留 501 口径（immutable 原则，修正解读不入产物）。
- n=1 run（锚 bit 级 + 零前向缓解）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2929_response_structure_atlas.py: 82dff3df
- execution.json: fed4302e（created 2026-09-19T13:37:44）
- result.json: f761cf15（final_verdict=skeleton_event_aligned，runtime 0.9 s）
- response_structure_atlas.npz: 57ed5651（rho_grid、p_perm、null_p95、skel_mask、event_flag）
- 源：2927 probe_relativity.npz 84fec594；2928 survivor_core_anatomy.npz 2db8dfca；2917 event_atlas.npz 02343146
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2929/response_structure_atlas/
- Ledger：M2929_response_structure_atlas 入账，measurements 67->68，L14 connects 35->36，ledger sha256-8 = __LEDGER_SHA__

### 接续（2930 候选）
- A（主选）：**方向偏置对照**——dirs_word 反向组约定或每极平衡对重构（一次前向），检验 2928 L+ 0/4 全灭是否方向伪象；同时检验骨架对探针方向约定的稳健性（2927 已证 margin 符号翻转不变，rho 未证）。
- B（零前向）：h4 L1<->L19 复用子空间主角度（2918 唯一真复用通道，roadmap 遗留项）。
- C：rho 骨架跨模型复现——glm4 双口径一次前向 + 全格 rho 图谱（2927 协议移植）。
- D：骨架格功能判据——高 rho 格 vs 低 rho 格的因果消融读出差（一次前向），检验"响应结构不变"是否伴随"干预功能不变"。
'''

ledger_sha = sha8(LP)
section = SECTION.replace('__LEDGER_SHA__', ledger_sha)

with open(P, 'r', encoding='utf-8') as f:
    old = f.read()
if '## Phase 2929:' in old:
    raise SystemExit('Phase 2929 section already present - abort')
with open(P, 'a', encoding='utf-8') as f:
    f.write(section)

# disk recheck
with open(P, 'r', encoding='utf-8') as f:
    new = f.read()
n_lines = new.count('\n') + 1
title_ok = '## Phase 2929: 全格响应结构图谱——探针不变骨架确立 + 事件耦合 [2026-09-19 13:37]' in new
tail_ok = new.rstrip().endswith('D：骨架格功能判据——高 rho 格 vs 低 rho 格的因果消融读出差（一次前向），检验"响应结构不变"是否伴随"干预功能不变"。')
rep = ['title_ok: %s' % title_ok,
       'tail_ok: %s' % tail_ok,
       'total lines: %d' % n_lines,
       'ledger sha256-8 (post-update): %s' % ledger_sha]
with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
print('memo append OK')
