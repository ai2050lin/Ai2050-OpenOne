# -*- coding: utf-8 -*-
"""Append Phase 2936 section to AGI_GPT5_MEMO.md."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2936_memo_append_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


sec = """
## Phase 2936: 语义上下文抑制律形式化——锚定模型否定与 scale 口径审计 [2026-09-19 17:21]

### 原理与问题
2935 发现 null 上下文 CI"放大"（LOAD x1.83、DEEP x2.00）且放大梯度反转（放大最多处恰是 CI 最大处）。本 Phase 形式化"锚定假说"：语义上下文把消融敏感度压低一个近似常量的加性基底（supp = a + b*ci, a>0，则 amp = 1 + a/ci + b，梯度反转是推论），对比乘性模型（supp ∝ ci）。**零新前向**，全部分析复用 2935 npz（func + 4 组重采样 null，raw CI = ci_rel x scale 可恢复）+ 2934 npz（same/func/null x 3 配置）+ 2930 lin_r + 2931 masks + 2933 func CI。预注册冻结：execution.json 先落盘（rng 2917 截距置换 10000 次/组）。

### 锚（5/5，两次 bit 级）
- a1：2935 func ci_rel vs 2933 npz **max abs diff = 0.00e+00**（bit 级）
- a4：2934 pos1/func raw CI vs 2935 func raw **max abs diff = 0.00e+00**（bit 级，跨相位 batch57 同组成确定性）
- a2 格网三方一致；a3 源 SHA 链与 2935 execution 一致；a5 masks 469/382/295

### 主结果
**P1 加性锚定模型否定**：supp_raw = raw_null − raw_func 与 raw_func 的 Spearman 为**负**（4 组 −0.3232/−0.4168/−0.3517/−0.3552，median −0.3535）；R2_linear 仅 0.14-0.20（电池门槛 0.8）；截距 a>0 显著（p_a 1e-4）但模型解释力远不足；**supp_raw<=0 的格 750-805/1120**——raw 口径下多数格 null 扰动并不比 func 大。P3 same 条件同样否定（rho −0.2668、R2lin 0.0477）。判决按冻结映射落 **anchoring_not_established**。

**seal scale 口径审计（本 Phase 登记级发现）**：
1. **"null 放大"完全是分母驱动**：null/func scale 比 0.41-0.54（基线读出量级在 null 上下文塌缩 92.29 → 37-49）；raw 口径 amp median **0.86-0.90 < 1**（null 绝对扰动反而缩小，LOAD 缩更多 0.835 vs DEEP 0.919），rel 口径 1.65-2.13 > 1。2934/2935 的全部 CI"放大"结论是 **rel 口径陈述**；"放大梯度反转"叙事是 scale 塌缩的 rel 口径伪象。rel 口径 supp_rel 与 ci_func 相关 +0.30..+0.53，与 raw 口径符号相反。
2. **幸存核 raw 口径二分**：(5,6) 1.61 / (8,2) 1.42 / (21,6) 1.41 / (7,19) 1.29 / (1,6) 1.14（raw 增强型）vs (14,9) 0.50 / (20,8) 0.58（raw 腰斩型）——两类成员对上下文统计的绝对敏感性方向相反。
3. 2934 same/func raw amp 0.9302（语义上下文 same 也使 raw 扰动略小于 func）。
4. 层内 rho(supp, ci_func) median −0.5524（层内更大 CI 的格 raw 缩得更少）。

### 结论
加性与乘性模型双双否定；真实结构是"分母塌缩"——语义上下文的效果主要是把基线读出量级锚定放大（scale 92→41），而绝对消融扰动在无语义上下文时反而缩小 ~11-14%。lin_r-CI 律本身（band 结构方向）双口径保持，但其所有幅度陈述必须带口径标签。教训延伸纪律 16：**倍率主张必须登记分母口径**。

### 硬伤
- 纯 rel/raw 二分：未测第三口径（如 per-word 归一）；scale 塌缩的机制（为何 null 上下文使 |s_base| 减半）未解释——2937 直接候选。
- supp 的噪声底：raw CI 含前向数值噪声（bit 级复现排除漂移但不排除系统偏差）；负 supp 格集中在 L7-L30 中段的机制未解剖。
- run 前 seal 脚本 rel 公式双重除法 bug（两处 Edit 修复后重跑，主脚本不受影响）。

### 文件与 SHA256-8
- 脚本 tests/glm5/phase2936_anchoring_law.py: d948a623
- execution.json: a38e8fcc（created 2026-09-19T17:21:46）
- result.json: 87003d1e（final_verdict=anchoring_not_established，runtime 2.1 s）
- anchoring_law.npz: 1b3eaa93（cells/supp_null0/supp_null1/supp_same/raw_func/wl_rhos）
- 源：2927 a0e1a7dd 系（execution sources 记录全链）
- 产物目录 tests/glm5/result/rdc_query_construction_20260913/phase2936/anchoring_law/
- Ledger：M2936_anchoring_law 入账，measurements 74->75，L14 connects 42->43，ledger sha256-8 = 72e60a0b

### 接续（2937 候选）
- A（主选）：**scale 塌缩机制**——s_base 分解（|s| 词位项 vs 上下文项，norm/方向分解，零前向可初探 + 一次前向验证）：为何 null 上下文使基线读出量级腰斩？
- B：eps 扫描线性度（lin_r ∈ {0.1, 0.3, 1.0} 偶阶 ~eps 缩放，一次前向族）。
- C（零前向）：h4 L1<->L19 复用子空间主角度（roadmap 遗留项）。
- D：承重带跨模型复现（glm4 双口径消融子采样，一次前向；只主张 lin_r<0.9 层 + raw/rel 双口径）。
"""

with open(P, 'a', encoding='utf-8') as f:
    f.write('\n' + sec)

rep = ['memo lines now %d'
       % len(open(P, encoding='utf-8').read().splitlines()),
       'memo sha8 %s' % sha8(P)]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('OK memo 2936', flush=True)
