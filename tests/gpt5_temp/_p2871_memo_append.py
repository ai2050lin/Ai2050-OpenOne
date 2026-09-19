import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = '''

## Phase 2871（2026-09-18）：显著头三重身份解剖 —— 一致核心确认（零前向 0.2s）

**设计**：三种独立的"重要头"定义统一解剖——A = 2846 因果负载 top-64（mean drop）；B = 2864 η² 显著头（p_per_head<0.01，43 头 = 2868 B2s 同源）；C = 2859 bootstrap 稳定前缘 top-10。超几何富集（Bonferroni ×4）+ 2862 冻结分位角色分类 + 负载份额。

**产物**：`tests/glm5/phase2871_sig_heads_triple.py`（sha 91377ec6 [execution.json] / 59830517 [result.json] / bd056954 [sig_heads_triple.npz]），目录 `phase2871/sig_heads_triple/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| **U1** 交集富集 | A∩B=9（p=3.6e-4）/ A∩C=**10**（p≈0，C⊆A！）/ B∩C=4（p=3.0e-4）/ 三重=4 | **true → core_confirmed** |
| **U2** 核心角色（≥2 定义，15 头） | 形成器 2 / 放大器 **0** / 其他 13 | 与 2862 前缘纯形成器一致 |
| **U3** 负载份额 | 代数和 −0.26（**负值域失义**）；正归一化 **23.1%** | 字面 false（core_minor_load）；判据缺陷注记 |

**三重一致核心（4 头）：L5H25 / L23H7 / L23H29 / L26H4**；core2（15 头）含 **L13H30**（A∩C：因果负载 top1 兼稳定前缘，drop 0.1012 = core2 正负载的 1/5）。横跨因果+类方差+稳定三定义的交集全部超几何富集——**"重要头"不是单一协议的伪影，存在跨定义一致的真实核心**。

**两项登记**：
1. **2864 勘误注记**：`n_sig_heads_p95=102` 用了 `p_per_head >= 0.95`（错误尾向，描述性附带未进判决）；正确的 η² 显著大集合 = p<0.01（43 头）。
2. **负值域教训（第二次）**：mean_drop 697/1152 头为负（sum −1.91），凡"份额/占比"判据预注册时必须先声明归一化基准的正负处理。正归一化交叉验证：top64 = 51.6% 正负载，与 2846 逐位一致 ✓。core2 密度：1.3% 头数承载 23.1% 正负载。

### 接续

2872 主选：**属性轴因果 census**（2846 协议移植到属性对词表——属性轴机制侧定位，双轴关联第二样例；需完整预注册设计 + 词表构造，~1 前向/词）。备选：门控融合增长率曲线第三点（密度门控协议推广）。
'''

with io.open(MEMO, 'r', encoding='utf-8') as f:
    before = f.read()
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
print('APPENDED: %d -> %d lines' % (before.count(chr(10)), (before + section).count(chr(10))))
