# -*- coding: utf-8 -*-
"""Append Phase 2878+2879 section to AGI_GPT5_MEMO.md (append-only)."""
import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
REPORT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_memo_append_check.txt'

SEC = u"""

## Phase 2878+2879：语法轴 mlp 载体阳性 + 翻译轴真阴性（统一跨语言方向不存在）

**日期**：2026-09-18。**脚本**：`phase2878_syntax_trans_vocab.py`（零前向，
Gen1→Gen2）、`phase2879_syntax_mlp_spectrum.py`（16.0s，144 前向）。产物：
`phase2878/syntax_trans_vocab/` {exec `d7165ca0`, result `66b282e3`,
npz `0bb3ec1e`}；`phase2879/syntax_mlp_spectrum/` {exec `e1da7fa7`,
result `44a659bd`, npz `24eb4941`}。

### 2878：语法/翻译轴词表（双轨判决，Gen2）

- **Gen1 教训（同形词污染）**：`single_token_id` 优先 `' '+t`，`chat` 解析为
  英语"聊天" token 而非法语"猫"——翻译对方向被英文义污染；VG1 翻译族仅
  2/3（fr 4/8 对），vocab_legal=false。
- **Gen2 修正（2874 扩池先例，门线不动）**：翻译池扩容（fr 19/de 22/es 30
  对）+ 预注册同形排除表（chat/sol/pan/Mann/Gold/Winter/Hand/Bank 等 32 词，
  冻结于脚本）+ comparative 扩至 15 对。
- **判决**：VG1 语法 4/4、翻译 3/3 全过；VG2a max cos 0.1292 < 0.488；
  VG3 最差比 1.1654 < 3.0。
  **syn_legal=true**（number/gerund/comparative 存活，48 词 22 对；
  tense VG2b 0.1902 差 0.01 隔离）。
  **trans_legal=false —— 真阴性：去污染后 VG2b = fr −0.0007 / de −0.0029 /
  es 0.0011，精确零**。

### 2879：语法轴 mlp 响应谱（B3_syntax，48×10，2877 协议逐字移植）

| 判决 | 观测 | 结果 |
|---|---|---|
| **v1** 重算确定性 | max rel err = **0.000e+00** | **true** |
| kernel 上下文噪声 | 1.685e-3（描述性） | — |
| **E1** 轴检索 | acc = **0.7083** vs null p95 0.5208（1.82×，null mean 0.3887） | **mlp_carries_syntax_axis** |
| **E2** 同轴余弦边际 | 0.0744 vs null p95 0.0889（null mean 0.0041） | **margin_absent**（弱趋势不达门线） |
| E3 层分布 | L35 主导（0.2425），深层递增 | 类/属性轴同型 |

### 核心结论（重复三遍）

**mlp 臂载体定律第三轴复制：syntax E1 阳性（0.7083，零新 head 组件，同一
36×32 布局内换通道读出）——载体曲线 class 0.875 / attr 0.500 / syntax
0.7083；同时 E2 边际阴性揭示语法轴载体结构与类/属性轴不同：检索成功但无
全域同轴聚类（3 轴不平衡 null 基线高，p95 0.5208）。翻译轴则根本不存在统一
轴方向（VG2b≈0）——tied unembed 中 en→L 差被词形/概念差主导，语言恒定分量
≈0；"轴"这一语言族谱概念在翻译域不成立，在语法域以形态对成立。**

### 方法论入账

- **同形词消歧必须预注册**：多语词表中 `' '+t` 优先的 tokenizer 规则会把
  法语 chat/sol/table 解析为英语 token；扩池时同步冻结同形排除表（VG0）。
- 双轨判决（syn_legal / trans_legal 分开登记）避免一族真阴性拖垮另一族的
  测量——真阴性是结果，不是词表失败。

### 接续

2880 候选：A（主选）语法轴 drop 谱 census（2875 协议，48 词全谱，
通道分化四象限表补齐 syntax 象限）；B B3_class∪B3_attr∪B3_syntax 三族
联合词坐标与密度门控融合（增长率曲线三族交汇）；C 语法轴 E2 阴性解剖
（per-axis margin 分解：number/gerund/comparative 哪轴无边际）。
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SEC)

# disk recheck
with io.open(MEMO, 'r', encoding='utf-8') as f:
    lines = f.readlines()
ok = ('Phase 2878+2879' in lines[-1] or 'Phase 2878+2879'
      in ''.join(lines[-3:]))
with io.open(REPORT, 'w', encoding='utf-8') as g:
    g.write('total_lines=%d\n' % len(lines))
    g.write('tail_check=%s\n' % ok)
    g.write('last_line=%s\n' % lines[-1][:100])
print('appended')
