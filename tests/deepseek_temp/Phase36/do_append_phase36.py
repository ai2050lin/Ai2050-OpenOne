# -*- coding: utf-8 -*-
"""Append Phase 36 (E4/E4b) entry to AGI_DEEPSEEK_MEMO.md (append-only)."""
import hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'
before = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]

entry = """
## Phase 36: E4/E4b 词频 x 嵌入有效维度——「常用词满秩、生僻词低秩」双向否证 + 自纠 C2 的 K 失配判据（零 GPU；用户指令队列外插入）[2026-10-03 05:45]

> 类型：**探索性探针（E 系列，零 GPU，仅读 embed_tokens 行）**。触发：用户问「常用词（好/的）的词嵌入是不是满秩的，生僻字（麒麟）是不是低秩的」。授权：用户直接指令，宪法 30-Phase 队列外插入（同 E1/E2/E3 先例，探索性、不作确认性结论）。
> seal：`tests/deepseek/Phase36/E4_design_seal.json`（b1e616c2，观测前冻结词表/指标/判据）。脚本 `e4_freq_rank_probe.py`（077d946a）、`e4b_domain_control.py`（ecd8004e）。报告 4 件：`tests/deepseek_temp/Phase36/e4_report_{qwen3-4b(68a66e5e),qwen2.5-3b-instruct(fee79b52),glm4-9b-chat-hf(87357034)}.txt`、`e4b_domain_control_report.txt`（3992557a）。追加前 memo sha8 5be5bf12（685,649 B）。

### 1. 操作化（单向量无"秩"，必须先定义）

- **A 向量级有效维度**：p_i = v_i^2/Σv_j^2；PR = 1/Σp_i^2（高斯随机期望 (d+2)/3）；熵维 = exp(H(p))；top10 份额；max 份额。
- **B 组级矩阵有效秩**：K x d 行矩阵 SVD 谱（raw + 去组均值 centered）；PR_spec、熵秩、数值秩（ε=1e-2）；组内两两 cos；K=20 子采样 x50。

### 2. 实际结果（3 模型：qwen3-4b d=2560 / qwen2.5-3b d=2048 / glm4-9b d=4096；合计 22s 零 GPU）

**(a) 向量级：没有任何一组是"低秩向量"。** qwen3-4b：HIGH PR 中位 901 [879,937]、MID 878、RARE 815 [784,846]、随机 CJK 844、高斯带 854±27——全部落在数百有效维（d=2560）。用户点名：好 PR=907/熵维 1287，的 PR=918/熵维 1376，麒 831/1216，麟 857/1244，麒麟(单 token id=113981) 867/1236。

**(b) 向量级方向与假设相反（qwen3-4b +10.5%、qwen2.5 +7.9%，均 p<1e-5，rank-biserial -0.82~-0.90）。** 常用词的分量能量比生僻字**更摊开**（更接近均匀幅度的码），生僻字略更集中（top10 0.043 vs 0.037）且更接近高斯随机带（带内比例 57-79% vs 随机 CJK 应约 95%——介于随机与常用之间，即"训练过但欠训练"）。C1 判据：qwen3-4b 达 10% 门槛成立、qwen2.5 未达（7.9%）但方向一致显著。

**(c) 组级：E4 初判"RARE 低秩 11.5/14 vs HIGH 18.1/20（比 1.57，C2 反向）"——被 E4b 对照臂当场否决为 K 失配伪影。** PR_spec 受 K 上界约束，14 与 20 不可直接比。E4b 匹配 K=20 后：低频器物域 A2=18.4、低频神兽域 A3=17.8、中频动物域 A1=18.3、HIGH=18.0——**四组同档，无组级秩差异**。RARE(14) 数值秩 13/13（ε=1%），无线性退化。**"生僻字低秩"与"常用词满秩"在组级均不成立。**

**(d) 真实的组级差异是域相干性，不是频率。** 组内 raw cos：HIGH（跨域虚词）+0.065 < A2 器物 +0.111 ≈ A1 动物 +0.120 < A3 神兽 +0.141；最近邻 cos：HIGH 0.206 < RARE 0.316。同语义域的词（无论频率高低）互相更近——把频率与语义域分离后，"生僻词抱团"主要反映**它们取自同一个小语义域**，而非生僻本身。

**(e) 模型间差异显著。** glm4-9b 词表对生僻字覆盖差（20 选 3 存活，C1-C3 跳过）；glm4 嵌入整体更各向同性（随机 CJK 组内 cos +0.013 vs qwen +0.077；范数 0.1 vs 1.1），且 glm4 的 HIGH PR=1340 **低于**随机 1366（qwen 相反）——向量级细结构是**模型特异**的，"频率-有效维"关系不跨族。

### 3. 分析结论

1. **用户假设双向否证**：常用词不是"更满秩"（向量级略更摊开但组级无差异）；生僻字不是"低秩"（向量级数百有效维、组级匹配 K 后满档、数值秩满）。
2. **自纠登记（R1 复核纪律的现场执行）**：C2 判据设计缺陷——预注册时未规定 PR_spec 的跨 K 可比性，E4 首跑产出"反向"假阳性，E4b 域匹配臂（预注册判据"若 A1 也低秩则频率结论收回"）触发修正。教训：**凡比 PR_spec/熵秩，必须先固定 K 或同 K 子采样**。
3. **机制拼图增量**：嵌入词典的索引页——常用词页码彼此散开（近正交索引、相干性 0.065），同域词页码成簇（0.11-0.14），生僻欠训练词介于随机初始化与常用码之间（带内 57-79%）。与 E1"实例词类谱塌缩"、3139"读出不读行"互洽：生僻词连**身份索引的可分性**都差，其语义必须在层计算中才可读。
4. **RDC**：不新增公式；为"嵌入=一级词典"提供词典几何的量化注脚（索引密度分层：常用近正交 / 同域成簇 / 欠训练趋随机）。

### 4. 硬伤

1. 词频分组为人工判定，非语料频率实测；HIGH 与 RARE 的"频率差"混着"虚词 vs 实词"的词性差（E4b 部分缓解：A1 实词组仍近满秩）。
2. 向量级效应量小（中位差 8-10%），仅秩检验显著；不构成"常用词=均匀码"的强结论。
3. RARE 组 n=14（qwen），组内分布宽 [784,846]；E4b 的 A2/A3 n=20 补足。
4. glm4 只能作各向同性对照，判据组未成。
5. 静态行观察，不涉及层计算；与 N2h1 的"5 维类别子空间"是不同协议，禁互相印证。
6. 生僻组候选词过不了分词的（龘/靐/麤 等在 Qwen 为 2-token）被剔除，存活偏倚朝"较常用生僻字"方向。

### 5. 后续

- 频率-有效维关系的确认版（若需要）：语料频率实测分组 + held-out 词 + 3 模型，入队列为候选（不入当前 B 闸门主链）。
- 与 E1 的"实例词类谱塌缩"合并成一个"词典几何"小节候选：索引密度 = 频率的函数（近正交 → 域簇 → 趋随机）。

**资源**：零 GPU，3 探针合计约 22s（含 tokenizer 加载）；产物全部落盘 `tests/deepseek/Phase36/`（脚本+seal）与 `tests/deepseek_temp/Phase36/`（报告+sha8 登记）。
"""

with open(P, 'r', encoding='utf-8') as f:
    raw = f.read()
crlf = raw.count('\r\n')
style = '\r\n' if crlf > (raw.count('\n') - crlf) else '\n'
if not raw.endswith(style):
    raw += style
body = entry.replace('\n', style)
with open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(body)
after = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('before=%s after=%s appended=%d chars style=%s' % (before, after, len(body), repr(style)))
