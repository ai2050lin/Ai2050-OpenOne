import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

TEXT = """

---

## Phase 2866 (2026-09-18) — per-word 机制坐标 v0：机制谱词级类结构成立（W1=true），类结构弥散于词级坐标（2863 失败精确定位）

### 动机与协议（预注册冻结于任何观测前）
TMA 词族图谱最小实体：为 80 词（SEED=2855 词表确定性重建）构造双块坐标——
B1 unembed 投影块 proj[w,c] = E_w·dW_unit[c]（10 维，unit）；B2 因果谱块
drops_all[w].flatten()（1152 维，unit，2846 不可变产物）。零前向（10.6s，
模型仅加载读 W_U）。判据（200 次标签置换 null，SEED=2866）：
  W1 机制谱词级类结构：margin[w] = mean cos(w,同类) − mean cos(w,异类)，
      mean margin > null p95 → true；
  W2 unembed 侧同判据（正控制，2857 预测 true）；
  W3 块互补性：ρ(上三角 cos_B1, cos_B2)（描述）；
  W4 词特异性下界：每词对其他词的最大 cos（描述分布）；
  W5 留一最近邻同类检索 accuracy > null p95。

### 结果（phase2866/word_coords/；exec 14adb2f6 / result f89485e6 / npz fff91b4f）
| 判决 | 观测 | null p95 | 结果 |
|---|---|---|---|
| **W1** 机制谱词级类结构 | margin 均值 **0.0976** | 0.0170（5.7×） | **true** |
| **W2** unembed 类结构（正控） | margin 均值 **1.067** | 0.0325（32.9×） | **true** |
| **W3** 块互补性 | ρ = **0.114** | — | 两块携带不同信息 |
| **W4** 词特异性下界 | 最近邻 max cos min **0.144** / p50 0.391 | — | **无两词共享机制**（NN 距离下界 0.86） |
| **W5** 同类检索 | accuracy **0.325**（chance 0.111） | 0.1375 | **true**（2.4× null） |

### 科学结论
1. **2863 失败精确定位**：类均值谱无类信号（J1=false），但 **per-word margin 有**
   （W1=true，5.7× null）——类结构存在于机制谱中，但**弥散在词级坐标里，
   无法压缩为类级原型**。"类"是弱先验/检索视角，不是机制组织轴。
2. **unembed 侧 >> 机制侧**（margin 1.067 vs 0.098，11 倍）：类信息在 unembed
   几何中远比在因果谱中浓缩——双图谱的两侧不对称性首次定量。
3. **W3=0.114**：词的 unembed 投影坐标与因果谱坐标几乎独立——词族图谱与
   响应图谱的"关联机制"问题在词级就有实体（两坐标系之间的变换是研究对象，
   2865 已给出第一个机制级样例：mlp 间接写出）。
4. **W4 直接定量支撑 token 特殊性原则**（用户假设）：80 词中最近邻机制距离
   下界 0.86（min max-cos 0.144）——不存在机制相同的词对；同时 p50 0.391
   说明词间有实质共享成分——**共享基 + 坐标差异**的组合编码图像。

### 阶段性目标达成小结（用户指令：连续推进至此）
- 响应图谱：四轴确立（因果 2846 / mlp 响应 2861 / OV 静态 2862 / η² 类方差 2864），
  互不预测（ρ −0.026 / −0.079 / 0.15）。
- 词族图谱：词坐标 v0（B1+B2）落地，类结构词级成立、类均值不成立。
- 关联机制：第一个完整样例 = 2865（L13H30 因果 → mlp 间接写出，e_ff ~30× 静态）。
- 类组织三连判决：2863 类均值 false / 2864 类方差弱且正交 / 2866 词级 margin true。

### 接续（2867 候选）
- 主选：词坐标 v1——加入第三块（2861 mlp 响应谱 per-word / η² 加权谱），
  检验 W3 互补性是否推广到三块；检索准确率随块数的增量（机制基增长率曲线
  的第一个点：每加一个坐标块，词间分辨/类检索提升多少）。
- 备选：属性轴预研（反义词对协议，接 400b）。
- 备选：199 词普查重启（G2 门线需用户决策）。
（脚本 tests/glm5/phase2866_word_coords.py；产物 phase2866/word_coords/）
"""

with io.open(MEMO, 'r', encoding='utf-8') as f:
    before = f.read().count('\n')
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(TEXT)
with io.open(MEMO, 'r', encoding='utf-8') as f:
    after = f.read().count('\n')
print('LINES %d -> %d' % (before, after))
