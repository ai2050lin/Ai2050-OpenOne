# Phase 269 & 270: Scientific Falsification & Baseline Verification

## 任务分解

- [x] **1. 高维诅咒基线测试 (Orthogonality Falsification)**
  - [x] 编写并运行 `tests/claude/269_high_dim_orthogonality_baseline.py`。
  - [x] 随机采样十万组 3584 维向量，计算其余弦相似度。
  - [x] 若观察值 0.0162 落在随机噪声标准差（$\approx 0.0167$）内，则彻底推翻正交叠加假说。
- [x] **2. 随机网络拓扑测试 (Omega Falsification)**
  - [x] 编写并运行 `tests/claude/270_untrained_omega_baseline.py`。
  - [x] 实例化无预训练权重的乱码网络，执行 SVD 并计算 $\Omega$ 压缩比。
  - [x] 验证低秩坍缩到底是“推理逻辑”的产物，还是架构（如 LayerNorm/Residual）的天然属性。
- [x] **3. 文献档案自洽手术**
  - [x] 修改 `coding_mechanism_puzzle_ledger.md`：记录两组证伪实验的发现。
  - [x] 修改 `neuron_attribute_mapping.md`：抹除具体的单点坐标（如 Neuron #2596），替换为子空间概念，修正各层深度百分比一致性假说。
  - [x] 修改 `llm_first_principles_mathematics.md`：整合并修正六大假说，删除伪装的现象学描述，直面高维高斯噪声和架构平凡属性。
  - [x] 将更新同步至 `docs` 目录。
- [x] **4. 制作证伪汇报**
  - [x] 产出 `walkthrough.md`。
