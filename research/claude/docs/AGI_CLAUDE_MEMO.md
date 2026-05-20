
## Phase 234: 语言编码机制破解 — 差分编码 + 控制信号几何 + k90@not-token [2026-05-20 21:30]

### 实验设计
测试三个核心假说：
- Part A: 差分编码（d_apple = e_apple - mean(e_fruits) 是否捕捉苹果独特属性）
- Part B: k90@"not"token 位置（复现 Phase 236 的 k90=1 声称）
- Part C: 控制信号几何（极性/时态/语态/疑问变换方向是否正交）

模型: Qwen3 → GLM4 → DS7B (BF16 + device_map=auto)
Run1: 3类×5词, 15对, 8对/类 | Run2: 5类×10词, 80对, 40对/类

### Part A 结果：差分编码（否定结论）
三模型一致结果:
- k90_d = 8/10（差分向量占 10 维的 8 维主成分）
- mean_pairwise_cos_d ≈ -0.11（差分向量两两接近正交，略有负相关）
- d_w 的 top-20 token 仍以该词本身及变体为主，不显示明显语义分化

**结论：embedding 层的差分编码（d_w = e_w - mean_cat）未产生语义清晰的独特特征投影。**
差分向量的方向高度分散（k90=8/10），说明每个概念偏差向量几乎独立，没有共享轴，但也没有语义对齐的独特信号。可能原因：embedding 空间本身是高维各向同性的（每个词独占一个方向），所以减去均值仍然是高维噪声，而非有语义意义的偏差。

### Part B 结果：k90@not-token（Phase 236 未复现）
Run2 (n=80):
- Qwen3 (L36): k90=40-53 for all layers (not-token pos)
- GLM4 (L40):  k90=54-63 for all layers (not-token pos)
- DS7B (L28):  k90=56-61 for L0-L26, **L27: A_k90=18, B_k90=14**

**结论：Phase 236 声称的 DS7B 中层 k90=1 未复现。**
"not" token 位置的 delta（h_neg[not_pos] - h_base[adj_pos]）在所有中间层都是 k90=56-61（高维），只有最后一层（L27）出现压缩（k90=14-18）。这表明 Phase 236 的 k90=1 可能来自不同测量方式（绝对 hidden state 而非 delta，或特定 attention head 的 value）。

### Part C 结果：控制信号几何（最强发现）
Run2 (n=40 per type) 各模型 polarity_vs_question cosine:
- Qwen3: 0.245–0.365 (全层一致)
- GLM4:  0.265–0.395 (全层，L30峰值=0.395)
- DS7B:  0.224–0.380 (全层，L20峰值=0.380)

其他对（polarity_vs_tense, polarity_vs_voice, tense_vs_voice, tense_vs_question, voice_vs_question）全部接近零（-0.08 到 +0.09）。

**核心发现：极性变换（"X is Y" → "X is not Y"）和疑问变换（"X is Y" → "Is X Y?"）在隐藏状态空间中不正交，三模型一致（cosine≈0.25–0.40），所有层一致。**

其余三对变换（时态/语态/疑问 vs 极性/时态/语态）高度正交，说明存在独立的控制信号方向。

**语言学解释：** 极性和疑问共享一个几何轴，因为两者都在对"X is Y"的命题真值进行"操作"（polarity 否定真值，question 悬置真值）。时态和语态是正交的，因为它们操作的是谓词的时间维度和施受关系，与真值操作无关。

DS7B 最后一层（L27）出现控制信号维度崩塌：
- polarity: k90=6, top1=0.771
- tense:    k90=15, top1=0.613  
- question: k90=11, top1=0.647
- voice:    k90=22, top1=0.449

### 命令与脚本
```bash
# 主脚本
python tests/claude/234_encoding_mechanism_2026-05-20.py --run run1 --model qwen3
python tests/claude/234_encoding_mechanism_2026-05-20.py --run run2 --model all
```
结果文件: tests/claude_temp/234_run{1,2}_{qwen3,glm4,ds7b}.json

### 研究进展总结
Phase 232-234 连贯结论：
1. **词法≠句法否定**（Phase 232）：embedding 中否定方向≠动态 hidden state 否定变化
2. **否定不是低维压缩**（Phase 233-234）：k90@not-token 在中层=56-63，不是 k90=1
3. **极性与疑问共享真值操作轴**（Phase 234，最新）：三模型一致，cosine≈0.25-0.40
4. **其他控制信号正交**（Phase 234）：时态/语态/疑问 vs 极性/时态/语态 均近零
5. **DS7B 最后层特有压缩**（Phase 233-234）：k90 急剧下降到 14-22

