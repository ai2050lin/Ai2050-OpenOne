import sys, os
sys.stdout.reconfigure(encoding='utf-8')

memo_text = """
## Phase 98: 语言切换机制与词表归一化分析 [2026-05-08 22:54]

### 批判评估

| 批判点 | 评估 | Phase 98验证 |
|--------|------|------------|
| "振荡可能是tokenizer artifact" | ✅**部分正确** | Qwen3: 簇聚合后振荡只减16%(语义级); GLM4: 减少60%(部分artifact) |
| "Cosine≈-1+Pearson≈0有数学问题" | ✅**部分正确** | 不是稀疏+重尾(峰度≈3)，是"全局反平行+局部无关"的真实结构 |
| "只有Qwen3能翻译是prompt mismatch" | ✅**完全正确** | GLM4 Top-5=100%(en_prefix), DS7B Top-5=73%(en_prefix) |
| "翻译不是序列流程证据不够" | ✅**正确** | 簇聚合后振荡仍存在，但原语(对齐/压制/解码)仍可能是底层机制 |
| "局部计算原语>统一方程" | ✅**极可能正确** | Path Patching显示L26 Attn是唯一注意力关键层，L31 MLP最关键 |

---

### 实验1: 词表语义簇分析 — 区分"分词动力学"和"语义动力学"

**Qwen3 (20对)**:
- 单token振荡: 3.05次 → 语义簇振荡: 2.55次 (仅减少16%)
- **→ 振荡主要是语义级别的，不是tokenizer artifact**
- 语义切换层: L28.2 (78.4%深度) — 和Phase 96/97的patching一致
- 英文簇覆盖率: 5.97x — 变体token分走了大量概率
- 逐层语义分裂: L28 en=0.256, L30 en=0.640, L32 en=0.828

**GLM4 (20对)** — 完全不同的模式:
- 单token振荡: 2.35次 → 语义簇振荡: 0.95次 (**减少60%**)
- **→ GLM4的振荡大部分是tokenizer artifact**
- 语义切换层: L36.5 (91.2%深度) — 远深于Qwen3
- 逐层语义分裂: 全程other≈100%，en/zh概率<0.005
- **→ GLM4的翻译信息在logits空间几乎不可见**

**跨模型核心差异**:
| 特征 | Qwen3 | GLM4 |
|------|-------|------|
| 振荡是语义级的? | 是(仅16%减少) | 部分(60%是artifact) |
| 切换深度 | 78% | 91% |
| 翻译信息在logits可见? | 是 | 几乎不可见 |

---

### 实验2: 贡献向量分布分析 — 解决Cosine≈-1+Pearson≈0谜题

**关键发现: 不是稀疏+重尾，是"全局反平行+局部无关"**

- 峰度: 翻译=2.89(≈正态), 补全=6.08(轻重尾) — 不是极端重尾
- 极端值比例: Top-10只占2.5% — 不是被少数值主导
- 去除极端值后: Cosine仍≈-0.998, Pearson仍≈-0.02 → 两者都是真实的

**数学解释**:
- Cosine≈-1: 所有head的翻译贡献都是负的(-0.061)，补全贡献都是正的(+0.034) → 消融任何head都让翻译变好
- Pearson≈0: 具体哪个head贡献多/少，在两个任务间无相关性 → 不是同一批head

**结论**: 消融方法本身有问题——**所有head对翻译的"贡献"都是负的**，说明消融改变了整体网络状态（不是简单地去掉功能），而翻译功能比补全更容易被网络重配改善。

---

### 实验3: Path Patching — 从residual级升级到attention path级

**Qwen3关键发现: 翻译信息主要通过MLP路径传递**

| 层 | Attn Leak | MLP Leak | 主导路径 |
|---|---|---|---|
| L26 | **0.00730** | 0.00011 | **Attn** ← 唯一attn关键层 |
| L31 | 0.00003 | **0.01965** | **MLP** ← 最关键层 |
| L34 | 0.00024 | 0.00209 | MLP |
| L30 | -0.00001 | 0.00175 | MLP |

- **MLP主导19层 vs Attn主导10层**
- **L26的attention可能是"语言切换路由"** — 把中文语义空间的信息导向英文空间
- **L31的MLP是"翻译信息放大器"** — 将portable信息放大为可解码输出

**结合Phase 96/97的发现**:
- Phase 96: L28是portable information首次形成的层
- Phase 98: L26的attn是关键 → L26-28之间attn在执行"语言切换"
- L31的MLP是翻译信息的"输出放大"

---

### 实验4: 模型专属prompt验证

**完全推翻Phase 97的"只有Qwen3能翻译"结论！**

| 模型 | 最佳prompt | Top-5准确率 | Top-1准确率 | en_cluster概率 |
|------|-----------|------------|------------|--------------|
| Qwen3 | X的英文是 | ~95% | ~90% | ~0.80 |
| GLM4 | Translate X to English: | **100%** | **93.33%** | **0.8554** |
| DS7B | Translate X to English: | 73.33% | 46.67% | 0.2585 |

**关键修正**:
- GLM4翻译能力**不弱于Qwen3** — 只是prompt格式不同
- DS7B翻译能力**确实弱** — 最佳格式下Top-5只有73%
- Phase 96/97的所有跨模型对比都有系统性偏差（prompt mismatch）

---

### 被证伪的

1. 只有Qwen3能翻译 → 错！GLM4用en_prefix格式100%Top-5
2. "振荡是语义切换"普适 → 错！GLM4的振荡60%是tokenizer artifact
3. Cosine≈-1+Pearson≈0是因为稀疏重尾 → 错！是真实的"全局反平行+局部无关"

### 被确认的

1. **Qwen3的振荡是语义级切换** — 簇聚合后仅减少16%
2. **不同模型的振荡机制根本不同** — Qwen3语义级 vs GLM4 tokenizer级
3. **翻译信息主要通过MLP传递** — L31 MLP最关键，L26 Attn是唯一注意力关键层
4. **消融方法有根本问题** — 所有head的翻译贡献都是负的
5. **GLM4翻译信息在logits空间不可见** — 需要非logits的分析方法

---

### Phase 98最重要的发现

# 1. "语言切换"是语义级现象(Qwen3)还是tokenizer artifact(GLM4)? — 两者都有!
# 2. L26的attention是"语言切换路由" — 这是目前找到的最接近"计算原语"的结构
# 3. Prompt mismatch导致了之前所有跨模型对比的系统性偏差

---

### 硬伤与瓶颈

1. **GLM4的Exp 1仍用"X的英文是"** — 应换成"Translate X to English:"，但翻译信息在logits层不可见的问题可能更深层
2. **Path Patching只做了"加法"** — 是添加source的attn/MLP输出，不是替换；结果可能和residual patching不同
3. **DS7B的翻译能力确实弱(73%)** — 但不是0，所以可以做跨模型对比
4. **"全局反平行+局部无关"的解释可能不完整** — 消融方法的根本问题需要新方法论
5. **未做因果中介分析** — path patching≠causal mediation

---

### 下一步关键方向

1. **为每个模型使用最佳prompt重新做全部分析** — 修正prompt mismatch偏差
2. **L26 Attn深入分析** — 找到具体哪些head在执行"语言切换路由"
3. **因果中介分析** — 不只是patch，而是真正测量causal mediation effect
4. **非logits层的翻译信息追踪** — GLM4的翻译信息在hidden space中存在但不在logits中出现
5. **最小充分电路** — 找到翻译任务的最小子网络
"""

outpath = "research/glm5/docs/AGI_GLM5_MEMO.md"
with open(outpath, "a", encoding="utf-8") as f:
    f.write(memo_text)

print(f"Memo已追加到 {outpath}")
print(f"追加长度: {len(memo_text)} 字符")
