# 神经元两极图谱：从采集到填图的完整流程 v1

**状态**：方法论文档，零 GPU 产出；代码段为结构正确但**未执行**（未实测），首次落地跑通后需补 verdict 与位级锚。
**适用**：qwen3-4b / Qwen3-14B / glm4-9b（三者均为 SwiGLU + SiLU，实测 config 确认）。
**来源**：Goodfire《Neural Networks Are Already Clean》(Cammarata) 的操作方法，落地为本项目口径。
**不占用 Phase 编号**；若转为正式实验，按五段式开 Phase 并写 MEMO。

---

## 0 全景：五个阶段

| 阶段 | 一句话 | 输入 | 输出 | 通过判据 |
|---|---|---|---|---|
| A 采集 | 宽语料跑一遍，存门控 pre-activation 与残差流 | 语料 | `G[N,inter]`、`H[N,hidden]` | 有效文本数 ≥ 200 |
| B 选通道 | 按集中度排序，先读最容易读的 | G、H | 候选通道清单 | 邻域一致性 > 0.5 优先 |
| C 读两极 | top-k / bottom-k，同时摊开 | 某通道 z | P、N、Δ | 两极各自内部语义一致 |
| D 解释 | 四条互不相关的读出路径收敛 | Δ | 一句描述 + 强度 | ≥3 路指向同一处 |
| E 归族 | 与他通道/他层的 Δ 比角度 | 全部 Δ | 族与地图 | cos 显著高于随机 |

主线一句话：**z 分两极 → Δ = 两极均值差 → Δ 的方向就是那条轴 → 用四路检验它是不是真轴。**

---

## 1 阶段 A：采集

### 1.1 三个量，别混

qwen3 系列的 MLP 是 SwiGLU，前向为：

```
g  = W_gate · x + b_gate        # 门控 pre-activation，有符号
u  = W_up   · x + b_up          # 值通路，有符号
a  = silu(g) * u                # 真正进 down_proj 的量
x  = post_attention_layernorm(residual)
```

同时需要**残差流**本身（mlp 之前、norm 之前）：

```
residual  = 该层 attention 输出加回后的值
```

三者关系：`residual` → norm → `x` → gate/up → `a`。

### 1.2 极性读在 g 上，不读在 a 上（最关键的架构差异）

ReLU 网络里读 `z`（pre-activation）。SwiGLU 里对应物是 `g`，理由更强：

- `silu(g)` 在 g<0 时仍为**小负数**，不会归零 → 负向信息没被完全抹掉，这是好消息；
- 但 `a = silu(g) * u` 是**乘积**，符号是 g 与 u 的联合符号。g<0 且 u<0 时 a>0，**两个负号抵消**。这就是文章提到的"双重否定"。

所以：
- 想读"这个通道在区分什么" → **读 g**；
- 想读"它对下游贡献了什么" → 读 a；
- 用 a 判极性会把"门关着但值是负的"误判成"正极激活"。

`silu` 最小值约 **−0.2785**（在 g≈−1.278 处），所以 g 极负时 a 的贡献是有界的、不会爆炸。

### 1.3 采集点（两个 hook，不要搞错）

| 要的东西 | hook 位置 | 取什么 |
|---|---|---|
| 残差流 `H` | `layers[L].post_attention_layernorm` 的 **pre**-hook | `args[0]` |
| 门控 `G` | `layers[L].mlp.gate_proj` 的 **post**-hook | `output` |

**常见错误**：在 `mlp` 的 pre-hook 取输入当残差流。那是 norm **之后**的值，每个位置缩放不同，用它算 Δ 会引入位置相关的伪方向，且无法用于 steering。

### 1.4 数据规模与去重

- 位置数 `N` 要 ≥ 20 万（1000 段 × 200 token 量级）；
- 但**有效样本数按文本数算**：同段文本相邻 token 的 h 高度自相关，若 k=100 只来自 5 段，标准差被低估一个量级 → 假显著；
- 规则：**k 个极值样本至少覆盖 k/5 段独立文本**，不足就扩语料或减 k。

---

## 2 阶段 B：选通道

文章没给这一步（他是闲逛遇到的）。工程上必须补，用三个不需要语义标注的代理指标。

### 2.1 三个指标

**① 激活频率** `freq_j = mean(g_j > 0)`
> 0.5 以上通常是通用通道（几乎对所有输入开门），信息量低；< 1e−4 则样本不足。

**② 分离度** `sep_j = mean(top-k g) / std(all g)`
> 越大说明极值离主体越远，越像"专门检测器"。

**③ 邻域一致性**（最实用）
```
coh_j = mean over pairs (i,i') in top-k of cos(H_i, H_i')
```
> 极值位置的残差流彼此越像，说明它抓的是同一件事。**无需语义标注就能算**，且直接对应"集中度"。

### 2.2 排序与判据

综合分 `score_j = zscore(sep_j) + zscore(coh_j)`，降序读。

| coh | 解释 | 动作 |
|---|---|---|
| > 0.6 | 高度集中 | 优先读，最可能一次读懂 |
| 0.3–0.6 | 部分集中 | 可读，注意是否两个子群 |
| < 0.3 | 分散 | 先怀疑语料覆盖不足，再怀疑多义 |

**注意**：coh 低不直接等于多义。文章两个失败案例（长上下文、健康 SEO）都是"语料不够 → 看起来多义"。排序前先确认语料覆盖。

---

## 3 阶段 C：读两极

### 3.1 计算

```
P = top-k { g_j[p] }        # 正极：样本地址集合 {(文本 id, 位置)}
N = bottom-k { g_j[p] }     # 负极
H_P = H[P]                  # [k, hidden]
H_N = H[N]                  # [k, hidden]
Δ   = mean(H_P) - mean(H_N) # [hidden]
```

三层要分清：P 是**地址集合**，`H[P]` 是**矩阵**，`mean(H[P])` 才变成**向量**。

### 3.2 k 的选择

| k | 后果 |
|---|---|
| < 20 | 采样噪声主导 Δ |
| 50–200 | 推荐区间 |
| > 1% 的 N | 混入中间态，Δ 被稀释 |

### 3.3 输出形态（唯一有意义的验收入口）

必须打出**真实文本片段**，不是向量摘要：

```
[P] rank  z      位置   片段
     1   +4.82   tok 6  ...他把苹果削了皮，切成小块...
     2   +4.51   tok 5  ...果园里的苹果熟了，压弯枝头...
[N]  1   -5.13   tok 4  ...分析师上调苹果目标价...
```

看到这张表，"正极/负极"才从符号变成可检验的东西。

### 3.4 为什么 Δ 能工作（整套方法的数学根据）

设该层残差流 `h ~ N(μ, Σ)`，通道的 `z = wᵀh + b`。`(h,z)` 联合高斯时条件期望是线性的：

```
E[h | z] = μ + Σw · (z - wᵀμ) / (wᵀΣw)
```

于是两极均值差：

```
Δ = E[h | z>t] - E[h | z<-t] = c · Σw ,   c > 0
```

即 **Δ 与真实权重方向只差一个协方差矩阵**：Δ ∝ Σw。

三条推论，每条都改变做法：

1. **Δ ≠ w 但同向**——这就是玩具例里 cos=0.972、而分量绝对值不可比的来源。Δ 是被 Σ 拉歪的轴。
2. **白化可复原**：`Σ⁻¹Δ ∝ w`。所以"轴纯度"的严格定义不是 `cos(Δ,d)`，而是 `cos(Σ⁻¹Δ, d)`。
3. **Σ⁻¹Δ 正是 Fisher/LDA 判别方向**。整套流程在数学上等价于：估计两个条件分布的均值差，再按背景协方差白化。

这条同时给 AGENTS.md §5.2「统一背景白化」提供了根据：不白化也能拿到方向，但跨族比较时各族的 Σ 不同，cos 会被 Σ 污染而不是被语义污染。

---

## 4 阶段 D：解释（四路独立检验）

### 4.1 路径一：logit lens

把 Δ 投影到输出侧词表：

```
S = Δ · U^T                 # U = lm_head.weight, [vocab, hidden]
top/bottom 各取 30 个 token
```

**跨模型注意（实测）**：

| 模型 | tie_word_embeddings | 该乘什么 |
|---|---|---|
| qwen3-4b | **True** | embed 与 lm_head 同一矩阵，等价 |
| Qwen3-14B | False | 必须用 `lm_head.weight` |
| glm4-9b | False | 必须用 `lm_head.weight` |

用错矩阵会得到一个语义不通的 token 列表，且不会报错——静默错误。

### 4.2 路径二：steering

```
h' = residual + α · Δ̂          # Δ̂ = Δ/‖Δ‖
```
α 从小到大扫（0.5 / 1 / 2 / 4 × ‖residual‖ 的典型分量），看生成内容变化。
**判据**：调正后出现正极语义，调负后出现负极语义，才算这条轴成立。

### 4.3 路径三：对偶检验（否证）

> P 共享性质 A，N 共享性质 B，且 A、B 是**同一语义轴的两端** → 单义门控。
> N 内部无共同性质 → A 不是真轴，仍多义。

这是唯一能杀掉假设的一路。正极只能提假设。

### 4.4 路径四：定向测例（针对语言侧）

设计只在轴上变化的最小对，例如：
- 苹果（水果义）vs 苹果（公司义）
- is-a（上位）vs part-of（组成）
- 陈述 vs 疑问

**四路必须互不相关**：logit lens 看词表、steering 看生成、对偶看语义、测例看行为。任意一路分叉就回阶段 C 重猜，不是"大部分对就算过"。

---

## 5 阶段 E：归族与填图

### 5.1 族判据

两条 Δ 是否同一族：`cos(Δ_a, Δ_b)`

| cos | 含义 |
|---|---|
| > 0.7 | 同轴，大概率同族 |
| 0.3–0.7 | 相关但不完全同 |
| < 0.3 | 不同族（或同族但被层间变换旋转） |

**跨层比较的陷阱**：残差流各层共享同一空间，但同一功能在相邻层可能是同一方向、隔多层可能被旋转。所以跨层 cos 必须**相邻层**才可信，跨远层要先用同层内的"桥接通道"对齐。

### 5.2 登记表（一格一卡）

| 字段 | 内容 |
|---|---|
| id | `L{层}.N{通道}` + 模型快照 |
| 极轴 | 一句动词性描述（它在区分什么） |
| P 摘要 | top-5 片段 + k + coh |
| N 摘要 | 同上 |
| Δ sha8 | 方向指纹，便于复现 |
| logit lens | 两端各 10 token |
| steering | α 区间 + 行为描述 |
| 对偶 | 通过/失败/未测 |
| 族 | 同族 id 列表 + cos |
| 证据级 | E0 观察 / E1 留出复现 / E2 跨模型 / E3 干预 |

### 5.3 复利从哪来

填图的加速不是"读得多"，是**判据变准**：
- 有了族，新通道的候选解释被压缩到同族范围；
- 有了失败账本，coh 低时能立刻区分"语料不足"还是"真多义"；
- 剩余未解释部分若集中在某处 → 指向工具本身有问题（文章发现 SwiGLU 双重否定即此路径）。

---

## 6 完整脚本（结构正确，未执行）

```python
# tests/gpt5/<phase>_neuron_pole_probe.py
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"D:/AI2050/Ai2050-OpenOne/models/hf/qwen3-4b"
LAYER, K, MAX_LEN = 18, 100, 128
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map=DEV).eval()

cap = {}
layer = model.model.layers[LAYER]

def hook_resid(mod, args):
    cap["h"] = args[0].detach()          # 残差流（norm 之前）

def hook_gate(mod, inp, out):
    cap["g"] = out.detach()              # 门控 pre-activation

layer.post_attention_layernorm.register_forward_pre_hook(hook_resid)
layer.mlp.gate_proj.register_forward_hook(hook_gate)

G, H, meta = [], [], []
with torch.no_grad():
    for i, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=MAX_LEN).to(DEV)
        model(**enc)
        g = cap["g"][0].float().cpu()    # [T, inter]
        h = cap["h"][0].float().cpu()    # [T, hidden]
        G.append(g); H.append(h)
        meta += [{"doc": i, "pos": p} for p in range(g.shape[0])]
        cap.clear()

G = torch.cat(G)                         # [N, inter]
H = torch.cat(H)                         # [N, hidden]
N = G.shape[0]

def read_poles(j, k=K):
    z = G[:, j]
    idxP = z.topk(k).indices
    idxN = z.topk(k, largest=False).indices
    dP, dN = H[idxP], H[idxN]
    delta = dP.mean(0) - dN.mean(0)
    coh = torch.nn.functional.cosine_similarity(
        dP, dP.mean(0, keepdim=True)).mean().item()
    return {"j": j, "zP": z[idxP].tolist(), "zN": z[idxN].tolist(),
            "delta": delta, "coh": coh,
            "docsP": sorted({meta[i]["doc"] for i in idxP.tolist()}),
            "docsN": sorted({meta[i]["doc"] for i in idxN.tolist()})}

def logit_lens(delta, topn=30):
    U = model.lm_head.weight.detach().float().cpu()   # [vocab, hidden]
    s = delta @ U.T
    hi = s.topk(topn).indices.tolist()
    lo = s.topk(topn, largest=False).indices.tolist()
    return ([tok.decode([t]) for t in hi],
            [tok.decode([t]) for t in lo])
```

调用：

```python
rec = read_poles(j)
print(len(rec["docsP"]), len(rec["docsN"]))   # 去重检查：应 >= k/5
print(rec["coh"])
print(logit_lens(rec["delta"]))
```

**未实现但必须补的**：steering（需在 forward 中改残差流）、对偶检验（人工或 LLM 判定）、跨层族比较。

---

## 7 判据汇总：什么算"读懂了"

| 检查 | 阈值 | 失败含义 |
|---|---|---|
| 有效文本数 | ≥ k/5 | 样本不独立，Δ 方差被低估 |
| coh | > 0.3 | 语料覆盖不足，或真多义 |
| 两极内部语义一致 | 人工判定 | 未找到轴 |
| 两极构成对偶 | 人工判定 | 是混合物 |
| logit lens 两端通顺 | 人工判定 | Δ 落在无语义子空间 |
| steering 双向对应 | 行为可辨 | 几何存在但行为不控（同 F20） |
| 留出文本复现 | 换语料仍成立 | 过拟合到语料 |

七条里至少五条过，才能登记为 E1。

---

## 8 十个坑

1. **读 a 不读 g** → SwiGLU 双重否定，极性全错。
2. **在 mlp pre-hook 取残差流** → 拿到 norm 后的值，Δ 带位置伪影。
3. **k 按位置数算有效样本** → 一段文本贡献 100 个"样本"，方差低估。
4. **语料太窄** → 伪造多义（文章两个失败案例同因）。
5. **用输入 embed 做 logit lens** → 14B/GLM4 不 tie，静默错；qwen3-4b 恰好 tie 所以看不出来。
6. **跨远层比 cos** → 同功能被旋转，误判不同族。
7. **只读正极** → 必然把"轴的两端"当成"多个概念"。
8. **用梯度优化找极值图** → reward hacking，得到的是最容易生成的样本而不是最该响应的样本。
9. **coh 低就判多义** → 九成是覆盖不足。
10. **steering 有效就宣布机制成立** → 几何可控 ≠ 计算上是必要环节（F20 已证）。

---

## 9 这套流程的边界（必须写在结论前面）

- **它是描述性方法**：给出"这个通道在区分什么"，不给出"它在计算中是否必要"。必要性需要消融或归因，属另一条线。
- **单通道视角看不到组合**：一个语言功能的实现通常跨多个通道与层，单格读懂不等于该功能读懂。
- **阈值全是经验值**：coh>0.3、cos>0.7 等来自视觉模型经验，语言侧需本项目实测重定，不能直接引用。
- **原文证据以视觉为主**：语言侧的量化的姊妹论文未发布，本文档的判据是从方法论推的，不是从原文语言实验抄的。
- **不替代 dist_protocol**：需要推广性主张时，仍须走模板族 + 行为门 + lockbox，本文档只覆盖"单通道读法"。

---

## 10 与现有线的接口

- 与 **3246**（FLIP=INFORMATION_GAP、XC_p5_b8 现役）：本流程的 Δ 可直接作为"轴纯度"指标，检验 XC_p5_b8 是否满足 Δ_d ∥ d 的不动点条件。
- 与 **3255 双轨道几何**：本流程用于验证"词表侧不可补"是否在单通道层面也成立（若单通道两极都落在词表外，则双轨道判断得到独立支持）。
- 与 **F20**：steering 一栏必须与"读出几何"分开记，避免重犯"几何有行为无"。
