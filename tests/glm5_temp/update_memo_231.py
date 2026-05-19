"""Update MEMO with Phase 231 results"""
import time

content = """

## Phase 231: Operator Mechanics - Feature是方向还是算子？ [2026-05-19 09:15]

### 核心问题
Phase 230否定: 形容词方向不存在跨名词稳定性(sep=1.0-1.2x)
Phase 231问题: 如果feature不是方向, 那它是算子吗?
  方向模型: h(red apple) = h(apple) + v_red
  算子模型: h(red apple) = h(apple) + W_red * h(apple)

### ExpA: 线性算子拟合 — 部分突破

方法: 30名词训练+10名词测试, 对每个形容词拟合Ridge回归(W_pca), PCA降维到20维
在held-out名词上测试R2

| 指标 | Qwen3(36L,d=2560) | GLM4(40L,d=4096) | DS7B(28L,d=3584) |
|------|-------------------|------------------|------------------|
| 最佳层dir_R2 | -1.70(L3) | -0.98(L4) | -0.21(L24) |
| 最佳层op_R2 | -1.63(L3) | -0.80(L4) | +0.56(L24) |
| op_advantage | 0.07 | 0.18 | 0.77 |
| 评价类adv | -0.06 | 0.14 | 0.88 |
| 颜色类adv | -0.13 | 0.13 | 0.86 |
| 状态类adv | 0.73 | 0.35 | 0.75 |

DS7B的算子R2=+0.56是正值! 方向R2=-0.21! 算子模型在DS7B上有效!
但Qwen3/GLM4的R2都极负, 算子模型不优于方向模型

层序模式(DS7B):
- L2: op_adv=-0.58(算子比方向差)
- L6: op_adv=+0.45(算子开始超越)
- L10-20: op_adv=0.51-0.71(算子持续优势)
- L24: op_adv=0.77(最大优势)

### ExpB: 操作因果注入 — 效果微弱

用beta=10-500在embedding层注入操作方向
KL散度一般<0.1, top-1很少改变, op相关token很少出现

### ExpC: 预测回路发现 — 最一致的发现

否定翻转(三模型一致):
| 模型 | flip_ratio | KL散度 |
|------|-----------|--------|
| Qwen3 | 0.10 | 3.15 |
| GLM4 | 0.07 | 2.10 |
| DS7B | 0.08 | 3.08 |

"not"将top-10概率压制到原来的7-10%! 这是10-14x的压制!
这是三模型最一致的发现, 也是目前最强的"预测修正器"信号

### ExpD: 算子非交换性

cos_context_dep(形容词"red"在"big"之后的delta vs 单独的delta的cosine):
| 深层cos_ctx | Qwen3 | GLM4 | DS7B |
|------------|-------|------|------|
| L25+ | 0.33-0.36 | 0.19-0.21 | 0.46-0.62 |

GLM4上下文依赖最强(cos_ctx=0.19), DS7B最弱(0.46-0.62)
context_dep_ratio 约 0.8-1.2: 上下文依赖和独立delta同量级

### 综合判决

1. 方向模型全面失败 (三模型R2都极负) -> Feature不是简单的方向向量
2. 线性算子模型仅在DS7B成功 (op_R2=+0.56, advantage=0.77)
   Qwen3/GLM4的R2都极负, 算子不优于方向
   -> 算子假设有条件成立
3. 否定翻转是最强预测修正器
   "not"将概率压制到7-10% (三模型一致)
   -> 支持"feature是概率流变换器"理论
4. 上下文依赖性确认
   形容词效果随层深越来越上下文依赖
   GLM4: cos_ctx=0.19(极强上下文依赖)
   -> 支持feature依赖于base流形的理论

### 硬伤分析

1. PCA+Ridge pipeline的有效性取决于信噪比, 不一定是真正的线性算子
2. 方向模型R2极负可能因为d_model太大(2560-4096) vs 样本量(30)
3. 否定翻转虽然强, 但只测了5个模板, 需要更多数据
4. ExpB因果注入失败, 不能确认操作方向的因果有效性

### 下一步方向

1. 否定回路深入: "not"是真正的feature primitive, 需要定位它的circuit
2. norm归一化后重测: 控制信噪比, 看算子优势是否普遍
3. 更多行为级feature: 时态约束、括号闭合、引用匹配
4. Activation Patching: 不用方向注入, 用真实activation替换
5. SAE训练: 如果能训练SAE, 可以找到更基础的feature primitive
"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'a', encoding='utf-8') as f:
    f.write(content)
print('MEMO updated at', time.strftime('%Y-%m-%d %H:%M:%S'))
