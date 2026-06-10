import sys
sys.stdout.reconfigure(encoding='utf-8')

memo_content = """

## Phase 448: 关系槽位主导的绑定态私有化 [2026-06-10 09:35]

### 实验1: SlotMediation拆分 — 模板先验 vs 对象知识 vs 冲突恢复

**三类模板:**
- no_obj: "A thing is a kind of ___" (纯模板先验)
- with_obj: "The apple is a kind of ___" (对象条件化)
- conflict: "Although the apple is described as an animal, it is a kind of ___" (冲突下对象知识)

**三模型对比 (apple):**
| 指标 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| avg_prior | +2.772 | **-1.683** | +2.879 |
| avg_obj_cond | +0.802 | +1.214 | +1.714 |
| avg_conflict_delta | +0.800 | **+2.468** | +0.440 |
| PriorScore | **0.776** | 0.581 | 0.627 |
| ObjCondScore | 0.224 | **0.419** | 0.373 |
| ConflictResilience | 0.997 | **2.033** | 0.257 |

**关键发现:**
1. GLM4的模板先验为负值(-1.68): GLM4主动抑制属性logit,对象知识才能解锁
2. GLM4的ConflictResilience=2.033(最高): 冲突模板下对象知识比无冲突更强
3. Qwen3的PriorScore最高(0.776): 模板先验占主导,对象知识贡献少
4. DS7B的ConflictResilience最低(0.257): 冲突下对象知识几乎不保留

**按槽位分析 (apple, is_a模板):**
| 模型 | prior | obj_cond | conflict_delta |
|------|-------|----------|---------------|
| Qwen3 | 2.15 | 2.48 | 2.58 |
| GLM4 | -1.15 | 2.07 | **4.22** |
| DS7B | 4.32 | -1.53 | -0.01 |

is_a是唯一所有模型obj_cond都强的模板。GLM4在is_a+conflict下增幅最大(4.22)。

### 实验2: 共享→私有化动力学 + MLP/Attn消融

**逐层shared_ratio趋势 (fruit):**
| 层 | Qwen3 | GLM4 | DS7B |
|----|-------|------|------|
| L0 | 0.898 | **1.000** | 0.959 |
| L12 | 0.681 | 0.855 | 0.646 |
| L24 | 0.488 | 0.856 | 0.359 |
| Last | 0.407 | **0.814** | 0.159 |

**消融结果 — MLP/Attn对shared_ratio的影响 (delta_shared after ablation):**
| 消融 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| L0 MLP消融 | **-0.192** | -0.084 | +0.059 |
| L0 Attn消融 | +0.063 | -0.018 | **-0.030** |
| L12 MLP消融 | -0.052 | -0.002 | +0.006 |
| L12 Attn消融 | +0.019 | +0.005 | +0.065 |

**极其重要的发现: MLP/Attn在私有化中的角色在不同模型中相反!**

- **Qwen3**: MLP维持共享(-0.192), Attn促进私有化(+0.063)
  → Qwen3的attention是把共享方向转化为对象私有方向的主要驱动力
- **GLM4**: MLP维持共享(-0.084), Attn影响很小(-0.018)
  → GLM4的私有化极慢,MLP和Attn都倾向于维持共享
- **DS7B**: MLP促进私有化(+0.059), Attn维持共享(-0.030)
  → DS7B的MLP是私有化的主要驱动力(与Qwen3相反!)

这解释了为什么GLM4深层shared_ratio仍然高:它的MLP和Attn都倾向于保持共享。
Qwen3的Attn促进私有化,所以shared_ratio下降快。
DS7B的MLP促进私有化,所以shared_ratio下降最快。

### 实验3: Alpha机制区间扫描

**自然/强制区间边界:**
| 对象 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| apple | 自然:0.1-1.5, 过渡:2.0 | 自然:0.75-1.5, 强制:2.0 | 自然:0.1-1.0, 强制:1.5+ |
| dog | 自然:0.1-1.5, 强制:2.0 | 过渡:0.1-0.5,0.75=自然 | 自然:0.1-1.0, 过渡:1.5+ |
| knife | 自然:0.1-2.0(全区间!) | 强制:0.25, 自然:1.0 | 自然:0.1-0.75, 过渡:1.0+ |

**GLM4对knife极不稳定**: alpha=0.25就进入forced区间(entropy从4.9跳到8.3)。
Qwen3最稳定: knife在alpha=2.0下仍为natural。
DS7B中等: apple在alpha=1.5就forced。

### Phase 448 核心发现总结

1. **SlotMediation的组成在不同模型中根本不同**:
   - Qwen3: 模板先验为主(77%),对象条件化为辅(23%)
   - GLM4: 对象条件化占比更高(42%),且先验为负(抑制性)
   - DS7B: 对象条件化占37%,但冲突恢复力弱

2. **私有化的驱动机制因模型而异**:
   - Qwen3: Attn是私有化驱动力
   - GLM4: 两种路径都保持共享(不私有化)
   - DS7B: MLP是私有化驱动力

3. **GLM4的负先验+高冲突恢复力**是一个全新的发现:
   GLM4默认抑制属性,对象知识"解锁"属性;
   冲突模板下对象被再次提及,解锁更强。

4. **alpha区间**: Qwen3最鲁棒,GLM4对某些对象极敏感,DS7B中等。
"""

output_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
with open(output_path, 'a', encoding='utf-8') as f:
    f.write(memo_content)

print(f"MEMO updated at {output_path}")
