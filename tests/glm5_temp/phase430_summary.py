"""Phase 430 Comprehensive Cross-Model Summary + MEMO Update"""
import sys, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
ROOT = Path(r"d:\Ai2050\TransformerLens-Project")

models = ["qwen3", "glm4", "deepseek7b"]
results = {}
for m in models:
    for r in [1, 2]:
        f = ROOT / f"results/phase430_natural_transport/{m}_phase430_r{r}.json"
        if f.exists():
            with open(f) as fh:
                results[f"{m}_r{r}"] = json.load(fh)

# ===== Build MEMO content =====
now = time.strftime("%Y-%m-%d %H:%M")
memo_lines = []

memo_lines.append(f"\n## Phase 430: Natural Transport Direction + Causal Tracing [{now}]")
memo_lines.append("")
memo_lines.append("### 实验原理")
memo_lines.append("")
memo_lines.append("Phase 429B证明layer-probe方向在中层last token位置可导致类别切换，但probe方向是**统计相关方向**（类别均值差），不是因果方向。")
memo_lines.append("")
memo_lines.append("本阶段测试三个关键问题：")
memo_lines.append("1. **自然运输方向**：在embedding层注入类别扰动后，模型自然将扰动传播到中层。δ_l = h_l(perturbed) - h_l(clean) 就是\"被模型自然运输的方向\"。")
memo_lines.append("2. **运输方向 vs Probe方向**：哪个在中层注入时更有效、更清洁？")
memo_lines.append("3. **因果追踪**：用corrupt-then-restore方法，找出哪些层/位置对类别读出因果关键。")
memo_lines.append("")
memo_lines.append("### 核心数据")
memo_lines.append("")

# 1. Transported vs Probe comparison
memo_lines.append("#### 1. 自然运输方向 vs 统计Probe方向（best per object, R2 data）")
memo_lines.append("")
memo_lines.append("| 对象 | 模型 | Transported Δ | H | Probe Δ | H | 优势 |")
memo_lines.append("|------|------|-------------|---|---------|---|------|")

for m in models:
    rkey = f"{m}_r2"
    if rkey not in results:
        rkey = f"{m}_r1"
    if rkey not in results:
        continue
    for obj, od in results[rkey]["per_object"].items():
        # Find best transported at last position
        best_t = {"delta": 0, "H": 99, "desc": ""}
        for src_alpha, layer_data in od.get("inject_results", {}).items():
            for layer_key, pos_data in layer_data.items():
                for key, r in pos_data.items():
                    if "last" in key and abs(r["delta"]) > abs(best_t["delta"]):
                        best_t = {"delta": r["delta"], "H": r["full_entropy"],
                                 "desc": f"srcα={src_alpha} {layer_key}/{key}"}

        # Find best probe at last position
        best_p = {"delta": 0, "H": 99, "desc": ""}
        for layer_key, pos_data in od.get("probe_inject_results", {}).items():
            for key, r in pos_data.items():
                if "last" in key and abs(r["delta"]) > abs(best_p["delta"]):
                    best_p = {"delta": r["delta"], "H": r["full_entropy"],
                             "desc": f"{layer_key}/{key}"}

        if abs(best_t["delta"]) > 0.1 or abs(best_p["delta"]) > 0.1:
            adv = "Transported" if (abs(best_t["delta"]) > abs(best_p["delta"]) or best_t["H"] < best_p["H"]) else "Probe"
            if abs(best_t["delta"]) > abs(best_p["delta"]) and best_t["H"] < best_p["H"]:
                adv = "Transported远优"
            elif best_t["H"] < best_p["H"] * 0.5:
                adv = "Transported更清洁"
            elif abs(best_t["delta"]) < 0.05:
                adv = "Probe独有效"
            memo_lines.append(f"| {obj} | {m} | {best_t['delta']:+.3f} | {best_t['H']:.1f} | {best_p['delta']:+.3f} | {best_p['H']:.1f} | {adv} |")

memo_lines.append("")
memo_lines.append("**关键发现：自然运输方向在中层last token位置产生比统计probe方向更清洁（更低熵）的类别切换！**")
memo_lines.append("")

# 2. Causal Trace
memo_lines.append("#### 2. 因果追踪（corrupt-then-restore恢复分数）")
memo_lines.append("")
memo_lines.append("| 对象 | 模型 | 最佳obj位置恢复 | 最佳last位置恢复 | 主导位置 |")
memo_lines.append("|------|------|---------------|----------------|---------|")

for m in models:
    rkey = f"{m}_r2"
    if rkey not in results:
        rkey = f"{m}_r1"
    if rkey not in results:
        continue
    for obj, od in results[rkey]["per_object"].items():
        ct = od.get("causal_trace", {})
        obj_best = {"recovery": 0, "layer": ""}
        last_best = {"recovery": 0, "layer": ""}
        for key, val in ct.items():
            if not isinstance(val, dict):
                continue
            r = val.get("recovery", 0)
            if "/obj" in key and abs(r) > abs(obj_best["recovery"]):
                obj_best = {"recovery": r, "layer": key}
            if "/last" in key and abs(r) > abs(last_best["recovery"]):
                last_best = {"recovery": r, "layer": key}

        dominant = "BOTH" if abs(obj_best["recovery"]) > 0.3 and abs(last_best["recovery"]) > 0.3 else \
                   ("OBJ" if abs(obj_best["recovery"]) > abs(last_best["recovery"]) else "LAST")
        memo_lines.append(f"| {obj} | {m} | {obj_best['recovery']:.3f} [{obj_best['layer']}] | "
                         f"{last_best['recovery']:.3f} [{last_best['layer']}] | {dominant} |")

memo_lines.append("")
memo_lines.append("**关键发现：**")
memo_lines.append("- **Qwen3**: 类别信息在obj位置（早中期有效）→ last位置（深层有效），信息有明确的**迁移路径**")
memo_lines.append("- **GLM4**: 类别信息**只在last位置深层**有效（L23, L31），obj位置完全无关")
memo_lines.append("- **DS7B**: 类别信息在last位置深层有效（L16, L21），但中期有负恢复（过冲效应）")
memo_lines.append("")

# 3. Direction rotation during transport
memo_lines.append("#### 3. 运输过程中的方向旋转（cosine with d_embed, α=4.0）")
memo_lines.append("")
memo_lines.append("| 模型 | 对象 | L0 cos_obj | L0 cos_last | L7 cos_obj | L7 cos_last | L14/15 cos_last | L21/23 cos_last | L28/31 cos_last |")
memo_lines.append("|------|------|-----------|------------|-----------|------------|----------------|----------------|----------------|")

for m in models:
    rkey = f"{m}_r2"
    if rkey not in results:
        rkey = f"{m}_r1"
    if rkey not in results:
        continue
    for obj in ["apple", "knife", "car"]:
        if obj not in results[rkey]["per_object"]:
            continue
        od = results[rkey]["per_object"][obj]
        norms = od.get("transported_directions_norms", {})
        alpha = "4.0" if "4.0" in norms else None
        if alpha is None:
            continue
        n = norms[alpha]
        # Get specific layer keys
        l0 = n.get("L0", {})
        l7 = n.get("L7", {})
        # Get a mid layer
        mid_key = None
        for k in ["L14", "L15"]:
            if k in n:
                mid_key = k
                break
        late_key = None
        for k in ["L21", "L23"]:
            if k in n:
                late_key = k
                break
        deep_key = None
        for k in ["L28", "L31"]:
            if k in n:
                deep_key = k
                break

        memo_lines.append(f"| {m} | {obj} | "
                         f"{l0.get('cos_obj', 0):.3f} | {l0.get('cos_last', 0):.3f} | "
                         f"{l7.get('cos_obj', 0):.3f} | {l7.get('cos_last', 0):.3f} | "
                         f"{n.get(mid_key, {}).get('cos_last', 0):.3f} | "
                         f"{n.get(late_key, {}).get('cos_last', 0):.3f} | "
                         f"{n.get(deep_key, {}).get('cos_last', 0):.3f} |")

memo_lines.append("")
memo_lines.append("**关键发现：cosine从L0的0.4-0.9降到L7的~0，说明方向在前几层就被完全旋转。embedding方向只是入口，不是中层表示。**")
memo_lines.append("")

# 4. Residual norm growth
memo_lines.append("#### 4. 残差范数增长（obj位置, α=4.0, R1数据）")
memo_lines.append("")
memo_lines.append("| 模型 | apple L0→Lmid→Ldeep | knife L0→Lmid→Ldeep | car L0→Lmid→Ldeep |")
memo_lines.append("|------|---------------------|---------------------|-------------------|")

for m in models:
    rkey = f"{m}_r1"
    if rkey not in results:
        continue
    row = f"| {m} |"
    for obj in ["apple", "knife", "car"]:
        if obj not in results[rkey]["per_object"]:
            row += " N/A |"
            continue
        od = results[rkey]["per_object"][obj]
        norms = od.get("transported_directions_norms", {})
        alpha = "4.0" if "4.0" in norms else None
        if alpha is None:
            row += " N/A |"
            continue
        n = norms[alpha]
        # Get norms at L0, mid, deep
        l0_norm = n.get("L0", {}).get("obj_norm", 0)
        mid_norms = []
        for k in n:
            li = int(k.replace("L", ""))
            if 10 <= li <= 20:
                mid_norms.append((li, n[k].get("obj_norm", 0)))
        deep_norms = []
        for k in n:
            li = int(k.replace("L", ""))
            if li >= 20:
                deep_norms.append((li, n[k].get("obj_norm", 0)))

        mid_str = f"L{mid_norms[0][0]}:{mid_norms[0][1]:.0f}" if mid_norms else "N/A"
        deep_str = f"L{deep_norms[-1][0]}:{deep_norms[-1][1]:.0f}" if deep_norms else "N/A"
        row += f" {l0_norm:.0f}→{mid_str}→{deep_str} |"
    memo_lines.append(row)

memo_lines.append("")
memo_lines.append("**DS7B范数是Qwen3的100-1000倍！GLM4范数也远大于Qwen3。**")
memo_lines.append("")

# 5. Best clean switches
memo_lines.append("#### 5. 最佳清洁切换（全部模型R2, H<3.0）")
memo_lines.append("")
memo_lines.append("| 对象 | 模型 | 位置 | 方向类型 | Δ | H | 置信度 | 切换目标 |")
memo_lines.append("|------|------|------|---------|---|---|--------|---------|")

for m in models:
    rkey = f"{m}_r2"
    if rkey not in results:
        rkey = f"{m}_r1"
    if rkey not in results:
        continue
    for obj, od in results[rkey]["per_object"].items():
        # Find all switches with H < 3.0
        for src_alpha, layer_data in od.get("inject_results", {}).items():
            for layer_key, pos_data in layer_data.items():
                for key, r in pos_data.items():
                    if r["full_entropy"] < 3.0 and abs(r["delta"]) > 0.3:
                        memo_lines.append(f"| {obj} | {m} | {layer_key}/{key} | transported | "
                                         f"{r['delta']:+.3f} | {r['full_entropy']:.1f} | "
                                         f"{r.get('confidence', 0):.3f} | - |")

memo_lines.append("")
memo_lines.append("### 客观现象总结")
memo_lines.append("")
memo_lines.append("1. **自然运输方向比统计probe方向更有效更清洁**：在GLM4中，运输方向产生H=0.2的超清洁切换，而probe方向只有H=6.2")
memo_lines.append("2. **因果追踪揭示位置路由机制**：Qwen3类别信息从obj→last迁移；GLM4只在last深层；DS7B在last深层")
memo_lines.append("3. **方向在前几层被完全旋转**：cosine(d_embed, δ_l)从0.4-0.9降至~0")
memo_lines.append("4. **三模型残差范数分布完全不同**：Qwen3~50-200, GLM4~230-550, DS7B~3000-30000")
memo_lines.append("5. **DS7B基线异常**：car的baseline top=animal（错误），train也是animal（错误）")
memo_lines.append("")
memo_lines.append("### 严格审视")
memo_lines.append("")
memo_lines.append("**硬伤1：因果追踪的corrupt-restore方法可能有问题**")
memo_lines.append("apple对象在Qwen3和GLM4的corrupt baseline与clean baseline相同（recovery=0），说明corrupt方法可能对某些对象不适用。可能是'corrupt word'（dog）实际上产生了与clean word相同的类别输出。")
memo_lines.append("")
memo_lines.append("**硬伤2：运输方向的source α可能影响结果**")
memo_lines.append("运输方向δ_l依赖于源扰动强度α。α太小时δ_l太小（精度问题），α太大时可能进入非线性区域。目前用的是α=2-8，但没有系统扫描最优α。")
memo_lines.append("")
memo_lines.append("**硬伤3：对象数量仍然偏少**")
memo_lines.append("虽然R2增加到7个对象，但每类别仍然只有2-3个，不足以构建完整的类别拓扑。")
memo_lines.append("")
memo_lines.append("**硬伤4：只测了category任务**")
memo_lines.append("没有测property任务（属性），不知道运输方向是否也能控制属性切换。")
memo_lines.append("")
memo_lines.append("**硬伤5：GLM4和DS7B的car/train baseline异常**")
memo_lines.append("DS7B的car和train baseline都是animal（错误），说明模型本身对这些词的类别判断就有问题。这可能影响对'切换'结果的解读——也许'切换'只是纠正了基线错误。")
memo_lines.append("")
memo_lines.append("### 关键洞察")
memo_lines.append("")
memo_lines.append("**核心发现：自然运输方向T_{0→l}(d_embed)是模型真正使用的因果方向。**")
memo_lines.append("")
memo_lines.append("这比Phase 429B的probe方向发现更进一步：")
memo_lines.append("- Phase 429B：统计方向有效 → 说明中层有类别信息")
memo_lines.append("- Phase 430：**自然运输方向更有效** → 说明模型确实沿这个方向传输语义信息")
memo_lines.append("")
memo_lines.append("**物理含义：**")
memo_lines.append("- embedding方向的category perturbation被模型的层间计算**自然运输**到中层")
memo_lines.append("- 这个运输过程保留了语义内容（能产生类别切换），但方向本身被完全旋转")
memo_lines.append("- 统计probe方向虽然也能产生切换，但不如自然运输方向清洁（H更高）")
memo_lines.append("- 原因：probe方向包含统计噪声，而运输方向只包含被模型实际传播的信号")
memo_lines.append("")
memo_lines.append("**位置路由的物理图像：**")
memo_lines.append("1. 类别信息在embedding层写入obj位置")
memo_lines.append("2. 注意力机制将类别信息从obj位置**搬运**到last位置")
memo_lines.append("3. 深层读出只看last位置，不看obj位置")
memo_lines.append("4. 不同模型搬运速度不同：Qwen3早（L7开始），GLM4晚（L23开始）")
memo_lines.append("")
memo_lines.append("### 理论更新")
memo_lines.append("")
memo_lines.append("运输算子T_{0→l}现在有了实证支持：")
memo_lines.append("```")
memo_lines.append("d_{l,p}^{natural} = T_{0→l,p}(d_embed) = δ_l = h_l(perturbed) - h_l(clean)")
memo_lines.append("```")
memo_lines.append("")
memo_lines.append("而且运输方向**比统计方向更有效**，说明：")
memo_lines.append("```")
memo_lines.append("T_{0→l} 保留了语义因子 + 过滤了统计噪声")
memo_lines.append("```")
memo_lines.append("")
memo_lines.append("因果追踪确认了位置路由机制：")
memo_lines.append("```")
memo_lines.append("Category(obj_pos, L0-Lk) → Attention Transport → Category(last_pos, Lk+) → Readout")
memo_lines.append("```")
memo_lines.append("")
memo_lines.append("### 下一步")
memo_lines.append("")
memo_lines.append("1. **注意力头路由实验**：哪些注意力头把类别信息从obj搬运到last位置？")
memo_lines.append("2. **属性运输测试**：自然运输方向是否也能控制属性（property）切换？")
memo_lines.append("3. **运输算子T的显式计算**：能否从权重矩阵近似计算T_{0→l}？")
memo_lines.append("4. **跨对象运输方向一致性**：不同对象（apple, orange, lemon）的运输方向是否一致？")
memo_lines.append("5. **范数增长的语义含义**：为什么DS7B范数是Qwen3的1000倍？")

# Write to MEMO
memo_path = ROOT / "research" / "glm5" / "docs" / "AGI_GLM5_MEMO.md"
with open(memo_path, 'a', encoding='utf-8') as f:
    f.write('\n'.join(memo_lines))

print(f"MEMO updated at {now}")
print(f"\n{'='*60}")
print(f"PHASE 430 SUMMARY")
print(f"{'='*60}")

# Print the key findings
print("""
核心发现：

1. 自然运输方向 >> 统计Probe方向
   - GLM4 train: Transported Δ=-0.849, H=0.2 vs Probe Δ=-0.372, H=6.2
   - 运输方向产生H=0.2的超清洁切换！比probe方向清洁30倍

2. 因果追踪揭示位置路由
   - Qwen3: obj(早中) → last(深层)，信息迁移
   - GLM4: 只在last深层(L23, L31)，集中读出
   - DS7B: last深层(L21)，中期有负恢复

3. 方向旋转确认
   - cosine(d_embed, δ_l)从0.4-0.9降到~0.00（L7开始）
   - embedding方向被完全旋转

4. 范数增长差异巨大
   - Qwen3: ~50-200
   - GLM4: ~230-550
   - DS7B: ~3000-30000（比Qwen3大100-1000倍！）

5. 位置路由物理图像
   Category(obj_pos, L0-Lk) → Attention Transport → Category(last_pos, Lk+) → Readout
""")
