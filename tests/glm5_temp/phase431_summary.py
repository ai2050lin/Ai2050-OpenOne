"""Phase 431 Summary + MEMO Update"""
import sys, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
ROOT = Path(r"d:\Ai2050\TransformerLens-Project")

now = time.strftime("%Y-%m-%d %H:%M")
memo_lines = []

memo_lines.append(f"\n## Phase 431: Attention Head Routing [{now}]")
memo_lines.append("")
memo_lines.append("### 实验原理")
memo_lines.append("")
memo_lines.append("Phase 430因果追踪确认类别信息从obj位置经注意力搬运到last位置。本阶段用output_attentions=True提取所有层的注意力权重，找出哪些注意力头负责搬运。")
memo_lines.append("")
memo_lines.append("方法：")
memo_lines.append("1. 提取所有层所有头的注意力权重，计算last_pos→obj_pos的注意力")
memo_lines.append("2. 计算routing_score = attn_weight × |cos(W_o_head, d_cat)|")
memo_lines.append("3. 找出top routing heads")
memo_lines.append("")
memo_lines.append("### 核心数据")
memo_lines.append("")
memo_lines.append("#### 1. 通用复制注意力头（last→obj注意力 >0.85，跨对象一致）")
memo_lines.append("")
memo_lines.append("| 模型 | 层 | 头 | apple attn | knife attn | car attn | 特征 |")
memo_lines.append("|------|---|---|-----------|-----------|---------|------|")

# Load results
models_data = {}
for m in ["qwen3", "glm4", "deepseek7b"]:
    f = ROOT / f"results/phase431_attention_head_routing/{m}_phase431_r1.json"
    if f.exists():
        with open(f) as fh:
            models_data[m] = json.load(fh)

# Find universal copy heads (high attn across all objects)
for m, data in models_data.items():
    for obj, od in data["per_object"].items():
        cross_attn = od.get("cross_attention", {})
        # Collect all heads with their attention weights
        break

# Universal heads analysis
for m, data in models_data.items():
    # For each (layer, head), collect attn across objects
    head_attns = {}  # {(li, hi): {obj: attn}}
    for obj, od in data["per_object"].items():
        cross_attn = od.get("cross_attention", {})
        for li_str, heads in cross_attn.items():
            for hi_str, attn in heads.items():
                key = (int(li_str), int(hi_str))
                if key not in head_attns:
                    head_attns[key] = {}
                head_attns[key][obj] = attn

    # Find universal heads (high attn across all 3 objects)
    universal = []
    for (li, hi), obj_attns in head_attns.items():
        if len(obj_attns) >= 3:
            min_attn = min(obj_attns.values())
            avg_attn = sum(obj_attns.values()) / len(obj_attns)
            if min_attn > 0.7:  # Universal copy head
                universal.append((li, hi, avg_attn, obj_attns))

    universal.sort(key=lambda x: x[2], reverse=True)

    for li, hi, avg, obj_attns in universal[:5]:
        a_attn = obj_attns.get("apple", 0)
        k_attn = obj_attns.get("knife", 0)
        c_attn = obj_attns.get("car", 0)
        memo_lines.append(f"| {m} | L{li} | H{hi} | {a_attn:.3f} | {k_attn:.3f} | {c_attn:.3f} | avg={avg:.3f} |")

memo_lines.append("")
memo_lines.append("#### 2. 关键路由层（最高注意力权重层）")
memo_lines.append("")
memo_lines.append("| 模型 | 对象 | 最高注意力层 | 头 | 注意力权重 |")
memo_lines.append("|------|------|------------|---|-----------|")

for m, data in models_data.items():
    for obj, od in data["per_object"].items():
        cross_attn = od.get("cross_attention", {})
        best_li, best_hi, best_attn = 0, 0, 0
        for li_str, heads in cross_attn.items():
            for hi_str, attn in heads.items():
                if attn > best_attn:
                    best_attn = attn
                    best_li = int(li_str)
                    best_hi = int(hi_str)
        memo_lines.append(f"| {m} | {obj} | L{best_li} | H{best_hi} | {best_attn:.4f} |")

memo_lines.append("")
memo_lines.append("#### 3. Routing Score Top Heads")
memo_lines.append("")
memo_lines.append("| 模型 | 对象 | Top Head | Routing Score | Attn | cos_cat |")
memo_lines.append("|------|------|---------|-------------|------|---------|")

for m, data in models_data.items():
    for obj, od in data["per_object"].items():
        routing = od.get("routing_scores", {})
        best_head = ("", "", 0)
        for li_str, heads in routing.items():
            for hi_str, d in heads.items():
                if d["routing_score"] > best_head[2]:
                    best_head = (li_str, hi_str, d["routing_score"])
                    best_data = d
        if best_head[2] > 0:
            memo_lines.append(f"| {m} | {obj} | L{best_head[0]}/H{best_head[1]} | "
                             f"{best_head[2]:.6f} | {best_data['attn_last_to_obj']:.4f} | "
                             f"{best_data['cos_with_category']:.4f} |")

memo_lines.append("")
memo_lines.append("### 客观现象总结")
memo_lines.append("")
memo_lines.append("1. **存在通用复制注意力头**：GLM4的L4/H17在所有对象上都有0.92+的last→obj注意力，是通用的'从对象复制信息'头")
memo_lines.append("2. **DS7B的L27/H12/H10是深层复制头**：在最深层有0.96+的注意力，几乎100%从obj位置读取")
memo_lines.append("3. **Qwen3的L14/H12是中层层复制头**：在L14有0.97的注意力")
memo_lines.append("4. **三模型都有L3左右的早层通用头**：Qwen3 L3/H16, GLM4 L1-L4, DS7B L3/H23")
memo_lines.append("5. **Routing score很小**（最大0.028），因为cos_with_category很低（~0.05），说明W_o投影后方向与embedding类别方向的对齐很弱")
memo_lines.append("")
memo_lines.append("### 严格审视")
memo_lines.append("")
memo_lines.append("**硬伤1：注意力权重不等于因果贡献**")
memo_lines.append("高注意力权重只说明模型'看了'obj位置，但不代表这些信息被用于类别判断。需要zero-out头验证因果性。")
memo_lines.append("")
memo_lines.append("**硬伤2：没有做实际的头消融实验**")
memo_lines.append("由于HuggingFace API限制，没能实现单头消融。只有注意力权重的相关性证据，没有因果证据。")
memo_lines.append("")
memo_lines.append("**硬伤3：cos_with_category接近0**")
memo_lines.append("Routing score的核心问题是W_o投影后的方向与embedding类别方向几乎不正交。这说明用embedding类别方向来评估路由是不合适的——需要用层特异方向。")
memo_lines.append("")
memo_lines.append("**硬伤4：序列长度很短（6-8 tokens）**")
memo_lines.append("短序列中last→obj的注意力自然很高（位置选择有限）。需要更长的上下文来验证。")
memo_lines.append("")
memo_lines.append("### 关键洞察")
memo_lines.append("")
memo_lines.append("**核心发现：每个模型都有特定的'信息搬运'注意力头，它们将对象token的信息复制到last token位置。**")
memo_lines.append("")
memo_lines.append("这些搬运头的特征：")
memo_lines.append("- **高注意力权重**：last→obj注意力0.7-1.0")
memo_lines.append("- **跨对象通用**：同一个头在不同对象上都有高注意力")
memo_lines.append("- **层特异**：不同模型在不同层有搬运头")
memo_lines.append("  - Qwen3: L3-6（早）+ L14（中）")
memo_lines.append("  - GLM4: L1-4（极早）")
memo_lines.append("  - DS7B: L3（早）+ L14-16（中）+ L27（深）")
memo_lines.append("")
memo_lines.append("**物理图像更新：**")
memo_lines.append("```")
memo_lines.append("Category Embedding (obj_pos) ")
memo_lines.append("    ↓ Attention Heads (L3-L14) copy to last_pos")
memo_lines.append("Category Info (last_pos) ")
memo_lines.append("    ↓ Deep layers (L23-L31) refine for readout")
memo_lines.append("Final Category Prediction")
memo_lines.append("```")
memo_lines.append("")
memo_lines.append("### 下一步")
memo_lines.append("")
memo_lines.append("1. **实际头消融**：用TransformerLens或其他方法实现单头消融，验证因果性")
memo_lines.append("2. **长上下文测试**：用更长的prompt测试注意力模式是否稳定")
memo_lines.append("3. **跨类别注意力差异**：比较fruit vs animal vs tool的注意力模式差异")
memo_lines.append("4. **注意力头与运输方向的对应**：哪些头贡献了自然运输方向中的语义信息？")
memo_lines.append("5. **深层注意力头功能**：DS7B L27的复制头为什么在最深层？它在做什么？")

# Write to MEMO
memo_path = ROOT / "research" / "glm5" / "docs" / "AGI_GLM5_MEMO.md"
with open(memo_path, 'a', encoding='utf-8') as f:
    f.write('\n'.join(memo_lines))

print(f"MEMO updated at {now}")
print(f"\nPhase 431 complete. Key findings documented.")
