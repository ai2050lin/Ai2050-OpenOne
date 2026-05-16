"""Phase 192 综合对比: 路由因果性与最小语义回路 — 三模型"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import json
import numpy as np
from pathlib import Path

def load_results():
    base = Path("tests/glm5_temp")
    results = {}
    for model in ["qwen3", "glm4", "deepseek7b"]:
        files = sorted(base.glob(f"phase192_{model}_*.json"))
        if files:
            with open(files[-1], 'r', encoding='utf-8') as f:
                results[model] = json.load(f)
    return results

def main():
    results = load_results()
    models = list(results.keys())
    print(f"加载模型: {models}")

    # ===== 1. Head因果贡献 Top-10 对比 =====
    print(f"\n{'='*80}")
    print(f"1. Head因果贡献 Top-5 对比 (5种语义功能)")
    print(f"{'='*80}")

    func_types = ["negation", "tense", "role_binding", "question", "conditional"]
    for func in func_types:
        print(f"\n  --- {func} Top-5 heads ---")
        for model in models:
            exp1 = results[model].get("exp1", {})
            func_data = exp1.get(func, {})
            if not func_data:
                print(f"    {model}: 无数据")
                continue
            sorted_heads = sorted(func_data.items(), key=lambda x: float(x[1]), reverse=True)[:5]
            heads_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted_heads)
            print(f"    {model}: {heads_str}")

    # ===== 2. 关键Head层分布 =====
    print(f"\n{'='*80}")
    print(f"2. 关键Head层分布 (Top-20%)")
    print(f"{'='*80}")

    for func in func_types:
        print(f"\n  --- {func} ---")
        for model in models:
            exp1 = results[model].get("exp1", {})
            func_data = exp1.get(func, {})
            if not func_data:
                continue
            total = len(func_data)
            top_k = max(int(total * 0.20), 10)
            sorted_heads = sorted(func_data.items(), key=lambda x: float(x[1]), reverse=True)[:top_k]
            # 统计层分布
            layer_counts = {}
            for key, val in sorted_heads:
                layer = key.split("H")[0]
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            # 前后层分布
            n_layers = results[model]["model_info"]["n_layers"]
            early = sum(v for k, v in layer_counts.items() if int(k[1:]) < n_layers // 3)
            mid = sum(v for k, v in layer_counts.items() if n_layers // 3 <= int(k[1:]) < 2 * n_layers // 3)
            late = sum(v for k, v in layer_counts.items() if int(k[1:]) >= 2 * n_layers // 3)
            print(f"    {model}: early={early}, mid={mid}, late={late}")

    # ===== 3. 程序重叠度 (Jaccard) 对比 =====
    print(f"\n{'='*80}")
    print(f"3. 程序重叠度 (Jaccard) 对比")
    print(f"{'='*80}")

    pairs = [
        ("negation", "tense"), ("negation", "role_binding"),
        ("negation", "question"), ("negation", "conditional"),
        ("tense", "role_binding"), ("tense", "question"),
        ("tense", "conditional"), ("role_binding", "question"),
        ("role_binding", "conditional"), ("question", "conditional"),
    ]

    print(f"\n  {'Function Pair':<30}", end="")
    for model in models:
        print(f"  {model:>12}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for model in models:
        print(f"  {'-'*12}", end="")
    print()

    for f1, f2 in pairs:
        label = f"{f1} vs {f2}"
        print(f"  {label:<30}", end="")
        for model in models:
            exp3 = results[model].get("exp3", {})
            overlap = exp3.get("overlap", {})
            key = f"{f1}_vs_{f2}"
            jaccard = overlap.get(key, {}).get("jaccard", 0)
            print(f"  {jaccard:>12.4f}", end="")
        print()

    # ===== 4. 专用Head数对比 =====
    print(f"\n{'='*80}")
    print(f"4. 专用Head数 (仅1功能使用) 对比")
    print(f"{'='*80}")

    print(f"\n  {'Function':<20}", end="")
    for model in models:
        print(f"  {model:>12}", end="")
    print()
    for func in func_types:
        print(f"  {func:<20}", end="")
        for model in models:
            exp3 = results[model].get("exp3", {})
            dedicated = exp3.get("dedicated_by_function", {})
            print(f"  {dedicated.get(func, 0):>12}", end="")
        print()

    # ===== 5. 最小回路大小 =====
    print(f"\n{'='*80}")
    print(f"5. 最小回路大小 (50%/80%/95%贡献所需head数)")
    print(f"{'='*80}")

    for func in ["negation", "role_binding"]:
        print(f"\n  {func}:")
        for model in models:
            exp2 = results[model].get("exp2", {})
            func_data = exp2.get(func, {})
            if not func_data:
                print(f"    {model}: 无数据")
                continue
            k50 = func_data.get("k_50", 0)
            k80 = func_data.get("k_80", 0)
            k95 = func_data.get("k_95", 0)
            total = func_data.get("total_heads", 1)
            print(f"    {model}: 50%={k50}({k50/total*100:.1f}%), "
                  f"80%={k80}({k80/total*100:.1f}%), "
                  f"95%={k95}({k95/total*100:.1f}%)")

    # ===== 6. 路由图差异 =====
    print(f"\n{'='*80}")
    print(f"6. 通信图差异 (语义变体 vs 原句)")
    print(f"{'='*80}")

    print(f"\n  {'Function':<20}", end="")
    for model in models:
        print(f"  {model:>12}", end="")
    print()
    for func in func_types:
        print(f"  {func:<20}", end="")
        for model in models:
            exp4 = results[model].get("exp4", {})
            func_data = exp4.get(func, {})
            graph_diff = func_data.get("graph_diff", 0)
            print(f"  {graph_diff:>12.4f}", end="")
        print()

    # ===== 7. MLP贡献Top层 =====
    print(f"\n{'='*80}")
    print(f"7. MLP贡献Top-3层对比")
    print(f"{'='*80}")

    for func in func_types:
        print(f"\n  {func}:")
        for model in models:
            exp5 = results[model].get("exp5", {})
            func_data = exp5.get(func, {})
            if not func_data:
                print(f"    {model}: 无数据")
                continue
            sorted_layers = sorted(func_data.items(), key=lambda x: float(x[1]), reverse=True)[:3]
            layers_str = ", ".join(f"{k}={float(v):.1f}" for k, v in sorted_layers)
            print(f"    {model}: {layers_str}")

    # ===== 8. 核心发现总结 =====
    print(f"\n{'='*80}")
    print(f"8. 核心发现总结")
    print(f"{'='*80}")

    print("""
  ★ 发现1: 语义功能的Head特化模式
    - 否定(negation): 关键head在浅层(L0-1)和深层(L35/L39/L27)
    - 时态(tense): 贡献最分散, 无明显集中head (最小差异)
    - 角色绑定(role_binding): 关键head在中层(L11-20), 专用head最多
    - 疑问(question): 关键head在浅层(L0-5)
    - 条件(conditional): 关键head在浅层+深层, 与疑问高度重叠

  ★ 发现2: 程序复用模式 (三模型一致)
    - question ↔ conditional 重叠最高 (Jaccard≈0.45-0.57)
    - role_binding ↔ question 重叠最低 (Jaccard≈0.18-0.28)
    - → 疑问和条件共享"推理路由"子程序
    - → 角色绑定是独立的"绑定程序"

  ★ 发现3: 最小回路大小 (≈20% heads for 50%)
    - 否定: 50%贡献需≈20% heads
    - 角色绑定: 50%贡献需≈25-28% heads (更分散)
    - → 语义功能是"分布式程序", 不是单head功能

  ★ 发现4: 路由图差异 (通信模式)
    - 条件句通信图差异最大 (1.1-2.8)
    - 时态差异最小 (0.19-0.27)
    - → 条件句激活了全新的信息路由, 时态仅微调

  ★ 发现5: MLP门控集中在最后2-3层
    - 所有功能的MLP贡献都在最后3层最大
    - → MLP可能在"写入"最终语义状态
    - → Attention负责"路由", MLP负责"写入"

  ★★★ 关键洞察: 语义 = 分布式程序, 不是向量方向 ★★★
    1. 每个语义功能调用≈20-28%的heads (不是1个head)
    2. 不同功能有共享子程序 (Jaccard≈0.2-0.6)
    3. 程序有层次: 浅层路由 → 中层绑定 → 深层写入
    4. 条件/疑问共享推理子程序, 角色绑定独立
    5. 这支持"Transformer = 动态程序解释器"假说
    """)

if __name__ == "__main__":
    main()
