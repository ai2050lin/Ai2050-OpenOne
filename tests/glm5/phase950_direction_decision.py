#!/usr/bin/env python3
"""
Phase 950: 确定主攻方向 — 基于 Phase 946-949 的全部证据
=========================================================
综合分析三条路线的证据强度，确定下一阶段的主攻方向。

评分维度（每条路线5分制）：
- 跨模型一致性 (1-5): 发现在不同模型上是否一致
- 机制清晰度 (1-5): 因果机制是否清楚
- 理论深度 (1-5): 对解码语言数学结构的贡献
- 可操作性 (1-5): 能否设计精确的干预实验
- 整合潜力 (1-5): 能否与其他路线发现整合

总权重: 跨模型一致性×2 + 机制清晰度×1.5 + 理论深度×2 + 可操作性×1 + 整合潜力×1.5
"""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

import sys
sys.stdout.reconfigure(encoding="utf-8")

def log(msg):
    print(msg, flush=True)

def evaluate_routes():
    """Evaluate the three routes based on all evidence."""
    
    # ===== Evidence Matrix =====
    
    # Route A: Semantic Channel Bridge (Phase 940-947)
    route_a = {
        "name": "A: 语义通道桥接 (Semantic Channel Bridge)",
        "phases": [940, 941, 942, 943, 944, 947],
        "key_findings": [
            "color语义通过MLP gate通道编码 (qwen3: 3个共识通道, GLM4: 分布式编码)",
            "gate通道激活差异可达4.38 (GLM4 L25)",
            "跨模型桥接存在但编码模式不同 (qwen3集中 vs GLM4分布式)",
            "通道→边界移动的因果链尚未验证",
        ],
        "scores": {
            "cross_model_consistency": 2.5,
            "mechanism_clarity": 3.0,
            "theoretical_depth": 4.5,
            "operability": 3.5,
            "integration_potential": 3.5,
        },
        "rationale": {
            "cross_model_consistency": "桥接存在但模式不同 (qwen3 3-channel集中 vs GLM4 分布式)",
            "mechanism_clarity": "gate通道激活差异明确, 但通道→边界的因果链未验证",
            "theoretical_depth": "如果验证, 将证明语义以'通道激活模式'编码在MLP中",
            "operability": "可以精确操控特定通道进行干预实验",
            "integration_potential": "可以和Protocol场整合: 语义通道如何影响协议决策?",
        },
    }
    
    # Route B: Operator Algebra (Phase 946)
    route_b = {
        "name": "B: 算子代数 (Operator Algebra)",
        "phases": [946],
        "key_findings": [
            "算子形成3个独立族: 否定、量化、模态 (affinity cos<0.3)",
            "否定LOO余弦一致性: qwen3=0.59, GLM4=0.83, DS7B=0.77",
            "双否定闭合严重不一致: qwen3=0.87, GLM4=0.39, DS7B=0.77",
            "跨算子不变性: may→must ≈ might→must (cos=0.83-0.91)",
        ],
        "scores": {
            "cross_model_consistency": 1.5,
            "mechanism_clarity": 2.0,
            "theoretical_depth": 5.0,
            "operability": 2.0,
            "integration_potential": 2.0,
        },
        "rationale": {
            "cross_model_consistency": "双否定闭合差异极大 (0.39-0.87), 削弱了算子结构的通用性",
            "mechanism_clarity": "hidden state delta测量, 但缺乏归因和因果验证",
            "theoretical_depth": "如果成立, 将证明语言逻辑算子是嵌入空间的代数结构",
            "operability": "算子干预需要操控hidden state方向, 精度较差",
            "integration_potential": "与其它路线距离较远, 整合难度大",
        },
    }
    
    # Route C: Protocol Field (Phase 900-907, 948)
    route_c = {
        "name": "C: 协议场 (Protocol Field)",
        "phases": list(range(900, 908)) + [948],
        "key_findings": [
            "Protocol token受MLP主导 (Attn/MLP=0.44-0.79, 跨模型一致)",
            "最后一层MLP是protocol token logit的最强来源 (qwen3 L35_mlp: |Δ|=5.16)",
            "模型发现的protocol token是格式化标记 (空格、换行、冠词), 非预设语义类别",
            "Protocol MLP和语义MLP在层间负相关 (GLM4 Spearman r=-0.727)",
        ],
        "scores": {
            "cross_model_consistency": 5.0,
            "mechanism_clarity": 4.0,
            "theoretical_depth": 3.5,
            "operability": 4.0,
            "integration_potential": 4.0,
        },
        "rationale": {
            "cross_model_consistency": "MLP>Attn在所有三模型上一致, 最强的跨模型证据",
            "mechanism_clarity": "归零实验清晰显示最后一层MLP的主导性",
            "theoretical_depth": "揭示输出格式决策机制, 但理论普适性待验证",
            "operability": "归零+注入实验都可以精确执行",
            "integration_potential": "与语义通道在层级别有显著整合点 (L25重叠, Spearman r=-0.727)",
        },
    }
    
    return route_a, route_b, route_c


def compute_weighted_scores(routes):
    """Compute weighted scores and rank routes."""
    weights = {
        "cross_model_consistency": 2.0,
        "mechanism_clarity": 1.5,
        "theoretical_depth": 2.0,
        "operability": 1.0,
        "integration_potential": 1.5,
    }
    
    results = []
    for route in [routes[0], routes[1], routes[2]]:
        weighted = sum(route["scores"][k] * weights[k] for k in weights)
        results.append({
            "name": route["name"],
            "scores": route["scores"],
            "weighted_score": weighted,
            "max_possible": sum(5 * w for w in weights.values()),
        })
    
    results.sort(key=lambda x: -x["weighted_score"])
    return results


def generate_plan(ranked_routes, route_a, route_b, route_c):
    """Generate a phased research plan based on ranked routes."""
    
    plan = {
        "decision": "主攻路线C (Protocol场), 辅以路线A (语义通道桥接)",
        "rationale": {
            "why_c_as_main": [
                "跨模型一致性最高 (5.0/5): MLP>Attn在3模型上100%一致",
                "机制清晰度最高 (4.0/5): 归零实验结论明确, 因果方向清晰",
                "整合潜力最大: Protocol场和语义场在MLP层间的负相关揭示了关键的结构边界",
                "第二阶段可以自然延伸到'语义→协议转换机制'",
            ],
            "why_a_as_support": [
                "语义通道桥接的理论深度最高 (4.5/5), 是理解编码机制的关键",
                "通道操控实验精确, 适合作为因果验证工具",
                "与主线形成互补: A揭示'编码什么', C揭示'如何决策'",
            ],
            "why_not_b": [
                "跨模型一致性最差 (1.5/5), 特别是双否定闭合的巨大差异",
                "机制最不清晰 (2.0/5), hidden state delta缺乏归因",
                "与A/C路线距离较远, 难以整合",
                "但算子族的存在是重要观察, 将在Phase 951中做最小验证后归档",
            ],
        },
        "next_phases": [
            {
                "phase": 951,
                "title": "路线B清算: 算子代数最小化验证+归档",
                "goal": "验证算子族是否确实稳定存在, 如果是则归档为观察而非理论",
                "duration": "1 phase",
                "key_experiment": "用更大的刺激集(200+)仅测试算子族间的affinity矩阵, 确认3族结构",
            },
            {
                "phase": 952,
                "title": "Protocol场因果验证 (Route C深化)",
                "goal": "从归零(消融)升级为因果干预: 增强/抑制最后一层MLP的protocol方向",
                "duration": "2-3 phases",
                "key_experiments": [
                    "发现最后一层MLP中的'protocol方向' (通过protocol token的logit梯度)",
                    "沿protocol方向注入hidden state, 测量protocol token logit变化",
                    "测试protocol方向是否与语义方向正交",
                ],
            },
            {
                "phase": 953,
                "title": "语义→协议信息流追踪 (Route A+C 整合)",
                "goal": "追踪语义信息如何从中间MLP层传递到最后一层MLP",
                "duration": "2-3 phases",
                "key_experiments": [
                    "在L25注入color语义, 测量L39 hidden state的变化",
                    "残差流逐层追踪: 语义信号在残差流中的衰减/放大曲线",
                    "验证: L25激活→残差流→L39输入→protocol token logit 的因果链",
                ],
            },
            {
                "phase": 954,
                "title": "MLP双层编码理论: 框架构建",
                "goal": "基于Phase 952-953的数据, 构建'语义编码层+协议决策层'的双层MLP理论",
                "duration": "1-2 phases",
                "key_products": [
                    "MLP双层架构的数学模型",
                    "语义方向与协议方向的(非)正交性证明",
                    "跨模型统一性验证 (qwen3/GLM4/DS7B)",
                ],
            },
            {
                "phase": 955,
                "title": "第一性原理突破: 解码语言背后的数学结构",
                "goal": "基于完整数据集, 提出语言编码的数学理论",
                "duration": "1 phase (综合)",
                "approach": "不引入高级数学, 让拼图自然浮现",
            },
        ],
    }
    
    return plan


def main():
    log("=" * 70)
    log("Phase 950: 路线评估与方向决策")
    log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    log("=" * 70)
    
    route_a, route_b, route_c = evaluate_routes()
    routes = (route_a, route_b, route_c)
    
    # Print evaluation
    log("\n" + "=" * 70)
    log("一、三条路线证据矩阵")
    log("=" * 70)
    
    for route in routes:
        log(f"\n{'─'*50}")
        log(f"路线{route['name']}")
        log(f"涉及Phase: {route['phases']}")
        log(f"关键发现:")
        for f in route["key_findings"]:
            log(f"  - {f}")
        log(f"\n评分:")
        for k, v in route["scores"].items():
            name_map = {
                "cross_model_consistency": "跨模型一致性",
                "mechanism_clarity": "机制清晰度",
                "theoretical_depth": "理论深度",
                "operability": "可操作性",
                "integration_potential": "整合潜力",
            }
            log(f"  {name_map[k]:12s}: {v}/5 — {route['rationale'][k]}")
    
    # Compute weighted scores
    ranked = compute_weighted_scores(routes)
    
    log("\n" + "=" * 70)
    log("二、加权综合排名")
    log("=" * 70)
    log(f"\n{'路线':<40s} {'加权分':>8s} {'百分比':>8s}")
    log("-" * 56)
    for r in ranked:
        pct = r["weighted_score"] / r["max_possible"] * 100
        short_name = r["name"][:38]
        log(f"{short_name:<40s} {r['weighted_score']:>8.2f} {pct:>7.1f}%")
    
    # Generate plan
    plan = generate_plan(ranked, route_a, route_b, route_c)
    
    log("\n" + "=" * 70)
    log("三、主攻方向决策")
    log("=" * 70)
    log(f"\n决策: {plan['decision']}")
    
    log(f"\n为什么主攻Route C:")
    for r in plan["rationale"]["why_c_as_main"]:
        log(f"  [+] {r}")
    
    log(f"\n为什么辅以Route A:")
    for r in plan["rationale"]["why_a_as_support"]:
        log(f"  * {r}")
    
    log(f"\n为什么暂缓Route B:")
    for r in plan["rationale"]["why_not_b"]:
        log(f"  [-] {r}")
    
    log("\n" + "=" * 70)
    log("四、下一阶段路线图 (Phase 951-955)")
    log("=" * 70)
    
    for p in plan["next_phases"]:
        log(f"\nPhase {p['phase']}: {p['title']}")
        log(f"  目标: {p['goal']}")
        log(f"  预期时长: {p['duration']}")
        if "key_experiments" in p:
            log(f"  关键实验:")
            for exp in p["key_experiments"]:
                log(f"    - {exp}")
        if "key_products" in p:
            log(f"  关键产出:")
            for prod in p["key_products"]:
                log(f"    - {prod}")
    
    # Save
    output = {
        "phase": 950,
        "timestamp": datetime.now().isoformat(),
        "ranked_routes": [
            {"name": r["name"], "weighted_score": r["weighted_score"],
             "scores": r["scores"]}
            for r in ranked
        ],
        "decision": plan["decision"],
        "rationale": plan["rationale"],
        "next_phases": plan["next_phases"],
    }
    
    out_path = ROOT / "results" / "phase950_direction" / "decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n决策文件已保存: {out_path}")
    
    return output


if __name__ == "__main__":
    main()
