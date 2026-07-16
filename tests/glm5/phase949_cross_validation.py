#!/usr/bin/env python3
"""
Phase 949: 跨方向交叉验证 — 三条路线发现是否指向同一机制？
========================================================
综合 Phase 946（算子代数）, Phase 947（语义通道桥接）, Phase 948（协议场审计）的数据，
寻找跨路线的结构关系。

核心假设: 语义场和协议场是两个独立子系统，但共享MLP通道。
"""

import json, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
RESULT_946 = ROOT / "results" / "phase946_operator_algebra"
RESULT_947 = ROOT / "results" / "phase947_glm4_color_bridge"
RESULT_948 = ROOT / "results" / "phase948_protocol_field_audit"

MODELS = ["qwen3", "glm4", "deepseek7b"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def safe_decode(tokenizer, tid):
    try:
        r = tokenizer.decode([tid])
        return r if r else f"<tok_{tid}>"
    except:
        return f"<tok_{tid}>"


def analyze_phase946_layer_distribution(model_name):
    """Extract which layers show strongest operator encoding from Phase 946."""
    path = RESULT_946 / model_name / "summary.json"
    if not path.exists():
        log(f"  No Phase 946 data for {model_name}")
        return None

    data = load_json(path)
    
    # Try to find operator-related layer data
    result = {"model": model_name, "operator_layers": {}}
    
    # Check different possible data structures
    if "operator_metrics" in data:
        for op_name, op_data in data["operator_metrics"].items():
            if isinstance(op_data, dict) and "layer_stats" in op_data:
                layer_stats = op_data["layer_stats"]
                if isinstance(layer_stats, dict):
                    # Find layers with max cos
                    best_layer = max(layer_stats.items(), key=lambda x: float(x[1]) if x[1] is not None else 0)
                    result["operator_layers"][op_name] = {
                        "best_layer": best_layer[0],
                        "best_value": float(best_layer[1]) if best_layer[1] else 0,
                        "all_layers": {k: float(v) if v else 0 for k, v in layer_stats.items()}
                    }
    
    # Try alternative data structure
    if "negation" in data:
        neg = data["negation"]
        if "layer_loo_cos" in neg:
            result["negation_layers"] = {k: float(v) for k, v in neg["layer_loo_cos"].items()}
    
    # Try the full results format
    if "results" in data:
        results = data["results"]
        # Extract negation LOO scores per layer
        for key in results:
            if "negation" in key.lower() and "layer" in str(results[key]):
                result["negation_data"] = results[key]
    
    return result


def analyze_phase947_channel_distribution(model_name):
    """Extract which layers have strongest MLP channel activation gaps."""
    # Try multiple path patterns
    candidates = []
    for pattern in [f"{model_name}_minimal.json", f"{model_name}.json", 
                    f"**/{model_name}*.json", "**/summary.json"]:
        candidates.extend(list(RESULT_947.glob(pattern)))
    # Also try phase940 for qwen3
    if model_name == "qwen3":
        for pattern in ["**/qwen3*.json", "**/summary.json"]:
            p940 = ROOT / "results" / "phase940_color_boundary"
            if p940.exists():
                candidates.extend(list(p940.glob(pattern)))
    
    if not candidates:
        log(f"  No Phase 947 result files for {model_name}")
        return None
    
    path = candidates[0]
    log(f"  Loading Phase 947 from: {path}")
    
    data = load_json(path)
    
    result = {"model": model_name, "channel_layers": {}, "channel_layers_avg": {}}
    
    # Handle GLM4 minimal.json format: {"15": {act_gap_gate: ...}, "20": {...}, ...}
    if isinstance(data, dict):
        for key, val in data.items():
            # Check if key is a numeric layer index
            try:
                li = int(key)
            except (ValueError, TypeError):
                continue
            
            if isinstance(val, dict):
                # Extract gate_gap
                gap = val.get("act_gap_gate") or val.get("gate_gap") or val.get("mean_gap") or 0
                if gap:
                    result["channel_layers_avg"][li] = float(gap)
    
    # If still empty, try list format
    if not result["channel_layers_avg"]:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    li = item.get("layer") or item.get("layer_idx")
                    gap = item.get("gate_gap_max") or item.get("mean_gap") or item.get("max_gap", 0)
                    if li is not None:
                        result["channel_layers_avg"][int(li)] = float(gap) if gap else 0
    
    if result["channel_layers_avg"]:
        best = max(result["channel_layers_avg"].items(), key=lambda x: x[1])
        log(f"  Best channel layer: L{best[0]} (gap={best[1]:.3f})")
    
    return result


def analyze_phase948_protocol_distribution(model_name):
    """Extract Attn/MLP contribution per layer from Phase 948."""
    path = RESULT_948 / model_name / "component_attribution.json"
    if not path.exists():
        log(f"  No Phase 948 data for {model_name}")
        return None
    
    data = load_json(path)
    agg = data.get("attributions", {})
    
    # Aggregate by component and layer
    by_layer_comp = defaultdict(lambda: defaultdict(list))
    for tid_str, comp_data in agg.items():
        for key, val in comp_data.items():
            if "mean_delta" in val:
                # Parse L{layer}_{comp}
                parts = key.split("_")
                if len(parts) >= 2 and parts[0].startswith("L"):
                    layer = int(parts[0][1:])
                    comp = parts[1]
                    by_layer_comp[layer][comp].append(abs(val["mean_delta"]))
    
    result = {"model": model_name, "protocol_layers": {}}
    for layer, comps in by_layer_comp.items():
        for comp, vals in comps.items():
            key = f"L{layer}_{comp}"
            result["protocol_layers"][key] = {
                "mean_abs_delta": float(np.mean(vals)),
                "n_tokens": len(vals),
            }
    
    # Also extract overall summary
    summary_path = RESULT_948 / model_name / "summary.json"
    if summary_path.exists():
        s = load_json(summary_path)
        if "attribution" in s:
            result["attn_mlp_ratio"] = s["attribution"].get("attn_vs_mlp_ratio") or s["attribution"].get("attn_mlp_ratio", 0)
        if "protocol_discovery" in s:
            result["n_protocol_candidates"] = s["protocol_discovery"].get("n_candidates", 0)
            result["n_high_cross"] = s["protocol_discovery"].get("n_high_cross", 0)
    
    return result


def cross_reference_layers(p946, p947, p948, model_name):
    """Cross-reference: do semantic and protocol layers overlap?"""
    results = {"model": model_name}
    
    if not p948 or "protocol_layers" not in p948:
        return results
    
    # Get top protocol layers (by MLP contribution)
    mlp_layers = {k: v for k, v in p948["protocol_layers"].items() if "mlp" in k}
    sorted_mlp = sorted(mlp_layers.items(), key=lambda x: -x[1]["mean_abs_delta"])
    
    # Get top channel layers (Phase 947)
    channel_top = []
    if p947 and "channel_layers_avg" in p947:
        sorted_ch = sorted(p947["channel_layers_avg"].items(), key=lambda x: -x[1])
        channel_top = sorted_ch[:5]
    
    results["top_protocol_mlp_layers"] = [(k, v["mean_abs_delta"]) for k, v in sorted_mlp[:5]]
    results["top_channel_layers"] = channel_top
    
    # Check overlap
    protocol_layer_nums = set()
    for k in [x[0] for x in sorted_mlp[:5]]:
        num = int(k.split("_")[0][1:])
        protocol_layer_nums.add(num)
    
    channel_layer_nums = set()
    for k, _ in channel_top:
        channel_layer_nums.add(int(k) if isinstance(k, (int, str)) else k)
    
    results["overlapping_layers"] = list(protocol_layer_nums & channel_layer_nums)
    results["protocol_only_layers"] = list(protocol_layer_nums - channel_layer_nums)
    results["channel_only_layers"] = list(channel_layer_nums - protocol_layer_nums)
    
    # Compute rank correlation between protocol_mlp_strength and channel_gap across layers
    if p947 and "channel_layers_avg" in p947:
        all_layers = sorted(set(
            list(protocol_layer_nums) + list(channel_layer_nums)
        ))
        proto_vals, chan_vals = [], []
        for li in all_layers:
            proto_key = f"L{li}_mlp"
            if proto_key in mlp_layers:
                proto_vals.append(mlp_layers[proto_key]["mean_abs_delta"])
            else:
                proto_vals.append(0)
            if li in p947["channel_layers_avg"]:
                chan_vals.append(p947["channel_layers_avg"][li])
            else:
                chan_vals.append(0)
        
        if len(proto_vals) >= 3:
            try:
                from scipy.stats import spearmanr
                corr, pval = spearmanr(proto_vals, chan_vals)
                results["spearman_r_protocol_vs_channel"] = float(corr)
                results["spearman_p"] = float(pval)
            except ImportError:
                # Manual Spearman
                def rankdata(a):
                    n = len(a)
                    ikeys = sorted(range(n), key=lambda i: a[i])
                    result = [0] * n
                    i = 0
                    while i < n:
                        j = i
                        while j < n and a[ikeys[j]] == a[ikeys[i]]:
                            j += 1
                        rank = (i + j - 1) / 2.0
                        for k in range(i, j):
                            result[ikeys[k]] = rank
                        i = j
                    return result
                r1 = rankdata(proto_vals)
                r2 = rankdata(chan_vals)
                n = len(r1)
                d2 = sum((r1[i] - r2[i])**2 for i in range(n))
                corr = 1 - 6 * d2 / (n * (n**2 - 1))
                results["spearman_r_protocol_vs_channel"] = float(corr)
    
    return results


def synthesize_theory(model_results):
    """Synthesize a unified theory from cross-route evidence."""
    synthesis = {
        "route_summary": {},
        "unified_hypothesis": {},
        "evidence_for_unification": [],
        "evidence_against_unification": [],
    }
    
    for model_name, res in model_results.items():
        p946 = res.get("p946") or {}
        p948 = res.get("p948") or {}
        cross_ref = res.get("cross_ref") or {}
        
        synthesis["route_summary"][model_name] = {
            "route_b_operator": p946.get("operator_layers", {}),
            "route_a_channel": {
                "top_layers": cross_ref.get("top_channel_layers", [])[:3]
            },
            "route_c_protocol": {
                "attn_mlp_ratio": p948.get("attn_mlp_ratio", 0),
                "top_mlp_layers": cross_ref.get("top_protocol_mlp_layers", [])[:3],
                "n_candidates": p948.get("n_protocol_candidates", 0),
            },
        }
    
    # Unification evidence
    # Evidence 1: MLP dominance in both protocol and semantic encoding
    synthesis["evidence_for_unification"].append({
        "id": "E1_mlp_centrality",
        "description": "MLP在所有三条路线中都是关键组件：算子编码依赖MLP输出，"
                       "语义通道在MLP中实现，protocol token由MLP主导（Attn/MLP<1 in all models）",
        "strength": "strong",
    })
    
    # Evidence 2: Layer concentration pattern
    synthesis["evidence_for_unification"].append({
        "id": "E2_final_layer_convergence",
        "description": "Protocol token在最后一层MLP最强；语义编码在中层MLP最强；"
                       "暗示MLP各层分工：前中层做语义编码，最后一层做输出协议决策",
        "strength": "moderate",
    })
    
    # Evidence against
    synthesis["evidence_against_unification"].append({
        "id": "A1_different_dominance_patterns",
        "description": "语义编码（Phase 947）在不同模型用不同MLP通道，"
                       "但protocol编码（Phase 948）的Attn/MLP比例在跨模型间更一致。"
                       "这可能意味着它们不是同一个底层电路。",
        "strength": "moderate",
    })
    
    synthesis["evidence_against_unification"].append({
        "id": "A2_different_layer_profiles",
        "description": "Protocol token主要受最后一层MLP影响，"
                       "而语义通道在中间层分布（如qwen3 L18-22 color通道，GLM4 L25）。"
                       "层分布的系统性差异暗示不同的电路。",
        "strength": "strong",
    })
    
    # Unified hypothesis
    synthesis["unified_hypothesis"] = {
        "main": "MLP是统一的编码枢纽，但语义场和协议场使用MLP的不同层和不同通道。",
        "architecture": "前中层MLP → 语义编码（概念、算子、属性）→ 中层输出 → "
                        "最后一层MLP → 协议决策（格式、停止、结构）→ 最终logit",
        "channel_sharing": "可能存在跨层的通道复用：深层MLP通道同时参与语义和协议编码",
        "key_unknown": "语义→协议的转换是否在最后一层MLP中完成？"
                       "还是语义和协议在hidden state中是正交子空间？",
        "testable_prediction": "如果归零语义热点层（如L18）的MLP，protocol token logit不应显著变化；"
                              "如果归零最后一层MLP，protocol token logit应崩溃但语义方向应保留。",
    }
    
    return synthesis


def main():
    log("Phase 949: Cross-route Validation")
    log("=" * 60)
    
    model_results = {}
    
    for model_name in MODELS:
        log(f"\n--- {model_name} ---")
        
        p946 = analyze_phase946_layer_distribution(model_name)
        p947 = analyze_phase947_channel_distribution(model_name)
        p948 = analyze_phase948_protocol_distribution(model_name)
        
        log(f"  Phase 946 (operator): {'found' if p946 else 'NOT FOUND'}")
        log(f"  Phase 947 (channel):  {'found' if p947 else 'NOT FOUND'}")
        log(f"  Phase 948 (protocol): {'found' if p948 else 'NOT FOUND'}")
        
        cross_ref = cross_reference_layers(p946, p947, p948, model_name)
        
        model_results[model_name] = {
            "p946": p946,
            "p947": p947,
            "p948": p948,
            "cross_ref": cross_ref,
        }
        
        # Print cross-reference findings
        log(f"  Top protocol MLP layers: {cross_ref.get('top_protocol_mlp_layers', [])[:3]}")
        log(f"  Top channel layers: {cross_ref.get('top_channel_layers', [])[:3]}")
        log(f"  Overlapping layers: {cross_ref.get('overlapping_layers', [])}")
        
        if "spearman_r_protocol_vs_channel" in cross_ref:
            r = cross_ref["spearman_r_protocol_vs_channel"]
            p = cross_ref.get("spearman_p", 0)
            log(f"  Spearman r(protocol vs channel): {r:.3f} (p={p:.3f})")
    
    # Synthesis
    synthesis = synthesize_theory(model_results)
    
    # Save
    output = {
        "phase": 949,
        "model_results": {
            m: {
                "cross_ref": r["cross_ref"],
            }
            for m, r in model_results.items()
        },
        "synthesis": synthesis,
    }
    
    out_path = ROOT / "results" / "phase949_cross_validation" / "synthesis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nSaved to {out_path}")
    
    # Save to tmp for inspection
    tmp_path = ROOT / "tmp" / "phase949_synthesis.json"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Print synthesis
    log("\n" + "=" * 60)
    log("SYNTHESIS")
    log("=" * 60)
    h = synthesis["unified_hypothesis"]
    log(f"\nMain hypothesis: {h['main']}")
    log(f"\nArchitecture: {h['architecture']}")
    log(f"\nChannel sharing: {h['channel_sharing']}")
    log(f"\nKey unknown: {h['key_unknown']}")
    log(f"\nTestable prediction: {h['testable_prediction']}")
    
    log(f"\nEvidence FOR unification ({len(synthesis['evidence_for_unification'])}):")
    for e in synthesis["evidence_for_unification"]:
        log(f"  [{e['strength']}] {e['id']}: {e['description'][:100]}...")
    
    log(f"\nEvidence AGAINST unification ({len(synthesis['evidence_against_unification'])}):")
    for e in synthesis["evidence_against_unification"]:
        log(f"  [{e['strength']}] {e['id']}: {e['description'][:100]}...")
    
    return output


if __name__ == "__main__":
    result = main()
