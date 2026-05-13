"""Phase 147b: 随机权重对照 + Jacobian非线性交互 (Qwen3)
============================================================
Exp 2修复版: 先释放训练模型, 再加载随机模型到GPU
Exp 3: 非线性交互测试

用法:
  python tests/glm5_temp/phase147b_exp23.py qwen3
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "The relationship between language and thought is",
    "A fundamental principle of mathematics states that",
    "The history of civilization shows that humans",
    "In order to understand consciousness, we must",
]


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Exp 2: 随机权重对照 (先释放训练模型, 再创建随机模型)
# ============================================================
def exp2_random_control(model_name):
    """
    策略: 先加载训练模型测传播性质 → 释放 → 再加载随机模型测传播性质
    避免同时存在两个模型导致OOM
    """
    print("\n" + "="*60)
    print("Exp 2: 随机权重对照")
    print("="*60)

    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16

    # Step 1: 训练模型
    print("\n  [Step 1] 加载训练模型...")
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print("  [Step 1] 测量训练模型传播性质...")
    trained_stats = _measure_propagation(model, tokenizer, device, n_layers, d_model, n_sents=6)

    print("  [Step 1] 释放训练模型...")
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    # Step 2: 随机模型
    print("\n  [Step 2] 创建随机模型...")
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
        # 随机化权重
        def init_weights(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        model.apply(init_weights)
        model.eval()
        print("  [Step 2] 随机模型创建成功")

        print("  [Step 2] 测量随机模型传播性质...")
        random_stats = _measure_propagation(model, tokenizer, device, n_layers, d_model, n_sents=6)

        release_model(model)
        model = None
    except Exception as e:
        print(f"  [Step 2] 随机模型失败: {e}")
        random_stats = {"error": str(e)}
        if model is not None:
            release_model(model)

    # Step 3: 对比
    print("\n  === 训练 vs 随机 对比 ===")
    comparison = {}
    for metric_key in ["perturbation_growth", "direction_cosines"]:
        t = trained_stats.get(metric_key, {})
        r = random_stats.get(metric_key, {})
        for k in t:
            if k in r and "error" not in r:
                t_val = np.mean(t[k])
                r_val = np.mean(r[k])
                comparison[k] = {
                    "trained": float(t_val),
                    "random": float(r_val),
                    "diff": float(t_val - r_val),
                }
                print(f"    {k}: trained={t_val:.4f}, random={r_val:.4f}, diff={t_val-r_val:.4f}")

    # 关键判断
    growth_diffs = [abs(v["diff"]) for k, v in comparison.items() if "growth" in k]
    cos_diffs = [abs(v["diff"]) for k, v in comparison.items() if "cosine" in k]
    
    if growth_diffs:
        avg_growth_diff = np.mean(growth_diffs)
        print(f"\n  >>> 扰动增长平均差异: {avg_growth_diff:.4f}")
        if avg_growth_diff < 0.5:
            print("  >>> 扰动增长在训练和随机模型间差异不大 → 增长模式可能是架构效应")
        else:
            print("  >>> 扰动增长有显著训练效应")

    if cos_diffs:
        avg_cos_diff = np.mean(cos_diffs)
        print(f"  >>> 方向余弦平均差异: {avg_cos_diff:.4f}")
        if avg_cos_diff < 0.1:
            print("  >>> 方向一致性在训练和随机模型间差异不大 → 方向保持可能是架构效应!")
        else:
            print("  >>> 方向一致性有训练效应")

    return {"trained": trained_stats, "random": random_stats, "comparison": comparison}


def _measure_propagation(model, tokenizer, device, n_layers, d_model, n_sents=6):
    """测量模型传播性质: 扰动增长 + 方向余弦"""
    stats = {"perturbation_growth": {}, "direction_cosines": {}}
    
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    layers = get_layers(model)
    model_device = get_input_device(model)
    
    eps = 1.0
    
    for si in range(min(n_sents, len(TEST_PROMPTS))):
        prompt = TEST_PROMPTS[si]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(model_device)
        attn_mask = inputs["attention_mask"].to(model_device)
        
        # Clean forward
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        hs_clean = [h[0, -1, :].detach().cpu().float().numpy() for h in out.hidden_states]
        del out
        torch.cuda.empty_cache()
        
        for inj_l in sample_layers:
            v = np.random.randn(d_model).astype(np.float32)
            v = v / np.linalg.norm(v)
            delta_np = (eps * v).astype(np.float32)
            
            # 用hook在inj_l注入扰动, 在后续层读取
            captured = {}
            def make_capture(layer_idx):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[layer_idx] = h[0, -1, :].detach().cpu().float().numpy()
                return hook
            
            def inject_hook(mod, inp, out):
                h = inp[0] if isinstance(inp, tuple) else inp
                # 在output上加delta
                h_out = out[0] if isinstance(out, tuple) else out
                delta_t = torch.tensor(delta_np, dtype=h_out.dtype, device=h_out.device)
                h_new = h_out.clone()
                h_new[0, -1, :] += delta_t
                return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
            
            hooks = [layers[inj_l].register_forward_hook(inject_hook)]
            for li in sample_layers:
                if li > inj_l:
                    hooks.append(layers[li].register_forward_hook(make_capture(li)))
            
            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=False)
            except:
                pass
            finally:
                for h in hooks:
                    h.remove()
            
            for li in captured:
                h_pert = captured[li]
                h_c = hs_clean[li + 1]
                delta = h_pert - h_c
                
                growth = np.linalg.norm(delta) / eps
                key_g = f"L{inj_l}_to_L{li}"
                if key_g not in stats["perturbation_growth"]:
                    stats["perturbation_growth"][key_g] = []
                stats["perturbation_growth"][key_g].append(float(growth))
                
                cos_val = np.dot(h_c, h_pert) / (np.linalg.norm(h_c) * np.linalg.norm(h_pert) + 1e-10)
                if key_g not in stats["direction_cosines"]:
                    stats["direction_cosines"][key_g] = []
                stats["direction_cosines"][key_g].append(float(cos_val))
            
            torch.cuda.empty_cache()
    
    return stats


# ============================================================
# Exp 3: Jacobian非线性交互
# ============================================================
def exp3_nonlinearity(model_name):
    """需要重新加载模型"""
    print("\n" + "="*60)
    print("Exp 3: Jacobian非线性交互")
    print("="*60)

    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16

    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    layer_pairs = [
        (0, n_layers // 4),
        (0, n_layers // 2),
        (n_layers // 4, n_layers // 2),
        (n_layers // 4, 3 * n_layers // 4),
        (n_layers // 2, 3 * n_layers // 4),
    ]

    eps = 1.0
    results = {}
    model_device = get_input_device(model)

    for si in range(min(8, len(TEST_PROMPTS))):
        prompt = TEST_PROMPTS[si]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(model_device)
        attn_mask = inputs["attention_mask"].to(model_device)

        print(f"\n  Sent {si}: '{prompt[:40]}...'")

        # Clean
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        h_clean = out.hidden_states[-1][0, -1, :].detach().cpu().float().numpy()
        del out
        torch.cuda.empty_cache()

        for l1, l2 in layer_pairs:
            key = f"sent{si}_L{l1}L{l2}"

            v1 = np.random.randn(d_model).astype(np.float32)
            v1 = v1 / np.linalg.norm(v1) * eps
            v2 = np.random.randn(d_model).astype(np.float32)
            v2 = v2 / np.linalg.norm(v2) * eps

            h_l1 = _inject_get_final(model, input_ids, attn_mask, layers, [(l1, v1)])
            h_l2 = _inject_get_final(model, input_ids, attn_mask, layers, [(l2, v2)])
            h_both = _inject_get_final(model, input_ids, attn_mask, layers, [(l1, v1), (l2, v2)])

            if h_l1 is None or h_l2 is None or h_both is None:
                results[key] = {"error": "injection failed"}
                continue

            delta_l1 = h_l1 - h_clean
            delta_l2 = h_l2 - h_clean
            delta_both = h_both - h_clean
            delta_linear = delta_l1 + delta_l2

            delta_nl = delta_both - delta_linear
            nl_norm = np.linalg.norm(delta_nl)
            linear_norm = np.linalg.norm(delta_linear)
            both_norm = np.linalg.norm(delta_both)

            if both_norm > 1e-10 and linear_norm > 1e-10:
                cos_both_linear = np.dot(delta_both, delta_linear) / (both_norm * linear_norm)
            else:
                cos_both_linear = 1.0

            rel_nl = nl_norm / max(linear_norm, 1e-10)

            results[key] = {
                "l1": l1, "l2": l2,
                "cos_both_linear": float(cos_both_linear),
                "relative_nonlinearity": float(rel_nl),
            }
            print(f"    L{l1}+L{l2}: cos={cos_both_linear:.6f}, rel_nl={rel_nl:.6f}")
            torch.cuda.empty_cache()

    release_model(model)

    # 汇总
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        avg_cos = np.mean([v["cos_both_linear"] for v in valid.values()])
        avg_nl = np.mean([v["relative_nonlinearity"] for v in valid.values()])
        print(f"\n  === 非线性交互汇总 ===")
        print(f"  Avg cos(δ_both, δ_linear) = {avg_cos:.6f}")
        print(f"  Avg ||δ_nl||/||δ_linear|| = {avg_nl:.6f}")
        if avg_cos > 0.99:
            print("  → 系统近似线性, 流形可能存在")
        elif avg_cos > 0.95:
            print("  → 弱非线性, 可能有近似流形")
        else:
            print("  → 强非线性, 流形不存在!")

    return results


def _inject_get_final(model, input_ids, attn_mask, layers, specs):
    """在指定层注入扰动, 返回最终hidden state"""
    captured = {}

    def make_inject(delta_np):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            dt = torch.tensor(delta_np, dtype=h.dtype, device=h.device)
            h_new = h.clone()
            h_new[0, -1, :] += dt
            return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
        return hook

    def capture_final(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['final'] = h[0, -1, :].detach().cpu().float().numpy()

    hooks = [layers[li].register_forward_hook(make_inject(d)) for li, d in specs]
    hooks.append(layers[-1].register_forward_hook(capture_final))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=False)
    except:
        pass
    finally:
        for h in hooks:
            h.remove()

    return captured.get('final', None)


# ============================================================
import time

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\nPhase 147b: Exp 2+3 for {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    exp2 = exp2_random_control(model_name)
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    exp3 = exp3_nonlinearity(model_name)

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    def to_ser(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, dict): return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_ser(x) for x in obj]
        return obj

    full = {
        "model": model_name,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "exp2_random_control": to_ser(exp2),
        "exp3_nonlinearity": to_ser(exp3),
    }

    out_path = Path(f"tests/glm5_temp/phase147b_{model_name}_exp23_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
