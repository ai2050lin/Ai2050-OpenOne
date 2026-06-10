"""
Phase 436: 上下文化属性方向实验
===============================
目标: 测试属性信息是否存在于上下文化的hidden states中

核心思路:
Phase 432/432b证明静态W_E/W_U属性方向无效。但属性可能不在静态embedding中，
而在上下文化的hidden state中产生。

方法:
1. 构造属性对比句对:
   "The color of the apple is red." vs "The color of the apple is green."
2. 前向传播两个句子，提取各层last token的hidden state
3. 计算上下文化属性方向: d_attr = h(red_ctx) - h(green_ctx)
4. 将这个方向注入到另一句话的对应层，看能否操控属性读出
5. 与静态W_E属性方向对比

属性测试集:
- color: red vs green (apple), brown vs white (dog), silver vs red (car)
- taste: sweet vs sour (apple), salty vs sweet (orange)
- material: metal vs wood (knife), metal vs plastic (car)
- part: seeds vs wheels (apple vs car), blade vs handle (knife)

用法:
  python tests/glm5/phase436_contextualized_attribute.py qwen3 1
  python tests/glm5/phase436_contextualized_attribute.py glm4 1
  python tests/glm5/phase436_contextualized_attribute.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import os
import numpy as np
import torch
from datetime import datetime

from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)


# ===== 属性测试配置 =====
# 每个属性有: (对象, 关系槽位, 正面值, 反面值, 测试模板)
ATTRIBUTE_TESTS = [
    # color
    {
        "obj": "apple", "relation": "color", "pos_val": "red", "neg_val": "green",
        "pos_template": "The color of the apple is red.",
        "neg_template": "The color of the apple is green.",
        "test_template": "The color of the apple is",
        "target_tokens": ["red", "green", "yellow", "orange", "blue"],
        "category": "fruit",
    },
    {
        "obj": "dog", "relation": "color", "pos_val": "brown", "neg_val": "white",
        "pos_template": "The color of the dog is brown.",
        "neg_template": "The color of the dog is white.",
        "test_template": "The color of the dog is",
        "target_tokens": ["brown", "white", "black", "golden", "gray"],
        "category": "animal",
    },
    # taste
    {
        "obj": "apple", "relation": "taste", "pos_val": "sweet", "neg_val": "sour",
        "pos_template": "The taste of the apple is sweet.",
        "neg_template": "The taste of the apple is sour.",
        "test_template": "The taste of the apple is",
        "target_tokens": ["sweet", "sour", "bitter", "delicious", "tart"],
        "category": "fruit",
    },
    # material
    {
        "obj": "knife", "relation": "material", "pos_val": "metal", "neg_val": "wood",
        "pos_template": "The material of the knife is metal.",
        "neg_template": "The material of the knife is wood.",
        "test_template": "The material of the knife is",
        "target_tokens": ["metal", "wood", "steel", "plastic", "iron"],
        "category": "tool",
    },
    # part
    {
        "obj": "car", "relation": "part", "pos_val": "wheels", "neg_val": "wings",
        "pos_template": "A car has wheels.",
        "neg_template": "A car has wings.",
        "test_template": "A car has",
        "target_tokens": ["wheels", "wings", "doors", "seats", "engine"],
        "category": "vehicle",
    },
]


def get_hidden_states(model, tokenizer, device, prompt, n_layers):
    """获取各层last token的hidden state"""
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    # hidden_states: tuple of [1, seq_len, d_model], len=n_layers+1
    last_pos = input_ids.shape[1] - 1
    hs = {}
    for li in range(n_layers + 1):
        hs[li] = out.hidden_states[li][0, last_pos].float().cpu().numpy()
    
    return hs, out.logits[0, last_pos].float().cpu().numpy()


def inject_direction_at_layer(model, tokenizer, device, prompt, direction,
                               inject_layer, inject_pos, beta=1.0, n_layers=None):
    """在指定层指定位置注入方向"""
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    
    if inject_pos is None:
        inject_pos = seq_len - 1  # 默认last token
    
    layers = get_layers(model)
    
    # Hook: 在目标层注入方向
    direction_t = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    injected_done = [False]
    
    def inject_hook(module, input, output):
        if injected_done[0]:
            return
        if isinstance(output, tuple):
            h = output[0]
            # 注入到目标位置
            h[:, inject_pos, :] += (beta * direction_t).to(h.dtype)
            injected_done[0] = True
            return (h,) + output[1:]
        else:
            output[:, inject_pos, :] += (beta * direction_t).to(output.dtype)
            injected_done[0] = True
            return output
    
    hook = layers[inject_layer].register_forward_hook(inject_hook)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    hook.remove()
    
    return out.logits[0, -1].float().cpu().numpy()


def run_experiment(model_name: str, round_num: int = 1):
    """运行Phase 436实验"""
    print(f"\n{'='*60}")
    print(f"Phase 436: 上下文化属性方向 - {model_name} R{round_num}")
    print(f"{'='*60}")
    t_start = time.time()
    
    # 加载模型
    print(f"[1] Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  class={info.model_class}, n_layers={n_layers}, d_model={info.d_model}")
    
    # 获取W_E和W_U
    print(f"[2] Loading W_E and W_U...")
    W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    W_U = get_W_U(model, model_name)
    print(f"  W_E: {W_E.shape}, W_U: {W_U.shape}")
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    print(f"  Sample layers: {sample_layers}")
    
    results = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "d_model": info.d_model,
        "sample_layers": sample_layers,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_attribute": {},
    }
    
    for ai, attr_test in enumerate(ATTRIBUTE_TESTS):
        obj = attr_test["obj"]
        relation = attr_test["relation"]
        pos_val = attr_test["pos_val"]
        neg_val = attr_test["neg_val"]
        
        print(f"\n[3] === Attribute: {obj}/{relation} ({pos_val} vs {neg_val}) ===")
        
        # Step 1: 获取上下文化属性方向
        print(f"  [3a] Getting contextualized attribute directions...")
        t0 = time.time()
        
        hs_pos, logits_pos = get_hidden_states(model, tokenizer, device,
                                                attr_test["pos_template"], n_layers)
        hs_neg, logits_neg = get_hidden_states(model, tokenizer, device,
                                                attr_test["neg_template"], n_layers)
        
        # 上下文化属性方向 (各层)
        contextual_directions = {}
        contextual_norms = {}
        for li in sample_layers:
            d = hs_pos[li] - hs_neg[li]
            contextual_norms[li] = float(np.linalg.norm(d))
            if contextual_norms[li] > 1e-10:
                contextual_directions[li] = d / contextual_norms[li]
            else:
                contextual_directions[li] = d
        
        t1 = time.time()
        print(f"    Done in {t1-t0:.1f}s")
        print(f"    Contextual direction norms: " + 
              ", ".join(f"L{li}={contextual_norms[li]:.4f}" for li in sample_layers[:5]))
        
        # Step 2: 计算静态W_E属性方向
        pos_ids = tokenizer.encode(pos_val, add_special_tokens=False)
        neg_ids = tokenizer.encode(neg_val, add_special_tokens=False)
        
        we_direction = None
        if pos_ids and neg_ids:
            we_d = W_E[pos_ids[0]] - W_E[neg_ids[0]]
            we_norm = np.linalg.norm(we_d)
            if we_norm > 0:
                we_direction = we_d / we_norm
        
        # Step 3: 计算W_U属性方向
        wu_direction = None
        if pos_ids and neg_ids:
            wu_d = W_U[pos_ids[0]] - W_U[neg_ids[0]]
            wu_norm = np.linalg.norm(wu_d)
            if wu_norm > 0:
                wu_direction = wu_d / wu_norm
        
        # Step 4: 上下文化方向 vs 静态方向 的余弦
        cos_contextual_we = {}
        cos_contextual_wu = {}
        for li in sample_layers:
            cd = contextual_directions[li]
            if we_direction is not None and contextual_norms[li] > 1e-10:
                cos_contextual_we[li] = float(np.dot(cd, we_direction))
            else:
                cos_contextual_we[li] = 0.0
            
            if wu_direction is not None and contextual_norms[li] > 1e-10:
                cos_contextual_wu[li] = float(np.dot(cd, wu_direction))
            else:
                cos_contextual_wu[li] = 0.0
        
        print(f"    cos(contextual, W_E): " + 
              ", ".join(f"L{li}={cos_contextual_we[li]:.3f}" for li in sample_layers[:5]))
        print(f"    cos(contextual, W_U): " + 
              ", ".join(f"L{li}={cos_contextual_wu[li]:.3f}" for li in sample_layers[:5]))
        
        # Step 5: 在测试模板上注入上下文化方向
        print(f"  [3b] Injecting contextualized directions...")
        
        # 先获取baseline logits
        test_prompt = attr_test["test_template"]
        toks = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)
        
        with torch.no_grad():
            out_baseline = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_baseline = out_baseline.logits[0, -1].float().cpu().numpy()
        
        # 获取目标token的baseline logits
        target_token_ids = {}
        for tok_str in attr_test["target_tokens"]:
            tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if tok_ids:
                target_token_ids[tok_str] = tok_ids[0]
        
        baseline_target_logits = {}
        for tok_str, tok_id in target_token_ids.items():
            if tok_id < len(logits_baseline):
                baseline_target_logits[tok_str] = float(logits_baseline[tok_id])
        
        print(f"    Baseline target logits: {baseline_target_logits}")
        
        # 注入实验: 在各层注入上下文化方向
        injection_results = {}
        betas = [0.5, 1.0, 2.0]  # 注入强度（倍数于方向范数）
        
        for li in sample_layers:
            if contextual_norms[li] < 1e-10:
                continue
            
            direction = contextual_directions[li]
            layer_result = {}
            
            for beta in betas:
                # 正方向注入（应该增强pos_val）
                try:
                    logits_pos_inj = inject_direction_at_layer(
                        model, tokenizer, device, test_prompt,
                        direction * contextual_norms[li],  # 用原始范数缩放
                        inject_layer=li, inject_pos=None,  # last token
                        beta=beta, n_layers=n_layers
                    )
                    
                    # 负方向注入（应该增强neg_val）
                    logits_neg_inj = inject_direction_at_layer(
                        model, tokenizer, device, test_prompt,
                        -direction * contextual_norms[li],
                        inject_layer=li, inject_pos=None,
                        beta=beta, n_layers=n_layers
                    )
                    
                    # 计算目标token的logit变化
                    pos_inj_target = {}
                    neg_inj_target = {}
                    for tok_str, tok_id in target_token_ids.items():
                        if tok_id < len(logits_pos_inj):
                            pos_inj_target[tok_str] = {
                                "logit": float(logits_pos_inj[tok_id]),
                                "delta": float(logits_pos_inj[tok_id] - logits_baseline[tok_id])
                            }
                        if tok_id < len(logits_neg_inj):
                            neg_inj_target[tok_str] = {
                                "logit": float(logits_neg_inj[tok_id]),
                                "delta": float(logits_neg_inj[tok_id] - logits_baseline[tok_id])
                            }
                    
                    # 计算pos_val和neg_val的logit差
                    pos_val_logit_delta = 0
                    neg_val_logit_delta = 0
                    if pos_val in pos_inj_target:
                        pos_val_logit_delta = pos_inj_target[pos_val]["delta"]
                    if neg_val in neg_inj_target:
                        neg_val_logit_delta = neg_inj_target[neg_val]["delta"]
                    
                    # 也计算负方向注入的效果
                    pos_val_logit_delta_neg = 0
                    neg_val_logit_delta_neg = 0
                    if pos_val in neg_inj_target:
                        pos_val_logit_delta_neg = neg_inj_target[pos_val]["delta"]
                    if neg_val in neg_inj_target:
                        neg_val_logit_delta_neg = neg_inj_target[neg_val]["delta"]
                    
                    layer_result[beta] = {
                        "pos_injection": {
                            "pos_val_delta": round(pos_val_logit_delta, 4),
                            "neg_val_delta": round(neg_val_logit_delta, 4),
                            "switch_score": round(pos_val_logit_delta - neg_val_logit_delta, 4),
                            "all_targets": pos_inj_target,
                        },
                        "neg_injection": {
                            "pos_val_delta": round(pos_val_logit_delta_neg, 4),
                            "neg_val_delta": round(neg_val_logit_delta_neg, 4),
                            "switch_score": round(neg_val_logit_delta_neg - pos_val_logit_delta_neg, 4),
                        },
                    }
                    
                except Exception as e:
                    layer_result[beta] = {"error": str(e)}
            
            injection_results[li] = layer_result
            
            # 打印摘要
            if 1.0 in layer_result and "error" not in layer_result[1.0]:
                r = layer_result[1.0]
                pos_sw = r["pos_injection"]["switch_score"]
                neg_sw = r["neg_injection"]["switch_score"]
                print(f"    L{li} (beta=1.0): pos_inject switch={pos_sw:.4f}, "
                      f"neg_inject switch={neg_sw:.4f}")
        
        # Step 6: 对比 - 静态W_E属性方向注入（在相同层）
        print(f"  [3c] Comparing with static W_E attribute direction...")
        we_injection_results = {}
        if we_direction is not None:
            # 选择中层和深层各一层
            key_layers = [sample_layers[len(sample_layers)//3], 
                         sample_layers[2*len(sample_layers)//3]]
            key_layers = [li for li in key_layers if li < n_layers]
            
            for li in key_layers:
                try:
                    logits_we_inj = inject_direction_at_layer(
                        model, tokenizer, device, test_prompt,
                        we_direction,
                        inject_layer=li, inject_pos=None,
                        beta=1.0, n_layers=n_layers
                    )
                    
                    we_target_deltas = {}
                    for tok_str, tok_id in target_token_ids.items():
                        if tok_id < len(logits_we_inj):
                            we_target_deltas[tok_str] = round(float(logits_we_inj[tok_id] - logits_baseline[tok_id]), 4)
                    
                    we_injection_results[li] = we_target_deltas
                    print(f"    W_E L{li}: {we_target_deltas}")
                    
                except Exception as e:
                    we_injection_results[li] = {"error": str(e)}
        
        results["per_attribute"][f"{obj}_{relation}"] = {
            "pos_val": pos_val,
            "neg_val": neg_val,
            "pos_template": attr_test["pos_template"],
            "neg_template": attr_test["neg_template"],
            "test_template": test_prompt,
            "contextual_norms": {str(li): round(v, 6) for li, v in contextual_norms.items()},
            "cos_contextual_we": {str(li): round(v, 4) for li, v in cos_contextual_we.items()},
            "cos_contextual_wu": {str(li): round(v, 4) for li, v in cos_contextual_wu.items()},
            "baseline_target_logits": baseline_target_logits,
            "contextual_injection": {str(li): v for li, v in injection_results.items()},
            "we_injection": we_injection_results,
        }
        
        # 清理
        torch.cuda.empty_cache()
    
    # 保存结果
    os.makedirs("results/phase436_contextualized_attribute", exist_ok=True)
    out_path = f"results/phase436_contextualized_attribute/{model_name}_phase436_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[4] Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    
    t_total = time.time() - t_start
    print(f"[5] Total time: {t_total/60:.1f}min")
    
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    run_experiment(model_name, round_num)
