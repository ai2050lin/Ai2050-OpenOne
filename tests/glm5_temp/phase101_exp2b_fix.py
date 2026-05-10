"""
Phase 101 Exp2b: 修正版上下文化分析
====================================
核心修正: 提取目标词位置的hidden state，而非最后token
这是对批判"最后token hidden state ≠ 语义对象"的直接响应
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'glm5'))

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import gc

from model_utils import load_model, get_layers, get_model_info, release_model


def get_target_word_hiddens(model, input_ids, device, target_token_ids, n_layers):
    """提取目标词位置的hidden state"""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        all_hiddens = outputs.hidden_states  # tuple of (batch, seq_len, d_model)
    
    results = []
    batch_ids = input_ids[0].tolist()
    
    # 找到目标token的位置
    target_positions = []
    target_set = set(target_token_ids)
    for i, tid in enumerate(batch_ids):
        if tid in target_set:
            target_positions.append(i)
    
    if not target_positions:
        # 回退: 取第一个token
        target_positions = [0]
    
    # 取目标位置的hidden state (如果有多个目标token，取平均)
    for l in range(n_layers + 1):
        h = all_hiddens[l][0]  # (seq_len, d_model)
        target_h = h[target_positions].mean(dim=0)  # 平均目标token的表示
        results.append(target_h.float().cpu().numpy())
    
    return results, target_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Exp 2b: 修正版上下文化分析 — {args.model}")
    print(f"核心: 提取目标词位置(非最后token)的hidden state")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {args.model}, 层数: {n_layers}, d_model: {d_model}")
    
    # 测试词汇和上下文
    test_cases = {
        "苹果": {
            "contexts": [
                "苹果是一种水果",
                "苹果公司发布新手机",  
                "苹果手机很好用",
                "红色的苹果",
            ],
        },
        "猫": {
            "contexts": [
                "猫是一种动物",
                "猫喜欢吃鱼",
                "黑猫很可爱",
                "猫的英文是cat",
            ],
        },
        "水": {
            "contexts": [
                "水是生命之源",
                "喝水对身体好",
                "河水向东流",
                "水平面上升了",
            ],
        },
    }
    
    results = {}
    
    for word, data in test_cases.items():
        print(f"\n  === 处理词: {word} ===")
        contexts = data["contexts"]
        
        # 1. 孤立词的hidden state
        inputs = tokenizer(word, return_tensors="pt").to(device)
        word_token_ids = tokenizer.encode(word, add_special_tokens=False)
        hiddens_isolated, iso_pos = get_target_word_hiddens(
            model, inputs["input_ids"], device, word_token_ids, n_layers
        )
        print(f"    孤立词token ids: {word_token_ids}, 位置: {iso_pos}")
        
        # 2. 上下文中目标词位置的hidden state
        context_hiddens = {}
        for ctx in contexts:
            inputs = tokenizer(ctx, return_tensors="pt").to(device)
            full_ids = inputs["input_ids"][0].tolist()
            ctx_hiddens, target_pos = get_target_word_hiddens(
                model, inputs["input_ids"], device, word_token_ids, n_layers
            )
            context_hiddens[ctx] = {
                "hiddens": ctx_hiddens,
                "target_positions": target_pos,
                "full_token_ids": full_ids,
            }
            # 解码用于验证
            target_tokens = [tokenizer.decode([full_ids[p]]) for p in target_pos]
            print(f"    上下文: '{ctx[:15]}...' → 目标位置: {target_pos}, tokens: {target_tokens}")
        
        # 3. 上下文中最后token的hidden state (对比)
        context_last_hiddens = {}
        for ctx in contexts:
            inputs = tokenizer(ctx, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            last_hiddens = [outputs.hidden_states[l][0, -1, :].float().cpu().numpy() 
                          for l in range(n_layers + 1)]
            context_last_hiddens[ctx] = last_hiddens
        
        # 4. 分析
        word_results = {}
        key_layers = [0, 1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 30, 35]
        
        for l in key_layers:
            if l > n_layers:
                continue
            
            iso_h = hiddens_isolated[l]
            
            # 目标词位置分析
            target_dists_to_iso = []
            target_cross_dists = []
            last_dists_to_iso = []
            last_cross_dists = []
            
            ctx_names = list(context_hiddens.keys())
            
            for ctx in ctx_names:
                # 目标词位置 vs 孤立
                ctx_target_h = context_hiddens[ctx]["hiddens"][l]
                target_dists_to_iso.append(float(np.linalg.norm(ctx_target_h - iso_h)))
                
                # 最后token vs 孤立
                ctx_last_h = context_last_hiddens[ctx][l]
                last_dists_to_iso.append(float(np.linalg.norm(ctx_last_h - iso_h)))
            
            # 上下文间距离
            for i in range(len(ctx_names)):
                for j in range(i+1, len(ctx_names)):
                    c1, c2 = ctx_names[i], ctx_names[j]
                    # 目标词位置
                    h1_t = context_hiddens[c1]["hiddens"][l]
                    h2_t = context_hiddens[c2]["hiddens"][l]
                    target_cross_dists.append(float(np.linalg.norm(h1_t - h2_t)))
                    # 最后token
                    h1_l = context_last_hiddens[c1][l]
                    h2_l = context_last_hiddens[c2][l]
                    last_cross_dists.append(float(np.linalg.norm(h1_l - h2_l)))
            
            # 上下文化程度
            target_ctx_ratio = float(np.mean(target_dists_to_iso) / (np.mean(target_cross_dists) + 1e-8))
            last_ctx_ratio = float(np.mean(last_dists_to_iso) / (np.mean(last_cross_dists) + 1e-8))
            
            word_results[str(l)] = {
                "target_to_iso_mean": float(np.mean(target_dists_to_iso)),
                "target_cross_mean": float(np.mean(target_cross_dists)),
                "target_ctx_ratio": target_ctx_ratio,
                "last_to_iso_mean": float(np.mean(last_dists_to_iso)),
                "last_cross_mean": float(np.mean(last_cross_dists)),
                "last_ctx_ratio": last_ctx_ratio,
                # 关键指标: 上下文改变目标词表示的程度
                "target_contextualization": float(np.mean(target_cross_dists)),
            }
        
        results[word] = word_results
        
        # 打印关键层
        print(f"\n    --- 关键结果 ---")
        print(f"    {'层':>4} | {'目标词→孤立':>12} | {'目标词→跨上下文':>14} | {'目标词ctx比率':>12} | {'最后token→孤立':>14} | {'最后token→跨上下文':>16} | {'最后ctx比率':>12}")
        print(f"    {'-'*4}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*14}-+-{'-'*16}-+-{'-'*12}")
        for l in [0, 3, 6, 9, 18, 35]:
            if str(l) in word_results:
                r = word_results[str(l)]
                print(f"    L{l:>2} | {r['target_to_iso_mean']:>12.1f} | {r['target_cross_mean']:>14.1f} | {r['target_ctx_ratio']:>12.2f} | {r['last_to_iso_mean']:>14.1f} | {r['last_cross_mean']:>16.1f} | {r['last_ctx_ratio']:>12.2f}")
    
    # 保存
    save_path = f"tests/glm5_temp/phase101_exp2b_{args.model}_target_position.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {save_path}")
    
    # 跨词总结
    print(f"\n{'='*60}")
    print(f"跨词总结: 目标词位置 vs 最后token")
    print(f"{'='*60}")
    for l in [0, 3, 6, 9, 18, 35]:
        target_ratios = []
        last_ratios = []
        target_cross = []
        for word, word_data in results.items():
            if str(l) in word_data:
                target_ratios.append(word_data[str(l)]["target_ctx_ratio"])
                last_ratios.append(word_data[str(l)]["last_ctx_ratio"])
                target_cross.append(word_data[str(l)]["target_contextualization"])
        if target_ratios:
            print(f"  L{l}: 目标词ctx比率={np.mean(target_ratios):.2f}, "
                  f"最后token ctx比率={np.mean(last_ratios):.2f}, "
                  f"目标词跨上下文距离={np.mean(target_cross):.1f}")
    
    release_model(model)


if __name__ == "__main__":
    main()
