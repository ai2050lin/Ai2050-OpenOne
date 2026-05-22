"""
Subspace Topology Phase 1f: 语义子空间解码
=============================================

核心问题: Qwen3中间层的2维语义空间编码了什么？
方法: 
1. 收集对比句对的残差流
2. 去除Rank-1偏置方向
3. 在去偏置空间中找到对比方向
4. 验证这些方向是否与SVD的主方向对齐

Run:
  python tests/glm5/subspace_topology_phase1f_semantic_decode.py --model qwen3
  python tests/glm5/subspace_topology_phase1f_semantic_decode.py --model glm4
  python tests/glm5/subspace_topology_phase1f_semantic_decode.py --model deepseek7b
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
from pathlib import Path

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 对比句对: (base, variant, feature_name)
CONTRAST_PAIRS = [
    # 否定
    ("The apple is red.", "The apple is NOT red.", "negation"),
    ("The sky is blue.", "The sky is NOT blue.", "negation"),
    ("Ice is cold.", "Ice is NOT cold.", "negation"),
    ("Birds can fly.", "Birds cannot fly.", "negation"),
    ("Water is wet.", "Water is not wet.", "negation"),
    
    # 疑问
    ("The apple is red.", "Is the apple red?", "question"),
    ("The sky is blue.", "Is the sky blue?", "question"),
    ("Ice is cold.", "Is ice cold?", "question"),
    
    # 翻译
    ("The apple is red.", "Translate to French: The apple is red.", "translation"),
    ("Dogs are loyal.", "Translate to Chinese: Dogs are loyal.", "translation"),
    
    # 推理
    ("Cats are animals.", "If all cats are animals, and Whiskers is a cat, then", "reasoning"),
    ("A equals B.", "If A=B and B=C, then A=", "reasoning"),
    
    # 代码
    ("Sort this list.", "Write Python code to sort a list:", "code"),
    ("Reverse the string.", "Write a function to reverse a string:", "code"),
    
    # 算术
    ("What is five plus two?", "What is 5 + 2?", "arithmetic"),
    ("What is three times four?", "What is 3 × 4?", "arithmetic"),
    
    # 情感/风格
    ("Explain physics.", "In the style of Shakespeare, explain physics.", "style"),
    ("Describe the treasure.", "In the style of a pirate, describe the treasure.", "style"),
    
    # 中文
    ("The apple is red.", "苹果是红色的。", "language_zh"),
    ("Dogs are loyal.", "狗是忠诚的。", "language_zh"),
    
    # 双重否定
    ("The apple is red.", "It is not true that the apple is not red.", "double_neg"),
    ("The sky is blue.", "It is not true that the sky is not blue.", "double_neg"),
    
    # 反常识
    ("The earth revolves around the sun.", "The sun revolves around the earth.", "anti_fact"),
    ("Water freezes at 0 degrees.", "Water freezes at 100 degrees.", "anti_fact"),
    
    # 颜色属性
    ("The apple is red.", "The apple is green.", "color"),
    ("The sky is blue.", "The sky is red.", "color"),
    
    # 主语替换
    ("John gave Mary a book.", "Mary gave John a book.", "subject_swap"),
    ("Alice told Bob a secret.", "Bob told Alice a secret.", "subject_swap"),
]

# 额外的多样性句子用于SVD基
DIVERSE_SENTENCES = [
    "The apple is red.", "Paris is the capital of France.", 
    "Water boils at 100 degrees.", "Justice is a fundamental concept.",
    "Freedom means different things to different people.",
    "Time passes differently when you are happy.",
    "The earth revolves around the sun.",
    "Books are made of paper.", "Birds can fly in the sky.",
    "饕餮是一种传说中的神兽。",
]


def compute_participation_ratio(eigenvalues):
    lam = np.array(eigenvalues, dtype=np.float64)
    lam = lam[lam > 1e-12]
    if len(lam) == 0:
        return 0.0
    return float((np.sum(lam))**2 / np.sum(lam**2))


def robust_svd(matrix, k=None):
    """鲁棒SVD"""
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vt
    except np.linalg.LinAlgError:
        from sklearn.decomposition import TruncatedSVD
        k = k or min(100, matrix.shape[1] - 1, matrix.shape[0] - 1)
        svd_obj = TruncatedSVD(n_components=k, random_state=42)
        svd_obj.fit(matrix.astype(np.float32))
        return None, svd_obj.singular_values_, svd_obj.components_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    W_U = get_W_U(model, model_info.name)
    
    print(f"\n模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # ========================================
    # 步骤1: 收集多样性句子的残差流，建立SVD基
    # ========================================
    all_sentences = list(set(
        [s for pair in CONTRAST_PAIRS for s in pair[:2]] + DIVERSE_SENTENCES
    ))
    print(f"\n收集 {len(all_sentences)} 个句子的残差流，建立去偏置SVD基...")
    
    layer_acts = {li: [] for li in range(n_layers)}
    
    for si, sentence in enumerate(all_sentences):
        toks = tokenizer(sentence, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        captured = {}
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float()
                    else:
                        captured[key] = output.detach().float()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception:
                pass
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                acts = captured[key][0, -1, :].cpu().numpy()  # 只取最后一个token
                layer_acts[li].append(acts)
        
        del captured
        gc.collect()
    
    # ========================================
    # 步骤2: 为每层建立去偏置SVD基
    # ========================================
    print("\n为每层建立去偏置SVD基...")
    
    svd_bases = {}
    target_layers = sorted(set(
        [0, 1, 5, 6] + 
        list(range(0, n_layers, max(1, n_layers//8))) + 
        [n_layers-3, n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    for li in target_layers:
        acts = np.array(layer_acts[li])  # [n_sentences, d_model]
        mean = acts.mean(axis=0)
        centered = acts - mean
        
        U, S, Vt = robust_svd(centered)
        if U is not None:
            # 去偏置: 只保留第2个及之后的PC
            debiased = centered - U[:, :1] @ np.diag(S[:1]) @ Vt[:1, :]
        else:
            proj = centered @ Vt[:1, :].T
            debiased = centered - proj @ Vt[:1, :]
        
        U2, S2, Vt2 = robust_svd(debiased)
        svd_bases[li] = {
            "mean": mean,
            "Vt_full": Vt,
            "S_full": S,
            "U_full": U,
            "Vt_debiased": Vt2,
            "S_debiased": S2,
        }
        
        id_total = compute_participation_ratio(S**2 / (len(all_sentences) - 1))
        id_debiased = compute_participation_ratio(S2**2 / (len(all_sentences) - 1)) if S2 is not None else 0
        print(f"  L{li}: total_ID={id_total:.2f}, debiased_ID={id_debiased:.2f}, "
              f"top5 debiased S ratio: {[f'{S2[i]/S2[0]:.3f}' for i in range(min(5, len(S2)))]}")
    
    # ========================================
    # 步骤3: 对比句对分析 — 语义方向在去偏置空间中的投影
    # ========================================
    print(f"\n{'='*80}")
    print("对比句对分析: 语义方向在去偏置空间中的位置")
    print(f"{'='*80}")
    
    results = {}
    
    for li in target_layers:
        base = svd_bases[li]
        Vt_d = base["Vt_debiased"]
        S_d = base["S_debiased"]
        mean = base["mean"]
        
        if Vt_d is None or len(S_d) < 3:
            continue
        
        layer_results = []
        
        for base_sent, variant_sent, feature in CONTRAST_PAIRS:
            # 获取两个句子的残差流
            pair_acts = {}
            for label, sent in [("base", base_sent), ("variant", variant_sent)]:
                toks = tokenizer(sent, return_tensors="pt").to(device)
                seq_len = toks.input_ids.shape[1]
                embed_layer = model.get_input_embeddings()
                inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                
                captured = {}
                hooks = []
                for lidx in range(n_layers):
                    layer = layers[lidx]
                    def make_hook(key):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                captured[key] = output[0].detach().float()
                            else:
                                captured[key] = output.detach().float()
                        return hook
                    hooks.append(layer.register_forward_hook(make_hook(f"L{lidx}")))
                
                with torch.no_grad():
                    try:
                        _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
                    except Exception:
                        pass
                
                for h in hooks:
                    h.remove()
                
                key = f"L{li}"
                if key in captured:
                    pair_acts[label] = captured[key][0, -1, :].cpu().numpy()
                
                del captured
            
            if "base" not in pair_acts or "variant" not in pair_acts:
                continue
            
            # 计算对比方向
            delta = pair_acts["variant"] - pair_acts["base"]  # [d_model]
            delta_centered = delta - np.dot(delta, mean) * mean / np.dot(mean, mean) if np.dot(mean, mean) > 0 else delta
            
            # 去偏置: 减去Rank-1方向
            v1_full = base["Vt_full"][0]
            delta_debiased = delta_centered - np.dot(delta_centered, v1_full) * v1_full
            
            # 投影到去偏置SVD的前10个PC上
            n_proj = min(10, Vt_d.shape[0])
            proj_coeffs = np.array([np.dot(delta_debiased, Vt_d[i]) for i in range(n_proj)])
            
            # 解码每个PC方向
            pc_decoded = []
            for i in range(min(5, n_proj)):
                v = Vt_d[i]
                logits_v = W_U @ v
                top3_ids = np.argsort(logits_v)[-3:][::-1]
                top3_tokens = [tokenizer.decode([int(tid)]) for tid in top3_ids]
                pc_decoded.append({
                    "pc": i,
                    "top3_tokens": top3_tokens,
                    "var_ratio": float(S_d[i]**2 / np.sum(S_d**2)) if np.sum(S_d**2) > 0 else 0,
                })
            
            # 解码对比方向本身
            logits_delta = W_U @ (delta_debiased / (np.linalg.norm(delta_debiased) + 1e-10))
            top5_ids = np.argsort(logits_delta)[-5:][::-1]
            top5_tokens = [tokenizer.decode([int(tid)]) for tid in top5_ids]
            
            # 对比方向与各PC的对齐度
            alignment = [float(np.abs(np.dot(delta_debiased, Vt_d[i])) / (np.linalg.norm(delta_debiased) + 1e-10)) 
                        for i in range(n_proj)]
            
            result = {
                "feature": feature,
                "base": base_sent[:30],
                "variant": variant_sent[:30],
                "delta_decoded_top5": top5_tokens,
                "delta_alignment_with_PCs": alignment[:5],
                "PC_decoded": pc_decoded,
                "delta_norm": float(np.linalg.norm(delta)),
                "delta_debiased_norm": float(np.linalg.norm(delta_debiased)),
            }
            layer_results.append(result)
        
        results[f"L{li}"] = layer_results
        
        # 打印关键结果
        print(f"\n  === L{li} ===")
        for r in layer_results:
            al = r["delta_alignment_with_PCs"]
            dominant_pc = np.argmax(al) if max(al) > 0.1 else -1
            print(f"    {r['feature']:15s}: delta_norm={r['delta_debiased_norm']:.2f}, "
                  f"对齐PC{dominant_pc}({al[dominant_pc]:.3f})" if dominant_pc >= 0 else
                  f"    {r['feature']:15s}: delta_norm={r['delta_debiased_norm']:.2f}, 无强对齐",
                  end="")
            print(f", 解码→{r['delta_decoded_top5'][:3]}")
    
    # ========================================
    # 步骤4: 按特征类型聚合分析
    # ========================================
    print(f"\n{'='*80}")
    print("特征类型聚合: 各语义特征主要对齐哪个PC？")
    print(f"{'='*80}")
    
    for li in target_layers:
        layer_res = results.get(f"L{li}", [])
        if not layer_res:
            continue
        
        # 按特征分组
        feature_groups = {}
        for r in layer_res:
            feat = r["feature"]
            if feat not in feature_groups:
                feature_groups[feat] = []
            feature_groups[feat].append(r)
        
        print(f"\n  === L{li} ===")
        for feat, group in sorted(feature_groups.items()):
            avg_align = np.mean([r["delta_alignment_with_PCs"] for r in group], axis=0)
            dominant_pc = np.argmax(avg_align)
            avg_norm = np.mean([r["delta_debiased_norm"] for r in group])
            print(f"    {feat:15s}: avg_norm={avg_norm:.2f}, 主要对齐PC{dominant_pc}(avg_align={avg_align[dominant_pc]:.3f})")
    
    # 保存
    # 转换numpy为list以便json序列化
    for li in results:
        for r in results[li]:
            r["delta_alignment_with_PCs"] = [float(x) for x in r["delta_alignment_with_PCs"]]
    
    out_path = OUTPUT_DIR / f"exp1f_semantic_decode_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
