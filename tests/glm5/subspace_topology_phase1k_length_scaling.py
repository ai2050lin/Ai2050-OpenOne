"""
Subspace Topology Phase 1k: 序列长度 vs 总语义维度
====================================================

核心问题: 总语义维度(60-95)是否随句子长度增长？还是趋于饱和？

如果趋于饱和 → 语言是有限维流形，60-95维可能是内在维度
如果线性增长 → 维度是位置数的函数，语言没有内在上限

方法: 用不同长度的句子组分别计算总语义维度

Run:
  python tests/glm5/subspace_topology_phase1k_length_scaling.py --model qwen3
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import gc
import json
from pathlib import Path

from model_utils import load_model, get_layers, get_model_info, release_model

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 不同长度的句子组
# 短句(~4 tokens), 中句(~7 tokens), 长句(~12 tokens), 超长句(~18 tokens)
SENTENCE_GROUPS = {
    "short": [
        "Cats are animals.",
        "Ice is cold.",
        "Red is a color.",
        "Two plus two.",
        "Water flows down.",
        "Birds can fly.",
        "Fire is hot.",
        "Apples are fruit.",
        "Time passes on.",
        "Grass is green.",
        "Dogs bark loud.",
        "Snow is white.",
        "Rain falls down.",
        "Iron is heavy.",
        "Light travels fast.",
        "Salt tastes salty.",
    ],
    "medium": [
        "The apple is red and sweet.",
        "Paris is the capital of France.",
        "Water boils at 100 degrees.",
        "The sky is blue and clear.",
        "Justice is a fundamental concept.",
        "John gave Mary a book today.",
        "Ice melts when heated slowly.",
        "Books are made of paper too.",
        "Birds can fly in the sky.",
        "Freedom means different things.",
        "Time passes differently always.",
        "The grass is green and soft.",
        "Dogs are loyal and friendly.",
        "The snow is white and cold.",
        "Fire burns everything it touches.",
        "Iron is heavier than wood.",
    ],
    "long": [
        "The apple on the table is red and sweet to eat.",
        "Paris has been the capital of France for many centuries.",
        "Water always boils at exactly 100 degrees Celsius at sea level.",
        "The clear blue sky stretched endlessly across the horizon today.",
        "Justice is a fundamental concept that shapes our legal system.",
        "John carefully gave Mary a beautiful book as a present today.",
        "Ice begins to melt when heated slowly over a warm flame.",
        "Most books are traditionally made of paper and ink materials.",
        "Many birds can fly gracefully in the open sky above us.",
        "Freedom means different things to different people around the world.",
        "The soft green grass covered the entire hillside this spring.",
        "Dogs are known to be loyal and friendly companions always.",
        "The bright white snow blanketed the mountains overnight completely.",
        "Fire can quickly burn everything it touches if not controlled.",
        "Pure iron is significantly heavier than an equal volume of wood.",
        "The quick brown fox jumped over the lazy sleeping dog.",
    ],
    "very_long": [
        "The beautiful red apple sitting on the wooden kitchen table is sweet and delicious to eat for breakfast.",
        "Paris has been the proud capital of France for many centuries and attracts millions of tourists each year.",
        "Pure water always boils at exactly 100 degrees Celsius when measured at standard sea level atmospheric pressure.",
        "The remarkably clear blue sky stretched endlessly across the vast horizon making it a perfect day for hiking.",
        "Justice is a fundamental philosophical concept that has shaped our entire legal system throughout human history.",
        "John carefully wrapped and gave Mary a beautiful leather-bound book as a birthday present this afternoon.",
        "Solid ice begins to slowly melt when heated gradually over a warm candle flame in the kitchen.",
    ],
}


def compute_participation_ratio(eigenvalues):
    lam = np.array(eigenvalues, dtype=np.float64)
    lam = lam[lam > 1e-12]
    if len(lam) == 0:
        return 0.0
    return float((np.sum(lam))**2 / np.sum(lam**2))


def robust_svd(matrix):
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vt
    except np.linalg.LinAlgError:
        from sklearn.decomposition import TruncatedSVD
        k = min(200, matrix.shape[1] - 1, matrix.shape[0] - 1)
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
    
    print(f"\n模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # 选3个关键层
    target_layers = [1, n_layers // 2, n_layers - 2]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    results = {}
    
    for group_name, sentences in SENTENCE_GROUPS.items():
        print(f"\n{'='*70}")
        print(f"  句子组: {group_name} ({len(sentences)} 句)")
        print(f"{'='*70}")
        
        # tokenize
        max_len = 0
        tokenized = []
        for sent in sentences:
            toks = tokenizer(sent, return_tensors="pt")
            seq_len = toks.input_ids.shape[1]
            tokenized.append((sent, toks, seq_len))
            max_len = max(max_len, seq_len)
        
        print(f"  最大序列长度: {max_len}")
        
        # 收集激活
        pos_layer_acts = {pos: {li: [] for li in range(n_layers)} for pos in range(max_len)}
        
        for si, (sent, toks, seq_len) in enumerate(tokenized):
            toks = toks.to(device)
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
                    acts = captured[key][0, :, :].cpu().numpy()
                    for pos in range(seq_len):
                        pos_layer_acts[pos][li].append(acts[pos, :])
            
            del captured
            gc.collect()
        
        # 计算各层的总语义维度
        group_result = {"group": group_name, "n_sentences": len(sentences), "max_seq_len": max_len}
        
        for li in target_layers:
            # 有效位置(≥5个样本)
            valid_positions = [p for p in range(max_len) 
                              if len(pos_layer_acts[p][li]) >= 5]
            
            if len(valid_positions) < 2:
                continue
            
            # 方法3: 逐位置去偏置后合并
            per_position_debiased = []
            per_pos_ids = {}
            for pos in valid_positions:
                acts = np.array(pos_layer_acts[pos][li])
                if len(acts) < 3:
                    continue
                mean_p = acts.mean(axis=0)
                centered_p = acts - mean_p
                _, S_p, Vt_p = robust_svd(centered_p)
                
                # 单位置total ID
                id_p = compute_participation_ratio(S_p**2 / (len(acts) - 1)) if S_p is not None else 0
                
                # 去偏置(去PC1)
                if S_p is not None and len(S_p) > 1:
                    try:
                        U_p, S_p2, Vt_p2 = np.linalg.svd(centered_p, full_matrices=False)
                        debiased_p = centered_p - U_p[:, :1] @ np.diag(S_p2[:1]) @ Vt_p2[:1, :]
                    except:
                        proj = centered_p @ Vt_p[:1, :].T
                        debiased_p = centered_p - proj @ Vt_p[:1, :]
                    per_position_debiased.append(debiased_p)
                else:
                    per_position_debiased.append(centered_p)
                
                per_pos_ids[pos] = id_p
            
            if not per_position_debiased:
                continue
            
            all_debiased = np.vstack(per_position_debiased)
            _, S_all, _ = robust_svd(all_debiased)
            total_id = compute_participation_ratio(S_all**2 / (len(all_debiased) - 1)) if S_all is not None else 0
            
            avg_per_pos_id = np.mean(list(per_pos_ids.values()))
            n_pos = len(valid_positions)
            
            print(f"  L{li:2d}: n_pos={n_pos:2d}, avg_per_pos_ID={avg_per_pos_id:.1f}, "
                  f"total_ID={total_id:.1f}, ratio={total_id/(avg_per_pos_id*n_pos) if avg_per_pos_id*n_pos>0 else 0:.2f}")
            
            group_result[f"L{li}"] = {
                "n_positions": n_pos,
                "avg_per_pos_ID": avg_per_pos_id,
                "total_semantic_ID": total_id,
                "predicted_independent": avg_per_pos_id * n_pos,
            }
        
        results[group_name] = group_result
    
    # ========================================
    # 核心总结
    # ========================================
    print(f"\n{'='*80}")
    print("核心总结: 序列长度 vs 总语义维度")
    print(f"{'='*80}")
    
    for li in target_layers:
        print(f"\n  Layer {li}:")
        for group_name in ["short", "medium", "long", "very_long"]:
            r = results.get(group_name, {}).get(f"L{li}", {})
            total = r.get("total_semantic_ID", "?")
            n_pos = r.get("n_positions", "?")
            avg = r.get("avg_per_pos_ID", "?")
            print(f"    {group_name:10s}: n_pos={n_pos}, avg_per_pos={avg}, total={total}")
    
    # 保存
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    out_path = OUTPUT_DIR / f"exp1k_length_scaling_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
