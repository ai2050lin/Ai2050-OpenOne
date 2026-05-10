"""
Phase 79-Q: Computation Invariants on Qwen3
=============================================

在有能力模型上验证computation invariant假说

核心发现需要在Qwen3上验证:
  1. Routing topology: 同类任务routing >> 跨类任务routing?
  2. Boundary-conditioned computation: prefix是否改变routing policy?
  3. Transition direction: 同类任务transition direction对齐?

Qwen3有真正的推理和翻译能力, 如果computation policy invariant存在,
在有能力的模型中应该更加明显.

Usage:
  python ccml_phase79_qwen3_invariant.py --exp a
  python ccml_phase79_qwen3_invariant.py --exp c
  python ccml_phase79_qwen3_invariant.py --exp all
"""

import torch
import numpy as np
import argparse
from collections import defaultdict

def get_qwen3_model():
    """加载Qwen3-4B模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen3-4B"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer

def get_attention_patterns_qwen3(model, tokenizer, text, max_new_tokens=0):
    """获取Qwen3所有层的attention pattern"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
    
    # outputs.attentions 是一个tuple of tuples
    # 对于generate, 每步的attentions: tuple of layers, 每层 [batch, n_heads, seq_q, seq_k]
    # 我们只取第一步(prehfill)的attentions
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        first_step_attns = outputs.attentions[0]  # tuple of layers
        patterns = {}
        for layer_idx, attn in enumerate(first_step_attns):
            patterns[layer_idx] = attn[0].detach().cpu()  # [n_heads, seq_q, seq_k]
        return patterns
    
    return None

def get_attention_patterns_qwen3_forward(model, tokenizer, text):
    """直接forward获取attention patterns (更高效)"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    patterns = {}
    for layer_idx, attn in enumerate(outputs.attentions):
        patterns[layer_idx] = attn[0].detach().cpu()  # [n_heads, seq_q, seq_k]
    
    hidden_states = {}
    for layer_idx, hs in enumerate(outputs.hidden_states):
        hidden_states[layer_idx] = hs[0].detach().cpu()  # [seq_len, d_model]
    
    return patterns, hidden_states

def compute_topk_routing_overlap(pat1, pat2, k=5):
    """Top-k routing overlap比较"""
    n_heads = pat1.shape[0]
    min_len = min(pat1.shape[2], pat2.shape[2])
    
    overlaps = []
    for h in range(n_heads):
        head_overlap = 0
        count = 0
        for pos in range(min_len):
            topk1 = torch.topk(pat1[h, pos, :min_len], min(k, min_len)).indices
            topk2 = torch.topk(pat2[h, pos, :min_len], min(k, min_len)).indices
            set1 = set(topk1.tolist())
            set2 = set(topk2.tolist())
            if len(set1 | set2) > 0:
                head_overlap += len(set1 & set2) / len(set1 | set2)
            count += 1
        overlaps.append(head_overlap / max(count, 1))
    return overlaps

def compute_rank_correlation(pat1, pat2):
    """Rank correlation of attention distributions"""
    n_heads = pat1.shape[0]
    n_positions = min(pat1.shape[1], pat2.shape[1])  # query positions
    min_kv_len = min(pat1.shape[2], pat2.shape[2])    # key positions
    
    correlations = []
    for h in range(n_heads):
        head_corr = 0
        count = 0
        for pos in range(n_positions):
            row1 = pat1[h, pos, :min_kv_len].float()
            row2 = pat2[h, pos, :min_kv_len].float()
            rank1 = row1.argsort().argsort().float()
            rank2 = row2.argsort().argsort().float()
            r1c = rank1 - rank1.mean()
            r2c = rank2 - rank2.mean()
            denom = r1c.norm() * r2c.norm()
            if denom > 1e-8:
                corr = (r1c @ r2c) / denom
                head_corr += corr.item()
                count += 1
        correlations.append(head_corr / max(count, 1))
    return correlations

# ============================================================
# 实验A-Q: Routing Topology Invariant on Qwen3
# ============================================================

def exp_a_qwen3_routing(model, tokenizer):
    """在Qwen3上验证routing topology invariant"""
    print("=" * 70)
    print("实验A-Q: Routing Topology Invariant on Qwen3")
    print("=" * 70)
    
    n_layers = model.config.num_hidden_layers
    
    # ---- 加法任务 (Qwen3真的能做加法) ----
    add_tasks = {}
    for a, b in [(2,3),(7,4),(15,23),(9,6),(11,8),(3,5),(12,7),(4,9),(21,14),(6,2),
                 (8,1),(5,5),(10,3),(13,6),(1,7),(2,8),(4,3),(6,4),(9,2),(3,7)]:
        add_tasks[f"add_{a}_{b}"] = f"{a} + {b} ="
    
    # ---- 翻译任务 (Qwen3真的能翻译) ----
    trans_tasks = {}
    sentences = [
        "The cat is on the mat", "The dog runs in the park", "The bird sings a song",
        "The sun shines bright", "The water flows down", "The child plays outside",
        "The tree grows tall", "The rain falls softly", "The wind blows hard",
        "The moon rises slowly", "The fish swims deep", "The flower blooms red",
        "The snow falls white", "The fire burns hot", "The earth spins round",
        "The river runs wide", "The mountain stands high", "The cloud floats free",
        "The star shines far", "The ocean waves crash"
    ]
    for i, s in enumerate(sentences):
        trans_tasks[f"trans_{i}"] = f"Translate to French: {s}"
    
    # ---- 反义词任务 ----
    antonym_tasks = {}
    for w in ["hot","big","fast","happy","light","strong","loud","rough","wide","tall",
              "cold","small","slow","sad","dark","weak","quiet","smooth","narrow","short"]:
        antonym_tasks[f"ant_{w}"] = f"The opposite of {w} is"
    
    # ---- 数学推理任务 (Qwen3的强项) ----
    math_reason_tasks = {}
    for i, q in enumerate([
        "If x + 3 = 7, what is x?",
        "If x + 5 = 12, what is x?",
        "If x + 2 = 9, what is x?",
        "If x + 8 = 15, what is x?",
        "If x + 4 = 11, what is x?",
        "If 2x = 10, what is x?",
        "If 3x = 15, what is x?",
        "If 2x = 8, what is x?",
        "If 5x = 25, what is x?",
        "If 4x = 20, what is x?",
    ]):
        math_reason_tasks[f"math_{i}"] = q
    
    task_groups = {
        "addition": add_tasks,
        "translate_fr": trans_tasks,
        "antonym": antonym_tasks,
        "math_reasoning": math_reason_tasks,
    }
    
    # 采样 (太多会爆GPU内存)
    sample_size = 10
    
    all_patterns = {}
    for group_name, tasks in task_groups.items():
        print(f"\n  Processing {group_name} group ({min(len(tasks), sample_size)} samples)...")
        all_patterns[group_name] = {}
        task_items = list(tasks.items())[:sample_size]
        
        for task_name, text in task_items:
            patterns, _ = get_attention_patterns_qwen3_forward(model, tokenizer, text)
            all_patterns[group_name][task_name] = patterns
            # 清理GPU缓存
            torch.cuda.empty_cache()
    
    # ---- 分析1: 同组内routing相似度 (选几个代表层) ----
    print("\n" + "=" * 50)
    print("分析1: 同组内 vs 跨组 routing topology")
    print("=" * 50)
    
    target_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    for layer in target_layers:
        print(f"\n  Layer {layer}:")
        
        # 同组内相似度
        within_sims = {}
        for group_name, tasks in task_groups.items():
            task_names = list(tasks.keys())[:sample_size]
            sims = []
            for i in range(min(5, len(task_names))):
                for j in range(i+1, min(5, len(task_names))):
                    pat1 = all_patterns[group_name][task_names[i]][layer]
                    pat2 = all_patterns[group_name][task_names[j]][layer]
                    
                    if pat1 is None or pat2 is None:
                        continue
                    
                    rank_corrs = compute_rank_correlation(pat1, pat2)
                    sims.append(np.mean(rank_corrs))
            
            within_sims[group_name] = np.mean(sims) if sims else 0
            print(f"    {group_name} within: {within_sims[group_name]:.4f}")
        
        # 跨组相似度
        group_names = list(task_groups.keys())
        cross_sims = []
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i >= j:
                    continue
                t1 = list(task_groups[g1].keys())[0]
                t2 = list(task_groups[g2].keys())[0]
                pat1 = all_patterns[g1][t1][layer]
                pat2 = all_patterns[g2][t2][layer]
                
                if pat1 is None or pat2 is None:
                    continue
                
                rank_corrs = compute_rank_correlation(pat1, pat2)
                cross_sims.append(np.mean(rank_corrs))
        
        cross_avg = np.mean(cross_sims) if cross_sims else 0
        within_avg = np.mean(list(within_sims.values()))
        
        print(f"    ★ 组内均值: {within_avg:.4f}, 跨组均值: {cross_avg:.4f}, 差距: {within_avg - cross_avg:.4f}")
    
    # ---- 分析2: Top-k routing overlap (更本质) ----
    print("\n" + "=" * 50)
    print("分析2: Top-3 Routing Overlap (routing决策不变性)")
    print("=" * 50)
    
    for layer in target_layers:
        print(f"\n  Layer {layer}:")
        
        within_overlaps = {}
        for group_name, tasks in task_groups.items():
            task_names = list(tasks.keys())[:sample_size]
            overlaps = []
            for i in range(min(5, len(task_names))):
                for j in range(i+1, min(5, len(task_names))):
                    pat1 = all_patterns[group_name][task_names[i]][layer]
                    pat2 = all_patterns[group_name][task_names[j]][layer]
                    
                    if pat1 is None or pat2 is None:
                        continue
                    
                    ov = compute_topk_routing_overlap(pat1, pat2, k=3)
                    overlaps.append(np.mean(ov))
            
            within_overlaps[group_name] = np.mean(overlaps) if overlaps else 0
            print(f"    {group_name} within: {within_overlaps[group_name]:.4f}")
        
        group_names = list(task_groups.keys())
        cross_overlaps = []
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i >= j:
                    continue
                t1 = list(task_groups[g1].keys())[0]
                t2 = list(task_groups[g2].keys())[0]
                pat1 = all_patterns[g1][t1][layer]
                pat2 = all_patterns[g2][t2][layer]
                
                if pat1 is None or pat2 is None:
                    continue
                
                ov = compute_topk_routing_overlap(pat1, pat2, k=3)
                cross_overlaps.append(np.mean(ov))
        
        cross_avg = np.mean(cross_overlaps) if cross_overlaps else 0
        within_avg = np.mean(list(within_overlaps.values()))
        
        print(f"    ★ 组内均值: {within_avg:.4f}, 跨组均值: {cross_avg:.4f}, 差距: {within_avg - cross_avg:.4f}")


# ============================================================
# 实验C-Q: Boundary-Conditioned Computation on Qwen3
# ============================================================

def exp_c_qwen3_boundary(model, tokenizer):
    """在Qwen3上验证boundary-conditioned computation"""
    print("=" * 70)
    print("实验C-Q: Boundary-Conditioned Computation on Qwen3")
    print("=" * 70)
    
    n_layers = model.config.num_hidden_layers
    
    shared_content = "the cat sat on the mat"
    
    prefixes = {
        "translate_fr": "Translate to French:",
        "translate_de": "Translate to German:",
        "summarize": "Summarize:",
        "explain": "Explain:",
        "continue": "Continue the text:",
        "question": "Question about:",
        "analyze": "Analyze the grammar of:",
        "paraphrase": "Paraphrase:",
    }
    
    all_patterns = {}
    all_hidden = {}
    for pname, prefix in prefixes.items():
        text = f"{prefix} {shared_content}"
        patterns, hidden = get_attention_patterns_qwen3_forward(model, tokenizer, text)
        all_patterns[pname] = patterns
        all_hidden[pname] = hidden
        torch.cuda.empty_cache()
    
    # ---- 分析1: Prefix对content最后位置routing的影响 ----
    print("\n--- 分析1: Prefix对content最后位置routing的影响 ---")
    
    target_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    for layer in target_layers:
        # 收集各prefix的content最后位置attention
        last_attns = {}
        for pname in prefixes:
            if layer in all_patterns[pname]:
                pat = all_patterns[pname][layer]
                last_attns[pname] = pat[:, -1, :]  # [n_heads, seq_len]
        
        # 两两比较routing rank correlation
        pnames = list(last_attns.keys())
        pair_corrs = []
        for i in range(len(pnames)):
            for j in range(i+1, len(pnames)):
                p1, p2 = pnames[i], pnames[j]
                # last_attns是[n_heads, seq_len], 需要扩展为[n_heads, 1, seq_len]
                a1 = last_attns[p1].unsqueeze(1)  # [n_heads, 1, seq_len]
                a2 = last_attns[p2].unsqueeze(1)
                rank_corrs = compute_rank_correlation(a1, a2)
                pair_corrs.append(np.mean(rank_corrs))
        
        avg_corr = np.mean(pair_corrs) if pair_corrs else 0
        std_corr = np.std(pair_corrs) if pair_corrs else 0
        print(f"  Layer {layer}: cross-prefix routing correlation = {avg_corr:.4f} +/- {std_corr:.4f}")
    
    # ---- 分析2: 同类prefix vs 异类prefix ----
    print("\n--- 分析2: 同类prefix vs 异类prefix ---")
    
    same_category_pairs = [
        ("translate_fr", "translate_de"),  # 都是翻译
        ("summarize", "paraphrase"),        # 都是改写
    ]
    
    diff_category_pairs = [
        ("translate_fr", "summarize"),
        ("translate_fr", "continue"),
        ("translate_fr", "analyze"),
        ("summarize", "continue"),
        ("translate_fr", "question"),
        ("explain", "continue"),
    ]
    
    for layer in target_layers:
        # 同类routing sim
        same_sims = []
        for p1, p2 in same_category_pairs:
            if layer in all_patterns[p1] and layer in all_patterns[p2]:
                pat1 = all_patterns[p1][layer][:, -1, :].unsqueeze(1)  # [n_heads, 1, seq_len]
                pat2 = all_patterns[p2][layer][:, -1, :].unsqueeze(1)
                rank_corrs = compute_rank_correlation(pat1, pat2)
                same_sims.append(np.mean(rank_corrs))
        
        # 异类routing sim
        diff_sims = []
        for p1, p2 in diff_category_pairs:
            if layer in all_patterns[p1] and layer in all_patterns[p2]:
                pat1 = all_patterns[p1][layer][:, -1, :].unsqueeze(1)
                pat2 = all_patterns[p2][layer][:, -1, :].unsqueeze(1)
                rank_corrs = compute_rank_correlation(pat1, pat2)
                diff_sims.append(np.mean(rank_corrs))
        
        same_avg = np.mean(same_sims) if same_sims else 0
        diff_avg = np.mean(diff_sims) if diff_sims else 0
        
        print(f"  Layer {layer}: 同类={same_avg:.4f}, 异类={diff_avg:.4f}, 差距={same_avg-diff_avg:.4f}")
    
    # ---- 分析3: Transition direction ----
    print("\n--- 分析3: Prefix对transition direction的影响 ---")
    
    for layer in target_layers:
        if layer == 0:
            continue  # L0没有前置层
        
        transition_dirs = {}
        for pname in prefixes:
            if layer in all_hidden[pname] and (layer-1) in all_hidden[pname]:
                h_in = all_hidden[pname][layer-1][-1]  # 上一层的最后位置
                h_out = all_hidden[pname][layer][-1]    # 当前层的最后位置
                transition = h_out - h_in
                transition_dirs[pname] = transition / (transition.norm() + 1e-8)
        
        if len(transition_dirs) < 2:
            continue
        
        # 两两比较transition方向
        pnames = list(transition_dirs.keys())
        dir_sims = []
        for i in range(len(pnames)):
            for j in range(i+1, len(pnames)):
                sim = torch.nn.functional.cosine_similarity(
                    transition_dirs[pnames[i]].unsqueeze(0),
                    transition_dirs[pnames[j]].unsqueeze(0),
                ).item()
                dir_sims.append(sim)
        
        print(f"  Layer {layer}: transition direction sim = {np.mean(dir_sims):.4f} +/- {np.std(dir_sims):.4f}")
    
    # ---- 分析4: Head sensitivity ----
    print("\n--- 分析4: Head-level prefix sensitivity ---")
    
    for layer in [n_layers//4, n_layers//2, 3*n_layers//4]:
        if layer not in all_patterns[list(prefixes.keys())[0]]:
            continue
        
        n_heads = all_patterns[list(prefixes.keys())[0]][layer].shape[0]
        
        head_sensitivity = []
        for h in range(min(n_heads, 32)):  # Qwen3可能head数多
            attn_vecs = []
            for pname in prefixes:
                if layer in all_patterns[pname]:
                    pat = all_patterns[pname][layer]
                    attn_vecs.append(pat[h, -1, :].float())
            
            if len(attn_vecs) < 2:
                continue
            
            pair_sims = []
            for i in range(len(attn_vecs)):
                for j in range(i+1, len(attn_vecs)):
                    min_len = min(attn_vecs[i].shape[0], attn_vecs[j].shape[0])
                    sim = torch.nn.functional.cosine_similarity(
                        attn_vecs[i][:min_len].unsqueeze(0),
                        attn_vecs[j][:min_len].unsqueeze(0),
                    ).item()
                    pair_sims.append(sim)
            
            head_sensitivity.append((h, np.mean(pair_sims)))
        
        head_sensitivity.sort(key=lambda x: x[1])
        
        print(f"\n  Layer {layer}:")
        print(f"    最prefix-sensitive heads:")
        for h, sim in head_sensitivity[:5]:
            print(f"      Head {h}: avg cross-prefix sim = {sim:.4f}")
        print(f"    最prefix-insensitive heads:")
        for h, sim in head_sensitivity[-5:]:
            print(f"      Head {h}: avg cross-prefix sim = {sim:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", choices=["a", "c", "all"])
    args = parser.parse_args()
    
    print("Phase 79-Q: Computation Invariants on Qwen3")
    print("=" * 70)
    
    model, tokenizer = get_qwen3_model()
    
    if args.exp in ["a", "all"]:
        exp_a_qwen3_routing(model, tokenizer)
    
    if args.exp in ["c", "all"]:
        exp_c_qwen3_boundary(model, tokenizer)
    
    # 清理
    del model
    torch.cuda.empty_cache()
    
    print("\nPhase 79-Q 完成")
