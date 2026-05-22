"""
Phase 57b: 控制模板差异的复用/差异化验证
==========================================
核心问题: AND/OR的低cos是因为模板语法差异还是逻辑操作差异?

验证1: 语法匹配的AND/OR模板 — 消除语法差异, 只保留逻辑差异
验证2: apple/fruit用不同模板 — 消除模板相似性偏差
验证3: 骨干子空间语义解码 — 投影骨干到W_U解码语义
"""

import sys, os, json, time, argparse, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.decomposition import PCA

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import (
    load_model, get_layers, get_model_info, get_W_U,
    release_model, safe_decode, MODEL_CONFIGS
)

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    # 安全输出: 替换非ASCII字符
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)


# ===== 验证1: 语法匹配的AND/OR模板 =====
# 关键: 使用完全相同的句子框架, 只替换AND/OR
CONTROLLED_AND_OR = [
    "You can have tea {w} coffee with your breakfast",
    "She likes reading {w} writing in her free time", 
    "He needs patience {w} skill to win the game",
    "They want justice {w} peace for their country",
    "We need love {w} trust to build a family",
    "The project requires time {w} money to succeed",
    "She has talent {w} ambition for this career",
    "You should eat fruits {w} vegetables every day",
    "He speaks English {w} French fluently",
    "The room has a desk {w} chair for studying",
    "I need paper {w} pen to write a letter",
    "She bought bread {w} milk from the store",
    "They have cars {w} bikes for transportation",
    "He plays guitar {w} piano in the band",
    "We see stars {w} moon at night",
    "The soup contains salt {w} pepper for flavor",
    "She uses laptop {w} phone for work",
    "He wears shirt {w} pants to the office",
    "They grow corn {w} wheat on the farm",
    "The box has nails {w} screws inside",
    "I drink tea {w} juice in the morning",
    "She reads books {w} magazines at the library",
    "He likes cats {w} dogs as pets",
    "They build houses {w} apartments in the city",
    "We cook rice {w} pasta for dinner",
    "The park has trees {w} flowers everywhere",
    "She paints portraits {w} landscapes beautifully",
    "He studies math {w} science at school",
    "They sell shirts {w} dresses at the store",
    "We need food {w} water to survive",
]

# ===== 验证2: 语法不同的apple/fruit模板 =====
DIFFERENT_TEMPLATES_APPLE = [
    "I think about {w} when I am hungry",
    "Someone mentioned {w} at the dinner table",
    "The recipe calls for {w} as an ingredient",
    "She bought some {w} from the farmer",
    "He grew {w} in his backyard garden",
    "My grandmother makes the best {w} pie",
    "The supermarket has fresh {w} today",
    "I need {w} for the salad",
    "She picked {w} from the orchard",
    "The {w} season starts in September",
    "A ripe {w} fell from the tree",
    "We had {w} for dessert last night",
    "The market sells organic {w}",
    "I prefer green {w} over red ones",
    "She sliced the {w} carefully",
    "The {w} juice tastes refreshing",
    "He offered me a piece of {w}",
    "The {w} orchard looks beautiful in fall",
    "I found wild {w} in the forest",
    "The {w} harvest was abundant this year",
    "She made {w} preserves for winter",
    "The {w} basket was almost empty",
    "He composts {w} scraps in the bin",
    "I crave {w} on hot summer days",
    "The {w} tree needs more water",
    "She chooses {w} that are firm",
    "The {w} basket overflowed",
    "He washed the {w} thoroughly",
    "The smell of {w} filled the kitchen",
    "I enjoy {w} with peanut butter",
]


def find_target_token_pos_in_full(tokenizer, input_ids, target_word):
    """在完整token序列中找目标词位置"""
    tokens_list = input_ids[0].tolist()
    
    for i in range(len(tokens_list)):
        decoded = tokenizer.decode(tokens_list[i])
        if target_word.lower() in decoded.lower():
            return i, 1
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            if target_word.lower() == decoded.strip().lower():
                return i, j - i
    
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped and target_word.lower() in stripped and len(stripped) <= len(target_word) + 2:
                return i, j - i
    
    return None, None


def collect_activations(model, tokenizer, device, templates, target_word, 
                        n_layers, target_layers):
    """收集目标词位置的激活"""
    activations = {li: [] for li in target_layers}
    found = 0
    
    for template in templates:
        sentence = template.replace("{w}", target_word)
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs.input_ids.to(device)
        seq_len = input_ids.shape[1]
        
        pos, tlen = find_target_token_pos_in_full(tokenizer, input_ids, target_word)
        if pos is None or pos >= seq_len:
            continue
        
        found += 1
        actual_pos = min(pos + (tlen // 2), seq_len - 1)
        
        layers = get_layers(model)
        captured = {}
        
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in target_layers:
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        import torch
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids)
            except:
                for h in hooks: h.remove()
                continue
        
        for h in hooks: h.remove()
        
        for li in target_layers:
            key = f"L{li}"
            if key in captured and actual_pos < captured[key].shape[1]:
                act = captured[key][0, actual_pos, :].numpy()
                activations[li].append(act)
    
    return activations, found


def compute_subspace_analysis(activations_a, activations_b, n_dims=15):
    """分析两个概念激活的共享/独特子空间"""
    if len(activations_a) < 5 or len(activations_b) < 5:
        return None
    
    A = np.array(activations_a)
    B = np.array(activations_b)
    
    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)
    A_centered = A - mean_a
    B_centered = B - mean_b
    
    cos_mean = float(np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-10))
    
    n_comp = min(n_dims, min(A_centered.shape) - 1, min(B_centered.shape) - 1)
    n_comp = max(n_comp, 2)
    
    pca_a = PCA(n_components=n_comp)
    pca_a.fit(A_centered)
    pca_b = PCA(n_components=n_comp)
    pca_b.fit(B_centered)
    
    V_a = pca_a.components_
    V_b = pca_b.components_
    
    overlap_matrix = V_a @ V_b.T
    subspace_overlap = float(np.sum(overlap_matrix ** 2) / n_comp)
    
    # delta_unique
    delta_mean = mean_a - mean_b
    delta_proj_B = V_b.T @ (V_b @ delta_mean)
    delta_unique_vec = delta_mean - delta_proj_B
    shared_delta_ratio = float(np.sum(delta_proj_B**2) / max(np.sum(delta_mean**2), 1e-10))
    unique_delta_ratio = float(np.sum(delta_unique_vec**2) / max(np.sum(delta_mean**2), 1e-10))
    
    # shared ratio
    A_proj_B = V_b.T @ (V_b @ A_centered.T)
    shared_ratio_A = float(np.sum(A_proj_B ** 2) / max(np.sum(A_centered ** 2), 1e-10))
    
    B_proj_A = V_a.T @ (V_a @ B_centered.T)
    shared_ratio_B = float(np.sum(B_proj_A ** 2) / max(np.sum(B_centered ** 2), 1e-10))
    
    return {
        "cos_mean": cos_mean,
        "subspace_overlap": subspace_overlap,
        "shared_ratio_A": shared_ratio_A,
        "shared_ratio_B": shared_ratio_B,
        "avg_shared_ratio": (shared_ratio_A + shared_ratio_B) / 2,
        "unique_delta_ratio": unique_delta_ratio,
        "shared_delta_ratio": shared_delta_ratio,
        "n_a": len(activations_a),
        "n_b": len(activations_b),
    }


def decode_backbone_directions(model, tokenizer, activations_dict, W_U, top_k=20):
    """解码骨干方向的语义内容"""
    all_acts = []
    for word, acts in activations_dict.items():
        if len(acts) >= 5:
            all_acts.append(np.array(acts))
    
    if len(all_acts) < 2:
        return None
    
    # 合并所有概念, 提取骨干方向(跨概念共享的PCA维度)
    combined = np.vstack(all_acts)
    combined_centered = combined - combined.mean(axis=0)
    pca = PCA(n_components=min(15, min(combined_centered.shape) - 1))
    pca.fit(combined_centered)
    
    # 解码前5个骨干方向
    results = {}
    for i in range(min(5, pca.n_components_)):
        direction = pca.components_[i]  # [d_model]
        # 投影到W_U, 找最相关的token
        scores = W_U @ direction  # [vocab_size]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        bottom_indices = np.argsort(scores)[:top_k]
        
        top_tokens = [(safe_decode(tokenizer, idx), float(scores[idx])) for idx in top_indices]
        bottom_tokens = [(safe_decode(tokenizer, idx), float(scores[idx])) for idx in bottom_indices]
        
        results[f"PC{i}"] = {
            "variance_ratio": float(pca.explained_variance_ratio_[i]),
            "top_tokens": top_tokens,
            "bottom_tokens": bottom_tokens,
        }
    
    return results


def run_validation(model_name):
    import torch
    
    log_time(f"=" * 60)
    log_time(f"Phase 57b: 控制模板验证 - {model_name}")
    log_time(f"=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    attn_impl = "eager" if model_name == "deepseek7b" else "sdpa"
    
    if model_name == "qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="cpu",
            trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True,
            attn_implementation=attn_impl, use_cache=False,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl, use_cache=False,
        )
    
    model.eval()
    device = next(model.parameters()).device
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    target_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    log_time(f"采样层: {target_layers}")
    
    results = {"model": model_name, "validations": {}}
    
    # ===== 验证1: 语法匹配的AND/OR =====
    log_time(f"\n验证1: 语法匹配的AND/OR (30句, 相同模板框架)")
    act_and, n_and = collect_activations(model, tokenizer, device, CONTROLLED_AND_OR, "and", n_layers, target_layers)
    act_or, n_or = collect_activations(model, tokenizer, device, CONTROLLED_AND_OR, "or", n_layers, target_layers)
    log_time(f"  找到: and={n_and}, or={n_or}")
    
    and_or_controlled = {}
    for li in target_layers:
        analysis = compute_subspace_analysis(act_and.get(li, []), act_or.get(li, []))
        if analysis:
            and_or_controlled[str(li)] = analysis
            log_time(f"  L{li}: cos={analysis['cos_mean']:.4f} overlap={analysis['subspace_overlap']:.4f} "
                     f"shared={analysis['avg_shared_ratio']:.4f} delta_unique={analysis['unique_delta_ratio']:.4f}")
    
    results["validations"]["and_or_controlled"] = and_or_controlled
    
    del act_and, act_or
    torch.cuda.empty_cache()
    
    # ===== 验证2: 语法不同的apple/fruit =====
    log_time(f"\n验证2: 语法不同的apple/fruit (不同模板)")
    act_apple_diff, n_apple = collect_activations(model, tokenizer, device, DIFFERENT_TEMPLATES_APPLE, "apple", n_layers, target_layers)
    
    # fruit用同一个不同模板
    act_fruit_diff, n_fruit = collect_activations(model, tokenizer, device, DIFFERENT_TEMPLATES_APPLE, "fruit", n_layers, target_layers)
    log_time(f"  找到: apple={n_apple}, fruit={n_fruit}")
    
    apple_fruit_diff = {}
    for li in target_layers:
        analysis = compute_subspace_analysis(act_apple_diff.get(li, []), act_fruit_diff.get(li, []))
        if analysis:
            apple_fruit_diff[str(li)] = analysis
            log_time(f"  L{li}: cos={analysis['cos_mean']:.4f} overlap={analysis['subspace_overlap']:.4f} "
                     f"shared={analysis['avg_shared_ratio']:.4f} delta_unique={analysis['unique_delta_ratio']:.4f}")
    
    results["validations"]["apple_fruit_different_template"] = apple_fruit_diff
    
    # ===== 验证3: 骨干子空间语义解码 =====
    log_time(f"\n验证3: 骨干子空间语义解码")
    mid_layer = n_layers // 2
    
    # 收集所有概念的激活
    concept_acts = {}
    for word, templates in [("apple", DIFFERENT_TEMPLATES_APPLE), ("fruit", DIFFERENT_TEMPLATES_APPLE)]:
        act, n = collect_activations(model, tokenizer, device, templates, word, n_layers, [mid_layer])
        if mid_layer in act and len(act[mid_layer]) >= 5:
            concept_acts[word] = act[mid_layer]
    
    # 也收集AND/OR
    for word, templates in [("and", CONTROLLED_AND_OR), ("or", CONTROLLED_AND_OR)]:
        act, n = collect_activations(model, tokenizer, device, templates, word, n_layers, [mid_layer])
        if mid_layer in act and len(act[mid_layer]) >= 5:
            concept_acts[word] = act[mid_layer]
    
    if len(concept_acts) >= 2:
        W_U = get_W_U(model, model_name)
        decoded = decode_backbone_directions(model, tokenizer, concept_acts, W_U)
        if decoded:
            results["validations"]["backbone_decode"] = decoded
            for pc, data in decoded.items():
                log_time(f"  {pc} (var={data['variance_ratio']:.4f}): "
                         f"top={data['top_tokens'][:5]}")
    
    del act_apple_diff, act_fruit_diff, concept_acts
    torch.cuda.empty_cache()
    
    # 保存结果
    output_dir = PROJECT / "results" / "subspace_topology"
    output_file = output_dir / f"exp3b_controlled_{model_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    log_time(f"结果已保存: {output_file}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    run_validation(args.model)
