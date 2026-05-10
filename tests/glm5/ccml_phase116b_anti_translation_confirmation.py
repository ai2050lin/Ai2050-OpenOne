"""
Phase 116b: 反翻译码假说确认 — 大样本验证 + 跨模型验证 + 分层amplify
===================================================================

Phase 116发现: L15-L27的spike是"反翻译码"(移除帮助翻译),
L30-L35是"助翻译码"(移除损害翻译), 因果相变在L27→L30。

但存在关键疑点:
1. 样本量偏小(50词对), 因果效应的稳健性需要验证
2. L21/L24的amplify和remove都帮助, 矛盾
3. 只在Qwen3上做了, 需要跨模型验证
4. "反翻译"的机制不明 — 是spike整体反翻译, 还是spike中含两种成分?

实验设计:

Exp A: 大样本因果扫描 (100词对, Qwen3)
   - 验证因果效应的稳健性
   - 特别关注L21/L24的矛盾效应

Exp B: 分层spike分析 — spike中是否同时含助/反翻译成分?
   - 将L21的20维spike分成top-k和bottom-k
   - 分别干预top-k和bottom-k, 看方向是否不同
   - 如果top-k助翻译, bottom-k反翻译 → spike含混合成分

Exp C: 跨模型因果扫描 (DeepSeek7B)
   - DeepSeek7B是否也有L27的反翻译效应?
   - DeepSeek7B的因果相变在哪里?

理论纪律:
- "反翻译"是可测量的统计量, 不是理论推测
- 不假设"势垒"存在, 只报告因果效应的方向
- 矛盾效应优先按数据解释, 不强行统一
"""

import os, sys, json, gc, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, get_layers, get_model_info, release_model

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 扩大词对到150个
WORD_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"),
    ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"), ("鸡", "chicken"), ("鼠", "mouse"),
    ("水", "water"), ("火", "fire"), ("土", "earth"), ("风", "wind"), ("金", "gold"),
    ("木", "wood"), ("铁", "iron"), ("石", "stone"), ("沙", "sand"), ("冰", "ice"),
    ("月", "moon"), ("星", "star"), ("云", "cloud"), ("雨", "rain"), ("雪", "snow"),
    ("日", "sun"), ("天", "sky"), ("海", "sea"), ("河", "river"), ("山", "mountain"),
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"), ("黑", "black"),
    ("黄", "yellow"), ("紫", "purple"), ("灰", "gray"), ("棕", "brown"), ("粉", "pink"),
    ("手", "hand"), ("足", "foot"), ("目", "eye"), ("耳", "ear"), ("口", "mouth"),
    ("心", "heart"), ("头", "head"), ("骨", "bone"), ("血", "blood"), ("发", "hair"),
    ("父", "father"), ("母", "mother"), ("子", "son"), ("女", "daughter"), ("友", "friend"),
    ("王", "king"), ("师", "teacher"), ("医", "doctor"), ("兵", "soldier"), ("农", "farmer"),
    ("爱", "love"), ("恨", "hate"), ("善", "good"), ("恶", "evil"), ("真", "truth"),
    ("美", "beauty"), ("智", "wisdom"), ("力", "power"), ("光", "light"), ("影", "shadow"),
    ("走", "walk"), ("跑", "run"), ("飞", "fly"), ("吃", "eat"), ("喝", "drink"),
    ("看", "see"), ("听", "hear"), ("说", "say"), ("写", "write"), ("读", "read"),
    ("年", "year"), ("月", "month"), ("日", "day"), ("时", "hour"), ("分", "minute"),
    ("米", "rice"), ("茶", "tea"), ("酒", "wine"), ("肉", "meat"), ("盐", "salt"),
    ("车", "car"), ("船", "ship"), ("路", "road"), ("桥", "bridge"), ("门", "door"),
    ("花", "flower"), ("草", "grass"), ("树", "tree"), ("叶", "leaf"), ("根", "root"),
    ("喜", "joy"), ("怒", "anger"), ("哀", "sorrow"), ("惧", "fear"), ("思", "thought"),
    ("法", "law"), ("国", "country"), ("城", "city"), ("家", "home"), ("书", "book"),
    ("电", "electricity"), ("网", "network"), ("数", "number"), ("算", "compute"), ("器", "device"),
    ("湖", "lake"), ("岛", "island"), ("春", "spring"), ("夏", "summer"), ("秋", "autumn"),
    ("冬", "winter"), ("晨", "morning"), ("暮", "dusk"), ("雷", "thunder"), ("雾", "fog"),
    ("龙", "dragon"), ("蛇", "snake"), ("虎", "tiger"), ("鹿", "deer"), ("兔", "rabbit"),
    ("道", "way"), ("德", "virtue"), ("礼", "ritual"), ("义", "justice"), ("信", "trust"),
    ("剑", "sword"), ("笔", "pen"), ("琴", "lute"), ("画", "painting"), ("棋", "chess"),
    ("砖", "brick"), ("瓦", "tile"), ("丝", "silk"), ("布", "cloth"), ("纸", "paper"),
]

seen_zh = set()
UNIQUE_PAIRS = []
for zh, en in WORD_PAIRS:
    if zh not in seen_zh:
        seen_zh.add(zh)
        UNIQUE_PAIRS.append((zh, en))
WORD_PAIRS = UNIQUE_PAIRS[:150]

SAMPLE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
KNOWN_SIGNAL_DIMS_QWEN = {0: 25, 3: 24, 6: 23, 9: 21, 12: 4, 15: 14, 18: 8, 21: 20, 24: 21, 27: 18, 30: 23, 33: 24, 35: 14}
KNOWN_SIGNAL_DIMS_DS7B = {0: 29, 3: 27, 6: 22, 9: 22, 12: 21, 15: 21, 18: 19, 21: 24, 24: 26, 27: 7}


def collect_mlp_outputs(model, tokenizer, device, n_layers, word_pairs, batch_size=5):
    layers = get_layers(model)
    all_mlp = {'zh': defaultdict(list), 'trans': defaultdict(list)}
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        for zh, en in batch:
            zh_prompt = f"翻译以下中文词：{zh}"
            trans_prompt = f"Translate the following Chinese word: {zh}"
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                mlp_outs = {}
                hooks = []
                def make_mlp_hook(l):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            mlp_outs[l] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            mlp_outs[l] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                for l, layer in enumerate(layers):
                    if hasattr(layer, 'mlp'):
                        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))
                with torch.no_grad():
                    _ = model(inputs["input_ids"])
                for h in hooks:
                    h.remove()
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                for l in mlp_outs:
                    all_mlp[task][l].append(mlp_outs[l])
        print(f"  [collect_mlp] {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)}")
    result = {}
    for task in ['zh', 'trans']:
        result[task] = {}
        for l in all_mlp[task]:
            result[task][l] = np.array(all_mlp[task][l])
    return result


def get_spike_subspace(diffs, k, N, P):
    diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    k_actual = min(k, Vt.shape[0])
    return {
        'V': Vt[:k_actual, :].T,
        'S': S[:k_actual],
        'Vt': Vt[:k_actual, :],
        'k': k_actual,
        'coefficients': U[:, :k_actual] * S[:k_actual],
        'mean_diff': diffs.mean(axis=0),
    }


def make_remove_hook(proj_matrix_np, scale=1.0):
    proj_matrix = torch.tensor(proj_matrix_np, dtype=torch.float32, device='cuda:0')
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].detach().clone().float()
            last_h = h[0, -1, :].unsqueeze(0)
            proj = proj_matrix @ (proj_matrix.T @ last_h.T)
            h[0, -1, :] -= proj.squeeze() * scale
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.detach().clone().float()
            last_h = h[0, -1, :].unsqueeze(0)
            proj = proj_matrix @ (proj_matrix.T @ last_h.T)
            h[0, -1, :] -= proj.squeeze() * scale
            return h.to(output.dtype)
    return hook_fn


def make_amplify_hook(proj_matrix_np, factor=1.0):
    proj_matrix = torch.tensor(proj_matrix_np, dtype=torch.float32, device='cuda:0')
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].detach().clone().float()
            last_h = h[0, -1, :].unsqueeze(0)
            proj = proj_matrix @ (proj_matrix.T @ last_h.T)
            h[0, -1, :] += proj.squeeze() * factor
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.detach().clone().float()
            last_h = h[0, -1, :].unsqueeze(0)
            proj = proj_matrix @ (proj_matrix.T @ last_h.T)
            h[0, -1, :] += proj.squeeze() * factor
            return h.to(output.dtype)
    return hook_fn


def evaluate_translation_batch(pairs, model, tokenizer, device, layers_list, layer_idx, hook_fn=None):
    """批量评估翻译log_prob"""
    log_probs = []
    for zh, en in pairs:
        trans_prompt = f"Translate the following Chinese word: {zh}"
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        hooks = []
        if hook_fn is not None and layer_idx < len(layers_list):
            hooks.append(layers_list[layer_idx].register_forward_hook(hook_fn))
        with torch.no_grad():
            outputs = model(inputs["input_ids"])
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        target_ids = tokenizer.encode(en, add_special_tokens=False)
        target_log_prob = -100
        for tid in target_ids:
            if tid < len(probs):
                target_log_prob = max(target_log_prob, float(torch.log(probs[tid] + 1e-10)))
        log_probs.append(target_log_prob)
        for h in hooks:
            h.remove()
        del inputs, outputs, logits, probs
        gc.collect()
        torch.cuda.empty_cache()
    return float(np.mean(log_probs)), float(np.std(log_probs))


# ============================================================
# Exp A: 大样本因果扫描
# ============================================================
def expA_large_sample_causal_sweep(model, tokenizer, device, n_layers, mlp_outs, d_model,
                                    model_name, n_test=100):
    print("\n" + "="*70)
    print(f"Exp A: 大样本因果扫描 ({n_test}词对, {model_name})")
    print("="*70)
    
    layers_list = get_layers(model)
    dims = KNOWN_SIGNAL_DIMS_QWEN if model_name == "qwen3" else KNOWN_SIGNAL_DIMS_DS7B
    test_pairs = WORD_PAIRS[:n_test]
    results = {}
    
    # 先计算baseline (所有层共用)
    print("  Computing baseline...")
    baseline_lp, baseline_std = evaluate_translation_batch(
        test_pairs, model, tokenizer, device, layers_list, -1, hook_fn=None)
    print(f"  Baseline: log_prob={baseline_lp:.3f}±{baseline_std:.3f}")
    
    for l in SAMPLE_LAYERS:
        if l >= len(layers_list) or l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        k = dims.get(l, min(10, N-1))
        
        diffs = trans_data - zh_data
        spike = get_spike_subspace(diffs, k, N, P)
        V = spike['V']
        
        # Remove spike
        remove_lp, _ = evaluate_translation_batch(
            test_pairs, model, tokenizer, device, layers_list, l,
            make_remove_hook(V))
        
        # Amplify spike
        amplify_lp, _ = evaluate_translation_batch(
            test_pairs, model, tokenizer, device, layers_list, l,
            make_amplify_hook(V, factor=1.0))
        
        # Remove random
        V_random = np.random.randn(P, spike['k'])
        V_random, _ = np.linalg.qr(V_random)
        random_lp, _ = evaluate_translation_batch(
            test_pairs, model, tokenizer, device, layers_list, l,
            make_remove_hook(V_random))
        
        delta_remove = remove_lp - baseline_lp
        delta_amplify = amplify_lp - baseline_lp
        delta_random = random_lp - baseline_lp
        causal_effect = delta_remove - delta_random
        
        results[f"L{l}"] = {
            "spike_dim": spike['k'],
            "baseline_log_prob": float(baseline_lp),
            "delta_remove": float(delta_remove),
            "delta_amplify": float(delta_amplify),
            "delta_random": float(delta_random),
            "causal_effect": float(causal_effect),
            "n_test": n_test,
        }
        
        print(f"  L{l}: Δ_remove={delta_remove:+.3f}, Δ_amplify={delta_amplify:+.3f}, "
              f"Δ_random={delta_random:+.3f}, causal={causal_effect:+.3f}")
    
    return results


# ============================================================
# Exp B: 分层spike分析 — top-k vs bottom-k
# ============================================================
def expB_stratified_spike(model, tokenizer, device, n_layers, mlp_outs, d_model,
                           target_layers=[21, 24, 27], n_test=80):
    """
    分层spike分析: 将spike子空间分成top-k(强信号)和bottom-k(弱信号), 
    分别干预看因果效应方向是否不同
    
    如果L21的矛盾效应(两者都帮助)是因为top-k助翻译+bottom-k反翻译,
    那么分别干预应该显示不同方向
    """
    print("\n" + "="*70)
    print("Exp B: 分层spike分析 — top-k vs bottom-k")
    print("="*70)
    
    layers_list = get_layers(model)
    test_pairs = WORD_PAIRS[:n_test]
    results = {}
    
    # Baseline
    baseline_lp, _ = evaluate_translation_batch(
        test_pairs, model, tokenizer, device, layers_list, -1, hook_fn=None)
    print(f"  Baseline: log_prob={baseline_lp:.3f}")
    
    for l in target_layers:
        if l >= len(layers_list) or l not in mlp_outs['zh']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        k = KNOWN_SIGNAL_DIMS_QWEN.get(l, min(10, N-1))
        
        diffs = trans_data - zh_data
        spike = get_spike_subspace(diffs, k, N, P)
        V = spike['V']  # [P, k]
        S = spike['S']  # [k]
        
        # 分成top-k/2和bottom-k/2
        k_half = max(1, spike['k'] // 2)
        V_top = V[:, :k_half]  # 前k/2个最大特征值方向
        V_bottom = V[:, k_half:]  # 后k/2个最小特征值方向
        
        # 干预top-k
        remove_top_lp, _ = evaluate_translation_batch(
            test_pairs, model, tokenizer, device, layers_list, l,
            make_remove_hook(V_top))
        amplify_top_lp, _ = evaluate_translation_batch(
            test_pairs, model, tokenizer, device, layers_list, l,
            make_amplify_hook(V_top, factor=1.0))
        
        # 干预bottom-k
        remove_bottom_lp, _ = evaluate_translation_batch(
            test_pairs, model, tokenizer, device, layers_list, l,
            make_remove_hook(V_bottom))
        amplify_bottom_lp, _ = evaluate_translation_batch(
            test_pairs, model, tokenizer, device, layers_list, l,
            make_amplify_hook(V_bottom, factor=1.0))
        
        delta_remove_top = remove_top_lp - baseline_lp
        delta_remove_bottom = remove_bottom_lp - baseline_lp
        delta_amplify_top = amplify_top_lp - baseline_lp
        delta_amplify_bottom = amplify_bottom_lp - baseline_lp
        
        results[f"L{l}"] = {
            "spike_dim": spike['k'],
            "k_top": k_half,
            "k_bottom": spike['k'] - k_half,
            "top_eigenvalues": [float(x) for x in S[:k_half]],
            "bottom_eigenvalues": [float(x) for x in S[k_half:]],
            "delta_remove_top": float(delta_remove_top),
            "delta_remove_bottom": float(delta_remove_bottom),
            "delta_amplify_top": float(delta_amplify_top),
            "delta_amplify_bottom": float(delta_amplify_bottom),
        }
        
        print(f"  L{l} (k={spike['k']}, top={k_half}, bottom={spike['k']-k_half}):")
        print(f"    Remove top:  Δ={delta_remove_top:+.3f}")
        print(f"    Remove bottom: Δ={delta_remove_bottom:+.3f}")
        print(f"    Amplify top: Δ={delta_amplify_top:+.3f}")
        print(f"    Amplify bottom: Δ={delta_amplify_bottom:+.3f}")
        
        # 判定: top和bottom是否有不同方向?
        if delta_remove_top * delta_remove_bottom < 0:
            print(f"    → TOP和BOTTOM因果方向不同! spike含混合成分")
        else:
            print(f"    → TOP和BOTTOM因果方向相同, spike成分一致")
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["all", "A", "B"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*70}")
    print(f"Phase 116b: 反翻译码假说确认")
    print(f"模型: {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    global SAMPLE_LAYERS
    if model_name == "deepseek7b":
        SAMPLE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    elif n_layers <= 36:
        SAMPLE_LAYERS = [l for l in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35] if l < n_layers]
    
    print(f"模型: {model_info.model_class}, {n_layers}层, d_model={d_model}")
    
    all_results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    print("\n--- 收集MLP输出 ---")
    mlp_outs = collect_mlp_outputs(model, tokenizer, device, n_layers, WORD_PAIRS)
    
    # Exp A: 大样本因果扫描
    if args.exp in ["all", "A"]:
        print("\n--- Exp A: 大样本因果扫描 ---")
        expA_result = expA_large_sample_causal_sweep(
            model, tokenizer, device, n_layers, mlp_outs, d_model, model_name, n_test=100)
        all_results["expA_large_sample"] = expA_result
        
        out_path = os.path.join(OUT_DIR, f"phase116b_expA_{model_name}_large_sample.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(expA_result, f, indent=2, ensure_ascii=False)
    
    # Exp B: 分层spike分析
    if args.exp in ["all", "B"]:
        print("\n--- Exp B: 分层spike分析 ---")
        target_layers = [l for l in [15, 18, 21, 24, 27] if l < n_layers]
        expB_result = expB_stratified_spike(
            model, tokenizer, device, n_layers, mlp_outs, d_model,
            target_layers=target_layers, n_test=80)
        all_results["expB_stratified"] = expB_result
        
        out_path = os.path.join(OUT_DIR, f"phase116b_expB_{model_name}_stratified.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(expB_result, f, indent=2, ensure_ascii=False)
    
    all_out_path = os.path.join(OUT_DIR, f"phase116b_{model_name}_all_results.json")
    with open(all_out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n全部结果保存到 {all_out_path}")
    
    release_model(model)
    print("\nPhase 116b 完成!")


if __name__ == "__main__":
    main()
